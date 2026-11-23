"""Input sources delivering current control values {channel: value}.

read() must be non-blocking and return the *latest* value, so the FE loop
never queues behind the experiment. A real device/socket source plugs in by
implementing the same interface. status() describes the source in one line,
which is shown on screen above the results.
"""
import collections
import http.client
import math
import re
import sys
import threading
import time
import urllib.parse
import urllib.request


class Source:
    def __init__(self, channels):
        self.channels = list(channels)

    def read(self):
        raise NotImplementedError

    def status(self):
        """One line describing the source and its current state, shown on screen
        after the "input:" label.

        It may change over time (e.g. connection state or a measured value);
        the demo redraws the status text whenever it does.
        """
        return type(self).__name__

    def close(self):
        pass


class SineSource(Source):
    """Oscillates every channel between 0 and amp: amp*(1-cos(2*pi*t/period))/2."""

    def __init__(self, channels, amp=5.0, period=4.0):
        super().__init__(channels)
        self.amp, self.period, self.t0 = amp, period, time.perf_counter()

    def read(self):
        t = time.perf_counter() - self.t0
        v = 0.5 * self.amp * (1.0 - math.cos(2.0 * math.pi * t / self.period))
        return {c: v for c in self.channels}

    def status(self):
        return f"sine 0–{self.amp:g} mm, period {self.period:g} s"


class RampSource(Source):
    """Triangular ramp 0 -> amp -> 0 with given period."""

    def __init__(self, channels, amp=5.0, period=4.0):
        super().__init__(channels)
        self.amp, self.period, self.t0 = amp, period, time.perf_counter()

    def read(self):
        ph = ((time.perf_counter() - self.t0) / self.period) % 1.0
        v = self.amp * (2.0 * ph if ph < 0.5 else 2.0 * (1.0 - ph))
        return {c: v for c in self.channels}

    def status(self):
        return f"ramp 0–{self.amp:g}–0 mm, period {self.period:g} s"


class ManualSource(Source):
    """Value set externally, e.g. by a slider in the plot window (see renderers)."""

    def __init__(self, channels, amp=5.0, initial=0.0):
        super().__init__(channels)
        self.amp = amp
        self.values = {c: initial for c in self.channels}

    def set(self, channel, value):
        self.values[channel] = float(value)

    def read(self):
        return dict(self.values)

    def status(self):
        return f"manual slider 0–{self.amp:g} mm"


TELEMETRY_PORT = 8765
DEFAULT_TELEMETRY_URL = f"http://127.0.0.1:{TELEMETRY_PORT}/api/values"


class TelemetrySource(Source):
    """Displacement measured on the testing machine, read over the LAN.

    Polls the telemetry bridge (Experiment/server.py) at GET /api/values,
    which answers either "relative_mm,force_N" (plain text, 2 decimals) or
    204 No Content when the machine is not publishing fresh data.

    A background thread does the HTTP requests and keeps only the latest
    sample, so read() never waits for the network and the FE loop never
    queues behind the experiment. When the data stop (204, network error),
    the last displacement is held, so the model does not jump back to zero;
    status() then shows why and for how long.
    """

    # same format check as Experiment/receiver.py (parse_values)
    VALUES_RE = re.compile(r"(-?\d+\.\d{2}),(-?\d+\.\d{2})")

    # link states (self._state)
    CONNECTING = "connecting"   # no answer from the server yet
    OK = "ok"                   # last request returned a sample
    NO_DATA = "no_data"         # server answers 204: the machine is not sending
    ERROR = "error"             # no connection or invalid answer

    def __init__(self, channels, url=DEFAULT_TELEMETRY_URL, rate=15.0, scale=1.0, timeout=1.0):
        super().__init__(channels)
        self.url = url
        self.rate = float(rate)        # polling frequency [Hz]
        self.scale = float(scale)      # channel value = scale * measured displacement
        self.timeout = float(timeout)  # per-request timeout [s]
        # "host:port" for the status line
        parts = urllib.parse.urlsplit(url)
        self.address = f"{parts.hostname}:{parts.port or (443 if parts.scheme == 'https' else 80)}"

        # latest sample and link state, shared with the polling thread (guarded by _lock)
        self._lock = threading.Lock()
        self._disp = 0.0          # last valid displacement [mm]; 0 before the first sample
        self._force = 0.0         # last valid force [N]
        self._n_ok = 0            # number of valid samples received
        self._state = self.CONNECTING
        self._since = time.monotonic()   # when the current state was entered
        self._error = ""          # short error text for the status line (state ERROR)
        self._t_ok = collections.deque(maxlen=64)   # times of recent samples (read rate)

        # no proxies: the machine PC is on the local network (like trust_env=False)
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="telemetry", daemon=True)
        self._thread.start()

    @classmethod
    def parse(cls, text):
        """'1.23,-45.60' -> (1.23, -45.6); raise ValueError on anything else."""
        m = cls.VALUES_RE.fullmatch(text.strip())
        if m is None:
            raise ValueError(f"invalid /api/values response {text[:40]!r}")
        disp, force = map(float, m.groups())
        if not (math.isfinite(disp) and math.isfinite(force)):
            raise ValueError("non-finite measurement")
        return disp, force

    @staticmethod
    def _log(msg):
        print(f"[telemetry] {msg}", file=sys.stderr, flush=True)

    def _set_state(self, state, error=""):
        """Switch the link state (call with _lock held); keeps the start time
        while the state does not change, for the "(N s)" shown on screen."""
        if state != self._state:
            self._state, self._since = state, time.monotonic()
        self._error = error

    def _poll_once(self):
        """One GET request; updates the shared state. Raises on failure."""
        with self._opener.open(self.url, timeout=self.timeout) as resp:
            status = resp.status
            body = resp.read(256).decode("ascii", "replace")
        if status == 204:
            # server is up, but the machine is not sending: keep the last values
            with self._lock:
                self._set_state(self.NO_DATA)
            return
        disp, force = self.parse(body)
        now = time.monotonic()
        with self._lock:
            self._disp, self._force = disp, force
            self._n_ok += 1
            self._t_ok.append(now)
            self._set_state(self.OK)

    @staticmethod
    def _short_error(exc):
        """Compact error text, e.g. 'Connection refused' or 'timed out'."""
        reason = getattr(exc, "reason", exc)          # URLError wraps the cause
        return getattr(reason, "strerror", None) or str(reason) or type(exc).__name__

    def _run(self):
        """Polling loop of the background thread."""
        interval = 1.0 / self.rate
        delay = 1.0               # reconnect delay, grows on repeated errors
        logged = None             # last message written to stderr (avoid flooding)
        next_tick = time.monotonic()
        while not self._stop.is_set():
            try:
                self._poll_once()
            except (OSError, ValueError, http.client.HTTPException) as exc:
                # OSError covers URLError, HTTPError, timeouts, refused connections
                short = self._short_error(exc)
                with self._lock:
                    self._set_state(self.ERROR, short)
                msg = f"{type(exc).__name__}: {short}"
                if msg != logged:
                    self._log(f"{msg} ({self.url}); retrying")
                    logged = msg
                # back off like Experiment/receiver.py: 1 s, x1.7, at most 5 s
                self._stop.wait(delay)
                delay = min(5.0, delay * 1.7)
                next_tick = time.monotonic()
                continue

            # report state changes (receiving / machine not sending) once
            with self._lock:
                state = self._state
            msg = "receiving data" if state == self.OK else "no fresh data from the machine"
            if msg != logged:
                self._log(msg)
                logged = msg
            delay = 1.0

            # fixed polling rate; if a request took too long, skip missed ticks
            next_tick += interval
            now = time.monotonic()
            if next_tick < now:
                next_tick = now
            self._stop.wait(next_tick - now)

    def read(self):
        with self._lock:
            v = self.scale * self._disp
        return {c: v for c in self.channels}

    def status(self):
        now = time.monotonic()
        with self._lock:
            state, since, error, n_ok, force = (self._state, self._since, self._error,
                                                self._n_ok, self._force)
            recent = [t for t in self._t_ok if now - t <= 2.0]
        age = now - since
        if state == self.OK:
            # samples per second over the last 2 s
            rate = (len(recent) - 1) / (recent[-1] - recent[0]) if len(recent) > 1 else 0.0
            link = f"receiving, {rate:.0f} reads/s"
        elif state == self.NO_DATA:
            link = f"NO FRESH DATA from the machine for {age:.0f} s"
        elif state == self.ERROR:
            link = f"NO CONNECTION ({error[:40]}) for {age:.0f} s"
        else:
            link = f"connecting… {age:.0f} s"
        text = f"telemetry {self.address} · {link}"
        if n_ok:
            # last measured force; held (not current) unless the link is OK
            text += f" · measured F = {force:.2f} N" + ("" if state == self.OK else " (held)")
        return text

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2.0 * self.timeout + 1.0)


SOURCES = {"sine": SineSource, "ramp": RampSource, "manual": ManualSource,
           "telemetry": TelemetrySource}
