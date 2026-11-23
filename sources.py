"""Input sources delivering current control values {channel: value}.

read() must be non-blocking and return the *latest* value, so the FE loop
never queues behind the experiment. A real device/socket source plugs in by
implementing the same interface.
"""
import math
import time


class Source:
    def __init__(self, channels):
        self.channels = list(channels)

    def read(self):
        raise NotImplementedError

    def info(self):
        """Short extra text for the status line (e.g. a measured value or a
        connection warning). Synthetic sources have nothing to add."""
        return ""

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


class RampSource(Source):
    """Triangular ramp 0 -> amp -> 0 with given period."""

    def __init__(self, channels, amp=5.0, period=4.0):
        super().__init__(channels)
        self.amp, self.period, self.t0 = amp, period, time.perf_counter()

    def read(self):
        ph = ((time.perf_counter() - self.t0) / self.period) % 1.0
        v = self.amp * (2.0 * ph if ph < 0.5 else 2.0 * (1.0 - ph))
        return {c: v for c in self.channels}


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


SOURCES = {"sine": SineSource, "ramp": RampSource, "manual": ManualSource}
