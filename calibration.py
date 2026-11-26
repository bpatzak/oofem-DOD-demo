"""Autocalibration of Young's modulus from the force measured on the rig.

For a linear elastic specimen loaded by a prescribed displacement, the
reaction is proportional to Young's modulus: F_sim(E) = (E / E0) * F_sim(E0).
If the measured force F_meas differs from F_sim, the modulus that matches it
for the current displacement is simply

    E* = E0 * |F_meas| / |F_sim(E0)|.

The calibrator works with the factor k = E / E0 (applied to the result by
Result.stiffness_scaled(), no re-solve needed):

* Deviation d = (|F_sim(E)| - |F_meas|) / |F_meas|, smoothed in time (time
  constant `tau`), so single noisy samples neither start nor stop anything.
  If the smoothed |d| exceeds `tol` (default 10 %), calibration starts.
* While calibrating, k moves continuously towards k* = |F_meas| / |F_sim(E0)|
  by first-order smoothing with the same time constant, which averages out
  the noise of the force signal: k += (1 - exp(-dt/tau)) * (k* - k).
* Once the smoothed |d| is below `settle` (default 1 %), calibration stops, so
  the modulus stays put instead of chasing noise, until |d| exceeds `tol` again.
* No update while there is no fresh measurement, or while the simulated force
  is below `min_force`: near zero load the force ratio is dominated by noise
  and offsets.

Magnitudes are compared, so the sign convention of the rig's force channel
does not matter.
"""
import math
import time


class AutoCalibrator:
    # states for the status line
    OFF, PAUSED, ADJUSTING, OK = "off", "paused", "adjusting", "ok"

    def __init__(self, E0, tol=0.10, settle=0.01, tau=1.0, min_force=0.0, enabled=True):
        """E0: modulus the FE model was built with [MPa]; min_force [N] refers to
        the simulated reaction at E0."""
        if not 0.0 < settle < tol:
            raise ValueError("need 0 < settle < tol")
        self.E0 = float(E0)
        self.tol, self.settle, self.tau = float(tol), float(settle), float(tau)
        self.min_force = float(min_force)
        self.enabled = enabled
        self.factor = 1.0          # k = E / E0
        self.calibrating = False   # True between exceeding tol and reaching settle
        self.deviation = None      # smoothed relative force deviation d, None if unknown
        self.reason = "no data yet"   # why the calibration waits (status line)
        self._t_last = None        # time of the last update with valid data

    @property
    def E(self):
        """Current (calibrated) Young's modulus [MPa]."""
        return self.E0 * self.factor

    @property
    def state(self):
        if not self.enabled:
            return self.OFF
        if self.deviation is None:
            return self.PAUSED       # no fresh measurement or load too small
        return self.ADJUSTING if self.calibrating else self.OK

    def update(self, sim_reactions, measured, now=None):
        """Compare forces and adapt the factor; return True if it changed.

        sim_reactions: {channel: simulated reaction at E0} (unscaled result),
        measured: {channel: measured force} or None (no fresh measurement).
        """
        if not self.enabled:
            return False
        now = time.monotonic() if now is None else now
        if not measured:
            return self._wait("no fresh measurement")
        chans = [c for c in measured if c in sim_reactions]
        f_sim0 = sum(abs(sim_reactions[c]) for c in chans)     # at E0
        f_meas = sum(abs(measured[c]) for c in chans)
        if not chans or f_sim0 < self.min_force or f_meas <= 0.0:
            return self._wait("load too small")

        self.reason = ""
        # time-based smoothing weight: same behaviour at any frame or sample
        # rate; dt is capped, so a resume after a pause does not jump at once
        if self._t_last is None:
            alpha = 1.0                     # first sample after a pause
        else:
            dt = min(now - self._t_last, 0.25)
            alpha = 1.0 - math.exp(-dt / self.tau) if self.tau > 0 else 1.0
        self._t_last = now

        changed = False
        if self.calibrating:
            # follow the modulus that matches the measured force
            target = f_meas / f_sim0
            new = self.factor + alpha * (target - self.factor)
            changed = new != self.factor
            self.factor = new

        # smoothed deviation of the simulated force (current E) from the measured one
        d = (self.factor * f_sim0 - f_meas) / f_meas
        self.deviation = d if self.deviation is None else self.deviation + alpha * (d - self.deviation)
        if not self.calibrating and abs(self.deviation) > self.tol:
            self.calibrating = True         # starts adapting from the next sample
        elif self.calibrating and abs(self.deviation) < self.settle:
            self.calibrating = False
        return changed

    def _wait(self, reason):
        """No usable data now: keep the factor, restart the smoothing clock."""
        self.deviation, self.reason, self._t_last = None, reason, None
        return False

    def describe(self):
        """Short status text, e.g. 'E = 52.3 MPa (autocal: adjusting, ΔF = +12.4 %)'.

        ΔF is the smoothed deviation of the simulated force from the measured one.
        """
        text = f"E = {self.E:.4g} MPa"
        if self.state == self.OFF:
            return text
        if self.state == self.PAUSED:
            return f"{text} (autocal: paused, {self.reason})"
        return f"{text} (autocal: {self.state}, ΔF = {100.0 * self.deviation:+.1f} %)"
