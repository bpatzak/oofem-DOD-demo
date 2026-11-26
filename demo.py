#!/usr/bin/env python3
"""Real-time FE demo: prescribed displacement in -> deformed shape + stress/strain out.

Examples:
    python demo.py                                   # beam3pb, PyVista, sine input
    python demo.py --backend mpl --field mises --theme dark
    python demo.py --h 1 --smooth --bands 12         # banded contour look
    python demo.py --source manual                   # slider controls the displacement
    python demo.py --bench 100                       # solve timings for several mesh sizes
    python demo.py --list                            # available geometries and fields
"""
import argparse
import json
import time

import numpy as np

from calibration import AutoCalibrator
from fields import FIELDS, color_limits
from geometries import GEOMETRIES
from sources import (DEFAULT_TELEMETRY_URL, SOURCES, TELEMETRY_PORT, ManualSource, Source,
                     TelemetrySource)
from viz_base import THEMES, Scene


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--geometry", default="beam3pb", help="specimen geometry (see --list)")
    ap.add_argument("--param", action="append", default=[], metavar="K=V",
                    help="geometry parameter override, repeatable (e.g. --param span=180)")
    ap.add_argument("--config", help="JSON file with {'geometry':..., 'params':{...}, other options}")
    ap.add_argument("--h", type=float, default=2.0, help="target element size [mm]")
    ap.add_argument("--max", type=float, default=5.0,
                    help="maximum prescribed displacement [mm]; fixes the colour scale "
                         "and the amplitude of the sine/ramp/manual source")
    ap.add_argument("--field", default="sxx", choices=sorted(FIELDS))
    ap.add_argument("--backend", default="pv", choices=("pv", "mpl"))
    ap.add_argument("--theme", default="light", choices=sorted(THEMES))
    ap.add_argument("--source", default="sine", choices=sorted(SOURCES))
    ap.add_argument("--period", type=float, default=6.0, help="period of sine/ramp source [s]")
    # --source telemetry: displacement measured on the rig (Experiment/server.py)
    # URL or IP shortcut, not both (the shortcut would silently override the URL)
    where = ap.add_mutually_exclusive_group()
    where.add_argument("--telemetry-url", default=DEFAULT_TELEMETRY_URL,
                       help="telemetry /api/values URL (default %(default)s)")
    where.add_argument("--telemetry-ip", metavar="IP",
                       help=f"machine PC address; shortcut for "
                            f"--telemetry-url http://IP:{TELEMETRY_PORT}/api/values")
    ap.add_argument("--telemetry-rate", type=float, default=15.0,
                    help="telemetry polling frequency [Hz] (default 15)")
    ap.add_argument("--telemetry-scale", type=float, default=1.0,
                    help="channel value = scale * measured displacement; -1 flips the sign")
    # autocalibration of Young's modulus from the measured force (telemetry source)
    ap.add_argument("--autocalibrate", action="store_true",
                    help="rescale Young's modulus continuously so that the simulated force "
                         "follows the force measured on the rig")
    ap.add_argument("--autocalib-tol", type=float, default=0.10, metavar="TOL",
                    help="start calibrating when the force deviation exceeds TOL "
                         "(default 0.10 = 10 %%); stops below 1 %%")
    ap.add_argument("--autocalib-tau", type=float, default=1.0, metavar="S",
                    help="smoothing time constant of the calibration [s] (default 1)")
    ap.add_argument("--autocalib-min", type=float, default=0.1, metavar="FRAC",
                    help="calibrate only while the simulated force exceeds FRAC times the "
                         "force at --max (default 0.1); small loads are dominated by noise")
    ap.add_argument("--mode", default="solve", choices=("solve", "scale"),
                    help="solve: OOFEM solve each frame; scale: superpose precomputed unit solutions")
    ap.add_argument("--skip-unchanged", type=float, default=None, metavar="TOL",
                    help="do not re-solve while every input differs from the last solved "
                         "value by at most TOL (input units, e.g. 0.001 mm); off by default")
    ap.add_argument("--warp", type=float, default=1.0, help="displacement magnification for display")
    ap.add_argument("--smooth", action="store_true", help="smooth nodal colouring (PyVista)")
    ap.add_argument("--bands", type=int, default=None, help="number of discrete colour bands (PyVista)")
    ap.add_argument("--no-mesh", action="store_true", help="do not draw the element edges")
    ap.add_argument("--logo-height", type=int, default=46, help="height of the logos in pixels")
    ap.add_argument("--mesh-alpha", type=float, default=0.35,
                    help="opacity of the element edges, 0..1 (default 0.35)")
    ap.add_argument("--report", type=int, default=50, help="print timing statistics every N frames")
    ap.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = until window closed)")
    ap.add_argument("--no-render", action="store_true", help="run the loop without graphics")
    ap.add_argument("--bench", type=int, default=0, metavar="N",
                    help="benchmark N solves for each mesh size in --bench-h and exit")
    ap.add_argument("--bench-h", default="5,2,1,0.5", help="mesh sizes for --bench")
    ap.add_argument("--no-profileopt", action="store_true",
                    help="disable OOFEM equation renumbering (profile optimization)")
    ap.add_argument("--oofem-build", help="directory with oofempy*.so (default: $OOFEM_BUILD)")
    ap.add_argument("--list", action="store_true", help="list geometries and fields and exit")
    args = ap.parse_args(argv)
    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
        args.geometry = cfg.pop("geometry", args.geometry)
        args.param = [f"{k}={v}" for k, v in cfg.pop("params", {}).items()] + args.param
        for k, v in cfg.items():
            setattr(args, k.replace("-", "_"), v)
    return args


def geometry_params(args):
    out = {}
    for kv in args.param:
        k, _, v = kv.partition("=")
        out[k.strip()] = v.strip()
    return out


# labels of the on-screen text lines (title on top, status at the bottom),
# padded to one width so the values start in one column; both renderers draw
# these lines in a monospace font
SCREEN_LABELS = ("geometry:", "model:", "input:", "simulation:", "stat:")
LABEL_WIDTH = max(map(len, SCREEN_LABELS)) + 1


def labelled(lines):
    """[(label, text), ...] -> one string, one padded label per line."""
    return "\n".join(f"{label:<{LABEL_WIDTH}}{text}" for label, text in lines)


class Stats:
    """Per-phase timing statistics (mean and 95th percentile, in ms)."""

    def __init__(self):
        self.data = {}

    def add(self, **t):
        for k, v in t.items():
            self.data.setdefault(k, []).append(v)

    def report(self, reset=True):
        parts = []
        for k, v in self.data.items():
            a = 1e3 * np.asarray(v)
            parts.append(f"{k} {a.mean():6.1f}/{np.percentile(a, 95):6.1f}")
        if reset:
            self.data = {}
        return "  ".join(parts) + "   [ms mean/p95]"


def benchmark(args):
    from fe_model import FEModel
    geom = GEOMETRIES[args.geometry]()
    params = geometry_params(args)
    rng = np.random.default_rng(0)
    print(f"{'h':>5} {'nodes':>7} {'elems':>7} {'eqs':>7} {'setup':>8} {'1st':>8} "
          f"{'solve':>8} {'p95':>8} {'extract':>8} {'total':>8}   [ms]")
    for h in (float(x) for x in args.bench_h.split(",")):
        spec = geom.build(h, **params)
        m = FEModel(spec, mode="solve", nodal=args.smooth, profileopt=not args.no_profileopt,
                    oofem_build=args.oofem_build)
        t0 = time.perf_counter()
        m.solve({c: 1.0 for c in spec.controls})          # assembles + factorizes K
        first = time.perf_counter() - t0
        ts, te = [], []
        for _ in range(args.bench):
            m.solve({c: rng.uniform(0, args.max) for c in spec.controls})
            ts.append(m.timing["solve"])
            te.append(m.timing["extract"])
        ts, te = 1e3 * np.array(ts), 1e3 * np.array(te)
        print(f"{h:5g} {spec.n_nodes:7d} {spec.n_cells:7d} {m.n_equations:7d} "
              f"{1e3 * m.timing['setup']:8.1f} {1e3 * first:8.1f} {ts.mean():8.2f} "
              f"{np.percentile(ts, 95):8.2f} {te.mean():8.2f} {(ts + te).mean():8.2f}")


def run(args):
    from fe_model import FEModel
    spec = GEOMETRIES[args.geometry]().build(args.h, **geometry_params(args))
    smooth = args.smooth and args.backend == "pv"
    model = FEModel(spec, mode=args.mode, nodal=smooth, profileopt=not args.no_profileopt,
                    oofem_build=args.oofem_build)
    channels = list(spec.controls)
    print(f"{spec.title}: {spec.n_nodes} nodes, {spec.n_cells} elements, "
          f"{model.n_equations} equations, setup {1e3 * model.timing['setup']:.0f} ms")

    field = FIELDS[args.field]
    max_inputs = {c: args.max for c in channels}
    clim = color_limits(field, model.unit_results(), max_inputs,   # also factorizes K
                        smooth=smooth)
    print(f"field {field.label}: fixed colour range {clim[0]:.4g} .. {clim[1]:.4g} {field.unit} "
          f"for input up to {args.max:g}")

    if args.source == "manual":
        source = ManualSource(channels, amp=args.max)
    elif args.source == "telemetry":
        url = (f"http://{args.telemetry_ip}:{TELEMETRY_PORT}/api/values"
               if args.telemetry_ip else args.telemetry_url)
        print(f"telemetry: polling {url} at {args.telemetry_rate:g} Hz, "
              f"scale {args.telemetry_scale:g}")
        source = TelemetrySource(channels, url=url, rate=args.telemetry_rate,
                                 scale=args.telemetry_scale)
    else:
        source = SOURCES[args.source](channels, amp=args.max, period=args.period)

    # Autocalibration: E0 is the modulus the model was built with; the result is
    # rescaled by calib.factor = E / E0 (exact for this linear problem, no re-solve).
    E0 = spec.material["E"]
    f_at_max = sum(abs(r) for r in model.unit_results()[channels[0]].reactions.values()) * args.max
    calib = AutoCalibrator(E0, tol=args.autocalib_tol, settle=min(0.01, 0.5 * args.autocalib_tol),
                           tau=args.autocalib_tau, min_force=args.autocalib_min * f_at_max,
                           enabled=args.autocalibrate)
    if args.autocalibrate:
        print(f"autocalibration: E0 = {E0:g} MPa, tolerance {100 * args.autocalib_tol:g} %, "
              f"tau {args.autocalib_tau:g} s, active above F = {calib.min_force:.3g} N")
        if type(source).measured is Source.measured:
            print(f"autocalibration: source '{args.source}' measures no force; "
                  f"it needs --source telemetry")
    clim0 = clim                 # colour range for E0; stress ranges scale with E
    clim_factor = 1.0            # calibration factor the colour range was set for

    renderer = None
    if not args.no_render:
        from viz_base import make_renderer
        renderer = make_renderer(args.backend)
        if args.backend == "pv" and args.bands:
            renderer.n_bands = args.bands
        slider = None
        if isinstance(source, ManualSource):
            slider = (f"{channels[0]} [mm]", 0.0, args.max,
                      lambda v: [source.set(c, v) for c in channels])
        unit = next(iter(model.unit_results().values()))
        umax = np.abs(unit.disp).max() * args.max * args.warp
        info = (f"{spec.n_nodes} nodes · {spec.n_cells} elements · {model.n_equations} DOFs · "
                f"h = {args.h:g} mm · E = {E0:g} MPa · setup {1e3 * model.timing['setup']:.0f} ms")
        renderer.setup(Scene(coords=unit.coords, cells=unit.cells, cell_type=spec.cell_type,
                             field_label=f"{field.label} [{field.unit}]", clim=clim,
                             signed=field.symmetric,
                             title=labelled([("geometry:", spec.title), ("model:", info)]),
                             groups=spec.groups,
                             warp_margin=umax, theme=THEMES[args.theme], smooth=smooth,
                             show_mesh=not args.no_mesh, mesh_alpha=args.mesh_alpha,
                             logo_height=args.logo_height,
                             slider=slider))

    stats = Stats()
    frame = 0
    # running averages shown on screen (exponential smoothing avoids flicker)
    ema = {"solve": 0.0, "draw": 0.0, "frame": 0.0}

    def ema_update(key, value, alpha=0.1):
        ema[key] = value if ema[key] == 0.0 else (1 - alpha) * ema[key] + alpha * value
    res, skipped = None, 0
    raw = None                   # last solved result for E0 (before calibration scaling)
    rescaled = 0                 # frames redrawn for a new E only (input unchanged)
    shown_factor = 1.0           # calibration factor of the result `res` (on screen)
    tol = args.skip_unchanged

    def unchanged(inputs):
        """True if no input moved more than tol from the last *solved* values.

        Comparing with the last solved (not the last read) values means slow
        drift below tol still accumulates and eventually triggers a solve.
        """
        return (tol is not None and res is not None
                and all(abs(v - res.inputs.get(k, 0.0)) <= tol for k, v in inputs.items()))

    def screen_state():
        """Texts that may change without a new solve: source and calibration status."""
        return source.status(), (calib.describe() if calib.enabled else "")

    def make_status(state):
        """Three-line status text: input source, last solved result `res`, timings.

        `state` is screen_state(): source.status() (e.g. the telemetry address,
        link state and the force measured on the rig) and the calibration text.
        Each line starts with a label (input:, simulation:, stat:).
        """
        src_line, calib_text = state
        # controlled displacement and the corresponding reaction per channel
        vin = "   ".join(
            f"{k} = {v:6.3f} mm   F{'' if len(res.inputs) == 1 else '_' + k} = "
            f"{res.reactions.get(k, float('nan')):7.4f} N"
            for k, v in res.inputs.items())
        fps = 1.0 / ema["frame"] if ema["frame"] > 0 else 0.0
        return labelled([
            ("input:", src_line),
            ("simulation:", f"{vin}   {field.label} max {np.abs(cvals).max():.4g} {field.unit}"
                            + (f"   {calib_text}" if calib_text else "")),
            ("stat:", f"solve {1e3 * ema['solve']:.1f} ms · render {1e3 * ema['draw']:.1f} ms · "
                      f"frame {1e3 * ema['frame']:.1f} ms ({fps:.0f} fps)"),
        ])

    # what is on screen: last drawn (points, cell values, node values) and screen_state()
    last_frame, shown_state = None, None
    calib_logged = calib.state   # last calibration state written to the console

    while renderer is None or renderer.is_open():
        t0 = time.perf_counter()
        inputs = source.read()
        t1 = time.perf_counter()
        resolve = not unchanged(inputs)
        if resolve:
            raw = model.solve(inputs)
        # Compare the simulated with the measured force and adapt E, on every
        # pass (time-based smoothing), also while the input is held: the force
        # may still change. A new E alone triggers a redraw only once it differs
        # by more than 0.1 % from the E on screen, so the loop does not spin.
        calib.update(raw.reactions, source.measured())
        recalibrated = abs(calib.factor / shown_factor - 1.0) > 1e-3
        if calib.state != calib_logged and calib.enabled:
            calib_logged = calib.state
            print(f"autocalibration: {calib.describe()}", flush=True)
        if not resolve and not recalibrated:
            # nothing to compute or redraw: keep the window responsive and
            # yield the CPU briefly instead of spinning
            skipped += 1
            if renderer is not None:
                state = screen_state()
                if state != shown_state and last_frame is not None:
                    # the input is frozen (e.g. the rig paused or disconnected) but
                    # the source or calibration status changed: refresh the text
                    shown_state = state
                    renderer.update(*last_frame, make_status(state))
                else:
                    renderer.idle()
            time.sleep(0.005)
            frame += 1
            if args.frames and frame >= args.frames:
                break
            continue
        # result for the current (calibrated) modulus; no new solve needed
        res = raw.stiffness_scaled(calib.factor)
        shown_factor = calib.factor
        if (renderer is not None and field.kind == "stress"
                and abs(calib.factor / clim_factor - 1.0) > 0.02):
            # stresses scale with E: keep the colour range in proportion, but only
            # update it on changes above 2 % so the colour bar does not flicker
            clim_factor = calib.factor
            renderer.set_clim(tuple(clim_factor * c for c in clim0))
        rescaled += not resolve
        t_solve = model.timing["solve"] if resolve else 0.0
        t_extract = model.timing["extract"] if resolve else 0.0
        t2 = time.perf_counter()
        cvals, nvals = field.evaluate(res)
        t3 = time.perf_counter()
        if renderer is not None:
            shown_state = screen_state()
            last_frame = (res.coords + args.warp * res.disp, cvals, nvals)
            renderer.update(*last_frame, make_status(shown_state))
        t4 = time.perf_counter()
        t_frame = t4 - t0
        ema_update("solve", t_solve + t_extract)
        ema_update("draw", t4 - t3)
        ema_update("frame", t_frame)
        stats.add(input=t1 - t0, solve=t_solve, extract=t_extract,
                  post=t3 - t2, draw=t4 - t3, frame=t_frame)
        frame += 1
        if args.report and frame % args.report == 0:
            note = f"   skipped (unchanged input): {skipped}" if tol is not None else ""
            print(f"[{frame:6d}] {stats.report()}{note}", flush=True)
        if args.frames and frame >= args.frames:
            break
    if renderer is not None:
        renderer.close()
    source.close()      # stops e.g. the telemetry polling thread
    print(f"frames {frame}: solved {frame - skipped - rescaled}, rescaled only (new E) {rescaled}, "
          f"skipped (unchanged input) {skipped}")
    if calib.enabled:
        print(f"autocalibration: final E = {calib.E:.4g} MPa (E0 = {E0:g} MPa, "
              f"factor {calib.factor:.4f})")


def main(argv=None):
    args = parse_args(argv)
    if args.list:
        for g in GEOMETRIES.values():
            print(g.describe())
        print("fields:", ", ".join(f"{k} ({f.label})" for k, f in FIELDS.items()))
        return
    if args.bench:
        benchmark(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
