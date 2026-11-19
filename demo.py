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

from fields import FIELDS, color_limits
from geometries import GEOMETRIES
from sources import SOURCES, ManualSource
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
    ap.add_argument("--mode", default="solve", choices=("solve", "scale"),
                    help="solve: OOFEM solve each frame; scale: superpose precomputed unit solutions")
    ap.add_argument("--warp", type=float, default=1.0, help="displacement magnification for display")
    ap.add_argument("--smooth", action="store_true", help="smooth nodal colouring (PyVista)")
    ap.add_argument("--bands", type=int, default=None, help="number of discrete colour bands (PyVista)")
    ap.add_argument("--no-edges", action="store_true", help="hide element edges")
    ap.add_argument("--report", type=int, default=50, help="print timing statistics every N frames")
    ap.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = until window closed)")
    ap.add_argument("--no-render", action="store_true", help="run the loop without graphics")
    ap.add_argument("--bench", type=int, default=0, metavar="N",
                    help="benchmark N solves for each mesh size in --bench-h and exit")
    ap.add_argument("--bench-h", default="5,2,1,0.5", help="mesh sizes for --bench")
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
        m = FEModel(spec, mode="solve", oofem_build=args.oofem_build)
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
    model = FEModel(spec, mode=args.mode, oofem_build=args.oofem_build)
    channels = list(spec.controls)
    print(f"{spec.title}: {spec.n_nodes} nodes, {spec.n_cells} elements, "
          f"{model.n_equations} equations, setup {1e3 * model.timing['setup']:.0f} ms")

    field = FIELDS[args.field]
    max_inputs = {c: args.max for c in channels}
    clim = color_limits(field, model.unit_results(), max_inputs,   # also factorizes K
                        smooth=args.smooth and args.backend == "pv")
    print(f"field {field.label}: fixed colour range {clim[0]:.4g} .. {clim[1]:.4g} {field.unit} "
          f"for input up to {args.max:g}")

    if args.source == "manual":
        source = ManualSource(channels, amp=args.max)
    else:
        source = SOURCES[args.source](channels, amp=args.max, period=args.period)

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
        renderer.setup(Scene(coords=unit.coords, cells=unit.cells, cell_type=spec.cell_type,
                             field_label=f"{field.label} [{field.unit}]", clim=clim,
                             signed=field.symmetric, title=spec.title, groups=spec.groups,
                             warp_margin=umax, theme=THEMES[args.theme], smooth=args.smooth,
                             show_edges=not args.no_edges, slider=slider))

    stats = Stats()
    frame, t_frame = 0, 0.0
    while renderer is None or renderer.is_open():
        t0 = time.perf_counter()
        inputs = source.read()
        t1 = time.perf_counter()
        res = model.solve(inputs)
        t2 = time.perf_counter()
        cvals, nvals = field.evaluate(res)
        t3 = time.perf_counter()
        if renderer is not None:
            vin = ", ".join(f"{k} = {v:6.3f} mm" for k, v in inputs.items())
            status = (f"{vin}   {field.label} max {np.abs(cvals).max():.4g} {field.unit}"
                      f"   frame {1e3 * t_frame:5.1f} ms")
            renderer.update(res.coords + args.warp * res.disp, cvals, nvals, status)
        t4 = time.perf_counter()
        t_frame = t4 - t0
        stats.add(input=t1 - t0, solve=model.timing["solve"], extract=model.timing["extract"],
                  post=t3 - t2, draw=t4 - t3, frame=t_frame)
        frame += 1
        if args.report and frame % args.report == 0:
            print(f"[{frame:6d}] {stats.report()}", flush=True)
        if args.frames and frame >= args.frames:
            break
    if renderer is not None:
        renderer.close()


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
