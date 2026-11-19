"""Four-point bending of a simply supported rectangular beam.

Also serves as the example of adding a geometry: one file, registered by the
decorator and imported in geometries/__init__.py; nothing else changes.
"""
import numpy as np

from spec import ProblemSpec
from .base import Geometry, register
from .mesh_utils import breakpoints_grid, single_node


@register("beam4pb")
class FourPointBendingBeam(Geometry):
    """Rectangular beam, two bottom supports, two top load points driven by the same displacement."""

    default_params = {
        "L": 200.0,          # beam length [mm]
        "H": 19.8,           # beam height [mm]
        "t": 9.84,           # thickness [mm]
        "span": 180.0,       # distance between supports [mm]
        "load_span": 60.0,   # distance between the two load points [mm]
        "E": 8.6,            # Young's modulus [MPa]
        "nu": 0.15,          # Poisson's ratio
    }
    param_help = {
        "L": "beam length [mm]", "H": "beam height [mm]", "t": "thickness [mm]",
        "span": "support distance [mm]", "load_span": "distance of load points [mm]",
        "E": "Young's modulus [MPa]", "nu": "Poisson's ratio",
    }

    def build(self, h, **overrides):
        p = self.params(**overrides)
        L, H, span, ls = p["L"], p["H"], p["span"], p["load_span"]
        a, xc = 0.5 * (L - span), 0.5 * L
        if not 0 <= ls < span <= L:
            raise ValueError("require 0 <= load_span < span <= L")
        xl, xr = xc - 0.5 * ls, xc + 0.5 * ls
        coords, cells = breakpoints_grid([0.0, a, xl, xc, xr, L - a, L], [0.0, H], h)

        supports = np.concatenate([single_node(coords, a, 0.0), single_node(coords, L - a, 0.0)])
        loads = np.concatenate([single_node(coords, xl, H), single_node(coords, xr, H)])
        # horizontal rigid-body motion: fix u at the top-centre node (symmetry point)
        centre = single_node(coords, xc, H)

        return ProblemSpec(
            coords=coords, cells=cells, cell_type="quad",
            thickness=p["t"], material={"E": p["E"], "nu": p["nu"]},
            fixed=[(supports, "v"), (centre, "u")],
            controls={"w": [(loads, "v", -1.0)]},
            groups={"supports": supports, "load": loads},
            title=f"4-point bending  L={L:g}  H={H:g}  span={span:g}  load span={ls:g} mm",
        )
