"""Three-point bending of a simply supported rectangular beam."""
import numpy as np

from spec import ProblemSpec
from .base import Geometry, register
from .mesh_utils import breakpoints_grid, single_node


@register("beam3pb")
class ThreePointBendingBeam(Geometry):
    """Rectangular beam, two bottom supports, displacement-controlled load at top centre."""

    default_params = {
        "L": 200.0,      # beam length [mm]
        "H": 19.8,       # beam height [mm]
        "t": 9.84,       # thickness [mm]
        "span": 180.0,   # distance between supports [mm]
        "E": 8.6,        # Young's modulus [MPa]
        "nu": 0.15,      # Poisson's ratio
    }
    param_help = {
        "L": "beam length [mm]", "H": "beam height [mm]", "t": "thickness [mm]",
        "span": "support distance [mm]", "E": "Young's modulus [MPa]", "nu": "Poisson's ratio",
    }

    def build(self, h, **overrides):
        p = self.params(**overrides)
        L, H, span = p["L"], p["H"], p["span"]
        a = 0.5 * (L - span)                     # overhang beyond supports
        if a < 0:
            raise ValueError("span must not exceed beam length")
        xc = 0.5 * L
        # breakpoints guarantee nodes at both supports and the load point
        coords, cells = breakpoints_grid([0.0, a, xc, L - a, L], [0.0, H], h)

        sup_l = single_node(coords, a, 0.0)
        sup_r = single_node(coords, L - a, 0.0)
        load = single_node(coords, xc, H)
        supports = np.concatenate([sup_l, sup_r])

        return ProblemSpec(
            coords=coords, cells=cells, cell_type="quad",
            thickness=p["t"], material={"E": p["E"], "nu": p["nu"]},
            # supports fix vertical displacement only; load point fixes horizontal
            fixed=[(supports, "v"), (load, "u")],
            # positive w = downward deflection of the load point
            controls={"w": [(load, "v", -1.0)]},
            groups={"supports": supports, "load": load},
            title=f"3-point bending  L={L:g}  H={H:g}  span={span:g} mm",
        )
