"""Scalar fields derived from a Result for colour plots.

Tensors from the OOFEM VTK export are full 3x3, row-major:
    [xx, xy, xz, yx, yy, yz, zx, zy, zz]
Strain is the tensor strain, so engineering shear gamma_xy = 2 * eps_xy.
Each field returns (cell_values, node_values); node values are the smoothed
(recovered) nodal values, used for contour plots.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np

XX, XY, YY, ZZ = 0, 1, 4, 8


def _von_mises(s):
    sx, sy, sz, txy = s[:, XX], s[:, YY], s[:, ZZ], s[:, XY]
    return np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3.0 * txy ** 2)


def _comp(i, scale=1.0):
    return lambda t: scale * t[:, i]


@dataclass
class Field:
    label: str
    unit: str
    kind: str          # "stress" | "strain" | "disp"
    fn: Callable       # applied to the (n, 9) tensor or (n, 2) displacement array
    symmetric: bool    # colour range symmetric around zero (signed quantities)

    def evaluate(self, res):
        if self.kind == "stress":
            return self.fn(res.cell_stress), self.fn(res.node_stress)
        if self.kind == "strain":
            return self.fn(res.cell_strain), self.fn(res.node_strain)
        nodal = self.fn(res.disp)
        return nodal[res.cells].mean(axis=1), nodal


FIELDS = {
    "sxx": Field("σxx", "MPa", "stress", _comp(XX), True),
    "syy": Field("σyy", "MPa", "stress", _comp(YY), True),
    "sxy": Field("τxy", "MPa", "stress", _comp(XY), True),
    "mises": Field("σ von Mises", "MPa", "stress", _von_mises, False),
    "exx": Field("εxx", "-", "strain", _comp(XX), True),
    "eyy": Field("εyy", "-", "strain", _comp(YY), True),
    "gxy": Field("γxy", "-", "strain", _comp(XY, 2.0), True),
    "ux": Field("ux", "mm", "disp", lambda d: d[:, 0], True),
    "uy": Field("uy", "mm", "disp", lambda d: d[:, 1], True),
    "umag": Field("|u|", "mm", "disp", lambda d: np.hypot(d[:, 0], d[:, 1]), False),
}


def color_limits(field, unit_results, max_inputs, smooth=False):
    """Fixed colour range for the expected input range (no flicker).

    For a linear model the extreme value of a signed field over the input range
    [-max, max] of each channel is bounded by sum(|unit field| * max). For a
    single channel the bound is exact. Only the values actually displayed
    (cell or smoothed nodal) are considered.
    """
    bound = None
    for ch, res in unit_results.items():
        c, n = field.evaluate(res)
        m = np.abs(n if smooth else c).max() * abs(max_inputs.get(ch, 1.0))
        bound = m if bound is None else bound + m
    bound = bound or 1.0
    return (-bound, bound) if field.symmetric else (0.0, bound)
