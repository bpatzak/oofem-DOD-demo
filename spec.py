"""Neutral data contracts between geometry, FE model and visualization.

A geometry produces a :class:`ProblemSpec` (pure numpy, no OOFEM objects);
the FE model consumes it and returns :class:`Result` objects.  Renderers only
see arrays, so new geometries need no changes in solver or plotting code.
"""
from dataclasses import dataclass, field

import numpy as np

# Supported DOF names for 2D plane stress
DOFS = ("u", "v")


@dataclass
class ProblemSpec:
    coords: np.ndarray                 # (nn, 2) node coordinates
    cells: np.ndarray                  # (ne, nen) 0-based connectivity, counter-clockwise
    thickness: float
    material: dict                     # {"E": ..., "nu": ...}
    cell_type: str = "quad"            # "quad" -> planestress2d, "tri" -> trplanestress2d
    # DOFs prescribed to zero: list of (node indices, dof name)
    fixed: list = field(default_factory=list)
    # Control channels driven by the input source:
    #   name -> list of (node indices, dof name, sign)
    # prescribed value on these DOFs = sign * channel value
    controls: dict = field(default_factory=dict)
    # Named node groups (e.g. supports, load points) for plot markers
    groups: dict = field(default_factory=dict)
    title: str = ""

    @property
    def n_nodes(self):
        return self.coords.shape[0]

    @property
    def n_cells(self):
        return self.cells.shape[0]


@dataclass
class Result:
    coords: np.ndarray        # (nn, 2) undeformed node coordinates (export ordering)
    cells: np.ndarray         # (ne, nen) 0-based connectivity (export ordering)
    disp: np.ndarray          # (nn, 2) nodal displacements
    cell_stress: np.ndarray   # (ne, 9) element-averaged stress, full 3x3 row-major
    cell_strain: np.ndarray   # (ne, 9) element-averaged strain tensor (not engineering shear)
    node_stress: np.ndarray   # (nn, 9) smoothed nodal stress (for contours)
    node_strain: np.ndarray   # (nn, 9) smoothed nodal strain
    inputs: dict              # channel values this result corresponds to
    # Reaction per control channel: the force the actuator applies, i.e. the sum
    # of nodal forces on the controlled DOFs weighted by the channel signs;
    # positive in the direction of a positive channel value.
    reactions: dict = field(default_factory=dict)

    def scaled(self, factor):
        """Return result multiplied by a scalar (valid for linear problems)."""
        return Result(self.coords, self.cells, factor * self.disp,
                      factor * self.cell_stress, factor * self.cell_strain,
                      factor * self.node_stress, factor * self.node_strain,
                      {k: factor * v for k, v in self.inputs.items()},
                      {k: factor * v for k, v in self.reactions.items()})

    def __add__(self, other):
        def add(a, b):
            out = dict(a)
            for k, v in b.items():
                out[k] = out.get(k, 0.0) + v
            return out
        return Result(self.coords, self.cells, self.disp + other.disp,
                      self.cell_stress + other.cell_stress,
                      self.cell_strain + other.cell_strain,
                      self.node_stress + other.node_stress,
                      self.node_strain + other.node_strain,
                      add(self.inputs, other.inputs), add(self.reactions, other.reactions))
