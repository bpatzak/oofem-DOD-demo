"""Generic OOFEM linear plane-stress model built from a ProblemSpec.

Nothing in here is specific to a particular specimen. The model is created
once; each call to :meth:`FEModel.solve` only updates the prescribed values of
the control boundary conditions and runs one LinearStatic step. LinearStatic
assembles and factorizes the stiffness matrix in its first step only, so later
steps cost one right-hand-side assembly plus a back-substitution.
"""
import time

import numpy as np

from oofem_env import import_oofem
from spec import DOFS, Result

ELEMENT_FACTORY = {"quad": "planestress2d", "tri": "trplanestress2d"}


class FEModel:
    def __init__(self, spec, mode="solve", oofem_build=None):
        """
        mode: "solve" -- run an OOFEM solve for every new input,
              "scale" -- precompute unit solutions per control channel and
                         superpose them (exact for the linear problem).
        """
        if mode not in ("solve", "scale"):
            raise ValueError(f"unknown mode {mode}")
        self.spec = spec
        self.mode = mode
        self.oofempy, self.util = import_oofem(oofem_build)
        self.timing = {}      # timings of the last solve [s]
        t0 = time.perf_counter()
        self._build()
        self.timing["setup"] = time.perf_counter() - t0
        self._unit = None
        if mode == "scale":
            self._unit = {c: self._solve_oofem({k: (1.0 if k == c else 0.0) for k in spec.controls})
                          for c in spec.controls}

    # ------------------------------------------------------------------ build
    def _build(self):
        oofempy, util, spec = self.oofempy, self.util, self.spec
        problem = oofempy.linearStatic(nSteps=1, outFile="/dev/null", suppress_output=True)
        domain = oofempy.domain(1, 1, problem, oofempy.domainType._2dPlaneStressMode,
                                tstep_all=True, dofman_all=True, element_all=True)
        problem.setDomain(1, domain, True)

        ltf = oofempy.constantFunction(1, domain, f_t=1.0)

        # BC 1: homogeneous (fixed) DOFs; one further BC per (channel, sign),
        # since setPrescribedValue() assigns the same value to all DOFs of a BC.
        bcs = [oofempy.boundaryCondition(1, domain, loadTimeFunction=1, prescribedValue=0.0)]
        self._control_bcs = {}    # channel -> list of (bc object, sign)
        dof_bc = np.zeros((spec.n_nodes, len(DOFS)), dtype=int)   # 0 = free

        def assign(ids, dof, bcnum):
            j = DOFS.index(dof)
            clash = dof_bc[ids, j][(dof_bc[ids, j] != 0) & (dof_bc[ids, j] != bcnum)]
            if clash.size:
                raise ValueError(f"DOF {dof} prescribed by more than one condition")
            dof_bc[ids, j] = bcnum

        for ids, dof in spec.fixed:
            assign(np.atleast_1d(ids), dof, 1)
        for ch, items in spec.controls.items():
            by_sign = {}
            for ids, dof, sign in items:
                if sign not in by_sign:
                    num = len(bcs) + 1
                    bc = oofempy.boundaryCondition(num, domain, loadTimeFunction=1, prescribedValue=0.0)
                    bcs.append(bc)
                    by_sign[sign] = bc
                    self._control_bcs.setdefault(ch, []).append((bc, sign))
                assign(np.atleast_1d(ids), dof, by_sign[sign].number)

        nodes = [oofempy.node(i + 1, domain, coords=(float(x), float(y), 0.0),
                              bc=tuple(int(b) for b in dof_bc[i]))
                 for i, (x, y) in enumerate(spec.coords)]

        mat = oofempy.isoLE(1, domain, d=1.0, E=spec.material["E"], n=spec.material["nu"], tAlpha=0.0)
        cs = oofempy.simpleCS(1, domain, thick=spec.thickness, material=1, set=1)

        make_elem = getattr(oofempy, ELEMENT_FACTORY[spec.cell_type])
        elems = [make_elem(i + 1, domain, nodes=tuple(int(n) + 1 for n in c))
                 for i, c in enumerate(spec.cells)]
        eset = oofempy.createSet(1, domain, elements=tuple(range(1, spec.n_cells + 1)))

        util.setupDomain(domain, nodes, elems, (cs,), (mat,), bcs, (), (ltf,), (eset,))

        # In-memory VTK export: displacement (primvar 1), stress/strain (IST 1, 4)
        # as element averages (cellvars) and smoothed nodal values (vars).
        self._export = oofempy.vtkmemory(1, problem, domain_all=True, tstep_all=True,
                                         dofman_all=True, element_all=True,
                                         vars=(1, 4), primvars=(1,), cellvars=(1, 4),
                                         stype=1, pythonExport=1)

        problem.checkProblemConsistency()
        problem.init()
        problem.postInitialize()
        problem.initMetaStepAttributes(problem.giveMetaStep(1))

        self.problem, self.domain = problem, domain
        # The domain takes ownership of these objects, but Python must keep
        # its references alive as well.
        self._keep = (ltf, bcs, nodes, mat, cs, elems, eset)
        self._geom = None   # (coords, cells) in export ordering, filled lazily

    # ------------------------------------------------------------------ solve
    def solve(self, values):
        """Solve for given control values {channel: value}; return Result."""
        if self.mode == "scale":
            t0 = time.perf_counter()
            res = None
            for ch, unit in self._unit.items():
                part = unit.scaled(values.get(ch, 0.0))
                res = part if res is None else res + part
            self.timing.update(solve=time.perf_counter() - t0, extract=0.0)
            return res
        return self._solve_oofem(values)

    def _solve_oofem(self, values):
        oofempy, problem = self.oofempy, self.problem
        t0 = time.perf_counter()
        for ch, lst in self._control_bcs.items():
            v = float(values.get(ch, 0.0))
            for bc, sign in lst:
                bc.setPrescribedValue(sign * v)
        problem.preInitializeNextStep()
        problem.giveNextStep()
        ts = problem.giveCurrentStep()
        problem.initializeYourself(ts)
        problem.solveYourselfAt(ts)
        problem.updateYourself(ts)      # recomputes stresses at Gauss points
        problem.terminate(ts)           # runs export modules (vtkmemory)
        t1 = time.perf_counter()
        res = self._extract(values)
        self.timing.update(solve=t1 - t0, extract=time.perf_counter() - t1)
        return res

    def _extract(self, values):
        ist = self.oofempy.InternalStateType
        reg = self._export.getExportRegions()[0]
        if self._geom is None:
            coords = np.array(reg.getVertices())[:, :2].copy()
            # NOTE: reg.getCellConnectivity() is not used: its pybind11 wrapper
            # (intarray2numpy in oofem.cpp) returns a view of a freed temporary
            # and yields garbage. Read the connectivity per cell instead; this
            # is done once only.
            cells = np.array([[int(n) - 1 for n in reg.giveCellConnectivity(i + 1)]
                              for i in range(self.spec.n_cells)], dtype=int)
            self._geom = (coords, cells)
        coords, cells = self._geom
        disp = np.array(reg.getPrimaryVertexValues(self.oofempy.UnknownType.DisplacementVector))[:, :2]
        return Result(coords=coords, cells=cells, disp=disp,
                      cell_stress=np.array(reg.getCellValues(ist.IST_StressTensor)),
                      cell_strain=np.array(reg.getCellValues(ist.IST_StrainTensor)),
                      node_stress=np.array(reg.getInternalVertexValues(ist.IST_StressTensor)),
                      node_strain=np.array(reg.getInternalVertexValues(ist.IST_StrainTensor)),
                      inputs=dict(values))

    @property
    def n_equations(self):
        return 2 * self.spec.n_nodes - sum(len(np.atleast_1d(i)) for i, _ in self.spec.fixed) \
            - sum(len(np.atleast_1d(i)) for items in self.spec.controls.values() for i, _, _ in items)
