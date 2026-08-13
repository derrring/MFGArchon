"""Equivalence tests for B1.5b.4: HJBFDMSolver + FixedPointSolver rename.

Last iteration (4/6) of the B1.5b series. Legacy `damping_factor` ctor kwarg
continues to work via `@deprecated_parameter` + body redirect. Silent
`@property` aliases on both classes keep `solver.damping_factor` attribute
reads working without warning flooding.

Removal scheduled for v0.25.0.
"""

from __future__ import annotations

import warnings

import numpy as np

from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.utils.numerical.nonlinear_solvers import FixedPointSolver


def _make_problem():
    H = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: m,
        coupling_dm=lambda m: 1.0,
    )
    c = MFGComponents(
        hamiltonian=H,
        u_terminal=lambda x: 0.0,
        m_initial=lambda x: 1.0,
    )
    return MFGProblem(
        geometry=TensorProductGrid(
            bounds=[(0.0, 1.0)], Nx_points=[11 + 1], boundary_conditions=no_flux_bc(dimension=1)
        ),
        T=0.2,
        Nt=5,
        sigma=0.3,
        components=c,
    )


# ------------------------------------------------------------------
# FixedPointSolver (standalone, utils/numerical/nonlinear_solvers.py)
# ------------------------------------------------------------------


class TestFixedPointSolverAlias:
    def test_canonical_kwarg_works(self):
        s = FixedPointSolver(relaxation=0.7)
        assert s.relaxation == 0.7

    def test_legacy_kwarg_redirects(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            s = FixedPointSolver(damping_factor=0.7)
        assert s.relaxation == 0.7

    def test_legacy_and_canonical_equivalent(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            legacy = FixedPointSolver(damping_factor=0.6)
        canonical = FixedPointSolver(relaxation=0.6)
        assert legacy.relaxation == canonical.relaxation == 0.6

    def test_legacy_kwarg_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            FixedPointSolver(damping_factor=0.5)
            dep = [x for x in w if issubclass(x.category, DeprecationWarning) and "damping_factor" in str(x.message)]
        assert len(dep) >= 1

    def test_canonical_kwarg_silent(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            FixedPointSolver(relaxation=0.5)
            dep = [x for x in w if issubclass(x.category, DeprecationWarning) and "damping_factor" in str(x.message)]
        assert len(dep) == 0

    def test_damping_factor_property_silent(self):
        s = FixedPointSolver(relaxation=0.4)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assert s.damping_factor == 0.4
            dep = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep) == 0

    def test_solve_still_works_with_legacy_kwarg(self):
        """End-to-end smoke: legacy kwarg produces a functional solver."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            solver = FixedPointSolver(damping_factor=0.5, max_iterations=50)
        # Solve x = cos(x), fixed point ~0.739
        x, info = solver.solve(lambda x: np.cos(x), x0=np.asarray(0.5))
        assert info.converged
        assert abs(float(x) - 0.739) < 1e-3


# ------------------------------------------------------------------
# HJBFDMSolver
# ------------------------------------------------------------------


class TestHJBFDMSolverAlias:
    def test_canonical_kwarg_works(self):
        p = _make_problem()
        s = HJBFDMSolver(p, relaxation=0.7)
        assert s.relaxation == 0.7

    def test_legacy_kwarg_redirects(self):
        p = _make_problem()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            s = HJBFDMSolver(p, damping_factor=0.7)
        assert s.relaxation == 0.7

    def test_legacy_kwarg_warns(self):
        p = _make_problem()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            HJBFDMSolver(p, damping_factor=0.5)
            dep = [x for x in w if issubclass(x.category, DeprecationWarning) and "damping_factor" in str(x.message)]
        assert len(dep) >= 1

    def test_damping_factor_property_silent(self):
        p = _make_problem()
        s = HJBFDMSolver(p, relaxation=0.6)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assert s.damping_factor == 0.6
            dep = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep) == 0

    def test_internal_fixed_point_solver_receives_canonical_kwarg(self):
        """When HJBFDMSolver constructs an internal FixedPointSolver for nD,
        it should forward `relaxation`, not `damping_factor`."""
        # 1D never builds the internal solver at all (hjb_fdm.py guards it with
        # `if self.dimension > 1`), so `HJBFDMSolver(p1D)` has no `nonlinear_solver`
        # attribute; on 1D this only checks that the canonical kwarg is silent.
        p = _make_problem()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            HJBFDMSolver(p, relaxation=0.5)
            # No DeprecationWarning should fire from inside HJBFDMSolver
            # (which previously would have, if it still passed damping_factor internally)
            dep = [x for x in w if issubclass(x.category, DeprecationWarning) and "damping_factor" in str(x.message)]
        assert len(dep) == 0

        # The forwarding this test is named for needs BOTH dimension > 1 and
        # solver_type="fixed_point" (the default is "newton", which builds a NewtonSolver with
        # no relaxation at all).  Under those two conditions the value must arrive intact.
        H = SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=1.0),
            coupling=lambda m: m,
            coupling_dm=lambda m: 1.0,
        )
        p2d = MFGProblem(
            geometry=TensorProductGrid(
                bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[8, 8], boundary_conditions=no_flux_bc(dimension=2)
            ),
            T=0.2,
            Nt=5,
            sigma=0.3,
            components=MFGComponents(hamiltonian=H, u_terminal=lambda x: 0.0, m_initial=lambda x: 1.0),
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            s2d = HJBFDMSolver(p2d, relaxation=0.37, solver_type="fixed_point")
            dep2 = [x for x in w if issubclass(x.category, DeprecationWarning) and "damping_factor" in str(x.message)]
        assert len(dep2) == 0
        assert isinstance(s2d.nonlinear_solver, FixedPointSolver)
        assert s2d.nonlinear_solver.relaxation == 0.37
