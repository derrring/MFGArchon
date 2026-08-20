"""Issue #2021: `HJBWENOSolver` carried the dimensional solve three times, and one copy was dead.

`_solve_hjb_system_2d`, `_3d` and `_nd` were the same backward loop written out three times. They
differed in slice notation (`[:, :]` / `[:, :, :]` / `[...]`, and the last covers the others), in
logging, and in **three different sources for one array shape** — 2d read
`self.num_grid_points_*`, 3d read `M_density.shape[1:]`, nd read `U_terminal.shape`, with nothing
checking they agree.

**The 3D copy did not run at all.** Its first statement called `self._get_logger()`, which exists
nowhere in the MRO, so `HJBWENOSolver` in 3D raised `AttributeError` immediately and always had.
Twenty-three test files mention WENO and none is 3-D, which is what a duplicated path with no oracle
looks like from the outside: green.

WHAT THIS FILE PINS
-------------------
Per `domains/cs/_core.md`, a consolidation owes one owner, the deletion of the rivals, and a pin
against the **pre-consolidation output** — agreement between the paths afterwards is tautological,
since they route through the same code. The 2-D numbers below were captured before the change.
"""

from __future__ import annotations

import inspect

import pytest

import numpy as np

from mfgarchon.alg.numerical.hjb_solvers import hjb_weno
from mfgarchon.alg.numerical.hjb_solvers.hjb_weno import HJBWENOSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def _solve(dim, n, nt):
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0)] * dim, Nx_points=[n] * dim, boundary_conditions=no_flux_bc(dimension=dim)
    )
    problem = MFGProblem(
        geometry=grid,
        T=0.2,
        Nt=nt,
        sigma=0.3,
        coupling_coefficient=0.0,
        components=MFGComponents(
            m_initial=lambda q: 1.0,
            u_terminal=lambda q: 0.0,
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0),
                potential=lambda x, t: float(np.sum(np.cos(np.pi * np.atleast_1d(x)))),
            ),
        ),
    )
    solver = HJBWENOSolver(problem)
    x = np.linspace(0.0, 1.0, n)
    u_t = np.cos(np.pi * x)
    for _ in range(dim - 1):
        u_t = np.add.outer(u_t, np.cos(np.pi * x))
    shape = tuple([n] * dim)
    return solver, np.asarray(
        solver.solve_hjb_system(
            M_density=np.ones((nt + 1, *shape)),
            U_terminal=u_t.reshape(shape),
            U_coupling_prev=np.zeros((nt + 1, *shape)),
        )
    )


def test_the_2d_solve_still_reproduces_the_pre_consolidation_output():
    """The required pin: captured from the three-copy version BEFORE the rivals were deleted.

    Comparing the paths against each other instead would prove nothing -- after the consolidation
    they ARE the same code, so agreement is tautological and passes over a broken owner.
    """
    _solver, u = _solve(2, 11, 6)
    assert u.shape == (7, 11, 11)
    assert float(u.sum()) == pytest.approx(-4.825099898418e02, rel=1e-10)
    assert float(u[0].sum()) == pytest.approx(-1.298054011318e02, rel=1e-10)


def test_the_3d_solve_runs_at_all():
    """It did not. `_solve_hjb_system_3d` opened with `self._get_logger()`, absent from the MRO, so
    every 3-D WENO solve raised AttributeError. This is the whole of what changed for d = 3."""
    _solver, u = _solve(3, 7, 4)
    assert u.shape == (5, 7, 7, 7)
    assert np.all(np.isfinite(u)), "3-D WENO must produce a finite field"
    assert float(u.sum()) == pytest.approx(-1.942069586894e03, rel=1e-8)


def test_there_is_one_dimensional_solve_and_one_cfl_step():
    """The consolidation gate from `domains/cs/_core.md` is the implementation count, and nothing
    else: after the change the number of places computing the quantity must drop."""
    src = inspect.getsource(hjb_weno)
    for stem, expected in (
        ("def _solve_hjb_system_", 2),  # _1d (not a split path, and the only source_term carrier) + _nd
        ("def _compute_dt_stable_", 2),  # _1d + _nd
        ("def _step_", 1),  # _nd_split
    ):
        assert src.count(stem) == expected, (
            f"expected {expected} `{stem}*` implementations, found {src.count(stem)}. Six rivals "
            f"were deleted in #2021; a new one is a new place for the copies to diverge."
        )
    for gone in ("_solve_hjb_system_2d", "_solve_hjb_system_3d", "_step_2d_split", "_step_3d_split"):
        assert f"def {gone}" not in src, f"{gone} came back"


def test_the_cfl_step_has_no_floor_and_names_the_zero_gradient_case():
    """The surviving CFL step takes the deleted 3-D copy's semantics, which were the corrected ones.

    `max(dt_stable, 1e-10)` said "ensure positive time step", but positivity is not the property
    required: when the diffusion-limited step is genuinely smaller, the floor returns a step ABOVE
    the stability bound and the solve runs unstably, where without it `_advance_full_interval`
    reaches its `max_substeps` guard and fails loud.
    """
    solver, _u = _solve(2, 11, 6)
    n = 11
    flat = np.ones((n, n))
    dt_flat = solver._compute_dt_stable_nd(flat, np.ones((n, n)))
    diff_bound = solver.diffusion_stability_factor * solver.grid_spacing[0] ** 2 / solver.problem.sigma**2
    assert dt_flat == pytest.approx(diff_bound, rel=1e-12), (
        "with no gradient anywhere the CFL limit is absent and the diffusion bound must govern "
        "exactly -- an epsilon in the denominator would return a slightly different number and hide "
        "the case instead of naming it"
    )

    # Comments stripped before searching. Without that this assertion matches the comment that
    # EXPLAINS the removal -- the instrument reading the documentation of the thing instead of the
    # thing, which is how it first failed.
    def _code_only(fn):
        return "\n".join(ln for ln in inspect.getsource(fn).splitlines() if not ln.lstrip().startswith("#"))

    assert "max(dt_stable, 1e-10)" not in _code_only(HJBWENOSolver._compute_dt_stable_nd), (
        "the floor came back in the nd owner"
    )

    # SCOPED, and the scope is the finding. `_compute_dt_stable_1d` still carries the same floor.
    # It is not a rival of the nd owner -- the 1-D path is not a dimensional split and is the only
    # one carrying `source_term` -- so #2021 does not touch it, and widening this change to reach it
    # would be a behaviour change smuggled into a consolidation. Recorded here so the remaining
    # instance is visible rather than assumed gone.
    assert "max(dt_stable, 1e-10)" in _code_only(HJBWENOSolver._compute_dt_stable_1d), (
        "the 1-D CFL step no longer has the floor. If that was deliberate, delete this assertion and "
        "say so; if it drifted, the two paths have diverged again in the opposite direction."
    )
