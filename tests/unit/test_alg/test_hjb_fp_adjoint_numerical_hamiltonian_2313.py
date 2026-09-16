"""Under Engquist-Osher the FDM HJB linearisation transposes the FDM FP operator, local maxima included (#2313).

The default FDM pair, `HJBFDMSolver` with `FPFDMSolver(advection_scheme="divergence_upwind")`, is advertised as
discretely adjoint (#580, #622), and `build_linearized_operator` is what the strict-adjoint coupling hands the FP side
as ``A_fp = J^T`` (#707). At 6c0610d2 the HJB took the Rouy-Tourin momentum, whose linearisation at a local maximum
uses one one-sided difference while the FP face upwinding pairs with their sum. The discrepancy is confined to the
entries a maximum touches: at 8debbbbc, over the entries whose row or column is a discrete local maximum (3 of the 39
interior rows in 1-D, 54 of 195 in 2-D) max|A_fp - J^T| is 18.5 and 32.4, while over the entries that touch none it is
2.8e-14 and 4.4e-14 -- the round-off Engquist-Osher reaches everywhere. EO, the default since #2313, is ACD's Example 1
form, whose linearisation is that sum.

Oracle: ``A_fp`` is the library's own FP assembly (`add_interior_entries_divergence_upwind`, dt -> infinity,
sigma = 0) and ``J`` the HJB solver's own `build_linearized_operator`; neither is written here. The states have local
maxima along every axis, and Rouy-Tourin is the control that shows the fixture reaches them.

**Interior only, and why.** `build_linearized_operator` zeroes its wall rows (it hardcodes no-flux, #1564), while the FP
has a real flux from the wall cell, so including the wall columns gives max|A_fp - J^T| = 94.6 in 1-D and 60.4 in 2-D
for BOTH presets -- the same entry either way (row 1, column 0: fp -94.608 against J^T 0.000). Measured at e7aad397.
That gap is the zeroed wall rows, not the numerical Hamiltonian; it is #2338's territory.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import pytest

import numpy as np

from mfgarchon import MFGProblem
from mfgarchon.alg.numerical.fp_solvers.fp_fdm_alg_divergence_upwind import add_interior_entries_divergence_upwind
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.alg.numerical.hjb_solvers.base_hjb import _compute_gradient_array_1d, compute_hjb_residual
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.operators.stencils.finite_difference import upwind_momentum
from mfgarchon.utils.pde_coefficients import fp_drift_coefficient

if TYPE_CHECKING:
    from collections.abc import Sequence

_FD_EPSILON = 1e-6
"""Step of the finite-difference oracle in `_hamiltonian_columns`, in state units.

`_kinked_rows` scales its tolerance to it.
"""


def _transpose_gap(shape: tuple[int, ...], numerical_hamiltonian: str | None) -> float:
    """``None`` constructs the solver with no preset, so the constructor default decides."""
    dimension = len(shape)
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0)] * dimension, Nx_points=list(shape), boundary_conditions=no_flux_bc(dimension=dimension)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        problem = MFGProblem(
            geometry=grid,
            Nt=10,
            T=1.0,
            sigma=0.0,
            components=MFGComponents(
                m_initial=lambda x: 1.0,
                u_terminal=lambda x: 0.0,
                hamiltonian=SeparableHamiltonian(
                    control_cost=QuadraticControlCost(control_cost=1.0), coupling=lambda m: m, coupling_dm=lambda m: 1.0
                ),
            ),
        )
    x = np.meshgrid(*grid.coordinates, indexing="ij")
    if dimension == 1:
        u = 0.3 * np.cos(2 * np.pi * x[0]) + 0.2 * np.cos(6 * np.pi * x[0] + 0.4)
    else:
        u = (
            0.3 * np.cos(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1] + 0.4)
            + 0.2 * np.sin(6 * np.pi * x[0] + 0.3)
            + 0.15 * np.cos(4 * np.pi * x[1])
        )
    total = int(np.prod(shape))
    spacing = tuple(grid.get_grid_spacing())

    def indices(margin: int) -> list[int]:
        return [
            k
            for k in range(total)
            if all(margin <= i < n - margin for i, n in zip(np.unravel_index(k, shape), shape, strict=True))
        ]

    interior = indices(1)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for k in interior:
        multi = tuple(int(i) for i in np.unravel_index(k, shape))
        add_interior_entries_divergence_upwind(
            rows, cols, vals, k, multi, shape, dimension, 1e300, 0.0, 1.0, spacing, u.ravel(), grid,
            no_flux_bc(dimension=dimension),
        )  # fmt: skip
    fp = np.zeros((total, total))
    np.add.at(fp, (rows, cols), vals)
    solver = (
        HJBFDMSolver(problem)
        if numerical_hamiltonian is None
        else HJBFDMSolver(problem, numerical_hamiltonian=numerical_hamiltonian)
    )
    hjb = solver.build_linearized_operator(u, np.ones(shape))
    return float(np.abs(fp - hjb.toarray().T)[np.ix_(interior, interior)].max())


@pytest.mark.parametrize("shape", [(41,), (17, 15)], ids=["1d", "2d"])
def test_engquist_osher_linearises_to_the_transpose_of_the_fp_operator(shape):
    """Measured at e7aad397: 2.8e-14 (1-D) and 4.4e-14 (2-D), against 18.5 and 32.4 for Rouy-Tourin."""
    assert _transpose_gap(shape, "engquist_osher") < 1e-10
    assert _transpose_gap(shape, "rouy_tourin") > 1.0, "no local maximum reaches the operator; nothing is tested"


@pytest.mark.parametrize("shape", [(41,), (17, 15)], ids=["1d", "2d"])
def test_the_solver_default_is_the_preset_whose_linearisation_transposes(shape):
    """The user ruling of 2026-09-16 (#2313) in behavioural form: a default-constructed solver must be Engquist-Osher.

    Flipping `DEFAULT_NUMERICAL_HAMILTONIAN` or the constructor default to `"rouy_tourin"` left the whole suite green
    before this test existed (review of #2340, blocker 1), so a one-line revert restored the previous default silently.
    """
    assert _transpose_gap(shape, None) < 1e-10


def test_the_default_momentum_is_engquist_osher_at_a_local_maximum():
    """The module-level default, which the residual primitives read when no preset is passed.

    At a maximum with a+ = -b- = 1 the two presets differ by construction: sqrt(2) against 1.
    """
    backward, forward = np.array([1.0]), np.array([-1.0])
    default = _compute_gradient_array_1d(np.array([-1.0, 0.0, -1.0]), 1.0, bc=None, upwind=True)[1]
    assert default == pytest.approx(float(upwind_momentum(backward, forward, "engquist_osher")[0]))
    assert default != pytest.approx(float(upwind_momentum(backward, forward, "rouy_tourin")[0]))


def _hamiltonian_columns(solver, problem, u, m, preset: str, eps: float = _FD_EPSILON) -> np.ndarray:
    """Column-wise derivative of the Hamiltonian term the residual evaluates, with no Jacobian code involved.

    ``sigma = 0`` on these fixtures, so in nD `_evaluate_hamiltonian_nd` is the Hamiltonian alone, and in 1-D the
    residual is the Hamiltonian plus ``U/dt``, whose derivative is the identity over ``dt``.
    """
    flat = u.ravel()
    total = flat.size

    def hamiltonian(state: np.ndarray) -> np.ndarray:
        if u.ndim == 1:
            residual = compute_hjb_residual(
                state, np.zeros(total), m.ravel(), problem, 5, None, 0.0, True,
                bc=problem.geometry.get_boundary_conditions(), numerical_hamiltonian=preset,
            )  # fmt: skip
            return np.asarray(residual, dtype=float) - state / problem.dt
        gradients = solver._compute_gradients_nd(state.reshape(u.shape), time=0.0)
        return np.asarray(solver._evaluate_hamiltonian_nd(state.reshape(u.shape), m, gradients), dtype=float).ravel()

    columns = np.zeros((total, total))
    for j in range(total):
        step = np.zeros(total)
        step[j] = eps
        columns[:, j] = (hamiltonian(flat + step) - hamiltonian(flat - step)) / (2 * eps)
    return columns


def _kinked_rows(u: np.ndarray, spacings: Sequence[float], shape: tuple[int, ...]) -> list[int]:
    """Interior rows where some axis has ``a+ = -b- > 0``: a discrete local maximum whose two one-sided differences
    are equal in magnitude, where ``rouy_tourin``'s ``max`` has no derivative and returns a Clarke element instead,
    which a two-sided finite difference of H cannot match.

    The positivity is the whole predicate. ``a+ = -b- = 0`` says only that the row is a local MINIMUM: both parts are
    clipped to zero, the momentum is zero on a neighbourhood, and the derivative exists and is zero -- both presets
    agree with the oracle there to 0.0 in 1-D and 1.747e-10 in 2-D. An earlier version of this filter tested the
    equality without the positivity and so excluded the minima, which is 2 of the 19 interior rows in 1-D and 9 of the
    42 in 2-D, while excluding no kink (review of #2340, round 2). ``engquist_osher`` is differentiable even at a real
    kink, since ``sqrt((a+)^2 + (b-)^2)`` is smooth wherever the pair is nonzero; the fixtures below carry no kink for
    either preset, so the tests assert that and then compare every interior row.
    """
    kinked = np.zeros(shape, dtype=bool)
    # The oracle perturbs one node by `_FD_EPSILON`, which is `_FD_EPSILON / h` in difference units, so a row within
    # a few of those of a tie is one the perturbation can carry ACROSS the tie -- the oracle is then invalid there
    # even though the row itself is not tied. A tolerance in state units would be ~4x too narrow to cover that band
    # (review of #2340, round 3: with a planted tie, rows at |a+ + b-| between 6e-06 and 1.6e-05 hard-failed the
    # comparison instead of being caught here). Both fixtures clear this by ~1e4 oracle steps.
    for axis, spacing in enumerate(spacings):
        backward = (u - np.roll(u, 1, axis=axis)) / spacing
        forward = (np.roll(u, -1, axis=axis) - u) / spacing
        a_plus, b_minus = np.maximum(backward, 0.0), np.minimum(forward, 0.0)
        tolerance = 4.0 * _FD_EPSILON / min(spacings)
        kinked |= (np.abs(a_plus + b_minus) < tolerance) & (a_plus > tolerance)
    return [k for k in _interior_rows(shape) if kinked[np.unravel_index(k, shape)]]


def _interior_rows(shape: tuple[int, ...]) -> list[int]:
    """Flat indices of the rows `build_linearized_operator` fills; wall rows are zeroed by design (#1564)."""
    return [
        k
        for k in range(int(np.prod(shape)))
        if all(1 <= i < n - 1 for i, n in zip(np.unravel_index(k, shape), shape, strict=True))
    ]


@pytest.mark.parametrize("numerical_hamiltonian", ["engquist_osher", "rouy_tourin"])
@pytest.mark.parametrize("shape", [(21,), (9, 8)], ids=["1d", "2d"])
def test_the_linearised_operator_is_the_derivative_of_the_hamiltonian_it_linearises(shape, numerical_hamiltonian):
    """`build_linearized_operator` must be the derivative of the residual's Hamiltonian term, preset by preset.

    The adjointness test above cannot see a preset dropped inside this operator: it would still differ from the FP
    transpose, only differently (review of #2340, blocker 2). This compares it against a finite difference of the
    Hamiltonian the solver itself evaluates, over every interior row. Wall rows are zeroed by design (#1564).
    """
    dimension = len(shape)
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0)] * dimension, Nx_points=list(shape), boundary_conditions=no_flux_bc(dimension=dimension)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        problem = MFGProblem(
            geometry=grid,
            Nt=10,
            T=1.0,
            sigma=0.0,
            components=MFGComponents(
                m_initial=lambda x: 1.0,
                u_terminal=lambda x: 0.0,
                hamiltonian=SeparableHamiltonian(
                    control_cost=QuadraticControlCost(control_cost=1.0), coupling=lambda m: m, coupling_dm=lambda m: 1.0
                ),
            ),
        )
    x = np.meshgrid(*grid.coordinates, indexing="ij")
    u = (
        0.3 * np.cos(4 * np.pi * x[0]) + 0.1 * x[0]
        if dimension == 1
        else 0.3 * np.cos(4 * np.pi * x[0]) * np.cos(2 * np.pi * x[1] + 0.3) + 0.11 * x[1] + 0.07 * x[0]
    )
    m = np.ones(shape)
    solver = HJBFDMSolver(problem, numerical_hamiltonian=numerical_hamiltonian)
    assembled = solver.build_linearized_operator(u, m).toarray()
    measured = _hamiltonian_columns(solver, problem, u, m, numerical_hamiltonian)
    kinked = _kinked_rows(u, grid.get_grid_spacing(), shape)
    assert not kinked, f"rows {kinked} sit on a kink, where rouy_tourin has no derivative; retilt the fixture"
    rows = _interior_rows(shape)
    other = "rouy_tourin" if numerical_hamiltonian == "engquist_osher" else "engquist_osher"
    separation = np.abs(
        assembled - HJBFDMSolver(problem, numerical_hamiltonian=other).build_linearized_operator(u, m).toarray()
    )[rows].max()
    assert separation > 1.0, f"the presets build the same operator ({separation:.3e}); nothing is discriminated"
    gap = np.abs(assembled - measured)[rows]
    assert gap.max() < 1e-4, f"{numerical_hamiltonian}: worst gap {gap.max():.3e}"


def test_the_solver_refuses_a_preset_it_does_not_have():
    """The validation branch added with the parameter (#2313), untested until the review of #2340 asked for it.

    A silent fallback to the default here would make a typo look like a working choice of preset -- the failure this
    whole file is about, since the two presets differ only at a local maximum and a fallback would be invisible on any
    fixture without one.
    """
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[9], boundary_conditions=no_flux_bc(dimension=1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        problem = MFGProblem(
            geometry=grid,
            Nt=4,
            T=1.0,
            sigma=0.0,
            components=MFGComponents(
                m_initial=lambda x: 1.0,
                u_terminal=lambda x: 0.0,
                hamiltonian=SeparableHamiltonian(
                    control_cost=QuadraticControlCost(control_cost=1.0), coupling=lambda m: m, coupling_dm=lambda m: 1.0
                ),
            ),
        )
    with pytest.raises(ValueError, match=r"numerical_hamiltonian.*godunov.*2313"):
        HJBFDMSolver(problem, numerical_hamiltonian="godunov")  # the name #2313 ruled out


def _public_entry_problem(n: int = 41, Nt: int = 4):
    """1-D no-flux fixture whose terminal condition has interior local maxima, where the presets separate."""
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[n], boundary_conditions=no_flux_bc(dimension=1))
    x = np.asarray(grid.coordinates[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        problem = MFGProblem(
            geometry=grid,
            Nt=Nt,
            T=0.4,
            sigma=0.05,
            components=MFGComponents(
                m_initial=lambda _x: 1.0,
                u_terminal=lambda _x: np.cos(4 * np.pi * np.asarray(_x)),
                hamiltonian=SeparableHamiltonian(
                    control_cost=QuadraticControlCost(control_cost=1.0), coupling=lambda m: m, coupling_dm=lambda m: 1.0
                ),
            ),
        )
    return problem, x


@pytest.mark.parametrize("numerical_hamiltonian", ["engquist_osher", "rouy_tourin"])
def test_the_backward_solve_returns_a_root_of_its_own_presets_residual(numerical_hamiltonian):
    """`solve_hjb_system` is the solver's public entry point, and nothing pinned the preset through it (#2313).

    Hardcoding the preset at the `solve_hjb_system_backward` call site left the whole suite green (review of #2340,
    round 3), so a user's choice could have been dropped at the outermost hop while every inner pin stayed green.

    Oracle: the definition of the scheme rather than any Jacobian. A backward solve returns U such that each step's
    residual VANISHES, so the returned U must be a root of the residual of the preset the solver was constructed with
    and not of the other one. Measured at 54c1d9dc: 6.3e-08 under its own preset against 5.6 (engquist_osher) and 10.8
    (rouy_tourin) under the other -- eight orders, so the two cannot be confused.
    """
    problem, x = _public_entry_problem()
    m = np.ones((problem.Nt + 1, x.size))
    solver = HJBFDMSolver(problem, numerical_hamiltonian=numerical_hamiltonian)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u = np.asarray(
            solver.solve_hjb_system(
                M_density=m, U_terminal=np.cos(4 * np.pi * x), U_coupling_prev=np.zeros((problem.Nt + 1, x.size))
            )
        )
    bc = no_flux_bc(dimension=1)

    def worst_residual(preset: str) -> float:
        return max(
            float(
                np.abs(
                    compute_hjb_residual(u[k], u[k + 1], m[k + 1], problem, k, bc=bc, numerical_hamiltonian=preset)
                ).max()
            )
            for k in range(problem.Nt)
        )

    other = "rouy_tourin" if numerical_hamiltonian == "engquist_osher" else "engquist_osher"
    assert worst_residual(numerical_hamiltonian) < 1e-6
    assert worst_residual(other) > 1.0, "the presets agree on this state; the fixture reaches no local maximum"


@pytest.mark.parametrize("numerical_hamiltonian", ["engquist_osher", "rouy_tourin"])
def test_the_advection_matrix_carries_the_solvers_own_momentum(numerical_hamiltonian):
    """`build_advection_matrix` is the other public surface that reads the preset, and it was unpinned too (#2313).

    `BlockIterator`'s velocity modes and `FPFDMSolver.solve_fp_step_adjoint_mode` consume this matrix, so a preset
    dropped here moves an FP solve, not just an HJB one.

    Oracle: the operator's own documented form. It upwinds by ``v = -coupling * p``, so its diagonal is ``|v| / dx``
    with ``p`` the owner's momentum for that preset -- computed here from one-sided differences, with no solver code
    in the oracle. Interior nodes only: the wall nodes' differences come from the BC padding, which is #1564's
    subject rather than the preset's. Measured at 54c1d9dc: exact agreement, against 28.58 for the other preset.
    """
    problem, x = _public_entry_problem()
    dx = float(problem.geometry.get_grid_spacing()[0])
    u = np.cos(4 * np.pi * x) + 0.1 * x
    solver = HJBFDMSolver(problem, numerical_hamiltonian=numerical_hamiltonian)
    diagonal = np.diag(solver.build_advection_matrix(u).toarray())[1:-1]
    backward, forward = (u[1:-1] - u[:-2]) / dx, (u[2:] - u[1:-1]) / dx

    def expected(preset: str) -> np.ndarray:
        return np.abs(-fp_drift_coefficient(problem) * upwind_momentum(backward, forward, preset)) / dx

    other = "rouy_tourin" if numerical_hamiltonian == "engquist_osher" else "engquist_osher"
    np.testing.assert_allclose(diagonal, expected(numerical_hamiltonian), rtol=0, atol=1e-12)
    separation = float(np.abs(diagonal - expected(other)).max())
    assert separation > 1.0, f"the presets build the same diagonal ({separation:.3e}); nothing is discriminated"
