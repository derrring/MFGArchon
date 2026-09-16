"""`BlockIterator(adjoint_verify=True)` compares two operators, or refuses; it never silently passes (#2338).

At 24246a20 both the per-step check and `_generate_adjoint_report` asked the FP solver for a `_build_advection_matrix`
that no FP solver has ever defined -- the name exists only on `HJBFDMSolver` -- so `getattr` returned None and both
skipped, for every configuration including the default FDM pair. A diagnostic that is present and never runs reads as
"adjoint verified" to whoever enabled it, which is worse than not offering it.

What it compares now (user ruling, 2026-09-16): the FP solver's OWN advection operator against
`build_linearized_operator`, the exact Jacobian, which is the pairing that actually holds. The velocity-mode
`build_advection_matrix` the old code reached for does not: measured at 5d232f62 on the 41-point fixture below, its
relative error against the FP operator is 9.991e-01 under `engquist_osher` and 9.996e-01 under `rouy_tourin`, so no
tolerance separates a defect from the design there and wiring the check as written would have warned on correct code.

Oracle: #2313's own measurement, which this check exists to have caught. Over the interior window, relative error is
1.1e-16 under `engquist_osher` against 3.0e-02 under `rouy_tourin` (absolute max|A_fp - J^T| 2.8e-14 against 18.5),
either side of the 1e-10 tolerance the check already used.
"""

from __future__ import annotations

import logging
import warnings

import pytest

import numpy as np

from mfgarchon import MFGProblem
from mfgarchon.alg.numerical.coupling.block_iterators import BlockIterator
from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

_N = 41


def _bump(dimension: int):
    """Centred Gaussian, written per dimension because the IC signature is resolved by arity."""
    if dimension == 1:
        return lambda x: np.exp(-10 * (np.asarray(x) - 0.5) ** 2)
    return lambda x, y: np.exp(-10 * ((np.asarray(x) - 0.5) ** 2 + (np.asarray(y) - 0.5) ** 2))


def _ripple(dimension: int):
    """Terminal condition with interior local maxima along every axis, where the two presets separate."""
    if dimension == 1:
        return lambda x: np.cos(4 * np.pi * np.asarray(x))
    return lambda x, y: np.cos(4 * np.pi * np.asarray(x)) * np.cos(4 * np.pi * np.asarray(y))


def _problem(n: int = _N, shape: tuple[int, ...] | None = None) -> MFGProblem:
    """No-flux fixture whose terminal condition has interior local maxima, where the presets separate."""
    points = list(shape) if shape is not None else [n]
    dimension = len(points)
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0)] * dimension, Nx_points=points, boundary_conditions=no_flux_bc(dimension=dimension)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MFGProblem(
            geometry=grid,
            Nt=8,
            T=0.4,
            sigma=0.05,
            components=MFGComponents(
                m_initial=_bump(dimension),
                u_terminal=_ripple(dimension),
                hamiltonian=SeparableHamiltonian(
                    control_cost=QuadraticControlCost(control_cost=1.0), coupling=lambda m: m, coupling_dm=lambda m: 1.0
                ),
            ),
        )


@pytest.mark.parametrize(
    ("numerical_hamiltonian", "expect_mismatch"), [("engquist_osher", False), ("rouy_tourin", True)]
)
def test_the_check_runs_and_reports_which_pairing_is_adjoint(numerical_hamiltonian, expect_mismatch, caplog):
    """The pin #2338 is about: the comparison happens, and it separates an adjoint pairing from a non-adjoint one.

    `adjoint_rows_compared` is the positive control. Without it a run that skipped every comparison is indistinguishable
    from one that made them all and found nothing -- which is precisely the state this issue reports, and a
    `mismatch_count` of 0 is what it looked like.
    """
    problem = _problem()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        iterator = BlockIterator(
            problem,
            HJBFDMSolver(problem, numerical_hamiltonian=numerical_hamiltonian),
            FPFDMSolver(problem),
            adjoint_verify=True,
            adjoint_mode="jacobian_transpose",
        )
        with caplog.at_level(logging.WARNING, logger="mfgarchon.alg.numerical.coupling.block_iterators"):
            result = iterator.solve(max_iterations=2, tolerance=1e-6)

    metadata = result.metadata
    assert metadata["adjoint_rows_compared"] == _N - 2, "the check did not compare the interior rows"
    if expect_mismatch:
        assert metadata["adjoint_mismatch_count"] > 0
        assert metadata["adjoint_max_rel_error"] > 1e-3
        assert any("Adjoint mismatch at step" in message for message in caplog.messages), (
            "the mismatch was counted but never reported, which is the failure mode of #2338 one level in"
        )
    else:
        assert metadata["adjoint_mismatch_count"] == 0
        assert metadata["adjoint_max_rel_error"] < 1e-10


def test_it_refuses_an_fp_solver_that_cannot_supply_an_operator():
    """A solver can satisfy the strict-adjoint protocol and still have no matrix to compare (#2338).

    `WeakFormFPSolver` is the in-tree case: it defines `solve_fp_step_adjoint_mode`, so
    `validate_adjoint_capability` admits it, and it assembles no advection operator. The stand-in below is that shape
    with none of its construction cost. Skipping here is what the issue reports; refusing is the user ruling of
    2026-09-16.
    """
    problem = _problem(n=11)

    class OperatorlessFPSolver(FPFDMSolver):
        """Passes the strict-adjoint protocol check, supplies no operator."""

        build_advection_operator = None  # type: ignore[assignment]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        iterator = BlockIterator(
            problem,
            HJBFDMSolver(problem),
            OperatorlessFPSolver(problem),
            adjoint_verify=True,
            adjoint_mode="jacobian_transpose",
        )
        with pytest.raises(NotImplementedError, match=r"adjoint_verify=True.*build_advection_operator.*2338"):
            iterator.solve(max_iterations=1, tolerance=1e-6)


def test_it_refuses_verification_in_a_mode_that_builds_no_adjoint_coupling():
    """`adjoint_mode='off'` is the default, so `adjoint_verify=True` alone used to promise a check and run none."""
    problem = _problem(n=11)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match=r"adjoint_verify=True requires adjoint_mode.*2338"):
            BlockIterator(problem, HJBFDMSolver(problem), FPFDMSolver(problem), adjoint_verify=True)


def test_the_operator_is_the_scheme_the_solver_is_configured_with():
    """`build_advection_operator` must read `self.advection_scheme`, not a hardcoded one.

    Otherwise the check verifies a discretisation the FP solve does not use, and it would pass while the configured
    scheme disagreed -- a green check for the wrong operator.
    """
    problem = _problem()
    x = np.asarray(problem.geometry.coordinates[0])
    u = np.cos(4 * np.pi * x) + 0.1 * x
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        divergence = FPFDMSolver(problem, advection_scheme="divergence_upwind").build_advection_operator(u)
        gradient = FPFDMSolver(problem, advection_scheme="gradient_upwind").build_advection_operator(u)
    separation = float(np.abs((divergence - gradient).toarray()).max())
    assert separation > 1.0, f"the two schemes build the same operator ({separation:.3e}); the scheme is not read"


def _verified_run(problem: MFGProblem, numerical_hamiltonian: str, adjoint_mode: str = "jacobian_transpose", sweeps=2):
    """Solve with the check armed and hand back the metadata it filled."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        iterator = BlockIterator(
            problem,
            HJBFDMSolver(problem, numerical_hamiltonian=numerical_hamiltonian),
            FPFDMSolver(problem),
            adjoint_verify=True,
            adjoint_mode=adjoint_mode,
        )
        return iterator.solve(max_iterations=sweeps, tolerance=1e-6).metadata


def test_the_report_agrees_with_the_check_it_ships_beside():
    """The report was the check's false alarm (review of #2344, blocker 1).

    `_generate_adjoint_report` had never run before #2338 -- it skipped on the same missing method -- and wiring it
    live without the interior window made its FIRST output on a correct pair `severity=HIGH, source=BOUNDARY,
    total=1.866e-01`, recommending the user go and fix their boundary conditions, while the per-step check beside it
    reported 0 mismatches at 1.28e-16. The whole residual sat at (interior row, wall column), where
    `build_linearized_operator` zeroes by design (#1564) and the FP operator has real outflow.

    Two things are pinned here: that the report exists at all, and that it agrees with the check. Returning None, and
    restoring the transpose that `diagnose_adjoint_error` already applies, each survived the whole suite before this.
    """
    metadata = _verified_run(_problem(), "engquist_osher")
    report = metadata["adjoint_report"]
    assert report is not None, "the report skipped; before #2338 that was its only behaviour"
    assert metadata["adjoint_mismatch_count"] == 0, "fixture no longer adjoint; the rest of this test says nothing"
    assert report.total_error < 1e-10, (
        f"the report says {report.total_error:.3e} on a pair the check calls adjoint to 1e-16 "
        f"(severity {report.severity}, source {report.error_source})"
    )


def test_the_report_still_sees_a_pair_that_is_not_adjoint():
    """The other side of the pin above: a windowed report must not be a silent one."""
    metadata = _verified_run(_problem(), "rouy_tourin")
    report = metadata["adjoint_report"]
    assert report is not None
    assert metadata["adjoint_mismatch_count"] > 0
    assert report.total_error > 1e-3, f"the report is blind to a pair the check rejects ({report.total_error:.3e})"


def test_the_check_runs_in_every_adjoint_mode():
    """The call sits outside the mode branch, and nothing pinned that (review of #2344: the mutation moving it
    inside `jacobian_transpose` survived the suite).

    `transpose` is deprecated, so the warning is consumed here rather than left to widen the warning census.
    """
    problem = _problem()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.warns(DeprecationWarning, match=r"adjoint_mode='transpose' is deprecated"):
            iterator = BlockIterator(
                problem,
                HJBFDMSolver(problem),
                FPFDMSolver(problem),
                adjoint_verify=True,
                adjoint_mode="transpose",
            )
        metadata = iterator.solve(max_iterations=1, tolerance=1e-6).metadata
    assert metadata["adjoint_rows_compared"] == _N - 2
    assert metadata["adjoint_mismatch_count"] == 0


def test_the_window_is_the_grid_it_is_given():
    """`adjoint_rows_compared` pinned on one 1-D fixture leaves a hardcoded 39 indistinguishable from the real count."""
    metadata = _verified_run(_problem(shape=(11, 11)), "engquist_osher", sweeps=1)
    assert metadata["adjoint_rows_compared"] == 9 * 9
    assert metadata["adjoint_mismatch_count"] == 0


def test_it_refuses_a_grid_with_no_interior():
    """An empty window would compare nothing and pass, which is the shape of #2338 itself."""
    problem = _problem(n=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        iterator = BlockIterator(
            problem,
            HJBFDMSolver(problem),
            FPFDMSolver(problem),
            adjoint_verify=True,
            adjoint_mode="jacobian_transpose",
        )
        with pytest.raises(ValueError, match=r"no\s+interior node|interior node"):
            iterator.solve(max_iterations=1, tolerance=1e-6)


def test_it_refuses_an_hjb_solver_that_builds_no_linearised_operator():
    """`BaseHJBSolver` declares no `build_linearized_operator` and only `HJBFDMSolver` defines one, so every call
    site reached through the base class would have raised `AttributeError` mid-solve (review of #2344 found this
    through the mypy scope ratchet). The refusal that replaced it was itself unexercised.
    """
    problem = _problem(n=11)

    class OperatorlessHJBSolver(HJBFDMSolver):
        """Satisfies nothing that `LinearizedOperatorCapable` asks for."""

        build_linearized_operator = None  # type: ignore[assignment]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        iterator = BlockIterator(
            problem,
            OperatorlessHJBSolver(problem),
            FPFDMSolver(problem),
            adjoint_verify=True,
            adjoint_mode="jacobian_transpose",
        )
        with pytest.raises(NotImplementedError, match=r"LinearizedOperatorCapable|linearised HJB operator"):
            iterator.solve(max_iterations=1, tolerance=1e-6)
