#!/usr/bin/env python3
"""HJBGFDM's batch path must consume a spatial volatility field, not its mean (Issue #1725).

The batch residual and Jacobian resolved sigma through ``_get_sigma_value(None)``, whose
documented contract collapses an array to its mean (``pde_coefficients.py``: "the batch path
applies one global scalar"). ``assemble_hjb_residual`` has accepted a per-node field since
Issue #1071 phase 7 -- the LLF path already passes one -- so the field was being discarded at
coefficient retrieval, one layer above an assembly that could consume it.

Every pre-existing array test used a CONSTANT array (``np.full(Nx, 0.5)``), which is its own
mean and therefore passes under both behaviours. These tests use a non-constant field, which is
the only kind that can tell them apart.
"""

import pytest

import numpy as np

from mfgarchon.alg.numerical.hjb_solvers import HJBGFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary.conditions import no_flux_bc

_N = 21
_SIGMA_LO, _SIGMA_HI = 0.1, 0.5  # mean 0.3


def _problem(sigma=0.3):
    return MFGProblem(
        geometry=TensorProductGrid(
            bounds=[(0.0, 1.0)],
            Nx_points=[_N],
            boundary_conditions=no_flux_bc(dimension=1),
        ),
        components=MFGComponents(
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0),
                coupling=lambda m: m,
                coupling_dm=lambda m: 1.0,
            ),
            m_initial=lambda x: np.exp(-10 * (np.asarray(x) - 0.5) ** 2),
            u_terminal=lambda x: 0.5 * np.asarray(x) ** 2,
        ),
        T=0.1,
        Nt=5,
        sigma=sigma,
    )


def _solve(volatility_field):
    problem = _problem()
    x = np.linspace(0.0, 1.0, _N)
    solver = HJBGFDMSolver(problem, collocation_points=x.reshape(-1, 1))
    M = np.tile(np.exp(-10 * (x - 0.5) ** 2), (problem.Nt + 1, 1))
    return solver.solve_hjb_system(M, 0.5 * x**2, np.zeros((problem.Nt + 1, _N)), volatility_field=volatility_field)


@pytest.mark.unit
def test_nonconstant_array_field_is_not_collapsed_to_its_mean():
    """The defect, stated as the test that was missing.

    linspace(0.1, 0.5) has mean 0.3. Before the fix these two solves were byte-identical.
    """
    field = np.linspace(_SIGMA_LO, _SIGMA_HI, _N)
    u_field = _solve(field)
    u_mean = _solve(float(field.mean()))

    assert not np.array_equal(u_field, u_mean), (
        "the spatial volatility field was collapsed to its mean: the solve is byte-identical "
        f"to passing the scalar {field.mean()}"
    )
    # And the difference must be of the size the field's spread implies, not float noise.
    assert np.max(np.abs(u_field - u_mean)) > 1e-6


@pytest.mark.unit
def test_nonconstant_callable_field_is_not_collapsed_to_the_domain_center():
    """Same defect via the callable branch, which evaluated once at the domain center."""

    def sigma_of_x(pt):
        return _SIGMA_LO + (_SIGMA_HI - _SIGMA_LO) * np.asarray(pt).reshape(-1)[0]

    u_callable = _solve(sigma_of_x)
    u_center = _solve(float(sigma_of_x(np.array([0.5]))))

    assert not np.array_equal(u_callable, u_center), (
        "the callable volatility was evaluated once at the domain center rather than per node"
    )


@pytest.mark.unit
def test_scalar_field_is_byte_identical_to_the_problem_sigma():
    """The fix must not perturb the scalar path.

    Byte-identity, not a tolerance: consolidating coefficient retrieval is a refactor on this
    path and any float difference would mean it is not.
    """
    u_override = _solve(0.3)
    u_problem = _solve(None)
    np.testing.assert_array_equal(
        u_override,
        u_problem,
        err_msg="scalar volatility_field=0.3 must reproduce problem.sigma=0.3 bit for bit",
    )


@pytest.mark.unit
def test_residual_and_jacobian_read_the_same_sigma_source():
    """Residual and Jacobian resolved sigma through two byte-identical copies of one expression.

    Consolidating them is the point of the fix -- this pins that they cannot drift apart again by
    asserting they resolve through the single owner rather than by comparing their outputs, which
    would be an agreement test between two paths that now share an implementation and would
    therefore go inert exactly when the consolidation succeeds.
    """
    problem = _problem()
    x = np.linspace(0.0, 1.0, _N)
    solver = HJBGFDMSolver(problem, collocation_points=x.reshape(-1, 1))
    field = np.linspace(_SIGMA_LO, _SIGMA_HI, _N)
    solver._volatility_field_override = field

    resolved = solver._batch_sigma()
    assert isinstance(resolved, np.ndarray), "batch sigma must stay a per-node field"
    np.testing.assert_array_equal(resolved, field)

    solver._volatility_field_override = 0.3
    assert isinstance(solver._batch_sigma(), float), "a scalar override must resolve to a scalar"
