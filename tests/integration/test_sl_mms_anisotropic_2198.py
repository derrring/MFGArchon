#!/usr/bin/env python3
"""First MMS order measurement of the Semi-Lagrangian family's HJB half, and the first anisotropic
ORDER measurement anywhere in this repository (Issue #2198).

WHY THIS FIXTURE EXISTS

#2198 measured a disjointness: the only solver family that discretises anisotropic
cross-derivatives (Semi-Lagrangian, via ``hjb_sl_adi.apply_cross_diffusion_explicit``) was the only
family a manufactured solution could not reach, because ``HJBSemiLagrangianSolver`` did not thread
``source_term`` -- zero occurrences of the name in that file, against 21 in ``base_hjb``. The
capability and the verification were in disjoint solvers, so ``D_ij d2u/dx_i dx_j`` had never had
its convergence order measured. This closes that: the solver now threads the source through its
operator-splitting path, and this study measures the order with an off-diagonal ``Sigma``.

WHAT IT CATCHES, MEASURED BY MUTATION

Disabling ``apply_cross_diffusion_explicit`` -- the #1079 O(1) drop, injected:

    sigma            clean EOC        cross-derivative dropped
    scalar 0.3       1.039, 1.017     1.039, 1.017      <- UNCHANGED (see below)
    tensor S         1.040, 1.017     0.718, 0.385      <- collapses; e(31) 1.2999e-02 -> 2.1544e-02

The scalar row is not decoration: a diagonal ``Sigma`` has no cross term to drop, so it must be
bit-unchanged by that mutation. It failing would mean the mutation reached something else and the
tensor row proves nothing. That is ALL it establishes -- it is not evidence that the tensor row
measures the stencil correctly, and calling the pair a "built-in control" overstated it.

WHAT IT CANNOT SEE

- **The three other SL variants.** ``canonical_cs``, the L-based DPP path and
  ``stochastic`` replace the splitting path rather than adding to it, so they refuse
  ``source_term`` and are unmeasured. Their order on ANY manufactured problem is still unknown.
- **The FP side.** ``FPSLAdjointSolver`` also carries the ADI cross-derivative and also does not
  thread ``source_term``; #2198 asks for both and this fixture delivers the HJB half only.
- **A coupled system.** ``m`` is constant here, so this measures the HJB scheme, not the coupling.

THE SIGMA ROUTE IS A KNOWN HOLE, AND THIS FIXTURE DOCUMENTS IT RATHER THAN HIDING IT

``problem.sigma`` is assigned AFTER construction below, because no constructor route accepts a
``(d, d)`` volatility. Measured: ``sigma=S``, ``volatility=S`` and ``diffusion=0.5*S@S.T`` all raise
the same ``ValidationError`` -- ``volatility_field has shape (2, 2), expected (11, 11)`` -- because
every array is validated as a spatial field on the grid. So the anisotropic capability has no
supported entry point from a user's problem; it is reachable only by the assignment below or by
calling ``adi_diffusion_step`` directly. Filed separately.
"""

from __future__ import annotations

from unittest import mock

import pytest

import numpy as np

from mfgarchon.alg.numerical.hjb_solvers import HJBSemiLagrangianSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.utils.manufactured import ManufacturedPair, check_pair, hjb_source

L, T, LAMBDA, M_CONST = 1.0, 0.05, 1.0, 1.0
C = np.pi / L
# Symmetric standard-deviation matrix S; D = 1/2 S S^T (RFC #1596). The off-diagonal is what this
# fixture exists for -- with it zero, every assertion below still passes and measures nothing new.
S_ANISOTROPIC = np.array([[0.30, 0.12], [0.12, 0.24]])
SIGMA_SCALAR = 0.3
LEVELS = ((11, 8), (21, 16), (31, 24))


def _a(t):
    return 1.0 + 0.5 * (T - t) / T


def _ap(_t):
    return -0.5 / T


def u_star(t, x):
    return _a(t) * np.cos(C * x[..., 0]) * np.cos(C * x[..., 1])


def _hess_u(t, x):
    """u* is a PRODUCT, not a separable sum, so its cross derivative is non-zero.

    This is the whole design constraint. `test_coupled_mms_2d_no_flux.py`'s u* is a SUM
    (cos(c x1) + beta cos(c x2)), so d2u/dx1dx2 == 0 identically and no choice of Sigma can make
    that fixture exercise the HJB cross term. Measured on this test's own
    200-point sample at t = 0.3T: max|d2u/dx1dx2| = 1.33e+01 -- a sample maximum, not a bound.
    """
    hess = np.zeros((len(x), 2, 2))
    c1, c2 = np.cos(C * x[..., 0]), np.cos(C * x[..., 1])
    s1, s2 = np.sin(C * x[..., 0]), np.sin(C * x[..., 1])
    hess[:, 0, 0] = hess[:, 1, 1] = -_a(t) * C**2 * c1 * c2
    hess[:, 0, 1] = hess[:, 1, 0] = _a(t) * C**2 * s1 * s2
    return hess


PAIR = ManufacturedPair(
    u=u_star,
    u_t=lambda t, x: _ap(t) * np.cos(C * x[..., 0]) * np.cos(C * x[..., 1]),
    grad_u=lambda t, x: np.stack(
        [
            -_a(t) * C * np.sin(C * x[..., 0]) * np.cos(C * x[..., 1]),
            -_a(t) * C * np.cos(C * x[..., 0]) * np.sin(C * x[..., 1]),
        ],
        axis=-1,
    ),
    hess_u=_hess_u,
    m=lambda t, x: np.full(len(x), M_CONST),
    m_t=lambda t, x: np.zeros(len(x)),
    grad_m=lambda t, x: np.zeros((len(x), 2)),
    hess_m=lambda t, x: np.zeros((len(x), 2, 2)),
    name="sl_anisotropic_2d",
)

HAMILTONIAN = SeparableHamiltonian(
    control_cost=QuadraticControlCost(lambda_=LAMBDA), coupling=lambda m: 0.0 * m, coupling_dm=lambda _m: 0.0
)


def _solve(nx: int, nt: int, sigma, sigma_kind: str | None) -> float:
    grid = TensorProductGrid(bounds=[(0.0, L)] * 2, Nx_points=[nx] * 2, boundary_conditions=no_flux_bc(dimension=2))
    components = MFGComponents(
        m_initial=lambda p: M_CONST,
        u_terminal=lambda p: float(u_star(T, np.asarray(p).reshape(1, 2))[0]),
        hamiltonian=HAMILTONIAN,
    )
    problem = MFGProblem(geometry=grid, T=T, Nt=nt, sigma=SIGMA_SCALAR, components=components, coupling_coefficient=1.0)
    if sigma_kind == "tensor":
        # See the module docstring: no constructor route accepts a (d, d) volatility.
        problem.sigma = sigma
    points = problem.geometry.get_spatial_grid()
    shape = tuple(problem.geometry.Nx_points)
    solver = HJBSemiLagrangianSolver(problem)
    U = solver.solve_hjb_system(
        M_density=np.full((nt + 1, *shape), M_CONST),
        U_terminal=u_star(T, points).reshape(shape),
        U_coupling_prev=np.stack([u_star(k * T / nt, points).reshape(shape) for k in range(nt + 1)]),
        source_term=hjb_source(PAIR, HAMILTONIAN, sigma, sigma_kind=sigma_kind),
    )
    error = np.asarray(U)[0] - u_star(0.0, points).reshape(shape)
    return float(np.sqrt((error**2).sum()) * (L / (nx - 1)))


def _eoc(errors):
    h = [L / (nx - 1) for nx, _ in LEVELS]
    return [float(np.log(errors[i] / errors[i + 1]) / np.log(h[i] / h[i + 1])) for i in range(len(errors) - 1)]


@pytest.mark.integration
def test_the_pair_has_a_nonzero_cross_derivative():
    """Without this the study is blind, and it would look exactly as healthy."""
    x = np.random.default_rng(1).uniform(0.05, 0.95, size=(200, 2))
    check_pair(PAIR, 0.3 * T, x)
    assert np.max(np.abs(_hess_u(0.3 * T, x)[:, 0, 1])) > 1.0


@pytest.mark.integration
def test_the_three_other_sl_variants_refuse_the_source():
    """Refusal is a behaviour; an absent signature is not (#2020). Each of these replaces the
    splitting path, so where the forcing enters has not been derived -- and a manufactured source in
    the wrong place verifies a different equation while still converging."""
    grid = TensorProductGrid(bounds=[(0.0, L)] * 2, Nx_points=[11] * 2, boundary_conditions=no_flux_bc(dimension=2))
    components = MFGComponents(m_initial=lambda p: M_CONST, u_terminal=lambda p: 0.0, hamiltonian=HAMILTONIAN)
    problem = MFGProblem(geometry=grid, T=T, Nt=4, sigma=SIGMA_SCALAR, components=components, coupling_coefficient=1.0)
    shape = (11, 11)
    kwargs = {
        "M_density": np.full((5, *shape), M_CONST),
        "U_terminal": np.zeros(shape),
        "U_coupling_prev": np.zeros((5, *shape)),
    }
    for method in ("canonical_cs", "stochastic"):
        solver = HJBSemiLagrangianSolver(problem, diffusion_method=method)
        with pytest.raises(NotImplementedError, match="operator-splitting path only"):
            solver.solve_hjb_system(**kwargs, source_term=lambda t, x: np.zeros(len(x)))
        # CONTROL: the same solver runs when no source is asked for, so the refusal is about the
        # source and not about the configuration being broken.
        solver.solve_hjb_system(**kwargs)

    # The THIRD variant. `_use_dpp` is a derived property -- it needs a lagrangian_class plus a
    # non-smooth Hamiltonian -- so the branch is reached by patching the dispatch condition the
    # refusal itself reads. An earlier version of this test asserted only the two above while three
    # separate artifacts claimed three were asserted.
    dpp_solver = HJBSemiLagrangianSolver(problem)
    with mock.patch.object(type(dpp_solver), "_use_dpp", property(lambda _self: True)):
        assert dpp_solver._use_dpp is True, "the patch did not take; the assertion below proves nothing"
        with pytest.raises(NotImplementedError, match="operator-splitting path only"):
            dpp_solver.solve_hjb_system(**kwargs, source_term=lambda t, x: np.zeros(len(x)))


@pytest.mark.integration
def test_sl_is_first_order_with_an_isotropic_sigma():
    """The control leg. It also establishes that the source channel itself is correct: a wrong sign
    or a misplaced forcing shows up here, where no cross term is involved."""
    errors = [_solve(nx, nt, SIGMA_SCALAR, None) for nx, nt in LEVELS]
    order = _eoc(errors)
    assert all(0.85 <= o <= 1.3 for o in order), f"scalar-sigma order {order} (errors {errors})"
    # An ERROR CEILING beside the order, because EOC is a ratio and is blind to an error scaled
    # uniformly across levels. Review measured that a source displaced by one grid cell inflates
    # the error 6.1x and still passes the order assertion. Clean finest-level error is 1.32e-02
    # (scalar) and 1.30e-02 (tensor); 2.0e-02 sits above both and well under a 6x inflation.
    assert errors[-1] < 2.0e-02, (
        f"scalar-sigma error level {errors[-1]:.4e} at the finest grid, with the order still in band "
        f"({order}). A misplaced or mis-scaled source converges at the right ORDER with the wrong "
        f"constant, and only a level bound can see that."
    )


@pytest.mark.integration
def test_sl_is_first_order_with_an_off_diagonal_sigma():
    """The first anisotropic order measurement in this repository.

    It answers the warning `_adi_diffusion_step` emits for this configuration -- "full tensor with
    off-diagonal terms ... ADI scheme may be inaccurate. Consider ... Craig-Sneyd scheme for mixed
    derivatives". At these resolutions first order is preserved: 1.040, 1.017 against the isotropic
    1.039, 1.017. That is a statement about these resolutions, not a proof that the explicit
    cross-term treatment is unconditionally fine.
    """
    errors = [_solve(nx, nt, S_ANISOTROPIC, "tensor") for nx, nt in LEVELS]
    order = _eoc(errors)
    assert all(0.85 <= o <= 1.3 for o in order), f"tensor-sigma order {order} (errors {errors})"
    # An ERROR CEILING beside the order, because EOC is a ratio and is blind to an error scaled
    # uniformly across levels. Review measured that a source displaced by one grid cell inflates
    # the error 6.1x and still passes the order assertion. Clean finest-level error is 1.32e-02
    # (scalar) and 1.30e-02 (tensor); 2.0e-02 sits above both and well under a 6x inflation.
    assert errors[-1] < 2.0e-02, (
        f"tensor-sigma error level {errors[-1]:.4e} at the finest grid, with the order still in band "
        f"({order}). A misplaced or mis-scaled source converges at the right ORDER with the wrong "
        f"constant, and only a level bound can see that."
    )
