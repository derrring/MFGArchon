#!/usr/bin/env python3
"""Guards for ``mfgarchon.utils.manufactured`` -- the assembly of MMS source terms (#2201).

WHAT EACH GROUP PINS, AND WHAT WOULD CATCH IT

- ``TestAgreesWithTheHandAssembly`` -- the pre-consolidation output, captured here as an
  INDEPENDENT copy of the arithmetic ``test_coupled_mms_2d_no_flux.py`` wrote by hand before this
  module existed. It is deliberately NOT imported from that fixture: once the fixture is migrated
  onto this module, importing its source would make the comparison tautological, which is the
  failure mode ``domains/cs/_core.md`` names for a consolidation's own pin.
- ``TestConventionsComeFromTheirOwners`` -- the assembly convention, adjudicated against this
  package's single-source owners (``H.optimal_control`` for the drift, ``diffusion_from_volatility``
  for sigma -> D) rather than against this module's own arithmetic. This is the group that fails if
  the drift sign or the volatility power is re-derived here.
- ``TestRefusals`` -- every documented scope boundary, asserted to actually raise. The SCOPE
  docstring previously promised refusals that no code performed.
- ``TestCheckPair`` -- the finite-difference audit, whose oracle is calculus and not another
  implementation. It is the only check that sees a wrong cross-derivative, and the first test in it
  measures why: under an isotropic sigma the assembled source cannot.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.core.hamiltonian import (
    L1ControlCost,
    OptimizationSense,
    QuadraticControlCost,
    SeparableHamiltonian,
)
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCType, no_flux_bc
from mfgarchon.utils.manufactured import (
    ManufacturedPair,
    _diffusion_tensor,
    check_pair,
    fp_source,
    hjb_source,
    pair_derivative_errors,
    pair_for,
)
from mfgarchon.utils.pde_coefficients import diffusion_from_volatility

L, T, SIGMA, ZETA, BETA = 20.0, 4.0, 1.0, 0.5, 0.6
C = np.pi / L


def _a1(t):
    return 1.0 + 0.5 * (T - t) / T


def _a1p(_t):
    return -0.5 / T


def _a2(t):
    return 0.4 * np.cos(np.pi * t / (2.0 * T))


def _a2p(t):
    return -0.4 * (np.pi / (2.0 * T)) * np.sin(np.pi * t / (2.0 * T))


def _u(t, x):
    return _a1(t) * (np.cos(C * x[..., 0]) + BETA * np.cos(C * x[..., 1]))


def _m(t, x):
    return (1.0 + _a2(t) * np.cos(2 * C * x[..., 0]) * np.cos(4 * C * x[..., 1])) / L**2


def _hess_u(t, x):
    hess = np.zeros((len(x), 2, 2))
    hess[:, 0, 0] = -_a1(t) * C**2 * np.cos(C * x[..., 0])
    hess[:, 1, 1] = -BETA * _a1(t) * C**2 * np.cos(C * x[..., 1])
    return hess


def _hess_m(t, x):
    hess = np.zeros((len(x), 2, 2))
    common = _a2(t) * np.cos(2 * C * x[..., 0]) * np.cos(4 * C * x[..., 1]) / L**2
    hess[:, 0, 0] = -4 * C**2 * common
    hess[:, 1, 1] = -16 * C**2 * common
    hess[:, 0, 1] = hess[:, 1, 0] = 8 * C**2 * _a2(t) * np.sin(2 * C * x[..., 0]) * np.sin(4 * C * x[..., 1]) / L**2
    return hess


@pytest.fixture
def pair():
    return ManufacturedPair(
        u=_u,
        u_t=lambda t, x: _a1p(t) * (np.cos(C * x[..., 0]) + BETA * np.cos(C * x[..., 1])),
        grad_u=lambda t, x: np.stack(
            [-_a1(t) * C * np.sin(C * x[..., 0]), -BETA * _a1(t) * C * np.sin(C * x[..., 1])], axis=-1
        ),
        hess_u=_hess_u,
        m=_m,
        m_t=lambda t, x: _a2p(t) * np.cos(2 * C * x[..., 0]) * np.cos(4 * C * x[..., 1]) / L**2,
        grad_m=lambda t, x: np.stack(
            [
                -2 * C * _a2(t) * np.sin(2 * C * x[..., 0]) * np.cos(4 * C * x[..., 1]) / L**2,
                -4 * C * _a2(t) * np.cos(2 * C * x[..., 0]) * np.sin(4 * C * x[..., 1]) / L**2,
            ],
            axis=-1,
        ),
        hess_m=_hess_m,
        name="coupled_2d_no_flux",
    )


def _hamiltonian(lam=1.0, sense=OptimizationSense.MINIMIZE):
    return SeparableHamiltonian(
        control_cost=QuadraticControlCost(lambda_=lam, sense=sense),
        coupling=lambda m: ZETA * m,
        coupling_dm=lambda _m: ZETA,
        sense=sense,
    )


@pytest.fixture
def points():
    return np.random.default_rng(20260831).uniform(0.0, L, size=(200, 2))


class TestAgreesWithTheHandAssembly:
    """The pre-consolidation output, at t = 1.3, on 200 interior points."""

    def _hand_hjb(self, t, x):
        x1, x2 = x[:, 0], x[:, 1]
        du_dt = _a1p(t) * (np.cos(C * x1) + BETA * np.cos(C * x2))
        grad_sq = (_a1(t) * C) ** 2 * (np.sin(C * x1) ** 2 + BETA**2 * np.sin(C * x2) ** 2)
        lap_u = -_a1(t) * C**2 * (np.cos(C * x1) + BETA * np.cos(C * x2))
        return -du_dt + 0.5 * grad_sq + ZETA * _m(t, x) - 0.5 * SIGMA**2 * lap_u

    def _hand_fp(self, t, x):
        x1, x2 = x[:, 0], x[:, 1]
        dm_dt = _a2p(t) * np.cos(2 * C * x1) * np.cos(4 * C * x2) / L**2
        gm1 = -2 * C * _a2(t) * np.sin(2 * C * x1) * np.cos(4 * C * x2) / L**2
        gm2 = -4 * C * _a2(t) * np.cos(2 * C * x1) * np.sin(4 * C * x2) / L**2
        gu1 = -_a1(t) * C * np.sin(C * x1)
        gu2 = -BETA * _a1(t) * C * np.sin(C * x2)
        lap_u = -_a1(t) * C**2 * (np.cos(C * x1) + BETA * np.cos(C * x2))
        lap_m = -20 * C**2 * _a2(t) * np.cos(2 * C * x1) * np.cos(4 * C * x2) / L**2
        return dm_dt - (gm1 * gu1 + gm2 * gu2 + _m(t, x) * lap_u) - 0.5 * SIGMA**2 * lap_m

    def test_hjb_source_reproduces_it(self, pair, points):
        got = hjb_source(pair, _hamiltonian(), SIGMA)(1.3, points)
        np.testing.assert_allclose(got, self._hand_hjb(1.3, points), rtol=0, atol=1e-15)

    def test_fp_source_reproduces_it(self, pair, points):
        got = fp_source(pair, _hamiltonian(), SIGMA)(1.3, points)
        np.testing.assert_allclose(got, self._hand_fp(1.3, points), rtol=0, atol=1e-18)

    def test_the_comparison_discriminates(self, pair, points):
        """Mutation: drop the coupling from the Hamiltonian. Kills both assertions above -- so they
        are not passing because both sides are zero or because the tolerance is loose."""
        uncoupled = SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0))
        got = hjb_source(pair, uncoupled, SIGMA)(1.3, points)
        assert np.max(np.abs(got - self._hand_hjb(1.3, points))) > 1e-4


class TestConventionsComeFromTheirOwners:
    def test_the_drift_sign_follows_the_optimization_sense(self, pair, points):
        """#1542 class: the FP transport is ``div(m alpha*)`` and ``alpha*`` is ``-grad u / lambda``
        for MINIMIZE but ``+grad u / lambda`` for MAXIMIZE. Deriving it here instead of reading
        ``H.optimal_control`` made the two senses byte-identical and the MAXIMIZE source 195% wrong.
        """
        minimize = fp_source(pair, _hamiltonian(lam=2.0), SIGMA)(1.3, points)
        maximize = fp_source(pair, _hamiltonian(lam=2.0, sense=OptimizationSense.MAXIMIZE), SIGMA)(1.3, points)
        assert np.max(np.abs(minimize - maximize)) > 1e-6, "fp_source is sense-blind"

    @pytest.mark.parametrize("sense", [OptimizationSense.MINIMIZE, OptimizationSense.MAXIMIZE])
    def test_matches_an_assembly_built_through_optimal_control(self, pair, points, sense):
        """Independent assembly: take the drift from the owner, expand the divergence by hand."""
        t = 1.3
        hamiltonian = _hamiltonian(lam=2.0, sense=sense)
        m, grad_u, grad_m = pair.m(t, points), pair.grad_u(t, points), pair.grad_m(t, points)
        alpha = np.asarray(hamiltonian.optimal_control(points, m, grad_u, t), dtype=float)
        coefficient = float(alpha.flat[0] / grad_u.flat[0])
        expected = (
            pair.m_t(t, points)
            + (grad_m * alpha).sum(axis=-1)
            + m * coefficient * np.einsum("nii->n", pair.hess_u(t, points))
            - 0.5 * SIGMA**2 * np.einsum("nii->n", pair.hess_m(t, points))
        )
        np.testing.assert_allclose(fp_source(pair, hamiltonian, SIGMA)(t, points), expected, rtol=0, atol=1e-18)

    @pytest.mark.parametrize("sigma", [0.1, 0.7, 1.0, np.sqrt(2.0), 2.5, 3.0])
    def test_scalar_sigma_resolves_through_the_converter(self, sigma):
        np.testing.assert_allclose(_diffusion_tensor(sigma, 2, None), diffusion_from_volatility(sigma) * np.eye(2))

    @pytest.mark.parametrize("sigma", [0.1, 0.7, 1.0, 2.5])
    def test_scalar_and_its_diagonal_tensor_spelling_agree(self, sigma):
        """#1506 class, at this module's own boundary: an isotropic sigma written as a scalar and
        written as ``diag([sigma, sigma])`` must give the same D. Returning a ``(d, d)`` argument
        unsquared -- reading the standard-deviation matrix as a covariance -- made these differ by
        exactly ``sigma``, silently, on the anisotropic branch this module exists for."""
        np.testing.assert_allclose(
            _diffusion_tensor(sigma, 2, None), _diffusion_tensor(np.diag([sigma, sigma]), 2, "tensor")
        )

    def test_tensor_sigma_resolves_through_the_converter(self):
        volatility = np.array([[0.5, 0.1], [0.1, 0.3]])
        np.testing.assert_allclose(
            _diffusion_tensor(volatility, 2, "tensor"), diffusion_from_volatility(volatility, kind="tensor")
        )


class TestRefusals:
    def test_a_bare_1d_array_is_refused(self, pair, points):
        """Everywhere else in this package a 1-D sigma array is a spatially varying FIELD. Reading
        it here as per-axis variances applied the grid-summed sigma^2 at every point, silently."""
        with pytest.raises(ValueError, match="sigma_kind"):
            hjb_source(pair, _hamiltonian(), np.array([0.3, 0.7]))(1.3, points)

    def test_a_spatially_varying_sigma_is_refused(self, pair, points):
        with pytest.raises(ValueError, match="spatially varying"):
            hjb_source(pair, _hamiltonian(), 0.3 + 0.4 * np.linspace(0.0, 1.0, 11))(1.3, points)

    def test_a_tensor_of_the_wrong_dimension_is_refused(self, pair, points):
        with pytest.raises(ValueError, match=r"shape \(2, 2\)"):
            hjb_source(pair, _hamiltonian(), np.eye(3), sigma_kind="tensor")(1.3, points)

    def test_an_asymmetric_tensor_is_refused(self, pair, points):
        with pytest.raises(ValueError, match="symmetric"):
            hjb_source(pair, _hamiltonian(), np.array([[1.0, 0.4], [0.0, 1.0]]), sigma_kind="tensor")(1.3, points)

    def test_a_scalar_with_a_kind_is_refused(self, pair, points):
        with pytest.raises(ValueError, match="unambiguous"):
            hjb_source(pair, _hamiltonian(), 0.5, sigma_kind="tensor")(1.3, points)

    def test_a_nonlinear_drift_is_refused_when_the_source_is_built(self, pair):
        """L1: ``div(alpha*)`` is not ``c tr(Hess u)``. The refusal must happen at build time, not
        midway through a solve."""
        with pytest.raises(NotImplementedError, match="not linear in p"):
            fp_source(pair, SeparableHamiltonian(control_cost=L1ControlCost()), SIGMA)

    def test_a_non_separable_hamiltonian_is_refused(self, pair):
        class NotSeparable:
            pass

        with pytest.raises(NotImplementedError, match="SeparableHamiltonian"):
            fp_source(pair, NotSeparable(), SIGMA)

    def test_a_field_of_the_wrong_shape_is_refused(self, pair, points):
        """An ``(N, 1)`` where an ``(N,)`` is expected broadcasts to ``(N, N)`` with no error."""
        wrong = ManufacturedPair(**{**pair.__dict__, "m_t": lambda t, x: pair.m_t(t, x)[:, None]})
        with pytest.raises(ValueError, match=r"m_t must return shape \(200,\)"):
            fp_source(wrong, _hamiltonian(), SIGMA)(1.3, points)

    def test_flat_points_are_refused(self, pair, points):
        with pytest.raises(ValueError, match=r"shape \(N, d\)"):
            hjb_source(pair, _hamiltonian(), SIGMA)(1.3, points[:, 0])


class TestCheckPair:
    @staticmethod
    def _cross_flipped(pair):
        return ManufacturedPair(
            **{**pair.__dict__, "hess_m": lambda t, x: _hess_m(t, x) * np.array([[1, -1], [-1, 1]])}
        )

    def test_an_isotropic_source_cannot_see_a_wrong_cross_derivative(self, pair, points):
        """The measurement that justifies check_pair existing: ``tr(D . Hess)`` with a diagonal D
        multiplies every off-diagonal Hessian entry by exactly zero, so the assembled source is
        bit-identical for a pair whose cross-derivative has the wrong sign."""
        good = fp_source(pair, _hamiltonian(), SIGMA)(1.3, points)
        bad = fp_source(self._cross_flipped(pair), _hamiltonian(), SIGMA)(1.3, points)
        assert np.max(np.abs(good - bad)) == 0.0

    def test_an_anisotropic_source_does_see_it(self, pair, points):
        """Positive control for the test above: the path exists and is live, it is simply not
        entered under an isotropic sigma. This is the #2198 cross-derivative term."""
        volatility, kwargs = np.array([[1.0, 0.4], [0.4, 1.0]]), {"sigma_kind": "tensor"}
        good = fp_source(pair, _hamiltonian(), volatility, **kwargs)(1.3, points)
        bad = fp_source(self._cross_flipped(pair), _hamiltonian(), volatility, **kwargs)(1.3, points)
        assert np.max(np.abs(good - bad)) > 1e-5

    def test_a_true_pair_passes(self, pair, points):
        check_pair(pair, 1.3, points)

    def test_a_wrong_cross_derivative_is_caught(self, pair, points):
        with pytest.raises(ValueError, match="hess_m"):
            check_pair(self._cross_flipped(pair), 1.3, points)

    def test_every_field_is_audited(self, pair, points):
        """A field the audit does not reach is a field nothing checks. Mutate each in turn; each
        must move its OWN entry, so the six entries are not aliases of one measurement."""
        baseline = pair_derivative_errors(pair, 1.3, points)
        assert set(baseline) == {"u_t", "grad_u", "hess_u", "m_t", "grad_m", "hess_m"}
        assert all(value < 1e-4 for value in baseline.values()), baseline
        for field in baseline:
            mutated = ManufacturedPair(**{**pair.__dict__, field: lambda t, x, f=field: 1.3 * getattr(pair, f)(t, x)})
            assert pair_derivative_errors(mutated, 1.3, points)[field] > 1e-2, field


# --------------------------------------------------------------------------------------------
# pair_for  --  Issue #2201
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("family", ["no_flux", "periodic", "dirichlet"])
@pytest.mark.parametrize("dim", [1, 2])
def test_a_generated_pair_satisfies_the_boundary_condition_it_names(family, dim):
    """Checked by the BC's OWN definition, face by face, not by a norm over the whole gradient.

    This is the property the generator exists to provide, and its failure is silent: a pair whose
    profile no longer satisfies the wall still runs, still converges, and measures the order of a
    problem nobody posed. Measured by mutating `_profile`: swapping the NO_FLUX profile from
    `cos` to `sin` moves that family's worst wall residual from 2.5e-16 to 2.0e+00, and swapping
    DIRICHLET's from `sin` to `cos` moves its 8.0e-17 to 6.5e-01 -- each contaminating only its
    own family, which is why the parametrisation is per family rather than one aggregate.

    The normal component is what the wall constrains. An earlier version of this check took
    `max|grad u|` over all components and read 2.0e+00 in 2-D as a failure; that was the
    TANGENTIAL derivative, which no boundary condition here says anything about.
    """
    bounds = [(0.0, 1.0)] if dim == 1 else [(0.0, 2.0), (0.0, 1.0)]
    points = [21] if dim == 1 else [9, 7]
    geometry = TensorProductGrid(bounds=bounds, Nx_points=points, boundary_conditions=no_flux_bc(dimension=dim))
    generated = pair_for(geometry, BCType(family))
    x = geometry.get_spatial_grid()

    worst, where = 0.0, ""
    for axis, (low, high) in enumerate(bounds):
        for value in (low, high):
            face = np.abs(x[:, axis] - value) < 1e-12
            assert face.any(), f"no grid point on axis {axis} at {value} -- the probe found no wall"
            wall = x[face]
            if family == "dirichlet":
                residual = float(np.abs(generated.pair.u(0.3, wall)).max())
            else:
                residual = float(np.abs(generated.pair.grad_u(0.3, wall)[:, axis]).max())
            if residual > worst:
                worst, where = residual, f"axis {axis} at {value}"

    assert worst < 1e-12, (
        f"{family} in {dim}-D: worst wall residual {worst:.3e} at {where}, against a machine-zero "
        f"baseline of 2.5e-16 (no_flux), 1.0e-15 (periodic), 8.0e-17 (dirichlet). The generated "
        f"pair does not satisfy the boundary condition it is named for, so any study using it "
        f"measures a different problem."
    )


def test_a_generated_density_stays_positive_at_a_dirichlet_wall():
    """`m_floor` is measured, not stylistic, and this pins the reason.

    A homogeneous-Dirichlet density is zero at the wall, so the exact solution touches zero and any
    negative truncation error there IS a negative density -- driving `FPFDMSolver` with such a pair
    raises at timestep 1 of 17, `density went to -2.928e-02`, from `clip_nonnegative_or_raise`.
    """
    geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1))
    generated = pair_for(geometry, BCType("dirichlet"), m_floor=1.0)
    x = geometry.get_spatial_grid()
    m = generated.pair.m(0.3, x)
    assert m.min() > 0.0, f"generated density reaches {m.min():.3e} -- a solver would refuse it"
    walls = np.abs(m[[0, -1]] - 1.0)
    assert walls.max() < 1e-12, f"m at the walls is not m_floor: off by {walls.max():.3e}"


def test_robin_is_refused_and_says_where_the_blocker_is():
    """Refusing is a behaviour; the message has to carry the reason or the next reader re-derives it."""
    geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1))
    with pytest.raises(NotImplementedError, match="ROBIN"):
        pair_for(geometry, BCType("robin"))
