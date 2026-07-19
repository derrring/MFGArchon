"""Tests for interaction energy functionals (Issues #1023, #1642).

Three groups, each pinning a distinct failure:

* **Gate 2 / A2 bridge** -- ``flat_derivative`` equals the FD entry gradient of
  ``.energy`` *divided by the quadrature weights*. Catches either side of the
  pair drifting: a missing ``dx`` in ``energy``, a spurious one in
  ``flat_derivative``, or the bridge factor being applied twice / not at all.
  On the non-uniform grids below a scalar-``dx`` repair does **not** pass.
* **A2 physicality** -- ``energy()`` converges under refinement instead of
  scaling as ``1/h``. Catches the regression directly, without reference to any
  derivative.
* **A1 contract freeze** -- keyword-only ``t``, the ``weights`` member, the
  ``second_variation`` slot, and the loud single-population refusal. Catches a
  silent arity/membership change to the ``runtime_checkable`` Protocol.

Non-uniform weights (A3) are exercised throughout: every pre-#1642 test built a
uniform ``linspace`` grid, but ``ConvolutionCouplingOperator(cell_volume=<array>)``
is a shipped constructor for scattered GFDM clouds.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.operators.interaction.convolution import ConvolutionCouplingOperator
from mfgarchon.operators.interaction.energy_functionals import (
    CombinedEnergy,
    EnergyFunctional,
    PotentialEnergy,
    QuadraticInteractionEnergy,
    as_single_population,
    flat_derivative_from_energy_gradient,
    validate_weights,
)
from mfgarchon.operators.interaction.kernels import GaussianKernel, WendlandKernel
from mfgarchon.utils.functional_calculus import FiniteDifferenceFunctionalDerivative


def _grid(N):
    """Uniform grid: points and the scalar cell volume."""
    x = np.linspace(0.0, 1.0, N)
    return x, x[1] - x[0]


def _uniform_weights(N):
    x, dx = _grid(N)
    return x, np.full(N, dx)


def _scattered(N, seed=0):
    """Scattered 1-D cloud with genuinely non-uniform trapezoid cell volumes.

    Weights vary by more than an order of magnitude across the cloud, so a
    scalar-``dx`` convention cannot accidentally satisfy the weighted pins.
    """
    rng = np.random.default_rng(seed)
    pts = np.sort(rng.uniform(0.0, 1.0, N))
    w = np.empty(N)
    w[0] = (pts[1] - pts[0]) / 2.0
    w[-1] = (pts[-1] - pts[-2]) / 2.0
    w[1:-1] = (pts[2:] - pts[:-2]) / 2.0
    return pts, w


def _fd_flat_derivative(energy_fn, m, weights, epsilon=1e-5):
    """FD entry gradient of ``energy_fn`` routed through the single bridge owner."""
    fd = FiniteDifferenceFunctionalDerivative(epsilon=epsilon, method="central")
    gradient = fd.compute(energy_fn, m, x_points=None, y_points=np.arange(len(m)))
    return flat_derivative_from_energy_gradient(gradient, weights)


class TestProtocolConformance:
    def test_quadratic_is_energy_functional(self):
        _, dx = _grid(20)
        conv = ConvolutionCouplingOperator(GaussianKernel(1.0, 0.1), grid_shape=(20,), spacings=[dx])
        assert isinstance(QuadraticInteractionEnergy(conv), EnergyFunctional)

    def test_potential_is_energy_functional(self):
        _, w = _uniform_weights(20)
        assert isinstance(PotentialEnergy(np.ones(20), w), EnergyFunctional)

    def test_combined_is_energy_functional(self):
        _, w = _uniform_weights(20)
        assert isinstance(CombinedEnergy([PotentialEnergy(np.ones(20), w)]), EnergyFunctional)

    def test_protocol_requires_all_frozen_members(self):
        """A1: the frozen member set is {weights, energy, flat_derivative, second_variation}.

        ``runtime_checkable`` checks presence only, so this is the only place the
        membership is asserted. A near-miss implementation must NOT satisfy
        ``isinstance`` -- ``create_lions_source`` dispatches on exactly this
        check, and a false positive would call a method that does not exist
        inside a Picard iteration.
        """

        class MissingSecondVariation:
            weights = np.ones(3)

            def energy(self, m, *, t=0.0):
                return 0.0

            def flat_derivative(self, m, *, t=0.0):
                return np.zeros(3)

        class MissingWeights:
            def energy(self, m, *, t=0.0):
                return 0.0

            def flat_derivative(self, m, *, t=0.0):
                return np.zeros(3)

            def second_variation(self, m, *, t=0.0):
                return None

        class Complete(MissingSecondVariation):
            def second_variation(self, m, *, t=0.0):
                return None

        assert not isinstance(MissingSecondVariation(), EnergyFunctional)
        assert not isinstance(MissingWeights(), EnergyFunctional)
        assert isinstance(Complete(), EnergyFunctional)

    def test_subclassing_protocol_supplies_default_second_variation(self):
        """Implementers may inherit the ``None`` default by subclassing explicitly."""

        class Inherited(EnergyFunctional):
            weights = np.ones(3)

            def energy(self, m, *, t=0.0):
                return 0.0

            def flat_derivative(self, m, *, t=0.0):
                return np.zeros(3)

        obj = Inherited()
        assert obj.second_variation(np.ones(3)) is None
        assert isinstance(obj, EnergyFunctional)

    @pytest.mark.parametrize("method", ["energy", "flat_derivative", "second_variation"])
    @pytest.mark.parametrize("which", ["quadratic", "potential", "combined"])
    def test_t_is_keyword_only(self, method, which):
        """D-3: ``t`` is keyword-only on every method of every shipped class, so
        a positional third argument (e.g. a caller passing ``x``) fails at the
        call site instead of being silently bound to ``t``.

        Parametrized over all three classes: a per-class arity drift is exactly
        what ``runtime_checkable`` cannot see -- ``isinstance`` checks member
        presence only, so a positional ``t`` on one implementer would surface as
        a ``TypeError`` inside a Picard iteration, not at construction.
        """
        N = 12
        x, dx = _grid(N)
        conv = ConvolutionCouplingOperator(GaussianKernel(1.0, 0.1), grid_shape=(N,), spacings=[dx])
        inter = QuadraticInteractionEnergy(conv)
        pot = PotentialEnergy(np.cos(x), conv.weights)
        energy = {"quadratic": inter, "potential": pot, "combined": CombinedEnergy([inter, pot])}[which]

        m = np.ones(N)
        getattr(energy, method)(m, t=0.5)  # keyword form works
        with pytest.raises(TypeError):
            getattr(energy, method)(m, 0.5)


class TestGate2AnalyticVsFD:
    """``flat_derivative`` == FD entry gradient of ``energy`` / weights."""

    @pytest.mark.parametrize("kernel", [GaussianKernel(1.3, 0.1), WendlandKernel(2.0, 0.25)])
    def test_quadratic_interaction_derivative_uniform(self, kernel):
        N = 60
        x, dx = _grid(N)
        conv = ConvolutionCouplingOperator(kernel, grid_shape=(N,), spacings=[dx], use_fft=False)
        energy = QuadraticInteractionEnergy(conv)
        m = np.sin(np.pi * x) + 1.2

        analytic = energy.flat_derivative(m)
        numeric = _fd_flat_derivative(energy.energy, m, energy.weights)
        rel = np.max(np.abs(analytic - numeric)) / np.max(np.abs(analytic))
        assert rel < 1e-6

    @pytest.mark.parametrize("kernel", [GaussianKernel(1.3, 0.1), WendlandKernel(2.0, 0.25)])
    def test_quadratic_interaction_derivative_nonuniform(self, kernel):
        """A3: same pin on a scattered cloud with per-point cell volumes."""
        N = 60
        pts, w = _scattered(N)
        assert w.max() / w.min() > 5.0, "cloud not non-uniform enough to discriminate"
        conv = ConvolutionCouplingOperator(kernel, points=pts, cell_volume=w)
        energy = QuadraticInteractionEnergy(conv)
        m = np.sin(np.pi * pts) + 1.2

        analytic = energy.flat_derivative(m)
        numeric = _fd_flat_derivative(energy.energy, m, energy.weights)
        rel = np.max(np.abs(analytic - numeric)) / np.max(np.abs(analytic))
        assert rel < 1e-6

    def test_potential_derivative_uniform(self):
        N = 40
        x, w = _uniform_weights(N)
        V = np.cos(2 * np.pi * x)
        energy = PotentialEnergy(V, w)
        m = np.abs(np.sin(np.pi * x)) + 0.5

        np.testing.assert_allclose(energy.flat_derivative(m), V, atol=1e-14)
        numeric = _fd_flat_derivative(energy.energy, m, energy.weights)
        np.testing.assert_allclose(V, numeric, rtol=1e-6, atol=1e-7)

    def test_potential_derivative_nonuniform(self):
        """A3: ``delta F / delta m = V`` has no quadrature factor even when the
        weights vary -- the FD entry gradient is ``w_k V_k`` and only the bridge
        removes the ``w_k``."""
        N = 40
        pts, w = _scattered(N, seed=3)
        V = np.cos(2 * np.pi * pts)
        energy = PotentialEnergy(V, w)
        m = np.abs(np.sin(np.pi * pts)) + 0.5

        np.testing.assert_allclose(energy.flat_derivative(m), V, atol=1e-14)
        numeric = _fd_flat_derivative(energy.energy, m, energy.weights)
        np.testing.assert_allclose(V, numeric, rtol=1e-5, atol=1e-7)

    def test_combined_derivative_is_additive(self):
        N = 50
        x, dx = _grid(N)
        conv = ConvolutionCouplingOperator(GaussianKernel(1.0, 0.12), grid_shape=(N,), spacings=[dx])
        inter = QuadraticInteractionEnergy(conv)
        pot = PotentialEnergy(3.0 * (x - 0.5) ** 2, conv.weights)
        combined = CombinedEnergy([inter, pot])
        m = np.exp(-((x - 0.5) ** 2) / 0.05)

        np.testing.assert_allclose(
            combined.flat_derivative(m),
            inter.flat_derivative(m) + pot.flat_derivative(m),
            atol=1e-12,
        )
        assert combined.energy(m) == pytest.approx(inter.energy(m) + pot.energy(m))

    def test_combined_derivative_vs_fd_nonuniform(self):
        """A3: the additive pin on a scattered cloud."""
        N = 50
        pts, w = _scattered(N, seed=7)
        conv = ConvolutionCouplingOperator(GaussianKernel(1.5, 0.12), points=pts, cell_volume=w)
        combined = CombinedEnergy([QuadraticInteractionEnergy(conv), PotentialEnergy(3.0 * (pts - 0.5) ** 2, w)])
        m = np.exp(-((pts - 0.5) ** 2) / 0.05) + 0.3

        analytic = combined.flat_derivative(m)
        numeric = _fd_flat_derivative(combined.energy, m, combined.weights)
        rel = np.max(np.abs(analytic - numeric)) / np.max(np.abs(analytic))
        assert rel < 1e-6


class TestA2PhysicalQuadrature:
    """``energy()`` is the physical integral, not the integral over a cell volume."""

    @staticmethod
    def _reference_quadratic(kernel, pts, w, m):
        W = kernel.matrix(pts)
        return 0.5 * float((m * w) @ W @ (m * w))

    def test_quadratic_energy_matches_double_quadrature_nonuniform(self):
        N = 80
        pts, w = _scattered(N, seed=11)
        kernel = GaussianKernel(1.3, 0.1)
        conv = ConvolutionCouplingOperator(kernel, points=pts, cell_volume=w)
        energy = QuadraticInteractionEnergy(conv)
        m = np.exp(-((pts - 0.5) ** 2) / 0.02) + 0.3

        reference = self._reference_quadratic(kernel, conv.points, w, m)
        assert energy.energy(m) == pytest.approx(reference, rel=1e-12)

    def test_potential_energy_matches_weighted_sum_nonuniform(self):
        N = 80
        pts, w = _scattered(N, seed=13)
        V = np.cos(2 * np.pi * pts)
        m = np.exp(-((pts - 0.5) ** 2) / 0.02) + 0.3
        assert PotentialEnergy(V, w).energy(m) == pytest.approx(float(np.sum(w * V * m)), rel=1e-12)

    def test_energy_converges_under_refinement(self):
        """The regression that A2 repairs, stated without any derivative.

        Before the fix ``energy()`` returned ``F[m]/cell_volume``: measured
        ratios were exactly ``1/dx`` (31, 63, 127, 255 at N=32..256), i.e. the
        value doubled with every refinement. Now successive refinements must
        contract toward a limit.
        """
        kernel = GaussianKernel(1.0, 0.1)

        def m_of(x):
            return np.exp(-((x - 0.5) ** 2) / 0.02) + 0.3

        values = []
        for N in (64, 128, 256, 512):
            x, dx = _grid(N)
            conv = ConvolutionCouplingOperator(kernel, grid_shape=(N,), spacings=[dx], use_fft=False)
            values.append(QuadraticInteractionEnergy(conv).energy(m_of(x)))

        diffs = np.abs(np.diff(values))
        assert np.all(diffs[1:] < diffs[:-1]), f"not contracting: {values}"
        assert diffs[-1] < 0.02 * abs(values[-1]), f"not converged: {values}"

    def test_uniform_grid_is_the_scattered_case_with_equal_weights(self):
        """Cross-path pin: the FFT/grid construction and the scattered
        construction are two code paths for the same quantity. On a uniform
        cloud they must return the same energy -- the grid path must not carry a
        private ``cell_volume`` convention."""
        N = 48
        x, dx = _grid(N)
        kernel = GaussianKernel(1.1, 0.13)
        grid_energy = QuadraticInteractionEnergy(ConvolutionCouplingOperator(kernel, grid_shape=(N,), spacings=[dx]))
        cloud_energy = QuadraticInteractionEnergy(
            ConvolutionCouplingOperator(kernel, points=x, cell_volume=np.full(N, dx))
        )
        m = np.sin(np.pi * x) + 1.2
        assert grid_energy.energy(m) == pytest.approx(cloud_energy.energy(m), rel=1e-10)


class TestWeightsOntology:
    """D-2: cell volumes (sum to |Omega|) and particle masses (sum to 1) are
    both quadrature weights and take the same code path. One test each."""

    def test_cell_volume_ontology_sums_to_domain_measure(self):
        N = 64
        x, dx = _grid(N)
        conv = ConvolutionCouplingOperator(GaussianKernel(1.0, 0.1), grid_shape=(N,), spacings=[dx])
        w = conv.weights
        assert w.sum() == pytest.approx(N * dx)
        energy = QuadraticInteractionEnergy(conv)
        m = np.sin(np.pi * x) + 1.2
        numeric = _fd_flat_derivative(energy.energy, m, w)
        rel = np.max(np.abs(energy.flat_derivative(m) - numeric)) / np.max(np.abs(energy.flat_derivative(m)))
        assert rel < 1e-6

    def test_particle_mass_ontology_sums_to_one(self):
        """Empirical measure: weights are masses summing to 1, ``m`` is all-ones.

        The bridge factor is ``w_k`` in both ontologies; nothing renormalizes.
        """
        rng = np.random.default_rng(5)
        N = 40
        pts = np.sort(rng.uniform(0.0, 1.0, N))
        masses = rng.uniform(0.5, 2.0, N)
        masses = masses / masses.sum()
        assert masses.sum() == pytest.approx(1.0)

        conv = ConvolutionCouplingOperator(GaussianKernel(1.0, 0.15), points=pts, cell_volume=masses)
        energy = QuadraticInteractionEnergy(conv)
        m = np.ones(N)

        analytic = energy.flat_derivative(m)
        numeric = _fd_flat_derivative(energy.energy, m, energy.weights)
        rel = np.max(np.abs(analytic - numeric)) / np.max(np.abs(analytic))
        assert rel < 1e-6

        V = np.cos(2 * np.pi * pts)
        pot = PotentialEnergy(V, masses)
        assert pot.energy(m) == pytest.approx(float(np.sum(masses * V)))

    def test_bridge_owner_is_a_plain_division(self):
        w = np.array([0.25, 0.5, 1.0])
        np.testing.assert_allclose(
            flat_derivative_from_energy_gradient(np.array([1.0, 2.0, 3.0]), w),
            np.array([4.0, 4.0, 3.0]),
        )

    def test_bridge_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match="does not match weights shape"):
            flat_derivative_from_energy_gradient(np.ones(4), np.ones(3))

    @pytest.mark.parametrize("bad", [np.array([1.0, 0.0]), np.array([1.0, -0.5])])
    def test_nonpositive_weights_refused(self, bad):
        """A zero or negative cell volume is a broken discretization; the bridge
        divides by it, so it must not reach an ``inf``."""
        with pytest.raises(ValueError, match="strictly positive"):
            validate_weights(bad, "test")

    def test_nonfinite_weights_refused(self):
        with pytest.raises(ValueError, match="finite"):
            validate_weights(np.array([1.0, np.nan]), "test")


class TestSecondVariation:
    """D-4: the optional second-variation slot, and what it means."""

    def test_quadratic_second_variation_is_the_kernel_matrix(self):
        N = 30
        pts, w = _scattered(N, seed=17)
        kernel = GaussianKernel(1.2, 0.15)
        conv = ConvolutionCouplingOperator(kernel, points=pts, cell_volume=w)
        energy = QuadraticInteractionEnergy(conv)
        m = np.sin(np.pi * pts) + 1.0

        S = energy.second_variation(m)
        assert S.shape == (N, N)
        np.testing.assert_allclose(S, S.T, atol=1e-14)
        np.testing.assert_allclose(S, kernel.matrix(conv.points), atol=1e-14)

    def test_second_variation_is_jacobian_of_flat_derivative_over_weights(self):
        """The documented relation, on non-uniform weights where it is sharp:
        ``d(flat_derivative)_k / d m_l == S_kl * w_l``. Returning the operator
        matrix ``W diag(w)`` instead of ``W`` (or vice versa) fails here.

        ``flat_derivative`` is exactly linear in ``m`` for this functional, so a
        unit central difference carries no truncation error and no cancellation
        noise -- unlike a small-epsilon step, which cannot resolve the smallest
        kernel entries."""
        N = 24
        pts, w = _scattered(N, seed=19)
        conv = ConvolutionCouplingOperator(GaussianKernel(1.2, 0.15), points=pts, cell_volume=w)
        energy = QuadraticInteractionEnergy(conv)
        m = np.sin(np.pi * pts) + 1.0

        jac = np.empty((N, N))
        for ell in range(N):
            e = np.zeros(N)
            e[ell] = 0.5
            jac[:, ell] = energy.flat_derivative(m + e) - energy.flat_derivative(m - e)

        np.testing.assert_allclose(jac, energy.second_variation(m) * w[None, :], rtol=1e-10, atol=1e-14)

    def test_potential_second_variation_is_zero(self):
        N = 15
        x, w = _uniform_weights(N)
        S = PotentialEnergy(np.cos(x), w).second_variation(np.ones(N))
        assert S.shape == (N, N)
        np.testing.assert_array_equal(S, np.zeros((N, N)))

    def test_combined_second_variation_is_additive(self):
        N = 20
        x, dx = _grid(N)
        conv = ConvolutionCouplingOperator(GaussianKernel(1.0, 0.12), grid_shape=(N,), spacings=[dx])
        inter = QuadraticInteractionEnergy(conv)
        pot = PotentialEnergy(3.0 * (x - 0.5) ** 2, conv.weights)
        m = np.ones(N)
        np.testing.assert_allclose(
            CombinedEnergy([inter, pot]).second_variation(m),
            inter.second_variation(m) + pot.second_variation(m),
            atol=1e-14,
        )

    def test_combined_second_variation_none_propagates(self):
        """A component without a second variation makes the sum unavailable, not
        partial: a partial sum is a wrong operator silently offered as a right one."""
        N = 10
        x, w = _uniform_weights(N)

        class NoSecondVariation:
            weights = w

            def energy(self, m, *, t=0.0):
                return 0.0

            def flat_derivative(self, m, *, t=0.0):
                return np.zeros(N)

            def second_variation(self, m, *, t=0.0):
                return None

        combined = CombinedEnergy([PotentialEnergy(np.cos(x), w), NoSecondVariation()])
        assert combined.second_variation(np.ones(N)) is None


class TestSinglePopulationRefusal:
    """D-5: a stacked (K*N,) multi-population density is refused loudly."""

    @staticmethod
    def _functionals(N):
        x, dx = _grid(N)
        conv = ConvolutionCouplingOperator(GaussianKernel(1.0, 0.1), grid_shape=(N,), spacings=[dx])
        inter = QuadraticInteractionEnergy(conv)
        pot = PotentialEnergy(np.cos(x), conv.weights)
        return [inter, pot, CombinedEnergy([inter, pot])]

    @pytest.mark.parametrize("method", ["energy", "flat_derivative", "second_variation"])
    def test_stacked_density_refused_on_every_method(self, method):
        N = 20
        stacked = np.concatenate([np.ones(N), 2.0 * np.ones(N)])
        for f in self._functionals(N):
            with pytest.raises(ValueError, match="single-population functional"):
                getattr(f, method)(stacked)

    def test_potential_flat_derivative_no_longer_silently_returns_v(self):
        """The one method that used to accept a stacked array and return a
        wrong-length, plausible answer: ``PotentialEnergy.flat_derivative``
        ignored ``m`` entirely and returned ``V.copy()``."""
        N = 20
        x, w = _uniform_weights(N)
        pot = PotentialEnergy(np.cos(x), w)
        with pytest.raises(ValueError, match=r"got shape \(40,\)"):
            pot.flat_derivative(np.ones(2 * N))

    def test_refusal_names_the_expected_shape(self):
        N = 12
        x, w = _uniform_weights(N)
        with pytest.raises(ValueError, match=r"expected a density of shape \(12,\)"):
            PotentialEnergy(np.cos(x), w).energy(np.ones(7))

    def test_as_single_population_accepts_matching_shape(self):
        w = np.ones(4)
        np.testing.assert_array_equal(as_single_population([1, 2, 3, 4], w, "test"), np.arange(1, 5))


class TestConstruction:
    def test_quadratic_energy_nonnegative_for_repulsive(self):
        """For a positive-definite kernel, F[m] = (1/2)<m, K*m> >= 0."""
        N = 64
        x, dx = _grid(N)
        conv = ConvolutionCouplingOperator(WendlandKernel(1.0, 0.2), grid_shape=(N,), spacings=[dx])
        energy = QuadraticInteractionEnergy(conv)
        m = np.abs(np.sin(2 * np.pi * x)) + 0.1
        assert energy.energy(m) > 0

    def test_combined_requires_components(self):
        with pytest.raises(ValueError):
            CombinedEnergy([])

    def test_combined_refuses_mismatched_weights(self):
        """Summing energies discretized against different measures is not
        defined; before the freeze it produced a number."""
        N = 20
        x, dx = _grid(N)
        conv = ConvolutionCouplingOperator(GaussianKernel(1.0, 0.1), grid_shape=(N,), spacings=[dx])
        wrong = PotentialEnergy(np.cos(x), np.full(N, 2.0 * dx))
        with pytest.raises(ValueError, match="same quadrature weights"):
            CombinedEnergy([QuadraticInteractionEnergy(conv), wrong])

    def test_combined_refuses_mismatched_weight_length(self):
        N = 20
        x, dx = _grid(N)
        conv = ConvolutionCouplingOperator(GaussianKernel(1.0, 0.1), grid_shape=(N,), spacings=[dx])
        short = PotentialEnergy(np.cos(x[:10]), np.full(10, dx))
        with pytest.raises(ValueError, match="same quadrature weights"):
            CombinedEnergy([QuadraticInteractionEnergy(conv), short])

    def test_potential_requires_weights(self):
        """No default: a silent unit cell volume would reintroduce exactly the
        mesh-dependence A2 removes."""
        with pytest.raises(TypeError):
            PotentialEnergy(np.ones(10))

    def test_potential_scalar_weight_broadcasts(self):
        N = 10
        pot = PotentialEnergy(np.ones(N), 0.1)
        assert pot.weights.shape == (N,)
        assert pot.energy(np.ones(N)) == pytest.approx(1.0)

    def test_potential_refuses_weight_shape_mismatch(self):
        with pytest.raises(ValueError, match="does not match potential shape"):
            PotentialEnergy(np.ones(10), np.ones(7))

    def test_operator_weights_is_always_an_array(self):
        """``ConvolutionCouplingOperator.weights`` is the single spelling of the
        quadrature weight, an ``(N,)`` array on both construction routes."""
        N = 16
        x, dx = _grid(N)
        grid_op = ConvolutionCouplingOperator(GaussianKernel(1.0, 0.1), grid_shape=(N,), spacings=[dx])
        cloud_op = ConvolutionCouplingOperator(GaussianKernel(1.0, 0.1), points=x, cell_volume=dx)
        for op in (grid_op, cloud_op):
            assert op.weights.shape == (N,)
            np.testing.assert_allclose(op.weights, dx)

    def test_kernel_matrix_times_weights_is_as_dense(self):
        """Pins the two matrix accessors against each other: ``as_dense`` is the
        operator ``W diag(w)``, ``kernel_matrix`` the unweighted ``W``."""
        N = 18
        pts, w = _scattered(N, seed=23)
        op = ConvolutionCouplingOperator(GaussianKernel(1.0, 0.12), points=pts, cell_volume=w)
        np.testing.assert_allclose(op.kernel_matrix() * w[None, :], op.as_dense(), atol=1e-15)

    def test_kernel_matrix_returns_a_copy(self):
        N = 8
        _, dx = _grid(N)
        op = ConvolutionCouplingOperator(GaussianKernel(1.0, 0.1), grid_shape=(N,), spacings=[dx])
        first = op.kernel_matrix()
        first[0, 0] = 12345.0
        assert op.kernel_matrix()[0, 0] != 12345.0
