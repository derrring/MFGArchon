"""Integration tests for the interaction Lions bridge and ring equilibrium.

Issue #1023, Phase 2.

Gate 3 (lions-bridge equivalence): create_lions_source with an EnergyFunctional
produces the same HJB source as the FD path fed that functional's OWN
``.energy``, and as the optimized create_nonlocal_source path. Feeding the FD
branch a re-derived lambda instead is what let a ``w``-factor fork between the
two branches pass this gate.

Also pinned here: the per-slice ``t`` reaching the functional (D-3 transport,
not merely arity), and the loud refusal of a near-miss EnergyFunctional (D-6)
that would otherwise be silently downgraded to the FD path.

Gate 4 (ring-equilibrium demo): a coupled HJB-FP solve with a repulsive
interaction kernel plus a central attractive potential depletes the centre and
pushes density outward, relative to the attractive-only baseline. This is the
non-local towel-on-the-beach signature that local f(m) cannot produce.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.alg.numerical.coupling import FixedPointIterator
from mfgarchon.alg.numerical.coupling.lions_correction import (
    create_lions_source,
    create_nonlocal_source,
)
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.config import MFGSolverConfig
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.operators.interaction import (
    CombinedEnergy,
    ConvolutionCouplingOperator,
    GaussianKernel,
    PotentialEnergy,
    QuadraticInteractionEnergy,
)
from mfgarchon.utils.functional_calculus import FiniteDifferenceFunctionalDerivative


class TestGate3LionsBridgeEquivalence:
    """The analytic EnergyFunctional path matches the legacy FD / nonlocal paths."""

    def test_analytic_matches_nonlocal_source_exactly(self):
        N = 60
        x = np.linspace(0.0, 1.0, N)
        dx = x[1] - x[0]
        kernel = GaussianKernel(amplitude=1.3, length_scale=0.1)

        conv = ConvolutionCouplingOperator(kernel, grid_shape=(N,), spacings=[dx], use_fft=False)
        energy = QuadraticInteractionEnergy(conv)
        source_analytic = create_lions_source(energy)

        W = kernel.matrix(x)  # raw K(x_i, x_j) matrix
        source_nonlocal = create_nonlocal_source(W, grid_spacing=dx)

        m = np.sin(np.pi * x) + 1.2
        v = np.zeros(N)
        r_analytic = source_analytic(x, m, v, 0.0)
        r_nonlocal = source_nonlocal(x, m, v, 0.0)
        # Both equal (W @ m) * dx to machine precision.
        np.testing.assert_allclose(r_analytic, r_nonlocal, atol=1e-12)

    def test_analytic_matches_fd_lambda_path(self):
        """The two branches of create_lions_source agree on ONE energy object.

        The FD branch is fed ``energy.energy`` itself, not a hand-rolled lambda:
        a re-derived scalar can silently encode a different quadrature
        convention than the functional it is supposed to mirror, which is
        exactly how a ``w``-factor fork between the two branches stayed
        invisible to this gate.
        """
        N = 60
        x = np.linspace(0.0, 1.0, N)
        dx = x[1] - x[0]
        kernel = GaussianKernel(amplitude=1.3, length_scale=0.1)

        conv = ConvolutionCouplingOperator(kernel, grid_shape=(N,), spacings=[dx], use_fft=False)
        energy = QuadraticInteractionEnergy(conv)
        source_analytic = create_lions_source(energy)

        # FD path: the SAME energy object as a plain F[m] -> float callable.
        fd = FiniteDifferenceFunctionalDerivative(epsilon=1e-5, method="central")
        source_fd = create_lions_source(energy.energy, fd, weights=energy.weights)

        m = np.sin(np.pi * x) + 1.2
        v = np.zeros(N)
        r_analytic = source_analytic(x, m, v, 0.0)
        r_fd = source_fd(x, m, v, 0.0)
        rel = np.max(np.abs(r_analytic - r_fd)) / np.max(np.abs(r_analytic))
        assert rel < 1e-6

    def test_fd_path_matches_a_user_written_physical_energy(self):
        """A user's own F[m] -> float, byte-identical to energy.energy, agrees too.

        Guards the convention from the outside: the FD branch must not require
        the caller to pre-divide by the mesh. ``F_physical`` is the textbook
        double quadrature, and the source it yields is the pointwise
        ``delta F / delta m``, not ``w * delta F / delta m``.
        """
        N = 60
        x = np.linspace(0.0, 1.0, N)
        dx = x[1] - x[0]
        kernel = GaussianKernel(amplitude=1.3, length_scale=0.1)
        conv = ConvolutionCouplingOperator(kernel, grid_shape=(N,), spacings=[dx], use_fft=False)
        energy = QuadraticInteractionEnergy(conv)
        W = kernel.matrix(x)

        def f_physical(m):
            m = np.asarray(m).ravel()
            # F[m] = (1/2) sum_ij w_i w_j W_ij m_i m_j -- BOTH quadrature factors.
            return 0.5 * float(np.sum(m * (W @ m))) * dx**2

        m = np.sin(np.pi * x) + 1.2
        assert f_physical(m) == pytest.approx(energy.energy(m), rel=1e-12)

        fd = FiniteDifferenceFunctionalDerivative(epsilon=1e-5, method="central")
        source_user = create_lions_source(f_physical, fd, weights=dx)
        r_user = source_user(x, m, np.zeros(N), 0.0)
        r_analytic = create_lions_source(energy)(x, m, np.zeros(N), 0.0)
        rel = np.max(np.abs(r_analytic - r_user)) / np.max(np.abs(r_analytic))
        assert rel < 1e-6

    def test_fd_path_refuses_without_weights(self):
        """No default weights: assuming w = 1 forks the two branches by w."""
        fd = FiniteDifferenceFunctionalDerivative(epsilon=1e-5, method="central")

        def energy_lambda(m):
            return 0.5 * float(np.sum(np.asarray(m) ** 2)) * 0.02

        with pytest.raises(ValueError, match=r"weights= is required on the finite-difference path"):
            create_lions_source(energy_lambda, fd)

    def test_analytic_path_refuses_redundant_weights(self):
        """An EnergyFunctional carries its own weights; a second set is refused."""
        N = 20
        dx = 1.0 / (N - 1)
        conv = ConvolutionCouplingOperator(GaussianKernel(1.0, 0.1), grid_shape=(N,), spacings=[dx])
        energy = QuadraticInteractionEnergy(conv)
        with pytest.raises(ValueError, match=r"weights= is for the finite-difference path only"):
            create_lions_source(energy, weights=dx)

    def test_fd_path_refuses_non_positive_weights(self):
        """A zero or negative cell volume is a broken discretization, not a case to absorb."""
        fd = FiniteDifferenceFunctionalDerivative(epsilon=1e-5, method="central")

        def energy_lambda(m):
            return float(np.sum(np.asarray(m)))

        with pytest.raises(ValueError, match=r"strictly positive"):
            create_lions_source(energy_lambda, fd, weights=0.0)
        with pytest.raises(ValueError, match=r"strictly positive"):
            create_lions_source(energy_lambda, fd, weights=np.array([0.1, -0.1, 0.1]))

    def test_fd_path_refuses_weights_of_the_wrong_length(self):
        """A weights/density length mismatch must not broadcast into a wrong number."""
        fd = FiniteDifferenceFunctionalDerivative(epsilon=1e-5, method="central")

        def energy_lambda(m):
            return 0.5 * float(np.sum(np.asarray(m) ** 2)) * 0.02

        source = create_lions_source(energy_lambda, fd, weights=np.full(7, 0.02))
        with pytest.raises(ValueError, match=r"does not match weights shape"):
            source(np.linspace(0, 1, 10), np.ones(10), np.zeros(10), 0.0)

    def test_fd_path_requires_functional_derivative(self):
        """Backward compat: plain callable without FD instance raises clearly."""

        def energy_lambda(m):
            return 0.5 * np.sum(m**2)

        with pytest.raises(ValueError):
            create_lions_source(energy_lambda)

    def test_time_space_array_rejected(self):
        """A 2-D (Nt+1, Nx) trajectory is a caller error: the source pipeline
        time-slices before calling, so a per-time source must get a 1-D spatial
        slice. Passing the full trajectory raises (was a silent m[-1] fallback
        that reintroduced the Issue #1285 wrong-slice bug). The 1-D slice path
        still works on both the analytic and the optimized nonlocal source."""
        N = 30
        x = np.linspace(0.0, 1.0, N)
        dx = x[1] - x[0]
        conv = ConvolutionCouplingOperator(GaussianKernel(1.0, 0.1), grid_shape=(N,), spacings=[dx])
        source = create_lions_source(QuadraticInteractionEnergy(conv))
        W = np.exp(-((x[:, None] - x[None, :]) ** 2) / (2 * 0.2**2))
        source_nonlocal = create_nonlocal_source(W, grid_spacing=dx)

        m_slice = np.sin(np.pi * x) + 1.0
        m_traj = np.tile(m_slice, (5, 1))  # (Nt+1, Nx), constant in time

        # 1-D slice path still works.
        r_slice = source(x, m_slice, np.zeros(N), 0.0)
        assert r_slice.shape == (N,)

        # 2-D trajectory rejected on both paths.
        with pytest.raises(ValueError, match=r"1-D spatial density"):
            source(x, m_traj, np.zeros(N), 0.0)
        with pytest.raises(ValueError, match=r"1-D spatial density"):
            source_nonlocal(x, m_traj, np.zeros(N), 0.0)


class _TimeStampedEnergy:
    """Non-autonomous probe functional: ``F_t[m] = t * sum_k w_k m_k``.

    Every shipped functional is autonomous, so no shipped test can observe the
    per-slice ``t`` arriving. This one encodes ``t`` in its return values *and*
    records what it was handed, so dropping ``t=t`` anywhere along the transport
    chain is a wrong number, not an invisible default.
    """

    def __init__(self, weights, scale=1.0):
        self._w = np.asarray(weights, dtype=float).ravel()
        self._scale = float(scale)
        self.seen_energy: list[float] = []
        self.seen_flat: list[float] = []
        self.seen_second: list[float] = []

    @property
    def weights(self):
        return self._w

    def energy(self, m, *, t=0.0):
        self.seen_energy.append(float(t))
        return self._scale * float(t) * float(np.sum(self._w * np.asarray(m).ravel()))

    def flat_derivative(self, m, *, t=0.0):
        self.seen_flat.append(float(t))
        return np.full(self._w.shape, self._scale * float(t))

    def second_variation(self, m, *, t=0.0):
        self.seen_second.append(float(t))
        return np.full((self._w.size, self._w.size), self._scale * float(t))


class TestD3TimeTransport:
    """The per-slice ``t`` reaches the functional -- not merely is accepted.

    ``create_lions_source`` and ``CombinedEnergy`` both forward ``t``. Because
    the three shipped functionals ignore it, arity pins alone leave the
    *transport* unobserved: dropping ``t=t`` at either hop silently substitutes
    the ``t=0.0`` default.
    """

    def test_analytic_dispatch_forwards_t(self):
        N = 12
        w = np.full(N, 0.05)
        probe = _TimeStampedEnergy(w)
        source = create_lions_source(probe)

        x = np.linspace(0.0, 1.0, N)
        m = np.ones(N)
        r = source(x, m, np.zeros(N), 2.5)

        assert probe.seen_flat == [2.5]
        np.testing.assert_allclose(r, np.full(N, 2.5), rtol=0, atol=0)

    def test_analytic_dispatch_t_is_not_a_constant(self):
        """Two different slices must give two different sources."""
        N = 8
        probe = _TimeStampedEnergy(np.full(N, 0.125))
        source = create_lions_source(probe)
        x = np.linspace(0.0, 1.0, N)
        m = np.ones(N)

        r0 = source(x, m, np.zeros(N), 0.0)
        r1 = source(x, m, np.zeros(N), 1.0)
        r2 = source(x, m, np.zeros(N), 3.0)

        np.testing.assert_allclose(r0, np.zeros(N), rtol=0, atol=0)
        np.testing.assert_allclose(r1, np.ones(N), rtol=0, atol=0)
        np.testing.assert_allclose(r2, np.full(N, 3.0), rtol=0, atol=0)
        assert probe.seen_flat == [0.0, 1.0, 3.0]

    def test_combined_energy_forwards_t_to_every_component(self):
        N = 6
        w = np.full(N, 0.2)
        a = _TimeStampedEnergy(w, scale=1.0)
        b = _TimeStampedEnergy(w, scale=10.0)
        combined = CombinedEnergy([a, b])
        m = np.ones(N)

        # energy: t * (1 + 10) * sum_k w_k m_k
        assert combined.energy(m, t=2.0) == pytest.approx(2.0 * 11.0 * float(np.sum(w)))
        assert a.seen_energy == [2.0]
        assert b.seen_energy == [2.0]

        np.testing.assert_allclose(combined.flat_derivative(m, t=3.0), np.full(N, 3.0 * 11.0), rtol=0, atol=0)
        assert a.seen_flat == [3.0]
        assert b.seen_flat == [3.0]

        np.testing.assert_allclose(combined.second_variation(m, t=4.0), np.full((N, N), 4.0 * 11.0), rtol=0, atol=0)
        assert a.seen_second == [4.0]
        assert b.seen_second == [4.0]

    def test_t_survives_dispatch_and_delegation_together(self):
        """End-to-end: source_term_hjb(x, m, v, t) -> CombinedEnergy -> component."""
        N = 5
        w = np.full(N, 0.25)
        a = _TimeStampedEnergy(w, scale=2.0)
        b = _TimeStampedEnergy(w, scale=5.0)
        source = create_lions_source(CombinedEnergy([a, b]))

        r = source(np.linspace(0.0, 1.0, N), np.ones(N), np.zeros(N), 1.75)

        np.testing.assert_allclose(r, np.full(N, 1.75 * 7.0), rtol=0, atol=0)
        assert a.seen_flat == [1.75]
        assert b.seen_flat == [1.75]


class _NearMissEnergy:
    """Provides three of the four required members, and is callable.

    The callable-ness is the dangerous part: without an explicit refusal it
    satisfies the FD branch's duck test and takes that path, so its exact
    ``flat_derivative`` is never called and nothing reports it.
    """

    def __init__(self, weights):
        self._w = np.asarray(weights, dtype=float).ravel()
        self.flat_derivative_calls = 0

    @property
    def weights(self):
        return self._w

    def energy(self, m, *, t=0.0):
        return float(np.sum(self._w * np.asarray(m).ravel()))

    def flat_derivative(self, m, *, t=0.0):
        self.flat_derivative_calls += 1
        return np.full(self._w.shape, 999.0)

    def __call__(self, m):
        return self.energy(m)


class TestNearMissRefusedLoudly:
    """A partial EnergyFunctional is refused, not silently downgraded (D-6)."""

    def test_callable_near_miss_is_refused_and_names_the_missing_member(self):
        N = 10
        probe = _NearMissEnergy(np.full(N, 0.1))
        fd = FiniteDifferenceFunctionalDerivative(epsilon=1e-5, method="central")

        with pytest.raises(TypeError) as excinfo:
            create_lions_source(probe, fd, weights=0.1)

        message = str(excinfo.value)
        assert "second_variation" in message
        assert "_NearMissEnergy" in message
        for provided in ("energy", "flat_derivative", "weights"):
            assert provided in message
        # The silent path is what we are refusing: it never called this.
        assert probe.flat_derivative_calls == 0

    def test_near_miss_is_refused_on_the_analytic_call_too(self):
        """Refusal does not depend on the caller having supplied an FD engine."""
        probe = _NearMissEnergy(np.full(4, 0.25))
        with pytest.raises(TypeError, match=r"second_variation"):
            create_lions_source(probe)

    def test_plain_callable_is_still_accepted(self):
        """The refusal must not swallow the legitimate FD path."""
        fd = FiniteDifferenceFunctionalDerivative(epsilon=1e-5, method="central")
        source = create_lions_source(lambda m: 0.5 * float(np.sum(np.asarray(m) ** 2)) * 0.1, fd, weights=0.1)
        m = np.linspace(1.0, 2.0, 8)
        np.testing.assert_allclose(source(np.linspace(0, 1, 8), m, np.zeros(8), 0.0), m, rtol=1e-6)

    def test_non_callable_non_functional_is_refused_with_a_diagnostic(self):
        """Was an undiagnostic 'object is not callable' from deep inside the FD engine."""
        fd = FiniteDifferenceFunctionalDerivative(epsilon=1e-5, method="central")
        with pytest.raises(TypeError, match=r"is not callable"):
            create_lions_source(object(), fd, weights=0.1)


def _ring_problem(grid_only=False, amp=5.0, length_scale=0.15, bowl=4.0):
    """Build a 1D towel-on-the-beach problem.

    Central attractive potential (bowl, cost-signed) always present; the
    repulsive interaction is added only when ``grid_only`` is False.
    """
    H = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: 0.0,
        coupling_dm=lambda m: 0.0,
    )

    def m_initial(xx):
        return np.exp(-((xx - 0.5) ** 2) / (2 * 0.12**2))

    components = MFGComponents(hamiltonian=H, u_terminal=lambda xx: 0.0, m_initial=m_initial)
    problem = MFGProblem(
        geometry=TensorProductGrid(
            bounds=[(0.0, 1.0)], Nx_points=[21 + 1], boundary_conditions=no_flux_bc(dimension=1)
        ),
        T=0.5,
        Nt=4,
        sigma=0.2,
        components=components,
    )
    g = problem.geometry.get_spatial_grid().ravel()
    dx = g[1] - g[0]
    potential = PotentialEnergy(bowl * (g - 0.5) ** 2, dx)  # attractive bowl (cost away from centre)
    if grid_only:
        energy = potential
    else:
        conv = ConvolutionCouplingOperator(
            GaussianKernel(amplitude=amp, length_scale=length_scale),
            grid_shape=(len(g),),
            spacings=[dx],
            use_fft=True,
        )
        energy = CombinedEnergy([QuadraticInteractionEnergy(conv), potential])
    problem.source_term_hjb = create_lions_source(energy)
    return problem, g


def _solve_terminal_density(problem, g, iters=3):
    hjb = HJBFDMSolver(problem)
    fp = FPFDMSolver(problem)
    iterator = FixedPointIterator(problem, hjb, fp, config=MFGSolverConfig(max_iterations=iters))
    result = iterator.solve()
    M = result.M if hasattr(result, "M") else result[1]
    m_terminal = M[-1].ravel()
    m_terminal = m_terminal / np.trapezoid(m_terminal, g)
    return m_terminal


class TestGate4RingEquilibrium:
    """Non-local repulsion depletes the centre and spreads density outward."""

    @pytest.mark.slow  # coupled FixedPointIterator solve; deselected on PR-CI (30-min budget)
    def test_central_depletion_and_outward_spread(self):
        prob_attract, g = _ring_problem(grid_only=True)
        m_attract = _solve_terminal_density(prob_attract, g)

        prob_ring, g = _ring_problem(grid_only=False)
        m_ring = _solve_terminal_density(prob_ring, g)

        assert np.all(np.isfinite(m_attract))
        assert np.all(np.isfinite(m_ring))

        ci = int(np.argmin(np.abs(g - 0.5)))
        # Central depletion: the repulsive non-local coupling lowers the centre.
        assert m_ring[ci] < 0.7 * m_attract[ci]

        # Outward spread: variance about the centre increases with interaction.
        var_attract = np.trapezoid(m_attract * (g - 0.5) ** 2, g)
        var_ring = np.trapezoid(m_ring * (g - 0.5) ** 2, g)
        assert var_ring > 1.5 * var_attract

        # Density stays a non-negative normalized measure.
        assert np.all(m_ring >= -1e-9)
        assert np.trapezoid(m_ring, g) == pytest.approx(1.0)
