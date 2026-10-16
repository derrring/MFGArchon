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

from mfgarchon import Conditions, Model
from mfgarchon.alg.numerical.coupling import FixedPointIterator
from mfgarchon.alg.numerical.coupling.lions_correction import (
    create_lions_source,
    create_nonlocal_source,
)
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.config import MFGSolverConfig, PicardConfig
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
from mfgarchon.utils.numerical.quadrature import quadrature_weights_1d


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
        source_nonlocal = create_nonlocal_source(W, quadrature_weights_1d(x))

        m = np.sin(np.pi * x) + 1.2
        v = np.zeros(N)
        r_analytic = source_analytic(t=0.0, x=x, v=v, m=m)
        r_nonlocal = source_nonlocal(t=0.0, x=x, v=v, m=m)
        # Both equal W @ (w * m) on the grid's trapezoid weights to machine precision, the end nodes included (#1642).
        np.testing.assert_allclose(r_analytic, r_nonlocal, atol=1e-12)

    def test_the_nonlocal_source_refuses_one_cell_volume_and_reads_the_measure(self):
        """A scalar weight is the rectangle rule: each end node gets a full cell past the domain, an O(dx) error. It
        is refused by name, and the remedy the refusal names, the problem's measure, solves (#1642)."""
        problem = MFGProblem(
            model=Model(
                hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)), volatility=0.2
            ),
            domain=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[41], boundary_conditions=no_flux_bc(1)),
            conditions=Conditions(u_terminal=lambda xx: 0.0, m_initial=lambda xx: 1.0, T=0.5),
            Nt=4,
        )
        x = problem.geometry.get_spatial_grid().ravel()
        W = GaussianKernel(amplitude=1.0, length_scale=0.1).matrix(x)
        with pytest.raises(ValueError, match=r"rectangle rule.*problem\.spatial_measure\(\)\.weights"):
            create_nonlocal_source(W, x[1] - x[0])
        source = create_nonlocal_source(W, problem.spatial_measure().weights)
        m = np.ones_like(x)
        np.testing.assert_allclose(
            source(t=0.0, x=x, v=np.zeros_like(x), m=m), W @ quadrature_weights_1d(x), atol=1e-14
        )

    @pytest.mark.parametrize(
        "form",
        [
            "dense",
            "csr_matrix",
            "csr_array",
            "LinearOperator",
            "an object with @",
            "a wrapper of csr_array",
            "a wrapper of a LinearOperator",
            "an object with @ and a list for its shape",
        ],
    )
    def test_the_nonlocal_source_applies_a_sparse_or_matrix_free_kernel(self, form):
        """The nonlocal source is the path for a kernel that is not translation-invariant, which is where a sparse or
        matrix-free W is used: anything with a shape that applies itself with @ is applied as it is. np.asarray wrapped
        them in a 0-d object array, and the shape check refused them (#2726, reviews 1 and 2). A wrapper that forwards
        its shape through __getattr__ is one too; a check that looked the shape up statically missed it (review 3)."""
        from scipy.sparse import csr_array, csr_matrix
        from scipy.sparse.linalg import aslinearoperator

        from mfgarchon.operators.interaction.kernels import WendlandKernel

        x = np.linspace(0.0, 1.0, 41)
        W = WendlandKernel(1.0, 0.1).matrix(x)

        class _AppliesItself:
            shape = W.shape

            def __matmul__(self, v):
                return W @ v

        class _ListShape:
            def __init__(self):
                self.shape = list(W.shape)

            def __matmul__(self, v):
                return W @ v

        class _Forwarding:
            def __init__(self, inner):
                self._inner = inner

            def __getattr__(self, name):
                return getattr(self._inner, name)

            def __matmul__(self, v):
                return self._inner @ v

        kernel = {
            "dense": W,
            "csr_matrix": csr_matrix(W),
            "csr_array": csr_array(W),
            "LinearOperator": aslinearoperator(W),
            "an object with @": _AppliesItself(),
            "a wrapper of csr_array": _Forwarding(csr_array(W)),
            "a wrapper of a LinearOperator": _Forwarding(aslinearoperator(W)),
            "an object with @ and a list for its shape": _ListShape(),
        }[form]
        w, m = quadrature_weights_1d(x), 1.0 + x
        source = create_nonlocal_source(kernel, w)
        np.testing.assert_allclose(source(t=0.0, x=x, v=np.zeros_like(x), m=m), W @ (w * m), atol=1e-15)

    @pytest.mark.parametrize("operator", ["ConvolutionCouplingOperator", "GraphonCouplingOperator"])
    def test_the_nonlocal_source_refuses_the_library_s_weighted_operators(self, operator):
        """The library's weighted operators' @ is already the integral, weights included, so as W they would weight
        m twice (#2726, reviews 2 and 3). Each is refused, and the remedies its message names run: for the
        convolution, its own channel create_lions_source(QuadraticInteractionEnergy(op)) and its kernel_matrix(),
        both equal to op @ m; for the graphon, the matrix of its kernel, here the constant 0.5, whose integral against
        m = 1 on [0, 1] is 0.5."""
        from mfgarchon.operators.integro_diff.graphon_coupling import GraphonCouplingOperator
        from mfgarchon.operators.integro_diff.graphon_kernels import ConstantGraphon

        x = np.linspace(0.0, 1.0, 21)
        w = quadrature_weights_1d(x)
        v = np.zeros_like(x)
        if operator == "ConvolutionCouplingOperator":
            op = ConvolutionCouplingOperator(GaussianKernel(1.0, 0.1), grid_shape=(21,), spacings=[x[1]])
            with pytest.raises(ValueError, match=r"weight m twice.*QuadraticInteractionEnergy\(op\).*kernel_matrix"):
                create_nonlocal_source(op, w)
            m = 1.0 + x
            channel = create_lions_source(QuadraticInteractionEnergy(op))
            np.testing.assert_allclose(channel(t=0.0, x=x, v=v, m=m), op @ m, atol=1e-14)
            np.testing.assert_allclose(
                create_nonlocal_source(op.kernel_matrix(), w)(t=0.0, x=x, v=v, m=m), op @ m, atol=1e-14
            )
            return
        kernel = ConstantGraphon(p=0.5)
        with pytest.raises(ValueError, match=r"GraphonCouplingOperator.*weight m twice.*kernel\.evaluate\(x, x\)"):
            create_nonlocal_source(GraphonCouplingOperator(x, kernel), w)
        source = create_nonlocal_source(kernel.evaluate(x, x), w)
        np.testing.assert_allclose(source(t=0.0, x=x, v=v, m=np.ones_like(x)), 0.5, atol=1e-14)

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
        r_analytic = source_analytic(t=0.0, x=x, v=v, m=m)
        r_fd = source_fd(t=0.0, x=x, v=v, m=m)
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
        w = quadrature_weights_1d(x)  # the trapezoid weights of the grid's measure (#1642)

        def f_physical(m):
            m = np.asarray(m).ravel()
            # F[m] = (1/2) sum_ij w_i w_j W_ij m_i m_j -- BOTH quadrature factors.
            return 0.5 * float(np.sum(w * m * (W @ (w * m))))

        m = np.sin(np.pi * x) + 1.2
        assert f_physical(m) == pytest.approx(energy.energy(m), rel=1e-12)

        fd = FiniteDifferenceFunctionalDerivative(epsilon=1e-5, method="central")
        source_user = create_lions_source(f_physical, fd, weights=w)
        r_user = source_user(t=0.0, x=x, v=np.zeros(N), m=m)
        r_analytic = create_lions_source(energy)(t=0.0, x=x, v=np.zeros(N), m=m)
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
        """A weights/density length mismatch must not broadcast into a wrong number: the density is refused by name,
        a slice on another grid than the weights' (#2728)."""
        fd = FiniteDifferenceFunctionalDerivative(epsilon=1e-5, method="central")

        def energy_lambda(m):
            return 0.5 * float(np.sum(np.asarray(m) ** 2)) * 0.02

        source = create_lions_source(energy_lambda, fd, weights=np.full(7, 0.02))
        with pytest.raises(ValueError, match=r"one time-t density slice of 7 values.*shape \(10,\).*another grid"):
            source(t=0.0, x=np.linspace(0, 1, 10), v=np.zeros(10), m=np.ones(10))

    def test_fd_path_requires_functional_derivative(self):
        """Backward compat: plain callable without FD instance raises clearly."""

        def energy_lambda(m):
            return 0.5 * np.sum(m**2)

        with pytest.raises(ValueError):
            create_lions_source(energy_lambda)

    def test_time_space_array_rejected(self):
        """A (Nt+1, Nx) trajectory is a caller error: the source pipeline
        time-slices before calling, so a per-time source must get one spatial
        slice. Passing the full trajectory raises (was a silent m[-1] fallback
        that reintroduced the Issue #1285 wrong-slice bug). The 1-D slice path
        still works on both the analytic and the optimized nonlocal source.

        On a 7 x 5 grid a (3, 7, 5) trajectory is refused by name on all three
        routes -- the analytic, the optimized nonlocal and the finite-difference
        source with one scalar weight -- since its size is not the 35 nodes.
        Read by ``ndim`` it was not 2-D, so it fell through: the analytic and
        nonlocal routes raised a shape error naming no trajectory, and the
        scalar-weight route raised nothing and returned a field of 105 values
        (#2728)."""
        N = 30
        x = np.linspace(0.0, 1.0, N)
        dx = x[1] - x[0]
        conv = ConvolutionCouplingOperator(GaussianKernel(1.0, 0.1), grid_shape=(N,), spacings=[dx])
        source = create_lions_source(QuadraticInteractionEnergy(conv))
        W = np.exp(-((x[:, None] - x[None, :]) ** 2) / (2 * 0.2**2))
        source_nonlocal = create_nonlocal_source(W, quadrature_weights_1d(x))

        m_slice = np.sin(np.pi * x) + 1.0
        m_traj = np.tile(m_slice, (5, 1))  # (Nt+1, Nx), constant in time

        # 1-D slice path still works.
        r_slice = source(t=0.0, x=x, v=np.zeros(N), m=m_slice)
        assert r_slice.shape == (N,)

        # 2-D trajectory rejected on both paths.
        with pytest.raises(ValueError, match=r"one time-t density slice of 30 values.*shape \(5, 30\)"):
            source(t=0.0, x=x, v=np.zeros(N), m=m_traj)
        with pytest.raises(ValueError, match=r"one time-t density slice of 30 values.*shape \(5, 30\)"):
            source_nonlocal(t=0.0, x=x, v=np.zeros(N), m=m_traj)

        # On a 7 x 5 grid, a (3, 7, 5) trajectory on every route.
        from mfgarchon.utils.functional_calculus import FiniteDifferenceFunctionalDerivative

        xs, ys = np.linspace(0.0, 1.0, 7), np.linspace(0.0, 2.0, 5)
        nodes = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)
        cell = (xs[1] - xs[0]) * (ys[1] - ys[0])
        plane = ConvolutionCouplingOperator(
            GaussianKernel(1.0, 0.5), grid_shape=(7, 5), spacings=[xs[1] - xs[0], ys[1] - ys[0]]
        )
        W2 = np.exp(-(np.linalg.norm(nodes[:, None] - nodes[None], axis=2) ** 2))
        routes = {
            "analytic": create_lions_source(QuadraticInteractionEnergy(plane)),
            "nonlocal": create_nonlocal_source(W2, np.full(35, cell)),
            "finite-difference": create_lions_source(
                lambda m: float(np.sum(cell * np.asarray(m, dtype=float))),
                FiniteDifferenceFunctionalDerivative(),
                weights=cell,
            ),
        }
        m_grid_traj = np.ones((3, 7, 5))
        for name, routed in routes.items():
            with pytest.raises(ValueError, match=r"one time-t density slice of 35 values.*shape \(3, 7, 5\)"):
                routed(t=0.0, x=nodes, v=np.zeros(35), m=m_grid_traj)
            assert np.asarray(routed(t=0.0, x=nodes, v=np.zeros(35), m=m_grid_traj[0])).shape == (35,), name


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
        r = source(t=2.5, x=x, v=np.zeros(N), m=m)

        assert probe.seen_flat == [2.5]
        np.testing.assert_allclose(r, np.full(N, 2.5), rtol=0, atol=0)

    def test_analytic_dispatch_t_is_not_a_constant(self):
        """Two different slices must give two different sources."""
        N = 8
        probe = _TimeStampedEnergy(np.full(N, 0.125))
        source = create_lions_source(probe)
        x = np.linspace(0.0, 1.0, N)
        m = np.ones(N)

        r0 = source(t=0.0, x=x, v=np.zeros(N), m=m)
        r1 = source(t=1.0, x=x, v=np.zeros(N), m=m)
        r2 = source(t=3.0, x=x, v=np.zeros(N), m=m)

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
        """End-to-end: source_term_hjb(t, x, v, m) -> CombinedEnergy -> component."""
        N = 5
        w = np.full(N, 0.25)
        a = _TimeStampedEnergy(w, scale=2.0)
        b = _TimeStampedEnergy(w, scale=5.0)
        source = create_lions_source(CombinedEnergy([a, b]))

        r = source(t=1.75, x=np.linspace(0.0, 1.0, N), v=np.zeros(N), m=np.ones(N))

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

    def test_plain_callable_is_still_accepted(self):
        """The refusal must not swallow the legitimate FD path."""
        fd = FiniteDifferenceFunctionalDerivative(epsilon=1e-5, method="central")
        source = create_lions_source(lambda m: 0.5 * float(np.sum(np.asarray(m) ** 2)) * 0.1, fd, weights=0.1)
        m = np.linspace(1.0, 2.0, 8)
        np.testing.assert_allclose(source(t=0.0, x=np.linspace(0, 1, 8), v=np.zeros(8), m=m), m, rtol=1e-6)


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
        volatility=0.2,
        components=components,
    )
    g = problem.geometry.get_spatial_grid().ravel()
    # Both energies on the problem's measure (#1642): an attractive bowl (cost away from the centre) and the interaction.
    potential = PotentialEnergy(bowl * (g - 0.5) ** 2, problem.spatial_measure().weights)
    if grid_only:
        energy = potential
    else:
        conv = ConvolutionCouplingOperator.from_problem(
            GaussianKernel(amplitude=amp, length_scale=length_scale), problem, use_fft=True
        )
        energy = CombinedEnergy([QuadraticInteractionEnergy(conv), potential])
    problem.source_term_hjb = create_lions_source(energy)
    return problem, g


def _solve_terminal_density(problem, g, iters=3):
    hjb = HJBFDMSolver(problem)
    fp = FPFDMSolver(problem)
    iterator = FixedPointIterator(problem, hjb, fp, config=MFGSolverConfig(picard=PicardConfig(max_iterations=iters)))
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
