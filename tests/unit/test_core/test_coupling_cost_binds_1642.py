"""#1642 stage 1: ``Model(coupling_cost=...)`` binds to the problem's grid and measure, and the HJB carries its flat
derivative ``delta F / delta m`` as a cost-signed source.

- The plumbing: a linear F = int V m, bound by ``PotentialCost(V)``, against the same V as the Hamiltonian's potential.
- The measure: the bound ``InteractionCost`` against the exact box integral of its kernel.
- An m-dependent F: the linear-quadratic game of Bensoussan, Frehse and Yam (2013), ch. 6.2, in closed form.
- The pairing: what the coupling part receives, slice and time, per solver.
- Per population: each population's F reads its own density.
- What is refused, and the route each refusal names, run.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import pytest

import numpy as np
from scipy.special import erf

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.coupling import NewtonMFGSolver
from mfgarchon.alg.numerical.coupling.multi_population_iterator import MultiPopulationIterator
from mfgarchon.alg.numerical.coupling.source_composition import _coupling_part, compose_hjb_source
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver, HJBSemiLagrangianSolver, HJBWENOSolver
from mfgarchon.config import MFGSolverConfig, PicardConfig
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.multi_population import MultiPopulationProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.operators.interaction import InteractionCost, PotentialCost
from mfgarchon.operators.interaction.energy_functionals import energy_functional_members
from mfgarchon.operators.interaction.kernels import GaussianKernel, RadialKernel
from mfgarchon.types import NumericalScheme


def _hamiltonian(**kwargs):
    return SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0), **kwargs)


def _grid(n, bounds=(0.0, 1.0)):
    return TensorProductGrid(bounds=[bounds], Nx_points=[n], boundary_conditions=no_flux_bc(dimension=1))


def _unit_mass(density, n):
    """``density`` on [0, 1], scaled to mass 1 on the n-node grid's measure, the trapezoid."""
    xs = np.linspace(0.0, 1.0, n)
    mass = np.trapezoid(density(xs), xs)
    return lambda x: density(np.asarray(x, dtype=float)) / mass


def _x(problem):
    return np.asarray(problem.geometry.get_spatial_grid(), dtype=float).ravel()


def _V(x):
    x = np.asarray(x, dtype=float)
    return 0.8 * np.sin(2 * np.pi * x) + 0.5 * x


def _plumbing_problem(where, n=41, nt=20):
    """V as the Hamiltonian's potential (``"H"``), as ``PotentialCost(V)`` (``"F"``), or neither."""
    potential = (lambda t, x: _V(np.asarray(x, dtype=float)[..., 0])) if where == "H" else None
    return MFGProblem(
        model=Model(
            hamiltonian=_hamiltonian(potential=potential),
            volatility=0.3,
            coupling_cost=PotentialCost(_V) if where == "F" else None,
        ),
        domain=_grid(n),
        conditions=Conditions(
            u_terminal=lambda x: (np.asarray(x, dtype=float) - 0.7) ** 2,
            m_initial=_unit_mass(lambda x: np.exp(-30 * (x - 0.3) ** 2), n),
            T=0.5,
        ),
        Nt=nt,
    )


# Five sweeps, not a converged solve: both forms take the same iterates, so they agree at every sweep.
_PICARD = MFGSolverConfig(picard=PicardConfig(max_iterations=5, tolerance=1e-6, relaxation=0.5, verbose=False))


def _picard(where, **kwargs):
    result = _plumbing_problem(where).solve(config=_PICARD, **kwargs)
    return np.asarray(result.U, dtype=float)


def _newton(where):
    """One Newton step from the initial guess, on an 11 x 5 grid: its Jacobian is by finite differences."""
    problem = _plumbing_problem(where, n=11, nt=5)
    U, _, _ = NewtonMFGSolver(
        problem, HJBFDMSolver(problem), FPFDMSolver(problem), picard_warmup=0, newton_max_iterations=1
    ).solve(max_iterations=1, tolerance=1e-14, verbose=False)
    return np.asarray(U, dtype=float)


@pytest.mark.parametrize("route", ["Picard, HJB-FDM", "Newton, HJB-FDM", "Picard, HJB-SL"])
def test_a_linear_coupling_cost_solves_as_the_same_potential_in_the_hamiltonian(route):
    """F = int V m has ``delta F / delta m = V``, so ``PotentialCost(V)`` and V as H's potential pose one game, and
    each route takes the same iterates with either.

    - HJB-FDM adds the source where it adds V. Five Picard sweeps agree to 1.1e-16; one Newton step, to 6.9e-18.
    - HJB-SL takes V inside its update from H and adds the source at t_{n+1}, so the two differ by O(dt): after five
      sweeps, 8.8e-3 at Nt = 20 and 6.7e-3 at Nt = 40. The bound is 0.015 at Nt = 20.
    - Without either, five sweeps move U by 0.26: the scale a dropped part would show.
    """
    if route == "Picard, HJB-FDM":
        gap, bound = np.max(np.abs(_picard("H") - _picard("F"))), 1e-14
    elif route == "Newton, HJB-FDM":
        gap, bound = np.max(np.abs(_newton("H") - _newton("F"))), 1e-14
    else:
        sl = {"scheme": NumericalScheme.SL_LINEAR}
        gap, bound = np.max(np.abs(_picard("H", **sl) - _picard("F", **sl))), 0.015
    assert gap <= bound, f"{route}: PotentialCost(V) and V in H differ by {gap:.3e} (bound {bound:.1e})"
    if route == "Picard, HJB-FDM":
        dropped = np.max(np.abs(_picard("H") - _picard("neither")))
        assert dropped > 0.1, f"the fixture does not see V: dropping it moves U by {dropped:.3e}"


def test_the_bound_interaction_cost_integrates_on_the_problems_measure():
    """``InteractionCost(K).bind(problem)``'s ``flat_derivative(1)`` is ``(K * 1)(x)``, here the exact box integral of a
    Gaussian of amplitude A and length l on [0, 1]: ``A l sqrt(pi/2) [erf((1 - x)/(sqrt2 l)) + erf(x/(sqrt2 l))]``.

    On the grid's measure, the trapezoid, it is within 4.9e-6 at N = 321 and second order; the same sum with dx at
    every node, rectangle weights, is 1.6e-3 off, first order."""
    amplitude, length = 1.0, 0.1
    problem = MFGProblem(
        model=Model(
            hamiltonian=_hamiltonian(),
            volatility=0.1,
            coupling_cost=InteractionCost(GaussianKernel(amplitude=amplitude, length_scale=length)),
        ),
        domain=_grid(321),
        conditions=Conditions(u_terminal=lambda x: 0.0, m_initial=lambda x: 1.0, T=0.1),
        Nt=2,
    )
    x = _x(problem)
    exact = (
        amplitude
        * length
        * np.sqrt(np.pi / 2)
        * (erf((1 - x) / (np.sqrt(2) * length)) + erf(x / (np.sqrt(2) * length)))
    )
    gap = np.max(np.abs(problem.coupling_energy.flat_derivative(np.ones(x.size)) - exact))
    assert gap < 2e-5, f"the bound convolution misses the exact (K * 1)(x) by {gap:.3e}"


class _Quadratic(RadialKernel):
    """``K(r) = c r^2``: ``delta F / delta m = c [(x - z)^2 + Var(m)]`` for a density of mass 1 and mean z."""

    def __init__(self, c):
        self.c = c

    def profile(self, r):
        return self.c * np.asarray(r, dtype=float) ** 2

    @property
    def is_repulsive(self):
        return False


def test_an_m_dependent_coupling_cost_solves_the_linear_quadratic_game():
    """Bensoussan, Frehse and Yam, *Mean Field Games and Mean Field Type Control Theory* (2013), ch. 6.2, pp. 46-48.

    In 1-D take A = Abar = Q = 0, B = R = 1, Qbar = 2c, S = 1, Q_T = q, Qbar_T = 0. Their running cost
    ``(1/2)(x - S z)* Qbar (x - S z)``, z the mean of m, is then ``c (x - z)^2``: the flat derivative of
    ``F[m] = (1/2) int int c |x - y|^2 m(x) m(y)`` up to ``c Var(m)``, which does not depend on x. So
    ``u_x = P(t) x + r(t)`` (6.6), with, in tau = T - t and k = sqrt(2c):
    - (6.9) ``dP/dt - P^2 + 2c = 0``, P(T) = q: ``P = k (q + k tanh(k tau)) / (k + q tanh(k tau))``;
    - (6.21) ``dSigma/dt - Sigma^2 = 0``, Sigma(T) = q: ``Sigma = q / (1 + q tau)``;
    - (6.17), (6.20) ``z' = -Sigma z``: ``z = z0 (1 + q tau) / (1 + q T)``; and (6.22) ``r = (Sigma - P) z``.

    The solve on [-3, 3], 121 nodes and 50 steps, matches u_x within 1 of the mean at t = 0 to 2.55e-2. The
    difference is first order: 1.29e-2 and 6.5e-3 at twice and four times the resolution. Without F it is 0.83;
    with m frozen at m_0, 0.125."""
    c, q, T, z0, s0 = 1.0, 1.0, 1.0, 0.5, 0.3
    problem = MFGProblem(
        model=Model(hamiltonian=_hamiltonian(), volatility=0.4, coupling_cost=InteractionCost(_Quadratic(c))),
        domain=_grid(121, (-3.0, 3.0)),
        conditions=Conditions(
            u_terminal=lambda x: 0.5 * q * np.asarray(x, dtype=float) ** 2,
            m_initial=lambda x: (
                np.exp(-((np.asarray(x, dtype=float) - z0) ** 2) / (2 * s0**2)) / np.sqrt(2 * np.pi * s0**2)
            ),
            T=T,
        ),
        Nt=50,
    )
    result = problem.solve(
        config=MFGSolverConfig(picard=PicardConfig(max_iterations=60, tolerance=1e-6, relaxation=0.7, verbose=False))
    )
    assert result.converged
    k, tau = np.sqrt(2 * c), T
    P = k * (q + k * np.tanh(k * tau)) / (k + q * np.tanh(k * tau))
    z = z0  # z(t = 0)
    r = (q / (1 + q * tau) - P) * z
    x = _x(problem)
    u_x = np.gradient(np.asarray(result.U, dtype=float)[0], x[1] - x[0])
    near = np.abs(x - z) < 1.0
    gap = np.max(np.abs(u_x - (P * x + r))[near])
    assert gap < 0.04, f"u_x at t = 0 misses P x + r = {P:.4f} x {r:+.4f} by {gap:.3e} near the mean"


def test_the_bound_nodes_and_the_density_slice_share_the_grids_order():
    """The node order has one owner, the rows of ``geometry.get_spatial_grid()``: ``bind`` samples V and reads the
    weights in it, and the coupling part flattens each density slice, shape (Nx, Ny), into it. On a 7 x 5 grid, with a
    V and an m that are not symmetric in x and y, the composed source at those rows is V there, and ``(K * m)`` there
    summed node by node. A slice flattened in the other order would put the values on the wrong nodes."""
    xs, ys = np.linspace(0.0, 1.0, 7), np.linspace(0.0, 2.0, 5)
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 2.0)], Nx_points=[7, 5], boundary_conditions=no_flux_bc(dimension=2)
    )
    kernel = GaussianKernel(amplitude=1.0, length_scale=0.5)

    def V(x):
        return x[0] + 3.0 * x[1] ** 2

    def m(x, y):
        return 1.0 + x + 2.0 * y**2

    def problem(cost):
        return MFGProblem(
            model=Model(hamiltonian=_hamiltonian(), volatility=0.3, coupling_cost=cost),
            domain=grid,
            conditions=Conditions(u_terminal=lambda x: 0.0, m_initial=lambda x: 0.5, T=0.2),
            Nt=2,
        )

    X, Y = np.meshgrid(xs, ys, indexing="ij")
    M = np.array([m(X, Y)] * 3)
    U = np.zeros_like(M)
    nodes = np.asarray(grid.get_spatial_grid(), dtype=float)
    for cost, expected in (
        (PotentialCost(V), np.array([V(p) for p in nodes])),
        (
            InteractionCost(kernel),
            np.array(
                [
                    np.sum(
                        kernel(np.linalg.norm(nodes - p, axis=1))
                        * problem(None).spatial_measure().weights.ravel()
                        * m(nodes[:, 0], nodes[:, 1])
                    )
                    for p in nodes
                ]
            ),
        ),
    ):
        p = problem(cost)
        source = compose_hjb_source(p, M, U)(0.0, nodes)
        np.testing.assert_allclose(source, expected, rtol=1e-12, atol=1e-12, err_msg=type(cost).__name__)


class _RecordingCost:
    """A coupling cost whose bound energy is ``PotentialEnergy`` of zero, recording each ``flat_derivative`` call's
    (t, m)."""

    def __init__(self):
        self.calls = []

    def bind(self, problem):
        energy = PotentialCost(lambda x: 0.0).bind(problem)
        calls = self.calls

        class _Recording:
            def __getattr__(self, name):
                return getattr(energy, name)

            def flat_derivative(self, m, *, t=0.0):
                calls.append((float(t), np.array(m, dtype=float)))
                return energy.flat_derivative(m, t=t)

        recording = _Recording()
        assert all(hasattr(recording, name) for name in energy_functional_members())
        return recording


@pytest.mark.parametrize(
    ("solver", "levels"),
    [
        (HJBFDMSolver, "t_n with m_n"),
        (HJBSemiLagrangianSolver, "t_{n+1} with m_{n+1}"),
        (HJBWENOSolver, "each stage time t with m at round(t / dt)"),
    ],
)
def test_the_coupling_part_reads_the_density_slice_at_the_time_it_is_given(solver, levels):
    """The part evaluates ``delta F / delta m`` at the slice of m its solver's time selects; it compensates for
    nothing. Which (t, slice) each solver pairs is the solvers' and the source primitive's (#2557): an O(dt) offset
    in the part would leave a converged-solution oracle green, so this pins it per call. Each slice of m is distinct."""
    cost = _RecordingCost()
    nt = 4
    problem = MFGProblem(
        model=Model(hamiltonian=_hamiltonian(), volatility=0.3, coupling_cost=cost),
        domain=_grid(21),
        conditions=Conditions(u_terminal=lambda x: 0.0, m_initial=lambda x: 1.0, T=0.2),
        Nt=nt,
    )
    x = _x(problem)
    M = np.array([(1.0 + n) * (1.0 + x) for n in range(nt + 1)])
    U = np.zeros_like(M)
    hjb = solver(problem)
    hjb.solve_hjb_system(M, U[-1], U, source_term=compose_hjb_source(problem, M, U))
    dt = problem.dt
    assert cost.calls, "the coupling part was never called"
    for t, m in cost.calls:
        np.testing.assert_array_equal(m, M[round(t / dt)], err_msg=f"at t = {t}")
    indices = sorted({round(t / dt) for t, _ in cost.calls})
    expected = {
        "t_n with m_n": list(range(nt)),
        "t_{n+1} with m_{n+1}": list(range(1, nt + 1)),
        "each stage time t with m at round(t / dt)": list(range(nt + 1)),
    }[levels]
    assert indices == expected, f"{solver.__name__} pairs {levels}: read slices {indices}"


def test_each_population_s_coupling_cost_reads_its_own_density():
    """Population k's ``Model`` carries its own F_k, composed on m_k (``hjb_solve_kwargs(populations[k], ...,
    M=M[k])``). Two populations with no cross-coupling, each with its own kernel and its own initial density,
    solve as two separate games: each population with the other present equals it alone in the iterator, and that
    equals the single-population Picard solve, to 1e-9. The last comparison is the oracle: the first runs one route
    twice, so an error the iterator makes for every K cancels in it. F is what each solves: without it, U is 0, and
    with it 0.40 and 0.23 at most, so two games that never composed F would agree as well."""

    def problem(amplitude, centre, cost=True):
        return MFGProblem(
            model=Model(
                hamiltonian=_hamiltonian(),
                volatility=0.3,
                coupling_cost=InteractionCost(GaussianKernel(amplitude=amplitude, length_scale=0.2)) if cost else None,
            ),
            domain=_grid(31),
            conditions=Conditions(
                u_terminal=lambda x: 0.0,
                m_initial=_unit_mass(lambda x: np.exp(-40 * (x - centre) ** 2), 31),
                T=0.3,
            ),
            Nt=6,
        )

    specs = [(2.0, 0.3), (-1.0, 0.7)]

    def solve(problems):
        result = MultiPopulationIterator(
            MultiPopulationProblem(populations=problems, population_names=[f"p{k}" for k in range(len(problems))]),
            [HJBFDMSolver(p, newton_tolerance=1e-13) for p in problems],
            [FPFDMSolver(p) for p in problems],
            relaxation=0.5,
        ).solve(max_iterations=200, tolerance=1e-12)
        assert result.converged
        return result.U

    together = solve([problem(*s) for s in specs])
    without = solve([problem(*s, cost=False) for s in specs])
    for k, spec in enumerate(specs):
        alone = solve([problem(*spec)])[0]
        gap = np.max(np.abs(np.asarray(together[k]) - np.asarray(alone)))
        assert gap < 1e-9, f"population {k} with the other present differs from it alone by {gap:.3e}"
        single = problem(*spec)
        result = single.solve(
            hjb_solver=HJBFDMSolver(single, newton_tolerance=1e-13),
            fp_solver=FPFDMSolver(single),
            max_iterations=200,
            tolerance=1e-12,
        )
        assert result.converged
        gap = np.max(np.abs(np.asarray(alone) - np.asarray(result.U)))
        assert gap < 1e-9, f"population {k} in the iterator differs from its single-population solve by {gap:.3e}"
        moved = np.max(np.abs(np.asarray(together[k]) - np.asarray(without[k])))
        assert moved > 0.1, f"F_{k} moves population {k}'s U by {moved:.3e}: the fixture does not see it"


def test_a_users_coupling_cost_sums_with_the_librarys():
    """A ``CouplingCost`` is an instance of anything with ``bind(problem)``. ``+`` takes a user's beside the library's
    in either order; ``sum`` does where its first element is the library's, since it starts from ``0 + costs[0]``,
    which a user's class without ``__radd__`` cannot answer. Each binds to the sum of the parts' flat derivatives.
    A part whose binding is not an ``EnergyFunctional`` is refused at construction by its own name, as it is alone; in
    a sum it was admitted, and the first evaluation raised an AttributeError."""

    class Mine:
        def bind(self, problem):
            return PotentialCost(lambda x: 2.0 * x).bind(problem)

    problem = _plumbing_problem("neither", n=11)
    x = _x(problem)
    for cost in (PotentialCost(_V) + Mine(), Mine() + PotentialCost(_V), sum([PotentialCost(_V), Mine()])):
        np.testing.assert_allclose(
            cost.bind(problem).flat_derivative(np.ones(x.size)), _V(x) + 2.0 * x, rtol=0, atol=1e-14, err_msg=repr(cost)
        )

    class Broken:
        def bind(self, problem):
            energy = PotentialCost(_V).bind(problem)
            return SimpleNamespace(weights=energy.weights, energy=energy.energy)

    with pytest.raises(TypeError, match=r"Broken\.bind\(problem\) returned SimpleNamespace.*lacks .*flat_derivative"):
        MFGProblem(
            model=Model(hamiltonian=_hamiltonian(), volatility=0.3, coupling_cost=PotentialCost(_V) + Broken()),
            domain=_grid(11),
            conditions=Conditions(u_terminal=lambda x: 0.0, m_initial=lambda x: 1.0, T=0.2),
            Nt=2,
        )


def test_what_stage_one_does_not_bind_is_refused_and_each_named_route_runs():
    """- A plain ``F[m] -> float`` carries its own quadrature, so it is refused; the route the message names, the
      finite-difference Lions source on the grid's weights, recovers ``delta F / delta m = V`` for F = int V m. Run as
      written through ``MFGProblem`` on a 7 x 5 grid, where the composer hands it a (7, 5) slice, it is V within
      4.2e-10, with the grid's weights or with one scalar weight, dx dy. The other two Lions sources read that slice
      too: ``create_nonlocal_source(W, w)`` gives W @ (w m), and ``create_lions_source`` on a bound energy its flat
      derivative (#2728). Each refused the slice as a trajectory.
    - ``PotentialCost(V)`` with a ``V(t, x)`` is refused: binding reads V once. The route it names, the Hamiltonian's
      ``potential(t, x)``, reads V at each time, at an array of points, so the 1-D V reads ``x[..., 0]`` as the
      message says: with ``5 x (1 + 10 t)`` it solves 0.57 away from ``5 x``. A V the reader cannot take, ``V(x, t)``,
      is refused naming the same route.
    - A ``terminal_coupling`` that is not a coupling cost, a plain ``G(x, m)``, is refused, at the ``Model`` and again where
      a problem binds it, so one set after the ``Model`` was built is refused there too.
    - On a network no solve takes a ``Model`` (#2045); the network's own route, ``node_potential_func``, reaches its
      HJB's source.
    - The part evaluates only at the nodes it was bound on: an x of another row count, the nodes in another order, or
      other points are refused, at the part and through the composer, which passes the grid's nodes."""
    from mfgarchon.alg.numerical.coupling.lions_correction import create_lions_source
    from mfgarchon.utils.functional_calculus import FiniteDifferenceFunctionalDerivative

    grid = _grid(21)
    w = grid.measure_weights().ravel()
    x = np.asarray(grid.get_spatial_grid(), dtype=float).ravel()

    def F(m):
        return float(np.sum(w * _V(x) * np.asarray(m, dtype=float)))

    with pytest.raises(TypeError, match=r"create_lions_source.*source_term_hjb"):
        Model(hamiltonian=_hamiltonian(), coupling_cost=F)
    source = create_lions_source(F, FiniteDifferenceFunctionalDerivative(), weights=grid.measure_weights().ravel())
    np.testing.assert_allclose(source(0.0, x, np.zeros_like(x), np.ones_like(x)), _V(x), rtol=0, atol=1e-6)
    plane = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 2.0)], Nx_points=[7, 5], boundary_conditions=no_flux_bc(dimension=2)
    )
    nodes = np.asarray(plane.get_spatial_grid(), dtype=float)
    V2 = nodes[:, 0] + 3.0 * nodes[:, 1] ** 2

    def F2(m):
        return float(np.sum(plane.measure_weights().ravel() * V2 * np.asarray(m, dtype=float).ravel()))

    advised = MFGProblem(
        model=Model(hamiltonian=_hamiltonian(), volatility=0.3),
        domain=plane,
        conditions=Conditions(u_terminal=lambda x: 0.0, m_initial=lambda x: 0.5, T=0.2),
        Nt=2,
        source_term_hjb=create_lions_source(
            F2, FiniteDifferenceFunctionalDerivative(), weights=plane.measure_weights().ravel()
        ),
    )
    M = np.ones((3, 7, 5))
    np.testing.assert_allclose(compose_hjb_source(advised, M, 0 * M)(0.0, nodes), V2, rtol=0, atol=1e-8)
    cell = float(np.prod(plane.get_grid_spacing()))
    scalar = MFGProblem(
        model=Model(hamiltonian=_hamiltonian(), volatility=0.3),
        domain=plane,
        conditions=Conditions(u_terminal=lambda x: 0.0, m_initial=lambda x: 0.5, T=0.2),
        Nt=2,
        source_term_hjb=create_lions_source(
            lambda m: float(np.sum(cell * V2 * np.asarray(m, dtype=float).ravel())),
            FiniteDifferenceFunctionalDerivative(),
            weights=cell,
        ),
    )
    np.testing.assert_allclose(compose_hjb_source(scalar, M, 0 * M)(0.0, nodes), V2, rtol=0, atol=1e-8)
    from mfgarchon.alg.numerical.coupling.lions_correction import create_nonlocal_source

    W = np.exp(-(np.linalg.norm(nodes[:, None] - nodes[None], axis=2) ** 2))
    energy = InteractionCost(GaussianKernel(amplitude=1.0, length_scale=0.5)).bind(advised)
    m2 = (1.0 + nodes[:, 0] + 2.0 * nodes[:, 1] ** 2).reshape(7, 5)
    for source, expected in (
        (
            create_nonlocal_source(W, plane.measure_weights().ravel()),
            W @ (plane.measure_weights().ravel() * m2.ravel()),
        ),
        (create_lions_source(energy), energy.flat_derivative(m2.ravel())),
    ):
        routed = MFGProblem(
            model=Model(hamiltonian=_hamiltonian(), volatility=0.3),
            domain=plane,
            conditions=Conditions(u_terminal=lambda x: 0.0, m_initial=lambda x: 0.5, T=0.2),
            Nt=2,
            source_term_hjb=source,
        )
        Ms = np.array([m2] * 3)
        np.testing.assert_allclose(compose_hjb_source(routed, Ms, 0 * Ms)(0.0, nodes), expected, rtol=1e-12, atol=1e-12)

    def timed(**kwargs):
        return MFGProblem(
            model=Model(volatility=0.3, **kwargs),
            domain=grid,
            conditions=Conditions(u_terminal=lambda x: 0.0, m_initial=lambda x: 1.0, T=0.2),
            Nt=4,
        )

    with pytest.raises(
        TypeError, match=r"V takes \(t, x\).*SeparableHamiltonian\(potential=lambda t, x.*x\[\.\.\., 0\]"
    ):
        timed(hamiltonian=_hamiltonian(), coupling_cost=PotentialCost(lambda t, x: x * (1 + t)))
    with pytest.raises(TypeError, match=r"V takes the point x alone.*SeparableHamiltonian\(potential=lambda t, x"):
        timed(hamiltonian=_hamiltonian(), coupling_cost=PotentialCost(lambda x, t: x * (1 + t)))
    sweeps = MFGSolverConfig(picard=PicardConfig(max_iterations=3, tolerance=1e-6, verbose=False))
    moving, frozen = (
        np.asarray(timed(hamiltonian=_hamiltonian(potential=V)).solve(config=sweeps).U)
        for V in (
            lambda t, x: 5.0 * np.asarray(x)[..., 0] * (1 + 10 * t),
            lambda t, x: 5.0 * np.asarray(x)[..., 0],
        )
    )
    assert np.max(np.abs(moving - frozen)) > 0.1, "the Hamiltonian's potential(t, x) is not read at each time"

    with pytest.raises(TypeError, match=r"Model\(terminal_coupling=\.\.\.\) takes a coupling cost.*not function"):
        Model(hamiltonian=_hamiltonian(), terminal_coupling=lambda x, m: 0.0)
    mutated = Model(hamiltonian=_hamiltonian(), volatility=0.3, coupling_cost=PotentialCost(_V))
    mutated.terminal_coupling = lambda x, m: 0.0
    with pytest.raises(TypeError, match=r"Model\(terminal_coupling=\.\.\.\) takes a coupling cost.*not function"):
        MFGProblem(
            model=mutated,
            domain=grid,
            conditions=Conditions(u_terminal=lambda x: 0.0, m_initial=lambda x: 1.0, T=0.2),
            Nt=2,
        )

    from scipy.sparse import csr_matrix

    from mfgarchon.alg.numerical.network_solvers.hjb_network import NetworkHJBSolver
    from mfgarchon.extensions.topology import NetworkMFGComponents, NetworkMFGProblem
    from mfgarchon.geometry.graph.network_geometry import CustomNetwork, GridNetwork

    pytest.importorskip("igraph")
    ring = np.zeros((6, 6))
    for i in range(6):
        ring[i, (i + 1) % 6] = ring[(i + 1) % 6, i] = 1.0
    with pytest.raises(NotImplementedError, match=r"network.*NetworkMFGComponents\(node_potential_func="):
        MFGProblem(
            model=Model(hamiltonian=_hamiltonian(), coupling_cost=PotentialCost(lambda x: 0.0)),
            domain=CustomNetwork(csr_matrix(ring)),
            conditions=Conditions(u_terminal=lambda x: 0.0, m_initial=lambda x: 1.0 / 6.0, T=0.2),
            Nt=4,
        )
    net = GridNetwork(width=5, height=1)
    net.create_network()
    network = NetworkMFGProblem(
        geometry=net, T=0.2, Nt=4, components=NetworkMFGComponents(node_potential_func=lambda t, i: 0.3 * i)
    )
    n = network.num_nodes
    spread = np.ptp(NetworkHJBSolver(network)._source_terms(np.ones(n) / n, 0.0))
    assert spread > 0.5, f"the node potential 0.3 i over 5 nodes spans 1.2 in the source; it spans {spread:.3e}"

    energy = PotentialCost(_V).bind(SimpleNamespace(geometry=grid, spatial_measure=lambda: SimpleNamespace(weights=w)))
    for other, rows in ((x[:20, None], 20), (x[::-1, None], 21), (x[:, None] + 0.01, 21)):
        with pytest.raises(
            ValueError, match=rf"bound on this problem's 21 grid nodes.*x with {rows} rows that are not"
        ):
            _coupling_part(energy, x[:, None], np.ones(21), other, 0.0)
    np.testing.assert_allclose(_coupling_part(energy, x[:, None], np.ones(21), x[:, None], 0.0), _V(x), rtol=0, atol=0)
    composed = _plumbing_problem("F", n=21, nt=2)
    M = np.ones((3, 21))
    with pytest.raises(ValueError, match=r"x with 21 rows that are not.*until #1642's stage 1b-1"):
        compose_hjb_source(composed, M, 0 * M)(0.0, x[::-1, None])


def test_a_terminal_coupling_solves_the_linear_quadratic_game():
    """Bensoussan, Frehse and Yam (2013), ch. 6, pp. 45-47, with the mean field at T only (#1642 stage 2).

    In 1-D take A = Abar = Q = Qbar = 0, B = R = 1, Q_T = 0, Qbar_T = k, S_T = c. The terminal cost (6.3), (6.4) is
    then ``(k/2)(x - c z(T))^2``, z the mean of m: up to a constant in x, the flat derivative at m(T), of mass 1, of
    ``G[m] = int (k/2)(1 - c) x^2 m + (1/2) int int (kc/2)|x - y|^2 m(x) m(y)``, PotentialCost plus InteractionCost.
    (6.9) gives ``P' = P^2``, P(T) = k; (6.12), (6.13) ``r' = P r``, ``r(T) = -k c z(T)``; (6.10) ``z' = -(P z + r)``.
    So P z + r is constant, k (1 - c) z(T), and ``z(T) = z0 / (1 + k (1 - c) T)``, whatever sigma.

    k = 2, c = 1/2, T = 1, z0 = 1/2: z(T) = 1/4. On [-3, 3] with 61 nodes and 25 steps the solve's mean at T misses it
    by 2.0e-2, first order: 1.0e-2 and 5.1e-3 at twice and four times the resolution. With the terminal frozen at m_0
    the solve converges to ``z0 (1 + k c T) / (1 + k T)`` = 1/3 instead, and misses 1/4 by 9.3e-2."""
    k, c, T, z0, s0 = 2.0, 0.5, 1.0, 0.5, 0.3
    problem = MFGProblem(
        model=Model(
            hamiltonian=_hamiltonian(),
            volatility=0.4,
            terminal_coupling=PotentialCost(lambda x: 0.5 * k * (1 - c) * x**2)
            + InteractionCost(_Quadratic(k * c / 2)),
        ),
        domain=_grid(61, (-3.0, 3.0)),
        conditions=Conditions(
            u_terminal=lambda x: 0.0,
            m_initial=lambda x: (
                np.exp(-((np.asarray(x, dtype=float) - z0) ** 2) / (2 * s0**2)) / np.sqrt(2 * np.pi * s0**2)
            ),
            T=T,
        ),
        Nt=25,
    )
    x = _x(problem)
    w = np.asarray(problem.spatial_measure().weights, dtype=float).ravel()
    m0 = np.asarray(problem.get_m_initial(), dtype=float).ravel()
    offset = np.asarray(problem.terminal_value(m0), dtype=float) - (0.5 * k * x**2 - k * c * (w @ (x * m0)) * x)
    assert np.ptp(offset) < 1e-12, (
        f"u(T) at m_0 is not (k/2) x^2 - k c z x up to a constant: spread {np.ptp(offset):.3e}"
    )
    result = problem.solve(
        config=MFGSolverConfig(picard=PicardConfig(max_iterations=80, tolerance=1e-6, relaxation=0.7, verbose=False))
    )
    assert result.converged
    mT = np.asarray(result.M, dtype=float)[-1]
    gap = abs(float((w @ (x * mT)) / (w @ mT)) - z0 / (1 + k * (1 - c) * T))
    assert gap < 0.03, f"the mean at T misses z0 / (1 + k (1 - c) T) = 1/4 by {gap:.3e}"


def _terminal_coupled(n=21, nt=5, centre=0.3):
    """G = (1/2) int int |x - y|^2 m m + int 3 x m on [0, 1]: u(T) moves with m(T), and m(T) moves from sweep to sweep
    because the initial density is off-centre. Two nodes or regimes take different centres, so neither holds the other's
    density."""
    return MFGProblem(
        model=Model(
            hamiltonian=_hamiltonian(),
            volatility=0.3,
            terminal_coupling=InteractionCost(_Quadratic(1.0)) + PotentialCost(lambda x: 3.0 * x),
        ),
        domain=_grid(n),
        conditions=Conditions(
            u_terminal=lambda x: 0.0, m_initial=_unit_mass(lambda x: np.exp(-40 * (x - centre) ** 2), n), T=0.3
        ),
        Nt=nt,
    )


def _recording(solver, calls, diverge_at=None):
    """``solver.solve_hjb_system`` recording its problem, the density and the terminal of each call, its signature
    kept. From call ``diverge_at`` on, counted over every solver sharing ``calls``, it returns a value function with one
    NaN, as a diverged inner Newton does."""
    import functools

    original = solver.solve_hjb_system

    @functools.wraps(original)
    def solve_hjb_system(M_density, U_terminal, *args, **kwargs):
        calls.append((solver.problem, np.array(M_density, dtype=float), np.array(U_terminal, dtype=float)))
        U = original(M_density, U_terminal, *args, **kwargs)
        if diverge_at is not None and len(calls) > diverge_at:
            U = np.array(U, dtype=float)
            U[0, 0] = np.nan
        return U

    solver.solve_hjb_system = solve_hjb_system
    return solver


def _run_loop(name, calls, diverge_at=None):
    """``(published, expected)``: the U(T) row a loop publishes and the u(T) it must equal, or ``(None, None)`` where
    the loop publishes no re-imposed or restored row."""
    from mfgarchon.alg.numerical.coupling.block_iterators import BlockGaussSeidelIterator, BlockJacobiIterator
    from mfgarchon.alg.numerical.coupling.fictitious_play import FictitiousPlayIterator
    from mfgarchon.alg.numerical.coupling.fixed_point_iterator import FixedPointIterator
    from mfgarchon.alg.numerical.coupling.graph_coupling import AdjacencyCoupling
    from mfgarchon.alg.numerical.coupling.graph_mfg_solver import GraphMFGSolver
    from mfgarchon.alg.numerical.coupling.regime_switching_iterator import RegimeSwitchingIterator
    from mfgarchon.core.regime_switching import RegimeSwitchingConfig

    loops = {
        "FixedPointIterator": FixedPointIterator,
        "FictitiousPlayIterator": FictitiousPlayIterator,
        "BlockGaussSeidelIterator": BlockGaussSeidelIterator,
        "BlockJacobiIterator": BlockJacobiIterator,
    }
    if name in loops:
        problem = _terminal_coupled()
        loop = loops[name](problem, _recording(HJBFDMSolver(problem), calls, diverge_at), FPFDMSolver(problem))
        loop.solve(max_iterations=4, tolerance=1e-6)
        return np.asarray(loop.U, dtype=float)[-1], calls[-1][2]
    if name == "NewtonMFGSolver":
        problem = _terminal_coupled(n=11)
        NewtonMFGSolver(
            problem,
            _recording(HJBFDMSolver(problem), calls),
            FPFDMSolver(problem),
            picard_warmup=2,
            newton_max_iterations=1,
        ).solve(max_iterations=1, tolerance=1e-14, verbose=False)
        return None, None
    if name == "NewtonMFGSolver warm-up":
        # A warm start whose m(T) is not m_0, and a tolerance the warm-up meets at once: Newton never runs, and the
        # solve publishes the warm-up's iterate, re-imposed (or restored, when its solve diverged) at that solve's u(T).
        problem = _terminal_coupled(n=11)
        solver = NewtonMFGSolver(
            problem, _recording(HJBFDMSolver(problem), calls, diverge_at), FPFDMSolver(problem), picard_warmup=1
        )
        M_w = np.tile(_unit_mass(lambda x: np.exp(-40 * (x - 0.7) ** 2), 11)(_x(problem)), (problem.Nt + 1, 1))
        solver.set_warm_start_data((np.zeros_like(M_w), M_w))
        U, _, info = solver.solve(max_iterations=1, tolerance=1e30, verbose=False)
        reason = "diverged_nan" if diverge_at is not None else "Converged during Picard warm-up"
        assert info["convergence_reason"] == reason, info["convergence_reason"]
        return np.asarray(U, dtype=float)[-1], np.asarray(problem.terminal_value(M_w[-1]), dtype=float)
    if name == "BaseHJBSolver.solve":
        problem = _terminal_coupled()
        _recording(HJBFDMSolver(problem), calls).solve()
        return None, None
    problems = [_terminal_coupled(centre=0.3), _terminal_coupled(centre=0.7)]
    hjb = [_recording(HJBFDMSolver(p), calls, diverge_at) for p in problems]
    fp = [FPFDMSolver(p) for p in problems]
    if name == "RegimeSwitchingIterator":
        result = RegimeSwitchingIterator(
            problems,
            RegimeSwitchingConfig(transition_matrix=np.array([[-0.5, 0.5], [0.5, -0.5]])),
            hjb,
            fp,
            max_iterations=4,
            tolerance=1e-6,
            damping=0.3,
        ).solve()
    else:
        result = GraphMFGSolver(
            problems,
            AdjacencyCoupling(np.array([[0.0, 1.0], [1.0, 0.0]]), alpha=0.05, beta=0.02),
            hjb,
            fp,
            max_iterations=4,
            tolerance=1e-6,
            damping=0.3,
        ).solve()
    if diverge_at is None:
        return None, None  # neither re-imposes u(T) after damping
    k = next(i for i, p in enumerate(problems) if p is calls[-1][0])
    return np.asarray(result.values[k], dtype=float)[-1], calls[-1][2]


_LOOPS = [
    "FixedPointIterator",
    "FictitiousPlayIterator",
    "BlockGaussSeidelIterator",
    "BlockJacobiIterator",
    "NewtonMFGSolver",
    "NewtonMFGSolver warm-up",
    "RegimeSwitchingIterator",
    "GraphMFGSolver",
    "BaseHJBSolver.solve",
]
#: The loops that restore u(T) after a diverged HJB solve (#1718), each diverging at the first call whose density is not
#: m_0's: the second sweep's, the second sweep's first node, or the warm start's.
_RESTORING = dict.fromkeys(_LOOPS[:4], 1) | {"NewtonMFGSolver warm-up": 0} | dict.fromkeys(_LOOPS[6:8], 2)


@pytest.mark.parametrize(
    ("loop", "diverge_at"),
    [(loop, None) for loop in _LOOPS] + [(loop, at) for loop, at in _RESTORING.items()],
    ids=list(_LOOPS) + [f"{loop}, diverged" for loop in _RESTORING],
)
def test_each_hjb_solve_is_given_u_T_at_its_own_density(loop, diverge_at):
    """Every HJB solve receives ``problem.terminal_value`` at the density at t = T of that same call (#1642 stage 2):
    each Picard sweep, each step of Newton's warm-up and each residual evaluation of its finite-difference Jacobian,
    each regime and graph node, and the standalone solve at its frozen density. A loop that re-imposes u(T) after its
    update publishes that solve's u(T), and so does each loop's restore after a diverged solve (#1718), Newton's warm-up
    included, where Newton never runs.

    A terminal frozen at m_0, lagged by a sweep, read one time step early, read at another node's density, or cached in
    Newton's residual or its warm-up each gives some call, or some published row, another density's u(T). The rows are
    built so that none passes unseen: the bound terminal moves over the solve, two nodes hold different densities, and
    each published row's u(T) is not the one at m_0."""
    calls = []
    published, expected = _run_loop(loop, calls, diverge_at)
    assert calls, f"{loop}: no HJB solve was recorded"
    for i, (problem, M, U_T) in enumerate(calls):
        own = np.asarray(problem.terminal_value(M[-1]), dtype=float).reshape(U_T.shape)
        np.testing.assert_array_equal(U_T, own, err_msg=f"{loop}: HJB call {i} received another density's u(T)")
    if len(calls) > 1:
        moved = max(np.max(np.abs(U_T - calls[0][2])) for _, _, U_T in calls)
        assert moved > 1e-3, f"{loop}: u(T) moved {moved:.3e} over the solve; the fixture cannot see a stale terminal"
    if published is not None:
        at_m0 = np.asarray(calls[-1][0].terminal_value(calls[-1][0].get_m_initial()), dtype=float).ravel()
        assert np.max(np.abs(expected.ravel() - at_m0)) > 1e-3, f"{loop}: the expected u(T) is the one at m_0"
        np.testing.assert_array_equal(
            published.ravel(), expected.ravel(), err_msg=f"{loop}: the published U(T) is not its own solve's u(T)"
        )


def test_what_stage_two_does_not_bind_is_refused():
    """A terminal coupling is refused in the multi-population iterator, which passes the stacked densities (#1642, the
    user's ruling 5), and on a network, where no coupled solve takes a problem (#2568); the network's message names its
    own route for u(T), ``terminal_node_value_func``. F and G alike are refused on FEM and meshless geometries, whose
    HJB solvers own their discretization (stage 1b)."""
    from scipy.sparse import csr_matrix

    from mfgarchon.geometry import Hyperrectangle
    from mfgarchon.geometry.graph.network_geometry import CustomNetwork
    from mfgarchon.geometry.meshes.mesh_1d import Mesh1D

    mesh = Mesh1D(bounds=(0.0, 1.0), num_elements=16)
    mesh.generate_mesh()
    mesh.boundary_conditions = no_flux_bc(dimension=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # the density's mass on these measures is not this pin's subject
        for field in ("coupling_cost", "terminal_coupling"):
            for domain in (Hyperrectangle(np.array([[0.0, 1.0]])), mesh):
                refusal = rf"Model\({field}=\.\.\.\) binds on a TensorProductGrid.*{type(domain).__name__}"
                with pytest.raises(NotImplementedError, match=refusal):
                    MFGProblem(
                        model=Model(hamiltonian=_hamiltonian(), volatility=0.3, **{field: PotentialCost(lambda x: x)}),
                        domain=domain,
                        conditions=Conditions(u_terminal=lambda x: 0.0, m_initial=lambda x: 1.0, T=0.5),
                        Nt=4,
                    )

    problems = [_terminal_coupled(), _terminal_coupled()]
    with pytest.raises(NotImplementedError, match=r"population\(s\) \[0, 1\] set Model\(terminal_coupling=\.\.\.\)"):
        MultiPopulationIterator(
            MultiPopulationProblem(populations=problems, population_names=["a", "b"]),
            [HJBFDMSolver(p) for p in problems],
            [FPFDMSolver(p) for p in problems],
        ).solve(max_iterations=2, tolerance=1e-6)

    pytest.importorskip("igraph")
    ring = np.zeros((6, 6))
    for i in range(6):
        ring[i, (i + 1) % 6] = ring[(i + 1) % 6, i] = 1.0
    with pytest.raises(NotImplementedError, match=r"terminal_coupling.*network.*terminal_node_value_func="):
        MFGProblem(
            model=Model(hamiltonian=_hamiltonian(), terminal_coupling=PotentialCost(lambda x: 0.0)),
            domain=CustomNetwork(csr_matrix(ring)),
            conditions=Conditions(u_terminal=lambda x: 0.0, m_initial=lambda x: 1.0 / 6.0, T=0.2),
            Nt=4,
        )
