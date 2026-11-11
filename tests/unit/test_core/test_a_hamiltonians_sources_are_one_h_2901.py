"""A Hamiltonian's sources are one H: a user's Hamiltonian whose sources disagree is refused at construction (#2901).

The library reads a Hamiltonian through several sources: `__call__` in the HJB residual, `dp` in its Jacobian,
`optimal_control` in the Fokker–Planck drift, the doors over them, and, for a separable H, its cost. Before #2901 a
problem checked `dp` and `dm` against `__call__` (`validate_hamiltonian_consistency`, #1642), and nothing else, so a
subclass whose `optimal_control` was another H's was solved as a mixture with no error. The user ruled (A) on
2026-11-11: refuse it at construction. The cost stands for H only as the library ships it (#2914's owner).

- **The control against -dp** (a mutation-verified pin). `Mix` and the issue's reproduction are refused at
  construction, and the message names the pair that disagrees and the probe (t, x, m, p) where it does.
- **The probes vary t and x** (a mutation-verified pin). A control that departs from -dp only after t = 0, and one that
  departs only away from the domain's middle, are refused: the probes run at t = 0, T/2 and T, and at the middle node and
  each axis's two end nodes.
- **A door against its method** (a mutation-verified pin). A subclass whose `evaluate_optimal_control` returns another
  control than its `optimal_control` is refused.
- **A network Hamiltonian's rates against its dp** (a mutation-verified pin). The network's own identity, at node
  probes, refuses a subclass whose rates are not dp's.
- **`canonical_cs` reads the cost only as shipped** (a mutation-verified pin). It refuses a subclass, and a
  `QuadraticControlCost` subclass, that change nothing, as FP-FDM does since #2915, and says what was run and on what.
  `BaseHJBSolver._control_cost_lambda`, which reads the cost's `lambda_`, refuses the same, naming the owner's reason.
- **The control every reader must accept** (a necessary oracle). Every library Hamiltonian is accepted.
"""

from __future__ import annotations

import logging
import re
import warnings

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.core import hamiltonian as hm
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.utils.validation import ValidationError, validate_custom_functions


def f(m):
    return 0.5 * m


class _OneHalfQuadratic(hm.SeparableHamiltonian):
    """`__call__` and `dp` are the lambda = 1/3 Hamiltonian, 1.5|p|^2 - f(m) and 3p, and they agree."""

    def __call__(self, t, x, p, m):
        return 1.5 * np.sum(np.asarray(p, dtype=float) ** 2, axis=-1) - f(np.asarray(m, dtype=float))

    def dp(self, t, x, p, m):
        return 3.0 * np.asarray(p, dtype=float)


class _Mix(_OneHalfQuadratic):
    """And its optimal control is the lambda = 1/2 one."""

    def optimal_control(self, t, x, p, m):
        return -2.0 * np.asarray(p, dtype=float)


class _TimeScaled(hm.SeparableHamiltonian):
    """The library's H and dp, and its control times 1 + 4t: they agree at t = 0 and not after."""

    def optimal_control(self, t, x, p, m):
        return (1.0 + 4.0 * t) * np.asarray(super().optimal_control(t=t, x=x, p=p, m=m), dtype=float)


class _AwayFromTheMiddle(hm.SeparableHamiltonian):
    """The library's H and dp, and its control times 1 + 3(x - 0.5)^2: they agree at x = 0.5 and not elsewhere."""

    def optimal_control(self, t, x, p, m):
        control = np.asarray(super().optimal_control(t=t, x=x, p=p, m=m), dtype=float)
        return (1.0 + 3.0 * (np.asarray(x[0], dtype=float) - 0.5) ** 2) * control


class _DoorDisagrees(hm.SeparableHamiltonian):
    """The library's quadratic Hamiltonian, whose control door returns twice its control."""

    def evaluate_optimal_control(self, state):
        return 2.0 * super().evaluate_optimal_control(state)


class _Plain(hm.SeparableHamiltonian):
    """A subclass that changes nothing."""


class _PlainCost(hm.QuadraticControlCost):
    """A cost subclass that changes nothing."""


class _QuadraticLagrangian(hm.LagrangianBase):
    """A user's Lagrangian, not a SeparableLagrangian, so the library forms its Hamiltonian numerically."""

    def __call__(self, t, x, alpha, m):
        return 0.5 * np.sum(np.asarray(alpha, dtype=float) ** 2, axis=-1)


def _quiet(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logging.disable(logging.CRITICAL)
        try:
            return fn()
        finally:
            logging.disable(logging.NOTSET)


def _grid(dim=1, n=11):
    return TensorProductGrid(
        bounds=[(0.0, 1.0)] * dim, Nx_points=[n] * dim, boundary_conditions=no_flux_bc(dimension=dim)
    )


def _problem(h, dim=1):
    return MFGProblem(
        model=Model(hamiltonian=h, volatility=0.3),
        domain=_grid(dim),
        conditions=Conditions(
            m_initial=lambda x: 0.2 + 0.0 * np.asarray(x[0], dtype=float),
            u_terminal=lambda x: 2.0 * (np.asarray(x[0], dtype=float) - 0.8) ** 2,
            T=0.5,
        ),
        Nt=5,
    )


def _quadratic(cost=1.0):
    return hm.QuadraticControlCost(control_cost=cost)


@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize(
    ("make", "control"),
    [
        (lambda: _Mix(_quadratic(), coupling=f), "0.8314"),
        (lambda: _OneHalfQuadratic(_quadratic(), coupling=f), "0.4157"),  # the issue's reproduction
    ],
    ids=["Mix", "the issue's reproduction"],
)
def test_a_control_that_is_not_minus_dp_is_refused_at_construction(make, control, dim):
    probe = "[0.5]" if dim == 1 else "[0.5 0.5]"
    momentum = "[0.4157]" if dim == 1 else "[0.4157 0.4157]"
    message = (
        rf"optimal_control disagrees with dp and with H at t=0, x={re.escape(probe)}, m=0\.5, p={re.escape(momentum)}: "
        rf"-optimal_control\[0\]={control}, dp\[0\]=1\.2471, and the central difference of H is 1\.2471"
    )
    with pytest.raises(ValidationError, match=message):
        _quiet(lambda: _problem(make(), dim))


@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize(
    ("make", "t", "x"),
    [(_TimeScaled, "0.25", lambda dim: [0.5] * dim), (_AwayFromTheMiddle, "0", lambda dim: [0.0] + [0.5] * (dim - 1))],
    ids=["after t = 0", "away from the middle"],
)
def test_a_control_that_departs_from_minus_dp_off_one_point_and_time_is_refused(make, t, x, dim):
    """T = 0.5, so T/2 = 0.25; along axis 0 the first end node is x = 0, the others at the middle."""
    probe = np.array2string(np.array(x(dim), dtype=float), precision=4)
    momentum = np.array2string(np.full(dim, 0.4157), precision=4)
    message = (
        rf"optimal_control disagrees with dp and with H at t={re.escape(t)}, x={re.escape(probe)}, m=0\.5, "
        rf"p={re.escape(momentum)}: "
    )
    with pytest.raises(ValidationError, match=message):
        _quiet(lambda: _problem(make(_quadratic(), coupling=f), dim))


def test_a_door_that_disagrees_with_its_method_is_refused_at_construction():
    with pytest.raises(
        ValidationError, match=r"evaluate_optimal_control and optimal_control disagree at t=0, x=\[0\.5\]"
    ):
        _quiet(lambda: _problem(_DoorDisagrees(_quadratic(), coupling=f)))


def test_a_network_hamiltonian_whose_rates_are_not_its_dp_is_refused():
    pytest.importorskip("igraph")
    from mfgarchon.extensions.topology import NetworkHamiltonian, NetworkMFGProblem
    from mfgarchon.geometry.graph.network_geometry import GridNetwork

    net = GridNetwork(width=3, height=3)
    net.create_network()
    problem = _quiet(lambda: NetworkMFGProblem(geometry=net, T=0.5, Nt=5))
    library = problem.hamiltonian_class

    class TwiceTheRates(NetworkHamiltonian):
        def optimal_control(self, t, x, p, m):
            return 2.0 * super().optimal_control(t=t, x=x, p=p, m=m)

    doubled = TwiceTheRates(network_data=library.network_data)
    accepted = _quiet(
        lambda: validate_custom_functions(
            library, library.dm, library.dp, problem.geometry, optimal_control=library.optimal_control
        )
    )
    refused = _quiet(
        lambda: validate_custom_functions(
            doubled, doubled.dm, doubled.dp, problem.geometry, optimal_control=doubled.optimal_control
        )
    )
    assert accepted.is_valid, [str(i) for i in accepted.issues]
    assert not refused.is_valid
    assert any("optimal_control and dp disagree on the network at t=0, node" in str(i) for i in refused.issues)


@pytest.mark.parametrize(
    ("make", "reason"),
    [
        (lambda: _Plain(_quadratic(), coupling=f), r"its class is _Plain"),
        (
            lambda: hm.SeparableHamiltonian(_PlainCost(control_cost=1.0), coupling=f),
            r"its control cost is a _PlainCost",
        ),
    ],
    ids=["a subclass that changes nothing", "a QuadraticControlCost subclass that changes nothing"],
)
def test_canonical_cs_reads_the_cost_only_as_the_library_ships_it(make, reason):
    from mfgarchon.alg.numerical.hjb_solvers import HJBSemiLagrangianSolver

    problem = _quiet(lambda: _problem(make()))
    message = (
        r"HJBSemiLagrangianSolver\(diffusion_method='canonical_cs'\), run on a 1-D problem: .* this problem's \w+ is not "
        rf"\({reason}"
    )
    with pytest.raises(NotImplementedError, match=message):
        HJBSemiLagrangianSolver(problem, diffusion_method="canonical_cs")


def test_the_control_cost_lambda_is_read_only_as_the_library_ships_it():
    from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver

    solver = HJBFDMSolver(_quiet(lambda: _problem(_Plain(_quadratic(), coupling=f))))
    with pytest.raises(
        NotImplementedError,
        match=r"^HJBFDMSolver reads the control cost's lambda_ .* this problem's _Plain is not \(its class is _Plain\)",
    ):
        solver._control_cost_lambda()


@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize(
    "make",
    [
        lambda: hm.SeparableHamiltonian(_quadratic(0.7), coupling=f),
        lambda: hm.SeparableHamiltonian(hm.L1ControlCost(control_cost=1.0), coupling=f),
        lambda: hm.SeparableHamiltonian(hm.BoundedControlCost(control_cost=1.0, max_control=2.0), coupling=f),
        lambda: hm.QuadraticMFGHamiltonian(),
        lambda: hm.CongestionHamiltonian(
            control_cost=_quadratic(),
            congestion_factor=lambda m: 1 + 0.8 * m,
            congestion_factor_dm=lambda m: 0.8 + 0 * m,
        ),
        lambda: hm.SeparableHamiltonian(hm.L1ControlCost(control_cost=1.0)).regularize(0.1),
        lambda: hm.SeparableLagrangian(control_cost=_quadratic()).as_hamiltonian(),
        lambda: _QuadraticLagrangian().legendre_transform(),
    ],
    ids=[
        "quadratic",
        "L1",
        "bounded",
        "QuadraticMFGHamiltonian",
        "congestion",
        "Moreau-Yosida",
        "as_hamiltonian",
        "Dual",
    ],
)
def test_every_library_hamiltonian_is_accepted(make, dim):
    """The control every reader must accept: the library's own Hamiltonians, whose sources agree."""
    assert _quiet(lambda: _problem(make(), dim)).hamiltonian_class is not None
