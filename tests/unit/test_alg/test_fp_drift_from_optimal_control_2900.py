"""A SeparableHamiltonian whose optimal_control is its own is driven by it, not by -c*grad(U) (#2900).

The FP drift of a smooth separable H was formed as -c*grad(U), its control cost's, by any solver handed the value
function, so a subclass that redefined H solved with the base class's drift on problem.solve()'s default scheme, with
no error. The oracle is the library SeparableHamiltonian for the same H: the subclass must solve as it does. Where the
FP solver forms -c*grad(U) and its velocity route is not measured equal to its value-function route, the solve refuses,
stating what was run and on what, and naming no other scheme (user ruling, 2026-11-06). The last rows run the CONVENTIONS
sentences on the solvers that read optimal_control themselves.
"""

from __future__ import annotations

import functools
import importlib
import logging
import pkgutil
import sys
import warnings

import pytest

import numpy as np

import mfgarchon.alg.numerical
from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.coupling.block_iterators import BlockIterator
from mfgarchon.alg.numerical.coupling.fictitious_play import FictitiousPlayIterator
from mfgarchon.alg.numerical.coupling.fixed_point_utils import fp_solver_sig_params, resolve_fp_drift_kwargs
from mfgarchon.alg.numerical.coupling.graph_coupling import AdjacencyCoupling
from mfgarchon.alg.numerical.coupling.graph_mfg_solver import GraphMFGSolver
from mfgarchon.alg.numerical.coupling.mfg_residual import MFGResidual
from mfgarchon.alg.numerical.coupling.multi_population_iterator import MultiPopulationIterator
from mfgarchon.alg.numerical.coupling.regime_switching_iterator import RegimeSwitchingIterator
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver, FPParticleSolver
from mfgarchon.alg.numerical.fp_solvers.base_fp import BaseFPSolver
from mfgarchon.alg.numerical.fp_solvers.fp_fvm import FPFVMSolver
from mfgarchon.alg.numerical.fp_solvers.fp_semi_lagrangian_adjoint import FPSLSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.alg.numerical.meshless_galerkin.fp_solver import MeshlessGalerkinFPSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, QuadraticMFGHamiltonian, SeparableHamiltonian
from mfgarchon.core.multi_population import MultiPopulationProblem
from mfgarchon.core.regime_switching import RegimeSwitchingConfig
from mfgarchon.factory.scheme_factory import create_paired_solvers
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, dirichlet_bc, no_flux_bc
from mfgarchon.geometry.boundary.conditions import periodic_bc
from mfgarchon.types import NumericalScheme
from mfgarchon.utils import pde_coefficients
from mfgarchon.utils.pde_coefficients import (
    WHERE_OWN_CONTROLS_ARE_READ,
    drift_is_the_control_costs,
    fp_drift_coefficient,
)


def _coupling(m):
    return 0.5 * m


def _steep_control(self, t, x, p, m):
    return -3.0 * np.asarray(p, dtype=float)


class _Steep(SeparableHamiltonian):
    """H = 1.5|p|^2 - f(m) on a control_cost=1 base, whose control cost's drift is -grad(U), not -3 grad(U). The base
    reads the coupling, so the density it is handed is read as the library reads it."""

    def __call__(self, t, x, p, m):
        p_arr = np.atleast_1d(np.asarray(p, dtype=float))
        extra = np.sum(p_arr**2, axis=-1)
        return super().__call__(t, x, p, m) + (extra if p_arr.ndim == 2 else float(extra))

    def dp(self, t, x, p, m):
        return 3.0 * np.asarray(p, dtype=float)


def _library(**kwargs):
    return SeparableHamiltonian(QuadraticControlCost(control_cost=1 / 3), coupling=_coupling, **kwargs)


_LIBRARY = _library()


class _OwnControl(_Steep):
    optimal_control = _steep_control


class _InheritsOwnControl(_OwnControl):
    pass


class _WrappedOwnControl(_Steep):
    optimal_control = functools.wraps(_steep_control)(
        lambda self, *args, **kwargs: _steep_control(self, *args, **kwargs)
    )


class _OwnStatic(_Steep):
    optimal_control = staticmethod(lambda t, x, p, m: _LIBRARY.optimal_control(t, x, p, m))


class _OwnClassmethod(_Steep):
    @classmethod
    def optimal_control(cls, t, x, p, m):
        return _LIBRARY.optimal_control(t, x, p, m)


class _OwnProperty(_Steep):
    @property
    def optimal_control(self):
        return _LIBRARY.optimal_control


class _OwnGetattribute(_Steep):
    """__getattribute__ runs on every lookup, so the method served is the library H's."""

    def __getattribute__(self, name):
        if name == "optimal_control":
            return _LIBRARY.optimal_control
        return object.__getattribute__(self, name)


class _OwnGetattr(_Steep):
    """__getattr__ runs only when the normal lookup fails, and SeparableHamiltonian.optimal_control is found first."""

    def __getattr__(self, name):
        if name == "optimal_control":
            return _LIBRARY.optimal_control
        raise AttributeError(name)


class _OwnMixin:
    def optimal_control(self, t, x, p, m):
        return _LIBRARY.optimal_control(t, x, p, m)


class _OwnByMixin(_OwnMixin, _Steep):
    pass


_QUADRATIC_MFG_3 = QuadraticMFGHamiltonian(coupling_coefficient=3.0)


class _OwnQuadraticMFG(QuadraticMFGHamiltonian):
    """A QuadraticMFGHamiltonian(c=1) whose H, dp and optimal_control are QuadraticMFGHamiltonian(c=3)'s."""

    def __init__(self):
        super().__init__(coupling_coefficient=1.0)

    def __call__(self, t, x, p, m):
        return _QUADRATIC_MFG_3(t, x, p, m)

    def dp(self, t, x, p, m):
        return _QUADRATIC_MFG_3.dp(t, x, p, m)

    def optimal_control(self, t, x, p, m):
        return _QUADRATIC_MFG_3.optimal_control(t, x, p, m)


class _SteepCost(QuadraticControlCost):
    """A QuadraticControlCost(control_cost=1) whose evaluate, dp and optimal_control are control_cost=1/3's."""

    _target = QuadraticControlCost(control_cost=1 / 3)

    def optimal_control(self, p):
        return self._target.optimal_control(p)

    def evaluate(self, p):
        return self._target.evaluate(p)

    def dp(self, p):
        return self._target.dp(p)


def _steep(cls=_Steep, **kwargs):
    return cls(QuadraticControlCost(control_cost=1.0), coupling=_coupling, **kwargs)


def _assigned_on_the_instance():
    hamiltonian = _steep()
    hamiltonian.optimal_control = functools.partial(_steep_control, hamiltonian)
    return hamiltonian


def _library_method_assigned_on_the_instance():
    hamiltonian = _steep()
    hamiltonian.optimal_control = _LIBRARY.optimal_control
    return hamiltonian


#: name -> (the Hamiltonian, its reference: the library H for the same function, whether -c*grad(U) is its drift).
CASES = {
    "on_the_class": (lambda: _steep(_OwnControl), _library, False),
    "inherited_from_a_subclass": (lambda: _steep(_InheritsOwnControl), _library, False),
    "wrapped_with_functools_wraps": (lambda: _steep(_WrappedOwnControl), _library, False),
    "assigned_on_the_instance": (_assigned_on_the_instance, _library, False),
    "library_method_assigned_on_the_instance": (_library_method_assigned_on_the_instance, _library, False),
    "staticmethod": (lambda: _steep(_OwnStatic), _library, False),
    "classmethod": (lambda: _steep(_OwnClassmethod), _library, False),
    "property": (lambda: _steep(_OwnProperty), _library, False),
    "__getattribute__": (lambda: _steep(_OwnGetattribute), _library, False),
    "mixin_before_SeparableHamiltonian": (lambda: _steep(_OwnByMixin), _library, False),
    "QuadraticMFGHamiltonian_subclass": (
        _OwnQuadraticMFG,
        lambda: QuadraticMFGHamiltonian(coupling_coefficient=3.0),
        False,
    ),
    "QuadraticControlCost_subclass": (
        lambda: SeparableHamiltonian(_SteepCost(control_cost=1.0), coupling=_coupling),
        _library,
        False,
    ),
    "library_SeparableHamiltonian": (_library, _library, True),
    "library_QuadraticMFGHamiltonian": (
        lambda: QuadraticMFGHamiltonian(coupling_coefficient=3.0),
        lambda: QuadraticMFGHamiltonian(coupling_coefficient=3.0),
        True,
    ),
}

#: Hamiltonians that serve SeparableHamiltonian.optimal_control with a QuadraticControlCost's, but are not as the
#: library ships them, so the predicate does not hold. That their __call__ and dp are not that control's is #2901, out
#: of this fix's scope.
SERVES_THE_BASE_CONTROL = {"__getattr__": lambda: _steep(_OwnGetattr), "H_redefined_control_inherited": _steep}

NODES = np.linspace(0.0, 1.0, 21)


def _faces(left, right):
    """A 1-D BC written as one segment per face: (type, value) at x_min, then at x_max."""
    return BoundaryConditions(
        dimension=1,
        segments=[
            BCSegment(name=face, bc_type=kind, boundary=face, **({} if value is None else {"value": value}))
            for face, (kind, value) in (("x_min", left), ("x_max", right))
        ],
    )


WALLS = {
    "no_flux": no_flux_bc,
    "periodic": periodic_bc,
    "dirichlet": functools.partial(dirichlet_bc, value=0.0),
    "no_flux_per_face": lambda dimension: _faces((BCType.NO_FLUX, None), (BCType.NO_FLUX, None)),
    "no_flux_and_neumann_0": lambda dimension: _faces((BCType.NO_FLUX, None), (BCType.NEUMANN, 0.0)),
}


def _problem(hamiltonian, *, d=1, walls="no_flux", centre=0.3, volatility=0.3, n=None):
    n = n or (21 if d == 1 else 11)
    nodes = np.linspace(0.0, 1.0, n)
    mass = np.trapezoid(0.2 + np.exp(-40 * (nodes - centre) ** 2), nodes)  # the grid's measure, so the 1-D mass is 1
    grid = TensorProductGrid(bounds=[(0.0, 1.0)] * d, Nx_points=[n] * d, boundary_conditions=WALLS[walls](dimension=d))
    return MFGProblem(
        model=Model(hamiltonian=hamiltonian, volatility=volatility),
        domain=grid,
        conditions=Conditions(
            m_initial=lambda x: (0.2 + np.exp(-40 * (x[0] - centre) ** 2)) / mass,
            u_terminal=lambda x: 2.0 * sum((x[i] - 0.8) ** 2 for i in range(d)),
            T=0.5,
        ),
        Nt=10,
    )


def _quietly(run):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logging.disable(logging.CRITICAL)
        try:
            return run()
        finally:
            logging.disable(logging.NOTSET)


def _solve(hamiltonian, *, d=1, walls="no_flux", volatility=0.3, n=None, **solver):
    problem = _problem(hamiltonian, d=d, walls=walls, volatility=volatility, n=n)
    if "fp_solver" in solver:
        solver = {"hjb_solver": HJBFDMSolver(problem), "fp_solver": solver["fp_solver"](problem)}
    result = _quietly(lambda: problem.solve(max_iterations=6, tolerance=1e-12, verbose=False, **solver))
    return np.asarray(result.U), np.asarray(result.M)


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_predicate_reads_the_optimal_control_the_hamiltonian_serves(case):
    """drift_is_the_control_costs holds only for a Hamiltonian and its cost exactly as the library ships them: the class
    SeparableHamiltonian or QuadraticMFGHamiltonian, a QuadraticControlCost, and no family method on either instance
    (#2914)."""
    make, _, expected = CASES[case]
    assert drift_is_the_control_costs(make()) is expected


@pytest.mark.parametrize("case", sorted(SERVES_THE_BASE_CONTROL))
def test_a_hamiltonian_not_as_shipped_is_not_certified_whatever_it_serves(case):
    """These serve the base class's optimal_control, but are not as the library ships them, so the type rule does not
    certify -c*grad(U) as their drift and the velocity route reads what they serve (#2914, #2915 review 1). That
    their __call__ and dp say otherwise is #2901."""
    assert drift_is_the_control_costs(SERVES_THE_BASE_CONTROL[case]()) is False


@pytest.mark.parametrize("case", sorted(name for name, (_, _, holds) in CASES.items() if not holds))
def test_an_own_optimal_control_solves_as_the_library_hamiltonian_it_describes(case):
    """On the default scheme, 1-D with no-flux walls, each solves as the library H for the same function. At 6f02506c,
    where the default scheme drove each by -c*grad(U), the on_the_class row was 1.2074e+00 off in M on this fixture."""
    make, reference, _ = CASES[case]
    U_ref, M_ref = _solve(reference())
    U, M = _solve(make())
    np.testing.assert_allclose(U, U_ref, rtol=0, atol=1e-12)
    np.testing.assert_allclose(M, M_ref, rtol=0, atol=1e-12)


def _refused_saying(refused, *, site, ran_on, why=""):
    """A refusal of -c*grad(U) states what was run and on what, then why, and points to CONVENTIONS; it names no other
    scheme or operator (user ruling, 2026-11-06)."""
    said = str(refused.value)
    assert said.startswith(f"{site}, run on {ran_on}: "), said
    assert why in said, said
    assert said.endswith(WHERE_OWN_CONTROLS_ARE_READ), said
    for other in (
        "FVM_UPWIND",
        "FPFVMSolver",
        "FPParticleSolver",
        "build_linearized_operator",
        "streamline_diffusion_scale",
    ):
        assert other not in said, (other, said)


ROUTER = "FPFDMSolver.solve_fp_system(potential_field=U)"

#: The cells #2908 and #2915 widen the velocity route to: periodic and Dirichlet walls, 2-D and 3-D grids (#2908,
#: measured by #2909's census in #2904's test file), and FLUX walls handed to FP-FDM and 4-D grids (#2915). An own
#: control there solves as the library H it describes. Outside the comparison: FLUX walls in 2-D with a flux of 0.2,
#: where FP-FDM refuses the library H itself, since the flux's part of a step takes the density negative.
WIDENED = {
    "1-D periodic": {"walls": "periodic"},
    "2-D no-flux": {"d": 2, "n": 7},
    "1-D Dirichlet": {"walls": "dirichlet"},
    "3-D Dirichlet": {"d": 3, "n": 3, "walls": "dirichlet"},  # pins the router admitting 3-D; #2904's census, the route
    "1-D FLUX 0 on both faces": {
        "fp_solver": functools.partial(FPFDMSolver, boundary_conditions=_faces((BCType.FLUX, 0.0), (BCType.FLUX, 0.0)))
    },
    "1-D FLUX 0.2 on x_max": {
        "fp_solver": functools.partial(
            FPFDMSolver, boundary_conditions=_faces((BCType.NO_FLUX, 0.0), (BCType.FLUX, 0.2))
        )
    },
    "4-D no-flux, 3 nodes per axis": {"d": 4, "n": 3},
    "4-D no-flux, 4 nodes per axis": {"d": 4, "n": 4},  # about 110 s, so it runs outside the gate (slow)
}


@pytest.mark.parametrize(
    "where",
    [
        pytest.param(w, marks=pytest.mark.slow) if w.startswith("4-D") and WIDENED[w]["n"] > 3 else w
        for w in sorted(WIDENED)
    ],
)
def test_on_the_widened_set_an_own_control_solves_as_the_library_hamiltonian_it_describes(where):
    """#2908: the default scheme takes the own control as a velocity on these cells too, where it used to refuse (at
    87b145b1) or drive it by -c*grad(U) (at 6f02506c: 1.1121e+00 periodic, 2.5675e+00 in 2-D)."""
    _, M_ref = _solve(_library(), **WIDENED[where])
    _, M = _solve(_steep(_OwnControl), **WIDENED[where])
    np.testing.assert_allclose(M, M_ref, rtol=0, atol=1e-10)


def test_the_set_reads_the_walls_the_solver_was_handed():
    """FP-FDM handed walls other than the problem's solves an own control on the walls it was handed, as the library
    H does there (review 2 of #2902, W7 and its reverse; both kinds are in the set since #2908)."""
    for problem_walls, handed in (
        ("no_flux", functools.partial(FPFDMSolver, boundary_conditions=dirichlet_bc(value=0.0, dimension=1))),
        ("dirichlet", functools.partial(FPFDMSolver, boundary_conditions=no_flux_bc(dimension=1))),
    ):
        _, M_ref = _solve(_library(), walls=problem_walls, fp_solver=handed)
        _, M = _solve(_steep(_OwnControl), walls=problem_walls, fp_solver=handed)
        np.testing.assert_allclose(M, M_ref, rtol=0, atol=1e-12)


#: The sites that form -c*grad(U) when a solver is called directly, and the context each refusal names.
_U = np.tile(0.3 * np.sin(2 * np.pi * NODES) + 0.2 * NODES, (11, 1))
DIRECT_SITES = {
    "FPFDMSolver, from potential_field=U": lambda problem: FPFDMSolver(problem).solve_fp_system(
        problem.get_m_initial(), potential_field=_U, show_progress=False
    ),
    "FPFDMSolver.build_advection_operator": lambda problem: FPFDMSolver(problem).build_advection_operator(_U[0]),
    "FPSLSolver": lambda problem: FPSLSolver(problem).solve_fp_system(
        problem.get_m_initial(), potential_field=_U, show_progress=False
    ),
    "HJBFDMSolver.build_advection_matrix": lambda problem: HJBFDMSolver(problem).build_advection_matrix(_U[0]),
}


@pytest.mark.parametrize("site", sorted(DIRECT_SITES))
def test_a_site_called_directly_refuses_stating_what_it_ran(site):
    """Called directly, each site that forms -c*grad(U) refuses an own optimal control, naming itself and the problem's
    dimension, and no other scheme or operator."""
    with pytest.raises(NotImplementedError, match="#2900") as refused:
        _quietly(lambda: DIRECT_SITES[site](_problem(_steep(_OwnControl))))
    _refused_saying(refused, site=site, ran_on="a 1-D problem")


@pytest.mark.parametrize("shape", [(10, 21), (12, 21), (1, 21), (11, 20)])
def test_a_density_that_is_not_the_trajectory_is_refused_by_name(shape):
    """The velocity route reads M at every time level and grid node, so the set takes M of the trajectory's shape,
    (Nt + 1, Nx) = (11, 21), only. At 775f3b30 these routed, padded or cut without a word by compute_fp_velocity_field
    (review 2 of #2902)."""
    problem = _problem(_steep(_OwnControl))
    fp = FPFDMSolver(problem)
    U = np.zeros((11, 21))

    def route(M):
        return resolve_fp_drift_kwargs(
            problem, fp_solver_sig_params(fp), None, U, M, drift_convention=fp._drift_convention, fp_solver=fp
        )

    assert "drift_field" in _quietly(lambda: route(np.ones((11, 21))))[0]
    with pytest.raises(NotImplementedError, match=rf"M of shape \({shape[0]}, {shape[1]}\), not a density trajectory"):
        route(np.ones(shape))


def test_an_advection_scheme_that_reads_no_velocity_is_refused():
    """FDM_CENTERED's FP-FDM ignores a velocity, so its #1632 guard refuses the one the router sends, at the first FP
    solve, before it is dropped: 5.9844e-01 off in M at 6f02506c. At sigma = 0.8 the centred scheme solves the library
    H on this fixture."""
    with pytest.raises(NotImplementedError, match="#2900") as refused:
        _solve(_steep(_OwnControl), volatility=0.8, scheme=NumericalScheme.FDM_CENTERED)
    assert "advection_scheme='divergence_centered' does not read 'interface_velocity'" in str(refused.value)
    assert "Issue #1632" in str(refused.value)
    _solve(_library(), volatility=0.8, scheme=NumericalScheme.FDM_CENTERED)


def test_an_iterators_volatility_override_decides_what_the_solve_reads():
    """A FixedPointIterator handed volatility=0.3 on a problem whose own volatility is a tensor solves its FP with the
    scalar it was handed, whose step reads a velocity, so a plain subclass solves as the library H. At c425dae4 the
    router asked FP-FDM about the problem's tensor instead, and refused it (#2915 review 3, B1)."""
    from mfgarchon.alg.numerical.coupling.fixed_point_iterator import FixedPointIterator

    plain = type("Plain", (SeparableHamiltonian,), {})

    def solve(cls):
        problem = MFGProblem(
            model=Model(
                hamiltonian=cls(QuadraticControlCost(control_cost=1 / 3), coupling=_coupling),
                volatility=0.3 * np.eye(2),
                volatility_kind="tensor",
            ),
            domain=TensorProductGrid(
                bounds=[(0.0, 1.0)] * 2, Nx_points=[7, 7], boundary_conditions=no_flux_bc(dimension=2)
            ),
            conditions=Conditions(
                m_initial=lambda x: 1.0 + 0.3 * np.cos(np.pi * x[0]),
                u_terminal=lambda x: 0.2 * (x[0] - 0.8) ** 2 + 0.1 * x[1],
                T=0.2,
            ),
            Nt=40,
        )
        iterator = FixedPointIterator(problem, HJBFDMSolver(problem), FPFDMSolver(problem), volatility=0.3)
        return np.asarray(_quietly(lambda: iterator.solve(max_iterations=4, tolerance=1e-12, verbose=False)).M)

    np.testing.assert_allclose(solve(plain), solve(SeparableHamiltonian), rtol=0, atol=1e-12)


def test_a_solver_that_takes_the_value_function_only_is_refused():
    """FP-SL forms the drift as -c*grad(U) and takes no velocity, so it refuses rather than solve with the control
    cost's drift (1.1519e+00 off in M at 6f02506c on this fixture, SL_LINEAR)."""
    with pytest.raises(NotImplementedError, match="#2900") as refused:
        _solve(_steep(_OwnControl), scheme=NumericalScheme.SL_LINEAR)
    _refused_saying(
        refused,
        site="FPSLSolver.solve_fp_system(potential_field=U)",
        ran_on="a 1-D grid, one population, M of shape (11, 21)",
        why="FPSLSolver takes the value function only.",
    )


def _plain_with_its_control_assigned():
    hamiltonian = SeparableHamiltonian(QuadraticControlCost(control_cost=1.0), coupling=_coupling)
    hamiltonian.optimal_control = functools.partial(_steep_control, hamiltonian)
    return hamiltonian


@pytest.mark.parametrize(
    ("make", "reason"),
    [
        (
            _plain_with_its_control_assigned,
            "SeparableHamiltonian is not as the library ships it (its instance holds ['optimal_control'] of its own)",
        ),
        (
            lambda: SeparableHamiltonian(_SteepCost(control_cost=1.0), coupling=_coupling),
            "SeparableHamiltonian is not as the library ships it (its control cost is a _SteepCost, not a "
            "QuadraticControlCost)",
        ),
        (lambda: _steep(_OwnControl), "_OwnControl is not as the library ships it (its class is _OwnControl)"),
    ],
)
def test_the_refusal_names_why_the_hamiltonian_is_not_as_shipped(make, reason):
    """The refusal names the owner's reason, the instance, the cost's type or the class (#2914), and only it: the
    second decider that picked a side by the type attribute is gone (#2915 review 1, B2). Pinned on the whole clause."""
    with pytest.raises(NotImplementedError, match="#2900") as refused:
        fp_drift_coefficient(_problem(make()), context="here")
    assert str(refused.value).startswith(
        f"here, run on a 1-D problem: {reason}, so -c*grad(U), the quadratic control cost's drift, is not known to be "
        "its drift, and here forms the drift from that scalar (#2900, #2914)."
    )


def _multi_population(fp_solver, centres=(0.3, 0.7)):
    def run(kind):
        hamiltonian = _library if kind == "library" else functools.partial(_steep, _OwnControl)
        problems = [_problem(hamiltonian(population_index=k), centre=c) for k, c in enumerate(centres)]
        iterator = MultiPopulationIterator(
            MultiPopulationProblem(populations=problems, population_names=[f"p{k}" for k in range(len(centres))]),
            [HJBFDMSolver(problem) for problem in problems],
            [fp_solver(problem) for problem in problems],
        )
        return np.asarray(_quietly(lambda: iterator.solve(max_iterations=6, tolerance=1e-12)).M)

    return run


@pytest.mark.parametrize("centres", [(0.3,), (0.3, 0.7)], ids=["K=1", "K=2"])
def test_a_multi_population_solve_routes_each_population_at_its_own_density(centres):
    """MultiPopulationIterator hands FP-FDM each population's value function, and its velocity door reads each population
    at its own density (#2335), so on the measured set the own control routes there too, at K = 1 as at K = 2, and
    solves as the library H. At f59bb154, before #2335's door, both were refused."""
    run = _multi_population(FPFDMSolver, centres)
    np.testing.assert_allclose(run("own"), run("library"), rtol=0, atol=1e-12)


#: The router's callers not reached by problem.solve(), each on the measured set, own against library.
def _block(hamiltonian):
    problem = _problem(hamiltonian)
    iterator = BlockIterator(problem, HJBFDMSolver(problem), FPFDMSolver(problem))
    return np.asarray(_quietly(lambda: iterator.solve(max_iterations=6, tolerance=1e-12)).M)


def _fictitious_play(hamiltonian):
    problem = _problem(hamiltonian)
    iterator = FictitiousPlayIterator(problem, HJBFDMSolver(problem), FPFDMSolver(problem))
    return np.asarray(_quietly(lambda: iterator.solve(max_iterations=6, tolerance=1e-12)).M)


def _residual(hamiltonian):
    U, M = _solve(_library())
    problem = _problem(hamiltonian)
    return np.asarray(
        _quietly(lambda: MFGResidual(problem, HJBFDMSolver(problem), FPFDMSolver(problem)).compute_fp_output(U, M))
    )


@pytest.mark.parametrize(
    "caller", [_block, _fictitious_play, _residual], ids=["BlockIterator", "FictitiousPlayIterator", "MFGResidual"]
)
def test_each_router_caller_hands_the_solver_so_the_own_control_is_read(caller):
    """Each caller passes its FP solver to the router, which reads the class's attribute from it; without it the own
    control is refused there rather than routed."""
    np.testing.assert_allclose(caller(_steep(_OwnControl)), caller(_library()), rtol=0, atol=1e-12)


def test_regime_switching_refuses_without_a_density_trajectory():
    """RegimeSwitchingIterator hands the router a single density on its first sweep, not a trajectory (#2905), so FP-FDM
    refuses rather than reach the velocity route, which reads the trajectory."""
    problems = [_problem(_steep(_OwnControl), centre=c) for c in (0.3, 0.7)]
    iterator = RegimeSwitchingIterator(
        problems,
        RegimeSwitchingConfig(transition_matrix=np.array([[-0.5, 0.5], [0.5, -0.5]])),
        [HJBFDMSolver(problem) for problem in problems],
        [FPFDMSolver(problem) for problem in problems],
        max_iterations=6,
        tolerance=1e-12,
    )
    with pytest.raises(NotImplementedError, match=r"Here it is handed M of shape \(21,\), not a density trajectory"):
        _quietly(iterator.solve)


def test_a_graph_solve_takes_the_velocity_route_on_each_node():
    """GraphMFGSolver hands each node its own density trajectory and no cross density, so each node is one population
    on the measured set: the own optimal control is read, as the library H's control is (1.202e+00 off in M at
    6f02506c)."""

    def run(kind):
        problems = [_problem(_library() if kind == "library" else _steep(_OwnControl), centre=c) for c in (0.3, 0.7)]
        solver = GraphMFGSolver(
            problems,
            AdjacencyCoupling(np.ones((2, 2)) - np.eye(2), alpha=0.05, beta=0.02),
            [HJBFDMSolver(problem) for problem in problems],
            [FPFDMSolver(problem) for problem in problems],
            max_iterations=6,
            tolerance=1e-12,
        )
        return np.asarray([np.asarray(m) for m in _quietly(solver.solve).densities])

    np.testing.assert_allclose(run("own"), run("library"), rtol=0, atol=1e-12)


def test_particle_still_reads_an_own_optimal_control():
    """The particle solver reads optimal_control itself, so it keeps solving such a subclass: the scalar it builds is for a
    non-separable H only."""
    particles = functools.partial(FPParticleSolver, num_particles=4000, seed=7)
    _, M_ref = _solve(_library(), fp_solver=particles)
    _, M = _solve(_steep(_OwnControl), fp_solver=particles)
    np.testing.assert_allclose(M, M_ref, rtol=0, atol=1e-12)


def _meshless(hamiltonian, streamline_diffusion_scale, walls="no_flux"):
    problem = _problem(hamiltonian, walls=walls)
    config = {
        "collocation_points": NODES.reshape(-1, 1),
        "delta": 2.6 / np.sqrt(21),
        "degree": 2,
        "use_newton": True,
        "streamline_diffusion_scale": streamline_diffusion_scale,
    }
    hjb, fp = create_paired_solvers(problem, NumericalScheme.MESHLESS_GALERKIN, hjb_config=config)
    result = _quietly(
        lambda: problem.solve(max_iterations=6, tolerance=1e-12, verbose=False, hjb_solver=hjb, fp_solver=fp)
    )
    return np.asarray(result.U), np.asarray(result.M)


@pytest.mark.parametrize("walls", ["no_flux", "dirichlet"])
def test_the_meshless_stabilization_refuses_stating_what_it_ran(walls):
    """The meshless HJB's streamline diffusion forms its velocity from the scalar, so it refuses, naming no other scheme.
    streamline_diffusion_scale=0 is another, unstabilised discretisation, which on this fixture's Dirichlet walls the
    library H itself diverges under: max|M| 4.0577e+03 against 2.5047e+00 at 1 (review 2 of #2902)."""
    with pytest.raises(NotImplementedError, match="#2900") as refused:
        _meshless(_steep(_OwnControl), 1.0, walls)
    _refused_saying(refused, site="MeshlessGalerkinHJBSolver streamline diffusion", ran_on="a 1-D problem")


def test_without_the_stabilization_the_meshless_pair_forms_no_scalar():
    """At streamline_diffusion_scale=0 the meshless pair forms no drift from the scalar, so it runs the subclass, and on
    no-flux walls as the library H: 5.6634e-11 off in M, the pair's own convergence, not a drift difference."""
    U_ref, M_ref = _meshless(_library(), 0.0)
    U, M = _meshless(_steep(_OwnControl), 0.0)
    np.testing.assert_allclose(M, M_ref, rtol=0, atol=1e-9)
    np.testing.assert_allclose(U, U_ref, rtol=0, atol=1e-9)


# ---- the CONVENTIONS sentences on the solvers that read optimal_control themselves, each run once ------------------


def test_conventions_fvm_reads_the_own_control_at_its_faces():
    """FVM reads optimal_control at its faces, from U: FP alone on a periodic grid, across a U that jumps at the wrap
    face, U(1) - U(0) = 0.2, the own control solves as the library H."""
    nodes = np.linspace(0.0, 1.0, 21)
    U = np.stack([0.3 * np.sin(2 * np.pi * nodes) + 0.2 * nodes] * 11)
    m0 = 1.0 + 0.5 * np.cos(2 * np.pi * nodes)

    def fp_alone(hamiltonian):
        fp = FPFVMSolver(_problem(hamiltonian, walls="periodic"))
        return np.asarray(_quietly(lambda: fp.solve_fp_system(m0, potential_field=U, show_progress=False)))

    np.testing.assert_allclose(fp_alone(_steep(_OwnControl)), fp_alone(_library()), rtol=0, atol=1e-12)


class _ServesItsOwn(SeparableHamiltonian):
    """Serves an optimal_control of its own, which is the base's formula: H and dp stay the cost's."""

    def optimal_control(self, t, x, p, m):
        return SeparableHamiltonian.optimal_control(self, t, x, p, m)


def _moreau_yosida_own():
    from mfgarchon.core.hamiltonian import L1ControlCost

    return _ServesItsOwn(L1ControlCost(lambda_=0.5).regularize(0.2), coupling=_coupling)


#: What FVM refuses, as CONVENTIONS states it, each run on the own control: the problem, and what FVM says.
FVM_REFUSES = {
    "walls not one uniform condition": (
        lambda: _problem(_steep(_OwnControl), walls="no_flux_per_face"),
        "FP FVM solver supports only uniform boundary conditions",
    ),
    "Dirichlet walls": (
        lambda: _problem(_steep(_OwnControl), walls="dirichlet"),
        r"FP FVM \(v1\) does not support Dirichlet BC",
    ),
    "a volatility that is not one constant": (
        lambda: MFGProblem(
            model=Model(hamiltonian=_steep(_OwnControl), volatility=lambda t, x, m: 0.2 + 0.1 * np.asarray(x[0])),
            domain=_problem(_library()).geometry,
            conditions=Conditions(m_initial=lambda x: 1.0 + 0.0 * x[0], u_terminal=lambda x: 0.0 * x[0], T=0.5),
            Nt=10,
        ),
        "FP FVM solver supports only scalar/None volatility",
    ),
    "a control cost that is not quadratic": (
        lambda: _problem(_moreau_yosida_own()),
        r"FP FVM potential_field: the scalar FP drift .* is defined only for a quadratic SeparableHamiltonian",
    ),
}


@pytest.mark.parametrize("what", sorted(FVM_REFUSES))
def test_conventions_what_fvm_refuses(what):
    """CONVENTIONS: FVM refuses walls that are not one uniform condition, Dirichlet walls, a volatility that is not one
    constant, and, on its value-function route, a control cost that is not quadratic (#1542)."""
    make, says = FVM_REFUSES[what]
    problem = make()
    with pytest.raises(NotImplementedError, match=says):
        _quietly(
            lambda: problem.solve(max_iterations=2, tolerance=1e-12, verbose=False, scheme=NumericalScheme.FVM_UPWIND)
        )


def test_conventions_the_particle_solver_refuses_a_control_cost_that_is_not_quadratic():
    """CONVENTIONS: the particle solver refuses a control cost that is not quadratic (#1542)."""
    particles = functools.partial(FPParticleSolver, num_particles=500, seed=7)
    with pytest.raises(
        NotImplementedError, match=r"FPParticleSolver: the scalar FP drift .* defined only for a quadratic"
    ):
        _solve(_moreau_yosida_own(), fp_solver=particles)


# ---- the attribute the router reads, against what each FP solver does --------------------------------------------------


def _on_the_grid(fp_solver):
    def run():
        problem = _problem(_library())
        U = np.tile(0.5 * (NODES - 0.3) ** 2, (11, 1))
        fp_solver(problem).solve_fp_system(problem.get_m_initial(), potential_field=U, show_progress=False)

    return run


def _meshless_fp_alone():
    problem = _problem(_library())
    fp = MeshlessGalerkinFPSolver(problem, collocation_points=NODES[:, None], delta=2.6 / np.sqrt(21))
    fp.solve_fp_system(np.ones(21), potential_field=np.tile(0.5 * (NODES - 0.3) ** 2, (11, 1)))


def _fem_fp_alone():
    skfem = pytest.importorskip("skfem")
    from mfgarchon.alg.numerical.fem.fp_fem_solver import FPFEMSolver
    from mfgarchon.alg.numerical.fem.mesh_adapter import skfem_to_meshdata
    from mfgarchon.core.mfg_components import MFGComponents
    from mfgarchon.geometry.meshes.mesh_2d import Mesh2D

    mesh = skfem.MeshTri.init_sqsymmetric().refined(2)
    geometry = Mesh2D(domain_type="rectangle", bounds=(0.0, 1.0, 0.0, 1.0))
    geometry.mesh_data = skfem_to_meshdata(mesh)
    geometry.boundary_conditions = no_flux_bc(dimension=2)
    problem = MFGProblem(
        geometry=geometry,
        T=0.2,
        Nt=5,
        volatility=0.3,
        components=MFGComponents(m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0, hamiltonian=_library()),
    )
    U = np.tile(0.6 * mesh.p[0] ** 2 + 0.3 * mesh.p[1], (problem.Nt + 1, 1))
    FPFEMSolver(problem).solve_fp_system(np.ones(mesh.p.shape[1]), U)


#: Each FP solver, by name, run on its value-function route with the library H.
RUN_ON_THE_VALUE_FUNCTION = {
    "FPFDMSolver": _on_the_grid(FPFDMSolver),
    "FPSLSolver": _on_the_grid(FPSLSolver),
    "FPFVMSolver": _on_the_grid(FPFVMSolver),
    "FPParticleSolver": _on_the_grid(functools.partial(FPParticleSolver, num_particles=500, seed=7)),
    "MeshlessGalerkinFPSolver": _meshless_fp_alone,
    "FPFEMSolver": _fem_fp_alone,
}
#: FP solvers with no such run, and why.
NOT_RUN = {
    "FPNetworkSolver": "a network solver: its rates are NetworkHamiltonian.optimal_control's",
    "WeakFormFPSolver": "the weak-form family's base: FPFEMSolver and MeshlessGalerkinFPSolver are run",
}


def _every_fp_solver():
    pytest.importorskip("skfem")  # the FEM modules import it
    for module in pkgutil.walk_packages(mfgarchon.alg.numerical.__path__, "mfgarchon.alg.numerical."):
        importlib.import_module(module.name)

    def below(cls):
        for sub in cls.__subclasses__():
            yield sub
            yield from below(sub)

    return {cls.__name__: cls for cls in below(BaseFPSolver) if cls.__module__.startswith("mfgarchon.")}


def test_every_fp_solver_is_run_or_named_with_its_reason():
    """The population of the attribute pin below is every FP solver class of the package, so a new one is classified."""
    solvers = _every_fp_solver()
    assert set(solvers) == set(RUN_ON_THE_VALUE_FUNCTION) | set(NOT_RUN)
    assert not any(solvers[name]._forms_its_drift_from_u for name in NOT_RUN)


@pytest.mark.parametrize("name", sorted(RUN_ON_THE_VALUE_FUNCTION))
def test_the_attribute_the_router_reads_says_whether_the_solver_forms_its_drift_from_u(name, monkeypatch):
    """_forms_its_drift_from_u is True exactly for the solvers whose value-function route forms the drift from the
    scalar, -c*grad(U): the router reads it, and only it, to send such a solver an own optimal control as a velocity or
    refuse it. Every module's binding of fp_drift_coefficient is spied, so a call through any of them is counted."""
    calls = []

    def spy(*args, **kwargs):
        calls.append(kwargs.get("context", "fp_drift_coefficient"))
        return fp_drift_coefficient(*args, **kwargs)

    solvers = _every_fp_solver()
    for module_name, module in list(sys.modules.items()):
        if (
            module_name.startswith("mfgarchon")
            and getattr(module, "fp_drift_coefficient", None) is fp_drift_coefficient
        ):
            monkeypatch.setattr(module, "fp_drift_coefficient", spy)
    assert pde_coefficients.fp_drift_coefficient is spy
    _quietly(RUN_ON_THE_VALUE_FUNCTION[name])
    assert bool(calls) is solvers[name]._forms_its_drift_from_u, calls
