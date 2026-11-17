"""#2975: a value a solver reads during the solve that is non-finite makes the solve raise where it reads it, naming what
was read and where.

`refuse_a_non_finite_read` is the one owner and `NonFiniteReadError` the one error. Before it, the same value gave a
finite wrong U with `converged` False (HJB-FDM's Newton step), a wrong M with `converged` True (FP-FVM's mask, HJB-SL's
fallback, the network FP's dropped rate), a non-finite result returned without an error (the doors), or an error that
blamed the CFL condition or a linear solve. Each pin below runs through `problem.solve()` where the site is reachable
from it; the doors and the network FP are called as their callers call them.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

import numpy as np
from scipy.sparse.linalg import LinearOperator

import mfgarchon
from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.core.coupling_costs import PotentialCost
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry import boundary as bnd
from mfgarchon.types.schemes import NumericalScheme
from mfgarchon.utils.non_finite_read import NonFiniteReadError, refuse_a_non_finite_read

NAN = float("nan")
OWN = "is non-finite where it reads it"


# --- The owner. ---


def test_the_owner_names_the_first_non_finite_value_the_solver_and_where():
    with pytest.raises(NonFiniteReadError) as caught:
        refuse_a_non_finite_read("S", "time step 2 of 5", {"a": np.ones(3), "b": [1.0, np.inf, NAN], "c": NAN})
    assert str(caught.value) == f"S: b {OWN}: 2 of 3 values, the first inf, at time step 2 of 5 (#2975)."
    assert isinstance(caught.value, ValueError)


@pytest.mark.parametrize("bad", [NAN, np.inf, -np.inf], ids=repr)
def test_the_owner_names_the_first_non_finite_value_whatever_it_is(bad):
    with pytest.raises(NonFiniteReadError, match=rf"the first {bad}, at w "):
        refuse_a_non_finite_read("S", "w", {"a": [1.0, bad, NAN]})


def test_the_owner_passes_finite_values_and_skips_what_was_not_handed():
    refuse_a_non_finite_read("S", "its door", {"a": np.zeros((2, 3)), "b": None, "c": 1e308})


# --- Site 6: every concrete door reads its arrays through the owner first, read from source. ---

_PACKAGE = pathlib.Path(mfgarchon.__file__).parent
_DOOR_NAMES = ("solve_hjb_system", "solve_fp_system")


def _named(node: ast.expr, name: str) -> bool:
    return (isinstance(node, ast.Name) and node.id == name) or (isinstance(node, ast.Attribute) and node.attr == name)


def _reads_its_door_first(function: ast.FunctionDef) -> bool:
    """Whether the first statement after the docstring is the owner's call at "its door": a door reads what it is
    handed before any work. A call anywhere in the body is not enough, since a door whose loop also calls the owner, as
    FP-FVM's does, would pass with its own read deleted."""
    body = function.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    if not body or not isinstance(body[0], ast.Expr) or not isinstance(body[0].value, ast.Call):
        return False
    call = body[0].value
    return (
        _named(call.func, "refuse_a_non_finite_read")
        and len(call.args) >= 2
        and isinstance(call.args[1], ast.Constant)
        and call.args[1].value == "its door"
    )


def _doors() -> dict[str, ast.FunctionDef]:
    """Every `solve_hjb_system` and `solve_fp_system` defined in a class of the package, less a Protocol's stub and an
    abstract method, which declare a door and read nothing."""
    found = {}
    for path in sorted(_PACKAGE.rglob("*.py")):
        for cls in (n for n in ast.walk(ast.parse(path.read_text(encoding="utf-8"))) if isinstance(n, ast.ClassDef)):
            if any(_named(base, "Protocol") for base in cls.bases):
                continue
            for function in cls.body:
                if (
                    isinstance(function, ast.FunctionDef)
                    and function.name in _DOOR_NAMES
                    and not any(_named(d, "abstractmethod") for d in function.decorator_list)
                ):
                    found[f"{path.relative_to(_PACKAGE.parent).as_posix()}:{cls.name}.{function.name}"] = function
    return found


_DOORS = _doors()


def test_the_walk_finds_the_doors_known_to_exist():
    """The control: a walk that lost a directory or a class would pass the test below vacuously."""
    classes = {key.split(":")[1] for key in _DOORS}
    known = {
        "HJBFDMSolver.solve_hjb_system",
        "HJBSemiLagrangianSolver.solve_hjb_system",
        "HJBWENOSolver.solve_hjb_system",
        "HJBGFDMSolver.solve_hjb_system",
        "HJBHowardSolver.solve_hjb_system",
        "NetworkHJBSolver.solve_hjb_system",
        "NetworkPolicyIterationHJBSolver.solve_hjb_system",
        "FPFDMSolver.solve_fp_system",
        "FPFVMSolver.solve_fp_system",
        "FPParticleSolver.solve_fp_system",
        "FPSLSolver.solve_fp_system",
        "FPNetworkSolver.solve_fp_system",
    }
    assert known <= classes, sorted(known - classes)
    assert not any(key.endswith((":BaseHJBSolver.solve_hjb_system", ":BaseFPSolver.solve_fp_system")) for key in _DOORS)


_DOOR_SHAPES = {
    "reads first": (
        '"""Doc."""\nrefuse_a_non_finite_read("S", "its door", {"M_initial": M_initial})\nreturn M_initial',
        True,
    ),
    "no call": ('"""Doc."""\nreturn M_initial', False),
    "a call in its loop only": (
        'm = M_initial\nfor k in range(3):\n    refuse_a_non_finite_read("S", "step", {"m": m})\nreturn m',
        False,
    ),
    "a call after work": ('m = 2 * M_initial\nrefuse_a_non_finite_read("S", "its door", {"m": m})\nreturn m', False),
}


@pytest.mark.parametrize(("body", "reads"), list(_DOOR_SHAPES.values()), ids=list(_DOOR_SHAPES))
def test_the_door_predicate_tells_a_door_read_from_the_others(body, reads):
    indented = "".join(f"        {line}\n" for line in body.splitlines())
    tree = ast.parse(f"class A:\n    def solve_fp_system(self, M_initial):\n{indented}")
    assert _reads_its_door_first(tree.body[0].body[0]) is reads


@pytest.mark.parametrize("door", sorted(_DOORS))
def test_each_concrete_door_reads_its_arrays_through_the_owner_first(door):
    assert _reads_its_door_first(_DOORS[door]), f"{door} does not read the arrays it is handed through the owner first"


# --- The problems: 1-D, 11 nodes, T 0.5, Nt 5, u_terminal sin(pi x), whose gradient reaches pi. ---


def _at(x, where, bad, base=0.0):
    a = np.asarray(x[0], dtype=float)
    out = np.broadcast_to(np.asarray(base(a) if callable(base) else base, dtype=float), a.shape).copy()
    return np.where(where(a), bad, out)


def _near(c):
    return lambda a: np.isclose(a, c, rtol=0, atol=1e-9)


class _NaNWhereSteep(SeparableHamiltonian):
    """A Hamiltonian whose own method is NaN where |p| > 1, a p only the solve hands it."""

    method = ""

    def _steep(self, p, out):
        out = np.array(out, dtype=float, copy=True)
        steep = np.abs(np.asarray(p, dtype=float)).max(axis=-1) > 1.0
        out[steep] = np.nan
        return out

    def __call__(self, t, x, p, m):
        out = super().__call__(t=t, x=x, p=p, m=m)
        return self._steep(p, out) if self.method == "__call__" else out

    def dp(self, t, x, p, m):
        out = super().dp(t=t, x=x, p=p, m=m)
        return self._steep(p, out) if self.method == "dp" else out

    def optimal_control(self, t, x, p, m):
        out = super().optimal_control(t=t, x=x, p=p, m=m)
        return self._steep(p, out) if self.method == "optimal_control" else out


def _steep(method):
    hamiltonian = _NaNWhereSteep(QuadraticControlCost(control_cost=1.0))
    hamiltonian.method = method
    return hamiltonian


def _problem(hamiltonian=None, model_kw=None, dimension=1, bc=None, **kw):
    if hamiltonian is None or isinstance(hamiltonian, dict):
        hamiltonian = SeparableHamiltonian(QuadraticControlCost(control_cost=1.0), **(hamiltonian or {}))
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0)] * dimension,
        Nx_points=[11] if dimension == 1 else [6, 6],
        boundary_conditions=bc if bc is not None else bnd.no_flux_bc(dimension=dimension),
    )
    return MFGProblem(
        model=Model(hamiltonian=hamiltonian, **({"volatility": 0.2} | (model_kw or {}))),
        domain=grid,
        conditions=Conditions(
            m_initial=lambda x: 1.0 + 0.5 * np.asarray(x[0], dtype=float),
            u_terminal=lambda x: np.sin(np.pi * np.asarray(x[0], dtype=float)),
            T=0.5,
        ),
        Nt=5,
        **kw,
    )


def _operator_nan_in_row_0():
    def matvec(v):
        out = 0.1 * np.asarray(v, dtype=float).copy()
        out[0] = NAN
        return out

    return LinearOperator((11, 11), matvec=matvec, dtype=float)


# --- Site 1: HJB-FDM's Newton step names the term of its residual or Jacobian that is non-finite. ---

_SITE_1 = [
    ("the potential at x = 0.3", lambda: _problem({"potential": lambda t, x: _at(x, _near(0.3), NAN)}),
     r"SeparableHamiltonian's potential " + OWN + r": .*, the first nan, at t=[0-9.]+, x=\[0\.3"),
    ("the coupling where m < 1", lambda: _problem({"coupling": lambda m: np.where(np.asarray(m) < 1.0 - 1e-9, NAN, 0.5 * np.asarray(m))}),
     r"SeparableHamiltonian's coupling " + OWN + r": .*, the first nan, at t="),
    ("the HJB source at x = 0", lambda: _problem(source_term_hjb=lambda t, x: _at(x, _near(0.0), NAN)),
     r"source_term_hjb " + OWN + r": .*, the first nan, at t="),
    ("the coupling cost at x = 0.3", lambda: _problem(model_kw={"coupling_cost": PotentialCost(lambda x: _at(x, _near(0.3), NAN, lambda a: 0.1 * a))}),
     r"coupling_cost " + OWN + r": .*, the first nan, at t="),
    ("nonlocal_operator in row 0", lambda: _problem(nonlocal_operator=_operator_nan_in_row_0()), r"nonlocal_operator " + OWN + r": .*, the first nan, at t="),
    ("the wall at x = 0", lambda: _problem(state_penalty=lambda x: _at(x, _near(0.0), NAN)), r"state_penalty " + OWN + r": .*, the first nan, at t="),
    ("an own __call__ where |p| > 1", lambda: _problem(_steep("__call__")), r"_NaNWhereSteep's value " + OWN + r": .*, the first nan, at t="),
    ("an own dp where |p| > 1", lambda: _problem(_steep("dp")), r"_NaNWhereSteep's dp " + OWN + r": .*, the first nan, at t=.*, x=.*, p=.*, m="),
]  # fmt: skip


@pytest.mark.parametrize(("build", "names"), [row[1:] for row in _SITE_1], ids=[row[0] for row in _SITE_1])
def test_hjb_fdm_names_the_term_its_newton_step_read(build, names):
    """At 4ca7c9e0 each gave "the Newton step could not be computed" and a finite wrong U with `converged` False (max
    |U - U_twin| 0.054 to 0.65)."""
    with pytest.raises(NonFiniteReadError, match=r"HJBFDMSolver's Newton step: " + names):
        build().solve(max_iterations=20, verbose=False)


def test_hjb_fdm_in_two_dimensions_names_the_term():
    """The n-D residual HJB-FDM hands NewtonSolver: at 4ca7c9e0, "Matrix is exactly singular" 315 times and a finite
    wrong U."""
    problem = _problem(
        {"potential": lambda t, x: _at(x, lambda a: np.isclose(a, 0.2) & np.isclose(np.asarray(x[1]), 0.6), NAN)},
        dimension=2,
    )
    with pytest.raises(
        NonFiniteReadError,
        match=r"HJBFDMSolver's Newton step: SeparableHamiltonian's potential " + OWN + r": .*, the first nan, at t=",
    ):
        problem.solve(max_iterations=20, verbose=False)


# --- Sites 2 and 3: FP-FDM and FP-FVM read the drift and the source before a step advances them. ---


@pytest.mark.parametrize(
    ("scheme", "names"),
    [
        (
            NumericalScheme.FDM_UPWIND,
            r"compute_fp_velocity_field: the optimal control " + OWN + r": .*, the first nan, at t=",
        ),
        (
            NumericalScheme.FVM_UPWIND,
            r"the optimal control at the faces of axis 0 " + OWN + r": .*, the first nan, at time step",
        ),
    ],
    ids=["FDM_UPWIND", "FVM_UPWIND"],
)
def test_the_fp_names_a_non_finite_optimal_control(scheme, names):
    """At 4ca7c9e0, FDM blamed the CFL condition, and FVM returned a wrong M (max |M - M_twin| 9.09) with `converged`
    True: its mask read a NaN velocity as no advection."""
    with pytest.raises(NonFiniteReadError, match=names):
        _problem(_steep("optimal_control")).solve(scheme=scheme, max_iterations=20, verbose=False)


def test_fp_fdm_names_a_non_finite_source():
    problem = _problem(source_term_fp=lambda t, x: _at(x, lambda a: np.full(a.shape, abs(float(t) - 0.5) < 1e-12), NAN))
    with pytest.raises(
        NonFiniteReadError,
        match=r"FPFDMSolver: source_term " + OWN + r": .*, the first nan, at t=0\.5, the rows the step keeps",
    ):
        problem.solve(max_iterations=20, verbose=False)


def test_fp_fdm_does_not_judge_a_source_at_a_dirichlet_node_whose_row_discards_it():
    """A Dirichlet row reads m = g, so the source there never enters the step: at 4ca7c9e0 a NaN source only at a
    Dirichlet node solved bit for bit as its twin under FDM_UPWIND (#2974's sweep, 8 such cells), and it still does."""

    def source(bad):
        return lambda t, x: _at(x, _near(0.0), bad)

    bc = bnd.dirichlet_bc(value=0.0, dimension=1)
    nan, twin = (
        _problem(bc=bc, source_term_fp=source(bad)).solve(max_iterations=20, verbose=False) for bad in (NAN, 0.0)
    )
    assert np.array_equal(nan.U, twin.U)
    assert np.array_equal(nan.M, twin.M)


def test_fp_fdm_refuses_a_source_at_a_capped_exit_whose_cap_reads_it():
    """A capped exit reads the balance at its Dirichlet nodes, source included, to decide its switch (#2581), so the
    source there reaches the solution: at 4ca7c9e0 a NaN only at the capped exit node returned a finite M unlike its
    twin's (max |M - M_twin| 1.66) with no error. A coupled solve refuses a capped exit, so the FP is solved alone."""
    from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
    from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions

    capped = BoundaryConditions(
        dimension=1,
        segments=[
            BCSegment(name="x_min", bc_type=BCType.NO_FLUX, boundary="x_min"),
            BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=0.0, boundary="x_max", flux_capacity=0.5),
        ],
    )
    x = np.linspace(0.0, 1.0, 11)
    solver = FPFDMSolver(_problem(), boundary_conditions=capped)
    with pytest.raises(NonFiniteReadError, match=r"FPFDMSolver: source_term " + OWN + r": .*, the first nan, at t="):
        solver.solve_fp_system(
            np.exp(0.5 * x),
            potential_field=np.broadcast_to(-0.5 * x, (6, 11)).copy(),
            source_term=lambda t, x: _at(x, _near(1.0), NAN),
            show_progress=False,
        )


def test_fp_fdm_still_names_the_cfl_condition_for_a_step_that_overflows_from_finite_inputs():
    """The check after the step stays for a non-finite M the step computed from finite inputs."""
    from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver

    with pytest.raises(ValueError, match="Check CFL condition") as caught:
        FPFDMSolver(_problem()).solve_fp_system(np.full(11, 1e308), drift_field=np.full((6, 11), 1e3))
    assert not isinstance(caught.value, NonFiniteReadError)


# --- Site 4, HJB half: HJB-SL's n-D value update names the read behind a non-finite result. ---


def test_hjb_sl_names_the_read_where_its_fallback_stood():
    """At 4ca7c9e0 the update stored U at t_{n+1} for the one node and raised only past 10% of the points: max |U -
    U_twin| 0.0117 with `converged` True, and 95 log warnings."""
    problem = _problem(
        {"potential": lambda t, x: _at(x, lambda a: np.isclose(a, 0.4) & np.isclose(np.asarray(x[1]), 0.6), NAN)},
        dimension=2,
    )
    with pytest.raises(
        NonFiniteReadError,
        match=r"HJBSemiLagrangianSolver's value update: SeparableHamiltonian's potential "
        + OWN
        + r": .*, the first nan, at t=",
    ):
        problem.solve(scheme=NumericalScheme.SL_LINEAR, max_iterations=20, verbose=False)


# --- The network FP read a NaN transition rate as no transition. ---


def test_the_network_fp_refuses_a_non_finite_transition_rate(monkeypatch):
    from mfgarchon.alg.numerical.network_solvers.fp_network import FPNetworkSolver
    from mfgarchon.extensions.topology import NetworkMFGComponents, NetworkMFGProblem
    from mfgarchon.geometry.graph.network_geometry import GridNetwork

    network = GridNetwork(width=3, height=3)
    network.create_network()
    problem = NetworkMFGProblem(geometry=network, T=0.5, Nt=5, components=NetworkMFGComponents())
    H = problem.hamiltonian_class
    given = type(H).optimal_control

    def nan_at_node_4(self, *args, **kwargs):
        alpha = np.array(given(self, *args, **kwargs), dtype=float, copy=True)
        alpha[4] = np.nan
        return alpha

    monkeypatch.setattr(type(H), "optimal_control", nan_at_node_4)
    u = np.tile(np.linspace(0.0, 1.0, network.num_nodes), (6, 1))
    with pytest.raises(
        NonFiniteReadError,
        match=r"FPNetworkSolver: the transition rates out of node \d+ "
        + OWN
        + r": .*, the first nan, at t=.*, node \d+",
    ):
        FPNetworkSolver(problem).solve_fp_system(np.ones(network.num_nodes), potential_field=u)


# --- Site 6: an array handed at a door is refused there, by name. ---


def _doors_to_call():
    from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
    from mfgarchon.alg.numerical.fp_solvers.fp_fvm import FPFVMSolver
    from mfgarchon.alg.numerical.fp_solvers.fp_particle import FPParticleSolver
    from mfgarchon.alg.numerical.fp_solvers.fp_semi_lagrangian_adjoint import FPSLSolver
    from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver, HJBSemiLagrangianSolver
    from mfgarchon.alg.numerical.hjb_solvers.hjb_weno import HJBWENOSolver

    return {
        "HJBFDMSolver": ("hjb", HJBFDMSolver),
        "HJBSemiLagrangianSolver": ("hjb", HJBSemiLagrangianSolver),
        "HJBWENOSolver": ("hjb", HJBWENOSolver),
        "FPFDMSolver": ("drift_field", FPFDMSolver),
        "FPFVMSolver": ("drift_field", FPFVMSolver),
        "FPParticleSolver": ("drift_field", lambda p: FPParticleSolver(p, num_particles=500)),
        "FPSLSolver": ("potential_field", FPSLSolver),
    }


_DOOR_ROWS = [
    (solver, array)
    for solver, kind in [
        ("HJBFDMSolver", "hjb"),
        ("HJBSemiLagrangianSolver", "hjb"),
        ("HJBWENOSolver", "hjb"),
        ("FPFDMSolver", "fp"),
        ("FPFVMSolver", "fp"),
        ("FPParticleSolver", "fp"),
        ("FPSLSolver", "fp"),
    ]
    for array in (("U_terminal", "M_density", "U_coupling_prev") if kind == "hjb" else ("M_initial", "field"))
]


@pytest.mark.parametrize(("solver", "array"), _DOOR_ROWS, ids=[f"{s}-{a}" for s, a in _DOOR_ROWS])
def test_a_door_refuses_a_non_finite_array_by_name(solver, array):
    """At 4ca7c9e0 the HJB doors returned a non-finite U for a NaN U_terminal or M_density, and the FP doors blamed the
    CFL condition (#2976, folded here)."""
    kind, factory = _doors_to_call()[solver]
    door = factory(_problem())
    if kind == "hjb":
        args = {
            "M_density": np.ones((6, 11)),
            "U_terminal": np.sin(np.pi * np.linspace(0, 1, 11)),
            "U_coupling_prev": np.zeros((6, 11)),
        }
        args[array] = np.full_like(args[array], NAN)
        call = lambda: door.solve_hjb_system(**args)  # noqa: E731
    else:
        name = kind if array == "field" else array
        args = {"M_initial": np.ones(11), kind: np.zeros((6, 11))}
        args[name] = np.full_like(args[name], NAN)
        array = name
        call = lambda: door.solve_fp_system(**args)  # noqa: E731
    with pytest.raises(
        NonFiniteReadError, match=rf"{solver}\.solve_(hjb|fp)_system: {array} {OWN}: .*, the first nan, at its door"
    ):
        call()
