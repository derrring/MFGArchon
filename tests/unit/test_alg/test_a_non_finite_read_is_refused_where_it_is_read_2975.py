"""#2975: a non-finite value a solver reads makes the solve raise at the step whose result it reaches, or at the decision
it enters, naming the first non-finite value that step read and where. A value the step reads and discards is not
judged.

`refuse_a_non_finite_read` is the one owner and `NonFiniteReadError` the one error. Before it, the same value gave a
finite wrong U with `converged` False (HJB-FDM's Newton step), a wrong M with `converged` True (FP-FVM's mask, HJB-SL's
fallback, the network FP's dropped rate), a non-finite result returned without an error (the doors), or an error that
blamed the CFL condition or a linear solve. Each pin below runs through `problem.solve()` where the site is reachable
from it; the doors and the network FP are called as their callers call them.
"""

from __future__ import annotations

import ast
import pathlib
import warnings

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


def _door_reads(function: ast.FunctionDef) -> set[str]:
    """The names a door's read at "its door" maps, read from its first statement's mapping."""
    body = (
        function.body[1:]
        if isinstance(function.body[0], ast.Expr) and isinstance(function.body[0].value, ast.Constant)
        else function.body
    )
    mapping = body[0].value.args[2]
    assert isinstance(mapping, ast.Dict)
    return {key.value for key in mapping.keys if isinstance(key, ast.Constant)}


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


def _solution_rows(door: str) -> set[str]:
    if door.endswith(".solve_fp_system"):
        return {"M_initial", "initial_particles"} if "FPParticleSolver" in door else {"M_initial"}
    return {"U_terminal", "U_final_condition_at_T"} if "WeakFormHJBSolver" in door else {"U_terminal"}


@pytest.mark.parametrize("door", sorted(_DOORS))
def test_each_door_reads_the_solution_row_it_is_handed_and_nothing_else(door):
    """A door reads the solution's own row, its terminal value or its initial density; every other array it is handed is
    judged by the step that reads it, where it entered (#2975, the approver's ruling 3 of 2026-11-17)."""
    assert _door_reads(_DOORS[door]) == _solution_rows(door), door


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


def _problem(hamiltonian=None, model_kw=None, dimension=1, bc=None, u_terminal=None, m_initial=None, **kw):
    if hamiltonian is None or isinstance(hamiltonian, dict):
        hamiltonian = SeparableHamiltonian(QuadraticControlCost(control_cost=1.0), **(hamiltonian or {}))
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0)] * dimension,
        Nx_points=[11] if dimension == 1 else [6, 6],
        boundary_conditions=bc if bc is not None else bnd.no_flux_bc(dimension=dimension),
    )
    # m_initial 1 + x/2 keeps m >= 1 at the start, which the coupling row needs; its mass 1.25 is intended, so the
    # library's warning that it does not rescale it is silenced as the library says to.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="initial density mass", category=UserWarning)
        return MFGProblem(
            model=Model(hamiltonian=hamiltonian, **({"volatility": 0.2} | (model_kw or {}))),
            domain=grid,
            conditions=Conditions(
                m_initial=m_initial or (lambda x: 1.0 + 0.5 * np.asarray(x[0], dtype=float)),
                u_terminal=u_terminal or (lambda x: np.sin(np.pi * np.asarray(x[0], dtype=float))),
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


@pytest.mark.filterwarnings("ignore:Newton stopped:RuntimeWarning", "ignore::scipy.sparse.linalg.MatrixRankWarning")
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


@pytest.mark.filterwarnings("ignore::scipy.sparse.linalg.MatrixRankWarning")
@pytest.mark.parametrize(
    ("scheme", "names"),
    [
        (
            NumericalScheme.FDM_UPWIND,
            r"FPFDMSolver: drift_field " + OWN + r": .*, the first nan, at time step \d+ of 5, .*, where it entered",
        ),
        (
            NumericalScheme.FVM_UPWIND,
            r"FPFVMSolver's advection sub-step count: the optimal control at the faces of axis 0 "
            + OWN
            + r": .*, the first nan, at time step \d+ of 5, .*, face \(\d+,\), where it entered",
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
        match=r"FPFDMSolver: source_term "
        + OWN
        + r": .*, the first nan, at time step 5 of 5, from t=0\.4 to t=0\.5, node \(\d+,\), "
        r"where it entered the step",
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
    with pytest.raises(
        NonFiniteReadError,
        match=r"FPFDMSolver: source_term " + OWN + r": .*, the first nan, at t=.*, the capped exits' nodes \[10\]",
    ):
        solver.solve_fp_system(
            np.exp(0.5 * x),
            potential_field=np.broadcast_to(-0.5 * x, (6, 11)).copy(),
            source_term=lambda t, x: _at(x, _near(1.0), NAN),
            show_progress=False,
        )


def test_fp_fdm_still_names_the_cfl_condition_for_a_step_that_overflows_from_finite_inputs():
    """The check after the step stays for a non-finite M the step computed from finite inputs."""
    from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver

    # The step overflows by design; numpy's own warning about it is not this test's subject.
    with np.errstate(over="ignore", invalid="ignore"), pytest.raises(ValueError, match="Check CFL condition") as caught:
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
        FPNetworkSolver(problem, diffusion_coefficient=0.1).solve_fp_system(
            np.ones(network.num_nodes), potential_field=u
        )


# --- A door reads the solution's row it is handed; each other array is judged by the step that reads it. ---


def _door_calls():
    """Each concrete door, built on a small problem of its kind, with what it is handed under the names it takes."""
    from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
    from mfgarchon.alg.numerical.fp_solvers.fp_fvm import FPFVMSolver
    from mfgarchon.alg.numerical.fp_solvers.fp_particle import FPParticleSolver
    from mfgarchon.alg.numerical.fp_solvers.fp_semi_lagrangian_adjoint import FPSLSolver
    from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver, HJBSemiLagrangianSolver
    from mfgarchon.alg.numerical.hjb_solvers.hjb_gfdm import HJBGFDMSolver
    from mfgarchon.alg.numerical.hjb_solvers.hjb_weno import HJBWENOSolver
    from mfgarchon.alg.numerical.network_solvers.fp_network import FPNetworkSolver
    from mfgarchon.alg.numerical.network_solvers.hjb_network import NetworkHJBSolver, NetworkPolicyIterationHJBSolver
    from mfgarchon.extensions.topology import NetworkMFGComponents, NetworkMFGProblem
    from mfgarchon.geometry.graph.network_geometry import GridNetwork

    network = GridNetwork(width=3, height=3)
    network.create_network()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        on_network = NetworkMFGProblem(geometry=network, T=0.5, Nt=5, components=NetworkMFGComponents())
    nodes = network.num_nodes

    def gfdm():
        problem = _problem()
        return HJBGFDMSolver(problem, collocation_points=problem.geometry.get_spatial_grid().reshape(-1, 1), delta=0.3)

    hjb = {"M_density": np.ones((6, 11)), "U_terminal": np.zeros(11), "U_coupling_prev": np.zeros((6, 11))}
    fp = {"M_initial": np.ones(11), "potential_field": np.zeros((6, 11))}
    return {
        "HJBFDMSolver": (lambda: HJBFDMSolver(_problem()), "solve_hjb_system", hjb),
        "HJBSemiLagrangianSolver": (lambda: HJBSemiLagrangianSolver(_problem()), "solve_hjb_system", hjb),
        "HJBWENOSolver": (lambda: HJBWENOSolver(_problem()), "solve_hjb_system", hjb),
        "HJBGFDMSolver": (gfdm, "solve_hjb_system", hjb),
        "NetworkHJBSolver": (
            lambda: NetworkHJBSolver(on_network),
            "solve_hjb_system",
            {"M_density": np.ones((6, nodes)), "U_terminal": np.zeros(nodes)},
        ),
        "NetworkPolicyIterationHJBSolver": (
            lambda: NetworkPolicyIterationHJBSolver(on_network),
            "solve_hjb_system",
            {"M_density": np.ones((6, nodes)), "U_terminal": np.zeros(nodes)},
        ),
        "FPFDMSolver": (lambda: FPFDMSolver(_problem()), "solve_fp_system", fp),
        "FPFVMSolver": (lambda: FPFVMSolver(_problem()), "solve_fp_system", fp),
        "FPParticleSolver": (lambda: FPParticleSolver(_problem(), num_particles=500), "solve_fp_system", fp),
        "FPSLSolver": (lambda: FPSLSolver(_problem()), "solve_fp_system", fp),
        "FPNetworkSolver": (
            lambda: FPNetworkSolver(on_network, diffusion_coefficient=0.1),
            "solve_fp_system",
            {"M_initial": np.ones(nodes), "potential_field": np.zeros((6, nodes))},
        ),
    }


_CALLED_DOORS = [
    "HJBFDMSolver",
    "HJBSemiLagrangianSolver",
    "HJBWENOSolver",
    "HJBGFDMSolver",
    "NetworkHJBSolver",
    "NetworkPolicyIterationHJBSolver",
    "FPFDMSolver",
    "FPFVMSolver",
    "FPParticleSolver",
    "FPSLSolver",
    "FPNetworkSolver",
]


@pytest.mark.filterwarnings("ignore:HJBGFDMSolver:UserWarning")
@pytest.mark.parametrize("solver", _CALLED_DOORS)
def test_a_door_refuses_a_non_finite_solution_row_by_name(solver):
    """At 4ca7c9e0 the HJB doors returned a non-finite U for a NaN U_terminal, and the FP doors blamed the CFL condition
    (#2976, folded here). The weak forms' and Howard's doors are read by the AST tests above."""
    build, method, handed = _door_calls()[solver]
    row = "U_terminal" if method == "solve_hjb_system" else "M_initial"
    args = dict(handed)
    args[row] = np.full_like(args[row], NAN)
    with pytest.raises(NonFiniteReadError, match=rf"{solver}\.{method}: {row} {OWN}: .*, the first nan, at its door"):
        getattr(build(), method)(**args)


_X = np.linspace(0.0, 1.0, 11)


def _hjb_door_args(**changes):
    args = {
        "M_density": np.tile(1.0 + 0.5 * _X, (6, 1)),
        "U_terminal": np.sin(np.pi * _X),
        "U_coupling_prev": np.tile(np.sin(np.pi * _X), (6, 1)),
    }
    args.update(changes)
    return args


def _nan_at(array, index):
    out = np.array(array, dtype=float, copy=True)
    out[index] = NAN
    return out


@pytest.mark.parametrize(
    ("array", "index"),
    [("M_density", (-1, 3)), ("U_coupling_prev", (-1, 3)), ("U_coupling_prev", slice(None))],
    ids=["M_density at T", "U_coupling_prev at T", "U_coupling_prev whole"],
)
def test_the_1d_hjb_fdm_door_hands_on_what_its_step_does_not_read(array, index):
    """Review 1's A5 and C3: the 1-D backward loop reads no density row at T, and no row of U_coupling_prev at all (the
    Jacobian takes it and reads none of it), so a NaN there solves bit for bit as its twin, as at 4ca7c9e0."""
    from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver

    args = _hjb_door_args()
    nan = HJBFDMSolver(_problem()).solve_hjb_system(**_hjb_door_args(**{array: _nan_at(args[array], index)}))
    assert np.array_equal(nan, HJBFDMSolver(_problem()).solve_hjb_system(**args))


def test_the_fp_fdm_door_hands_on_a_velocity_row_its_steps_do_not_read():
    """Review 1's C3: the steps read the velocity at t_0 to t_{Nt-1}, so a NaN only at T solves as its twin."""
    from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver

    field = np.tile(0.1 * _X, (6, 1))
    nan = FPFDMSolver(_problem()).solve_fp_system(np.ones(11), drift_field=_nan_at(field, (-1, 3)), show_progress=False)
    assert np.array_equal(
        nan, FPFDMSolver(_problem()).solve_fp_system(np.ones(11), drift_field=field, show_progress=False)
    )


def test_the_newton_step_names_the_density_where_it_entered_h():
    """The density is read by H's coupling at each row, and named there (design call B, 2026-11-18)."""
    from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver

    problem = _problem({"coupling": lambda m: 0.5 * np.asarray(m, dtype=float)})
    args = _hjb_door_args(M_density=_nan_at(np.tile(1.0 + 0.5 * _X, (6, 1)), (slice(None), 4)))
    with pytest.raises(
        NonFiniteReadError,
        match=r"HJBFDMSolver's Newton step: M_density "
        + OWN
        + r": 1 of 1 values, the first nan, at t=[0-9.]+, x=\[0\.4",
    ):
        HJBFDMSolver(problem).solve_hjb_system(**args)


def test_a_density_h_does_not_read_is_discarded():
    """With no density term H reads no m: a NaN density alone solves as its twin, and beside a NaN potential at the same
    node the potential is named, not the density (design call B's discriminator)."""
    from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver

    nan_density = _nan_at(np.tile(1.0 + 0.5 * _X, (6, 1)), (slice(None), 4))
    alone = HJBFDMSolver(_problem()).solve_hjb_system(**_hjb_door_args(M_density=nan_density))
    assert np.array_equal(alone, HJBFDMSolver(_problem()).solve_hjb_system(**_hjb_door_args()))
    beside = _problem({"potential": lambda t, x: _at(x, _near(0.4), NAN)})
    with pytest.raises(NonFiniteReadError, match=r"SeparableHamiltonian's potential " + OWN):
        HJBFDMSolver(beside).solve_hjb_system(**_hjb_door_args(M_density=nan_density))


def test_an_inconclusive_density_test_does_not_name_the_density():
    """The density is named where H's value is finite with 1.0 in its place. A coupling NaN at m = 1 (#2974's row) leaves
    that test without an answer, so the density is not named, and the solve still raises (the approver's guard on
    design call B)."""
    from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver

    def coupling(m):
        m = np.asarray(m, dtype=float)
        return np.where(np.isclose(m, 1.0), NAN, 0.5 * m)

    problem = _problem({"coupling": coupling}, m_initial=lambda x: 2.0 + 0.0 * np.asarray(x[0], dtype=float))
    args = _hjb_door_args(M_density=_nan_at(np.full((6, 11), 2.0), (slice(None), 4)))
    with pytest.raises(NonFiniteReadError, match=r"SeparableHamiltonian's coupling " + OWN) as raised:
        HJBFDMSolver(problem).solve_hjb_system(**args)
    assert "M_density" not in str(raised.value)


@pytest.mark.filterwarnings("ignore:Newton stopped:RuntimeWarning", "ignore::scipy.sparse.linalg.MatrixRankWarning")
def test_the_nd_hjb_fdm_step_names_the_iterate_it_starts_from():
    """In n-D, Newton starts from the previous Picard pass's U at t_n, so a NaN in it is the first value the step reads
    (design call A, 2026-11-18)."""
    from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver

    terminal = np.sin(np.pi * np.linspace(0.0, 1.0, 6))[:, None] * np.ones((6, 6))
    previous = _nan_at(np.tile(terminal, (6, 1, 1)), (2, 3, 3))
    with pytest.raises(
        NonFiniteReadError,
        match=r"HJBFDMSolver's Newton step: U_coupling_prev " + OWN + r": .*, at the iterate it starts from, at t=0\.2",
    ):
        HJBFDMSolver(_problem(dimension=2)).solve_hjb_system(
            M_density=np.ones((6, 6, 6)), U_terminal=terminal, U_coupling_prev=previous
        )


def test_an_override_array_is_read_where_it_decides_the_diffusion_branch():
    """The override array decides the diffusion branch (scalar, field or tensor) before any step, and that decision
    consumes every entry, so it is read there: a NaN at a Dirichlet-only node is refused at the override's read (the
    approver's ruling of 2026-11-18 on `resolve_volatility_override`)."""
    from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver

    problem = _problem(bc=bnd.dirichlet_bc(value=0.0, dimension=1))
    with pytest.raises(
        NonFiniteReadError,
        match=r"1-D HJB-FDM \(solve_hjb_system_backward\): volatility "
        + OWN
        + r": 1 of 11 values, the first nan, at its "
        r"volatility= override",
    ):
        HJBFDMSolver(problem).solve_hjb_system(
            **_hjb_door_args(), volatility=_nan_at(np.full(11, 0.2), 0), volatility_kind="field"
        )


# --- Review 1's repros, each at its 4ca7c9e0 outcome (the approver's ruling 6 of 2026-11-17). ---


def test_the_per_point_residual_names_the_term_it_read():
    """A1: HJBFDMSolver(analytic_jacobian=False) on NumPyBackend takes the per-point residual, whose naming was off."""
    from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
    from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver

    problem = _problem({"potential": lambda t, x: _at(x, _near(0.3), NAN)})
    with pytest.raises(
        NonFiniteReadError,
        match=r"HJBFDMSolver's Newton step: SeparableHamiltonian's potential " + OWN + r": .*, x=\[0\.3",
    ):
        problem.solve(
            hjb_solver=HJBFDMSolver(problem, analytic_jacobian=False),
            fp_solver=FPFDMSolver(problem),
            max_iterations=20,
            verbose=False,
        )


@pytest.mark.filterwarnings("ignore:Newton stopped:RuntimeWarning", "ignore::scipy.sparse.linalg.MatrixRankWarning")
@pytest.mark.parametrize("dimension", [1, 2])
def test_a_source_that_returns_one_number_is_named(dimension):
    """A2: a source returning one number at t_0 raised IndexError inside the naming."""

    def source(t, x):
        return NAN if abs(float(np.ravel(t)[0])) < 1e-12 else 0.0

    with pytest.raises(
        NonFiniteReadError,
        match=r"HJBFDMSolver's Newton step: source_term_hjb " + OWN + r": .*, the first nan, at t=0, ",
    ):
        _problem(dimension=dimension, source_term_hjb=source).solve(max_iterations=20, verbose=False)


@pytest.mark.parametrize("policy", ["raise", "record"])
def test_a_value_iteration_that_diverges_from_finite_inputs_is_not_named(policy):
    """A3: a 2-D value iteration whose iterates overflow from finite inputs reads nothing non-finite at its start state,
    so the failure goes to #2644's policy, as at 4ca7c9e0: ConvergenceError under "raise", a non-finite U under
    "record"."""
    from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
    from mfgarchon.utils.exceptions import ConvergenceError

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        problem = MFGProblem(
            model=Model(hamiltonian=SeparableHamiltonian(QuadraticControlCost(control_cost=1.0)), volatility=1.0),
            domain=TensorProductGrid(
                bounds=[(0.0, 1.0)] * 2, Nx_points=[6, 6], boundary_conditions=bnd.no_flux_bc(dimension=2)
            ),
            conditions=Conditions(
                m_initial=lambda x: 1.0 + 0.5 * np.asarray(x[0], dtype=float),
                u_terminal=lambda x: np.sin(np.pi * np.asarray(x[0], dtype=float)),
                T=0.4,
            ),
            Nt=2,
        )
    solver = HJBFDMSolver(problem, solver_type="fixed_point", on_inner_failure=policy, max_newton_iterations=400)
    terminal = np.sin(np.pi * np.linspace(0.0, 1.0, 6))[:, None] * np.ones((6, 6))

    def solve():
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return solver.solve_hjb_system(
                M_density=np.ones((3, 6, 6)), U_terminal=terminal, U_coupling_prev=np.zeros((3, 6, 6))
            )

    if policy == "raise":
        with pytest.raises(ConvergenceError, match="fixed_point did not converge"):
            solve()
    else:
        assert not np.all(np.isfinite(solve()))


class _NaNOnlyWhereAnIterateGoes(SeparableHamiltonian):
    """An own value NaN at t = 0.3 where |p| > 1.2: past every start state at that time and short of Newton's iterates."""

    def __call__(self, t, x, p, m):
        out = np.array(super().__call__(t=t, x=x, p=p, m=m), dtype=float, copy=True)
        if abs(float(np.ravel(t)[0]) - 0.3) < 1e-9:
            momentum = np.abs(np.asarray(p, dtype=float))
            out[momentum.max(axis=-1) > 1.2 if momentum.ndim else momentum > 1.2] = NAN
        return out


@pytest.mark.filterwarnings("ignore:HJB inner Newton did not converge:RuntimeWarning")
def test_a_non_finite_h_only_at_a_newton_iterate_goes_to_the_inner_failure_policy():
    """Ruling 4's stated limit. With the potential 3 cos(pi x) and U = 0 at T, every start state at t = 0.3 has |p| at
    most 0.875 and Newton's iterates reach 1.657 (measured at 4ca7c9e0), so only an iterate meets the NaN. It is not
    named: the solve returns a finite U, which moved from its twin's, and at 4ca7c9e0 it returned the same U bit for
    bit (#2975's census)."""

    def solve(hamiltonian):
        return _problem(hamiltonian, u_terminal=lambda x: 0.0 * np.asarray(x[0], dtype=float)).solve(
            max_iterations=10, verbose=False
        )

    potential = {"potential": lambda t, x: 3.0 * np.cos(np.pi * np.asarray(x[0], dtype=float))}
    met = solve(_NaNOnlyWhereAnIterateGoes(QuadraticControlCost(control_cost=1.0), **potential))
    assert np.all(np.isfinite(met.U))
    assert not np.array_equal(met.U, solve(potential).U), "the NaN was not met: the case no longer reaches the limit"


def _potential_nan_at_04_06(t, x):
    return _at(x, lambda a: np.isclose(a, 0.4) & np.isclose(np.asarray(x[1]), 0.6), NAN)


@pytest.mark.parametrize("scheme", [NumericalScheme.SL_LINEAR, NumericalScheme.SL_CUBIC], ids=lambda s: s.name)
def test_hjb_sl_names_the_read_on_its_loop_without_sub_steps(scheme):
    """A4: U at T 0.2 sin(pi x) keeps every step's CFL number at most 1, so the loop without sub-steps runs."""
    problem = _problem(
        {"potential": _potential_nan_at_04_06},
        dimension=2,
        u_terminal=lambda x: 0.2 * np.sin(np.pi * np.asarray(x[0], dtype=float)),
    )
    with pytest.raises(
        NonFiniteReadError, match=r"HJBSemiLagrangianSolver's value update: SeparableHamiltonian's potential " + OWN
    ):
        problem.solve(scheme=scheme, max_iterations=20, verbose=False)


class _OwnDpNaNAtOneNode(SeparableHamiltonian):
    def dp(self, t, x, p, m):
        out = np.array(super().dp(t=t, x=x, p=p, m=m), dtype=float, copy=True)
        hit = np.isclose(np.asarray(x[0], dtype=float), 0.4) & np.isclose(np.asarray(x[1], dtype=float), 0.6)
        out[np.broadcast_to(hit.reshape(hit.shape + (1,) * (out.ndim - hit.ndim)), out.shape)] = NAN
        return out


@pytest.mark.filterwarnings("ignore::UserWarning:mfgarchon.core.mfg_problem")
def test_hjb_sl_names_an_own_dp_read_by_the_foot_through_the_solve():
    """An own dp NaN at one node, through problem.solve(): the CFL number reads the foot velocity, H's dp, ahead of the
    sub-step count and of the update's interpolation, and names it (census rows 2526 and 2539 at 4ca7c9e0)."""
    from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
    from mfgarchon.alg.numerical.hjb_solvers import HJBSemiLagrangianSolver

    problem = _problem(_OwnDpNaNAtOneNode(QuadraticControlCost(control_cost=1.0)), dimension=2)
    with pytest.raises(
        NonFiniteReadError,
        match=r"HJBSemiLagrangianSolver's CFL number: _OwnDpNaNAtOneNode's dp, the foot velocity "
        + OWN
        + r": .*, x=\[0\.4, 0\.6",
    ):
        problem.solve(
            hjb_solver=HJBSemiLagrangianSolver(problem),
            fp_solver=FPFDMSolver(problem),
            max_iterations=20,
            verbose=False,
        )


def test_a_source_at_an_uncapped_dirichlet_node_is_discarded_beside_a_capped_exit():
    """B1: the capped switch reads the balance at the capped nodes alone, so a NaN source at the other Dirichlet node is
    discarded by its row and solves bit for bit as its twin, as at 4ca7c9e0."""
    from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
    from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions

    capped = BoundaryConditions(
        dimension=1,
        segments=[
            BCSegment(name="entry", bc_type=BCType.DIRICHLET, value=0.0, boundary="x_min"),
            BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=0.0, boundary="x_max", flux_capacity=0.5),
        ],
    )

    def solve(bad):
        return FPFDMSolver(_problem(), boundary_conditions=capped).solve_fp_system(
            np.exp(0.5 * _X),
            potential_field=np.broadcast_to(-2.0 * _X, (6, 11)).copy(),
            source_term=lambda t, x: _at(x, _near(0.0), bad),
            show_progress=False,
        )

    assert np.array_equal(solve(NAN), solve(0.0))


class _ControlNaNWhere(SeparableHamiltonian):
    """An own optimal control NaN where ``where(t, x)`` holds."""

    where = None

    def optimal_control(self, t, x, p, m):
        out = np.array(super().optimal_control(t=t, x=x, p=p, m=m), dtype=float, copy=True)
        hit = np.broadcast_to(self.where(float(np.ravel(t)[0]), x), out.shape[: np.ndim(np.asarray(x[0]))])
        out[hit] = NAN
        return out


def _control_nan_where(where):
    hamiltonian = _ControlNaNWhere(QuadraticControlCost(control_cost=1.0))
    hamiltonian.where = where
    return hamiltonian


def _twin_runs(build, **solve_kw):
    nan, twin = (build(bad).solve(verbose=False, **solve_kw) for bad in (True, False))
    return nan, twin


@pytest.mark.parametrize(
    ("case", "dimension", "bc", "scheme"),
    [
        ("at T alone", 1, None, None),
        ("at T alone", 1, None, NumericalScheme.FVM_UPWIND),
        ("at T alone", 2, None, None),
        ("at faces only Dirichlet rows touch", 2, "dirichlet", None),
        ("at the repeated row x1 = 1", 2, "periodic", None),
        ("at the seam alone", 1, "periodic", NumericalScheme.FVM_UPWIND),
    ],
    ids=["T 1-D FDM", "T 1-D FVM", "T 2-D FDM", "Dirichlet faces 2-D", "repeated row 2-D FDM", "seam 1-D FVM"],
)
def test_an_optimal_control_the_fp_discards_solves_as_its_twin(case, dimension, bc, scheme):
    """B2, B3 and B4: a NaN optimal control only where the FP step discards it -- the level at T, which no step reads;
    the faces only Dirichlet rows touch; the repeated row of an endpoint-inclusive periodic grid; FP-FVM's seam -- solves
    bit for bit as its twin, as at 4ca7c9e0. The head of review 1 refused each."""
    where = {
        "at T alone": lambda t, x: np.full(np.shape(np.asarray(x[0])), abs(t - 0.5) < 1e-9),
        "at faces only Dirichlet rows touch": lambda t, x: np.isclose(np.asarray(x[0], dtype=float), 0.0),
        "at the repeated row x1 = 1": lambda t, x: np.isclose(np.asarray(x[1], dtype=float), 1.0),
        "at the seam alone": lambda t, x: np.asarray(x[0], dtype=float) > 1.0,
    }[case]
    boundary = {
        None: None,
        "dirichlet": bnd.dirichlet_bc(value=0.0, dimension=dimension),
        "periodic": bnd.periodic_bc(dimension=dimension),
    }[bc]

    def build(bad):
        return _problem(
            _control_nan_where(where if bad else (lambda t, x: np.zeros(np.shape(np.asarray(x[0])), bool))),
            dimension=dimension,
            bc=boundary,
        )

    nan, twin = _twin_runs(build, max_iterations=10, **({"scheme": scheme} if scheme else {}))
    assert np.array_equal(nan.U, twin.U)
    assert np.array_equal(nan.M, twin.M)


def test_a_multi_population_coupling_is_named_as_the_coupling():
    """C1: a density term re-read with the population rows the residual handed, so a coupling NaN only where the other
    population is dense is named as the coupling."""
    from mfgarchon.alg.numerical.coupling.multi_population_iterator import MultiPopulationIterator
    from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
    from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver
    from mfgarchon.core.mfg_components import MFGComponents
    from mfgarchon.core.multi_population import MultiPopulationProblem

    def make(k):
        def coupling(m, cross_density, pop_idx=k):
            if len(cross_density) > 1:
                other = np.asarray(cross_density[1 - pop_idx], dtype=float)
                return np.where(other > 0.9, NAN, 2.0 * other)
            return np.zeros_like(np.asarray(m, dtype=float))

        hamiltonian = SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=1.0),
            coupling=coupling,
            coupling_dm=lambda m: np.zeros_like(np.asarray(m, dtype=float)),
            population_index=k,
        )
        components = MFGComponents(
            m_initial=lambda x, k=k: np.exp(-((np.asarray(x[0]) - (0.3 + 0.4 * k)) ** 2) / 0.02),
            u_terminal=lambda x: np.asarray(x[0]) * 0.0,
            hamiltonian=hamiltonian,
        )
        grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=bnd.no_flux_bc(dimension=1))
        return MFGProblem(geometry=grid, Nt=8, T=1.0, volatility=0.15, components=components)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        problems = [make(k) for k in range(2)]
        iterator = MultiPopulationIterator(
            MultiPopulationProblem(populations=problems, population_names=["P0", "P1"]),
            [HJBFDMSolver(p) for p in problems],
            [FPFDMSolver(p) for p in problems],
            relaxation=0.5,
        )
        with pytest.raises(NonFiniteReadError, match=r"SeparableHamiltonian's coupling " + OWN):
            iterator.solve(max_iterations=6, tolerance=1e-10)


@pytest.mark.filterwarnings("ignore::scipy.sparse.linalg.MatrixRankWarning")
@pytest.mark.parametrize("kind", ["scalar", "array"])
def test_a_callable_volatility_is_named_by_the_step_that_reads_it(kind):
    """C4: a callable volatility returning a NaN at t = 0.2, one number or one per point, is named by the HJB's Newton
    step and by the FP's step, where it entered; `CoefficientField` no longer judges it where it is read (the approver's
    ruling of 2026-11-18)."""
    from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
    from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver

    def volatility(t, x):
        value = NAN if abs(float(t) - 0.2) < 1e-9 else 0.2
        return value if kind == "scalar" else np.full(np.shape(np.asarray(x[0] if isinstance(x, tuple) else x)), value)

    with pytest.raises(NonFiniteReadError, match=r"HJBFDMSolver's Newton step: volatility " + OWN + r": .*, at t=0\.2"):
        HJBFDMSolver(_problem()).solve_hjb_system(volatility=volatility, **_hjb_door_args())
    with pytest.raises(
        NonFiniteReadError, match=r"FPFDMSolver: volatility " + OWN + r": .*, where it entered the step"
    ):
        FPFDMSolver(_problem()).solve_fp_system(
            np.ones(11), potential_field=np.zeros((6, 11)), volatility=volatility, show_progress=False
        )


# --- The FP's failure path: the first non-finite read, in reading order, that entered the step. ---


def test_the_fp_step_names_the_entered_read_past_a_discarded_one():
    """A NaN source at a Dirichlet node, which its row discards, comes before a NaN source at an interior node in reading
    order. The bisection skips the first and names the second, at its node (the approver's ruling of 2026-11-18)."""
    from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver

    problem = _problem(bc=bnd.dirichlet_bc(value=0.0, dimension=1))
    with pytest.raises(
        NonFiniteReadError,
        match=r"FPFDMSolver: source_term "
        + OWN
        + r": 1 of 11 values, the first nan, at time step 1 of 5, from t=0 to t=0\.1, "
        r"node \(5,\), where it entered the step",
    ):
        FPFDMSolver(problem).solve_fp_system(
            np.ones(11),
            potential_field=np.zeros((6, 11)),
            source_term=lambda t, x: _at(x, lambda a: np.isclose(a, 0.0) | np.isclose(a, 0.5), NAN),
            show_progress=False,
        )


@pytest.mark.parametrize(
    ("solver", "names"),
    [
        ("FPFVMSolver", r"FPFVMSolver: source_term " + OWN + r": 1 of 11 values, .*, node \(4,\), where it entered"),
        ("FPFDMSolver", r"FPFDMSolver: source_term " + OWN + r": 1 of 11 values, .*, node \(4,\), where it entered"),
    ],
    ids=["FVM's Strang step", "FDM's explicit arm"],
)
def test_each_fp_arm_names_a_source_that_entered_its_step(solver, names):
    """FP-FVM's Strang step, and FP-FDM's explicit arm (a callable drift), on the same failure path."""
    from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
    from mfgarchon.alg.numerical.fp_solvers.fp_fvm import FPFVMSolver

    source = lambda t, x: _at(x, _near(0.4), NAN)  # noqa: E731
    if solver == "FPFVMSolver":
        solve = lambda: FPFVMSolver(_problem()).solve_fp_system(  # noqa: E731
            np.ones(11), drift_field=np.tile(0.1 * _X, (6, 1)), source_term=source
        )
    else:
        solve = lambda: FPFDMSolver(_problem()).solve_fp_system(  # noqa: E731
            np.ones(11),
            drift_field=lambda t, x, m: 0.1 * np.ones_like(np.asarray(m, dtype=float)),
            source_term=source,
            show_progress=False,
        )
    with pytest.raises(NonFiniteReadError, match=names):
        solve()


@pytest.mark.filterwarnings("ignore::scipy.sparse.linalg.MatrixRankWarning")
def test_a_volatility_nan_at_one_interior_node_takes_the_per_point_branch():
    """On the explicit arm (a callable drift), `np.ptp(sigma) > 1e-12` read an array holding a NaN as uniform, and the
    scalar branch kept entry 0 alone: a callable volatility NaN at one interior node was dropped. It now takes the
    per-point branch, where it enters its own column, and is named there (#2975)."""
    from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver

    def volatility(t, x):
        a = np.asarray(x[0] if isinstance(x, tuple) else x, dtype=float).ravel()
        return np.where(np.isclose(a, 0.5) & (abs(float(t) - 0.2) < 1e-9), NAN, 0.2)

    with pytest.raises(
        NonFiniteReadError,
        match=r"FPFDMSolver: volatility "
        + OWN
        + r": 1 of 11 values, the first nan, at time step 3 of 5, from t=0\.2 to t=0\.3, "
        r"node \(5,\), where it entered the step",
    ):
        FPFDMSolver(_problem()).solve_fp_system(
            np.ones(11),
            drift_field=lambda t, x, m: 0.1 * np.ones_like(np.asarray(m, dtype=float)),
            volatility=volatility,
            show_progress=False,
        )


def test_a_dual_hamiltonians_dp_is_non_finite_where_its_value_is():
    """The dual of a Lagrangian searches a grid of controls: its value, `np.max`, is NaN when the Lagrangian is NaN at a
    control the search reads, and its dp took `np.argmax`'s first NaN as the maximiser, a finite grid point. A step that
    read that dp, an FP's velocity, went on with it. dp is now NaN where the value is (#2975)."""
    from mfgarchon.core.hamiltonian import DualHamiltonian, SeparableLagrangian

    class _NaNAtATrial(SeparableLagrangian):
        def __call__(self, t, x, alpha, m):
            out = np.array(super().__call__(t=t, x=x, alpha=alpha, m=m), dtype=float, copy=True)
            a = np.asarray(alpha, dtype=float)
            return np.where(a.max(axis=-1) > 5.0 if a.ndim else a > 5.0, NAN, out)

    dual = DualHamiltonian(_NaNAtATrial(QuadraticControlCost(control_cost=1.0)))
    x, p, m = (np.array([0.5]),), np.array([0.3]), 1.0
    assert np.isnan(dual(t=0.0, x=x, p=p, m=m))
    assert np.all(np.isnan(dual.dp(t=0.0, x=x, p=p, m=m)))


def test_the_network_fp_names_the_value_field_its_rates_are_read_from():
    """The door no longer reads potential_field, so a NaN in it reaches the rates the step reads from it: the value field
    at the node and its neighbours is named first, ahead of the rates computed from it (#2975)."""
    from mfgarchon.alg.numerical.network_solvers.fp_network import FPNetworkSolver
    from mfgarchon.extensions.topology import NetworkMFGComponents, NetworkMFGProblem
    from mfgarchon.geometry.graph.network_geometry import GridNetwork

    network = GridNetwork(width=3, height=3)
    network.create_network()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        problem = NetworkMFGProblem(geometry=network, T=0.5, Nt=5, components=NetworkMFGComponents())
    u = np.tile(np.linspace(0.0, 1.0, network.num_nodes), (6, 1))
    u[0, 4] = NAN
    with pytest.raises(
        NonFiniteReadError,
        match=r"FPNetworkSolver: potential_field at node \d+ and its neighbours "
        + OWN
        + r": .*, the first nan, at t=0, ",
    ):
        FPNetworkSolver(problem, diffusion_coefficient=0.1).solve_fp_system(
            np.ones(network.num_nodes), potential_field=u
        )


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_the_nd_value_iteration_names_the_read_at_its_start_state():
    """HJB-FDM's n-D value iteration (solver_type="fixed_point") fails on a NaN potential at one node and names it at the
    step's start state, as Newton does; at 4ca7c9e0 it raised ConvergenceError after its budget (#2975)."""
    from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver

    def potential(t, x):
        return _at(
            x, lambda a: np.isclose(a, 0.8) & np.isclose(np.asarray(x[1], dtype=float), 0.2), NAN, lambda a: 0.1 * a
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        problem = MFGProblem(
            model=Model(
                hamiltonian=SeparableHamiltonian(QuadraticControlCost(control_cost=1.0), potential=potential),
                volatility=0.2,
            ),
            domain=TensorProductGrid(
                bounds=[(0.0, 1.0)] * 2, Nx_points=[6, 6], boundary_conditions=bnd.no_flux_bc(dimension=2)
            ),
            conditions=Conditions(
                m_initial=lambda x: 1.0 + 0.5 * np.asarray(x[0], dtype=float),
                u_terminal=lambda x: np.sin(np.pi * np.asarray(x[0], dtype=float)),
                T=0.1,
            ),
            Nt=5,
        )
    terminal = np.sin(np.pi * np.linspace(0.0, 1.0, 6))[:, None] * np.ones((6, 6))
    with pytest.raises(
        NonFiniteReadError,
        match=r"HJBFDMSolver's fixed-point step: SeparableHamiltonian's potential " + OWN + r": .*, x=\[0\.8, 0\.2\]",
    ):
        HJBFDMSolver(problem, solver_type="fixed_point", max_newton_iterations=400).solve_hjb_system(
            M_density=np.ones((6, 6, 6)), U_terminal=terminal, U_coupling_prev=np.zeros((6, 6, 6))
        )
