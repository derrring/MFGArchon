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


# --- Site 6: what every concrete door reads at its door, read from source. ---

_PACKAGE = pathlib.Path(mfgarchon.__file__).parent
_DOOR_NAMES = ("solve_hjb_system", "solve_fp_system")


def _named(node: ast.expr, name: str) -> bool:
    return (isinstance(node, ast.Name) and node.id == name) or (isinstance(node, ast.Attribute) and node.attr == name)


def _the_handed_row(name: str, value: ast.expr) -> bool:
    """Whether a door's read maps ``name`` to the argument handed under that name: the parameter itself, or the row as
    the solver holds it at level 0, built from it (`initial_density_as_held`, ruling 3 as amended after review 2), in a
    conditional that passes None through, ``None if name is None else ...``. A value such as None, another array, or a
    conditional on another argument (the weak-form alias read only where ``U_terminal`` is None, #2981's review 3)
    maps the name to something else, at least on one branch."""
    if isinstance(value, ast.IfExp):
        test = value.test
        passes_none_through = (
            isinstance(test, ast.Compare)
            and _named(test.left, name)
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Is)
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value is None
            and isinstance(value.body, ast.Constant)
            and value.body.value is None
        )
        return passes_none_through and _the_handed_row(name, value.orelse)
    if isinstance(value, ast.Call) and _named(value.func, "initial_density_as_held"):
        return bool(value.args) and _named(value.args[0], name)
    return _named(value, name)


def _door_reads(function: ast.FunctionDef) -> set[str]:
    """The names a door's read at "its door" maps: each name whose value is the argument handed under it
    (`_the_handed_row`), so a door that maps the name to something else does not read it. The read is the door's
    statement that calls the owner with "its door", wherever the door makes it; a door with none reads nothing. That
    each door refuses a NaN row before its work is the behavioural test's, which calls every door
    (`test_a_door_refuses_a_non_finite_solution_row_by_name`)."""
    calls = [
        stmt.value
        for stmt in function.body
        if isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and _named(stmt.value.func, "refuse_a_non_finite_read")
        and len(stmt.value.args) >= 3
        and isinstance(stmt.value.args[1], ast.Constant)
        and stmt.value.args[1].value == "its door"
    ]
    if not calls:
        return set()
    mapping = calls[0].args[2]
    assert isinstance(mapping, ast.Dict)
    return {
        key.value
        for key, value in zip(mapping.keys, mapping.values, strict=True)
        if isinstance(key, ast.Constant) and _the_handed_row(key.value, value)
    }


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
    """The control: a walk that lost a directory or a class would pass the door-read test vacuously."""
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


def _solution_rows(door: str) -> set[str]:
    if door.endswith(".solve_fp_system"):
        return {"M_initial", "initial_particles"} if "FPParticleSolver" in door else {"M_initial"}
    return {"U_terminal"}


_READ_SHAPES = {
    "the parameter": ('{"M_initial": M_initial}', {"M_initial"}),
    "the row as held": (
        '{"M_initial": None if M_initial is None else initial_density_as_held(M_initial, bc, grid, 1)}',
        {"M_initial"},
    ),
    "None": ('{"M_initial": None}', set()),
    "another array": ('{"M_initial": m}', set()),
    "a conditional on another argument": ('{"M_initial": None if m is not None else M_initial}', set()),
}


@pytest.mark.parametrize(("mapping", "reads"), list(_READ_SHAPES.values()), ids=list(_READ_SHAPES))
def test_the_door_read_predicate_tells_the_handed_row_from_another_value(mapping, reads):
    """The control for the test below: a door that maps its row's name to None, or to another array, reads nothing
    (review 2 of #2981 found a key-only reading passed those)."""
    tree = ast.parse(
        f'class A:\n    def solve_fp_system(self, M_initial):\n        refuse_a_non_finite_read("S", "its door", {mapping})\n'
    )
    assert _door_reads(tree.body[0].body[0]) == reads


@pytest.mark.parametrize(
    ("body", "reads"),
    [
        ('log("S")\nrefuse_a_non_finite_read("S", "its door", {"M_initial": M_initial})', {"M_initial"}),
        ('refuse_a_non_finite_read("S", "step", {"M_initial": M_initial})', set()),
    ],
    ids=["a statement ahead of the read", "no read at its door"],
)
def test_the_door_read_is_found_wherever_the_door_makes_it(body, reads):
    """The control for the test below on where the read sits: a statement ahead of it changes nothing, and a door with no
    read at its door reads nothing (#2981's review 3; the approver's ruling of 2026-11-19)."""
    indented = "".join(f"        {line}\n" for line in body.splitlines())
    tree = ast.parse(f"class A:\n    def solve_fp_system(self, M_initial):\n{indented}")
    assert _door_reads(tree.body[0].body[0]) == reads


@pytest.mark.parametrize("door", sorted(_DOORS))
def test_each_door_reads_the_solution_row_it_is_handed_and_nothing_else(door):
    """A door reads the solution's own row, its terminal value or its initial density; every other array it is handed is
    judged by the step that reads it, where it entered (#2975, the approver's ruling 3 of 2026-11-17)."""
    assert _door_reads(_DOORS[door]) == _solution_rows(door), door


def test_the_weak_form_door_reads_the_deprecated_alias_only_where_it_is_the_row():
    """The weak-form HJB takes ``U_final_condition_at_T`` only where ``U_terminal`` is None. Beside a finite
    ``U_terminal`` a NaN alias is discarded and solves bit for bit as the twin, as at 4ca7c9e0; alone it is the row, and
    refused at the door (#2981's review 3)."""
    from mfgarchon.alg.numerical.meshless_galerkin.hjb_solver import MeshlessGalerkinHJBSolver

    problem = _problem()
    points = np.asarray(problem.geometry.get_spatial_grid()).reshape(-1, 1)
    x = points.ravel()
    terminal = np.sin(np.pi * x)
    alias = terminal.copy()
    alias[3] = NAN

    def solve(**handed):
        solver = MeshlessGalerkinHJBSolver(problem, collocation_points=points, delta=0.35)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return solver.solve_hjb_system(
                M_density=np.tile(1.0 + 0.5 * x, (6, 1)), U_coupling_prev=np.tile(terminal, (6, 1)), **handed
            )

    assert np.array_equal(solve(U_terminal=terminal, U_final_condition_at_T=alias), solve(U_terminal=terminal))
    with pytest.raises(NonFiniteReadError, match=r"\.solve_hjb_system: U_final_condition_at_T " + OWN + r": 1 of 11 "):
        solve(U_final_condition_at_T=alias)


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
            + r": .*, the first nan, at time step \d+ of 5, .*, face \(\d+,\), formed from potential_field at nodes "
            r"\(\d+,\) and \(\d+,\), handed as [-0-9.e]+ and [-0-9.e]+, where it entered",
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
    twin's (max |M - M_twin| 1.66) with no error. The switch is part of the step, so the step's failure path names it
    where it entered (review 2 of #2981). A coupled solve refuses a capped exit, so the FP is solved alone."""
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
        match=r"FPFDMSolver: source_term " + OWN + r": 1 of 11 values, the first nan, at time step 1 of 5, from t=0 to "
        r"t=0\.1, node \(10,\), where it entered the step",
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
        match=r"FPNetworkSolver: the transition rate out of node 1 into node 4 "
        + OWN
        + r": 1 of 1 values, the first nan, at t=0 ",
    ):
        FPNetworkSolver(problem, diffusion_coefficient=0.1).solve_fp_system(
            np.ones(network.num_nodes), potential_field=u
        )


def test_the_network_fp_solves_a_non_finite_value_at_a_node_with_no_edges_as_its_twin():
    """A node with no edges has no rate out of it or into it, so no step reads its value: a NaN there solves bit for bit
    as the finite twin, as at 4ca7c9e0 (#2981's review 3), at the first level and the last."""
    from mfgarchon.alg.numerical.network_solvers.fp_network import FPNetworkSolver
    from mfgarchon.extensions.topology import NetworkMFGComponents, NetworkMFGProblem
    from mfgarchon.geometry.graph.network_geometry import CustomNetwork

    adjacency = np.zeros((5, 5))
    for i, j in ((0, 1), (1, 2), (2, 3)):
        adjacency[i, j] = adjacency[j, i] = 1.0
    network = CustomNetwork(adjacency)

    def solve(row):
        problem = NetworkMFGProblem(geometry=network, T=0.5, Nt=5, components=NetworkMFGComponents())
        u = np.tile(np.linspace(0.0, 1.0, 5), (6, 1))
        if row is not None:
            u[row, 4] = NAN
        return FPNetworkSolver(problem, diffusion_coefficient=0.1).solve_fp_system(np.ones(5), potential_field=u)

    twin = solve(None)
    for row in (0, 5):
        assert np.array_equal(solve(row), twin)


# --- A door reads the solution's row it is handed; each other array is judged by the step that reads it. ---


def _door_calls():
    """Each concrete door, built on a small problem of its kind, with what it is handed under the names it takes."""
    from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
    from mfgarchon.alg.numerical.fp_solvers.fp_fvm import FPFVMSolver
    from mfgarchon.alg.numerical.fp_solvers.fp_particle import FPParticleSolver
    from mfgarchon.alg.numerical.fp_solvers.fp_semi_lagrangian_adjoint import FPSLSolver
    from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver, HJBSemiLagrangianSolver
    from mfgarchon.alg.numerical.hjb_solvers.hjb_gfdm import HJBGFDMSolver
    from mfgarchon.alg.numerical.hjb_solvers.hjb_howard import HJBHowardSolver
    from mfgarchon.alg.numerical.hjb_solvers.hjb_weno import HJBWENOSolver
    from mfgarchon.alg.numerical.meshless_galerkin.fp_solver import MeshlessGalerkinFPSolver
    from mfgarchon.alg.numerical.meshless_galerkin.hjb_solver import MeshlessGalerkinHJBSolver
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

    def howard():
        problem = _problem()
        points = problem.geometry.get_spatial_grid().reshape(-1, 1)
        return HJBHowardSolver(
            problem,
            stencil_provider=HJBGFDMSolver(problem, collocation_points=points, delta=0.3),
            alpha_star=lambda t_idx, x, p, m: -p,
            max_iter=15,
        )

    def meshless(solver):
        problem = _problem()
        return solver(problem, collocation_points=problem.geometry.get_spatial_grid().reshape(-1, 1), delta=0.35)

    hjb = {"M_density": np.ones((6, 11)), "U_terminal": np.zeros(11), "U_coupling_prev": np.zeros((6, 11))}
    fp = {"M_initial": np.ones(11), "potential_field": np.zeros((6, 11))}
    return {
        "HJBFDMSolver": (lambda: HJBFDMSolver(_problem()), "solve_hjb_system", hjb),
        "HJBSemiLagrangianSolver": (lambda: HJBSemiLagrangianSolver(_problem()), "solve_hjb_system", hjb),
        "HJBWENOSolver": (lambda: HJBWENOSolver(_problem()), "solve_hjb_system", hjb),
        "HJBGFDMSolver": (gfdm, "solve_hjb_system", hjb),
        "HJBHowardSolver": (
            howard,
            "solve_hjb_system",
            {"M_density": hjb["M_density"], "U_terminal": hjb["U_terminal"]},
        ),
        "MeshlessGalerkinHJBSolver": (lambda: meshless(MeshlessGalerkinHJBSolver), "solve_hjb_system", hjb),
        "MeshlessGalerkinFPSolver": (lambda: meshless(MeshlessGalerkinFPSolver), "solve_fp_system", fp),
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
    "HJBHowardSolver",
    "MeshlessGalerkinHJBSolver",
    "MeshlessGalerkinFPSolver",
    "NetworkHJBSolver",
    "NetworkPolicyIterationHJBSolver",
    "FPFDMSolver",
    "FPFVMSolver",
    "FPParticleSolver",
    "FPSLSolver",
    "FPNetworkSolver",
]


@pytest.mark.filterwarnings("ignore:HJBGFDMSolver:UserWarning")
@pytest.mark.filterwarnings("ignore:HJBHowardSolver:UserWarning")
@pytest.mark.parametrize("solver", _CALLED_DOORS)
def test_a_door_refuses_a_non_finite_solution_row_by_name(solver):
    """At 4ca7c9e0 the HJB doors returned a non-finite U for a NaN U_terminal, and the FP doors blamed the CFL condition
    (#2976, folded here). Howard's and the weak forms' doors are called too, through the meshless solvers, which take
    the weak forms' doors as they are (review 2 of #2981: a door mapping its row to None passed the AST tests alone)."""
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


class _RaisesWhereSteepAt04(SeparableHamiltonian):
    """An own value that raises a plain ValueError at t = 0.4 where |p| > 2, a state only the step's trials reach."""

    def __call__(self, t, x, p, m):
        if abs(float(np.ravel(t)[0]) - 0.4) < 1e-9 and np.max(np.abs(np.asarray(p, dtype=float))) > 2.0:
            raise ValueError("_RaisesWhereSteepAt04: |p| > 2 is outside its domain")
        return super().__call__(t=t, x=x, p=p, m=m)


class _RaisesAtUnitDensityAt04(SeparableHamiltonian):
    """An own value that raises a plain ValueError at t = 0.4 where m == 1, the density design call B's test substitutes
    for a non-finite one."""

    def __call__(self, t, x, p, m):
        if abs(float(np.ravel(t)[0]) - 0.4) < 1e-9 and np.any(np.asarray(m, dtype=float) == 1.0):
            raise ValueError("_RaisesAtUnitDensityAt04: m == 1 is outside its domain")
        return super().__call__(t=t, x=x, p=p, m=m)


@pytest.mark.parametrize("policy", ["raise", "record"])
def test_a_naming_that_raises_something_else_leaves_the_steps_own_failure(policy):
    """Only a NonFiniteReadError leaves a failure-path naming (#2981's review 3). The n-D Newton step's start-state
    naming re-evaluates H; an H that raises a ValueError there let it out, where 4ca7c9e0 gave #2644's ConvergenceError,
    which stands again."""
    from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
    from mfgarchon.utils.exceptions import ConvergenceError

    problem = _problem(_RaisesWhereSteepAt04(QuadraticControlCost(control_cost=1.0)), dimension=2)
    x = np.linspace(0.0, 1.0, 6)
    terminal = np.outer(np.sin(np.pi * x), np.sin(np.pi * x))
    with pytest.raises(ConvergenceError, match="Newton solver failed: _RaisesWhereSteepAt04"):
        HJBFDMSolver(problem, on_inner_failure=policy).solve_hjb_system(
            np.ones((6, 6, 6)), terminal, np.tile(terminal, (6, 1, 1))
        )


def test_a_density_test_whose_substitute_raises_names_nothing_and_the_order_goes_on():
    """Design call B's density test substitutes 1.0 for a non-finite density; an H that raises there makes the test
    inconclusive, so it names nothing and the order goes on to the coupling the density entered (#2981's review 3).
    3e283757 let the ValueError out; 4ca7c9e0 returned a non-finite U without an error."""
    from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver

    x = np.linspace(0.0, 1.0, 11)
    density = np.tile(0.5 + 0.3 * x, (6, 1))
    density[:, 4] = NAN
    problem = _problem(
        _RaisesAtUnitDensityAt04(QuadraticControlCost(control_cost=1.0), coupling=lambda m: 0.5 * np.asarray(m))
    )
    with pytest.raises(NonFiniteReadError, match=r"_RaisesAtUnitDensityAt04's coupling " + OWN + r": .*, m=nan "):
        HJBFDMSolver(problem).solve_hjb_system(density, np.sin(np.pi * x), np.tile(np.sin(np.pi * x), (6, 1)))


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
def test_fp_fdm_names_a_value_field_that_entered_its_step_by_its_argument():
    """U handed as `potential_field` (the route `resolve_fp_drift_kwargs` takes for the quadratic separable class) is
    read at t_k by step k. A NaN at one interior node of row 1 enters step 2, and is named by the argument it arrived
    through, at its node (#2975)."""
    from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver

    with pytest.raises(
        NonFiniteReadError,
        match=r"FPFDMSolver: potential_field "
        + OWN
        + r": 1 of 11 values, the first nan, at time step 2 of 5, from t=0\.1 to t=0\.2, node \(5,\), where it "
        r"entered the step",
    ):
        FPFDMSolver(_problem()).solve_fp_system(
            np.ones(11), potential_field=_nan_at(np.zeros((6, 11)), (1, 5)), show_progress=False
        )


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


def test_the_network_fp_names_the_value_field_its_rates_are_read_from():
    """The door no longer reads potential_field, so a NaN in it reaches the rates the step reads from it: the value field
    is named first, ahead of the rates computed from it, by its own node and the rate that reads it. Node 1's rates are
    the first that read node 4, so the two nodes differ (review 2 of #2981)."""
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
        match=r"FPNetworkSolver: potential_field at node 4, read for the rate out of node 1 into node 4 "
        + OWN
        + r": 1 of 1 values, the first nan, at t=0 ",
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


# --- Review 2 of #2981: the pins its mutants showed missing, and F0, F1, F2. Each docstring names what it pins. ---


class _DpReadsM(SeparableHamiltonian):
    """H's value does not read m; its dp is dp * (m / m), NaN where m is NaN, so only the Jacobian reads the density."""

    def dp(self, t, x, p, m):
        out = np.array(super().dp(t=t, x=x, p=p, m=m), dtype=float, copy=True)
        fac = np.asarray(m, dtype=float) / np.asarray(m, dtype=float)
        return out * (fac.reshape(fac.shape + (1,) * (out.ndim - fac.ndim)) if out.ndim > fac.ndim else fac)


def _at_point(x, point):
    hit = np.ones(np.shape(np.asarray(x[0], dtype=float)), dtype=bool)
    for k, c in enumerate(point):
        hit &= np.isclose(np.asarray(x[k], dtype=float), c, rtol=0, atol=1e-9)
    return hit


# The step fails on the NaN before its naming runs, as it must: Newton's line search and its singular Jacobian warn.
@pytest.mark.filterwarnings("ignore:Newton stopped:RuntimeWarning")
@pytest.mark.filterwarnings("ignore::scipy.sparse.linalg.MatrixRankWarning")
def test_the_nd_newton_step_names_a_callable_volatility():
    """n-D HJB-FDM's naming hands the volatility to the start state's terms first. A run with it left out names the
    Hamiltonian's value instead (review 2's MK); at 4ca7c9e0 CoefficientField refused it at its read."""
    volatility = lambda t, x: np.where(_at_point(x, (0.6, 0.4)) & (abs(float(t) - 0.2) < 1e-9), NAN, 0.2)  # noqa: E731
    with pytest.raises(
        NonFiniteReadError,
        match=r"HJBFDMSolver's Newton step: volatility " + OWN + r": 1 of 1 values, the first nan, at t=0\.2, x=\[0\.6",
    ):
        _problem(dimension=2, model_kw={"volatility": volatility}).solve(max_iterations=20, verbose=False)


def test_the_jacobian_names_a_density_only_its_dp_reads():
    """The Jacobian's naming tests the handed density before H's dp, for an H whose value ignores m. Without that test it
    names H's dp (review 2's MG); at 4ca7c9e0 the solve returned a non-finite U."""
    from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver

    with pytest.raises(
        NonFiniteReadError,
        match=r"HJBFDMSolver's Newton step: M_density "
        + OWN
        + r": 1 of 1 values, the first nan, at t=0\.4, x=\[0\.4\],"
        r".*\(of the 1 non-finite row\(s\) the Jacobian keeps\)",
    ):
        HJBFDMSolver(_problem(_DpReadsM(QuadraticControlCost(control_cost=1.0)))).solve_hjb_system(
            **_hjb_door_args(M_density=_nan_at(np.tile(1.0 + 0.5 * _X, (6, 1)), (slice(None), 4)))
        )


_SL_TERM = 0.2 * np.sin(np.pi * np.linspace(0.0, 1.0, 6))[:, None] * np.ones((6, 6))


@pytest.mark.parametrize(
    ("hamiltonian", "where"),
    [
        ({"coupling": lambda m: 0.5 * np.asarray(m, dtype=float)}, r"value update: M_density " + OWN),
        (_DpReadsM(QuadraticControlCost(control_cost=1.0)), r"CFL number: M_density " + OWN),
    ],
    ids=["value update, a coupling", "CFL number, an H whose dp reads m"],
)
def test_hjb_sl_names_a_density_where_it_enters(hamiltonian, where):
    """HJB-SL's value update and its CFL number each test the handed density before the terms read from it. Without
    those tests they name the coupling, or H's dp (review 2's MH and MI); at 4ca7c9e0 the first returned a finite wrong
    U and the second raised "cfl_substeps got a non-finite CFL number"."""
    from mfgarchon.alg.numerical.hjb_solvers import HJBSemiLagrangianSolver

    M = _nan_at(np.ones((6, 6, 6)), (slice(None), 2, 3))
    with pytest.raises(NonFiniteReadError, match=rf"HJBSemiLagrangianSolver's {where}: 1 of 1 values, the first nan"):
        HJBSemiLagrangianSolver(_problem(hamiltonian, dimension=2)).solve_hjb_system(
            M_density=M, U_terminal=_SL_TERM, U_coupling_prev=np.tile(_SL_TERM, (6, 1, 1))
        )


def test_hjb_fdm_names_the_other_populations_row():
    """The residual's naming reads the other population's row, beside the own density. Without it the coupling is named
    (review 2's MQ); at 4ca7c9e0 the solve raised a ConvergenceError."""
    from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver

    def coupling(m, cross_density):
        return 2.0 * np.asarray(cross_density[1], dtype=float) if len(cross_density) > 1 else 0.0 * np.asarray(m)

    hamiltonian = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=coupling,
        coupling_dm=lambda m: np.zeros_like(np.asarray(m, dtype=float)),
        population_index=0,
    )
    stack = np.concatenate([np.tile(1.0 + 0.5 * _X, (6, 1)), np.tile(0.5 + 0.0 * _X, (6, 1))], axis=1)
    with pytest.raises(
        NonFiniteReadError,
        match=r"HJBFDMSolver's Newton step: cross_density "
        + OWN
        + r": 1 of 1 values, the first nan, at t=0\.4, x=\[0\.4\]",
    ):
        HJBFDMSolver(_problem(hamiltonian)).solve_hjb_system(
            **_hjb_door_args(), cross_density=_nan_at(stack, (slice(None), 15))
        )


def _nan_near(c, finite):
    return lambda t, x: np.where(np.isclose(np.asarray(x[0], dtype=float), c, rtol=0, atol=1e-9), NAN, finite)


@pytest.mark.parametrize(
    ("kwargs", "named"),
    [
        ({"model_kw": {"volatility": _nan_near(0.3, 0.2)}}, "volatility"),
        ({"hamiltonian": {"potential": _nan_near(0.3, 0.1)}}, "SeparableHamiltonian's potential"),
    ],
    ids=["the volatility before the source", "H's terms before the source"],
)
def test_hjb_fdm_names_in_its_stated_order(kwargs, named):
    """Two reads non-finite at one node: the residual names them in the order the body states, the volatility first,
    then H's terms, the source last. Named in another order, the source comes first (review 2's MA and MB)."""
    with pytest.raises(NonFiniteReadError, match=rf"HJBFDMSolver's Newton step: {named} {OWN}: 1 of 1 values"):
        _problem(source_term_hjb=_nan_near(0.3, 0.0), **kwargs).solve(max_iterations=20, verbose=False)


def _source_nan_at(nodes):
    return lambda t, x: np.where(np.isin(np.round(np.asarray(x[0], dtype=float), 9), np.round(_X[nodes], 9)), NAN, 0.0)


@pytest.mark.parametrize(
    ("nodes", "bc", "named"),
    [([3, 7], None, 3), ([5, 10], "dirichlet", 5)],
    ids=["two that enter", "one that enters before one a Dirichlet row discards"],
)
def test_the_fp_failure_path_names_the_first_read_that_entered(nodes, bc, named):
    """The bisection restores the entries in reading order and names the first whose restoration makes the step non-
    finite. A bisection that never restores names the last non-finite entry: node 7, or the discarded node 10 as if it
    had entered (review 2's mutant BISECT_NO_RESTORE)."""
    from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver

    problem = _problem(bc=bnd.dirichlet_bc(value=0.0, dimension=1) if bc else None)
    with pytest.raises(
        NonFiniteReadError,
        match=rf"FPFDMSolver: source_term {OWN}: 2 of 11 values, the first nan, at time step 1 of 5, .*, node \({named},\), "
        r"where it entered the step",
    ):
        FPFDMSolver(problem).solve_fp_system(
            np.ones(11), potential_field=np.zeros((6, 11)), source_term=_source_nan_at(nodes), show_progress=False
        )


# The overflow from finite reads is the input: it warns where the step divides.
@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
def test_a_computed_failure_beside_a_discarded_nan_names_nothing():
    """A NaN only at a Dirichlet node, which the step discards, beside an overflow from finite reads: the failure is
    computed, so the step's own message stands. Without the computed-failure branch the bisection names the discarded
    source as having entered (review 2's mutant NO_COMPUTED_FAILURE_BRANCH)."""
    from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver

    problem = _problem(bc=bnd.dirichlet_bc(value=0.0, dimension=1))
    with pytest.raises(ValueError, match=r"FP solver produced NaN/Inf at timestep 1/5") as caught:
        FPFDMSolver(problem).solve_fp_system(
            np.full(11, 1e308), drift_field=np.full((6, 11), 1e3), source_term=_source_nan_at([0]), show_progress=False
        )
    assert not isinstance(caught.value, NonFiniteReadError)


def _drift_nan_at(node):
    def drift(t, x, m):
        out = 0.1 * np.ones_like(np.asarray(m, dtype=float))
        if node is not None and abs(float(t) - 0.2) < 1e-9:
            out = out.copy()
            out[node] = NAN
        return out

    return drift


def test_a_drift_callable_is_named_where_it_enters_and_not_read_where_it_is_discarded():
    """A drift callable's output is handed to the step as read (#2975): named where it enters, at node 5, and not judged
    at the periodic repeated endpoint, which the step discards. Refused where CoefficientField reads it, as at
    4ca7c9e0, the repeated endpoint is refused too (review 2's mutant DRIFT_CALLABLE_READ_AGAIN)."""
    from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver

    with pytest.raises(
        NonFiniteReadError,
        match=r"FPFDMSolver: the drift callable " + OWN + r": 1 of 11 values, the first nan, at time step 3 of 5, .*, "
        r"node \(5,\), where it entered the step",
    ):
        FPFDMSolver(_problem()).solve_fp_system(np.ones(11), drift_field=_drift_nan_at(5), show_progress=False)
    periodic = _problem(bc=bnd.periodic_bc(dimension=1, domain_bounds=np.array([[0.0, 1.0]])))
    m0 = 1.0 + 0.5 * np.sin(2 * np.pi * _X)
    twin = FPFDMSolver(periodic).solve_fp_system(m0.copy(), drift_field=_drift_nan_at(None), show_progress=False)
    nan = FPFDMSolver(periodic).solve_fp_system(m0.copy(), drift_field=_drift_nan_at(10), show_progress=False)
    assert np.array_equal(nan, twin)


def _tensor_problem(bc):
    return MFGProblem(
        model=Model(hamiltonian=SeparableHamiltonian(QuadraticControlCost(control_cost=1.0)), volatility=0.2),
        domain=TensorProductGrid(bounds=[(0.0, 1.0)] * 2, Nx_points=[7, 7], boundary_conditions=bc),
        conditions=Conditions(m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0, T=0.05),
        Nt=4,
    )


def _tensor_nan_at(point):
    def tensor(t, x, m):
        S = np.array([[0.2, 0.05], [0.05, 0.25]])
        if point is not None and np.allclose(x, point):
            S = S.copy()
            S[0, 0] = NAN
        return S

    return tensor


def _solve_tensor(bc, point):
    from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver

    xs = np.linspace(0.0, 1.0, 7)
    U = np.tile((0.1 * np.add.outer(xs, xs))[None], (5, 1, 1))
    return FPFDMSolver(_tensor_problem(bc)).solve_fp_system(
        np.ones((7, 7)),
        potential_field=U,
        volatility=_tensor_nan_at(point),
        volatility_kind="tensor",
        show_progress=False,
    )


def test_a_callable_tensor_is_named_as_non_finite_where_it_enters_and_not_where_it_is_discarded():
    """A non-finite tensor entry is named as non-finite by the step, not under the PSD message its meaningless
    eigenvalues gave (review 2's F3), and the periodic repeated endpoint, which the step discards, is not judged.
    Refused where it is read, as at 4ca7c9e0, the repeated endpoint is refused too (review 2's mutant
    TENSOR_READ_AGAIN)."""
    with pytest.raises(
        NonFiniteReadError,
        match=r"FPFDMSolver: the volatility tensor "
        + OWN
        + r": 1 of 196 values, the first nan, .*, entry \(3, 3, 0, 0\) "
        r"of its \(7, 7, 2, 2\), where it entered the step",
    ):
        _solve_tensor(bnd.no_flux_bc(dimension=2), (0.5, 0.5))
    periodic = bnd.periodic_bc(dimension=2)
    assert np.array_equal(_solve_tensor(periodic, (1.0, 0.5)), _solve_tensor(periodic, None))


@pytest.mark.parametrize(
    ("matrix", "entry"),
    [([[0.2, NAN], [0.05, 0.25]], "3, 3, 0, 1"), ([[NAN, 0.9], [0.9, 0.25]], "3, 3, 0, 0")],
    ids=["asymmetric if replaced alone", "indefinite if replaced alone"],
)
def test_a_non_finite_tensor_entry_is_named_where_its_matrix_replaced_entry_by_entry_is_refused(matrix, entry):
    """The failure path set the non-finite entry alone to the identity's, so its re-run read a matrix the step refuses,
    asymmetric or not PSD, and that refusal left in place of the naming (#2981's review 3). It now sets the whole matrix
    at that node to the identity."""
    from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver

    def tensor(t, x, m):
        return np.array(matrix) if np.allclose(x, (0.5, 0.5)) else np.array([[0.2, 0.05], [0.05, 0.25]])

    xs = np.linspace(0.0, 1.0, 7)
    with pytest.raises(NonFiniteReadError, match=rf"the volatility tensor {OWN}: .*, entry \({entry}\) of its"):
        FPFDMSolver(_tensor_problem(bnd.no_flux_bc(dimension=2))).solve_fp_system(
            np.ones((7, 7)),
            potential_field=np.tile((0.1 * np.add.outer(xs, xs))[None], (5, 1, 1)),
            volatility=tensor,
            volatility_kind="tensor",
            show_progress=False,
        )


def _volatility_nan_at(node, varying):
    def volatility(t, x):
        a = np.asarray(x[0], dtype=float)
        out = 0.2 + 0.3 * a if varying else np.full(a.shape, 0.2)
        return out if node is None else np.where(np.isclose(a, node, rtol=0, atol=1e-9), NAN, out)

    return volatility


_U_FIELD = np.array([np.sin(np.pi * _X) * (1 + 0.1 * k) for k in range(6)])


@pytest.mark.filterwarnings("ignore::scipy.sparse.linalg.MatrixRankWarning")
@pytest.mark.parametrize("varying", [True, False], ids=["varying twin", "uniform twin"])
def test_the_implicit_step_names_a_volatility_nan_at_a_dirichlet_node_whatever_its_twin(varying):
    """F1: the implicit step's choice of the Ito branch reads a non-finite entry as varying, as the explicit step does,
    so a NaN at a Dirichlet node a kept row reads (row 1 reads node 0) reaches the step through D * g and is named,
    under a uniform twin as under a varying one. With `ptp > 1e-12`, a NaN chose the row-scaled branch, which drops it: a varying twin's solve came back finite
    and unlike its twin (max |dM| 0.142), with no error (review 2's F1)."""
    from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver

    problem = _problem(
        model_kw={"volatility": _volatility_nan_at(0.0, varying)}, bc=bnd.dirichlet_bc(value=0.0, dimension=1)
    )
    with pytest.raises(
        NonFiniteReadError,
        match=r"FPFDMSolver: volatility "
        + OWN
        + r": 1 of 11 values, the first nan, at time step 1 of 5, .*, node \(0,\), "
        r"where it entered the step",
    ):
        FPFDMSolver(problem).solve_fp_system(1.0 + 0.5 * _X, potential_field=_U_FIELD, show_progress=False)


def _periodic_endpoint_solve(nan, varying):
    """FP-FDM's implicit step on a 1-D endpoint-inclusive periodic grid, the volatility NaN only at x = 1 (review 2's p3)."""
    from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver

    def volatility(t, x):
        a = np.asarray(x[0], dtype=float)
        out = 0.2 + 0.1 * np.sin(2 * np.pi * a) + 0.05 * a * (1 - a) if varying else np.full(a.shape, 0.2)
        return np.where(np.isclose(a, 1.0) & nan, NAN, out)

    periodic = bnd.periodic_bc(dimension=1, domain_bounds=np.array([[0.0, 1.0]]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        problem = MFGProblem(
            model=Model(
                hamiltonian=SeparableHamiltonian(QuadraticControlCost(control_cost=1.0)), volatility=volatility
            ),
            domain=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=periodic),
            conditions=Conditions(m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0, T=0.5),
            Nt=5,
        )
    U = np.array([0.1 * np.sin(2 * np.pi * _X) * (1 + 0.1 * k) for k in range(6)])
    return FPFDMSolver(problem).solve_fp_system(
        1.0 + 0.3 * np.cos(2 * np.pi * _X), potential_field=U, show_progress=False
    )


@pytest.mark.parametrize("varying", [True, False], ids=["varying twin", "uniform twin"])
def test_a_volatility_nan_at_the_periodic_repeated_endpoint_does_not_choose_the_branch(varying):
    """F1: the repeated endpoint holds the value of the node it repeats, so neither branch reads its own entry, and the
    branch choice does not read it either: its NaN solves as the twin, bit for bit. Read there, a NaN under a uniform
    twin takes the Ito branch while the twin keeps the row-scaled one, 2.2e-16 apart; with `ptp > 1e-12` over every
    node, a NaN there moved a varying twin onto the row-scaled branch (max |dM| 1.39, the value-level census)."""
    assert np.array_equal(_periodic_endpoint_solve(True, varying), _periodic_endpoint_solve(False, varying))


def _capped_bc(capped):
    from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions

    return BoundaryConditions(
        dimension=1,
        segments=[
            BCSegment(name="entry", bc_type=BCType.DIRICHLET, value=0.0, boundary="x_min"),
            BCSegment(
                name="exit",
                bc_type=BCType.DIRICHLET,
                value=0.0,
                boundary="x_max",
                **({"flux_capacity": 0.5} if capped else {}),
            ),
        ],
    )


@pytest.mark.filterwarnings("ignore::scipy.sparse.linalg.MatrixRankWarning")
def test_the_capped_switch_is_part_of_the_step_and_its_wall_row_is_named():
    """F2: the capped exit's switch reads the wall row at its node (sigma, U, the face velocities), not only the source,
    and `> cap` read a NaN there as under the cap: a NaN volatility at the capped node gave a finite M 1.655 from its
    twin's. The switch is part of the step, so the step's failure path names the read (review 2's F2)."""
    from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver

    x = np.linspace(0.0, 1.0, 21)
    problem = MFGProblem(
        model=Model(
            hamiltonian=SeparableHamiltonian(QuadraticControlCost(control_cost=1.0)),
            volatility=_volatility_nan_at(1.0, False),
        ),
        domain=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=bnd.no_flux_bc(dimension=1)),
        conditions=Conditions(m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0, T=0.5),
        Nt=20,
    )
    with pytest.raises(
        NonFiniteReadError,
        match=r"FPFDMSolver: volatility "
        + OWN
        + r": 1 of 21 values, the first nan, at time step 1 of 20, .*, node \(20,\), "
        r"where it entered the step",
    ):
        FPFDMSolver(problem, boundary_conditions=_capped_bc(True)).solve_fp_system(
            np.exp(0.5 * x), potential_field=np.broadcast_to(-0.5 * x, (21, 21)).copy(), show_progress=False
        )


def _capped_column(capped, nan):
    from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
    from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions

    nx, ny, nt = 11, 9, 10
    problem = MFGProblem(
        model=Model(hamiltonian=SeparableHamiltonian(QuadraticControlCost(control_cost=1.0)), volatility=0.8),
        domain=TensorProductGrid(
            bounds=[(0.0, 1.0)] * 2, Nx_points=[nx, ny], boundary_conditions=bnd.no_flux_bc(dimension=2)
        ),
        conditions=Conditions(m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0, T=0.5),
        Nt=nt,
    )
    bc = BoundaryConditions(
        dimension=2,
        segments=[
            BCSegment(
                name="exit",
                bc_type=BCType.DIRICHLET,
                value=0.0,
                boundary="x_max",
                **({"flux_capacity": 0.5} if capped else {}),
            ),
            BCSegment(name="walls", bc_type=BCType.NO_FLUX, boundary="x_min"),
            BCSegment(name="bottom", bc_type=BCType.NO_FLUX, boundary="y_min"),
            BCSegment(name="top", bc_type=BCType.NO_FLUX, boundary="y_max"),
        ],
    )
    v = np.zeros((nt + 1, 2, nx, ny))
    v[:, 0] = 2.0
    if nan:
        v[:, 1, nx - 1, ny // 2] = NAN
    x = np.linspace(0.0, 1.0, nx)[:, None] * np.ones((1, ny))
    return FPFDMSolver(problem, boundary_conditions=bc).solve_fp_system(
        np.exp(0.5 * x), drift_field=v, show_progress=False
    )


def test_a_tangential_velocity_along_a_capped_column_is_named_and_solves_as_its_twin_uncapped():
    """F2's 2-D case, present at 4ca7c9e0 too: a NaN tangential velocity on a face along the capped column enters only
    the capped nodes' wall rows. Capped, the switch read it as under the cap (max |dM| 4.52 at both shas); it is named
    now. Uncapped, the Dirichlet rows discard it, and the solve is its twin's."""
    with pytest.raises(
        NonFiniteReadError,
        match=r"FPFDMSolver: drift_field " + OWN + r": 1 of 198 values, the first nan, .*, "
        r"entry \(1, 10, 4\) of its \(2, 11, 9\), where it entered the step",
    ):
        _capped_column(True, True)
    assert np.array_equal(_capped_column(False, True), _capped_column(False, False))


def test_fp_fdm_door_does_not_judge_an_initial_density_entry_its_dirichlet_wall_overwrites():
    """F0, ruling 3 as amended: a 1-D BC whose segments are all Dirichlet holds level 0 at g at the wall node, so an
    M_initial entry there is read and discarded. The door reads the row as the step holds it, and the solve is the
    twin's, bit for bit, as at 4ca7c9e0, where a door reading the handed row refused it (review 2's F0)."""
    from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver

    problem = _problem(bc=bnd.dirichlet_bc(value=0.0, dimension=1))
    m0 = 1.0 + 0.5 * _X
    nan = FPFDMSolver(problem).solve_fp_system(_nan_at(m0, 0), potential_field=_U_FIELD, show_progress=False)
    twin = FPFDMSolver(problem).solve_fp_system(m0, potential_field=_U_FIELD, show_progress=False)
    assert np.array_equal(nan, twin)
    assert nan[0, 0] == 0.0


def test_cfl_substeps_names_a_non_finite_cfl_number():
    """`cfl_substeps` reads its CFL number through the owner before `math.ceil`, which raised "cannot convert float NaN to
    integer". It is reached through HJB-SL's stage-2 paths, so it is pinned where it is read (review 2's mutant
    CFL_SUBSTEPS_UNREAD)."""
    from mfgarchon.alg.numerical.hjb_solvers.hjb_sl_characteristics import cfl_substeps

    with pytest.raises(
        NonFiniteReadError,
        match=r"cfl_substeps: the CFL number of the foot's velocity "
        + OWN
        + r": 1 of 1 values, the first nan, at its sub-step count",
    ):
        cfl_substeps(NAN)


def test_the_particle_fps_door_refuses_non_finite_initial_particles_by_name():
    """The particle FP's door reads the initial particles it is handed beside M_initial (review 2's door value mutant
    mapped them to None and passed)."""
    from mfgarchon.alg.numerical.fp_solvers.fp_particle import FPParticleSolver

    particles = np.linspace(0.05, 0.95, 50).reshape(-1, 1)
    particles[7, 0] = NAN
    with pytest.raises(
        NonFiniteReadError,
        match=r"FPParticleSolver\.solve_fp_system: initial_particles "
        + OWN
        + r": 1 of 50 values, the first nan, at its door",
    ):
        FPParticleSolver(_problem(), num_particles=50).solve_fp_system(
            initial_particles=particles, potential_field=np.zeros((6, 11))
        )


class _NaNAtTheTrialsOnly(SeparableHamiltonian):
    """NaN at x = 0.3, t = 0, where p lies within (5e-7, 5e-5) of the start state's momentum there: the per-point
    Jacobian's trials (U +- eps moves p by about 1e-6 at dx 0.1), never the start state."""

    p_start = None
    active = True

    def __call__(self, t, x, p, m):
        out = super().__call__(t=t, x=x, p=p, m=m)
        xs = np.ravel(np.asarray(x[0] if isinstance(x, tuple) else x, dtype=float))
        if (
            not self.active
            or self.p_start is None
            or xs.size != 1
            or abs(float(np.ravel(t)[0])) > 1e-12
            or abs(xs[0] - 0.3) > 1e-9
        ):
            return out
        return NAN if 5e-7 < abs(float(np.ravel(p)[0]) - self.p_start) < 5e-5 else out


def _solve_with_trial_nans(active):
    from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
    from mfgarchon.alg.numerical.hjb_solvers.base_hjb import _compute_gradient_array_1d

    hamiltonian = _NaNAtTheTrialsOnly(QuadraticControlCost(control_cost=1.0))
    hamiltonian.active = active
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=bnd.no_flux_bc(dimension=1))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="initial density mass", category=UserWarning)
        problem = MFGProblem(
            model=Model(hamiltonian=hamiltonian, volatility=0.2),
            domain=grid,
            conditions=Conditions(
                m_initial=lambda x: 1.0 + 0.5 * np.asarray(x[0], dtype=float),
                u_terminal=lambda x: np.sin(np.pi * np.asarray(x[0], dtype=float)),
                T=0.1,
            ),
            Nt=1,
        )
    solver = HJBFDMSolver(problem, analytic_jacobian=False)
    U_T = np.sin(np.pi * _X)
    momentum = _compute_gradient_array_1d(U_T, 0.1, bc=solver.get_boundary_conditions(), upwind=True, time=0.0)
    problem.hamiltonian_class.p_start = float(momentum[3])
    return solver.solve_hjb_system(
        M_density=np.tile(1.0 + 0.5 * _X, (2, 1)), U_terminal=U_T, U_coupling_prev=np.tile(U_T, (2, 1))
    )


def test_a_nan_only_at_the_per_point_jacobians_trials_solves_as_at_4ca7c9e0():
    """Ruling 4's edge on the per-point path (design call C): the Jacobian's per-point fill is 0.0, as at 4ca7c9e0, so
    a NaN only at its trial momenta leaves a finite U, 6.99e-9 from the twin; at 4ca7c9e0 the U had the same sha256,
    11199f8e. A fill of NaN instead stops the Newton step with a ConvergenceError (review 2's M15, which the body had
    said no pin could tell from 0.0)."""
    U = _solve_with_trial_nans(True)
    assert np.all(np.isfinite(U))
    assert float(np.max(np.abs(U - _solve_with_trial_nans(False)))) < 1e-7


def _mixed_2d_bc():
    from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions

    return BoundaryConditions(
        dimension=2,
        segments=[
            BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=0.0, boundary="x_min"),
            BCSegment(name="right", bc_type=BCType.NO_FLUX, boundary="x_max"),
            BCSegment(name="bottom", bc_type=BCType.NO_FLUX, boundary="y_min"),
            BCSegment(name="top", bc_type=BCType.NO_FLUX, boundary="y_max"),
        ],
    )


# grid -> (dimension, BC, a node and whether the kept rows read it): row 1 reads a 1-D Dirichlet node through D * g;
# no kept row reads a 2-D all-Dirichlet corner; the no-flux wall row beside it reads a corner on a Dirichlet face;
# neither branch reads a periodic repeated endpoint.
_READ_SET_GRIDS = {
    "1-D Dirichlet": (1, lambda: bnd.dirichlet_bc(value=0.0, dimension=1), 0, True),
    "2-D all Dirichlet": (2, lambda: bnd.dirichlet_bc(value=0.0, dimension=2), 0, False),
    "2-D Dirichlet face against no-flux faces": (2, _mixed_2d_bc, 0, True),
    "1-D periodic": (1, lambda: bnd.periodic_bc(dimension=1, domain_bounds=np.array([[0.0, 1.0]])), 6, False),
}
# The explicit step (a callable drift) takes only no-flux and periodic walls, and reads every node but a periodic
# repeated endpoint: (5, 5) is one on the 2-D periodic grid.
_EXPLICIT_READ_SET_GRIDS = {
    "1-D periodic": _READ_SET_GRIDS["1-D periodic"],
    "2-D periodic": (2, lambda: bnd.periodic_bc(dimension=2, domain_bounds=np.array([[0.0, 1.0]] * 2)), 35, False),
    "2-D no-flux": (2, lambda: bnd.no_flux_bc(dimension=2), 0, True),
}


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    ("arm", "grid"), [("implicit", g) for g in _READ_SET_GRIDS] + [("explicit", g) for g in _EXPLICIT_READ_SET_GRIDS]
)
def test_the_fp_steps_branch_choice_reads_exactly_the_volatility_its_matrix_reads(arm, grid, monkeypatch):
    """F1 (a'): each FP-FDM step's branch choice reads the volatility at the nodes whose value its matrix reads, and
    nowhere else: the implicit step's kept rows, the explicit step's shared coefficient. The oracle is the assembled
    matrix: a node is read when perturbing its volatility moves the matrix the step solves. So a 2-D Dirichlet corner
    no kept row reads is left out, one a no-flux wall row reads stays in, and a periodic repeated endpoint is left out
    by both steps."""
    import scipy.sparse.linalg

    from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
    from mfgarchon.alg.numerical.fp_solvers import fp_fdm_time_stepping as stepping

    dim, bc, node, read_there = (_READ_SET_GRIDS if arm == "implicit" else _EXPLICIT_READ_SET_GRIDS)[grid]
    shape = (7,) if dim == 1 else (6, 6)
    problem = MFGProblem(
        model=Model(hamiltonian=SeparableHamiltonian(QuadraticControlCost(control_cost=1.0)), volatility=0.2),
        domain=TensorProductGrid(bounds=[(0.0, 1.0)] * dim, Nx_points=list(shape), boundary_conditions=bc()),
        conditions=Conditions(m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0, T=0.1),
        Nt=1,
    )
    axes = np.meshgrid(*[np.linspace(0.0, 1.0, n) for n in shape], indexing="ij")
    sigma = 0.2 + 0.1 * axes[0] + (0.05 * axes[1] if dim == 2 else 0.0)
    handed = (
        {"potential_field": np.zeros((2, *shape))}
        if arm == "implicit"
        else {"drift_field": lambda t, x, m: np.zeros((dim, *shape)) if dim > 1 else np.zeros(shape)}
    )
    matrices, reads = [], []
    solve = scipy.sparse.linalg.spsolve
    choose = stepping.volatility_varies
    monkeypatch.setattr(
        scipy.sparse.linalg, "spsolve", lambda A, b, *a, **k: (matrices.append(A.toarray()), solve(A, b, *a, **k))[1]
    )
    monkeypatch.setattr(
        stepping, "volatility_varies", lambda v, read: (reads.append({int(i) for i in read}), choose(v, read))[1]
    )

    def matrix(field):
        matrices.clear()
        reads.clear()
        FPFDMSolver(problem).solve_fp_system(
            np.ones(shape), **handed, volatility=field, volatility_kind="field", show_progress=False
        )
        return matrices[0]

    base = matrix(sigma)
    read_set = reads[-1]  # the implicit step first asks whether the array varies over every node
    moved = set()
    for j in range(sigma.size):
        perturbed = sigma.copy()
        perturbed.flat[j] *= 1.5
        if not np.array_equal(matrix(perturbed), base):
            moved.add(j)
    assert read_set == moved, (sorted(read_set - moved), sorted(moved - read_set))
    assert (node in moved) is read_there
    assert 0 < len(moved) < sigma.size or read_there


def test_a_volatility_nan_at_a_corner_no_kept_row_reads_solves_as_its_twin():
    """F1 (a'): on an all-Dirichlet 2-D grid no kept row reads the corner's volatility, so its NaN neither enters the
    step nor chooses the branch, and the solve is its twin's, bit for bit. Read by the choice, as under (a), it took the
    Ito branch while the uniform twin kept the row-scaled one, 4.4e-16 apart under FDM_UPWIND (the outcome census)."""

    def solve(nan):
        volatility = (lambda t, x: np.where(_at_point(x, (0.0, 0.0)), NAN, 0.2)) if nan else 0.2
        return _problem(
            dimension=2, bc=bnd.dirichlet_bc(value=0.0, dimension=2), model_kw={"volatility": volatility}
        ).solve(scheme=NumericalScheme.FDM_UPWIND, max_iterations=20, verbose=False)

    nan, twin = solve(True), solve(False)
    assert np.array_equal(nan.U, twin.U)
    assert np.array_equal(nan.M, twin.M)


class _MassOfAStackNonFinite(TensorProductGrid):
    """A user geometry whose measure answers a stack of densities, one per time level, with a non-finite first mass,
    while it answers one density finitely: construction's own mass check passes, and the iterator's read is reached."""

    def integrate(self, field, *args, **kwargs):
        out = np.array(super().integrate(field, *args, **kwargs), dtype=float)
        if out.ndim == 1 and out.size == np.asarray(field).shape[0] and np.asarray(field).ndim > self.dimension:
            out[0] = NAN
        return out


def test_the_iterator_names_a_non_finite_initial_mass_its_geometry_reports():
    """The iterator reads the initial mass, the first of the per-step masses the geometry's measure gives, through the
    owner. Every density entry is finite, so only a geometry the user hands in reaches it (review 1's p8, as a
    subclass; review 2's mutant INITIAL_MASS_UNREAD)."""
    grid = _MassOfAStackNonFinite(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=bnd.no_flux_bc(dimension=1))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="initial density mass", category=UserWarning)
        problem = MFGProblem(
            model=Model(hamiltonian=SeparableHamiltonian(QuadraticControlCost(control_cost=1.0)), volatility=0.2),
            domain=grid,
            conditions=Conditions(
                m_initial=lambda x: 1.0 + 0.0 * np.asarray(x[0]), u_terminal=lambda x: 0.0 * np.asarray(x[0]), T=0.5
            ),
            Nt=5,
        )
    with pytest.raises(
        NonFiniteReadError,
        match=r"FixedPointIterator: the initial mass "
        + OWN
        + r": 1 of 1 values, the first nan, at the initial density's total mass",
    ):
        problem.solve(max_iterations=3, verbose=False)
