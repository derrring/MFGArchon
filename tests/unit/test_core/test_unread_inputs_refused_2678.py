"""An input no solve reads is refused by name when the problem is built (#2678, #2563).

Each of these was accepted and changed the default solve by 0.0 in U and in M, with nothing said: ``Model(drift_field=)``,
``drift=``, ``constraints=``, ``state_penalty_scale`` with no ``state_penalty``, ``suppress_warnings=True`` where no
feasibility check runs, ``hamiltonian=`` beside ``components=``, and three that are read on one route and dropped on
another: ``spatial_discretization=`` off a grid built from ``spatial_bounds=``, ``obstacles=`` off a geometry, and
``lambda_=`` on a grid. ``coupling_coefficient``, and ``lambda_`` where it is kept, are read by another Hamiltonian, and
stay with #2681.

``hamiltonian=`` is no longer a parameter, and the retired-keyword table is all that refuses it: without its row the
keyword warns as unrecognized and is dropped, which is #2563 again.

The refusal of a drift names routes, so each route has a row that follows it. Its bodies read both ``t`` and ``x``,
since a constant reads the same in either order (#2864).
"""

from __future__ import annotations

import functools

import pytest

import numpy as np
from scipy.integrate import trapezoid

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.coupling.fixed_point_iterator import FixedPointIterator
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver, FPParticleSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.geometry import Hypersphere, TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

# These rows hand the components= form what it refuses, so its notice that the form is planned to be replaced is noise.
pytestmark = pytest.mark.filterwarnings("ignore:The components= form of MFGProblem:PendingDeprecationWarning")

_ROUTES = ("solve_fp_system(drift_field=)", "FPFDMSolver", "FPParticleSolver", "FixedPointIterator(drift_field=)")


def _x(y):
    return np.asarray(y, dtype=float)[..., 0]


def _gaussian(points):
    return np.exp(-40 * (points - 0.3) ** 2)


_POINTS = np.linspace(0.0, 1.0, 21)
_MASS = float(trapezoid(_gaussian(_POINTS), _POINTS))


def _m_initial(x):
    return _gaussian(_x(x)) / _MASS


def _u_terminal(x):
    return 2.0 * (_x(x) - 0.7) ** 2


def _grid():
    return TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1))


def _hamiltonian(weight=1.0):
    return SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=functools.partial(np.multiply, weight),
        coupling_dm=lambda m: weight * np.ones_like(m),
    )


def _components(hamiltonian=None):
    return MFGComponents(hamiltonian=hamiltonian or _hamiltonian(), m_initial=_m_initial, u_terminal=_u_terminal)


def _model_form(**kwargs):
    model = Model(hamiltonian=_hamiltonian(), volatility=kwargs.pop("model_volatility", 0.4))
    conditions = Conditions(m_initial=_m_initial, u_terminal=_u_terminal, T=0.5)
    return MFGProblem(model=model, domain=_grid(), conditions=conditions, Nt=10, **kwargs)


def _components_form(**kwargs):
    kwargs.setdefault("components", _components())
    kwargs.setdefault("volatility", 0.4)
    if "network" not in kwargs and "spatial_bounds" not in kwargs:
        kwargs.setdefault("geometry", _grid())
    return MFGProblem(T=0.5, Nt=10, **kwargs)


_FORMS = {"model form": _model_form, "components form": _components_form}


def _drift(t, x, m):
    return 0.2 * (0.5 - np.asarray(x)) * (1.0 + np.asarray(t))


def _drift_written_x_first(x, t, m):
    return 0.2 * (0.5 - np.asarray(x)) * (1.0 + np.asarray(t))


def _drift_unnamed(a, b, c):
    return 0.2 * (0.5 - np.asarray(b)) * (1.0 + np.asarray(a))


def _zero_drift(t, x, m):
    return np.zeros_like(np.asarray(x, dtype=float))


def _density():
    return _gaussian(_POINTS) / _MASS


def _gap(a, b):
    return float(np.abs(np.asarray(a) - np.asarray(b)).max())


@pytest.mark.unit
@pytest.mark.parametrize("given", [{"hamiltonian": _hamiltonian()}, {}], ids=["beside a Hamiltonian", "alone"])
def test_a_model_refuses_a_drift_field_and_names_the_routes(given):
    """Alone it is told about the drift, not that it needs a Hamiltonian: the form the docstring called FP-only never
    reached a solve."""
    with pytest.raises(ValueError, match=r"Model\(drift_field=\.\.\.\) is refused.*#2678") as refused:
        Model(drift_field=_drift, **given)
    for route in _ROUTES:
        assert route in str(refused.value)
    for claim in (
        "pass a callable",
        "replaces the drift the Hamiltonian gives",
        "(t, x, m)",
        "Nt+2 slices (#2412)",
        "a velocity on FPFDMSolver, and a value function on FPParticleSolver",
        "A number has no route",
    ):
        assert claim in str(refused.value)


@pytest.mark.unit
def test_a_model_given_a_drift_field_after_construction_is_refused_where_a_problem_binds_it():
    """A ``Model`` is mutable, so the refusal is read again where a problem binds it. ``drift = model.drift_field``
    used to read it here, and store it for nothing."""
    model = Model(hamiltonian=_hamiltonian(), volatility=0.4)
    model.drift_field = _drift
    with pytest.raises(ValueError, match=r"Model\(drift_field=\.\.\.\) is refused.*#2678"):
        MFGProblem(
            model=model,
            domain=_grid(),
            conditions=Conditions(m_initial=_m_initial, u_terminal=_u_terminal, T=0.5),
            Nt=10,
        )


@pytest.mark.unit
def test_a_model_still_needs_a_hamiltonian_or_a_lagrangian():
    with pytest.raises(ValueError, match="requires a hamiltonian or a lagrangian"):
        Model()


@pytest.mark.unit
@pytest.mark.parametrize("form", _FORMS)
@pytest.mark.parametrize("drift", [_drift, 0.05, 0.0, np.zeros(21)], ids=["callable", "float", "zero", "array"])
def test_a_problem_refuses_a_drift_and_names_the_routes(form, drift):
    with pytest.raises(ValueError, match=r"MFGProblem\(drift=\.\.\.\) is refused.*#2678") as refused:
        _FORMS[form](drift=drift)
    for route in _ROUTES:
        assert route in str(refused.value)


@pytest.mark.unit
@pytest.mark.parametrize("form", _FORMS)
def test_constraints_are_refused_when_given(form):
    for given in (["anything"], []):
        with pytest.raises(ValueError, match=r"constraints=.*refused.*#2678"):
            _FORMS[form](constraints=given)
    assert _FORMS[form](constraints=None) is not None


@pytest.mark.unit
@pytest.mark.parametrize("form", _FORMS)
def test_a_state_penalty_scale_with_no_state_penalty_is_refused(form):
    for scale in (3.0, 0.5):
        with pytest.raises(ValueError, match=r"state_penalty_scale=.*no state_penalty=.*#2678"):
            _FORMS[form](state_penalty_scale=scale)
    assert _FORMS[form](state_penalty_scale=1.0) is not None


@pytest.mark.unit
def test_a_state_penalty_scale_beside_a_state_penalty_stays_and_moves_the_solve():
    wall = {"state_penalty": lambda x: 2.0 * _x(x)}
    one = _model_form(**wall).solve(verbose=False)
    three = _model_form(**wall, state_penalty_scale=3.0).solve(verbose=False)
    assert _gap(one.U, three.U) > 1e-2


@pytest.mark.unit
@pytest.mark.parametrize(
    ("build", "mode"),
    [
        (_model_form, "geometry"),
        (_components_form, "geometry"),
        (functools.partial(_components_form, network=object()), "network"),
    ],
    ids=["model form", "components form, geometry", "components form, network"],
)
def test_suppress_warnings_is_refused_where_no_feasibility_check_runs(build, mode):
    """The refusal comes before the network is read, so any object stands for one here and networkx is not needed."""
    with pytest.raises(ValueError, match=rf"suppress_warnings=True\) is refused on a {mode}.*#2678"):
        build(suppress_warnings=True)


@pytest.mark.unit
@pytest.mark.parametrize("form", _FORMS)
def test_suppress_warnings_false_is_accepted_on_a_geometry(form):
    assert _FORMS[form](suppress_warnings=False) is not None


@pytest.mark.unit
@pytest.mark.filterwarnings("ignore:No spatial domain specified", "ignore:initial density mass")
def test_suppress_warnings_is_accepted_on_the_default_domain():
    """The library builds that grid too, so the check runs there."""
    assert MFGProblem(components=_components(), T=0.5, Nt=10, volatility=0.4, suppress_warnings=True) is not None


_ROUTES_BUILT = {
    "a geometry": {"geometry": _grid()},
    "the default domain": {},
    "a network": {"network": object()},
    "spatial_bounds": {"spatial_bounds": [(0.0, 1.0)], "spatial_discretization": [20]},
}


def _on(route, **given):
    return MFGProblem(
        **{"components": _components(), "T": 0.5, "Nt": 10, "volatility": 0.4, **_ROUTES_BUILT[route], **given}
    )


@pytest.mark.unit
@pytest.mark.parametrize("route", ["a geometry", "the default domain", "a network"])
def test_spatial_discretization_is_refused_off_a_grid_built_from_spatial_bounds(route):
    with pytest.raises(ValueError, match=rf"spatial_discretization=\.\.\.\) is refused on {route}.*#2678"):
        _on(route, spatial_discretization=[40])


@pytest.mark.unit
@pytest.mark.filterwarnings("ignore:initial density mass")
def test_spatial_discretization_is_read_beside_spatial_bounds():
    assert _on("spatial_bounds").spatial_shape != _on("spatial_bounds", spatial_discretization=[40]).spatial_shape


@pytest.mark.unit
@pytest.mark.parametrize("route", ["spatial_bounds", "the default domain", "a network"])
def test_obstacles_are_refused_off_a_geometry(route):
    """On those routes the solve ran on a domain with no obstacle, with nothing said."""
    with pytest.raises(ValueError, match=r"obstacles=\.\.\.\) is refused on .*#2678"):
        _on(route, obstacles=[object()])


@pytest.mark.unit
def test_obstacles_are_read_on_a_geometry():
    assert _on("a geometry", obstacles=[Hypersphere(center=[0.5], radius=0.1)]).has_obstacles


@pytest.mark.unit
@pytest.mark.parametrize("route", ["spatial_bounds", "the default domain"])
def test_lambda_is_refused_on_a_grid_that_keeps_it_nowhere(route):
    with pytest.raises(ValueError, match=r"lambda_=\.\.\.\) is refused on .*#2678"):
        _on(route, lambda_=2.0)


@pytest.mark.unit
def test_lambda_is_kept_on_a_geometry():
    assert _on("a geometry", lambda_=2.0).lambda_ == 2.0


@pytest.mark.unit
@pytest.mark.parametrize(("suppress", "warned"), [(False, True), (True, False)])
def test_suppress_warnings_is_read_where_the_library_builds_the_grid(suppress, warned):
    """The path it stays on: the feasibility warning moves with the flag."""
    bounds = {"spatial_bounds": [(0.0, 1.0)] * 5, "spatial_discretization": [2] * 5}
    with pytest.warns() as caught:
        _components_form(suppress_warnings=suppress, **bounds)
    assert any("HIGH DIMENSION WARNING" in str(w.message) for w in caught) is warned


_HAMILTONIAN_ROW = r"'hamiltonian' -> Model\(hamiltonian=\.\.\.\), or MFGComponents\(hamiltonian=\.\.\.\).*#2563"


class _ProblemWithItsOwnFunctions(MFGProblem):
    """The one route that built with ``hamiltonian=`` and no components: no class in this repository takes it."""

    def _initialize_functions(self, **kwargs):
        pass


@pytest.mark.unit
@pytest.mark.parametrize(
    "hamiltonian",
    [_hamiltonian(3.0), lambda t, x, p, m: 0.0, None],
    ids=["a Hamiltonian", "a callable", "None"],
)
def test_a_hamiltonian_beside_components_is_refused_and_names_both_routes(hamiltonian):
    with pytest.raises(ValueError, match=_HAMILTONIAN_ROW) as refused:
        _components_form(hamiltonian=hamiltonian)
    assert "Beside components=, put the one you mean there" in str(refused.value)
    assert "pass it through Model" not in str(refused.value), "a components= caller is not told to migrate"


@pytest.mark.unit
@pytest.mark.parametrize(
    "hamiltonian", [_hamiltonian(3.0), lambda t, x, p, m: 0.0], ids=["a Hamiltonian", "a callable"]
)
def test_a_hamiltonian_with_no_components_is_refused_by_the_same_row(hamiltonian):
    with pytest.raises(ValueError, match=_HAMILTONIAN_ROW):
        MFGProblem(geometry=_grid(), T=0.5, Nt=10, volatility=0.4, hamiltonian=hamiltonian)


@pytest.mark.unit
def test_a_hamiltonian_beside_a_model_is_refused_by_the_same_row():
    with pytest.raises(ValueError, match=_HAMILTONIAN_ROW):
        _model_form(hamiltonian=_hamiltonian(3.0))


@pytest.mark.unit
def test_a_subclass_with_its_own_functions_is_refused_by_the_same_row():
    with pytest.raises(ValueError, match=_HAMILTONIAN_ROW):
        _ProblemWithItsOwnFunctions(geometry=_grid(), T=0.5, Nt=10, volatility=0.4, hamiltonian=_hamiltonian())


@pytest.mark.unit
def test_each_route_the_hamiltonian_row_names_moves_the_solve():
    one = _components_form().solve(verbose=False)
    through_components = _components_form(components=_components(_hamiltonian(3.0))).solve(verbose=False)
    through_model = MFGProblem(
        model=Model(hamiltonian=_hamiltonian(3.0), volatility=0.4),
        domain=_grid(),
        conditions=Conditions(m_initial=_m_initial, u_terminal=_u_terminal, T=0.5),
        Nt=10,
    ).solve(verbose=False)
    assert _gap(one.U, through_components.U) > 1e-2
    assert _gap(through_model.U, through_components.U) == 0.0


@pytest.mark.unit
def test_the_advice_on_fpfdmsolver_moves_the_density_and_reads_by_position():

    def solve(drift):
        return FPFDMSolver(_model_form()).solve_fp_system(_density(), drift_field=drift, show_progress=False)

    plain = solve(_zero_drift)
    written_as_told = solve(_drift)
    assert _gap(written_as_told, plain) > 1e-3
    assert _gap(written_as_told, solve(_drift_unnamed)) == 0.0
    assert _gap(written_as_told, solve(_drift_written_x_first)) > 1e-3


@pytest.mark.unit
def test_the_advice_on_fpparticlesolver_moves_the_density_and_reads_by_position():
    """Seeded alike, so the gap is the drift's and not the draw's. ``x`` is an array on this path (#2864)."""

    def solve(drift):
        solver = FPParticleSolver(_model_form(), num_particles=2000, seed=7)
        return solver.solve_fp_system(_density(), drift_field=drift, show_progress=False)

    repeat = solve(_zero_drift)
    assert _gap(repeat, solve(_zero_drift)) == 0.0
    written_as_told = solve(_drift)
    assert _gap(written_as_told, repeat) > 1e-3
    assert _gap(written_as_told, solve(_drift_unnamed)) == 0.0
    assert _gap(written_as_told, solve(_drift_written_x_first)) > 1e-3


@pytest.mark.unit
def test_an_array_is_a_velocity_on_fpfdmsolver():
    def solve(velocity):
        return FPFDMSolver(_model_form()).solve_fp_system(_density(), drift_field=velocity, show_progress=False)

    assert _gap(solve(np.full((11, 21), 0.5)), solve(np.zeros((11, 21)))) > 1e-2


@pytest.mark.unit
def test_an_array_is_a_value_function_on_fpparticlesolver():
    """A constant has no gradient, so the same constant that moves FP-FDM's density moves nothing here."""

    def solve(**kwargs):
        solver = FPParticleSolver(_model_form(), num_particles=2000, seed=7)
        return solver.solve_fp_system(_density(), show_progress=False, **kwargs)

    field = 0.2 * (0.5 - np.linspace(0.0, 1.0, 21)) * np.ones((11, 21))
    assert _gap(solve(drift_field=field), solve(potential_field=field)) == 0.0
    assert _gap(solve(drift_field=np.full((11, 21), 0.5)), solve(drift_field=np.zeros((11, 21)))) == 0.0


@pytest.mark.unit
@pytest.mark.parametrize("solver", [FPFDMSolver, FPParticleSolver])
def test_a_number_has_no_route(solver):
    with pytest.raises(TypeError):
        solver(_model_form()).solve_fp_system(_density(), drift_field=0.5, show_progress=False)


@pytest.mark.unit
def test_the_particle_route_returns_nt_plus_2_slices():
    """RECORDED DEFECT, not a contract: #2412. The refusal says so, and fixing it trips this row and that sentence."""
    solver = FPParticleSolver(_model_form(), num_particles=200, seed=7)
    density = solver.solve_fp_system(_density(), drift_field=_drift, show_progress=False)
    assert density.shape[0] == 12, "#2412 is fixed: remove this row and the sentence in UNREAD_DRIFT"


@pytest.mark.unit
def test_the_advice_on_the_iterator_replaces_the_drift_the_hamiltonian_gives_and_reads_by_position():
    """A zero drift is not "no drift added": it moves M away from the default solve, since it replaces H's."""

    def solve(**kwargs):
        problem = _model_form()
        iterator = FixedPointIterator(problem, HJBFDMSolver(problem), FPFDMSolver(problem), **kwargs)
        return iterator.solve(max_iterations=8, tolerance=1e-4, verbose=False)

    written_as_told = solve(drift_field=_drift).M
    assert _gap(solve(drift_field=_zero_drift).M, solve().M) > 1e-1
    assert _gap(written_as_told, solve(drift_field=_drift_unnamed).M) == 0.0
    assert _gap(written_as_told, solve(drift_field=_drift_written_x_first).M) > 1e-3


@pytest.mark.unit
def test_the_control_a_read_input_still_moves_the_solve():
    """The population's control: the same fixture moves when it is given an input a solve does read."""
    low = _model_form().solve(verbose=False)
    high = _model_form(model_volatility=0.8).solve(verbose=False)
    assert _gap(low.U, high.U) > 1e-2
