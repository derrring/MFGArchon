"""An input no solve reads is refused by name when the problem is built (#2678, #2563).

Each of these was accepted and changed the default solve by 0.0 in U and in M, with nothing said: ``Model(drift_field=)``,
``drift=``, ``constraints=``, ``state_penalty_scale`` with no ``state_penalty``, ``suppress_warnings=True`` where no
feasibility check runs, and ``hamiltonian=`` beside ``components=``. ``coupling_coefficient`` and ``lambda_`` are read by
another Hamiltonian, and stay with #2681.

The refusal of a drift names routes, so each route has a row that follows it. Its bodies read both ``t`` and ``x``,
since a constant reads the same in either order (#2864).
"""

from __future__ import annotations

import functools

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.coupling.fixed_point_iterator import FixedPointIterator
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver, FPParticleSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

_ROUTES = ("solve_fp_system(drift_field=)", "FPFDMSolver", "FPParticleSolver", "FixedPointIterator(drift_field=)")


def _x(y):
    return np.asarray(y, dtype=float)[..., 0]


def _m_initial(x):
    return np.exp(-40 * (_x(x) - 0.3) ** 2)


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
    points = np.linspace(0.0, 1.0, 21)
    values = np.exp(-40 * (points - 0.3) ** 2)
    return values / (values.sum() * 0.05)


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
    assert "replaces the drift the Hamiltonian gives" in str(refused.value)
    assert "(t, x, m)" in str(refused.value)


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
@pytest.mark.parametrize("drift", [_drift, 0.05, np.zeros(21)], ids=["callable", "float", "array"])
def test_a_problem_refuses_a_drift_and_names_the_routes(form, drift):
    with pytest.raises(ValueError, match=r"MFGProblem\(drift=\.\.\.\) is refused.*#2678") as refused:
        _FORMS[form](drift=drift)
    for route in _ROUTES:
        assert route in str(refused.value)


@pytest.mark.unit
@pytest.mark.parametrize("form", _FORMS)
def test_constraints_are_refused_when_given(form):
    with pytest.raises(ValueError, match=r"constraints=.*refused.*#2678"):
        _FORMS[form](constraints=["anything"])
    assert _FORMS[form](constraints=None) is not None


@pytest.mark.unit
@pytest.mark.parametrize("form", _FORMS)
def test_a_state_penalty_scale_with_no_state_penalty_is_refused(form):
    with pytest.raises(ValueError, match=r"state_penalty_scale=.*no state_penalty=.*#2678"):
        _FORMS[form](state_penalty_scale=3.0)
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
@pytest.mark.parametrize(("suppress", "warned"), [(False, True), (True, False)])
def test_suppress_warnings_is_read_where_the_library_builds_the_grid(suppress, warned):
    """The path it stays on: the feasibility warning moves with the flag."""
    bounds = {"spatial_bounds": [(0.0, 1.0)] * 5, "spatial_discretization": [2] * 5}
    with pytest.warns() as caught:
        _components_form(suppress_warnings=suppress, **bounds)
    assert any("HIGH DIMENSION WARNING" in str(w.message) for w in caught) is warned


@pytest.mark.unit
@pytest.mark.parametrize(
    "hamiltonian", [_hamiltonian(3.0), lambda t, x, p, m: 0.0], ids=["a Hamiltonian", "a callable"]
)
def test_a_hamiltonian_beside_components_is_refused_and_names_the_one_to_keep(hamiltonian):
    with pytest.raises(
        ValueError, match=r"hamiltonian=\.\.\., components=\.\.\.\) is refused \(#2563\).*Keep components="
    ):
        _components_form(hamiltonian=hamiltonian)


@pytest.mark.unit
def test_what_the_hamiltonian_refusal_says_to_do_moves_the_solve():
    one = _components_form().solve(verbose=False)
    three = _components_form(components=_components(_hamiltonian(3.0))).solve(verbose=False)
    assert _gap(one.U, three.U) > 1e-2


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
