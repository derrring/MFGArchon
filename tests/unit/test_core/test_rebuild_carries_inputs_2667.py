"""A rebuilt problem carries every input it was built from (#2667).

`solve(Nt=)`, `with_model`, `with_domain` and `with_conditions` rebuild a problem made in the model/domain/conditions
form. Each restated its own argument list -- model, domain, conditions, constraints, Nt -- so every other input the
constructor accepted was dropped: a problem built with ``source_term_hjb=50`` solved at another Nt as if it had none,
with nothing reported. The four now read one owner, `MFGProblem._rebuilt`, which builds from the record the
constructor keeps. Two inputs nothing read are refused by name instead: ``drift=`` in this form (#2678), and
``boundary_conditions=`` in every form.
"""

from __future__ import annotations

import warnings

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import dirichlet_bc, no_flux_bc


def _grid() -> TensorProductGrid:
    return TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1))


MODEL = Model(hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0)), volatility=0.3)
CONDITIONS = Conditions(u_terminal=lambda x: 0.0, m_initial=lambda x: 1.0, T=1.0)


def _problem(nt: int = 10, **inputs) -> MFGProblem:
    return MFGProblem(model=MODEL, domain=_grid(), conditions=CONDITIONS, Nt=nt, **inputs)


def _u0(problem: MFGProblem, **solve_kwargs) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(problem.solve(max_iterations=3, tolerance=1e-14, verbose=False, **solve_kwargs).U)[0]


def test_solve_at_another_nt_keeps_the_hjb_source():
    """The issue's measurement: with the source dropped, solve(Nt=20) gave max|U(with) - U(without)| = 0 where the
    construction Nt gave 43.75. The rebuilt solve is the solve of the problem built at Nt = 20."""

    def source(t, x, v, m):
        return 50.0 * np.ones(len(x))

    rebuilt = _u0(_problem(source_term_hjb=source), Nt=20)
    np.testing.assert_array_equal(rebuilt, _u0(_problem(nt=20, source_term_hjb=source)))
    assert np.max(np.abs(rebuilt - _u0(_problem(nt=20)))) > 1.0


def test_every_rebuild_carries_every_input_the_constructor_took():
    def capture(problem, *args, **kwargs):
        return problem  # solve(Nt=) rebuilds, then calls solve on the rebuilt problem without Nt

    inputs = {
        "source_term_hjb": lambda t, x, v, m: 50.0 * np.ones(len(x)),
        "source_term_fp": lambda t, x, v, m: np.zeros(len(x)),
        "nonlocal_operator": 0.3 * np.eye(21),
        "state_penalty": lambda x: np.maximum(0.0, np.asarray(x) - 0.5),
        "state_penalty_scale": 3.0,
        "obstacles": [object()],
        "lambda_": 2.0,
        "coupling_coefficient": 0.9,
    }
    problem = _problem(**inputs)
    real_solve = MFGProblem.solve
    rebuilds = {
        "solve(Nt=)": lambda p: real_solve(p, Nt=12),
        "with_model": lambda p: p.with_model(MODEL),
        "with_domain": lambda p: p.with_domain(_grid()),
        "with_conditions": lambda p: p.with_conditions(CONDITIONS),
    }
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(MFGProblem, "solve", lambda p, *a, **k: real_solve(p, *a, **k) if "Nt" in k else capture(p))
        for site, rebuild in rebuilds.items():
            rebuilt = rebuild(problem)
            for name, value in inputs.items():
                carried = getattr(rebuilt, name)
                assert carried == value if isinstance(value, float) else carried is value, (site, name)


@pytest.mark.parametrize(
    ("form", "keyword"),
    [
        ("model", {"drift": lambda t, x, m: np.zeros_like(np.asarray(x, dtype=float))}),
        ("model", {"boundary_conditions": dirichlet_bc(dimension=1, value=0.0)}),
        ("components", {"boundary_conditions": dirichlet_bc(dimension=1, value=0.0)}),
    ],
)
def test_an_input_nothing_reads_is_refused_by_name(form, keyword):
    """Measured before the refusal: a Dirichlet ``boundary_conditions=`` beside a no-flux geometry changed the solve by
    0, and a ``drift=`` changed it by 0 -- no solver reads the problem's drift field (#2678)."""
    name = next(iter(keyword))
    if form == "model":
        with pytest.raises((TypeError, ValueError), match=rf"{name}=.*refused.*#2667"):
            _problem(**keyword)
        return
    components = MFGComponents(
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0)),
        u_terminal=lambda x: 0.0,
        m_initial=lambda x: 1.0,
    )
    with pytest.raises(TypeError, match=rf"{name}=.*refused.*#2667"):
        MFGProblem(geometry=_grid(), T=1.0, Nt=10, volatility=0.3, components=components, **keyword)
