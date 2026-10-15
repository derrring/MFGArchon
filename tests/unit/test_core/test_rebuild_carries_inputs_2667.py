"""A rebuilt problem carries every input it was built from (#2667).

`solve(Nt=)`, `with_model`, `with_domain` and `with_conditions` rebuild a problem made in the model/domain/conditions
form. Each restated its own argument list -- model, domain, conditions, constraints, Nt -- so every other input the
constructor accepted was dropped: a problem built with ``source_term_hjb=50`` solved at another Nt as if it had none,
with nothing reported. The four now read one owner, `MFGProblem._rebuilt`, which builds from the record the
constructor keeps. Two inputs nothing read are refused by name instead: ``drift=`` in this form (#2678), and
``boundary_conditions=`` in every form.

The components= form has its own record (#2666), read by the one rebuild of that form the library makes:
`StochasticMFGProblem.create_conditional_problem`, which built each noise path's problem by hand from five fields.
"""

from __future__ import annotations

import dataclasses
import inspect
import warnings

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.stochastic import OrnsteinUhlenbeckProcess, StochasticMFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import dirichlet_bc, no_flux_bc


def _grid() -> TensorProductGrid:
    return TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1))


MODEL = Model(hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0)), volatility=0.3)
CONDITIONS = Conditions(
    u_terminal=lambda x: (np.asarray(x) - 0.3) ** 2,
    m_initial=lambda x: 1.0 + 0.5 * np.cos(np.pi * np.asarray(x)),
    T=1.0,
)


def _problem(nt: int = 10, model=MODEL, domain=None, conditions=CONDITIONS, **inputs) -> MFGProblem:
    return MFGProblem(model=model, domain=domain or _grid(), conditions=conditions, Nt=nt, **inputs)


def _u0(problem: MFGProblem, **solve_kwargs) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(problem.solve(max_iterations=3, tolerance=1e-14, verbose=False, **solve_kwargs).U)[0]


def test_every_rebuild_solves_as_the_fresh_build_of_its_inputs():
    """Each rebuild applies its change and carries every other input: solved, it is the solve of a problem built afresh
    from the new object and the same inputs, bit for bit, and not the solve it was rebuilt from. U(0) here depends on
    Nt, the model, the domain and the conditions, so a rebuild that ignored its change, or let the old objects override
    it, would solve as the original; one that composed the penalty or the source a second time would match neither.
    The issue's measurement is the last line: the source dropped, solve(Nt=20) gave the sourceless answer."""

    def source(t, x, v, m):
        return 50.0 * np.ones(len(x))

    inputs = {"source_term_hjb": source, "state_penalty": lambda x: np.maximum(0.0, np.asarray(x) - 0.5)}
    model = Model(hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=2.0)), volatility=0.45)
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
    conditions = Conditions(
        u_terminal=lambda x: (np.asarray(x) - 0.7) ** 2,
        m_initial=lambda x: 1.0 + 0.5 * np.cos(2 * np.pi * np.asarray(x)),
        T=2.0,
    )
    original = _problem(**inputs)
    before = _u0(original)
    cases = {
        "solve(Nt=)": (lambda: _u0(original, Nt=20), lambda: _u0(_problem(nt=20, **inputs))),
        "with_model": (lambda: _u0(original.with_model(model)), lambda: _u0(_problem(model=model, **inputs))),
        "with_domain": (lambda: _u0(original.with_domain(grid)), lambda: _u0(_problem(domain=grid, **inputs))),
        "with_conditions": (
            lambda: _u0(original.with_conditions(conditions)),
            lambda: _u0(_problem(conditions=conditions, **inputs)),
        ),
    }
    for site, (rebuilt, fresh) in cases.items():
        solved = rebuilt()
        np.testing.assert_array_equal(solved, fresh(), err_msg=site)
        assert solved.shape != before.shape or np.max(np.abs(solved - before)) > 1e-6, site
    assert np.max(np.abs(_u0(original, Nt=20) - _u0(_problem(nt=20)))) > 1.0


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


def test_the_conditional_problem_carries_every_input_of_the_components_form():
    """Measured before #2666 on CommonNoiseMFGSolver: a constant ``source_term_hjb`` or ``source_term_fp`` moved the mean
    by 0, called 0 times. Two enumerations keep the record whole as the constructor grows: every named parameter of
    ``MFGProblem.__init__`` outside the model form's three is in it, and every ``MFGComponents`` field reaches the
    conditional problem unless the frozen noise replaces it.

    A ``state_penalty`` cannot be carried: it is composed into a SeparableHamiltonian's potential, and the frozen-noise
    Hamiltonian is the user's ``conditional_hamiltonian``, which has none (#2700). It was dropped before; it is refused
    now, by name."""
    inputs = {
        "source_term_hjb": lambda t, x, v, m: 50.0 * np.ones(len(x)),
        "source_term_fp": lambda t, x, v, m: np.zeros(len(x)),
        "nonlocal_operator": 0.3 * np.eye(21),
        "lambda_": 2.0,
        "coupling_coefficient": 0.9,
    }

    def conditional_of(**given):
        components = MFGComponents(
            hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0)),
            u_terminal=lambda x: (np.asarray(x) - 0.3) ** 2,
            m_initial=lambda x: 1.0 + 0.5 * np.cos(np.pi * np.asarray(x)),
            parameters={"label": "parent"},
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parent = StochasticMFGProblem(
                geometry=_grid(),
                T=1.0,
                Nt=10,
                volatility=0.3,
                noise_process=OrnsteinUhlenbeckProcess(kappa=1.0, mu=0.0, sigma=0.1),
                conditional_hamiltonian=lambda x, p, m, theta: 0.5 * np.sum(p**2, axis=-1) + theta,
                components=components,
                **given,
            )
            return parent, parent.create_conditional_problem(parent.sample_noise_path(seed=0))

    parent, conditional = conditional_of(**inputs)

    for name, value in inputs.items():
        carried = getattr(conditional, name)
        assert carried == value if isinstance(value, float) else carried is value, name

    replaced = {"hamiltonian", "lagrangian", "u_terminal", "parameters", "description", "problem_type"}
    for field in dataclasses.fields(MFGComponents):
        if field.init and field.name not in replaced:
            assert getattr(conditional.components, field.name) is getattr(parent.components, field.name), field.name
    assert conditional.components.parameters == {"label": "parent"}
    assert conditional.components.parameters is not parent.components.parameters

    named = set(inspect.signature(MFGProblem.__init__).parameters) - {"self", "model", "domain", "conditions", "kwargs"}
    missing = sorted(named - set(parent._components_inputs))
    assert not missing, f"MFGProblem.__init__ takes {missing}, which the components form's record does not keep"

    with pytest.raises(NotImplementedError, match=r"common-noise problem declares state_penalty.*#2700"):
        conditional_of(state_penalty=lambda x: np.maximum(0.0, np.asarray(x) - 0.5))
