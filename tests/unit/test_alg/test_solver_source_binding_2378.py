"""A solver's `source_term(t, x)` is evaluated through its binding (#2375 ruling 8, #2378 phase 5 part 3c).

The order was always (t, x), so ruling 8 did not reorder it and unnamed parameters can only mean
that order. What the binding adds is the refusal: a source written (x, t) used to receive time as
space without an error.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.types.callable_protocols import evaluate_solver_source

_X = np.linspace(0.0, 1.0, 5)


@pytest.mark.parametrize(
    "source",
    [lambda t, x: t + x, lambda t, pts: t + pts, lambda a, b: a + b],
    ids=["named", "time named", "unnamed"],
)
def test_a_time_first_source_is_evaluated_at_t_and_x(source):
    np.testing.assert_array_equal(evaluate_solver_source(source, t=0.5, x=_X), 0.5 + _X)


@pytest.mark.parametrize("source", [lambda x, t: x, lambda pts, time: pts], ids=["x, t", "pts, time"])
def test_a_space_first_source_is_refused(source):
    with pytest.raises(TypeError, match=r"out of order"):
        evaluate_solver_source(source, t=0.5, x=_X)


def test_a_solver_refuses_a_space_first_source_instead_of_computing_with_it():
    """End to end, on the n-D FDM HJB path: it used to hand `t` to the parameter it believed was `x`."""
    from mfgarchon import Conditions, MFGProblem, Model
    from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
    from mfgarchon.geometry import TensorProductGrid
    from mfgarchon.geometry.boundary import no_flux_bc

    problem = MFGProblem(
        model=Model(hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost()), sigma=0.2),
        domain=TensorProductGrid(
            bounds=[(0.0, 1.0)] * 2, Nx_points=[5, 5], boundary_conditions=no_flux_bc(dimension=2)
        ),
        conditions=Conditions(u_terminal=lambda x: 0.0, m_initial=lambda x: 1.0, T=0.2),
        Nt=4,
    )
    shape = (problem.Nt + 1, *problem.geometry.get_grid_shape())
    args = (np.ones(shape), np.zeros(shape[1:]), np.zeros(shape))
    with pytest.raises(TypeError, match=r"source_term .* out of order"):
        HJBFDMSolver(problem).solve_hjb_system(*args, source_term=lambda x, t: np.zeros(np.shape(x)[0]))


def test_no_solver_calls_its_source_term_directly():
    """Every evaluation goes through `evaluate_solver_source`: a site that calls `source_term(...)`
    itself would pass an (x, t) source time as space again. The population is the package source,
    read statically, so a site no test reaches is covered too."""
    import ast
    from pathlib import Path

    import mfgarchon

    package = Path(mfgarchon.__file__).parent
    direct = []
    for path in sorted(package.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "source_term":
                direct.append(f"{path.relative_to(package)}:{node.lineno}")
    assert not direct, f"source_term called directly, not through evaluate_solver_source: {direct}"
