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

# The names under which a solver-level source reaches a call: a solver's own `source_term`, and the
# `coupling_source` a graph coupling operator returns (#2406 review).
_SOLVER_SOURCES = {"source_term", "coupling_source"}


@pytest.mark.parametrize(
    "source",
    [lambda t, x: t + x[0], lambda x, t: t + x[0]],
    ids=["named", "in either order"],
)
def test_a_time_first_source_is_evaluated_at_t_and_x(source):
    np.testing.assert_array_equal(evaluate_solver_source(source, t=0.5, x=_X, dimension=1), 0.5 + _X)


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
        if path.name == "callable_protocols.py":  # the owner
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _SOLVER_SOURCES:
                direct.append(f"{path.relative_to(package)}:{node.lineno}")
    assert not direct, f"source_term called directly, not through evaluate_solver_source: {direct}"


def test_the_variational_lagrangian_callables_are_bound_time_first():
    """`VariationalMFGComponents`' Lagrangian and its derivatives read (t, x, v, m) by name."""
    from mfgarchon.types.callable_protocols import VARIATIONAL_LAGRANGIAN_SLOTS, bind_user_callable

    bound = bind_user_callable(lambda t, x, v, m: (t, x[0], v, m), VARIATIONAL_LAGRANGIAN_SLOTS, role="lagrangian_func")
    assert bound(t=1, x=(2,), v=3, m=4) == (1, 2, 3, 4)


def test_no_variational_lagrangian_callable_is_called_directly():
    import ast
    from pathlib import Path

    import mfgarchon.alg.optimization.variational_problem as module

    tree = ast.parse(Path(module.__file__).read_text())
    direct = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr.startswith("lagrangian")
        and node.func.attr.endswith("_func")
        and "components" in ast.unparse(node.func.value)
    ]
    assert not direct, f"a Lagrangian callable called directly, not through its binding, at lines {direct}"


def _plain(t, x):
    return t + x


def _wrapper(*a, **k):
    return _plain(*a, **k)


def test_remembering_a_binding_keeps_no_solve_s_arrays_alive():
    """#2406 review: the couplers build a closure per Picard iteration that captures that iteration's
    arrays. A cache holding the callable kept all of them alive after the solve; the binding plan is
    now remembered weakly, so a closure and its arrays go when the solve drops them."""
    import gc
    import weakref

    def closure_over(k):
        captured = np.full(1000, float(k))

        def source(t, x):
            return captured[: np.shape(x[0])[0]] * 0.0 + t

        return source, weakref.ref(captured)

    alive = []
    for k in range(5):
        source, ref = closure_over(k)
        evaluate_solver_source(source, t=0.5, x=_X, dimension=1)
        alive.append(ref)
        del source
    gc.collect()
    assert [ref() is None for ref in alive] == [True] * 5

