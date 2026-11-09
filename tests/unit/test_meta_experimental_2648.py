"""`mfgarchon.meta`, kept as an experimental subpackage (#2648, ruling 4): what is pinned here, and what is not.

1. The flux that its code generator emits takes D from its one owner, `diffusion_from_volatility`, and does not
   restate `0.5 * sigma**2`. `scripts/single_source_baseline.json` cannot see source held in a string template,
   because the check blanks string tokens, so this test is the pin for that site.
2. Six inputs that never had an effect are deleted (the removal before 1.0): five dataclass fields, and `backend=`
   at the three generator entry points. Passing one by keyword is Python's own `TypeError`, which names it; a
   positional argument at or past its slot is a `TypeError` too, because the parameters after it are keyword-only,
   so it cannot land silently on the next one.
3. `dataclasses.replace` and the `asdict` / `astuple` round trips still work (review 1 of #2650, B1).

Not pinned: that a generated solver runs. Its source does not compile today, because the template substitution does
not indent multi-line snippets; repaired by hand, the generated FP solver still raises `NameError` on an undefined
`x_idx` (#2655). The package docstring says so.
"""

from __future__ import annotations

import copy
import dataclasses
import inspect
import pickle

import pytest

from mfgarchon.meta import generate_discretization, generate_solver_class
from mfgarchon.meta.code_generation import CodeGenerator, DiscretizationScheme, MFGSolverGenerator
from mfgarchon.meta.mathematical_dsl import quadratic_mfg_system
from mfgarchon.meta.optimization_meta import OptimizationHints, PerformanceProfile


def test_the_generated_flux_takes_the_diffusion_from_its_owner():
    system = quadratic_mfg_system(control_cost=0.5, state_cost=1.0)
    schemes = {"gradient": generate_discretization("gradient", 2), "divergence": generate_discretization("gradient", 2)}
    source = generate_solver_class("Pinned", system, schemes)
    flux = [line.strip() for line in source.splitlines() if line.strip().startswith("flux =")]
    assert flux == ["flux = m[x_idx] * optimal_control[x_idx] - diffusion_from_volatility(sigma) * m_grad"], flux
    assert "from mfgarchon.utils.pde_coefficients import diffusion_from_volatility" in source
    assert "0.5 * sigma**2" not in source


REFUSED = [
    (
        DiscretizationScheme,
        ("gradient", 2, [-1, 1], [-0.5, 0.5]),
        {"boundary_treatment": "periodic"},
        "boundary_treatment",
    ),
    (DiscretizationScheme, ("gradient", 2, [-1, 1], [-0.5, 0.5], "periodic"), {}, "boundary_treatment"),
    (PerformanceProfile, (100, "numpy"), {"target_precision": 1e-6}, "target_precision"),
    (PerformanceProfile, (100, "numpy", 1e-6), {}, "target_precision"),
    (PerformanceProfile, (100, "numpy"), {"time_constraint": 1.0}, "time_constraint"),
    (OptimizationHints, (), {"unroll_small_loops": True}, "unroll_small_loops"),
    (OptimizationHints, (), {"vectorize_inner_loops": False}, "vectorize_inner_loops"),
    (CodeGenerator, (), {"backend": "jax"}, "backend"),
    (CodeGenerator, ("numpy",), {}, "backend"),
    (MFGSolverGenerator, (), {"backend": "numba"}, "backend"),
    (MFGSolverGenerator, ("numpy",), {}, "backend"),
]


@pytest.mark.parametrize(("cls", "args", "kwargs", "name"), REFUSED)
def test_a_deleted_field_is_refused_by_keyword_and_by_position(cls, args, kwargs, name):
    """By keyword, Python names the field; by position, the call has one positional argument too many."""
    with pytest.raises(TypeError, match=rf"unexpected keyword argument '{name}'" if kwargs else r"positional argument"):
        cls(*args, **kwargs)


def test_the_solver_generator_takes_no_backend_and_runs_without_one():
    """`generate_solver_class` builds `MFGSolverGenerator`, which builds `CodeGenerator`; none takes a backend."""
    system = quadratic_mfg_system(control_cost=0.5, state_cost=1.0)
    schemes = {"gradient": generate_discretization("gradient", 2), "divergence": generate_discretization("gradient", 2)}
    assert "class GeneratedPinned" in generate_solver_class("Pinned", system, schemes)
    with pytest.raises(TypeError, match=r"generate_solver_class\(\) got an unexpected keyword argument 'backend'"):
        generate_solver_class("Pinned", system, schemes, backend="jax")
    with pytest.raises(TypeError, match=r"positional argument"):
        generate_solver_class("Pinned", system, schemes, "numpy")
    assert CodeGenerator().templates == {}
    assert MFGSolverGenerator().templates


def test_the_other_fields_still_bind_where_they_did():
    """Control: the deletions take nothing else away. A field before a deleted one keeps its position; a field after
    one is keyword-only."""

    def kinds(cls):
        return [(n, p.kind is inspect.Parameter.KEYWORD_ONLY) for n, p in inspect.signature(cls).parameters.items()]

    assert kinds(DiscretizationScheme) == [
        ("operator", False),
        ("order", False),
        ("stencil", False),
        ("coefficients", False),
    ]
    assert kinds(PerformanceProfile) == [
        ("problem_size", False),
        ("backend", False),
        ("memory_constraint", True),
        ("optimization_level", True),
    ]
    assert kinds(OptimizationHints) == [
        ("vectorizable", False),
        ("parallelizable", False),
        ("memory_bound", False),
        ("compute_bound", False),
        ("jax_hints", True),
        ("numba_hints", True),
    ]
    assert DiscretizationScheme("gradient", 2, [-1, 1], [-0.5, 0.5]).operator == "gradient"
    profile = PerformanceProfile(100, "numpy", memory_constraint=64, optimization_level="speed")
    assert (profile.memory_constraint, profile.optimization_level) == (64, "speed")
    hints = OptimizationHints(vectorizable=False, compute_bound=False)
    assert (hints.vectorizable, hints.compute_bound) == (False, False)


def test_a_round_trip_passes_every_field_back():
    """`replace` and the `asdict` / `astuple` round trips pass every field back in."""
    scheme = DiscretizationScheme("gradient", 2, [-1, 1], [-0.5, 0.5])
    profile = PerformanceProfile(100, "numpy")
    hints = OptimizationHints()
    assert dataclasses.replace(scheme, order=4).order == 4
    assert dataclasses.replace(profile, backend="jax").backend == "jax"
    assert dataclasses.replace(hints, vectorizable=False).vectorizable is False
    assert PerformanceProfile(**dataclasses.asdict(profile)) == profile
    assert DiscretizationScheme(*dataclasses.astuple(scheme)) == scheme
    for obj in (scheme, profile, hints):
        assert copy.copy(obj) == obj
        assert copy.deepcopy(obj) == obj
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            assert pickle.loads(pickle.dumps(obj, protocol=protocol)) == obj, protocol


def test_a_call_that_does_not_bind_keeps_the_function_in_its_message():
    """A call that does not bind gets the function's own error, which names it."""
    with pytest.raises(TypeError, match=r"OptimizationHints\.__init__\(\) got an unexpected keyword argument 'bogus'"):
        OptimizationHints(bogus=True)
