"""`mfgarchon.meta`, kept as an experimental subpackage (#2648, ruling 4): what is pinned here, and what is not.

1. The flux that its code generator emits takes D from its one owner, `diffusion_from_volatility`, and does not
   restate `0.5 * sigma**2`. `scripts/single_source_baseline.json` cannot see source held in a string template,
   because the check blanks string tokens, so this test is the pin for that site.
2. Six inputs that never had an effect are refused when passed, by keyword or by position: five dataclass fields,
   and `backend=` at the three generator entry points that accept it. The refusal stays until v0.25.0, in ruling
   16's shape. Each stays in the signature, so the positions of the parameters after it do not shift for a caller
   passing positionally.
3. Their default is `NEVER_READ`, which the refusal lets through, so `dataclasses.replace` and the `asdict` /
   `astuple` round trips -- which pass every field back in -- still work (review 1 of #2650, B1).

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
def test_a_field_that_never_had_an_effect_is_refused_by_name(cls, args, kwargs, name):
    """The message names the class called, not a base class the call reached through `super()`."""
    with pytest.raises(
        TypeError,
        match=rf"^{cls.__name__}\.__init__\(\) refuses {name}=: .+; it never had an effect, and the name goes in v0\.25\.0",
    ):
        cls(*args, **kwargs)


def test_the_solver_generator_refuses_a_backend_and_runs_without_one():
    """`generate_solver_class` forwards `backend` to `MFGSolverGenerator`, which forwards it to `CodeGenerator`; the
    default passes the whole chain, and a backend named at the entry point is refused there, by its own name."""
    system = quadratic_mfg_system(control_cost=0.5, state_cost=1.0)
    schemes = {"gradient": generate_discretization("gradient", 2), "divergence": generate_discretization("gradient", 2)}
    assert "class GeneratedPinned" in generate_solver_class("Pinned", system, schemes)
    refusal = r"generate_solver_class\(\) refuses backend=: .+; it never had an effect"
    with pytest.raises(TypeError, match=refusal):
        generate_solver_class("Pinned", system, schemes, backend="jax")
    with pytest.raises(TypeError, match=refusal):
        generate_solver_class("Pinned", system, schemes, "numpy")
    assert CodeGenerator().templates == {}
    assert MFGSolverGenerator().templates


def test_the_other_fields_still_bind_where_they_did():
    """Control: the refusals take nothing else away, and every field keeps the position it had at cb0724a9."""
    assert list(inspect.signature(DiscretizationScheme).parameters) == [
        "operator",
        "order",
        "stencil",
        "coefficients",
        "boundary_treatment",
    ]
    assert list(inspect.signature(PerformanceProfile).parameters) == [
        "problem_size",
        "backend",
        "target_precision",
        "memory_constraint",
        "time_constraint",
        "optimization_level",
    ]
    assert list(inspect.signature(OptimizationHints).parameters) == [
        "vectorizable",
        "parallelizable",
        "memory_bound",
        "compute_bound",
        "unroll_small_loops",
        "vectorize_inner_loops",
        "jax_hints",
        "numba_hints",
    ]
    assert DiscretizationScheme("gradient", 2, [-1, 1], [-0.5, 0.5]).operator == "gradient"
    profile = PerformanceProfile(100, "numpy", memory_constraint=64, optimization_level="speed")
    assert (profile.memory_constraint, profile.optimization_level) == (64, "speed")
    hints = OptimizationHints(vectorizable=False, compute_bound=False)
    assert (hints.vectorizable, hints.compute_bound) == (False, False)


def test_a_round_trip_passes_the_never_read_default_back():
    """`replace` and the `asdict` / `astuple` round trips pass every field back in, the refused ones included."""
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
    """The refusal binds the call first; a call that does not bind gets the function's own error, which names it."""
    with pytest.raises(TypeError, match=r"OptimizationHints\.__init__\(\) got an unexpected keyword argument 'bogus'"):
        OptimizationHints(bogus=True)
