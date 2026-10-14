"""`mfgarchon.meta`, kept as an experimental subpackage (#2648, ruling 4): what is pinned here, and what is not.

1. The flux that its code generator emits takes D from its one owner, `diffusion_from_volatility`, and does not
   restate `0.5 * sigma**2`. `scripts/single_source_baseline.json` cannot see source held in a string template,
   because the check blanks string tokens, so this test is the pin for that site.
2. Five dataclass fields that never had an effect are refused when passed, by keyword or by position. The refusal
   stays until v0.25.0, in ruling 16's shape. Each field stays in the signature, so the positions of the fields
   after it do not shift for a caller passing positionally.

Not pinned: that a generated solver runs. Its source does not compile today -- the HJB template's methods are
mis-indented -- and the package docstring says so.
"""

from __future__ import annotations

import pytest

from mfgarchon.meta import generate_discretization, generate_solver_class
from mfgarchon.meta.code_generation import DiscretizationScheme
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
]


@pytest.mark.parametrize(("cls", "args", "kwargs", "name"), REFUSED)
def test_a_field_that_never_had_an_effect_is_refused_by_name(cls, args, kwargs, name):
    with pytest.raises(
        TypeError, match=rf"refuses {name}=: .+; it never had an effect, and the name goes in v0\.25\.0"
    ):
        cls(*args, **kwargs)


def test_the_other_fields_still_bind_where_they_did():
    """Control: the refusals take nothing else away, and no field after a refused one has moved."""
    assert DiscretizationScheme("gradient", 2, [-1, 1], [-0.5, 0.5]).operator == "gradient"
    profile = PerformanceProfile(100, "numpy", memory_constraint=64, optimization_level="speed")
    assert (profile.memory_constraint, profile.optimization_level) == (64, "speed")
    hints = OptimizationHints(vectorizable=False, compute_bound=False)
    assert (hints.vectorizable, hints.compute_bound) == (False, False)
