r"""Config models reject unknown fields instead of dropping them (#1766).

Pydantic ignores extras by default, so `PicardConfig(anderson_acceleration=True)` constructed
cleanly, discarded the field, and left `anderson_memory` at 0 -- Anderson OFF while the caller
had just asked for it, with nothing raised. The API v1.0 design note taught that exact call.

Measured when the guard went in, the three real call sites it caught were all the same shape:
`MFGSolverConfig(max_iterations=3)` and `(max_iterations=5)` in integration tests -- that field
lives under `picard`, so those tests believed they ran 3 and 5 Picard iterations and were
actually running the default 100. Their runtime and their result were both not what they said.
"""

from __future__ import annotations

import warnings

import pytest
from pydantic import ValidationError

from mfgarchon.config import MFGSolverConfig, PicardConfig


def test_an_unknown_field_raises_instead_of_vanishing():
    with pytest.raises(ValidationError, match=r"[Ee]xtra inputs"):
        PicardConfig(anderson_acceleration=True)


def test_a_field_that_belongs_to_a_nested_model_raises_at_the_top():
    """`max_iterations` is a PicardConfig field, not an MFGSolverConfig one.

    This is the shape that silently defaulted three integration tests to 100 iterations.
    """
    with pytest.raises(ValidationError, match=r"[Ee]xtra inputs"):
        MFGSolverConfig(max_iterations=3)
    assert MFGSolverConfig(picard=PicardConfig(max_iterations=3)).picard.max_iterations == 3


def test_deprecated_aliases_still_work():
    """`extra="forbid"` must not kill the deprecation surface.

    The legacy names are translated by a `model_validator(mode="before")` that pops the old key
    before validation runs, so the forbid check never sees it. If that ordering ever changes,
    this goes red rather than the deprecation silently becoming an error.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config = PicardConfig(damping_factor=0.7)
    assert config.relaxation == 0.7
    assert any("deprecated" in str(w.message) for w in caught)


def test_the_yaml_bridge_drops_interpolation_anchors_but_says_so():
    """A transport boundary is not an API call.

    `base_tol: 1e-6` with `picard.tolerance: ${base_tol}` is a legitimate OmegaConf idiom: the
    anchor is scaffolding and has no field to land in. The bridge drops it -- and warns, because
    a silent drop at the boundary is exactly how a genuine top-level typo would disappear.
    """
    omegaconf = pytest.importorskip("omegaconf")
    from mfgarchon.config.bridge import bridge_to_pydantic

    cfg = omegaconf.OmegaConf.create({"base_tol": 1e-6, "picard": {"tolerance": "${base_tol}", "max_iterations": 100}})
    with pytest.warns(UserWarning, match="dropped 1 top-level key"):
        config = bridge_to_pydantic(cfg, MFGSolverConfig)
    assert config.picard.tolerance == 1e-6


def test_a_nested_typo_still_fails_through_the_bridge():
    """Only the TOP level is filtered. A misspelled nested key reaches its own model."""
    omegaconf = pytest.importorskip("omegaconf")
    from mfgarchon.config.bridge import bridge_to_pydantic

    cfg = omegaconf.OmegaConf.create({"picard": {"toleranse": 1e-6}})
    with pytest.raises(ValidationError, match=r"[Ee]xtra inputs"):
        bridge_to_pydantic(cfg, MFGSolverConfig)
