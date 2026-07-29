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


def test_the_config_packages_own_examples_construct():
    """Every self-contained `>>>` example in `mfgarchon.config` must run.

    `extra="forbid"` converts a misplaced key from silently-dropped to a hard raise, which turns
    every stale docstring example in this package from misleading-but-running into crashing. Four
    sites taught `FPConfig(method="particle", num_particles=5000)` -- `num_particles` lives under
    `fp.particle` -- and each was found one at a time, by a human reading, because nothing checks
    Python docstrings: `scripts/check_doc_api.py` globs `*.md`, and `pytest.ini` sets no
    `--doctest-modules`.

    Executes rather than parses. Blocks that touch the filesystem are skipped by their imports,
    not by matching on prose, so a block cannot opt itself out by rewording.
    """
    import importlib
    import inspect
    import pkgutil
    import re

    import mfgarchon.config as config_pkg

    filesystem_names = ("from_yaml", "load_solver_config", "save_effective_config", "load_effective_config")
    failures = []
    executed = 0

    modules = [config_pkg]
    for info in pkgutil.walk_packages(config_pkg.__path__, prefix="mfgarchon.config."):
        try:
            modules.append(importlib.import_module(info.name))
        except Exception:
            continue

    for module in modules:
        for holder in [module, *(o for _, o in inspect.getmembers(module, inspect.isclass))]:
            doc = inspect.getdoc(holder) or ""
            # One namespace per docstring, as doctest does: a later block legitimately builds on
            # an import from an earlier one, and executing blocks in isolation would report that
            # as a NameError the reader never sees.
            namespace: dict = {}
            for block in re.findall(r"((?:^|\n)>>> .*?)(?=\n\s*\n|\Z)", doc, re.S):
                lines = []
                for line in block.splitlines():
                    stripped = line.strip()
                    if stripped.startswith((">>> ", "... ")):
                        lines.append(stripped[4:])
                code = "\n".join(lines).strip()
                if not code or any(name in code for name in filesystem_names):
                    continue
                executed += 1
                try:
                    exec(compile(code, f"<{getattr(holder, '__name__', holder)}>", "exec"), namespace)
                except Exception as exc:
                    failures.append(f"{getattr(holder, '__name__', holder)}: {type(exc).__name__}: {exc}")

    assert executed >= 3, f"only {executed} example blocks found; the scan is probably not seeing docstrings"
    assert not failures, "config docstring examples that do not run:\n  " + "\n  ".join(failures)
