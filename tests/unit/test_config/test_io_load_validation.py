"""load_solver_config fails LOUD on a wrong-vocabulary YAML instead of silently
returning all-default config.

The solver-config schema (``MFGSolverConfig`` / its ``SolverConfig`` alias) is FLAT:
top-level keys are ``hjb``, ``fp``, ``picard``, ``backend``, ``logging``. A plain
``model_validate`` silently ignores unknown top-level keys, so a ``solver:``-wrapped
or OmegaConf-vocabulary YAML (e.g. the generated ``config/configs/*.yaml``, whose top
level is ``solver:``/``optimization:``/``debug:``) used to load as all-defaults -- the
user's config silently doing nothing. ``load_solver_config`` now rejects unknown
top-level keys (kernel fail-fast). The flat round-trip (save -> load) is unaffected.
"""

from __future__ import annotations

import pytest
import yaml

from mfgarchon.config import HJBConfig, MFGSolverConfig, PicardConfig, load_solver_config, save_solver_config


def test_save_load_round_trip(tmp_path):
    """A config saved by save_solver_config loads back identically (flat schema)."""
    cfg = MFGSolverConfig(
        hjb=HJBConfig(method="fdm", accuracy_order=2),
        picard=PicardConfig(max_iterations=37, tolerance=1e-7),
    )
    path = tmp_path / "cfg.yaml"
    save_solver_config(cfg, path)

    loaded = load_solver_config(path)
    assert loaded.picard.max_iterations == 37
    assert loaded.picard.tolerance == 1e-7
    assert loaded.hjb.method == "fdm"
    assert loaded.hjb.accuracy_order == 2
    # Full structural equality on the JSON-dumped form.
    assert loaded.model_dump(mode="json") == cfg.model_dump(mode="json")


def test_flat_yaml_values_are_honored(tmp_path):
    """A correctly flat YAML actually drives the config (not silently defaulted)."""
    path = tmp_path / "flat.yaml"
    path.write_text(
        "hjb:\n  method: gfdm\npicard:\n  max_iterations: 99\n  tolerance: 1.0e-8\n",
    )
    cfg = load_solver_config(path)
    assert cfg.hjb.method == "gfdm"
    assert cfg.picard.max_iterations == 99
    assert cfg.picard.tolerance == 1e-8


def test_solver_wrapped_yaml_is_rejected(tmp_path):
    """A ``solver:``-wrapped YAML (the OmegaConf/legacy vocabulary) is rejected loudly
    instead of silently returning all-defaults."""
    path = tmp_path / "wrapped.yaml"
    yaml.safe_dump(
        {"solver": {"type": "fixed_point", "max_iterations": 100}, "optimization": {}, "debug": {}},
        path.open("w"),
    )
    with pytest.raises(ValueError, match=r"Unknown top-level key"):
        load_solver_config(path)


def test_rejection_names_the_offending_keys(tmp_path):
    """The error message names the unknown keys and the valid ones (actionable)."""
    path = tmp_path / "wrapped.yaml"
    path.write_text("solver:\n  max_iterations: 50\n")
    with pytest.raises(ValueError) as exc:
        load_solver_config(path)
    msg = str(exc.value)
    assert "solver" in msg
    assert "picard" in msg  # a valid key is listed


def test_from_yaml_classmethod_uses_same_path(tmp_path):
    """MFGSolverConfig.from_yaml delegates to load_solver_config (same validation)."""
    path = tmp_path / "wrapped.yaml"
    path.write_text("solver:\n  max_iterations: 50\n")
    with pytest.raises(ValueError, match=r"Unknown top-level key"):
        MFGSolverConfig.from_yaml(path)
