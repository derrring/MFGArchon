"""`check_manifests` gates unguarded imports only, and its verdict must not move with the env.

Two properties, both learned the hard way.

**Guarded is not undeclared.** `pyyaml` was imported at module level by `config/io.py`, declared
nowhere, arriving transitively; dropping the packages that carried it would have raised ImportError
at import on a fresh install (#1687). A module-level `try: import cvxpy / except ImportError` cannot
do that — the module sets a flag and carries on. Gating those produces false findings, and a check
with false findings is one people learn to ignore.

**The verdict must not depend on what is installed.** `packages_distributions()` knows only the
current environment, so the modules it cannot map are exactly the undeclared, uninstalled ones the
check exists to catch. Nine of twenty-three module-level third-party imports were invisible under
CI's own install while the check printed green.
"""

from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location("_check_manifests", REPO / "scripts" / "check_manifests.py")
_CM = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CM)


def _tree(tmp_path: Path, source: str) -> Path:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "mod.py").write_text(textwrap.dedent(source))
    return pkg


@pytest.mark.parametrize(
    ("label", "source"),
    [
        ("bare", "import alpha\n"),
        ("from", "from alpha import thing\n"),
        ("inside a top-level if", "import sys\nif sys.platform == 'darwin':\n    import alpha\n"),
        ("inside a module-level with", "import contextlib\nwith contextlib.nullcontext():\n    import alpha\n"),
        ("inside a module-level for", "for _ in (1,):\n    import alpha\n"),
        ("inside a module-level while", "while False:\n    import alpha\n"),
        ("inside a class body", "class C:\n    import alpha\n"),
        ("inside a try that catches ValueError", "try:\n    import alpha\nexcept ValueError:\n    pass\n"),
        ("inside try/else", "try:\n    pass\nexcept ImportError:\n    pass\nelse:\n    import alpha\n"),
        ("inside try/finally", "try:\n    pass\nfinally:\n    import alpha\n"),
        ("inside except* that does not catch ImportError", "try:\n    import alpha\nexcept* ValueError:\n    pass\n"),
        ("inside a match case", "x = 1\nmatch x:\n    case 1:\n        import alpha\n"),
    ],
)
def test_these_shapes_execute_at_import_and_are_gated(tmp_path, label, source):
    """Every one of these raises ImportError at `import mod` if the package is gone."""
    unguarded, guarded = _CM._imports_of(_tree(tmp_path, source))
    assert "alpha" in unguarded, f"{label}: executes at import time and was not gated"
    assert "alpha" not in guarded


@pytest.mark.parametrize(
    ("label", "source"),
    [
        ("try/except ImportError", "try:\n    import alpha\nexcept ImportError:\n    alpha = None\n"),
        ("try/except ModuleNotFoundError", "try:\n    import alpha\nexcept ModuleNotFoundError:\n    alpha = None\n"),
        ("try/except tuple", "try:\n    import alpha\nexcept (OSError, ImportError):\n    alpha = None\n"),
        ("try/except Exception", "try:\n    import alpha\nexcept Exception:\n    alpha = None\n"),
        ("bare except", "try:\n    import alpha\nexcept:  # noqa: E722\n    alpha = None\n"),
        ("in the handler", "try:\n    import beta\nexcept ImportError:\n    import alpha\n"),
        (
            "contextlib.suppress(ImportError)",
            "import contextlib\nwith contextlib.suppress(ImportError):\n    import alpha\n",
        ),
        ("guarded inside an if", "if True:\n    try:\n        import alpha\n    except ImportError:\n        pass\n"),
    ],
)
def test_these_shapes_cannot_break_an_import_and_are_not_gated(tmp_path, label, source):
    unguarded, guarded = _CM._imports_of(_tree(tmp_path, source))
    assert "alpha" not in unguarded, f"{label}: cannot break an install and was gated anyway"
    assert "alpha" in guarded, f"{label}: must still be reported, not dropped"


def test_a_function_body_import_is_neither(tmp_path):
    """Lazy loads (#1930) are deliberate and there are hundreds; they are outside both sets."""
    unguarded, guarded = _CM._imports_of(_tree(tmp_path, "def f():\n    import alpha\n    return alpha\n"))
    assert "alpha" not in unguarded
    assert "alpha" not in guarded


def test_one_unguarded_site_outweighs_many_guarded_ones(tmp_path):
    """A package imported guardedly in ten files and bare in one still breaks that one file."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    for n in range(10):
        (pkg / f"soft{n}.py").write_text("try:\n    import alpha\nexcept ImportError:\n    alpha = None\n")
    (pkg / "hard.py").write_text("import alpha\n")
    unguarded, guarded = _CM._imports_of(pkg)
    assert "alpha" in unguarded
    assert "alpha" not in guarded, "a name with any unguarded site must not also be reported as optional"


def test_the_verdict_does_not_move_with_the_installed_set(tmp_path):
    """The control for the defect: with no distribution mapping, the answer must not change.

    Measured before the fallback existed: `colorlog`, `optax` and `ot` -- all real, all undeclared --
    were routed to an advisory list that never set the exit code, and six more joined them under
    CI's install.
    """
    pkg = _tree(tmp_path, "import alpha\nimport beta\n")
    declared = {"alpha"}
    real = _CM.importlib.metadata.packages_distributions
    try:
        _CM.importlib.metadata.packages_distributions = dict
        blind, _ = _CM._undeclared(pkg, declared)
    finally:
        _CM.importlib.metadata.packages_distributions = real
    assert blind == ["beta"], blind


def test_a_file_that_does_not_parse_is_refused_not_skipped(tmp_path):
    """A silent `continue` here makes the check quietest about the files most likely to be wrong.

    `check_single_source.py` raises on the same condition and `check_fail_fast.py` lets it
    propagate; #1629 records the ruling.
    """
    pkg = _tree(tmp_path, "import alpha\n")
    (pkg / "broken.py").write_text("def (:\n")
    with pytest.raises(SystemExit, match="does not parse"):
        _CM._imports_of(pkg)


def test_the_real_tree_is_clean_and_the_self_test_agrees():
    """The positive control for the two above: they must be measuring the shipped configuration."""
    assert _CM._self_test() == 0
    assert _CM.main.__module__ == "_check_manifests"


def test_the_name_map_is_not_identity():
    """`CONDA_TO_PYPI` went with `environment.yml` in #2167: there is no second manifest to
    normalise against, so a conda-to-PyPI rename had nothing left to rename."""
    assert not hasattr(_CM, "CONDA_TO_PYPI"), "the conda name map outlived the manifest it served"
    for name, target in _CM.IMPORT_TO_DISTRIBUTION.items():
        assert name != target, f"{name!r} maps to itself"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
