"""An availability flag reads `False` when its dependency is absent, and the package still imports (#2088).

- polars: `mfgarchon.utils.POLARS_AVAILABLE` and `mfgarchon.utils.data.POLARS_AVAILABLE` were set `True` by a guard
  around `utils.data.polars_integration`, which imports without polars, so they read `True` without it. They now
  re-export that module's own flag.
- psutil: `utils.memory_management` and `utils.performance.monitoring` import psutil at module top, and AGENTS.md calls
  psutil optional (#2930). Their guards in `mfgarchon.utils` stay, narrowed to psutil's absence, so without it the
  package imports and both flags read `False`.

Each case runs in a fresh interpreter, with the dependency refused before `mfgarchon` is first imported: a meta-path
finder raises `ModuleNotFoundError` for it, and `importlib.util.find_spec` returns `None` for it, as on an install that
lacks it. The child is pointed at this tree explicitly, because under `PYTHONSAFEPATH=1` it would resolve `mfgarchon`
from the editable install instead (see `test_utils_does_not_import_alg.py`), and it prints the module it loaded.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

_PROBE = """
import importlib, importlib.abc, importlib.util, json, sys
sys.path.insert(0, {repo!r})
ABSENT = {absent!r}

class _Absent(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] == ABSENT:
            raise ModuleNotFoundError(f"No module named {{name!r}}", name=name)
        return None

sys.meta_path.insert(0, _Absent())
_find_spec = importlib.util.find_spec
importlib.util.find_spec = lambda name, package=None: None if name.split(".")[0] == ABSENT else _find_spec(name, package)

import mfgarchon.utils as utils
print(json.dumps({{"tree": utils.__file__, "flags": {{f: getattr(importlib.import_module(m), f.split(".")[-1])
                                                      for m, f in {flags!r}}}}}))
"""

CASES = {
    "polars": [
        ("mfgarchon.utils", "utils.POLARS_AVAILABLE"),
        ("mfgarchon.utils.data", "utils.data.POLARS_AVAILABLE"),
        ("mfgarchon.utils.data.polars_integration", "polars_integration.POLARS_AVAILABLE"),
    ],
    "psutil": [
        ("mfgarchon.utils", "utils.MEMORY_MANAGEMENT_AVAILABLE"),
        ("mfgarchon.utils", "utils.PERFORMANCE_MONITORING_AVAILABLE"),
    ],
}


@pytest.mark.parametrize("absent", sorted(CASES))
def test_a_flag_reads_false_without_its_dependency_and_the_package_imports(absent):
    env = {**os.environ}
    env.pop("PYTHONSAFEPATH", None)
    code = _PROBE.format(repo=str(REPO), absent=absent, flags=CASES[absent])
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300, env=env)
    assert proc.returncode == 0, f"the package did not import without {absent}:\n{proc.stderr[-1500:]}"
    got = json.loads(proc.stdout.strip().splitlines()[-1])
    assert got["tree"].startswith(str(REPO / "mfgarchon")), got["tree"]
    assert len(got["flags"]) == len(CASES[absent]), got["flags"]
    assert got["flags"] == dict.fromkeys(got["flags"], False), got["flags"]


_NOT_PSUTIL = """
import importlib.abc, sys
sys.path.insert(0, {repo!r})

class _Broken(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == {module!r}:
            raise ModuleNotFoundError("No module named 'planted_dependency'", name="planted_dependency")
        return None

sys.meta_path.insert(0, _Broken())
import mfgarchon.utils
"""


@pytest.mark.parametrize("module", ["mfgarchon.utils.memory_management", "mfgarchon.utils.performance.monitoring"])
def test_a_psutil_guard_lets_any_other_import_failure_through(module):
    """The two guards stay for psutil alone. A module they guard that fails to import for any other reason is a defect,
    and the guard must not turn it into a `False` flag, which is what `except ImportError` did at the base."""
    env = {**os.environ}
    env.pop("PYTHONSAFEPATH", None)
    code = _NOT_PSUTIL.format(repo=str(REPO), module=module)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300, env=env)
    assert proc.returncode != 0, f"importing mfgarchon.utils swallowed a failure of {module} that is not psutil's"
    assert "planted_dependency" in proc.stderr, proc.stderr[-1500:]
