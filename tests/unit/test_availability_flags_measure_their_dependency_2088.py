"""An availability flag reads `False` when its dependency is absent (#2088).

polars: `mfgarchon.utils.POLARS_AVAILABLE` and `mfgarchon.utils.data.POLARS_AVAILABLE` were set `True` by a guard around
`utils.data.polars_integration`, which imports without polars, so they read `True` without it. They now re-export that
module's own flag. psutil's two flags were deleted when psutil became a plain import (#2930): a required package has no
availability flag.

Every case runs in a fresh interpreter pointed at this tree explicitly, because under `PYTHONSAFEPATH=1` it would
resolve `mfgarchon` from the editable install instead (see `test_utils_does_not_import_alg.py`). The case refuses the
dependency before `mfgarchon` is first imported: a meta-path finder raises `ModuleNotFoundError` for it, and
`importlib.util.find_spec` returns `None` for it, as on an install that lacks it. The child prints the module it
loaded, and the test checks it.
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
