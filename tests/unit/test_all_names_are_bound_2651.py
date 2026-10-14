"""Every name a module lists in `__all__` is bound by that module (#2651).

At cb0724a9, `from mfgarchon import *` raised `AttributeError`. #1989 rebound `create_general_mfg_problem` in
`factory/problem_factories.py` as the unbuilt stub that raises #1920's `NotImplementedError`, but did not import it
into `mfgarchon.factory` or `mfgarchon`, whose `__all__` lists still named it. `mfgarchon.alg.optimization` listed two
variational factories it never imports, in the branch it takes when the optional optimal-transport dependency, POT,
is absent.

1. The star import of the package, in a fresh interpreter.
2. Every `__all__` entry of every module resolves. The walk runs in a child interpreter: importing every module in
   the test process would pre-import the renamed shim modules for every later test on the same xdist worker, and the
   warning census would lose their DeprecationWarnings (review 1 of #2653; `test_hl_convention.py` found it first).
   The helper is checked on a planted module, and the walk is checked against the files on disk, since a module it
   skips is a module this test does not check.
3. `mfgarchon.alg.optimization` in both states of the optional dependency, forced in a subprocess. The environment
   running the suite decides only one of them.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CARRIER = str(ROOT / "mfgarchon" / "__init__.py")


def _run(code: str) -> list[str]:
    """Run `code` in a fresh interpreter whose `mfgarchon` is this tree; the code prints its carrier first.

    The tree goes on the child's PYTHONPATH. The working directory is not enough: the gate runs pytest under
    PYTHONSAFEPATH=1, which a child inherits, so `-c` adds no directory and the editable install's checkout answers.
    """
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")]))}
    proc = subprocess.run(
        [sys.executable, "-B", "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=300, check=False
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return proc.stdout.splitlines()


def test_the_star_import_of_the_package_succeeds():
    code = (
        "import warnings; warnings.simplefilter('ignore')\n"
        "import mfgarchon; print(mfgarchon.__file__)\n"
        "from mfgarchon import *\n"
        "try:\n"
        "    create_general_mfg_problem()\n"
        "except NotImplementedError as exc:\n"
        "    print('refused', '#1920' in str(exc))\n"
    )
    assert _run(code) == [CARRIER, "refused True"]


# Run in a CHILD interpreter by `test_every_name_in_every_all_is_bound`; see item 2 of the module docstring for why.
_CHILD_WALK = r"""
import importlib, json, pathlib, pkgutil, types, warnings
warnings.simplefilter("ignore")
import mfgarchon

# Each marker starts a fresh line: a module the walk imports may write to stdout without a newline.
print("\nCARRIER", mfgarchon.__file__)


def unbound(module):
    return [name for name in getattr(module, "__all__", ()) if not hasattr(module, name)]


planted = types.ModuleType("planted")
planted.__all__ = ["present", "absent"]
planted.present = 1

modules, failures, walked = [mfgarchon], [], {"mfgarchon"}
for info in pkgutil.walk_packages(mfgarchon.__path__, prefix="mfgarchon.", onerror=failures.append):
    walked.add(info.name)
    try:
        modules.append(importlib.import_module(info.name))
    except Exception as exc:
        failures.append(f"{info.name}: {type(exc).__name__}: {exc}")

# walk_packages descends only into directories with an __init__.py, so its population is checked against the disk.
root = pathlib.Path(mfgarchon.__file__).parent
on_disk = set()
for path in root.rglob("*.py"):
    parts = ("mfgarchon",) + path.relative_to(root).with_suffix("").parts
    on_disk.add(".".join(parts[:-1] if parts[-1] == "__init__" else parts))

print("\nRESULT", json.dumps({
    "control": unbound(planted),
    "failures": failures,
    "unwalked": sorted(on_disk - walked),
    "with_all": sorted(m.__name__ for m in modules if hasattr(m, "__all__")),
    "unbound": {m.__name__: u for m in modules if (u := unbound(m))},
}))
"""


def test_every_name_in_every_all_is_bound():
    lines = _run(_CHILD_WALK)
    assert [line for line in lines if line.startswith("CARRIER ")] == [f"CARRIER {CARRIER}"]
    # The last RESULT line is the walk's own: one printed by an imported module comes before it.
    result = json.loads([line for line in lines if line.startswith("RESULT ")][-1][len("RESULT ") :])
    assert result["control"] == ["absent"], "the helper does not see an unbound name on a planted module"
    assert result["failures"] == [], f"modules the walk could not import, so their __all__ went unchecked: {result}"
    assert result["unwalked"] == [], f"modules on disk the walk never reached: {result['unwalked']}"
    assert {"mfgarchon", "mfgarchon.factory", "mfgarchon.alg.optimization"} <= set(result["with_all"])
    assert result["unbound"] == {}, f"__all__ names these modules do not bind, so their star import raises: {result}"


FORCE = {
    False: "import sys; sys.modules['ot'] = None",
    True: "import sys, types; sys.modules['ot'] = types.ModuleType('ot')",
}


@pytest.mark.parametrize("pot", [False, True], ids=["pot-absent", "pot-present"])
def test_the_optimization_package_binds_its_all_with_and_without_pot(pot):
    code = (
        f"{FORCE[pot]}\n"
        "import warnings; warnings.simplefilter('ignore')\n"
        "import mfgarchon; print(mfgarchon.__file__)\n"
        "import mfgarchon.alg.optimization as opt\n"
        "print(opt.OT_IMPORTS_AVAILABLE)\n"
        "print([name for name in opt.__all__ if not hasattr(opt, name)])\n"
        "from mfgarchon.alg.optimization import *\n"
        "print('star import ok')\n"
    )
    assert _run(code) == [CARRIER, str(pot), "[]", "star import ok"]
