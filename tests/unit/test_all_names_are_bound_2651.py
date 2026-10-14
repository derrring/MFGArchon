"""Every name a module lists in `__all__` is bound by that module (#2651).

At cb0724a9, `from mfgarchon import *` raised `AttributeError`. #1989 rebound `create_general_mfg_problem` in
`factory/problem_factories.py` as the unbuilt stub that raises #1920's `NotImplementedError`, but did not import it
into `mfgarchon.factory` or `mfgarchon`, whose `__all__` lists still named it. `mfgarchon.alg.optimization` listed two
variational factories it never imports, in the branch it takes when the optional optimal-transport dependency, POT,
is absent.

1. The star import of the package, in a fresh interpreter.
2. Every `__all__` entry of every module resolves. The helper is checked on a planted module first, and the walk must
   import every module, since a module it skips is a module this test does not check.
3. `mfgarchon.alg.optimization` in both states of the optional dependency, forced in a subprocess. The environment
   running the suite decides only one of them.
"""

from __future__ import annotations

import importlib
import os
import pkgutil
import subprocess
import sys
import types
import warnings
from pathlib import Path

import pytest

import mfgarchon

ROOT = Path(__file__).resolve().parents[2]
CARRIER = str(ROOT / "mfgarchon" / "__init__.py")


def _unbound(module) -> list[str]:
    return [name for name in getattr(module, "__all__", ()) if not hasattr(module, name)]


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


def test_every_name_in_every_all_is_bound():
    planted = types.ModuleType("planted")
    planted.__all__ = ["present", "absent"]
    planted.present = 1
    assert _unbound(planted) == ["absent"]

    modules, failures = [mfgarchon], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for info in pkgutil.walk_packages(mfgarchon.__path__, prefix="mfgarchon."):
            try:
                modules.append(importlib.import_module(info.name))
            except Exception as exc:
                failures.append(f"{info.name}: {type(exc).__name__}: {exc}")
    assert failures == [], f"modules the walk could not import, so their __all__ went unchecked: {failures}"
    names = {m.__name__ for m in modules if hasattr(m, "__all__")}
    assert {"mfgarchon", "mfgarchon.factory", "mfgarchon.alg.optimization"} <= names
    unbound = {m.__name__: u for m in modules if (u := _unbound(m))}
    assert unbound == {}, f"__all__ names these modules do not bind, so their star import raises: {unbound}"


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
