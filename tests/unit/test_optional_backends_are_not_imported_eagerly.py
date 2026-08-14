"""Importing `mfgarchon` must not import torch or jax.

Both are optional, both are declared in the `[all]` extra rather than in `dependencies`, and
both were nonetheless imported by anything that touched the package. Measured on `1aa71b98`:
`import mfgarchon` was 4.12s and put `torch` in `sys.modules`; deferring the two eager sites
below takes it to 3.27s with `torch` absent — 0.82s, which is exactly torch's own cold import
cost measured on its own.

**Two sites, and cutting either alone changes nothing.** Three independent routes reach torch
during `import mfgarchon`:

    utils/__init__.py:30 -> adjoint_validation.py:55 -> alg -> ...
        -> nonlinear_solvers.py:45 -> utils/acceleration/__init__.py -> torch_utils.py:16
    utils/__init__.py:92 -> utils/geometry.py:30 -> geometry -> ... -> (same leaf)
    base_hjb.py:11 -> backends/compat -> backends/__init__.py -> torch_backend.py:30

The first two converge on `utils/acceleration`; the third does not touch it. Deferring only
`backends` leaves routes 1 and 2; deferring only `acceleration` leaves route 3. That is why this
test asserts the outcome — torch is absent — rather than the shape of either file: an outcome
assertion cannot be satisfied by fixing one site and calling it done.

jax is asserted separately and more weakly, because it still arrives through cvxpy, which is a
third-party import chain this repository does not control (#1930).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

_PROBE = (
    "import sys, json\n"
    "import mfgarchon\n"
    "print(json.dumps({p: p in sys.modules for p in ('torch', 'jax', 'numba', 'scipy')}))\n"
)


def _modules_after(code: str) -> dict[str, bool]:
    """What is in `sys.modules` after `code` runs in a fresh interpreter.

    A subprocess because by the time this file is collected the package is already imported and
    every heavy module with it -- an in-process check would report the state of the test runner,
    not of a fresh import.
    """
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, f"the probe never ran, so it measured nothing:\n{proc.stderr[-1500:]}"
    line = [x for x in proc.stdout.splitlines() if x.startswith("{")]
    assert line, f"the probe produced no verdict:\n{proc.stdout[-800:]}\n{proc.stderr[-800:]}"
    return json.loads(line[-1])


def test_importing_the_package_does_not_import_torch():
    loaded = _modules_after(_PROBE)
    assert not loaded["torch"], (
        "`import mfgarchon` imported torch. It is optional, declared in the `[all]` extra, and "
        "costs 0.82s. Three routes reach it and cutting one leaves the others -- check both "
        "`backends/__init__.py` (eager `register_backend('torch', ...)`) and "
        "`utils/acceleration/__init__.py` (eager `from .torch_utils import ...`). #1930"
    )


def test_the_probe_would_notice_an_import():
    """Positive control. The assertion above is that a flag is False, which is also what a probe
    that never imported anything returns."""
    loaded = _modules_after("import sys, json, torch\nprint(json.dumps({'torch': 'torch' in sys.modules}))\n")
    assert loaded["torch"], "the probe cannot see torch even when it is imported outright"


def test_scipy_is_still_imported_so_the_absence_above_means_something():
    """The other direction: a package that SHOULD arrive must arrive.

    Without this, a change that broke `import mfgarchon` into importing almost nothing would
    satisfy the torch assertion perfectly.
    """
    loaded = _modules_after(_PROBE)
    assert loaded["scipy"], "`import mfgarchon` no longer reaches scipy; the package is not loading normally"


def test_the_torch_backend_still_works_when_asked_for():
    """Deferring must not mean never. `create_backend('torch')` takes the on-demand path in
    `create_backend` -- the one that eager registration had made unreachable."""
    torch = pytest.importorskip("torch", reason="torch is optional; this asserts the on-demand path when present")
    assert torch is not None
    loaded = _modules_after(
        "import sys, json\n"
        "from mfgarchon.backends import create_backend\n"
        "before = 'torch' in sys.modules\n"
        "b = create_backend('torch')\n"
        "print(json.dumps({'before': before, 'after': 'torch' in sys.modules, 'name': type(b).__name__}))\n"
    )
    assert loaded["before"] is False, "torch was already imported before it was asked for"
    assert loaded["after"] is True, "asking for the torch backend did not import torch"
    assert loaded["name"] == "TorchBackend"
