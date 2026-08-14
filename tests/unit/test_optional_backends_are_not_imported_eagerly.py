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

jax is NOT asserted here. It still arrives through cvxpy, a third-party chain this repository
does not control, so there is no outcome to pin yet (#1930). The probe collects its presence for
the record only. (~~asserted separately and more weakly~~ [CORRECTED 2026-08-14] -- there was no
such assertion; the sentence described a test that was never written.)
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
    """Positive control, on a module that is always installed.

    ~~`import torch`~~ [CORRECTED 2026-08-14] -- torch is optional, declared in `[all]`, and is
    **not installed on any CI runner**: every workflow installs `-e .[dev]` or
    `-e ".[dev,numerical]"`. This control therefore failed on nightly, on `python-compat` for
    3.12/3.13/3.14, and on the release tier, while `test_importing_the_package_does_not_import_torch`
    was VACUOUSLY satisfied there -- the guard discriminated only on a machine with `[all]`
    installed. Found by review; the control now uses a module the runners have.
    """
    loaded = _modules_after(
        "import sys, json, decimal\nprint(json.dumps({'decimal': 'decimal' in sys.modules}))\n"
    )
    assert loaded["decimal"], "the probe cannot see a module even when it is imported outright"


@pytest.mark.optional_torch
def test_the_probe_would_notice_torch_specifically():
    """The same control for torch itself, where torch exists. Marked, so it is skipped rather
    than failed on the runners -- `test_the_probe_would_notice_an_import` above keeps the probe
    mechanism covered everywhere."""
    pytest.importorskip("torch", reason="torch is optional and absent on CI runners")
    loaded = _modules_after("import sys, json, torch\nprint(json.dumps({'torch': 'torch' in sys.modules}))\n")
    assert loaded["torch"], "the probe cannot see torch even when it is imported outright"


def test_scipy_is_still_imported_so_the_absence_above_means_something():
    """The other direction: a package that SHOULD arrive must arrive.

    Without this, a change that broke `import mfgarchon` into importing almost nothing would
    satisfy the torch assertion perfectly.
    """
    loaded = _modules_after(_PROBE)
    assert loaded["scipy"], "`import mfgarchon` no longer reaches scipy; the package is not loading normally"


@pytest.mark.optional_torch
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


LAZY_TORCH_NAMES = (
    "HAS_CUDA",
    "HAS_MPS",
    "HAS_TORCH",
    "GaussianKDE",
    "ensure_torch_available",
    "get_default_device",
    "to_numpy",
    "to_tensor",
    "torch_tridiagonal_solve",
)


@pytest.mark.optional_torch
@pytest.mark.parametrize("name", LAZY_TORCH_NAMES)
def test_each_deferred_name_still_resolves(name):
    """The deferral must not be a removal, per name.

    Review found the whole `__getattr__` could be replaced with `raise AttributeError` and the
    full suite still returned 6006 passed: **no file in the repository references any of these
    nine names through `mfgarchon.utils.acceleration`**, so nothing exercised the machinery this
    change introduced. The two mutation tests above pin that torch is not imported EAGERLY; they
    say nothing about whether it can be imported at all.
    """
    pytest.importorskip("torch", reason="torch is optional and absent on CI runners")
    import mfgarchon.utils.acceleration as acceleration

    resolved = getattr(acceleration, name)
    assert resolved is not None, f"{name} resolved to None"
    # Resolving twice must give the same object: `__getattr__` binds into `globals()` on first
    # use, and a second call that re-imported would be a different object each time.
    assert getattr(acceleration, name) is resolved


@pytest.mark.optional_torch
def test_the_renamed_export_points_at_the_right_function():
    """`torch_tridiagonal_solve` is `torch_utils.tridiagonal_solve` under another name — the one
    entry in the lazy map whose source attribute differs from its exported one, and the only
    place a copy-paste error would be silent."""
    pytest.importorskip("torch", reason="torch is optional and absent on CI runners")
    import mfgarchon.utils.acceleration as acceleration
    from mfgarchon.utils.acceleration import torch_utils

    assert acceleration.torch_tridiagonal_solve is torch_utils.tridiagonal_solve


def test_an_unknown_name_raises_attribute_error_not_import_error():
    """`hasattr` depends on this. `__getattr__` must raise `AttributeError` for a name it does
    not own; raising `ImportError` would make `hasattr` propagate instead of returning False."""
    import mfgarchon.utils.acceleration as acceleration

    with pytest.raises(AttributeError):
        _ = acceleration.definitely_not_a_real_name

    assert hasattr(acceleration, "get_acceleration_info"), "a real attribute stopped resolving"


def test_dir_lists_each_deferred_name_once():
    """`__dir__` merges `globals()` with the lazy map, and a name bound by an earlier
    `__getattr__` call is in both. Review measured three names listed twice."""
    import mfgarchon.utils.acceleration as acceleration

    listing = dir(acceleration)
    duplicated = sorted({n for n in listing if listing.count(n) > 1})
    assert not duplicated, f"dir() lists these more than once: {duplicated}"


def test_the_introspection_helper_still_works():
    """`get_acceleration_info()` reads `TORCH_UTILS_AVAILABLE`, which is now lazy.

    A module-level `__getattr__` is **not** consulted for a bare global lookup inside a function
    in the same module — that goes straight to `module.__dict__` and raises `NameError`. Deferring
    the flag therefore broke this function, and every test above still passed: they exercise
    import behaviour, not the package's own callers. `ruff F821` is what caught it.
    """
    import mfgarchon.utils.acceleration as acceleration

    info = acceleration.get_acceleration_info()
    assert "torch_utils_available" in info
    assert isinstance(info["torch_utils_available"], bool)
    assert "jax_utils_available" in info
