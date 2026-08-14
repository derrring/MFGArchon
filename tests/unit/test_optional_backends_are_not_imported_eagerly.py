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
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


# `cwd=REPO` alone does NOT pin which tree a probe imports. For `python -c`, `sys.path[0]` is the
# current directory and precedes PYTHONPATH -- so cwd wins, until `-P` / `PYTHONSAFEPATH=1` strips
# it, and `scripts/local_ci.sh:311` sets exactly that (deliberately: xdist workers do not inherit
# `-P`). `subprocess.run` inherits the environment, so under the gate the cwd entry is gone, nothing
# replaces it, and the probe imports whatever the EDITABLE INSTALL points at. On the canonical
# checkout the two trees coincide and the gate is honest; in a worktree -- which this repo MANDATES
# for pre-merge review -- they differ, and the probe reports on a tree that was never under test.
# Recorded from this repo's own #1906 review: an import-origin check is only evidence if the wrong
# answer is reachable. Hence both halves: put the tree back on the path, and have each probe print
# `mfgarchon.__file__` so the caller can assert it rather than trust the plumbing.
_PROBE_ENV = {
    **os.environ,
    "PYTHONPATH": os.pathsep.join(x for x in (str(REPO), os.environ.get("PYTHONPATH", "")) if x),
}

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
        env=_PROBE_ENV,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, f"the probe never ran, so it measured nothing:\n{proc.stderr[-1500:]}"
    line = [x for x in proc.stdout.splitlines() if x.startswith("{")]
    assert line, f"the probe produced no verdict:\n{proc.stdout[-800:]}\n{proc.stderr[-800:]}"
    return json.loads(line[-1])


def test_the_subprocess_probes_import_the_tree_under_test():
    """Validate the instrument before trusting any number it produces.

    Every probe in this file is a fresh interpreter, and which `mfgarchon` that interpreter finds
    is decided by `sys.path`, not by which directory the test file lives in. `cwd=REPO` puts the
    tree first only while `sys.path[0]` is the cwd; the gate runs under `PYTHONSAFEPATH=1`, which
    removes that entry, and `subprocess.run` inherits it -- so without `_PROBE_ENV` the probes
    report on the editable install. That is invisible on the canonical checkout, where the two
    coincide, and wrong in a worktree, where they do not.

    This does not stop the other probes from measuring the wrong tree; it makes the whole file go
    red when they would, which is the point. A check that only confirms the good case is
    decoration -- so the failure it names must be reachable, and in a worktree it is.
    """
    proc = subprocess.run(
        [sys.executable, "-c", "import json, mfgarchon; print(json.dumps({'tree': mfgarchon.__file__}))"],
        cwd=REPO,
        env=_PROBE_ENV,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, f"the instrument check never ran:\n{proc.stderr[-1500:]}"
    line = [x for x in proc.stdout.splitlines() if x.startswith("{")]
    assert line, f"no verdict:\n{proc.stdout[-600:]}\n{proc.stderr[-600:]}"
    tree = json.loads(line[-1])["tree"]
    # `resolve().is_relative_to`, not `startswith`: on macOS TMPDIR is a symlink, so a worktree's
    # logical and physical spellings differ and a string prefix compares False on a correct tree.
    assert Path(tree).resolve().is_relative_to(REPO.resolve()), (
        f"the probes import {tree}, not the tree under test at {REPO}. Every subprocess result in "
        f"this file is therefore about a different checkout. Check PYTHONSAFEPATH/PYTHONPATH."
    )


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
    loaded = _modules_after("import sys, json, decimal\nprint(json.dumps({'decimal': 'decimal' in sys.modules}))\n")
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


@pytest.mark.parametrize("name", LAZY_TORCH_NAMES)
def test_each_deferred_name_still_resolves(name):
    """The deferral must not be a removal, per name.

    Review found the whole `__getattr__` could be replaced with `raise AttributeError` and the
    full suite still returned 6006 passed: **no file in the repository references any of these
    nine names through `mfgarchon.utils.acceleration`**, so nothing exercised the machinery this
    change introduced. The two mutation tests above pin that torch is not imported EAGERLY; they
    say nothing about whether it can be imported at all.
    """
    # No `importorskip`: `torch_utils.py` wraps `import torch` in `try/except ImportError` and
    # publishes `HAS_TORCH = False`, so it imports cleanly without torch and every name below
    # resolves. Marking these `optional_torch` excluded them from every automated tier -- the gate,
    # nightly and python-compat all deselect that marker -- so the machinery they exist to pin was
    # covered by tests that never ran. Found by review, which also measured that they pass with
    # torch genuinely absent. The torch-less user is what CI is; that is the case worth pinning.
    import mfgarchon.utils.acceleration as acceleration

    resolved = getattr(acceleration, name)
    assert resolved is not None, f"{name} resolved to None"
    # Resolving twice must give the same object: `__getattr__` binds into `globals()` on first
    # use, and a second call that re-imported would be a different object each time.
    assert getattr(acceleration, name) is resolved


def test_the_renamed_export_points_at_the_right_function():
    """`torch_tridiagonal_solve` is `torch_utils.tridiagonal_solve` under another name — the one
    entry in the lazy map whose source attribute differs from its exported one, and the only
    place a copy-paste error would be silent."""
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
    `__getattr__` call is in both.

    **The binding happens inside this test.** Without it the two sets are disjoint and the buggy
    `sorted([...])` form produces no duplicates -- so the test passed with the defect present when
    run alone (1 passed) and under the gate's marker set (6 passed, 12 deselected), because the
    `optional_torch` cases that would have bound the names are excluded there. Only an unfiltered
    whole-file run caught it. That is the same failure mode as the registry test earlier in this
    branch: an assertion about a module global satisfied by whatever ran first. Found by review;
    order-independence is not something the surrounding suite can supply.
    """
    import mfgarchon.utils.acceleration as acceleration

    # `to_tensor`, not `TORCH_UTILS_AVAILABLE`. Not because one predates the branch -- neither
    # does; `git show 7ac9df18:mfgarchon/utils/acceleration/__init__.py | grep -c _LAZY_TORCH` is 0
    # and the whole map is new here. The distinction is that the flag's COMPUTATION was revised
    # twice inside this PR while `to_tensor` was not, so binding the flag would rest this test's
    # precondition on a decision the same PR is still moving. Measured: with the flag reverted to
    # eager AND the `sorted([...])` defect restored, the old form of this test goes green when run
    # alone; the `to_tensor` form reddens under the dir defect alone, torch-present and torch-free.
    acceleration.to_tensor  # noqa: B018 - bind one lazy name, which is the precondition
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

    # The precondition here is UNBOUND, and something else in the suite binds it. Not the dir test
    # above -- that binds `to_tensor`. The binder is
    # `tests/integration/test_cross_backend_consistency.py:16`, whose module-level
    # `from mfgarchon.utils.acceleration import ... TORCH_UTILS_AVAILABLE` triggers the PEP 562
    # `__getattr__`, which caches via `globals()[name] = value`. That runs at COLLECTION, so it has
    # already happened in every xdist worker before any test body starts, and then the bare lookup
    # inside `get_acceleration_info` finds the name in `__dict__` and no `NameError` can fire.
    # Measured with all three files under `-n 4`, 3 repeats: both call sites reverted -> 1 failed
    # with this pop, 61 passed without it. `vars(...).pop`, not `del`: `del` raises
    # `AttributeError` when nothing has bound the name yet.
    vars(acceleration).pop("TORCH_UTILS_AVAILABLE", None)

    info = acceleration.get_acceleration_info()
    assert "torch_utils_available" in info
    assert isinstance(info["torch_utils_available"], bool)
    assert "jax_utils_available" in info

    # Pin the lookup ORDER, not just the value. `__getattr__("TORCH_UTILS_AVAILABLE")` reaches the
    # module `__getattr__` directly and so skips `__dict__`, i.e. it reads PAST an explicit
    # override; `sys.modules[__name__].TORCH_UTILS_AVAILABLE` is normal attribute lookup and does
    # not.
    #
    # BOTH directions, and that is the whole point. Overriding only to `False` is vacuous wherever
    # torch does not import: `__getattr__` returns `bool(torch_utils.HAS_TORCH)` = False there, so
    # the reverted form satisfies the assertion and the pin is silent on exactly the machines CI
    # runs on (review measured 17 passed torch-free with both call sites reverted). Hardcoding the
    # dict entry to `False` survives torch-PRESENT for the mirror reason. Looping catches both,
    # because no single constant can equal both members of the loop.
    for override in (True, False):
        acceleration.TORCH_UTILS_AVAILABLE = override
        try:
            assert acceleration.get_acceleration_info()["torch_utils_available"] is override, (
                f"the helper reported {acceleration.get_acceleration_info()['torch_utils_available']!r} "
                f"under an explicit override of {override!r}: it is not going through attribute lookup"
            )
        finally:
            vars(acceleration).pop("TORCH_UTILS_AVAILABLE", None)


def test_the_availability_flag_reports_whether_torch_IMPORTS_not_whether_it_exists():
    """An installed-but-broken torch must give `False`.

    The first version of this deferral used `importlib.util.find_spec("torch") is not None`, which
    answers "is torch on disk". For a torch that is installed but raises on import, the eager form
    it replaced gave `False` and that gives `True` — and `get_acceleration_info()` then reported the
    self-contradictory `{"torch_utils_available": True, "torch_available": False}`.

    The fix was unpinned: review reverted it and the whole suite stayed green, because nothing in
    the tree simulates a broken torch. The guard that DID catch the other wrong fix — importing
    `torch_utils` eagerly — only sees the eagerness failure, not the semantics.
    """
    import textwrap

    code = textwrap.dedent(
        """
        import json, pathlib, sys, tempfile

        tmp = tempfile.mkdtemp()
        pkg = pathlib.Path(tmp) / "torch"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("raise ImportError('broken on purpose')\\n")
        sys.path.insert(0, tmp)

        import importlib.util
        # control: the shadow must be BOTH findable and un-importable, or this proves nothing
        assert importlib.util.find_spec("torch") is not None, "control: the shadow torch is not findable"
        try:
            import torch
            print(json.dumps({"control": "FAILED - the shadow torch imported"}))
            raise SystemExit(0)
        except ImportError:
            pass

        import mfgarchon.utils.acceleration as acc
        print(json.dumps({
            "control": "ok",
            "flag": bool(acc.TORCH_UTILS_AVAILABLE),
            "info": bool(acc.get_acceleration_info()["torch_utils_available"]),
        }))
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        env=_PROBE_ENV,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, f"the probe never ran:\n{proc.stderr[-1500:]}"
    line = [x for x in proc.stdout.splitlines() if x.startswith("{")]
    assert line, f"no verdict:\n{proc.stdout[-600:]}\n{proc.stderr[-600:]}"
    import json as _json

    got = _json.loads(line[-1])
    assert got["control"] == "ok", got["control"]
    assert got["flag"] is False, (
        "TORCH_UTILS_AVAILABLE is True for a torch that does not import. The flag answers "
        "'is torch on disk' rather than 'does importing it succeed'."
    )
    assert got["info"] is False, "get_acceleration_info() disagrees with the flag"
