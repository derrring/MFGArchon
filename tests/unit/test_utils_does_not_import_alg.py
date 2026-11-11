"""`mfgarchon.utils` is below `mfgarchon.alg` and must not import upward.

The inversion is what creates the cycle that blocks decoupling, and that is why this file exists.

~~Every heavy package enters through one line~~ and ~~`import mfgarchon` is 4.89s~~ [RETRACTED
2026-08-14, see #1930] — both were refuted by measurement after this file was written. It is not
a single entry point: with that line cut, torch arrives through `utils/__init__.py`'s
`from .geometry import` → `utils/geometry.py`, instead of its `from .adjoint_validation import`
→ `adjoint_validation.py`. `utils/__init__.py` is an
18-import, 106-name re-export hub, so cutting edges one at a time is a treadmill — all three
planned cuts applied together moved the total 4.31s → 4.62s, which is to say not at all.

~~torch arrives via `utils/data/polars_integration.py`~~ [CORRECTED 2026-08-14] — that named
the wrong witness. The probe watched `find_spec("torch")`, and polars PROBES for torch for its
`to_torch()` interop without importing it; `import polars` alone leaves `torch` out of
`sys.modules`. Re-measured by intercepting the module's actual execution: both routes converge
on the same leaf, `nonlinear_solvers.py` → `utils/acceleration/__init__.py` →
`torch_utils.py`. That convergence is the more useful fact and neither the original analysis
nor its first correction had it. The absolute figure is a property of
one process anyway (4.89s here, 6.37s cold and 4.04s warm elsewhere); what reproduces is the
decomposition, roughly half third-party and half the library importing itself.

A retracted rationale living on in a permanent test file is worse than none, which is why it is
struck here rather than quietly rewritten.

That line cannot simply be made lazy. Removing it, or replacing it with a PEP 562 `__getattr__`,
both fail with

    ImportError: cannot import name 'clip_nonnegative_or_raise' from partially initialized
    module 'mfgarchon.utils.numerical' (most likely due to a circular import)

because `utils/numerical/__init__.py` imports `alg`, which reaches `fp_network`, which needs a
name from `utils.numerical` while it is still stopped at that line. The eager import is
load-bearing: it completes `utils.numerical` by another route first and hides the cycle. #1930.

The count of module-level `utils` -> `alg` imports this file held, with its scanner, moved to
`scripts/check_layering.py`, the layering rule's one owner, which the gate runs (#2938). Where the
two disagreed, the layer table's reading stands: `utils/adjoint_validation.py` -> `alg.base_solver`
lies within one layer. The scanner's fourteen statement shapes and five relative cases are that
script's `STATEMENT_SHAPES` and `RELATIVE_CASES`. What stays here is not a count: the deleted shims
stay deleted, a lazy `__getattr__` cannot restore them, and the hub import can be deferred.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
UTILS = REPO / "mfgarchon" / "utils"


def test_the_deleted_shims_stay_deleted():
    """The two deprecation shims deleted from `utils/numerical/`, named, so neither comes back to keep an old import
    path alive."""
    for name in ("anderson_acceleration.py", "gfdm_strategies.py"):
        shim = UTILS / "numerical" / name
        assert not shim.exists(), (
            f"{shim.relative_to(REPO)} is back. It was a re-export of the canonical module in "
            f"`alg`, overdue for removal since v0.21.0. Import from `mfgarchon.alg` directly."
        )


def test_a_lazy_shim_cannot_restore_a_deleted_import_path():
    """The guard above counts imports; a PEP 562 `__getattr__` is not one.

    Review demonstrated the hole: appending a module-level `__getattr__` to
    `utils/numerical/__init__.py` that resolves `gfdm_strategies` and `anderson_acceleration` to
    their canonical `alg` modules restores BOTH deprecated import paths -- DeprecationWarning and
    all -- while adding no file and no module-level import. Every assertion above stayed green.

    That is not a hypothetical construction here: #1930's own step 4 proposes making
    `utils/__init__.py` a PEP 562 lazy re-export hub. This asserts the deleted paths in
    particular stay dead, whatever mechanism is used, so that work cannot restore them by
    accident on its way past.
    """
    import importlib

    parent = importlib.import_module("mfgarchon.utils.numerical")
    for name in ("gfdm_strategies", "anderson_acceleration"):
        full = f"mfgarchon.utils.numerical.{name}"
        # BOTH forms, because they take different routes. `import_module` does NOT consult a
        # module-level `__getattr__` -- verified: a package with `__getattr__` returning a module
        # answers `getattr(pkg, "sub")` and still raises ModuleNotFoundError for
        # `import_module("pkg.sub")`. A test that checked only the first would have passed over
        # exactly the lazy shim it exists to catch, which is how the first version of this test
        # was written.
        try:
            importlib.import_module(full)
        except ModuleNotFoundError:
            pass
        else:
            raise AssertionError(f"{full} imports again as a module.")
        resolved = getattr(parent, name, None)
        assert resolved is None, (
            f"`mfgarchon.utils.numerical.{name}` resolves again, as an attribute rather than a "
            f"file -- a lazy re-export restores the deprecated path while adding no module-level "
            f"import, so neither the count nor the file check above can see it. It was deleted as "
            f"a shim overdue since v0.21.0; import from `mfgarchon.alg` directly."
        )

    # Positive control: the canonical paths MUST import, or the loop above passes because the
    # package is broken rather than because the shims are gone.
    for path in (
        "mfgarchon.alg.numerical.gfdm_components.gfdm_strategies",
        "mfgarchon.alg.numerical.coupling.anderson_acceleration",
    ):
        importlib.import_module(path)


def test_the_hub_can_be_deferred_without_a_circular_import():
    """The cycle is broken, asserted as the thing that was blocked rather than as its cause.

    `utils/__init__.py`'s eager `from .adjoint_validation import (...)` is what makes every heavy
    package arrive on `import mfgarchon`, and #1930 step 5 is to defer it. Before the
    `gfdm_strategies` re-export was removed from `utils/numerical/__init__.py`, that deferral
    failed:

        ImportError: cannot import name 'clip_nonnegative_or_raise' from partially initialized
        module 'mfgarchon.utils.numerical' (most likely due to a circular import)

    because `utils.numerical` imported `alg`, which reached `fp_network.py`, which needed a name
    from `utils.numerical` while it was still executing that import. The eager hub import hid the
    cycle by completing `utils.numerical` through another route first -- so the cycle was only
    observable by attempting the very change it blocked.

    This performs the attempt in a subprocess against a patched copy of the tree, so the property
    is checked rather than remembered. If it ever fails again, an upward import has returned.
    """
    import shutil
    import subprocess
    import sys
    import tempfile

    hub = REPO / "mfgarchon" / "utils" / "__init__.py"
    source = hub.read_text()
    match = re.search(
        r"^# Adjoint duality validation \(Issue #580\)\nfrom \.adjoint_validation import \(([^)]*)\)\n", source, re.M
    )
    assert match, "the eager hub import moved; this test is pinning something that is no longer there"
    names = tuple(n.strip().rstrip(",") for n in match.group(1).split() if n.strip().rstrip(","))
    assert names, "parsed no re-exported names; the deferral below would prove nothing"

    lazy = (
        f"_LAZY = {names!r}\n\n\n"
        "def __getattr__(name):\n"
        "    if name in _LAZY:\n"
        "        from . import adjoint_validation as _m\n\n"
        "        return getattr(_m, name)\n"
        "    raise AttributeError(name)\n"
    )

    with tempfile.TemporaryDirectory() as tmp:
        tree = Path(tmp) / "tree"
        shutil.copytree(REPO / "mfgarchon", tree / "mfgarchon", ignore=shutil.ignore_patterns("__pycache__"))
        (tree / "mfgarchon" / "utils" / "__init__.py").write_text(source.replace(match.group(0), lazy, 1))
        # `PYTHONPATH` and an explicit `sys.path` insert, not cwd: `local_ci.sh` runs pytest
        # under `PYTHONSAFEPATH=1`, which drops cwd from `sys.path` -- and the subprocess then
        # resolved `mfgarchon` from the EDITABLE INSTALL, which imports fine, so this test passed
        # over a tree it never read. Measured: back-edge restored + PYTHONSAFEPATH=1 -> 1 passed,
        # same mutation without it -> 1 failed. Found by review.
        env = {**os.environ}
        # Prepend, do not overwrite: the previous form dropped the parent's PYTHONPATH, which
        # nothing here needs but a future runner might.
        env["PYTHONPATH"] = os.pathsep.join(x for x in (str(tree), os.environ.get("PYTHONPATH", "")) if x)
        # `pop`, not `= ""`. The empty string does clear the flag (CPython treats empty env vars
        # as unset), but it stays visible to anything checking presence rather than truth, and
        # "0" and " " both ENABLE it -- a value worth not having to remember.
        env.pop("PYTHONSAFEPATH", None)
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                f"import sys; sys.path.insert(0, {str(tree)!r})\nimport mfgarchon; print('TREE:' + mfgarchon.__file__)",
            ],
            cwd=tree,
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )

    assert proc.returncode == 0, (
        "deferring `utils/__init__.py`'s eager import of `adjoint_validation` fails, which means "
        "an upward `utils -> alg` import has reintroduced the cycle:\n" + proc.stderr[-2000:]
    )
    # The token identifies the tree, not just success. "OK" would be printed just as happily by
    # the installed package, which is the whole failure mode above.
    reported = [line[len("TREE:") :] for line in proc.stdout.splitlines() if line.startswith("TREE:")]
    assert reported, f"the probe printed no tree token:\n{proc.stdout[-600:]}"
    # `resolve().is_relative_to`, not `startswith`. On macOS TMPDIR is a symlink, so the logical
    # spelling (`/var/folders/...`) and the physical one `os.getcwd()` returns
    # (`/private/var/folders/...`) differ, and a string prefix compares False on the correct
    # tree. It passed only because `sys.path.insert` fed the import machinery the same string
    # this line compares against -- a guard whose correctness depended on a co-located
    # mechanism, which is the hazard this whole assertion exists to remove. Review measured it:
    # with the insert taken away the CLEAN tree reddens too, and a check red in both states is
    # stuck rather than discriminating.
    assert Path(reported[0]).resolve().is_relative_to(tree.resolve()), (
        f"the subprocess imported {reported[0]}, not the patched copy at {tree}. The result says "
        f"nothing about the patch."
    )
