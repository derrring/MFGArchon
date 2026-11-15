"""`mfgarchon.utils` is below `mfgarchon.alg` and must not import upward.

The inversion is what creates the cycle that blocks decoupling, and that is why this file exists.

~~Every heavy package enters through one line~~ and ~~`import mfgarchon` is 4.89s~~ [RETRACTED
2026-08-14, see #1930] — both were refuted by measurement after this file was written. It was not
a single entry point: with that line cut, torch arrived through `utils/__init__.py`'s
`from .geometry import` → `utils/geometry.py`, instead of its `from .adjoint_validation import`
→ `adjoint_validation.py`. `utils/__init__.py` was then an
18-import, 106-name re-export hub, so cutting edges one at a time was a treadmill — all three
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

That line could not simply be made lazy while `utils/numerical/__init__.py` imported `alg`. Removing it,
or replacing it with a PEP 562 `__getattr__`, both failed with

    ImportError: cannot import name 'clip_nonnegative_or_raise' from partially initialized
    module 'mfgarchon.utils.numerical' (most likely due to a circular import)

because `alg` reached `fp_network`, which needs a name from `utils.numerical` while that file is still
stopped at the import. The eager import completed `utils.numerical` by another route first and hid the
cycle. Once `utils/numerical/__init__.py` stopped importing `alg` (#1930), #2938's stage 2a deleted the
line, with every other upward re-export in `utils/__init__.py`.

The count of module-level `utils` -> `alg` imports this file held, with its scanner, was deleted when
`scripts/check_layering.py`, the layering rule's one owner, entered the gate (#2938). Where the two
disagreed, the layer table's reading stands: `utils/adjoint_validation.py` -> `alg.base_solver` lies
within one layer. The fourteen statement shapes and the five relative cases its tests held are the first
fourteen of that script's `STATEMENT_SHAPES` and the first five of its `RELATIVE_CASES`. What stays here
is not a count: the deleted shims stay deleted, and a lazy `__getattr__` cannot restore them.
"""

from __future__ import annotations

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
    """The file check above sees files; a PEP 562 `__getattr__` adds none.

    Review demonstrated the hole: appending a module-level `__getattr__` to
    `utils/numerical/__init__.py` that resolves `gfdm_strategies` and `anderson_acceleration` to
    their canonical `alg` modules restores BOTH deprecated import paths -- DeprecationWarning and
    all -- while adding no file and no module-level import. Every assertion this file then held
    stayed green.

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
            f"import, so the file check above cannot see it. It was deleted as "
            f"a shim overdue since v0.21.0; import from `mfgarchon.alg` directly."
        )

    # Positive control: the canonical paths MUST import, or the loop above passes because the
    # package is broken rather than because the shims are gone.
    for path in (
        "mfgarchon.alg.numerical.gfdm_components.gfdm_strategies",
        "mfgarchon.alg.numerical.coupling.anderson_acceleration",
    ):
        importlib.import_module(path)
