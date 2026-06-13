"""Deprecation + equivalence tests for the renamed integro-differential package.

Issue #1024: ``operators.nonlocal_ops`` is renamed to ``operators.integro_diff``.
The old path is retained as a deprecation shim that must, per the repository
deprecation policy, (a) emit ``DeprecationWarning`` on import and (b) redirect to
the new location, re-exporting *identical* class objects (immediate redirect +
equivalence).
"""

from __future__ import annotations

import importlib
import sys

import pytest


def _fresh_import(name: str):
    """Import ``name`` after evicting the shim package and its aliased submodules.

    The shim emits its ``DeprecationWarning`` at module-execution time, which only
    fires on a fresh (uncached) import. Evicting the cached modules lets the
    warning be observed deterministically regardless of test ordering.
    """
    for mod in list(sys.modules):
        if mod == "mfgarchon.operators.nonlocal_ops" or mod.startswith("mfgarchon.operators.nonlocal_ops."):
            del sys.modules[mod]
    return importlib.import_module(name)


def test_old_path_emits_deprecation_warning():
    with pytest.warns(DeprecationWarning, match="integro_diff"):
        _fresh_import("mfgarchon.operators.nonlocal_ops")


def test_package_level_symbols_are_identical():
    import mfgarchon.operators.integro_diff as new

    with pytest.warns(DeprecationWarning, match="integro_diff"):
        old = _fresh_import("mfgarchon.operators.nonlocal_ops")

    assert new.__all__  # guard against an empty public surface
    assert list(old.__all__) == list(new.__all__)
    for symbol in new.__all__:
        assert getattr(old, symbol) is getattr(new, symbol), symbol


def test_submodule_paths_resolve_to_same_objects():
    from mfgarchon.operators.integro_diff.graphon_coupling import (
        GraphonCouplingOperator as NewGraphon,
    )
    from mfgarchon.operators.integro_diff.graphon_kernels import (
        ConstantGraphon as NewConstantGraphon,
    )
    from mfgarchon.operators.integro_diff.levy_integro_diff import (
        LevyIntegroDiffOperator as NewLevy,
    )
    from mfgarchon.operators.integro_diff.levy_measures import (
        GaussianJumps as NewGaussian,
    )

    with pytest.warns(DeprecationWarning, match="integro_diff"):
        _fresh_import("mfgarchon.operators.nonlocal_ops")

    # Old dotted submodule paths must still resolve, to the same class objects.
    from mfgarchon.operators.nonlocal_ops.graphon_coupling import (
        GraphonCouplingOperator as OldGraphon,
    )
    from mfgarchon.operators.nonlocal_ops.graphon_kernels import (
        ConstantGraphon as OldConstantGraphon,
    )
    from mfgarchon.operators.nonlocal_ops.levy_integro_diff import (
        LevyIntegroDiffOperator as OldLevy,
    )
    from mfgarchon.operators.nonlocal_ops.levy_measures import (
        GaussianJumps as OldGaussian,
    )

    assert OldLevy is NewLevy
    assert OldGaussian is NewGaussian
    assert OldGraphon is NewGraphon
    assert OldConstantGraphon is NewConstantGraphon
