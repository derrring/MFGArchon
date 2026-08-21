"""Issue #1943: `is_on_boundary`'s tolerance argument had two spellings.

`GeometryProtocol` declares `tolerance=`, and eighteen implementers use it. The `ImplicitDomain`
family used `tol=`. Positional calls worked either way, so **any keyword call broke on one side of
the split**, with which side decided only by which geometry the caller happened to be handed:

    Hyperrectangle.is_on_boundary(pts, tol=0.03)        -> [True, False]
    Hyperrectangle.is_on_boundary(pts, tolerance=0.03)  -> TypeError

And `ImplicitApplicator._detect_boundary_points` used the keyword form — an applicator that exists
*for* this family, calling it in the spelling this family did not accept.

The issue counts seven diverging classes; measured, six of them **inherit** the method, so there was
one definition to change. The census below asserts against the resolved signature rather than the
count, since a subclass could override it back.
"""

from __future__ import annotations

import inspect
import pkgutil
import warnings
from importlib import import_module

import pytest

import numpy as np

import mfgarchon.geometry as _geometry
from mfgarchon.geometry.implicit.hyperrectangle import Hyperrectangle

_PTS = np.array([[0.0, 0.5], [0.5, 0.5]])


def _box():
    return Hyperrectangle(bounds=np.array([[0.0, 1.0], [0.0, 1.0]]))


def test_the_protocol_spelling_now_works_on_the_implicit_family():
    """The assertion #1943 is about."""
    assert np.array_equal(_box().is_on_boundary(_PTS, tolerance=0.03), np.array([True, False]))


def test_the_old_spelling_still_works_and_warns():
    """Deprecated, not removed: `tol=` is a released spelling and the two must not disagree while
    both exist."""
    box = _box()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        old = box.is_on_boundary(_PTS, tol=0.03)
    assert [w.category for w in caught] == [DeprecationWarning]
    assert np.array_equal(old, box.is_on_boundary(_PTS, tolerance=0.03)), (
        "the alias must be the same argument, not a second one that drifts"
    )


def test_passing_both_is_refused():
    """They are one argument. Accepting both would silently pick a winner, which is the class of
    defect this fix removes rather than one to introduce."""
    with pytest.raises(TypeError, match="same"):
        _box().is_on_boundary(_PTS, tolerance=0.03, tol=0.03)


def test_the_positional_call_is_unchanged():
    """Control: positional calls worked on both sides before, so a rename that broke them would be
    a regression the keyword tests could not see."""
    assert np.array_equal(_box().is_on_boundary(_PTS, 0.03), np.array([True, False]))


def test_every_implementer_accepts_the_protocol_spelling():
    """The gate, and it is behavioural rather than a name count: a subclass could re-override with
    `tol=` and a signature-name census keyed on the base would miss it.

    Walks `mfgarchon.geometry`, so a new geometry joins the census by existing.
    """
    offenders = []
    seen = set()
    for module in pkgutil.walk_packages(_geometry.__path__, _geometry.__name__ + "."):
        try:
            mod = import_module(module.name)
        except Exception:  # a module that cannot import contributes no implementer
            continue
        for obj in vars(mod).values():
            if not inspect.isclass(obj) or obj.__name__ in seen:
                continue
            fn = getattr(obj, "is_on_boundary", None)
            if fn is None or not callable(fn):
                continue
            seen.add(obj.__name__)
            try:
                params = inspect.signature(fn).parameters
            except (TypeError, ValueError):
                continue
            if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
                continue  # **kwargs accepts anything; a separate concern (#2020)
            if "tolerance" not in params:
                offenders.append(f"{obj.__name__}{inspect.signature(fn)}")

    assert "TensorProductGrid" in seen, "the walk found no known implementer -- the query is wrong"
    assert len(seen) >= 10, f"expected the geometry population, found {len(seen)}"
    assert offenders == [], (
        f"GeometryProtocol declares `tolerance=`; these do not accept it: {offenders}. A caller "
        f"cannot then write one keyword call that works for every geometry (#1943)."
    )
