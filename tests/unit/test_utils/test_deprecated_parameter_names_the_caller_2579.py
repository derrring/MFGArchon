"""A deprecated parameter's warning names the caller's file and line, under stacked decorators too (#2579).

Under two stacked `@deprecated_parameter` decorators the frame above the inner wrapper is the outer
wrapper, so a plain ``stacklevel=2`` attributed the inner warning to `mfgarchon/utils/deprecation.py`.
pytest.ini's ``ignore::DeprecationWarning:mfgarchon.*`` filters that module, so a call site left on the
inner parameter was invisible to the suite and to the warning census: `TensorProductGrid(dimension=...)`
×34 in the CI tier. The decorator now skips every frame of its own module (`WRAPPER_FRAMES`).

`pytest.warns` cannot see attribution, so these read ``warning.filename`` and ``warning.lineno``. With the
skip prefix mutated to the bare ``__file__``, which CPython 3.12 does not match against itself, the
stacked pair goes back to naming deprecation.py and the single-decorator control does not move.
"""

from __future__ import annotations

import warnings

from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.utils.deprecation import deprecated_parameter


@deprecated_parameter(param_name="old_outer", since="v0.0.0", replacement="new_outer")
@deprecated_parameter(param_name="old_inner", since="v0.0.0", replacement="new_inner")
def _stacked(new_outer=None, new_inner=None, old_outer=None, old_inner=None):
    return None


@deprecated_parameter(param_name="old_only", since="v0.0.0", replacement="new_only")
def _single(new_only=None, old_only=None):
    return None


def _attributions(call) -> dict[str, tuple[str, int]]:
    """``{deprecated parameter: (filename, lineno)}`` for each deprecated-parameter warning ``call`` raises."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        call()
    return {
        str(w.message).split("'")[1]: (w.filename, w.lineno)
        for w in caught
        if issubclass(w.category, DeprecationWarning) and "is deprecated" in str(w.message)
    }


def _call_line(call) -> int:
    """The line of the call in ``call``'s body, which starts on the line after its ``def``."""
    return call.__code__.co_firstlineno + 1


def test_both_warnings_of_a_stacked_pair_name_the_caller():
    def call():
        _stacked(old_outer=1, old_inner=2)

    assert _attributions(call) == {"old_outer": (__file__, _call_line(call)), "old_inner": (__file__, _call_line(call))}


def test_a_single_decorator_names_the_caller():
    """The control: a single wrapper was attributed correctly before the fix too."""

    def call():
        _single(old_only=1)

    assert _attributions(call) == {"old_only": (__file__, _call_line(call))}


def test_the_library_stacked_pair_names_the_caller():
    """`TensorProductGrid.__init__` stacks `num_points` over `dimension`, the census's 34 hidden call sites."""
    bc = no_flux_bc(dimension=1)

    def call():
        TensorProductGrid(bounds=[(0.0, 1.0)], num_points=[11], dimension=1, boundary_conditions=bc)

    line = _call_line(call)
    assert _attributions(call) == {"num_points": (__file__, line), "dimension": (__file__, line)}
