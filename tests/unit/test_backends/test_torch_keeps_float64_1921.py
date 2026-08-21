"""Issue #1921: the torch backend silently narrowed float64 to float32.

`TorchBackend(precision="float64")` -- the default -- returned float32, because `device="auto"`
selected MPS and MPS has no float64 at all. The backend warned and narrowed:

    warnings.warn("MPS does not support float64, using float32 instead. ...")

A `UserWarning` is not enough for that. This library's convergence tolerances are 1e-10 and 1e-12
throughout -- `hjb_residual_norm` against `DEFAULT_NEWTON_TOLERANCE`, the mass-conservation pins at
`abs=1e-12` -- so a float32 backend does not make them hard to hit, it makes them **unreachable in
principle**. Measured: `np.array([1.0 + 1e-10, 1.0])` came back with both entries equal.

THE ASYMMETRY, the same one as #1923. The precision was something the USER asked for; the device was
something the LIBRARY chose for them under `auto`. A choice made for the user must not override one
the user made. So `auto` now prefers CPU whenever float64 is requested -- which costs nothing here,
since torch is 9.2x-361x slower than numpy on this machine anyway -- and an EXPLICIT `device='mps'`
with `precision='float64'` raises, because there the caller asked for both and they genuinely
conflict.
"""

from __future__ import annotations

import pytest

import numpy as np

torch = pytest.importorskip("torch", reason="torch not installed")

from mfgarchon.backends import create_backend  # noqa: E402

# A difference float32 cannot hold.
_X = np.array([1.0 + 1e-10, 1.0], dtype=np.float64)


def test_the_default_torch_backend_keeps_float64():
    """The headline. Before, this came back float32 with both entries equal."""
    b = create_backend("torch")
    out = b.to_numpy(b.from_numpy(_X))
    assert out.dtype == np.float64, f"round trip narrowed to {out.dtype}"
    assert out[0] != out[1], (
        "1 + 1e-10 == 1 after the round trip: the backend is float32, and this library's 1e-10 / "
        "1e-12 tolerances are then unreachable in principle"
    )


def test_numpy_is_the_control():
    """Without it, a backend that returned constants would pass the test above."""
    b = create_backend("numpy")
    out = b.to_numpy(b.from_numpy(_X))
    assert out.dtype == np.float64
    assert out[0] != out[1]


def test_auto_does_not_select_mps_when_float64_is_asked_for():
    """The mechanism, not the symptom. `auto` is a choice made FOR the user; it must not spend the
    precision they asked for."""
    b = create_backend("torch", precision="float64")
    assert getattr(b, "device_type", None) != "mps", (
        "auto selected MPS under precision='float64'; MPS has no float64, so the precision would be narrowed silently"
    )
    assert b.precision == "float64", "the requested precision must survive device selection"


def test_auto_may_still_select_mps_for_float32():
    """Control: the skip is conditional on the precision, not a blanket refusal of MPS. Asserted
    only where MPS exists, so it does not become a platform assumption."""
    from mfgarchon.backends.torch_backend import MPS_AVAILABLE, MPS_FUNCTIONAL

    if not (MPS_AVAILABLE and MPS_FUNCTIONAL) or torch.cuda.is_available():
        pytest.skip("no MPS on this machine, or CUDA outranks it")
    b = create_backend("torch", precision="float32")
    assert b.device_type == "mps", "float32 must still reach MPS; the #1921 skip is precision-scoped"


def test_an_explicit_mps_float64_request_raises():
    """Both named by the caller, and they conflict. Silently resolving it is what this replaces."""
    from mfgarchon.backends.torch_backend import MPS_AVAILABLE, MPS_FUNCTIONAL

    if not (MPS_AVAILABLE and MPS_FUNCTIONAL):
        pytest.skip("no MPS on this machine")
    with pytest.raises(ValueError) as exc:
        create_backend("torch", device="mps", precision="float64")
    text = str(exc.value)
    assert "1921" in text
    # Both ways out, because which one is right depends on what the caller actually needs.
    assert "device='cpu'" in text, "the message must offer keeping float64"
    assert "precision='float32'" in text, "and it must offer keeping MPS"


def test_the_silent_narrowing_path_is_gone():
    """Pinned against the source: a warn-and-narrow is what made this survivable for so long."""
    import inspect

    from mfgarchon.backends import torch_backend

    src = "\n".join(
        ln
        for ln in inspect.getsource(torch_backend.TorchBackend._setup_backend).splitlines()
        if not ln.lstrip().startswith("#")
    )
    assert 'self.precision = "float32"' not in src, (
        "the backend reassigns its own precision again; a downgrade the caller did not ask for "
        "must raise, not be recorded after the fact (#1921)"
    )
