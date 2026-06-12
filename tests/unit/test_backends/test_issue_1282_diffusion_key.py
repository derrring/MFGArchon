"""Pinning tests for #1282 item 3: unify the backend diffusion-coefficient key.

Before the fix the four backends read the diffusion coefficient under three
different ``problem_params`` keys with three different semantics and defaults:

- numba  -> ``"sigma"``     (volatility, default 1.0)              -- canonical
- numpy  -> ``"sigma_sq"``  (sigma**2,   default 0.01)
- jax    -> ``"sigma_sq"``  (sigma**2,   default 0.01)
- torch  -> ``"diffusion"`` (volatility, default 0.1)

so a single ``problem_params`` dict could not be ported across backends. The
fix routes all four through ``resolve_volatility`` (prefer canonical ``"sigma"``;
fall back to the backend's legacy key with a one-time ``DeprecationWarning``;
else the backend's prior default).

Two pinning groups:

(1) Cross-backend portability -- one ``{"sigma": 0.3}`` dict yields the SAME
    effective ``D = sigma**2/2 = 0.045`` in every backend's ``fpk_step``.
(2) Backward-compat -- each backend's legacy key still resolves to the same
    sigma and emits a ``DeprecationWarning`` naming the canonical key.

Each group FAILs on the pre-fix code (legacy keys diverge / canonical ignored)
and PASSes after the fix.
"""

from __future__ import annotations

import warnings

import pytest

import numpy as np

from mfgarchon.utils import pde_coefficients
from mfgarchon.utils.pde_coefficients import resolve_volatility

# Per-backend (legacy_key, legacy_is_squared, default) — the exact arguments each
# backend's fpk_step passes to resolve_volatility.
BACKEND_DIFFUSION_CONFIG = {
    "numpy": {"legacy_key": "sigma_sq", "legacy_is_squared": True, "default": 0.1},
    "jax": {"legacy_key": "sigma_sq", "legacy_is_squared": True, "default": 0.1},
    "torch": {"legacy_key": "diffusion", "legacy_is_squared": False, "default": 0.1},
    "numba": {"legacy_key": None, "legacy_is_squared": False, "default": 1.0},
}

SIGMA = 0.3
EXPECTED_D = 0.5 * SIGMA * SIGMA  # 0.045


@pytest.fixture(autouse=True)
def _reset_legacy_warning_guard():
    """resolve_volatility warns only once per legacy key per process; clear the
    guard before each test so warning assertions are deterministic."""
    pde_coefficients._VOLATILITY_LEGACY_KEY_WARNED.clear()
    yield
    pde_coefficients._VOLATILITY_LEGACY_KEY_WARNED.clear()


# ---------------------------------------------------------------------------
# (1) Cross-backend portability at the resolver level (no backend libs needed)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend_name", list(BACKEND_DIFFUSION_CONFIG))
def test_canonical_sigma_gives_same_effective_D_all_backends(backend_name):
    """A single ``{"sigma": 0.3}`` dict resolves to sigma=0.3 (=> D=0.045) under
    every backend's resolver configuration."""
    cfg = BACKEND_DIFFUSION_CONFIG[backend_name]
    sigma = resolve_volatility({"sigma": SIGMA}, **cfg)
    assert sigma == SIGMA
    assert 0.5 * sigma * sigma == pytest.approx(EXPECTED_D)


def test_canonical_sigma_preferred_over_legacy_no_warning():
    """When both keys are present the canonical ``"sigma"`` wins and no
    DeprecationWarning is emitted."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes an error
        sigma = resolve_volatility(
            {"sigma": SIGMA, "sigma_sq": 0.09}, legacy_key="sigma_sq", legacy_is_squared=True, default=0.1
        )
    assert sigma == SIGMA


@pytest.mark.parametrize("backend_name", list(BACKEND_DIFFUSION_CONFIG))
def test_no_key_returns_backend_default(backend_name):
    """With neither canonical nor legacy key, the backend's prior default is
    returned and no warning is emitted (no behavior change for such callers)."""
    cfg = BACKEND_DIFFUSION_CONFIG[backend_name]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sigma = resolve_volatility({"unrelated": 1.0}, **cfg)
    assert sigma == cfg["default"]


# ---------------------------------------------------------------------------
# (2) Backward-compat: legacy keys still resolve + emit DeprecationWarning
# ---------------------------------------------------------------------------

# (legacy_key, legacy_is_squared, stored_value, expected_sigma)
LEGACY_CASES = [
    pytest.param("sigma_sq", True, 0.09, 0.3, id="numpy_jax_sigma_sq"),
    pytest.param("diffusion", False, 0.3, 0.3, id="torch_diffusion"),
]


@pytest.mark.parametrize(("legacy_key", "legacy_is_squared", "stored", "expected"), LEGACY_CASES)
def test_legacy_key_resolves_and_warns(legacy_key, legacy_is_squared, stored, expected):
    """Each legacy key resolves to the same sigma and emits a one-time
    DeprecationWarning naming the canonical 'sigma' key."""
    with pytest.warns(DeprecationWarning, match="sigma"):
        sigma = resolve_volatility(
            {legacy_key: stored}, legacy_key=legacy_key, legacy_is_squared=legacy_is_squared, default=0.1
        )
    assert sigma == pytest.approx(expected)


def test_legacy_warning_is_one_time_per_key():
    """The DeprecationWarning fires once per legacy key per process; a second
    resolve with the same legacy key is silent."""
    params = {"sigma_sq": 0.09}
    with pytest.warns(DeprecationWarning, match="sigma"):
        resolve_volatility(params, legacy_key="sigma_sq", legacy_is_squared=True, default=0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a second warning would raise
        sigma = resolve_volatility(params, legacy_key="sigma_sq", legacy_is_squared=True, default=0.1)
    assert sigma == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# (1') Cross-backend portability through the actual fpk_step code paths.
#
# For each available backend, ``fpk_step`` with the canonical ``{"sigma": 0.3}``
# dict must produce the IDENTICAL density evolution as ``fpk_step`` with that
# backend's own legacy key encoding sigma=0.3 (same backend, same discretisation,
# same dtype -> exact agreement).  Pre-fix the canonical key is ignored by
# numpy/jax/torch (they read sigma_sq/diffusion), so the two outputs diverge.
# ---------------------------------------------------------------------------


def _gaussian_density(nx: int = 41):
    dx = 1.0 / (nx - 1)
    x = np.linspace(0.0, 1.0, nx)
    M = np.exp(-0.5 * ((x - 0.5) / 0.15) ** 2)
    fn = getattr(np, "trapezoid", None) or np.trapz
    M = M / fn(M, dx=dx)
    U = np.zeros_like(x)  # zero drift -> pure diffusion isolates the D term
    return M, U, dx, x


def _to_numpy(arr):
    if hasattr(arr, "detach"):  # torch
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


# canonical dict and the legacy-key dict that encodes the SAME sigma=0.3
_LEGACY_ENCODING = {
    "numpy": {"sigma_sq": 0.09},
    "jax": {"sigma_sq": 0.09},
    "torch": {"diffusion": 0.3},
    "numba": {"sigma": 0.3},  # numba's canonical key IS its native key
}


def _make_backend(name):
    if name == "numpy":
        from mfgarchon.backends.numpy_backend import NumPyBackend

        return NumPyBackend()
    if name == "numba":
        pytest.importorskip("numba")
        from mfgarchon.backends.numba_backend import NumbaBackend

        return NumbaBackend()
    if name == "jax":
        pytest.importorskip("jax")
        from mfgarchon.backends.jax_backend import JAXBackend

        return JAXBackend()
    if name == "torch":
        pytest.importorskip("torch")
        from mfgarchon.backends.torch_backend import TorchBackend

        return TorchBackend()
    raise AssertionError(name)


@pytest.mark.parametrize("backend_name", list(BACKEND_DIFFUSION_CONFIG))
def test_canonical_sigma_dict_portable_through_fpk_step(backend_name):
    """``fpk_step({"sigma": 0.3})`` must match ``fpk_step({<legacy enc of 0.3>})``
    for the SAME backend.  Pre-fix numpy/jax/torch ignore the canonical key and
    fall to their default D, so the two outputs diverge -> this FAILs without the
    routing fix.  numba already used the canonical key (its case is the reference)."""
    backend = _make_backend(backend_name)
    M, U, dx, x = _gaussian_density()
    dt = 1e-3

    canonical = {"sigma": SIGMA, "x_grid": x}
    legacy = dict(_LEGACY_ENCODING[backend_name])
    legacy["x_grid"] = x

    out_canonical = _to_numpy(backend.fpk_step(M, U, dt, dx, canonical))
    out_legacy = _to_numpy(backend.fpk_step(M, U, dt, dx, legacy))

    np.testing.assert_allclose(
        out_canonical,
        out_legacy,
        rtol=1e-6,
        atol=1e-9,
        err_msg=(
            f"[{backend_name}] canonical {{'sigma': 0.3}} fpk_step output differs from the "
            f"legacy-key encoding of the same sigma -> the canonical key is not honored "
            f"(Issue #1282 item 3)."
        ),
    )


@pytest.mark.parametrize("backend_name", ["numpy", "jax", "torch"])
def test_canonical_sigma_changes_diffusion_vs_default(backend_name):
    """Guard against a resolver that always returns the default: for the backends
    whose default sigma (0.1) differs from 0.3, the canonical dict must change the
    diffusion-driven evolution relative to the no-key default."""
    backend = _make_backend(backend_name)
    M, U, dx, x = _gaussian_density()
    dt = 1e-3

    out_sigma = _to_numpy(backend.fpk_step(M, U, dt, dx, {"sigma": SIGMA, "x_grid": x}))
    out_default = _to_numpy(backend.fpk_step(M, U, dt, dx, {"x_grid": x}))

    assert not np.allclose(out_sigma, out_default, rtol=1e-3), (
        f"[{backend_name}] canonical sigma=0.3 produced the same output as the no-key "
        f"default; the diffusion coefficient is not being read from 'sigma'."
    )
