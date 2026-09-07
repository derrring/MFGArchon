- **Choosing the NumPy backend no longer imports cvxpy or numba** (Issue #1922).
  `import mfgarchon.backends.numpy_backend` loaded both: cvxpy through the package's only
  module-level `import cvxpy` (`gfdm_components/joint_socp.py`), numba through
  `utils/performance/optimization.py`, reached from `utils/__init__.py`. Each is now imported on
  first use by a cached helper, and every reader of the corresponding availability flag was moved
  to it in the same change — three gates for cvxpy (covering all 27 uses of `cp`), four readers for
  numba.

  **The flags keep their meaning.** `True` still means the import *succeeded*, not that the package
  is discoverable. Swapping in `importlib.util.find_spec` would have been a smaller diff and a
  behaviour change: the two answers diverge on an installed-but-broken package, which is exactly the
  case these callers degrade gracefully on, so it would have turned a handled fallback into a crash
  inside the solver. This repository already carries that divergence — `hjb_gfdm.py` computes a
  `CVXPY_AVAILABLE` from `find_spec` while `joint_socp.py` computes one from the import.

  Guarded by a **timing** assertion in `test_optional_backends_are_not_imported_eagerly.py`, not a
  scope one: it constrains *when* these may be imported, never *where* they may be used, so any
  number of deferred call sites satisfy it. It skips loudly rather than passing vacuously when the
  package is absent — CI installs no extras, so an unguarded assertion would have been green on
  every runner while measuring nothing.

  Not addressed here: jax, which reaches `sys.modules` by a different route
  (`utils/acceleration/__init__.py` runs `import jax` to compute `HAS_JAX`) and needs that package's
  jax-backed names re-exported lazily.
