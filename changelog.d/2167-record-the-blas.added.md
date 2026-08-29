- **The warning census records which BLAS produced it, and on which platform** (Issue #2167, step 5).
  Moving off conda changes the BLAS: measured on this machine, the PyPI numpy wheel links Apple
  Accelerate while the conda one links a generic `blas`. That is a real numerical change, and
  without this field the identity drift it causes is unattributable — the exact failure #2158 exists
  to prevent, arriving through a packaging decision instead of a version bump. Recorded as
  `numpy-blas`, `scipy-blas` and `platform`.
- **`numpy-blas` alone carries the transition**, even though its first value does not name an
  implementation: `blas` under conda, `accelerate` after the move. Measured in both environments.
- **Lowercased, because numpy and scipy disagree on the case of the same implementation** — numpy
  says `accelerate` and scipy says `Accelerate` on one PyPI environment. Compared raw across a
  re-baseline that reads as a package moving when nothing did. Both modules are read, because they
  are built independently and can genuinely differ.
- **A runtime reading via `threadpoolctl` was designed, measured, and then dropped.** It does name
  what conda hides — `openblas 0.3.33` in this project's environment, where `show_config` says only
  `blas` — but nothing here installs threadpoolctl, so the field is null in the environment being
  left **and** in the one being moved to, and its thread count moves with `OMP_NUM_THREADS`, which
  `.uvrc` sets to 4 and `ci.yml` sets to 1. A field that is null everywhere today and noisy the day
  it is not is a bad trade; recording drift on an axis nobody changed is the same objection that
  rejected the fuller platform string, and applying it to one field and not the other was the
  inconsistency. #2167 records what adding it would take.
- **The prior art is unanimous against this platform granularity, and the comment now says so.**
  `pandas.show_versions()`, `polars.show_versions()` and scikit-learn's all record
  `platform.platform()` — `macOS-26.5.2-arm64-arm-64bit` here. This records `darwin-arm64`. The
  reason is not that the finer string is wrong: it is that the OS version selects a *wheel*, and the
  wheel's BLAS is recorded directly, so a macOS upgrade that moves numpy from OpenBLAS to Accelerate
  shows up as `numpy-blas` changing. Presenting the prior art as endorsing the granularity, which an
  earlier version of this entry did, was a misreading of it.
- **Nothing bound `numpy-blas` to numpy.** Swapping the two right-hand sides passed the whole suite.
  The first fix did not close it either: it called `_blas_of` twice with marker configs and the
  mutation survived, because the swap is in the payload assembly and the test exercised the
  function. Driven through the hook now.
- **The machine half of `platform` was asserted by nothing** — the test checked
  `startswith(sys.platform)`, so dropping `platform.machine()` passed. That is the half that selects
  the wheel.
- **`.lower()` sat outside the `try`**, so a non-string name raised out of `pytest_terminal_summary`
  and lost the whole census. `bytes` was worse: `b"openblas".lower()` succeeds, so the value reached
  `json.dumps`, which raised past the payload's `except OSError`. `isinstance(name, str)`, inside
  the try.
- A dead `if scipy is not None` guard is gone — `_blas_of(None)` already answers `None` — and a
  redundant `import numpy` inside the hook is gone, the module already imports it as `np`.
- Eight mutations killed, including both field swaps, both halves of `platform`, the lowercase, the
  bytes path, and the `show_config` failure returning `""` instead of `None`.
