The `gradient_*` non-conservation warning states the drift dependence instead of understating the
leak by an order of magnitude (#2007).

It read *"leaks O(1e-2), even with zero drift"*. Measured:

| configuration | leak |
|---|---|
| genuinely zero drift, stationary initial density | **−1.7e-14** (none at all) |
| wall-normal drift A = 0.7, D = 1/8, d = 1 | **−1.4e-1** |
| wall-normal drift A = 0.7, D = 1/8, d = 2 | **−8.9e-1** |

So the `1.5e-2` figure was a **transient** density, which the string did not say, and a reader
budgeting against `O(1e-2)` was off by 10× exactly where the scheme is used.

The message also now names the **second** defect, because a reader who hears only "does not conserve
mass" will reach for a conservative wall and that would not fix it: `div(αm) = α·∇m + m ∇·α`, and the
gradient form drops the second term, so it does not discretize the FP operator even away from the
wall. Measured on a source-free instance, repointing the wall moves the error from 5.81e-1 to
8.02e-1 — EOC −0.007 → 0.108.

**What this change deliberately does not do.** #2007 recommends removing these schemes, and that
recommendation stands. It is not settled here: `test_gradient_centered_still_available_and_leaks`
records a standing decision to keep them explicitly selectable, and a warning is not the place to
overturn a recorded decision. A first attempt at this change refused the scheme under `no_flux` and
turned that test red — which is how the standing decision was found.

The corrected figures are pinned, including that `O(1e-2)` may appear only as a retraction and not as
a live claim, with a control confirming the `divergence_*` schemes do not warn.
