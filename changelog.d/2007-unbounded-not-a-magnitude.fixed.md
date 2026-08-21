The `gradient_*` non-conservation warning now states that the loss is **unbounded in the wall-normal
drift**, rather than quoting a single figure (#2007). Both earlier revisions — `O(1e-2), even with
zero drift`, then `-1.4e-1 at A = 0.7` — were true of their fixture and both read as a bounded error
a caller could budget against, which is the part that is false. Re-measured across cell Péclet
`v·dx/D` at `n=81, sigma=0.3, T=0.5` with the drift normal at both walls: `0 → +0.5%`,
`0.19 → -23.6%`, `0.89 → -99.97%`, against `divergence_upwind` at `0.0000%` for all three. At the
last of those the returned density is a relaxed uniform field, not an under-resolved correct one,
and the warning now says so. The schemes remain explicitly selectable — the standing decision in
`test_gradient_centered_still_available_and_leaks` is untouched.
