`compute_periodic_distance` now returns the true minimum image on `Hyperrectangle` and
`TensorProductGrid`. Both implemented the rule inline as `min(|d|, L - |d|)`, correct only for
`|d| <= L`: on a unit torus they returned 0.9 for a separation of 1.9 (true answer 0.1) and 6.7
for 7.7 — larger than the domain. Both now delegate to `wrap_displacement`, taking the count of
minimum-image implementations in the tree from 3 to 1 (#1841, #1847), and the rule stated
normatively in the `SupportsPeriodic` protocol — from which both had been transcribed — is
corrected with it. Both methods now delegate to a new `periodic_distance` owner beside
`wrap_displacement`, so the shape and dtype handling has one home too.

Distances for `|d| <= L` are unchanged on the eight values pinned in
`test_periodic_distance_minimum_image_1853.py`, captured by executing the pre-change
implementations and asserted with `==`. Two paths that were previously wrong are now fixed
rather than carried: integer input no longer truncates on write-back (it promoted to float, so
points half a non-integral period apart no longer report distance 0), and rank-3 or higher
input now raises instead of wrapping a single slab and returning a correctly-shaped array of
wrong distances. (#1853)
