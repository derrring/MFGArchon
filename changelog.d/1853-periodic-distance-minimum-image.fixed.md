`compute_periodic_distance` now returns the true minimum image on `Hyperrectangle` and
`TensorProductGrid`. Both implemented the rule inline as `min(|d|, L - |d|)`, correct only for
`|d| <= L`: on a unit torus they returned 0.9 for a separation of 1.9 (true answer 0.1) and 6.7
for 7.7 — larger than the domain. Both now delegate to `wrap_displacement`, taking the count of
minimum-image implementations in the tree from 3 to 1 (#1841, #1847), and the rule stated
normatively in the `SupportsPeriodic` protocol — from which both had been transcribed — is
corrected with it. For **floating-point** input, distances with `|d| <= L` are unchanged
bit-for-bit (0 ULP over ~82k samples per period, across eight periods, both classes). Integer
input with a non-integral period shifts, because both the old and new formulations truncate on
write-back into an integer array; that path was, and remains, wrong in both. (#1853)
