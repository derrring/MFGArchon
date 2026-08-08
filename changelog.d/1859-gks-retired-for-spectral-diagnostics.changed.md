**`check_gks_stability` is retired.** `mfgarchon.geometry.boundary.validation` now exports
`spectral_diagnostics`, which reports measured quantities and returns **no stability verdict**.

The retired function did not implement GKS. GKS is a normal-mode theory for hyperbolic
initial-boundary value problems producing a Kreiss determinant condition on the boundary scheme;
what was computed was the spectrum of a semi-discrete operator. One signature served three
incompatible theories and all three branches were defective: `parabolic` sampled the wrong end of
the spectrum above N=100 and reported stable for an operator with `max Re(λ) = +0.1`; `hyperbolic`
compared `max|Im(λ)|` against `10·max|λ|`, which holds for every complex number, so it reported
stable for every input ever constructed; `elliptic` answered a whole-spectrum question from a
one-ended truncated sample. `check_gks_convergence` is removed with them.

`spectral_diagnostics` reports the spectral abscissa `max Re(λ)` (necessary only — `> 0` proves
growth, `≤ 0` proves nothing for non-normal operators, and every one-sided boundary stencil
produces one) and adds the **numerical abscissa** `λ_max((L+Lᴴ)/2)`, which is sufficient:
`‖exp(tL)‖₂ ≤ exp(t·ω)`. Both are reported; neither is thresholded. It also reports whether the
whole spectrum was actually seen, since a truncated solve cannot support a whole-spectrum claim.

Boundary-condition correctness is validated here by exactness against an external field, discrete
conservation, and measured order of accuracy under refinement — which is what production PDE
libraries ship. A survey found none ships a GKS/Kreiss–Lopatinskii checker; the one research
package that does is restricted to 1D scalar constant-coefficient explicit schemes and does not
enforce its own CFL hypothesis. (#1859)
