- **The 1-D FDM Jacobian's diffusion block is now built from the operator the residual applies**
  (Issue #1894). It restated the interior three-point stencil, which is right in the interior and
  wrong at both ends: the true end rows are `[-1, 1]/dx²` under no-flux and `[-3, 1]/dx²` under
  Dirichlet against the hardcoded `[-2, 1]/dx²`. Measured by directional finite difference of
  `compute_hjb_residual`, `|Jv - dF/dv|` at row 0 was `6.5e-02 / 4.0e-01 / 1.6e+00 / 6.5e+00` at
  σ = `0.1 / 0.25 / 0.5 / 1.0` — the ratio to σ² constant at **6.4568 to five figures** — and is now
  `3.4e-09 / 3.2e-09 / 5.0e-09 / 7.7e-09`. The interior is unchanged at `3.3e-06`, which is the
  finite-difference truncation floor rather than zero.
- **Why nothing caught it.** A wrong Jacobian makes Newton converge slowly or not at all rather than
  return a wrong answer — the residual decides the root — so it surfaced as inner-solver stalls
  (#1878) and as the outer iteration consuming non-roots (#1873), neither of which points at a
  boundary. And it is identically absent at σ = 0, which is what the `fdm_upwind` capability fixture
  runs: every prior measurement of this Jacobian, including #1882's, was taken where the defect
  cannot appear.
- **Extracted, not restated.** `_bc_laplacian_bands` recovers the bands from
  `_compute_laplacian_1d` itself with three comb probes — the operator is tridiagonal, so within a
  row only one of `{i-1, i, i+1}` is congruent mod 3 and the columns do not alias. A second
  implementation of this operator is what caused the defect, so the fix does not add one. The
  tridiagonality is asserted against a fourth probe rather than assumed: BC providers resolve at
  iteration time (#574), so the shape cannot be taken on trust from the type.
- **Cost**: the FD fallback is unchanged (`4.28` vs `4.25` ms at Nx=21, `359` vs `361` ms at 1601 —
  four O(Nx) applications against an O(Nx²) loop). The analytic path stays O(Nx) and sub-millisecond
  (`0.89` vs `0.13` ms at Nx=1601), so #1607's reason for it survives.
- **Oracle**: `tests/unit/test_alg/test_hjb_jacobian_matches_residual_1894.py`, external — a
  directional finite difference of the residual is a law the Jacobian must reproduce, computed
  independently of it. Mutation-verified: restoring the hardcoded stencil turns 4 of 7 red and leaves
  the σ = 0 row green, which is what a control for this defect should do; removing the
  tridiagonality assertion turns exactly the test written for it red.
- **Two more shapes the first fix assumed away, both found by the gate rather than by reasoning.**
  Periodic BC makes the operator *cyclic* tridiagonal, which a `[-1, 0, +1]` band structure cannot
  hold — the hardcoded stencil was not wrong on the entries it had, it was two short. And the wrap's
  *position* depends on the convention: exclusive puts it at `(0, Nx-1)`, while
  `ENDPOINT_INCLUSIVE`, which `TensorProductGrid` uses, treats node 0 and node `Nx-1` as the same
  physical point and puts it at `(0, Nx-2)`. Assuming that position broke every grid-built periodic
  solve; the same hazard is #1832.
  Obstacle masks, source terms and nonlocal operators then produced operators no banded probe can
  attribute at all. So the extraction is two-tier: comb probes recover a banded operator in O(Nx),
  a control vector none of them was built from decides whether that held, and one probe per column —
  exact for any *linear* operator — is used when it did not. A nonlinear operator has no Jacobian to
  extract and raises.
- **Two tests moved, and neither was adjusted to match.**
  `test_jacobian_byte_identical_to_inline_assembly` compared against a reference that restated the
  same hardcoded stencil, so it pinned the defect rather than its own subject — which is #1071's
  inline `dp` form. The reference now takes its diffusion from the same operator; the diffusion half
  is tautological there and is pinned instead by the new external oracle.
  The golden baselines were regenerated. The **residual is untouched**, so the root they approach is
  unchanged; what moved is where a deliberately truncated iteration (`max_iterations=3`) stands after
  three sweeps with a corrected Newton step — 92 of 231 elements, max absolute difference `3.49e-08`.
  Recorded in the fixture file beside the #1745 and #1420 regenerations.
