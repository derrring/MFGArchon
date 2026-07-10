- **GFDM precompute log names the real SOCP-infeasible fallback** (Issue #1565). The
  post-precompute INFO log claimed SOCP-infeasible nodes "fall back to M-matrix QP (Phase 2)", but
  the SOCP loop iterates interior nodes only and the M-matrix-QP buffer is boundary-only, so an
  infeasible interior node actually falls through to the bare (non-monotone) Wendland-Taylor LSQ
  weights. The message now says so, so the monotone fraction of the cloud is not over-stated.
  Diagnostic-only; no numerics change.
