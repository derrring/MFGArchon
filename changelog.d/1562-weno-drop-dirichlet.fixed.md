- **HJB-WENO drops the unfaithful DIRICHLET capability** (Issue #1562, RFC #1574 Phase 0). WENO
  declared DIRICHLET and its comment claimed the ghost buffers "handle it faithfully", but the solve
  loop evolves the boundary node from the PDE RHS without pinning it to g, so Dirichlet was only weakly
  enforced (~O(h^1.5), not machine-zero / 5th order) and IC/BC-inconsistent data blew the degree-5
  Vandermonde ghost to NaN. DIRICHLET is removed from `_SUPPORTED_BC_TYPES` so it now fails loud at
  construction like Robin/Reflecting, rather than silently mis-enforcing. Strong boundary-node
  enforcement is a follow-up before Dirichlet is re-declared.
