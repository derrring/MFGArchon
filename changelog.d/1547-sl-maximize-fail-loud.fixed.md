- **HJB semi-Lagrangian fails loud on a MAXIMIZE control cost** (Issue #1547, RFC #1574 Phase 0). The
  characteristic-foot velocity dH/dp = p/lambda is hardcoded MINIMIZE-signed (alpha* = -grad(u)/lambda);
  a MAXIMIZE control cost (alpha* = +grad(u)/lambda) would trace the feet in the wrong direction, and
  because the MAXIMIZE-quadratic Hamiltonian is smooth the non-smooth DPP reroute never fires — so the
  wrong scheme ran silently. Construction now raises NotImplementedError (mirroring the GFDM Howard
  gate). MINIMIZE (every published config) is unaffected. The canonical_cs + non-quadratic gap remains
  a separate Phase-0 item on #1547.
