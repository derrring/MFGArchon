- **Network derivative dH/dm single-sourced** (Issue #1470 Strand A). `NetworkMFGProblem.hamiltonian_dm`
  now delegates to the wired Hamiltonian object's `dm`, which gained the EXACT analytic derivative for the
  default node congestion (`d/dm(0.5*m_own^2) = m_own[node]`, own-population slice) instead of finite
  difference. The dead `_default_density_coupling_derivative` helper (raw `m[node]`, reachable only via the
  uncalled `hamiltonian_dm`) is removed. Byte-identical single-population; own-slice for multi-population.
