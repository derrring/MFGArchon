- **Network source term single-sourced** (Issue #1470 Strand A). The p-independent source `V + f(m)` was
  computed on diverging paths — `NetworkHamiltonian` (own-population slice via `_extract_own_density`)
  vs the live `NetworkMFGProblem.density_coupling` / `hjb_network._source_terms` (raw stacked `m[node]`)
  — silently wrong for multi-population `m`, corrupting the HJB control isolation `h_total - source`.
  Now `NetworkHamiltonian` owns it via `source_term` / `node_potential_value` / `coupling_value`, and
  `_default_hamiltonian`, `hjb_network._source_terms`, and `NetworkMFGProblem.node_potential` /
  `density_coupling` all route through the wired object. Byte-identical single-population; fixes the
  multi-population fork on every consumer.
