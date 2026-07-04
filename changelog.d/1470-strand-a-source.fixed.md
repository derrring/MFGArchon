- **Network HJB source term single-sourced** (Issue #1470 Strand A). `hjb_network._source_terms` and
  `NetworkHamiltonian._default_hamiltonian` each computed the source `V + f(m)` independently — and
  diverged for stacked multi-population `m`: the HJB re-derived the coupling on the raw `m[node]` while
  the Hamiltonian used the own-population slice (`_extract_own_density`), corrupting the isolated control
  `h_total - source`. Added `NetworkHamiltonian.source_term` as the single source consumed by both;
  `_default_hamiltonian` is now `control + source_term`, and the HJB routes through the wired object.
  Byte-identical single-population; fixes the multi-population fork.
