- **Discrimination baseline + weekly sweep** (`scripts/discrimination_baseline.json`,
  `.github/workflows/discrimination.yml`, Refs #1701 #1715) — per-mutation kill counts,
  then ratcheted in both directions (one-way since #1715: a gain is listed and exits 0). Measured
  over 5,665 tests: 184 (3.2%) notice any of six
  single-sourced conventions, and `bc_type_to_geometric_operation(None)` is noticed by
  none. The ratchet is on counts, not test names, because this population cannot be
  selected by name.
