- **Test-discrimination harness** (`scripts/test_discrimination.py`, Refs #1701 #1715) —
  perturbs each convention the library single-sources and records which tests notice,
  producing a per-test kill matrix. Selects the population by behaviour rather than by
  name, since `*_agree` / `*_matches` / `*_equals` gives 51, 114 or 156 depending on the
  pattern. Handles the two traps that make mutation testing silently inert here: the
  editable install pinning imports to the main checkout (#1677), and a zero kill count
  being ambiguous between "no test covers this" and "the mutation never ran".
