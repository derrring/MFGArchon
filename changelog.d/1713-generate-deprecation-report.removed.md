- **`scripts/generate_deprecation_report.py`** (Issue #1713, #1706 Class A). 341 lines, **zero
  invokers** anywhere in CI, docs or scripts, consuming the broken AST discovery above — so the
  Markdown table it generated reported the same wrong `0`. It also cites
  `docs/development/DEPRECATION_LIFECYCLE_POLICY.md`, which does not exist. `--show` on the repaired
  checker lists the same information from the registry.
