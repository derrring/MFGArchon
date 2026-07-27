- **Docs no longer teach removed factory functions** (Issue #1709) — `CLAUDE.md`,
  `CONTRIBUTING.md` and five user documents presented `create_standard_solver` (ImportError)
  and four siblings that raise on call as the current API. Rewritten to `problem.solve()`.
  Two migration guides to an API that has since been removed, and a solver-selection guide
  organised entirely around the removed tiers, were deleted rather than rewritten.
  `tests/unit/test_docs_do_not_teach_removed_api.py` prevents recurrence, scanning fenced
  code blocks only so that prose about the removal stays legal.
