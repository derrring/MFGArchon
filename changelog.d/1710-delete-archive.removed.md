- **`archive/`, `scripts/archive/`, the seven `#580` review documents, and two write-only
  artefacts** (Issue #1710) — 16,674 lines. All were excluded from pytest, ruff and the
  fail-fast ratchet, so their cost of continued existence was zero, which is why they were
  handled last rather than first despite being the largest item by line count. The
  exclusion rules in `pytest.ini`, `.codecov.yml` and `pyproject.toml` go with them. Two
  unresolved findings buried in the archived summaries are recorded on #1655 first.
