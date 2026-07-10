- **Repoint the dangling `NAMING_CONVENTIONS.md` references** (Issue #1573). The convention doc was
  moved to `mfg-research/docs/archon-notes/development/guides/NAMING_CONVENTIONS.md` (#852) but 8 files
  (12 references, incl. the load-bearing Riccati-provenance comment in `test_hjb_howard_solver.py`)
  still pointed at the deleted in-repo `docs/NAMING_CONVENTIONS.md`. All now point at the new location.
  The Howard test's expected values remain provenanced by their inline derivation, not the external doc.
