`scripts/check_doc_api.py` — a ratchet that fails when the documentation teaches more API than
the package provides (#1759). Docs were the one artefact nothing ran: the test suite never
imports a doc example, so a rename left every tutorial using the old name teaching a
`NameError`. The first sweep found 259 such claims across 110 files — 102 imports of symbols
that do not exist, 99 calls to names neither defined nor provided, 53 parameters no signature
accepts, and 5 class sketches contradicting the real thing. Wired into `./scripts/local_ci.sh`
with a baseline, bidirectional like the other ratchets, and it refuses to run at all if its own
checks have gone inert.
