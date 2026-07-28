`scripts/check_doc_api.py` — a ratchet that fails when the documentation teaches more API than
the package provides (#1759). Docs were the one artefact nothing ran: the test suite never
imports a doc example, so a rename left every tutorial using the old name teaching a
`NameError`. The first sweep found 251 such claims in 30 files — 96 imports of symbols that do
not exist, 99 calls to names neither defined nor provided, 53 parameters no signature accepts,
and 3 class sketches contradicting the real thing. Scoped by `git ls-files` so the committed
baseline measures what is committed, wired into `./scripts/local_ci.sh`, bidirectional like the
other ratchets, and it refuses to run if its own checks have gone silent.
