- **The dev extra no longer caps mypy** (`mypy>=1.5`, was `>=1.5,<2.2`) (Issue #2082). The ceiling
  existed because `mypy mfgarchon/config --follow-imports=silent` — `ci.yml`'s *blocking* type gate —
  goes red from mypy 2.2 onwards on three valid `list[str]` slices in `omegaconf_manager.py`.
  Measured: 2.1.0 clean, 2.3.0 three errors, and CI reproduced exactly that on #2082. The cause is a
  typeshed overload that declares a slice's step `SupportsIndex` with no `| None`, which only bites
  under the `strict_optional` this project turns on for `mfgarchon.config.*`; it is still unfixed at
  2.3.1, so the old comment's "lift the pin once upstream fixes it" was an instruction with no expiry
  and nothing checking it. The three call sites now spell the step (`keys[: i + 1 : 1]`), which the
  gate accepts on 2.1 and 2.3 alike — so the ceiling was buying nothing while freezing out every
  unrelated fix in 2.2+, and dependabot re-proposed it on every release.
