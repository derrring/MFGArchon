- **`file:line` citations in code prose are replaced by the symbol they name.** A line number in a
  durable artifact is a claim with an expiry date and nothing marks it when it stops being true.
  Measured before the change, on the narrowest decidable property — does the cited line still hold
  code at all: **11 of 39** such citations in `mfgarchon/` and **23 of 103** in `tests/` pointed at a
  blank line, a comment, or past the end of the file. Both are zero now.

  **That is not the same property `check_citations.py` measures**, and its numbers move differently:
  `drifted` goes 17 → 13 of an `adjudicable` set that shrinks 38 → 32. It asks whether the symbol the
  prose names still describes the cited line, which is grammatical and which its own docstring says
  cannot be closed mechanically. This change removes citations rather than adjudicating them, so it
  lowers both instruments without closing either.

  Where the surrounding prose already named the symbol, the number was simply redundant; where it
  did not, the enclosing function or class replaces it.

- **Twelve changelog fragments cut to what a reader of `CHANGELOG.md` acts on**, 1235 lines to 253.
  What came out was the investigation behind each change — mutation kill tables, review-round
  narratives, retracted intermediate claims, measurements with no reader — all of which is in the
  commits and PRs those fragments were written beside. `changelog.d/README.md` specifies
  `- **Short title** (Issue #123). One-sentence what + why`; the largest of the twelve was 201 lines,
  and the one fact in it a library user acts on — that the analytic Jacobian is opt-in and the
  default finite-difference path is still wrong — sat at line 150.

  Three self-referential comment histories in `mfgarchon/` are rewritten as the trap they record
  rather than as the story of a previous revision.
