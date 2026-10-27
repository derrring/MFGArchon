- **AGENTS.md: a boundary or geometric-kernel PR is fuzzed before review 1** (user rulings, 2026-10-26 and
  2026-10-27), in § *Branches & PRs*, *Review before merge*.
  - The fuzz has an oracle written from the rules and a rival that must fire; its inputs are aimed at the edge of each
    branch condition the diff adds, removes or alters, with a population named for a boundary and for a kernel
    (the kernel's including scale per axis).
  - A correction that adds kernel or predicate logic after a review gets a fuzz of its own before the next review.
  - The fuzz stays in scratch; only the pins it motivates are committed.
- **AGENTS.md's no-clip section** cites the wall-clamp exclusion as "see #2485, class 10", since that exclusion covers
  9 of class 10's 11 calls (#2485, comment 5979417072).
