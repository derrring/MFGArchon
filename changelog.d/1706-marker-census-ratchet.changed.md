Eight pytest markers are removed and a ratchet added, because `--strict-markers` cannot see either
failure it leaves open. That flag rejects *undeclared* markers, so a declaration is exactly what
makes a meaningless marker look legitimate.

**Unreachable** — declared, applied by no test, named by no selector: `regression`, `network`,
`stochastic`, `numerical`.

**False promise** — the description claims a schedule that no selector implements. `tier1`–`tier4`
declared "run on every commit", "run on PRs", "run on merge to main" and "run weekly or manually",
with **zero** selectors referencing any of them. A test marked `tier4` in the belief it would be
deferred would have run in **every** tier, which is the opposite of the declaration. The family is
removed rather than re-described: the names carry the same suggestion as the text. Its four
applications were all in one file whose module docstring already explains each test.

`scripts/check_markers.py` now enforces both invariants, and runs in the local gate. Deliberately
not enforced: whether a marker is *useful*. `unit` (1404 applications) and `fast` (154) route
nothing automatically and remain legitimate — a developer types them by hand — and their
descriptions are descriptive rather than promissory.

This is the general shape of a defect an additive change produces. A modification is checked against
what was there before; an addition regresses nothing, so every gate passes it. The same class
produced two junk markers here on 2026-07-21, when a three-line description in the linelist
`markers` option registered `@pytest.mark.with` and `@pytest.mark.worse` — which `--strict-markers`
would then have accepted permanently. A test now rejects any multi-line declaration.
