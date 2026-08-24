**BREAKING for anyone relying on the fallback.** `HJBHowardSolver` now raises `RuntimeError` when
policy evaluation returns a non-finite value function, instead of logging a warning and returning
the previous timestep (#2072 item 3).

The old branch was a fail-silent fallback: the sweep completes, every value is finite, and the field
returned is a plausible-looking answer to a **different problem** — the previous timestep, carried
forward and then iterated on. It is what turned a Hamiltonian that slipped the decomposition guard
into an all-finite result **153% wrong** against Newton, rather than a NaN somebody would have
noticed.

**Nothing depended on it**, settled by **mutation** rather than by counting: replacing the fallback
with a hard raise on `main` leaves every test in the Howard-touching population with an identical
outcome.

Two successive counts of mine were wrong before that. "Five test references" was a loose grep
(`"policy iterate|non-finite|not finite"` filtered by `howard`, matching unrelated tests containing
both words). Its correction, "four test files import `HJBHowardSolver`", was a *second* loose grep —
**one** file imports it; the others name it in a docstring, a dict key and an `importlib` string —
and the predicate was wrong anyway, since a test reaches this path through `inner_solver="howard"`
without importing the class. The conclusion held both times; neither count did.

The raise names the count of non-finite entries, the time index and the Howard iteration, and points
at the two things that actually produce it: a Hamiltonian whose control cost is not the quadratic
Howard substitutes, and stencil conditioning. `inner_solver='newton'` needs no decomposition.

The new test patches `spsolve` rather than constructing a singular system, because the property
under test is the **policy** — what the solver does when evaluation fails — not the conditions that
make it fail; a matrix that happens to be singular would pin an accident of the fixture. It asserts
the solver produces a finite result **before** the patch, so the raise is attributable to the patch
and not to a broken fixture.

**Not resolved here, and worth a decision:** `fixed_point_iterator.py:645` (#1717) exists for a
diverged HJB — it publishes the field, sets `convergence_reason = "diverged_nan"`, and terminates in
a structured way, with a comment arguing that an escaping exception is the worse outcome. A Howard
divergence now escapes as an uncaught `RuntimeError` instead. That path never fired for Howard before
either (the fallback made the field finite), so this is not a regression — but Howard and Newton now
differ at the coupling layer for the same physical failure, and #2079 tracks it along with the silent
`max_iter` exhaustion in the same loop.
