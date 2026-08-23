**BREAKING for anyone relying on the fallback.** `HJBHowardSolver` now raises `RuntimeError` when
policy evaluation returns a non-finite value function, instead of logging a warning and returning
the previous timestep (#2072 item 3).

The old branch was a fail-silent fallback: the sweep completes, every value is finite, and the field
returned is a plausible-looking answer to a **different problem** — the previous timestep, carried
forward and then iterated on. It is what turned a Hamiltonian that slipped the decomposition guard
into an all-finite result **153% wrong** against Newton, rather than a NaN somebody would have
noticed.

**Nothing depended on it.** Zero tests assert on that warning, and none of the four test files
importing `HJBHowardSolver` reaches that path. An earlier estimate of "five test references" was a
loose grep — `"policy iterate|non-finite|not finite"` filtered by `howard` — matching unrelated
tests that contain both words; corrected on #2072.

The raise names the count of non-finite entries, the time index and the Howard iteration, and points
at the two things that actually produce it: a Hamiltonian whose control cost is not the quadratic
Howard substitutes, and stencil conditioning. `inner_solver='newton'` needs no decomposition.

The new test patches `spsolve` rather than constructing a singular system, because the property
under test is the **policy** — what the solver does when evaluation fails — not the conditions that
make it fail; a matrix that happens to be singular would pin an accident of the fixture. It asserts
the solver produces a finite result **before** the patch, so the raise is attributable to the patch
and not to a broken fixture.
