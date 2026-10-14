- **`mfgarchon.meta` is marked experimental, takes D from its one owner, and refuses its five unread fields** (#2648,
  ruling 4). The user ruled on 2026-10-14 to keep the subpackage, as an experimental feature.
  - Its docstring now says: experimental and unverified, no test checks that its output computes anything, and
    nothing in the library uses it. The solver source `generate_solver_class` emits does not compile today, because
    its template substitution does not indent multi-line snippets. Behind that, two generated methods use an
    undefined `x_idx`. No generated solver has run.
  - The flux it emits computed `0.5 * sigma**2` in place. It now calls `diffusion_from_volatility`, pinned by
    `tests/unit/test_meta_experimental_2648.py`.
  - These are refused when passed, by keyword or by position, until v0.25.0. None of them ever had an effect:
    - `DiscretizationScheme(boundary_treatment=)`;
    - `PerformanceProfile(target_precision=, time_constraint=)`;
    - `OptimizationHints(unroll_small_loops=, vectorize_inner_loops=)`.

    `target_precision` is now optional, where it was required. Each field keeps its position, so no positional
    argument after it shifts. `dataclasses.replace` and the `asdict` / `astuple` round trips still work.
  - `retired_parameters(never_read=True)` is the refusal for an input that never had an effect: it names the input and
    says so, rather than naming a replacement. The deprecation guide says the same.
