- **`mfgarchon.meta` is marked experimental, takes D from its one owner, and refuses its six unread inputs** (#2648,
  ruling 4). The user ruled on 2026-10-14 to keep the subpackage, as an experimental feature.
  - Its docstring now says: experimental and unverified, no test checks that its output computes anything, and
    nothing in the library uses it. No solver it generates runs as generated (#2655). The source
    `generate_solver_class` emits does not compile, because its template substitution does not indent multi-line
    snippets. Repaired by hand, the generated FP solver still raises `NameError` on an undefined `x_idx`.
  - The flux it emits computed `0.5 * sigma**2` in place. It now calls `diffusion_from_volatility`, pinned by
    `tests/unit/test_meta_experimental_2648.py`.
  - These are refused when passed, by keyword or by position, until v0.25.0. None of them ever had an effect:
    - `DiscretizationScheme(boundary_treatment=)`;
    - `PerformanceProfile(target_precision=, time_constraint=)`;
    - `OptimizationHints(unroll_small_loops=, vectorize_inner_loops=)`;
    - `backend=` in `CodeGenerator`, `MFGSolverGenerator` and `generate_solver_class`: every backend name produced
      the same source.

    `target_precision` is now optional, where it was required. Each field keeps its position, so no positional
    argument after it shifts. `dataclasses.replace` and the `asdict` / `astuple` round trips still work. Reading one
    of these attributes now gives `NEVER_READ`, shown as `<never read>`, where it gave the old default; nothing in the
    library reads them.
  - `retired_parameters(never_read=True)` is the refusal for an input that never had an effect: it names the input and
    says so, rather than naming a replacement. The deprecation guide says the same.
