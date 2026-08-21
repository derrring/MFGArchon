Docs and in-package docstrings no longer teach `solve_mfg()`, which does not exist (#1741):
`import mfgarchon; mfgarchon.solve_mfg` is `None`, so every example began with a line that raises
`ImportError`. Replaced with `problem.solve()` — the three-mode API of #580, and where
`create_solver()`'s own deprecation note (0.17.0) already pointed. The `phase2_features` guide's
"API Reference" block advertised a `method` preset, a `resolution` parameter and `**kwargs`, none
of which exist on `solve()`; it is now read off the real signature, and the note that `backend`
goes through `MFGSolverConfig` rather than a keyword is measured, not assumed. `AGENTS.md` cited
`solve_mfg()` as an example of a stable public API in its testing-tier table.
