- **The test suite no longer hangs on a GUI matplotlib backend** (Issue #2090).
  `ConvergenceInfo.plot_convergence()` ends in a bare `plt.show()` and a unit test calls it; on an
  interactive backend that call waits for the window to be dismissed, so `pytest tests/unit`
  stopped at 82% inside `tests/unit/test_types/test_state.py` and never returned. Measured: that
  test exits 124 under a 60s cap with the `macosx` backend and passes in 0.03s with `Agg`.
  `tests/conftest.py` now sets `MPLBACKEND=Agg` before anything imports pyplot, via `os.environ` so
  it reaches xdist workers, and with `setdefault` so a backend can still be forced deliberately.
  It survived this long because `scripts/local_ci.sh` runs under a conda environment where Agg is
  already active while `uv run --extra dev` resolves to `macosx` — the gate was green and a plain
  suite run hung, on the same tree.
