The Newton MFG solver integration tests run in no automatic tier. `slow` already kept them out of
the local gate; a new `manual` marker keeps them out of nightly as well. Run them deliberately with
`pytest tests/integration/test_newton_mfg_solver.py` or `pytest -m manual`.

Measured 2026-07-21: `test_newton_solver_executes` takes 822 s at `dcb8be82` and 842 s at
`c1a29a12`, and **passes serially at both** — the nightly failures were not a regression. Against
the 900 s timeout that is 8% of headroom, so under `-n auto` the workers contend and the group
crosses it. Eight tests at that duration also occupied roughly two hours of the integration shard,
so failures behind them were never reached.

The cost is stated rather than absorbed: nothing will now report when these break. The `manual`
marker's declared description says so, and the module docstring carries the measurement.
