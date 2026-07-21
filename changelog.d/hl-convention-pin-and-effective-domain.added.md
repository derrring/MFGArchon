`ControlCostBase.effective_domain()` — single owner of the admissible control set A = dom(L)
(Issue #1642, capability B3). `L1ControlCost` returns `(-1, 1)`, `BoundedControlCost` returns
`(-max_control, max_control)`, and the Moreau-Yosida wrapper delegates to the cost it wraps.
Both consumers now read it: `SeparableLagrangian.control_bounds()` and the semi-Lagrangian
DPP control sweep (`HJBSemiLagrangianSolver._solve_timestep_dpp`), each of which previously
re-derived A through its own isinstance ladder. API-level behavior change: a regularized
`L1`/`Bounded` cost previously reported an *unbounded* control set (the ladders matched
neither branch); it now reports the wrapped cost's bounds. No solver output moves — the
delegated control-candidate sets are byte-identical to the removed ladder for every
unwrapped cost, and wrapped costs do not reach the DPP path (`_use_dpp` requires
`not H.is_smooth()`, which Moreau-Yosida makes false).

The `(V, f)` sign convention between H and L (`L = L_ctrl - V - f`) is documented with its
derivation on `MFGOperatorBase` and pinned by `tests/unit/test_core/test_hl_convention.py`;
the `SeparableLagrangian` rows were strict-xfail pending Issue #1645, which landed in the same release and turned them green.
