`ControlCostBase.effective_domain()` — single owner of the admissible control set A = dom(L)
(Issue #1642, capability B3). `L1ControlCost` returns `(-1, 1)`, `BoundedControlCost` returns
`(-max_control, max_control)`, and the Moreau-Yosida wrapper delegates to the cost it wraps.
`SeparableLagrangian.control_bounds()` now delegates to it instead of re-deriving the same
answer through an isinstance ladder. Behavior change: a regularized `L1`/`Bounded` cost
previously reported an *unbounded* control set (the ladder matched neither branch); it now
reports the wrapped cost's bounds. The `(V, f)` sign convention between H and L
(`L = L_ctrl - V - f`) is documented on `MFGOperatorBase` and pinned by
`tests/unit/test_core/test_hl_convention.py`; the `SeparableLagrangian` rows are strict-xfail
pending Issue #1645.
