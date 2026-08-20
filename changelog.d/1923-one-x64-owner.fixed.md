`jax_enable_x64` has one owner (#1923).

It is a **process-global** switch, and three sites wrote it without agreeing:

| site | write |
|---|---|
| `backends/jax_backend.py` | `False` when `precision='float32'`, `True` otherwise — a **policy** |
| `utils/acceleration/jax_utils.py` | unconditional `True`, at **module import** |
| `alg/.../meshless_galerkin/mls_basis.py` | unconditional `True`, **per call** |

Last writer won, and which was last depended on import and call order. Nothing failed when they
disagreed: the result is a precision, not an exception. Measured on
`jax.experimental.sparse.linalg.spsolve` against scipy on a tridiagonal system with a known answer,
the error was `2.4e-07` — float32 — where scipy gave `4.4e-16`.

**The asymmetry that decides the design.** Two of those writes are *requirements* ("this computation
needs float64") and one is a *policy* ("the user asked for float32"). A requirement that silently
overwrites a policy is the defect — one MLS call turned a float32 process into a float64 one for
everything that followed — and a policy that silently starves a requirement is no better. Ordering
cannot settle it, because both are legitimate.

So the owner does not pick a winner: `set_x64_policy(enabled, source)` is the only write that may
turn x64 **off**, and `require_x64(reason)` **raises** when a policy forbids it, naming the policy,
its source, and the two ways out (construct the backend with `precision='float64'`, or run in a
separate process). It does not flip the switch on its way to raising.

Writers outside the owner: 3 → 0, pinned by a test that scans the package rather than a fixed list,
so a fourth joins loudly. Mutation-checked: reintroducing a direct write fails the SSOT gate, and
making `require_x64` override the policy fails the fail-loud test.

The tests mutate process-global state, which is the thing under test, so each restores both the flag
and the recorded policy — a test that leaked its own precision would be an instance of the bug.
