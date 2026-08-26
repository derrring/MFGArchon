- **`ConvergenceInfo.plot_convergence()`'s `except ImportError` now covers the import statement
  only, and the test named for its fallback branch actually reaches that branch** (Issue #2090,
  items 2 and 3; item 1 shipped in #2091, item 4 belongs to #2089).

  The `except` wrapped the whole method body. An `ImportError` raised from *inside* matplotlib —
  `plt.figure()` does exactly that on a backend whose own dependency is absent, which is the cairo
  case `test_headless_backend_2090.py` already documents — was reported as **"Matplotlib not
  available for plotting"**. That message names the one package that is installed, so it sends the
  reader to the wrong place and swallows the real one.

  The test was `test_convergence_info_plot_convergence_no_matplotlib`, documented as testing the
  path "without matplotlib". matplotlib is a hard **runtime** dependency — `[project] dependencies`
  carries `matplotlib>=3.8`, not a dev extra — so the `try` always succeeded and **the branch the
  test was named for is unreachable in any correct installation**, not merely unreached in the dev
  environment. It becomes reachable only once #2089 drops matplotlib from `dependencies`, which is
  the reason the fallback deserves a test. Its assertion was
  `assert isinstance(plot_succeeded, bool)` over a variable assigned only literal `True` and
  `False` — it could not fail, in any state of the code.

  Three tests replace it, each pinning a different thing, and each verified against a mutation that
  kills it and nothing else:

  | test | mutation that turns it red |
  |:-----|:---------------------------|
  | the fallback prints its message | delete the `print` |
  | the figure carries the residual history on a log axis | `semilogy` → `plot` |
  | an `ImportError` from inside matplotlib propagates | re-widen the `except` to the body |

  The third exists because narrowing the `except` was otherwise **unpinned** — the fallback test
  passes under either form, so the change would have been a claim with no measurement behind it.

  The gate's own assertion-strength ratchet reads the swap: **688 of 5467 → 687 of 5469**. The
  removed test was in the counted set and all three replacements are outside it, which is the
  check that the split bought discrimination rather than test count.

  The warnings ratchet (#2119/#2120) fired on this change and had to be re-recorded, which is the
  behaviour it was built for. Stubbing `plt.show()` retires
  `mfgarchon/types/state.py | UserWarning | FigureCanvasAgg is non-interactive` — exactly one
  identity, 225 → 224, verified against the gate's own population rather than a `tests/unit` subset.
  The sibling entry from `test_headless_backend_2090.py` stays, because that test calls `show()` on
  purpose to prove it returns.

  The fixture changed with them, and that closes the issue's **first** defect. It was
  `[1e-1, 1e-3, 1e-5]` — three exact powers of ten, constant ratio, so `semilogy` could render
  nothing but the straight line the user reported as making no sense. It was also what let the
  plotting test pass over a method that plotted a hardcoded literal: measured, replacing
  `self.residual_history` with `[1e-1, 1e-3, 1e-5]` left all 25 tests in the file green. With
  `[0.37, 4.1e-2, 6.2e-3]` that mutation turns the plotting test red.

  Not changed here: the method still ends in a bare `plt.show()`, and `SolverResult` already has
  the shape this one lacks — `plot_convergence(save_path=..., show=..., log_scale=...)` returning a
  figure, with **8** test methods exercising it across 9 call sites. Its `show` defaults to `True`,
  so the contrast is an *optional* show, not a non-showing default. Under #2089's policy this
  method goes away rather than
  being taught the sibling's signature, so the divergence is recorded there instead of half-closed
  here. The new plotting test asserts `show()` **was** called, so #2089 trips it deliberately.
