- **`ConvergenceInfo.plot_convergence()`'s `except ImportError` now covers the import statement
  only, and the test named for its fallback branch actually reaches that branch** (Issue #2090,
  items 2 and 3; item 1 shipped in #2091, item 4 belongs to #2089).

  The `except` wrapped the whole method body. An `ImportError` raised from *inside* matplotlib —
  `plt.figure()` does exactly that on a backend whose own dependency is absent, which is the cairo
  case `test_headless_backend_2090.py` already documents — was reported as **"Matplotlib not
  available for plotting"**. That message names the one package that is installed, so it sends the
  reader to the wrong place and swallows the real one.

  The test was `test_convergence_info_plot_convergence_no_matplotlib`, documented as testing the
  path "without matplotlib". matplotlib is a declared dev dependency, so the `try` always succeeded
  and **the branch the test was named for had never run**. Its assertion was
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

  Not changed here: the method still ends in a bare `plt.show()`, and `SolverResult` already has
  the shape this one lacks — `plot_convergence(save_path=..., show=False, log_scale=...)` returning
  a figure, with ten tests exercising it. Under #2089's policy this method goes away rather than
  being taught the sibling's signature, so the divergence is recorded there instead of half-closed
  here. The new plotting test asserts `show()` **was** called, so #2089 trips it deliberately.
