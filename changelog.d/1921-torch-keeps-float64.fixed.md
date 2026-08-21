The torch backend no longer narrows `float64` to `float32` behind the caller's back (#1921).

`TorchBackend(precision="float64")` — the default — returned float32, because `device="auto"`
selected MPS and MPS has no float64 at all. It warned and narrowed:

```
warnings.warn("MPS does not support float64, using float32 instead. ...")
```

A `UserWarning` is not enough for that. This library's convergence tolerances are 1e-10 and 1e-12
throughout, so a float32 backend does not make them *hard to hit* — it makes them **unreachable in
principle**. Measured: `np.array([1.0 + 1e-10, 1.0])` came back with both entries equal.

**The asymmetry, the same one behind #1923.** The precision was something the **user** asked for; the
device was something the **library** chose for them under `auto`. A choice made for the user must not
spend one the user made. So:

- `auto` now prefers CPU whenever `float64` is requested. That costs nothing here — measured on this
  machine over mean/max/trapezoid, torch is 9.2× slower than numpy at n = 1e6 and 361× at n = 1e4 —
  and it is precision-scoped, so `float32` still reaches MPS.
- An **explicit** `device='mps'` with `precision='float64'` raises. There the caller named both, they
  genuinely conflict, and the message offers both ways out rather than choosing: `device='cpu'` to
  keep float64, `precision='float32'` to keep MPS.

The other half of #1921 — `create_backend(None)` auto-selecting torch — was already corrected on
2026-08-17 and now returns numpy unconditionally.

Mutation-checked: letting `auto` pick MPS under float64 again fails both the round-trip test and the
device test. A control keeps the skip from becoming a blanket refusal of MPS, and the
warn-and-narrow path is pinned gone against the source.
