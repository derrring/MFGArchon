# Three-Mode Solving API Migration Guide

**Version**: 0.17.0
**Issue**: #580
**Status**: Recommended for all users

---

## Overview

The new three-mode solving API introduces **guaranteed adjoint duality** between HJB and FP solvers, preventing a subtle but critical numerical error that can break Nash equilibrium convergence. This guide helps you migrate to the new API.

**Key Benefit**: You can now safely experiment with different numerical schemes knowing that duality is guaranteed by construction.

---

## Quick Start

### If You're Using `problem.solve()` (Most Users)

**Good news**: Your code already works! The new API is backward compatible.

```python
# This still works (uses Auto Mode)
problem = MFGProblem(Nx=[40], Nt=20, T=1.0)
result = problem.solve()
```

**Recommended upgrade**: Make the scheme selection explicit for better clarity:

```python
from mfgarchon.types import NumericalScheme

# Explicit scheme selection (Safe Mode)
result = problem.solve(scheme=NumericalScheme.FDM_UPWIND)
```

### If You're Using `create_solver()`

**Status**: Deprecated (but still works)

```python
# Old pattern (deprecated)
from mfgarchon.factory import create_solver
solver = create_solver(problem, hjb_solver=hjb, fp_solver=fp)
result = solver.solve()
```

**New pattern** (Expert Mode):

```python
# New pattern (recommended)
result = problem.solve(hjb_solver=hjb, fp_solver=fp)
```

---

## Three Modes Explained

### Safe Mode: Automatic Dual Pairing

**When to use**: You want guaranteed correctness with zero configuration.

**Before**:
```python
from mfgarchon.alg.numerical import HJBFDMSolver, FPFDMSolver

problem = MFGProblem(Nx=[40], Nt=20, T=1.0)

# Manual solver creation (risk of mismatch)
hjb = HJBFDMSolver(problem)
fp = FPFDMSolver(problem)

# Hope they're compatible...
solver = create_solver(problem, hjb_solver=hjb, fp_solver=fp)
result = solver.solve()
```

**After**:
```python
from mfgarchon.types import NumericalScheme

problem = MFGProblem(Nx=[40], Nt=20, T=1.0)

# Factory creates validated pair automatically
result = problem.solve(scheme=NumericalScheme.FDM_UPWIND)
```

**Benefits**:
- Cannot create invalid pairings
- Factory handles all configuration
- Clearer intent in code

### Expert Mode: Manual Control with Validation

**When to use**: You need custom solver configuration.

**Before**:
```python
from mfgarchon.alg.numerical import HJBFDMSolver, FPFDMSolver
from mfgarchon.factory import create_solver

hjb = HJBFDMSolver(problem)
fp = FPFDMSolver(problem, advection_scheme="divergence_upwind")

solver = create_solver(problem, hjb_solver=hjb, fp_solver=fp)
result = solver.solve()
```

**After**:
```python
from mfgarchon.alg.numerical import HJBFDMSolver, FPFDMSolver

hjb = HJBFDMSolver(problem)
fp = FPFDMSolver(problem, advection_scheme="divergence_upwind")

# Duality automatically validated with warnings if mismatched
result = problem.solve(hjb_solver=hjb, fp_solver=fp)
```

**Benefits**:
- Full control over solver parameters
- Automatic duality validation
- Educational warnings if solvers don't match

### Auto Mode: Intelligent Defaults

**When to use**: Quick experiments, prototyping, teaching.

**Before**:
```python
problem = MFGProblem(Nx=[40], Nt=20, T=1.0)
result = problem.solve()  # Uses whatever default was hardcoded
```

**After**:
```python
problem = MFGProblem(Nx=[40], Nt=20, T=1.0)
result = problem.solve()  # Uses FDM_UPWIND (safe default)
# Future: Will analyze geometry and select optimal scheme
```

**Benefits**:
- Zero configuration needed
- Documented default behavior
- Future: Intelligent scheme selection based on geometry

---

## Common Migration Patterns

### Pattern 1: Testing Different Schemes

**Before** (manual, error-prone):
```python
from mfgarchon.alg.numerical import HJBFDMSolver, FPFDMSolver
from mfgarchon.factory import create_solver

problem = MFGProblem(Nx=[40], Nt=20, T=1.0)

# FDM test
hjb_fdm = HJBFDMSolver(problem)
fp_fdm = FPFDMSolver(problem)
solver_fdm = create_solver(problem, hjb_solver=hjb_fdm, fp_solver=fp_fdm)
result_fdm = solver_fdm.solve()

# Semi-Lagrangian test
hjb_sl = HJBSemiLagrangianSolver(problem)
fp_sl = FPSLAdjointSolver(problem)  # Must use Adjoint, not regular SL!
solver_sl = create_solver(problem, hjb_solver=hjb_sl, fp_solver=fp_sl)
result_sl = solver_sl.solve()
```

**After** (clean, safe):
```python
from mfgarchon.types import NumericalScheme

problem = MFGProblem(Nx=[40], Nt=20, T=1.0)

schemes = [NumericalScheme.FDM_UPWIND, NumericalScheme.SL_LINEAR]
results = {}

for scheme in schemes:
    results[scheme.value] = problem.solve(scheme=scheme)
```

### Pattern 2: Custom Solver Configuration

**Before**:
```python
from mfgarchon.alg.numerical import HJBFDMSolver, FPFDMSolver
from mfgarchon.factory import create_solver

problem = MFGProblem(Nx=[40], Nt=20, T=1.0)

hjb = HJBFDMSolver(problem)
fp = FPFDMSolver(problem, advection_scheme="gradient_centered")  # non-conservative; see note below

solver = create_solver(problem, hjb_solver=hjb, fp_solver=fp)
result = solver.solve()
```

> **The FP advection scheme changes in both "After" blocks, and that is not part of the
> migration.** The `gradient_*` family is non-conservative at a no-flux wall (#1075, #2007): on the
> `examples/basic/three_mode_api_demo.py` problem `gradient_upwind` lost 98.17% of the probability
> mass while the Picard residual stayed under tolerance (#2008), and `gradient_centered` aborts
> there outright. Migrating the API shape is independent of which scheme you pass — but a guide
> should not print a non-conservative one in the code it tells you to write, so these use
> `divergence_upwind`. Check `result.mass_conservation_error` either way; it is on `SolverResult`
> and nothing gates on it.
>
> Option 1 keeps `NumericalScheme.FDM_CENTERED` so that the `fp_config` override is a real
> non-default rather than a restatement of the factory default — `FDM_UPWIND` is already paired
> with `divergence_upwind`. Be aware of what that enum does and does not do: **`FDM_CENTERED`
> selects the FP half only.** Measured, `HJBFDMSolver.advection_scheme` is `gradient_upwind` under
> `FDM_UPWIND`, under `FDM_CENTERED`, and under `FDM_CENTERED` with this override alike — the HJB
> side never varies (#1866). So the pair Option 1 builds is the `FDM_UPWIND` pair; the enum is kept
> here for the shape of the example, not because it buys HJB order.

**After** (Option 1 - Safe Mode with config):
```python
from mfgarchon.factory import create_paired_solvers
from mfgarchon.types import NumericalScheme

hjb, fp = create_paired_solvers(
    problem,
    NumericalScheme.FDM_CENTERED,
    fp_config={"advection_scheme": "divergence_upwind"},
)

result = problem.solve(hjb_solver=hjb, fp_solver=fp)
```

**After** (Option 2 - Expert Mode):
```python
from mfgarchon.alg.numerical import HJBFDMSolver, FPFDMSolver

hjb = HJBFDMSolver(problem)
fp = FPFDMSolver(problem, advection_scheme="divergence_upwind")

# Direct Expert Mode - validates automatically
result = problem.solve(hjb_solver=hjb, fp_solver=fp)
```

### Pattern 3: GFDM with Collocation Points

**Before**:
```python
import numpy as np
from mfgarchon.alg.numerical import HJBGFDMSolver, FPGFDMSolver
from mfgarchon.factory import create_solver

problem = MFGProblem(Nx=[40], Nt=20, T=1.0)
points = np.linspace(0, 1, 30)[:, None]

hjb = HJBGFDMSolver(problem, collocation_points=points, delta=0.1)
fp = FPGFDMSolver(problem, collocation_points=points, delta=0.1)

solver = create_solver(problem, hjb_solver=hjb, fp_solver=fp)
result = solver.solve()
```

**After** (config threading):
```python
import numpy as np
from mfgarchon.factory import create_paired_solvers
from mfgarchon.types import NumericalScheme

problem = MFGProblem(Nx=[40], Nt=20, T=1.0)
points = np.linspace(0, 1, 30)[:, None]

# Config threading: specify once, used for both
hjb, fp = create_paired_solvers(
    problem,
    NumericalScheme.GFDM,
    hjb_config={"collocation_points": points, "delta": 0.1},
    # fp_config automatically inherits collocation_points and delta
)

result = problem.solve(hjb_solver=hjb, fp_solver=fp)
```

---

## Available Schemes

### Discrete Duality (Type A) - Exact Transpose

These schemes satisfy $L_{FP} = L_{HJB}^T$ exactly at matrix level:

| Scheme | Use Case | Order | Stability |
|:-------|:---------|:------|:----------|
| `FDM_UPWIND` | General purpose, monotone | 1st order | Excellent |
| `FDM_CENTERED` | Higher accuracy, smooth problems | 2nd order | Good (low Peclet)¹ |
| `SL_LINEAR` | Large time steps, transport-dominated | 1st order | Excellent |
| `SL_CUBIC` | High accuracy SL | 3rd order (HJB) | Good |

¹ **`FDM_CENTERED` differs from `FDM_UPWIND` in the FP half only** — measured,
`HJBFDMSolver.advection_scheme` is `gradient_upwind` under both (#1866). So the "2nd order" in this
row is the FP discretisation, and it is the half that will not run: `divergence_centered` is not
positivity-preserving, and when the density goes negative the mass-fabrication gate stops the solve
rather than clip it — at timestep 3 of 20 on the `three_mode_api_demo` problem. Overriding the FP
half to `divergence_upwind` therefore leaves you with exactly the `FDM_UPWIND` pair, not a
second-order HJB. The HJB-side knob is `HJBFDMSolver(advection_scheme=...)`, which the factory does
not set.

### Continuous Duality (Type B) - Asymptotic Adjoint

These schemes satisfy $L_{FP} = L_{HJB}^T + O(h)$ asymptotically:

| Scheme | Use Case | Order | Stability |
|:-------|:---------|:------|:----------|
| `GFDM` | Unstructured grids, complex geometries | 2nd order | Good |

**Note**: Type B schemes require renormalization for optimal Nash gap convergence.

---

## Checking Your Code for Issues

### Warning: Mismatched Solver Pairs

If you see this warning:

```
Expert Mode: Non-dual solver pair detected!
  HJB: HJBFDMSolver (fdm)
  FP: FPGFDMSolver (gfdm)
  Status: not_dual
This may lead to poor convergence or Nash gap issues.
Consider using Safe Mode for guaranteed duality.
```

**What it means**: Your HJB and FP solvers don't form an adjoint pair. The discrete operators won't satisfy $L_{FP} = L_{HJB}^T$.

**Consequence**: Nash gap may not converge, even as mesh size $h \to 0$.

**Fix**: Use Safe Mode to get a guaranteed dual pair:

```python
# Instead of mixing FDM with GFDM:
result = problem.solve(scheme=NumericalScheme.FDM_UPWIND)
# Or use matching schemes:
result = problem.solve(scheme=NumericalScheme.GFDM)
```

### Deprecation Warning: create_solver()

If you see:

```
DeprecationWarning: create_solver() is deprecated since v0.17.0 (Issue #580).
Use the new three-mode solving API instead:
  • Safe Mode: problem.solve(scheme=NumericalScheme.FDM_UPWIND)
  • Expert Mode: problem.solve(hjb_solver=hjb, fp_solver=fp)
  • Auto Mode: problem.solve()
```

**Action**: Migrate using the patterns above. The old function still works but will be removed in v1.0.0.

---

## Benefits of Migration

### Before Migration (Old API)

❌ **Easy to create invalid pairings** (FDM HJB + GFDM FP)
❌ **No validation** - errors discovered during solving
❌ **Verbose code** - many lines to create solver pair
❌ **Unclear intent** - what scheme are we actually using?

### After Migration (New API)

✅ **Cannot create invalid pairings** in Safe Mode
✅ **Automatic validation** in Expert Mode with warnings
✅ **Concise code** - one line to specify scheme
✅ **Clear intent** - scheme is explicitly named

### Performance

**No performance penalty**: The new API adds negligible overhead (<1%) for validation.

### Backward Compatibility

**100% backward compatible**: All existing `problem.solve()` calls continue to work.

---

## Examples

See **`examples/basic/three_mode_api_demo.py`** for a comprehensive demonstration of all three modes with:
- Side-by-side comparisons
- Scheme selection examples
- Visualization
- Convergence analysis

Run it:
```bash
python examples/basic/three_mode_api_demo.py
```

---

## FAQ

### Do I need to update my code?

**If you use `problem.solve()`**: No, but we recommend explicit scheme selection for clarity.

**If you use `create_solver()`**: Yes, it's deprecated and will be removed in v1.0.0.

### Which mode should I use?

- **Beginners / Production code**: Safe Mode
- **Advanced users / Research**: Expert Mode for custom config, Safe Mode otherwise
- **Quick experiments**: Auto Mode

### What if I need a specific solver configuration?

Use Expert Mode with manual solver creation, or Safe Mode with config dictionaries passed to `create_paired_solvers()`.

### Will this break my existing code?

No. The new API is fully backward compatible. Existing code continues to work.

### How do I know if my solvers are dual?

The system automatically validates:
- Safe Mode: Duality guaranteed by construction
- Expert Mode: Validation runs automatically, emits warnings if mismatched
- Use `check_solver_duality()` for manual verification

### What's the difference between FDM_UPWIND and FDM_CENTERED?

- **FDM_UPWIND**: First-order accurate, monotone (no oscillations), very stable
- **FDM_CENTERED**: differs from `FDM_UPWIND` in the **FP half only** (#1866), so the second order
  is the FP discretisation — and that half may oscillate for high Peclet numbers and then refuse to
  run rather than clip a negative density. See the footnote under the scheme table.

For most problems, start with FDM_UPWIND.

### Can I mix schemes?

**No** - mixing schemes (e.g., FDM HJB with GFDM FP) breaks adjoint duality. The system will warn you in Expert Mode and prevent it in Safe Mode.

### What about Semi-Lagrangian?

Use `SL_LINEAR` or `SL_CUBIC` in Safe Mode. The factory automatically pairs with `FPSLAdjointSolver` (forward splatting), not `FPSLSolver` (backward interpolation), to maintain duality.

---

## Getting Help

- **Documentation**: `docs/development/issue_580_adjoint_pairing_implementation.md`
- **Examples**: `examples/basic/three_mode_api_demo.py`
- **Issues**: Report problems at https://github.com/anthropics/mfgarchon/issues

---

## Summary

**Recommended Migration Path**:

1. ✅ Keep using `problem.solve()` for simple cases
2. ✅ Add explicit `scheme=` parameter for clarity
3. ✅ Replace `create_solver()` calls with `problem.solve(hjb_solver=..., fp_solver=...)`
4. ✅ Use `create_paired_solvers()` when you need custom configs
5. ✅ Run your tests - everything should still pass

**Timeline**:
- v0.17.0 (current): New API available, old API deprecated
- v1.0.0 (future): Old `create_solver()` removed

**Key Takeaway**: The new API makes it **impossible** to accidentally break adjoint duality, ensuring your MFG simulations converge correctly.
