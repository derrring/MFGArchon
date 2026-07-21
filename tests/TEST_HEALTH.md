# Test Suite Health Report

Last updated: 2025-12-05

## Summary

| Metric | Value |
|:-------|:------|
| Total test files | 170 |
| Files with xfail/skip | 49 |
| Tests requiring PyTorch | ~24 |
| Tests requiring h5py | ~13 |
| Pre-existing numerical issues | ~17 |

## Skip/XFail Categories

### 1. Optional Dependencies (Expected)

These tests are skipped when optional dependencies aren't installed:

| Dependency | Tests Affected | Status |
|:-----------|:---------------|:-------|
| PyTorch | ~24 | Expected - RL algorithms |
| h5py | ~13 | Expected - HDF5 I/O |
| scipy | ~9 | Expected - sparse operations |
| Gymnasium | ~6 | Expected - RL environments |
| Plotly/Bokeh | ~4 | Expected - interactive visualization |

### 2. Pre-existing Numerical Issues (Tracked)

| Issue | Tests | Root Cause | Tracking |
|:------|:------|:-----------|:---------|
| Semi-Lagrangian overflow | 17 | CFL condition violations causing instability | [#369](https://github.com/derrring/mfgarchon/issues/369) |
| Shape mismatch (solve_mfg) | 16 | FP solver returns (Nt+1,) vs (Nt,) expected | #365 |
| GFDM slow tests | 2 | Tests take 5+ minutes each | #365 |

### 3. API Migration Pending

| Issue | Tests | Target |
|:------|:------|:-------|
| Array validation | ~~11~~ **0 (fixed)** | ~~Phase 3.5~~ |
| Factory signatures | 7 | Issue #277 |
| Voronoi maze | 1 | Module not implemented |

## Which tier runs what

The `tier1`-`tier4` markers were deleted (#1706). They declared a schedule -- "every commit",
"on PRs", "on merge to main", "weekly or manually" -- that **no selector implemented**, so a test
marked `tier4` to defer it ran in every tier instead. `scripts/check_markers.py` now rejects any
declaration that promises a schedule without a selector to enforce it.

The markers that actually route are:

| Marker | Excluded from |
|:-------|:--------------|
| `slow` | the local gate, the PR smoke subset |
| `manual` | every automatic tier -- local gate, nightly, and release |
| `optional_torch`, `benchmark`, `experimental`, `environment` | the local gate, nightly, PR checks |

## Running Tests

```bash
# The authoritative gate (what a PR must pass)
./scripts/local_ci.sh

# What nightly runs
pytest tests/ -m "not optional_torch and not benchmark and not experimental and not environment and not manual"

# The manual-only set, which no automatic tier runs
pytest -m manual

# Skip optional dependencies
pytest tests/ --ignore=tests/unit/test_alg/test_neural
```

## Known Issues

### Critical (Blocking)
- None currently

### High Priority
- Semi-Lagrangian solver numerical stability (#365)
- solve_mfg shape mismatch (#365)

### Medium Priority
- ~~Array validation tests need fixing~~ (fixed)
- Factory signature validation

### Low Priority
- Voronoi maze module implementation
