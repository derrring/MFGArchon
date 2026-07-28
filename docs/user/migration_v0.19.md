# Migration Guide: v0.18.x → v0.19.0

This release removes the legacy `mfgarchon.config.pydantic_config` module. Every public symbol it exported is available — with possibly different defaults and fields — from the canonical `mfgarchon.config` package.

## Why this breaks

Audit of the dual config system found that `pydantic_config.py` was NOT a simple deprecation of the canonical `core.py` + `mfg_methods.py` — the two hierarchies had diverged into **structurally different APIs**:

- Fields present on one side were absent on the other (both directions).
- Default values differed, in some cases by three orders of magnitude (`PicardConfig.tolerance` was `1e-3` in legacy, `1e-6` in canonical).
- "Backward compatibility aliases" at the bottom of `pydantic_config.py` pointed at internal `_underscored` duplicates, not at the canonical classes.

Fixing this through a soft deprecation was impossible — the APIs weren't versions of one another, they were different APIs that happened to share names. The only honest path was a hard break, documented here.

## Step 1: Update imports (mechanical)

Every import from `mfgarchon.config.pydantic_config` becomes an import from `mfgarchon.config`:

```diff
- from mfgarchon.config.pydantic_config import MFGSolverConfig
+ from mfgarchon.config import MFGSolverConfig

- from mfgarchon.config.pydantic_config import NewtonConfig, PicardConfig
+ from mfgarchon.config import NewtonConfig, PicardConfig

- from mfgarchon.config.pydantic_config import HJBConfig, FPConfig
+ from mfgarchon.config import HJBConfig, FPConfig
```

`MFGSolverConfig` itself was already a transparent re-export of `core.MFGSolverConfig` in v0.18.x, so the behavior is unchanged for that one class. The other six classes were genuinely different — see Step 2.

## Step 2: Update constructor kwargs (field-by-field)

The divergence matrix below enumerates every field difference per class.  Column key:
- **Legacy**: what you got from `pydantic_config`.
- **Canonical**: what you get from `mfgarchon.config` in v0.19.0+.

### NewtonConfig

| Field | Legacy default | Canonical default | Action |
|---|---|---|---|
| `max_iterations` | 30 | **10** | Pass explicitly if you relied on 30 |
| `tolerance` | 1e-6 | 1e-6 | No change |
| `damping_factor` | 1.0 | **removed** | Use `relaxation` |
| `line_search` | False | **removed** | No canonical equivalent |
| `verbose` | False | **removed** | Use `configure_research_logging(level="DEBUG")` |
| `relaxation` | — | 1.0 | New field, under-relaxation parameter |

### PicardConfig (largest drift)

| Field | Legacy default | Canonical default | Action |
|---|---|---|---|
| `max_iterations` | 20 | **100** | Pass explicitly if you relied on 20 |
| `tolerance` | **1e-3** | **1e-6** | 1000x stricter — pass explicitly if you relied on 1e-3 |
| `verbose` | False | **True** | Pass `verbose=False` to silence progress |
| `anderson_memory` | — | set by canonical | New field |
| `damping_factor_M` | — | set by canonical | New field (separate damping for density) |
| `damping_schedule` | — | set by canonical | New field |
| `damping_schedule_M` | — | set by canonical | New field |

### GFDMConfig

| Field | Legacy | Canonical | Action |
|---|---|---|---|
| `weight_function` | `'gaussian'` | `'wendland'` | Pass `weight_function='gaussian'` if you relied on the old default |
| `constraint_tolerance` | 1e-6 (approx) | **removed** | No canonical equivalent; QP constraint tolerance now lives inside `qp` sub-config |
| `max_neighbors` | set | **removed** | Use `neighborhood` sub-config |
| `use_qp_constraints` | set | **removed** | Use `qp` sub-config presence to enable |
| `boundary_accuracy` | — | set | New: sub-config |
| `congestion_mode` | — | set | New |
| `derivative`, `neighborhood`, `qp` | — | set | New: sub-configs |
| `taylor_order`, `weight_scale` | — | set | New |

### ParticleConfig

| Field | Legacy | Canonical | Action |
|---|---|---|---|
| `kde_bandwidth` | `float` (default 0.01) | `float \| 'auto'` (default `'auto'`) | Pass `0.01` explicitly to restore old default |
| `adaptive_particles` | set | **removed** | No canonical equivalent |
| `boundary_treatment` | set | **removed** | Handled by solver-level BC |
| `resampling_method` | set | **removed** | Handled internally |
| `external_particles`, `mode`, `normalization` | — | set | New |

### HJBConfig

| Field | Legacy | Canonical | Action |
|---|---|---|---|
| `solver_type` | string | **removed** | Use `method` instead |
| `method` | — | required | New single field naming the HJB method (`"fdm"`, `"fem"`, `"gfdm"`, `"sl"`, `"weno"`) |
| `gfdm` | required (`_GFDMConfig`) | optional (`GFDMConfig \| None`) | Default is now `None` |
| `newton` | (`_NewtonConfig`) | (`NewtonConfig`) | Same pattern as above |
| `accuracy_order`, `boundary_conditions`, `fdm`, `fem`, `sl`, `weno` | — | set | New sub-configs |

### FPConfig

| Field | Legacy | Canonical | Action |
|---|---|---|---|
| `solver_type` | string | **removed** | Use `method` instead |
| `time_integration` | string | **removed** | Moved into method-specific sub-config |
| `particle` | required (`_ParticleConfig`) | optional (`ParticleConfig \| None`) | Default is `None` |
| `method`, `fdm`, `fem`, `network` | — | set | New |

### MFGSolverConfig

No field-level changes. `MFGSolverConfig` from `pydantic_config` was already a transparent re-export of `core.MFGSolverConfig`. Only the import path changed.

## Step 3: Phantom factories are not replaced

User docs previously referenced four functions that **never existed** as public API:

```python
# None of these work in v0.18.x or v0.19.0+
from mfgarchon.config.pydantic_config import (
    create_fast_config,         # never existed
    create_accurate_config,     # never existed
    create_research_config,     # never existed
    create_enhanced_config,     # never existed
)
```

If you were following those doc examples, you were already getting `ImportError` — the code simply was never there.

The `mfgarchon.factory` presets this section once pointed at as the replacement —
`create_fast_solver`, `create_accurate_solver`, `create_research_solver` — have since been
removed too, and raise `NotImplementedError` on call (Issue #1709). There is no preset
layer. Solve through the problem, and pass a `MFGSolverConfig` when you need explicit
control:

```python
result = problem.solve()                                  # defaults
result = problem.solve(max_iterations=200, tolerance=1e-8)
result = problem.solve(config=MFGSolverConfig(...))        # explicit control
```

## Step 4: Update YAML configs if affected

If you have YAML configuration files that set fields now removed from canonical (e.g. `newton.damping_factor`, `newton.line_search`, `newton.verbose`, `gfdm.constraint_tolerance`, or similar), Pydantic's strict validation at `bridge_to_pydantic()` will now raise `ValidationError` pointing at the unknown field. Either remove the field, or map it to its canonical replacement per the tables above.

## Questions

If the tables above leave a gap — a legacy field you relied on with no canonical equivalent listed — open an issue with the specific usage. Some features (e.g. `line_search` in Newton) were never actually wired into any solver even in legacy form; others (e.g. `damping_factor`) have direct replacements; a few may deserve reinstating as canonical fields if they represent real capability.

## Context

This release is the first of a 5-step internal refactor: B1 (this release) removes
the legacy `pydantic_config` module; B2–B5 follow as v0.19.x patches to add tests
for the canonical modules, migrate the YAML loader pipeline, remove the remaining
OmegaConf dataclass mirrors, and fix an unrelated `ExperimentConfig` forward-ref
issue surfaced by pydantic 2.12.5. None of those subsequent steps are user-facing.
