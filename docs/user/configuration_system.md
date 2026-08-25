# Configuration System User Guide

**Version**: 0.22+
**Last Updated**: 2026-08-25

---

## Overview

MFGArchon's configuration is **Pydantic models, and nothing else**. You build a config in Python or
load one from a flat YAML file; the models validate it, and a solver consumes the validated object.

This guide previously described a *dual-system* architecture — Pydantic for validation, OmegaConf
for YAML transport, with a bridge between them. **OmegaConf was removed in #1687** on the principle
that a numerical library should not own a config framework: it should accept typed objects, and
loading config files belongs to the application or experiment layer. What that removal took with it,
and where the responsibility went, is at the bottom of this page.

---

## Building a config in Python

Best for: unit tests, scripts, anything programmatic. This is the primary path.

```python
from mfgarchon.config import MFGSolverConfig, HJBConfig, FPConfig, ParticleConfig, PicardConfig

# The structure is nested: solver-specific settings live under the method that owns them, and
# iteration control lives under `picard`.
config = MFGSolverConfig(
    hjb=HJBConfig(method="gfdm", accuracy_order=2),
    fp=FPConfig(method="particle", particle=ParticleConfig(num_particles=5000)),
    picard=PicardConfig(max_iterations=100, tolerance=1e-8),
)

result = problem.solve(config=config)
```

### The shape matters, and a wrong key raises

Since #1766 the models carry `extra="forbid"` (`config/core.py`). An unknown key, or a nested key
written at the top level, **raises** instead of being silently dropped:

```python
MFGSolverConfig(tolerance=1e-8)          # raises: tolerance lives under `picard`
MFGSolverConfig(picard={"toleranse": 1})  # raises: extra inputs are not permitted
```

Deprecated field names are the one exception and still work, translated with a
`DeprecationWarning`. `LEGACY_FIELD_ALIASES` in `config/core.py` is the single owner of that list.

---

## Loading and saving YAML

`mfgarchon.config` exposes three functions, all in `config/io.py`:

```python
from mfgarchon.config import load_solver_config, save_solver_config, validate_yaml_config

config = load_solver_config("experiments/baseline.yaml")   # -> validated SolverConfig
save_solver_config(config, "experiments/effective.yaml")   # round-trips through model_dump
ok, message = validate_yaml_config("experiments/baseline.yaml")
```

The schema is **flat**: the YAML's top-level keys are the model's field names. A `solver:`-wrapped
file is a different format and `load_solver_config` refuses it by name rather than dropping the
keys — that shape came from the removed OmegaConf layer, and unwrapping it is your loader's job.

Interpolation (`${...}`), config composition and merging are OmegaConf features and are **gone**.
If an experiment needs them, they belong in the experiment's own loader, which can hand this
library a plain dict or a built model.

---

## Configuration classes

| Group | Classes |
|:------|:--------|
| Top level | `MFGSolverConfig`, `SolverConfig`, `ExperimentConfig`, `BaseConfig` |
| HJB | `HJBConfig`, `GFDMConfig`, `FDMConfig`, `FEMConfig`, `WENOConfig`, `SLConfig`, `NewtonConfig` |
| FP | `FPConfig`, `ParticleConfig`, `NetworkConfig` |
| Iteration | `PicardConfig` |
| Numerics | `DerivativeConfig`, `NeighborhoodConfig`, `QPConfig`, `BoundaryAccuracyConfig`, `CollocationConfig` |
| Infrastructure | `BackendConfig`, `LoggingConfig`, `MFGGridConfig`, `MFGArrays`, `ArrayValidationConfig` |

Translation helpers (`hjb_config_to_kwargs`, `fp_config_to_kwargs`,
`picard_config_to_iterator_kwargs`, `backend_config_to_kwargs`, `translate_solver_config`) turn a
validated config into the keyword arguments a solver constructor expects. They are the seam between
the config layer and the solvers; prefer them over reading fields by hand.

---

## What #1687 removed

| Removed | Where the responsibility went |
|:--------|:------------------------------|
| `OmegaConfManager`, `create_omega_manager` | The application's own loader |
| `bridge_to_pydantic` | `model_validate(...)` — with one config system there is nothing to bridge. Use `strict=True` for a dict you built in Python, where the bridge's behaviour carried over exactly: `{'max_iterations': '100'}` raised there and a plain `model_validate` coerces it to `int`. **Do not use it on data that came from `yaml.safe_load`** — see the note below |
| `save_effective_config` / `load_effective_config` | `save_solver_config` writes YAML; `model_dump_json` writes JSON |
| `create_parameter_sweep_configs` | The experiment layer; a sweep is a loop over built configs |
| `mfgarchon/config/configs/*.yaml` | Shipped defaults for the removed manager |
| YAML interpolation, composition, merging | OmegaConf features, not library features |

`OMEGACONF_AVAILABLE` is gone from `mfgarchon.config`. Code that branched on it can drop the branch.

### One behaviour change to know about: YAML scalars

PyYAML implements YAML 1.1, whose float rule requires **both a decimal point and a signed
exponent**. OmegaConf resolved scientific notation itself and handed Pydantic a `float`; PyYAML
hands it a `str`:

| spelling in YAML | `yaml.safe_load` gives |
|:-----------------|:-----------------------|
| `1e-8`, `1E-8`, `1e8`, `-1e-8` | `str` |
| `1.0e-8`, `1.e-8` | `float` |

`load_solver_config` is unaffected: it calls a non-strict `model_validate`, which coerces the
string back to `float`. So `tolerance: 1e-8` still loads correctly.

It matters if you validate YAML yourself with `strict=True`, which rejects the string outright
(`Input should be a valid number [type=float_type, input_value='1e-8', input_type=str]`). Either
drop `strict=True` on the YAML path, or write `1.0e-8`. Nothing shipped in this repository was
affected — the removed default configs all used the `1.0e-6` spelling.
