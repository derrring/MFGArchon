# Configuration System User Guide

**Version**: 0.16+
**Last Updated**: 2025-12-10

---

## Overview

MFGArchon uses a **Dual-System Configuration Architecture** that balances runtime safety with experimental flexibility. This guide explains how to use the configuration system effectively.

---

## Architecture Diagrams

### 1. The Funnel Model (Core Concept)

This is the most macroscopic view. The configuration system acts like a funnel:

- **Upper Layer (OmegaConf)**: Wide opening, allows flexibility (interpolation, overrides, loose types)
- **Middle (Bridge)**: Filter screen, performs strict conversion
- **Lower Layer (Pydantic)**: Narrow opening, only allows absolutely correct, type-safe data through

```mermaid
graph TD
    subgraph User_Space [User Space: Flexibility First]
        A[User Input<br>CLI / YAML] -->|Load| B(OmegaConf DictConfig<br><i>transport only</i>)
        style A fill:#e1f5fe,stroke:#01579b
        style B fill:#e1f5fe,stroke:#01579b
        B -->|Interpolation| B_Res[Resolved Config<br>User Intent]
        style B_Res fill:#b3e5fc,stroke:#01579b
    end

    subgraph The_Bridge [Conversion Layer: Generic Adapter]
        B_Res -->|Generic Bridge| C{Validation<br>& Parsing}
        style C fill:#fff9c4,stroke:#fbc02d
    end

    subgraph Kernel_Space [Kernel Space: Safety First]
        C -->|Success| D[Pydantic Object<br><b>*Config</b>]
        C -->|Failure| E[ValidationError<br>Stop Execution]
        style D fill:#e8f5e9,stroke:#2e7d32
        style E fill:#ffebee,stroke:#c62828
        D -->|Immutable Args| F[Solver / Math Kernel]
        style F fill:#e8f5e9,stroke:#2e7d32
    end
```

### 2. Standard Execution Flow (Data Lifecycle)

This sequence diagram shows the complete journey of a configuration file in **Mode 3 (Hybrid/Production)** mode. Note the exact moments when **"Generic Bridge"** and **"Snapshot"** occur.

```mermaid
sequenceDiagram
    participant U as User (YAML)
    participant O as OmegaConf (transport)
    participant M as Manager (Bridge)
    participant P as Pydantic (*Config)
    participant D as Disk (Snapshot)
    participant S as Solver (Math)

    Note over U, O: Phase 1: Orchestration (Interface)
    U->>O: 1. Load experiment.yaml
    O->>O: 2. Merge CLI Overrides
    O->>O: 3. Resolve Interpolations (${...})
    Note right of O: Data is still weakly-typed DictConfig at this point

    Note over O, P: Phase 2: The Bridge (Validation)
    O->>M: 4. Pass DictConfig
    M->>P: 5. model_validate(dict)
    alt Validation Failed
        P--xU: Throw ValidationError (Early Stop)
    else Validation Passed
        P->>M: Return Strict Model (*Config)
    end

    Note over M, S: Phase 3: Execution & Reproducibility
    M->>D: 6. Save resolved_config.json (SSOT)
    M->>S: 7. Inject Pydantic Model
    S->>S: 8. Run Math Kernel
```

### 3. Parameter Sweeps (Mode 4: Research)

This is where the architecture shines. OmegaConf generates multiple "intents", while Pydantic validates each one. This is like a **factory assembly line**.

```mermaid
graph LR
    subgraph Generator [OmegaConf Generator]
        Base[Base YAML]
        Sweep[Sweep Config<br>alpha=[0.1, 0.5, 0.9]]
        Base -->|Hydra/Loop| Gen{Generate<br>Variations}
    end

    subgraph Pipeline [Execution Pipeline]
        Gen -->|Var 1| C1[Schema 1]
        Gen -->|Var 2| C2[Schema 2]
        Gen -->|Var 3| C3[Schema 3]

        C1 -->|Bridge| V1((Validate))
        C2 -->|Bridge| V2((Validate))
        C3 -->|Bridge| V3((Validate))

        V1 -->|Config 1| S1[Snapshot 1] --> Run1[Run 1]
        V2 -->|Config 2| S2[Snapshot 2] --> Run2[Run 2]
        V3 -->|Error!| Err[Stop Run 3]
    end

    style V1 fill:#a5d6a7
    style V2 fill:#a5d6a7
    style V3 fill:#ef9a9a
    style Err fill:#c62828,color:white
```

### Key Insights from Diagrams

1. **Clear Division of Labor**: From the diagrams, **OmegaConf (blue)** handles all "messy" things (merging, substitution, generation), while **Pydantic (green)** only receives clean, determined data.

2. **Snapshot Position**: In diagram 2, snapshot saving occurs **after validation passes, before solver runs**. This is the golden time point for ensuring reproducibility.

3. **Bridge as Isolation**: The Generic Bridge is the only channel connecting the two worlds, isolating complexity.

---

## Two Configuration Systems

| System | File | Suffix | Role | When to Use |
|:-------|:-----|:-------|:-----|:------------|
| **Pydantic** | `core.py` | `*Config` | Runtime validation | Direct Python usage, API calls |
| **OmegaConf** | `omegaconf_manager.py` | — | YAML transport only | Experiments, parameter sweeps |

### Why Two Systems?

- **Pydantic** ensures mathematical correctness (e.g., `tolerance > 0`, `Nx > 0`)
- **OmegaConf** provides experiment flexibility (YAML files, interpolation, merging)

---

## Usage Patterns

### Pattern 1: Direct Python (Pydantic Only)

Best for: Unit tests, simple scripts, programmatic configuration.

```python
from mfgarchon.config import MFGSolverConfig, HJBConfig, FPConfig

# Create configuration directly in Python
config = MFGSolverConfig(
    hjb=HJBConfig(method="gfdm", accuracy_order=2),
    fp=FPConfig(method="particle", num_particles=5000),
    tolerance=1e-8,
    max_iterations=100,
)

# Use with solver
result = problem.solve(config=config)
```

### Pattern 2: YAML Configuration (OmegaConf Only)

Best for: Reproducible experiments, sharing configurations.

**experiment.yaml:**
```yaml
problem:
  name: "crowd_motion"
  T: 2.0
  Nx: 100
  Nt: 50

solver:
  type: "fixed_point"
  tolerance: 1e-8
  max_iterations: 200

  hjb:
    method: "gfdm"

  fp:
    method: "fdm"

experiment:
  name: "baseline_run"
  output_dir: "results/${experiment.name}"  # Interpolation!
```

**Python code:**
```python
from pathlib import Path

from mfgarchon.config import MFGSolverConfig, bridge_to_pydantic, create_omega_manager

manager = create_omega_manager()

# Pass an ABSOLUTE path. `load_config` resolves a relative name against the package's own
# config directory, so a bare "experiment.yaml" looks for mfgarchon/config/configs/, not
# your working directory.
cfg = manager.load_config(str(Path("experiment.yaml").resolve()))

# The OmegaConf object carries the YAML's own vocabulary
print(cfg.problem.T)        # 2.0
print(cfg.solver.tolerance)  # 1e-8

# Cross to Pydantic when you want validation. `bridge_to_pydantic` is that crossing
# (Issue #1010); the model's field names are its own (hjb, fp, picard, backend, logging),
# not the YAML's.
config = bridge_to_pydantic(cfg, MFGSolverConfig)
print(config.picard.max_iterations)
```

### Pattern 3: Hybrid (Production Recommended)

Best for: Production runs where both flexibility and validation matter.

```python
from pathlib import Path

from mfgarchon.config import MFGSolverConfig, bridge_to_pydantic, create_omega_manager

manager = create_omega_manager()

# 1. Load flexible YAML. Absolute path -- see the note in Pattern 2.
cfg = manager.load_config(str(Path("experiment.yaml").resolve()))

# 2. Compose, override or sweep while it is still an OmegaConf object
sweep = manager.create_parameter_sweep(cfg, {"solver.tolerance": [1e-6, 1e-8]})

# 3. Cross to Pydantic once, at the point where validation should happen (#1010)
config = bridge_to_pydantic(cfg, MFGSolverConfig)

# 4. Solve with the validated config
result = problem.solve(config=config)

# 4. Save effective config for reproducibility
manager.save_effective_config(pydantic_config, output_dir="results/run_001")
```

### Pattern 4: Parameter Sweeps (Research)

Best for: Running multiple experiments with varying parameters.

```python
from mfgarchon.config.omegaconf_manager import create_omega_manager

manager = create_omega_manager()
base_config = manager.load_config("experiment.yaml")

# Define sweep parameters
tolerances = [1e-6, 1e-8, 1e-10]
grid_sizes = [50, 100, 200]

for tol in tolerances:
    for Nx in grid_sizes:
        # Create variant
        config = manager.create_variant(base_config, {
            "solver.tolerance": tol,
            "problem.Nx": Nx,
        })

        # Bridge and validate
        pydantic_config = manager.bridge_to_pydantic(config, MFGSolverConfig)

        # Run experiment
        result = problem.solve(config=pydantic_config)

        # Save with effective config
        output_dir = f"results/tol_{tol}_Nx_{Nx}"
        manager.save_effective_config(pydantic_config, output_dir)
```

---

## Configuration Classes Reference

### Pydantic Classes (`mfgarchon.config`)

```python
from mfgarchon.config import (
    MFGSolverConfig,  # Root solver configuration
    HJBConfig,        # HJB equation solver settings
    FPConfig,         # Fokker-Planck solver settings
    NewtonConfig,     # Newton iteration settings
)
```

### YAML loading (`mfgarchon.config`)

There are no OmegaConf schema classes. `mfgarchon.config.structured_schemas` and its
`*Schema` dataclasses were removed in **v0.19.4** (`5f7d0800`, closing the B3+B4 items of
#1010): the config system has one schema authority, the Pydantic models above, and one
validation crossing. OmegaConf now handles YAML transport only.

```python
from mfgarchon.config import (
    create_omega_manager,    # YAML -> OmegaConf: composition, overrides, parameter sweeps
    bridge_to_pydantic,      # OmegaConf -> Pydantic: THE validation crossing (#1010)
    load_solver_config,      # YAML -> SolverConfig, for the FLAT solver schema only
    load_experiment_config,  # composed experiment config, with **overrides
    save_effective_config,   # write the config a run actually used, as resolved_config.json
    load_effective_config,   # read one of those snapshots BACK -- it parses JSON, not YAML
)
```

Two of these are easy to reach for and wrong:

- **`load_effective_config` does not read YAML.** It is the read-back half of
  `save_effective_config`, and it calls `json.load` on a `resolved_config.json` that a previous
  run wrote. Handing it a YAML file raises `JSONDecodeError` from inside a function that exists,
  which is a worse diagnostic than a missing name would have been.
- **`load_solver_config` does parse YAML, but only the flat solver schema.** A nested tree with
  `problem:` / `solver:` / `experiment:` keys is rejected with
  `Unknown top-level key(s) [...] The solver-config schema is flat`.

---

## Data Flow Summary

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  YAML File   │────►│   OmegaConf  │────►│   Pydantic   │────►│    Solver    │
│              │     │  (transport) │     │   *Config    │     │              │
│  (Flexible)  │     │  (untyped)   │     │  (Validated) │     │  (Executes)  │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                           │                     │
                           │                     │
                     Interpolation          Validation
                     ${...} syntax          tolerance > 0
                     Config merging         Nx > 0
                     CLI overrides          Type coercion
```

---

## Best Practices

### 1. Always Save Effective Configs

For reproducibility, always save the resolved configuration:

```python
# After bridging to Pydantic
manager.save_effective_config(pydantic_config, output_dir)
```

This creates `resolved_config.json` with all defaults filled and interpolations resolved.

### 2. Use Type Hints

Both systems support IDE autocompletion:

```python
# Pydantic - fully typed
config: MFGSolverConfig = MFGSolverConfig(...)
config.tolerance  # IDE knows this is float

# OmegaConf carries the YAML's own vocabulary; the Pydantic class owns its own schema.
# The two are different namespaces, and the crossing is where validation happens.
cfg = create_omega_manager().load_config(str(Path("config.yaml").resolve()))
cfg.solver.tolerance                          # the YAML's key, untyped
bridge_to_pydantic(cfg, MFGSolverConfig).picard.max_iterations  # the model's field, typed
```

### 3. Validate Early

Always bridge to Pydantic before running expensive computations:

```python
# Good: Validate before solve
pydantic_config = manager.bridge_to_pydantic(omega_config, MFGSolverConfig)
result = problem.solve(config=pydantic_config)  # Will not fail on bad config

# Bad: Validate during solve
result = problem.solve(config=omega_config)  # May fail mid-computation
```

### 4. Use Interpolation for DRY Configs

```yaml
experiment:
  name: "convergence_study"
  output_dir: "results/${experiment.name}"  # Automatically: results/convergence_study

problem:
  Nx: 100
  Nt: ${problem.Nx}  # Same as Nx for isotropic grids
```

---

## Troubleshooting

### "DictConfig has no attribute 'X'"

**Cause**: Using raw OmegaConf without structured schemas.

**Solution**: Use the structured loading functions:
```python
# Bad
config = OmegaConf.load("config.yaml")  # Returns DictConfig -- never validated

# Good
cfg = create_omega_manager().load_config(str(Path("config.yaml").resolve()))
config = bridge_to_pydantic(cfg, MFGSolverConfig)  # validated at the crossing
```

### Validation Error on Bridge

**Cause**: YAML values don't meet Pydantic constraints.

**Solution**: Check the error message and fix the YAML:
```
ValidationError: tolerance must be positive, got -1e-8
```

### Missing Fields After Load

**Cause**: YAML doesn't specify all fields, but schema has defaults.

**Solution**: This is normal - defaults are automatically filled. Check with:
```python
print(OmegaConf.to_yaml(config))  # See all values including defaults
```

---

## See Also

- `mfgarchon/config/core.py` - Pydantic class definitions
- `mfgarchon/config/omegaconf_manager.py` - Manager utilities
