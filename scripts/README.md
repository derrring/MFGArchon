# MFGArchon Scripts Directory

**Status**: ✅ Audited and Current (2025-10-04)

This directory contains utility scripts for development, environment management, and verification.

**Recent Changes**:
- ✅ Archived obsolete migration scripts (update_package_imports.py, fix_polars_stubs.py)
- ✅ Verified all remaining scripts are active and useful
- ✅ See `docs/development/SCRIPTS_AND_HOOKS_AUDIT.md` for comprehensive analysis

## Development Environment Setup

### `setup_development.sh` ⚡ **NEW**
Modern development environment setup supporting both UV and pip workflows.

**Usage:**
```bash
# Set up complete development environment (auto-detects UV or pip)
./scripts/setup_development.sh

# Verify existing installation
./scripts/setup_development.sh verify

# Show help
./scripts/setup_development.sh help
```

**Features:**
- ✅ Auto-detects UV for fast modern setup
- ✅ Falls back to traditional pip workflow
- ✅ Installs development dependencies
- ✅ Sets up pre-commit hooks
- ✅ Runs verification checks

## Development Tools

### `quick_type_check.py` ⚡ **NEW**
Rapid type checking and linting for development feedback.

**Usage:**
```bash
# Quick type check and linting
python scripts/quick_type_check.py

# From anywhere in project
./scripts/quick_type_check.py
```

**Features:**
- ✅ Fast mypy checking with error count
- ✅ Ruff linting integration
- ✅ Performance timing
- ✅ Focused output for rapid feedback

### `verify_modernization.py`
Comprehensive verification of package modernization and tool configuration.

**Usage:**
```bash
# Run full modernization verification
python scripts/verify_modernization.py
```

**Features:**
- ✅ pyproject.toml configuration verification
- ✅ Package import testing
- ✅ Typing modernization analysis
- ✅ Development tools validation

## Experimental Scripts

### `experimental/` Directory
Research and experimental API/type system design work.

**Contents:**
- `progressive_api_design.py` - API design experiments
- `smart_defaults_strategy.py` - Default value experiments
- `type_system_proposal.py` - Type system experiments

**Status:** Active research - proposals may be implemented in future versions

## Legacy Environment Management

### `manage_environments.sh`

> `create-dev` builds a conda **substrate** (interpreter plus BLAS-linked numpy/scipy) and installs
> the project from `pyproject.toml`; `environment.yml` no longer exists (#2167). `create-performance`
> reads `conda_performance.yml`, which was deleted in `dc47e5f7` and has been broken since — not
> fixed here, and not caused here.
Multi-environment management script for conda environments.

**Usage:**
```bash
# Create development environment
./scripts/manage_environments.sh create-dev

# Create performance-optimized environment
./scripts/manage_environments.sh create-performance

# Verify all environments
./scripts/manage_environments.sh verify

# Show environment information
./scripts/manage_environments.sh info
```

### `setup_env_vars.sh`
Environment variable configuration for development and research.

**Usage:**
```bash
# Source environment variables
source scripts/setup_env_vars.sh
```

## Quick Start Guide

For new developers:

```bash
# 1. Clone repository
git clone <repository-url>
cd MFGArchon

# 2. Run modern setup (handles everything)
./scripts/setup_development.sh

# 3. Activate environment
source .venv/bin/activate

# 4. Quick verification
python scripts/quick_type_check.py

# 5. Start developing! 🚀
```

## Development Guidelines

- All scripts should be executable (`chmod +x`)
- Include usage documentation in script headers
- Use consistent error handling and logging
- Test scripts in clean environments before committing
- Prefer UV-based workflows for performance when available

## Archived Scripts

### `archive/scripts/migration/`
Obsolete migration tools that successfully completed their tasks:
- `update_package_imports.py` - Paradigm reorganization complete
- `fix_polars_stubs.py` - Polars stubs now compatible

See `archive/scripts/migration/README.md` for details.

## Script Maintenance

Scripts are maintained alongside the main codebase. Update documentation when adding new scripts or modifying existing functionality.

**Recent Updates (2025-10-04):**
- ✅ Archived obsolete migration scripts
- ✅ Comprehensive audit completed (see `docs/development/SCRIPTS_AND_HOOKS_AUDIT.md`)
- ✅ All remaining scripts verified as active and useful
