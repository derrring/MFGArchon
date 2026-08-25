#!/usr/bin/env python3
"""
Unit tests for mfgarchon/utils/cli.py

Tests command-line interface utilities including:
- Argument parser creation and configuration
- Configuration file loading (JSON/YAML)
- Configuration file saving
- Configuration merging
- Argument to configuration conversion
- Error handling and validation
"""

import argparse
import json
import re
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from mfgarchon.utils.cli import (
    args_to_config,
    create_base_parser,
    create_solver_cli,
    load_config_file,
    merge_configs,
    save_config_file,
)

# ===================================================================
# Test Parser Creation
# ===================================================================


@pytest.mark.unit
def test_create_base_parser():
    """Test create_base_parser() creates valid ArgumentParser."""
    parser = create_base_parser()

    assert isinstance(parser, argparse.ArgumentParser)
    assert parser.description is not None
    assert "MFG" in parser.description


@pytest.mark.unit
def test_base_parser_default_arguments():
    """Test base parser has expected default arguments."""
    parser = create_base_parser()
    args = parser.parse_args([])

    # Problem configuration defaults
    assert args.T == 1.0
    assert args.Nt == 50
    assert args.xmin == 0.0
    assert args.xmax == 1.0
    assert args.Nx == 100

    # Solver configuration defaults
    assert args.solver_type == "fixed_point"
    assert args.preset == "balanced"
    assert args.max_iterations is None
    assert args.tolerance is None

    # Execution defaults
    assert args.verbose is False
    assert args.quiet is False
    assert args.progress is True
    assert args.timing is True


@pytest.mark.unit
def test_base_parser_custom_arguments():
    """Test base parser accepts custom arguments."""
    parser = create_base_parser()
    args = parser.parse_args(
        [
            "--T",
            "2.0",
            "--Nt",
            "100",
            "--Nx",
            "200",
            "--solver-type",
            "fixed_point",
            "--preset",
            "accurate",
            "--max-iterations",
            "500",
            "--tolerance",
            "1e-8",
            "--verbose",
        ]
    )

    assert args.T == 2.0
    assert args.Nt == 100
    assert args.Nx == 200
    assert args.solver_type == "fixed_point"
    assert args.preset == "accurate"
    assert args.max_iterations == 500
    assert args.tolerance == 1e-8
    assert args.verbose is True


@pytest.mark.unit
def test_base_parser_progress_flags():
    """Test progress and no-progress flags."""
    parser = create_base_parser()

    # Default: progress enabled
    args1 = parser.parse_args([])
    assert args1.progress is True

    # Explicit --progress
    args2 = parser.parse_args(["--progress"])
    assert args2.progress is True

    # Disable with --no-progress
    args3 = parser.parse_args(["--no-progress"])
    assert args3.progress is False


@pytest.mark.unit
def test_base_parser_timing_flags():
    """Test timing and no-timing flags."""
    parser = create_base_parser()

    # Default: timing enabled
    args1 = parser.parse_args([])
    assert args1.timing is True

    # Disable with --no-timing
    args2 = parser.parse_args(["--no-timing"])
    assert args2.timing is False


@pytest.mark.unit
def test_create_solver_cli():
    """Test create_solver_cli() creates CLI with subcommands."""
    parser = create_solver_cli()

    assert isinstance(parser, argparse.ArgumentParser)

    # Test with solve subcommand
    args = parser.parse_args(["solve"])
    assert args.command == "solve"


@pytest.mark.unit
def test_solver_cli_solve_subcommand():
    """Test solve subcommand with problem file."""
    parser = create_solver_cli()

    args = parser.parse_args(["solve", "problem.py", "--problem-class", "MyProblem"])
    assert args.command == "solve"
    assert args.problem_file == "problem.py"
    assert args.problem_class == "MyProblem"


@pytest.mark.unit
def test_solver_cli_config_generate_subcommand():
    """Test config generate subcommand."""
    parser = create_solver_cli()

    args = parser.parse_args(["config", "generate", "output.json", "--format", "json"])
    assert args.command == "config"
    assert args.config_command == "generate"
    assert args.output_file == "output.json"
    assert args.format == "json"


@pytest.mark.unit
def test_solver_cli_config_validate_subcommand():
    """Test config validate subcommand."""
    parser = create_solver_cli()

    args = parser.parse_args(["config", "validate", "config.json"])
    assert args.command == "config"
    assert args.config_command == "validate"
    assert args.config_file == "config.json"


# ===================================================================
# Test Configuration File Loading
# ===================================================================


@pytest.mark.unit
def test_load_config_file_json():
    """Test loading JSON configuration file."""
    config_data = {"problem": {"T": 2.0, "Nt": 100}, "solver": {"preset": "accurate"}}

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        loaded_config = load_config_file(config_path)
        assert loaded_config == config_data


@pytest.mark.unit
def test_load_config_file_yaml():
    """Test loading YAML configuration file."""
    pytest.importorskip("yaml")

    config_data = {"problem": {"T": 2.0, "Nt": 100}, "solver": {"preset": "accurate"}}

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"

        import yaml

        with open(config_path, "w") as f:
            yaml.safe_dump(config_data, f)

        loaded_config = load_config_file(config_path)
        assert loaded_config == config_data


@pytest.mark.unit
def test_load_config_file_not_found():
    """Test load_config_file() raises FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        load_config_file("nonexistent.json")


@pytest.mark.unit
def test_load_config_file_invalid_json():
    """Test load_config_file() raises ValueError for invalid JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "invalid.json"
        with open(config_path, "w") as f:
            f.write("{ invalid json }")

        with pytest.raises(ValueError, match="Error loading config file"):
            load_config_file(config_path)


@pytest.mark.unit
def test_load_config_file_unsupported_format():
    """Test load_config_file() raises ValueError for unsupported format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.txt"
        config_path.write_text("some text")

        with pytest.raises(ValueError, match="Unsupported config file format"):
            load_config_file(config_path)


@pytest.mark.unit
def test_pyyaml_is_declared_because_the_code_imports_it_unconditionally():
    """Replaces `test_load_config_file_yaml_without_pyyaml`, which #1687 made unreachable.

    That test patched `mfgarchon.utils.cli.YAML_AVAILABLE` to False to reach a `raise` guarding a
    missing PyYAML. The flag and the guard are gone: `pyyaml` is a declared core dependency, so the
    ImportError branch cannot be entered and a test that patches a module attribute out of
    existence pins nothing.

    What IS worth pinning is the reason it became unreachable. `mfgarchon/config/io.py` and this
    module both `import yaml` at module scope, unconditionally -- so `import mfgarchon.config`
    fails outright on a fresh install if the declaration is dropped. Before #1687 the dependency
    was satisfied only transitively (omegaconf, and jupyterlab via jupyter-server -> jupyter-events)
    while never being declared, which is a latent bug that removing omegaconf would have detonated.
    This asserts the declaration is present, which is the invariant the unconditional imports need.
    """
    import tomllib

    pyproject = Path(__file__).resolve().parents[3] / "pyproject.toml"
    assert pyproject.is_file(), f"expected pyproject.toml at {pyproject}"
    declared = tomllib.loads(pyproject.read_text())["project"]["dependencies"]
    names = {re.split(r"[<>=!~\[ ]", d, maxsplit=1)[0].lower().replace("_", "-") for d in declared}

    assert "numpy" in names, f"the parse is wrong, not the population: got {sorted(names)}"
    assert "pyyaml" in names, (
        f"mfgarchon/config/io.py and mfgarchon/utils/cli.py import yaml at module scope, so pyyaml "
        f"must be a declared core dependency; core dependencies are {sorted(names)}"
    )


# ===================================================================
# Test Configuration File Saving
# ===================================================================


@pytest.mark.unit
def test_save_config_file_json():
    """Test saving configuration to JSON file."""
    config_data = {"problem": {"T": 2.0, "Nt": 100}, "solver": {"preset": "accurate"}}

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.json"
        save_config_file(config_data, config_path)

        assert config_path.exists()

        with open(config_path) as f:
            loaded = json.load(f)
        assert loaded == config_data


@pytest.mark.unit
def test_save_config_file_yaml():
    """Test saving configuration to YAML file."""
    pytest.importorskip("yaml")

    config_data = {"problem": {"T": 2.0, "Nt": 100}, "solver": {"preset": "accurate"}}

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        save_config_file(config_data, config_path)

        assert config_path.exists()

        import yaml

        with open(config_path) as f:
            loaded = yaml.safe_load(f)
        assert loaded == config_data


@pytest.mark.unit
def test_save_config_file_default_format():
    """Test save_config_file() defaults to JSON for unknown format."""
    config_data = {"problem": {"T": 2.0}}

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.txt"
        save_config_file(config_data, config_path)

        assert config_path.exists()

        with open(config_path) as f:
            loaded = json.load(f)
        assert loaded == config_data


@pytest.mark.unit
def test_save_config_file_with_non_serializable():
    """Test save_config_file() handles non-serializable objects with default=str."""
    from pathlib import Path as PathClass

    config_data = {"path": PathClass("/some/path"), "value": 42}

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.json"
        save_config_file(config_data, config_path)

        assert config_path.exists()

        with open(config_path) as f:
            loaded = json.load(f)

        assert loaded["value"] == 42
        assert "/some/path" in loaded["path"]


# ===================================================================
# Test Configuration Merging
# ===================================================================


@pytest.mark.unit
def test_merge_configs_simple():
    """Test merging simple configuration dictionaries."""
    base = {"a": 1, "b": 2, "c": 3}
    override = {"b": 20, "d": 4}

    merged = merge_configs(base, override)

    assert merged["a"] == 1
    assert merged["b"] == 20
    assert merged["c"] == 3
    assert merged["d"] == 4


@pytest.mark.unit
def test_merge_configs_nested():
    """Test merging nested configuration dictionaries."""
    base = {"problem": {"T": 1.0, "Nt": 50}, "solver": {"preset": "balanced"}}
    override = {"problem": {"Nt": 100}, "execution": {"verbose": True}}

    merged = merge_configs(base, override)

    assert merged["problem"]["T"] == 1.0
    assert merged["problem"]["Nt"] == 100
    assert merged["solver"]["preset"] == "balanced"
    assert merged["execution"]["verbose"] is True


@pytest.mark.unit
def test_merge_configs_deep_nested():
    """Test merging deeply nested configuration dictionaries."""
    base = {"level1": {"level2": {"a": 1, "b": 2}}}
    override = {"level1": {"level2": {"b": 20, "c": 3}}}

    merged = merge_configs(base, override)

    assert merged["level1"]["level2"]["a"] == 1
    assert merged["level1"]["level2"]["b"] == 20
    assert merged["level1"]["level2"]["c"] == 3


@pytest.mark.unit
def test_merge_configs_override_with_non_dict():
    """Test merging where override replaces dict with non-dict."""
    base = {"solver": {"preset": "balanced", "max_iterations": 100}}
    override = {"solver": "custom_solver"}

    merged = merge_configs(base, override)

    assert merged["solver"] == "custom_solver"


@pytest.mark.unit
def test_merge_configs_empty_base():
    """Test merging with empty base configuration."""
    base = {}
    override = {"a": 1, "b": 2}

    merged = merge_configs(base, override)

    assert merged == override


@pytest.mark.unit
def test_merge_configs_empty_override():
    """Test merging with empty override configuration."""
    base = {"a": 1, "b": 2}
    override = {}

    merged = merge_configs(base, override)

    assert merged == base


# ===================================================================
# Test Arguments to Configuration Conversion
# ===================================================================


@pytest.mark.unit
def test_args_to_config_default():
    """Test args_to_config() with default arguments."""
    args = argparse.Namespace(
        T=1.0,
        Nt=50,
        xmin=0.0,
        xmax=1.0,
        Nx=100,
        solver_type="fixed_point",
        preset="balanced",
        max_iterations=None,
        tolerance=None,
        num_particles=None,
        warm_start=False,
        return_structured=True,
        verbose=False,
        quiet=False,
        progress=True,
        timing=True,
        profile=False,
        output=None,
        save_config=None,
    )

    config = args_to_config(args)

    assert config["problem"]["T"] == 1.0
    assert config["problem"]["Nt"] == 50
    assert config["problem"]["Nx"] == 100
    assert config["solver"]["type"] == "fixed_point"
    assert config["solver"]["preset"] == "balanced"
    assert config["execution"]["verbose"] is False
    assert config["execution"]["progress"] is True


@pytest.mark.unit
def test_args_to_config_with_overrides():
    """Test args_to_config() with override parameters."""
    args = argparse.Namespace(
        T=2.0,
        Nt=100,
        xmin=-1.0,
        xmax=2.0,
        Nx=200,
        solver_type="fixed_point",
        preset="accurate",
        max_iterations=500,
        tolerance=1e-8,
        num_particles=1000,
        warm_start=True,
        return_structured=True,
        verbose=True,
        quiet=False,
        progress=True,
        timing=True,
        profile=True,
        output="results.json",
        save_config="config.json",
    )

    config = args_to_config(args)

    assert config["problem"]["T"] == 2.0
    assert config["problem"]["Nt"] == 100
    assert config["problem"]["xmin"] == -1.0
    assert config["solver"]["max_iterations"] == 500
    assert config["solver"]["tolerance"] == 1e-8
    assert config["solver"]["num_particles"] == 1000
    assert config["solver"]["warm_start"] is True
    assert config["execution"]["verbose"] is True
    assert config["execution"]["profile"] is True
    assert config["io"]["output"] == "results.json"


@pytest.mark.unit
def test_args_to_config_quiet_overrides():
    """Test args_to_config() where quiet flag suppresses output."""
    args = argparse.Namespace(
        T=1.0,
        Nt=50,
        xmin=0.0,
        xmax=1.0,
        Nx=100,
        solver_type="fixed_point",
        preset="balanced",
        max_iterations=None,
        tolerance=None,
        num_particles=None,
        warm_start=False,
        return_structured=True,
        verbose=True,
        quiet=True,
        progress=True,
        timing=True,
        profile=False,
        output=None,
        save_config=None,
    )

    config = args_to_config(args)

    # quiet should suppress verbose, progress, and timing
    assert config["execution"]["verbose"] is False
    assert config["execution"]["progress"] is False
    assert config["execution"]["timing"] is False


@pytest.mark.unit
def test_args_to_config_no_optional_overrides():
    """Test args_to_config() when optional overrides are None."""
    args = argparse.Namespace(
        T=1.0,
        Nt=50,
        xmin=0.0,
        xmax=1.0,
        Nx=100,
        solver_type="fixed_point",
        preset="balanced",
        max_iterations=None,
        tolerance=None,
        num_particles=None,
        warm_start=False,
        return_structured=True,
        verbose=False,
        quiet=False,
        progress=True,
        timing=True,
        profile=False,
        output=None,
        save_config=None,
    )

    config = args_to_config(args)

    # Optional overrides should not appear in config
    assert "max_iterations" not in config["solver"]
    assert "tolerance" not in config["solver"]
    assert "num_particles" not in config["solver"]


# ===================================================================
# Test Integration with Config Files
# ===================================================================


@pytest.mark.unit
def test_cli_config_file_integration():
    """Test integration: parse args, load config, merge, convert."""
    parser = create_base_parser()

    # Create config file
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.json"
        file_config = {"problem": {"T": 3.0, "Nt": 150}, "solver": {"preset": "accurate"}}

        with open(config_path, "w") as f:
            json.dump(file_config, f)

        # Parse args with config file reference
        args = parser.parse_args(["--config", str(config_path), "--Nx", "300", "--verbose"])

        # Load config file
        loaded_config = load_config_file(args.config)

        # Convert args to config
        args_config = args_to_config(args)

        # Merge: args config as base, file config overrides file-level params
        # NOTE: args_to_config() always creates full config with all defaults,
        # so args config will override file config even for non-specified values.
        # This is the actual behavior - CLI args take precedence.
        final_config = merge_configs(loaded_config, args_config)

        # Verify merged config:
        # - args_to_config creates full config with all defaults
        # - args config overrides file config (even for defaults)
        # - Only Nx was explicitly set in CLI, but all args have defaults
        assert final_config["problem"]["Nx"] == 300  # explicitly set in CLI
        assert final_config["execution"]["verbose"] is True  # explicitly set in CLI

        # These come from args defaults (not file), showing CLI precedence:
        assert final_config["problem"]["T"] == 1.0  # args default overrides file
        assert final_config["solver"]["preset"] == "balanced"  # args default overrides file


# ===================================================================
# Test Error Handling
# ===================================================================


@pytest.mark.unit
def test_parser_invalid_solver_type():
    """Test parser rejects invalid solver type."""
    parser = create_base_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--solver-type", "invalid_solver"])


@pytest.mark.unit
def test_parser_invalid_preset():
    """Test parser rejects invalid preset."""
    parser = create_base_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--preset", "invalid_preset"])


@pytest.mark.unit
def test_save_config_handles_exception():
    """Test save_config_file() handles exceptions gracefully."""
    config_data = {"problem": {"T": 2.0}}

    with patch("builtins.open", side_effect=PermissionError("Cannot write")), patch("builtins.print") as mock_print:
        # Should not raise - handles exception gracefully
        save_config_file(config_data, "/invalid/path/config.json")

    # Verify it printed an error message (graceful handling)
    assert mock_print.called, "save_config_file should print error message on failure"
