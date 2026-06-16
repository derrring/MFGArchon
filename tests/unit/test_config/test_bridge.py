"""Tests for config bridge utilities."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from mfgarchon.config import MFGSolverConfig

# Skip all tests if OmegaConf is not available
pytest.importorskip("omegaconf")


class TestBridgeToPydantic:
    """Tests for bridge_to_pydantic function."""

    def test_simple_conversion(self) -> None:
        """Test basic OmegaConf to Pydantic conversion."""
        from omegaconf import OmegaConf

        from mfgarchon.config.bridge import bridge_to_pydantic

        # Create OmegaConf config with nested structure
        omega_cfg = OmegaConf.create(
            {
                "picard": {
                    "tolerance": 1e-8,
                    "max_iterations": 200,
                },
            }
        )

        # Convert to Pydantic
        config = bridge_to_pydantic(omega_cfg, MFGSolverConfig)

        assert config.picard.tolerance == 1e-8
        assert config.picard.max_iterations == 200

    def test_nested_config(self) -> None:
        """Test conversion with nested configurations."""
        from omegaconf import OmegaConf

        from mfgarchon.config.bridge import bridge_to_pydantic

        omega_cfg = OmegaConf.create(
            {
                "picard": {
                    "tolerance": 1e-6,
                },
                "hjb": {
                    "method": "gfdm",
                },
                "fp": {
                    "method": "particle",
                },
            }
        )

        config = bridge_to_pydantic(omega_cfg, MFGSolverConfig)

        assert config.picard.tolerance == 1e-6
        assert config.hjb.method == "gfdm"
        assert config.fp.method == "particle"

    def test_interpolation_resolution(self) -> None:
        """Test that OmegaConf interpolations are resolved."""
        from omegaconf import OmegaConf

        from mfgarchon.config.bridge import bridge_to_pydantic

        omega_cfg = OmegaConf.create(
            {
                "base_tol": 1e-6,
                "picard": {
                    "tolerance": "${base_tol}",  # Interpolation
                    "max_iterations": 100,
                },
            }
        )

        config = bridge_to_pydantic(omega_cfg, MFGSolverConfig)

        assert config.picard.tolerance == 1e-6

    def test_validation_error(self) -> None:
        """Test that Pydantic validation errors are raised."""
        from omegaconf import OmegaConf
        from pydantic import ValidationError

        from mfgarchon.config.bridge import bridge_to_pydantic

        omega_cfg = OmegaConf.create(
            {
                "picard": {
                    "tolerance": "not_a_number",  # Invalid type
                },
            }
        )

        with pytest.raises(ValidationError):
            bridge_to_pydantic(omega_cfg, MFGSolverConfig)


class TestSaveEffectiveConfig:
    """Tests for save_effective_config function."""

    def test_save_config(self) -> None:
        """Test saving config to JSON file."""
        from mfgarchon.config.bridge import save_effective_config

        config = MFGSolverConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_effective_config(config, tmpdir)

            assert path.exists()
            assert path.name == "resolved_config.json"

            with open(path) as f:
                saved = json.load(f)

            # Check nested structure
            assert "picard" in saved
            assert saved["picard"]["tolerance"] == 1e-6

    def test_save_creates_directory(self) -> None:
        """Test that output directory is created if it doesn't exist."""
        from mfgarchon.config.bridge import save_effective_config

        config = MFGSolverConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / "nested" / "output"
            path = save_effective_config(config, nested_path)

            assert path.exists()
            assert nested_path.exists()

    def test_custom_filename(self) -> None:
        """Test saving with custom filename."""
        from mfgarchon.config.bridge import save_effective_config

        config = MFGSolverConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_effective_config(config, tmpdir, filename="custom.json")

            assert path.name == "custom.json"

    def test_include_defaults(self) -> None:
        """Test that defaults are included by default."""
        from mfgarchon.config.bridge import save_effective_config

        config = MFGSolverConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_effective_config(config, tmpdir)

            with open(path) as f:
                saved = json.load(f)

            # All sections should be present with defaults
            assert "hjb" in saved
            assert "fp" in saved
            assert "picard" in saved
            assert "backend" in saved


class TestLoadEffectiveConfig:
    """Tests for load_effective_config function."""

    def test_load_config(self) -> None:
        """Test loading config from JSON file."""
        from mfgarchon.config.bridge import load_effective_config, save_effective_config

        original = MFGSolverConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_effective_config(original, tmpdir)
            loaded = load_effective_config(path, MFGSolverConfig)

            assert loaded.picard.tolerance == original.picard.tolerance
            assert loaded.picard.max_iterations == original.picard.max_iterations

    def test_roundtrip(self) -> None:
        """Test full save/load roundtrip preserves all values."""
        from mfgarchon.config.bridge import load_effective_config, save_effective_config

        original = MFGSolverConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_effective_config(original, tmpdir)
            loaded = load_effective_config(path, MFGSolverConfig)

            assert loaded == original

    def test_roundtrip_with_custom_values(self) -> None:
        """Test roundtrip with custom config values."""
        from mfgarchon.config import PicardConfig
        from mfgarchon.config.bridge import load_effective_config, save_effective_config

        original = MFGSolverConfig(
            picard=PicardConfig(
                tolerance=1e-10,
                max_iterations=500,
                relaxation=0.8,
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_effective_config(original, tmpdir)
            loaded = load_effective_config(path, MFGSolverConfig)

            assert loaded.picard.tolerance == 1e-10
            assert loaded.picard.max_iterations == 500
            assert loaded.picard.relaxation == 0.8


class TestNoVestigialOmegaToPydanticBridge1392:
    """Issue #1392: OmegaConf->Pydantic validation has ONE canonical crossing
    (bridge_to_pydantic). The pre-North-Star OmegaConfManager.create_pydantic_config /
    _map_omega_to_pydantic pair (a second, broken bridge that silently returned a default
    config) was removed and must not be silently re-introduced."""

    def test_create_pydantic_config_removed(self) -> None:
        from mfgarchon.config.omegaconf_manager import OmegaConfManager

        assert not hasattr(OmegaConfManager, "create_pydantic_config")
        assert not hasattr(OmegaConfManager, "_map_omega_to_pydantic")

    def test_bridge_to_pydantic_is_the_canonical_crossing(self) -> None:
        from mfgarchon.config import bridge_to_pydantic

        assert callable(bridge_to_pydantic)
