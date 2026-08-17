"""
Unit tests for Backend Factory.

Tests backend registration, discovery, creation, and auto-selection logic.
"""

import sys
from pathlib import Path

import pytest

from mfgarchon.backends import (
    create_backend,
    get_available_backends,
    get_backend_info,
    register_backend,
)
from mfgarchon.backends.numpy_backend import NumPyBackend


@pytest.fixture
def clean_backend_registry():
    """Clean backend registry before and after tests."""
    import mfgarchon.backends as backends_module

    original_backends = backends_module._BACKENDS.copy()
    yield
    # Restore original registry
    backends_module._BACKENDS = original_backends


class TestBackendRegistration:
    """Test backend registration system."""

    def test_register_backend(self, clean_backend_registry):
        """Test registering a new backend."""
        import mfgarchon.backends as backends_module

        class CustomBackend(NumPyBackend):
            @property
            def name(self):
                return "custom"

        register_backend("custom", CustomBackend)
        assert "custom" in backends_module._BACKENDS
        assert backends_module._BACKENDS["custom"] is CustomBackend

    def test_register_backend_overwrites(self, clean_backend_registry):
        """Test that re-registering overwrites previous backend."""
        import mfgarchon.backends as backends_module

        class Backend1(NumPyBackend):
            pass

        class Backend2(NumPyBackend):
            pass

        register_backend("test", Backend1)
        assert backends_module._BACKENDS["test"] is Backend1

        register_backend("test", Backend2)
        assert backends_module._BACKENDS["test"] is Backend2

    def test_numpy_backend_always_registered(self):
        """Test that NumPy backend is always registered."""
        import mfgarchon.backends as backends_module

        assert "numpy" in backends_module._BACKENDS
        assert backends_module._BACKENDS["numpy"] is NumPyBackend


class TestGetAvailableBackends:
    """Test backend availability detection."""

    def test_numpy_always_available(self):
        """Test that numpy is always available."""
        available = get_available_backends()
        assert available["numpy"] is True

    def test_returns_dict(self):
        """Test that function returns dictionary."""
        available = get_available_backends()
        assert isinstance(available, dict)

    def test_torch_keys_present(self):
        """Test that torch-related keys are present."""
        available = get_available_backends()
        assert "torch" in available
        assert "torch_cuda" in available
        assert "torch_mps" in available

    def test_jax_keys_present(self):
        """Test that jax-related keys are present."""
        available = get_available_backends()
        assert "jax" in available
        assert "jax_gpu" in available

    def test_availability_values_are_boolean(self):
        """Test that all availability values are boolean."""
        available = get_available_backends()
        for key, value in available.items():
            assert isinstance(value, bool), f"Key {key} has non-boolean value {value}"

    def test_torch_cuda_requires_torch(self):
        """Test that torch_cuda is False if torch is False."""
        available = get_available_backends()
        if not available["torch"]:
            assert available["torch_cuda"] is False
            assert available["torch_mps"] is False

    def test_jax_gpu_requires_jax(self):
        """Test that jax_gpu is False if jax is False."""
        available = get_available_backends()
        if not available["jax"]:
            assert available["jax_gpu"] is False


class TestCreateBackend:
    """Test backend creation and auto-selection."""

    def test_create_numpy_backend_explicit(self):
        """Test explicit numpy backend creation."""
        backend = create_backend("numpy")
        assert backend.name == "numpy"

    def test_create_numpy_backend_with_precision(self):
        """Test numpy backend creation with custom precision."""
        backend = create_backend("numpy", precision="float32")
        assert backend.precision == "float32"

    def test_create_backend_auto_selects_available(self):
        """Test auto backend selection."""
        backend = create_backend("auto")
        # Torch backend name includes device type (torch_cuda, torch_mps, torch_cpu)
        valid_names = ["torch_cuda", "torch_mps", "torch_cpu", "jax", "numpy"]
        assert backend.name in valid_names

    def test_create_backend_none_same_as_auto(self):
        """Test that None behaves same as 'auto'."""
        backend1 = create_backend(None)
        backend2 = create_backend("auto")
        # Both should select same backend (highest priority available)
        assert backend1.name == backend2.name

    def test_create_backend_invalid_name_raises_error(self):
        """Test that invalid backend name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            create_backend("nonexistent_backend")

    def test_create_backend_torch_when_unavailable(self, monkeypatch, clean_backend_registry):
        """A machine without PyTorch gets TorchBackend's own diagnostic.

        `torch_backend.py` imports fine without torch -- it degrades to `TORCH_AVAILABLE =
        False` -- so the error a real user sees comes from the constructor
        (`torch_backend.py:109`). Clearing the flag is what reproduces that path.

        ~~`backends/__init__.py` registers "torch" into `_BACKENDS` whether or not torch
        exists, so the `if backend_name not in _BACKENDS` branch is unreachable for torch~~
        [CORRECTED 2026-08-14] -- it is reachable now: #1930 stopped registering torch and jax
        eagerly, so `create_backend("torch")` imports and registers on demand. This test
        therefore REGISTERS torch as a side effect, which is why it takes
        `clean_backend_registry`. Without it, `test_optional_backends_registered_if_available`
        reads the residue and passes for the wrong reason -- serially and under a lucky xdist
        schedule -- while failing 3/3 on a directory-scoped `-n auto` run.
        """
        import mfgarchon.backends.torch_backend as torch_backend

        monkeypatch.setattr(torch_backend, "TORCH_AVAILABLE", False)

        with pytest.raises(ImportError, match="PyTorch is required for TorchBackend"):
            create_backend("torch")

    def test_create_backend_jax_when_unavailable(self, monkeypatch, clean_backend_registry):
        """A machine without JAX gets JAXBackend's own diagnostic.

        Same shape as the torch case: `jax_backend.py` imports without JAX, so the constructor
        raises (`jax_backend.py:66`). ~~so "jax" is always registered~~ [CORRECTED 2026-08-14]
        -- jax is registered on demand since #1930, so this test registers it and needs the
        cleanup fixture.
        """
        import mfgarchon.backends.jax_backend as jax_backend

        monkeypatch.setattr(jax_backend, "JAX_AVAILABLE", False)

        with pytest.raises(ImportError, match="JAX backend requested but JAX is not installed"):
            create_backend("jax")

    def test_kwargs_passed_to_backend(self):
        """Test that kwargs are passed to backend constructor."""
        backend = create_backend("numpy", precision="float32", custom_arg=42)
        assert backend.precision == "float32"
        assert backend.config.get("custom_arg") == 42


class TestGetBackendInfo:
    """Test backend information retrieval."""

    def test_returns_dict(self):
        """Test that function returns dictionary."""
        info = get_backend_info()
        assert isinstance(info, dict)

    def test_contains_available_backends(self):
        """Test that info contains available backends."""
        info = get_backend_info()
        assert "available_backends" in info
        assert isinstance(info["available_backends"], dict)

    def test_contains_default_backend(self):
        """Test that info contains default backend."""
        info = get_backend_info()
        assert "default_backend" in info
        assert info["default_backend"] == "numpy"

    def test_contains_registered_backends(self):
        """Test that info contains registered backends."""
        info = get_backend_info()
        assert "registered_backends" in info
        assert isinstance(info["registered_backends"], list)
        assert "numpy" in info["registered_backends"]

    def test_torch_info_when_available(self):
        """Test torch-specific info when torch is available."""
        available = get_available_backends()
        info = get_backend_info()

        if available["torch"]:
            assert "torch_info" in info
            assert "version" in info["torch_info"]
            assert "cuda_available" in info["torch_info"]
            assert "mps_available" in info["torch_info"]

    def test_torch_cuda_info_when_available(self):
        """Test CUDA-specific info when CUDA is available."""
        available = get_available_backends()
        info = get_backend_info()

        if available["torch"] and available["torch_cuda"]:
            assert "torch_info" in info
            assert "cuda_version" in info["torch_info"]
            assert "cuda_device_count" in info["torch_info"]
            assert "cuda_devices" in info["torch_info"]

    def test_jax_info_when_available(self):
        """Test jax-specific info when jax is available."""
        available = get_available_backends()
        info = get_backend_info()

        if available["jax"]:
            assert "jax_info" in info
            assert "version" in info["jax_info"]
            assert "devices" in info["jax_info"]
            assert "default_device" in info["jax_info"]
            assert "has_gpu" in info["jax_info"]


class TestEnsureNumpyBackend:
    """Test NumPy backend availability guarantee."""

    def test_ensure_numpy_backend_succeeds(self):
        """Test that ensure_numpy_backend completes without error."""
        from mfgarchon.backends import ensure_numpy_backend

        # Should not raise any errors
        ensure_numpy_backend()

        # NumPy should be registered
        import mfgarchon.backends as backends_module

        assert "numpy" in backends_module._BACKENDS

    def test_numpy_available_after_import(self):
        """Test that numpy backend is always available after module import."""
        import mfgarchon.backends as backends_module

        assert "numpy" in backends_module._BACKENDS
        backend = create_backend("numpy")
        assert backend.name == "numpy"


class TestModuleExports:
    """Test module __all__ exports."""

    def test_all_exports_callable(self):
        """Test that all exported items are callable."""
        import mfgarchon.backends as backends_module

        for name in backends_module.__all__:
            item = getattr(backends_module, name)
            assert callable(item), f"{name} should be callable"


class TestBackendInitialization:
    """Test backend module initialization."""

    def test_numpy_auto_initialized(self):
        """Test that NumPy backend is auto-initialized on import."""
        import mfgarchon.backends as backends_module

        assert "numpy" in backends_module._BACKENDS

    def test_optional_backends_are_not_registered_until_asked_for(self):
        """[SUPERSEDED 2026-08-14] Was `test_optional_backends_registered_if_available`, which
        asserted the opposite: that touching the package registers torch and jax when installed.

        #1930 stopped that, because registering them imported them -- ~0.8s of a ~4s
        `import mfgarchon` for anyone who never asked for a backend. `create_backend` already
        carried the on-demand path; eager registration was what made it unreachable.

        **In a subprocess, because `_BACKENDS` is a module global and this is an assertion about
        its state at import.** Any test anywhere that asks for a backend registers one, so an
        in-process check reads whatever the session happened to do first. The first attempt at
        this test used `clean_backend_registry` and still failed in the full suite -- the fixture
        restores the registry after the polluter, but not before this test, and under `-n auto`
        the order is not fixed. Order-independence is not something a fixture can retrofit onto
        a global; a fresh interpreter is the instrument.
        """
        import json
        import os
        import subprocess

        repo = Path(__file__).resolve().parents[3]
        # `cwd=repo` does not pin the tree. For `python -c`, `sys.path[0]` is the current directory
        # and precedes PYTHONPATH -- so cwd wins, until `-P` / `PYTHONSAFEPATH=1` strips it, which
        # is exactly what `scripts/local_ci.sh:311` sets and what `subprocess.run` inherits. Under
        # the gate the cwd entry is gone, nothing replaces it, and this probe measures the EDITABLE
        # INSTALL instead of the tree under test -- silently identical on the canonical checkout,
        # wrong in the worktree this repo mandates for pre-merge review. So: put the tree back on
        # the path, and report `mfgarchon.__file__` so the assertion below can check it rather than
        # trust the plumbing. (This repo's own #1906 review: an import-origin check is only evidence
        # if the wrong answer is reachable.)
        env = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join(x for x in (str(repo), os.environ.get("PYTHONPATH", "")) if x),
        }
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import json\n"
                "import mfgarchon.backends as b\n"
                "at_import = sorted(b._BACKENDS)\n"
                "asked = None\n"
                "if b.get_available_backends()['jax']:\n"
                "    b.create_backend('jax')\n"
                "    asked = sorted(b._BACKENDS)\n"
                "print(json.dumps({'at_import': at_import, 'after_asking': asked, 'tree': b.__file__}))\n",
            ],
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert proc.returncode == 0, f"the probe never ran:\n{proc.stderr[-1500:]}"
        line = [x for x in proc.stdout.splitlines() if x.startswith("{")]
        assert line, f"no verdict:\n{proc.stdout[-600:]}\n{proc.stderr[-600:]}"
        result = json.loads(line[-1])

        # A path check, not a string prefix: on macOS TMPDIR is a symlink, so a worktree's logical
        # and physical spellings differ and `startswith` compares False on the correct tree.
        assert Path(result["tree"]).resolve().is_relative_to(repo.resolve()), (
            f"the probe imported {result['tree']}, not the tree under test at {repo}. Its verdict "
            f"says nothing about this checkout."
        )

        assert result["at_import"] == ["numpy"], (
            f"at import the registry holds {result['at_import']}, expected only numpy. Registering "
            f"torch or jax imports it, which is what #1930 removed -- see "
            f"tests/unit/test_optional_backends_are_not_imported_eagerly.py"
        )
        if result["after_asking"] is not None:
            assert "jax" in result["after_asking"], "create_backend('jax') did not register it"


class TestBackendCreationEdgeCases:
    """Test edge cases in backend creation."""

    def test_create_backend_empty_kwargs(self):
        """Test backend creation with empty kwargs."""
        backend = create_backend("numpy")
        assert backend.config == {}

    def test_create_backend_multiple_kwargs(self):
        """Test backend creation with multiple kwargs."""
        backend = create_backend("numpy", arg1="value1", arg2=42, arg3=True)
        assert backend.config["arg1"] == "value1"
        assert backend.config["arg2"] == 42
        assert backend.config["arg3"] is True
