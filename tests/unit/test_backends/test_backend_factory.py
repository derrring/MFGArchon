"""
Unit tests for Backend Factory.

Tests backend registration, discovery, creation, and auto-selection logic.
"""

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

    def test_numba_key_present(self):
        """Test that numba key is present."""
        available = get_available_backends()
        assert "numba" in available

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

    def test_create_backend_torch_when_unavailable(self, monkeypatch):
        """Test error when requesting torch but it's not available."""
        available = get_available_backends()
        if available["torch"]:
            pytest.skip("PyTorch is available, cannot test unavailable scenario")

        # Mock torch as unavailable by removing from registry
        import mfgarchon.backends as backends_module

        original = backends_module._BACKENDS.copy()
        backends_module._BACKENDS.pop("torch", None)

        try:
            with pytest.raises(ImportError, match="PyTorch is required for TorchBackend"):
                create_backend("torch")
        finally:
            backends_module._BACKENDS = original

    def test_create_backend_jax_when_unavailable(self, monkeypatch):
        """Test error when requesting jax but it's not available."""
        available = get_available_backends()
        if available["jax"]:
            pytest.skip("JAX is available, cannot test unavailable scenario")

        import mfgarchon.backends as backends_module

        original = backends_module._BACKENDS.copy()
        backends_module._BACKENDS.pop("jax", None)

        try:
            with pytest.raises(ImportError, match="JAX backend requested"):
                create_backend("jax")
        finally:
            backends_module._BACKENDS = original

    def test_create_backend_numba_when_unavailable(self, monkeypatch):
        """Test error when requesting numba but it's not available."""
        available = get_available_backends()
        if available["numba"]:
            pytest.skip("Numba is available, cannot test unavailable scenario")

        import mfgarchon.backends as backends_module

        original = backends_module._BACKENDS.copy()
        backends_module._BACKENDS.pop("numba", None)

        try:
            with pytest.raises(ImportError, match="Numba backend requested"):
                create_backend("numba")
        finally:
            backends_module._BACKENDS = original

    def test_kwargs_passed_to_backend(self):
        """Test that kwargs are passed to backend constructor."""
        backend = create_backend("numpy", precision="float32", custom_arg=42)
        assert backend.precision == "float32"
        assert backend.config.get("custom_arg") == 42


class TestAutoBackendSelection:
    """Test automatic backend selection logic."""

    def test_auto_selection_priority(self, monkeypatch):
        """Test that auto-selection follows priority: torch > jax > numpy."""
        available = get_available_backends()

        backend = create_backend("auto")

        # Verify selection follows priority
        # Torch backend name includes device type
        if available["torch"]:
            assert backend.name.startswith("torch_")
        elif available["jax"]:
            assert backend.name == "jax"
        else:
            assert backend.name == "numpy"

    def test_auto_torch_cuda_device_selection(self, capfd):
        """Test that auto-selection sets CUDA device when available."""
        available = get_available_backends()

        if available["torch"] and available["torch_cuda"]:
            backend = create_backend("auto")
            assert backend.name == "torch_cuda"
            # Verify CUDA device reported in initialization output
            captured = capfd.readouterr()
            assert "cuda" in captured.out.lower()

    def test_auto_torch_mps_device_selection(self, capfd):
        """Test that auto-selection sets MPS device when CUDA unavailable."""
        available = get_available_backends()

        if available["torch"] and available["torch_mps"] and not available["torch_cuda"]:
            backend = create_backend("auto")
            assert backend.name == "torch_mps"
            # Verify MPS device reported in initialization output
            captured = capfd.readouterr()
            assert "mps" in captured.out.lower()

    def test_auto_jax_gpu_device_selection(self, capfd):
        """Test JAX GPU auto-selection when torch unavailable."""
        available = get_available_backends()

        if not available["torch"] and available["jax"] and available["jax_gpu"]:
            backend = create_backend("auto")
            assert backend.name == "jax"
            # Verify GPU device reported in initialization output
            captured = capfd.readouterr()
            assert "gpu" in captured.out.lower()


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

    def test_all_exports_defined(self):
        """Test that __all__ is defined and contains expected functions."""
        import mfgarchon.backends as backends_module

        assert hasattr(backends_module, "__all__")
        expected = {
            "create_backend",
            "get_available_backends",
            "get_backend_info",
            "register_backend",
            "warn_if_jax_scheme_downgraded",  # Issue #1072 interim warning helper
        }
        assert set(backends_module.__all__) == expected

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

    def test_optional_backends_registered_if_available(self):
        """Test that optional backends are registered when available."""
        import mfgarchon.backends as backends_module

        available = get_available_backends()

        # Torch should be registered if available
        if available["torch"]:
            assert "torch" in backends_module._BACKENDS

        # JAX should be registered if available
        if available["jax"]:
            assert "jax" in backends_module._BACKENDS


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

    def test_auto_selection_logging(self, capfd):
        """Test that auto-selection produces log messages."""
        backend = create_backend("auto")
        captured = capfd.readouterr()
        # Should have at least one log message about backend selection
        assert len(captured.out) > 0
        # Log message should mention the backend type (torch, jax, or numpy)
        log_text = captured.out.lower()
        # Extract backend type from name (e.g., "torch_mps" -> "torch")
        backend_type = backend.name.split("_")[0]
        assert backend_type in log_text
