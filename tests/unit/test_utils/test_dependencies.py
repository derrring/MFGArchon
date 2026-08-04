"""
Unit tests for Optional Dependency Checking System.

Tests the dependency utilities that provide helpful error messages for
missing optional dependencies.
"""

import pytest

# ============================================================================
# Test: is_available() Function
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_is_available_core_dependency():
    """Test is_available() returns True for core dependencies."""
    from mfgarchon.utils.dependencies import is_available

    # NumPy is a core dependency, should always be available
    assert is_available("numpy") is True
    assert is_available("scipy") is True
    assert is_available("matplotlib") is True


@pytest.mark.unit
@pytest.mark.fast
def test_is_available_nonexistent_package():
    """Test is_available() returns False for nonexistent packages."""
    from mfgarchon.utils.dependencies import is_available

    # Package that definitely doesn't exist
    assert is_available("this_package_does_not_exist_xyz123") is False


@pytest.mark.unit
@pytest.mark.fast
def test_is_available_optional_dependency():
    """Test is_available() correctly detects optional dependencies."""
    from mfgarchon.utils.dependencies import is_available

    # Test optional dependencies - results depend on environment
    torch_available = is_available("torch")
    jax_available = is_available("jax")
    plotly_available = is_available("plotly")

    # Should return boolean values
    assert isinstance(torch_available, bool)
    assert isinstance(jax_available, bool)
    assert isinstance(plotly_available, bool)


# ============================================================================
# Test: check_dependency() Function
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_check_dependency_available():
    """Test check_dependency() returns True for available packages."""
    from mfgarchon.utils.dependencies import check_dependency

    # Core dependency should pass
    assert check_dependency("numpy") is True


@pytest.mark.unit
@pytest.mark.fast
def test_check_dependency_missing_known_package():
    """Test check_dependency() raises helpful error for known packages."""
    from mfgarchon.utils.dependencies import check_dependency

    # Try to check a known optional dependency that might not be installed
    # We'll use a package unlikely to be installed
    try:
        check_dependency("this_package_does_not_exist_xyz123")
        pytest.fail("Expected ImportError for missing package")
    except ImportError as e:
        error_msg = str(e)
        # Should contain helpful message
        assert "required" in error_msg.lower()
        assert "install" in error_msg.lower()


@pytest.mark.unit
@pytest.mark.fast
def test_check_dependency_with_purpose_description():
    """Test check_dependency() includes purpose in error message."""
    from mfgarchon.utils.dependencies import check_dependency

    try:
        check_dependency("nonexistent_package_xyz", purpose="test feature")
        pytest.fail("Expected ImportError")
    except ImportError as e:
        error_msg = str(e)
        # Should include purpose description
        assert "test feature" in error_msg


@pytest.mark.unit
@pytest.mark.fast
def test_check_dependency_raise_on_missing_false():
    """Test check_dependency() returns False when raise_on_missing=False."""
    from mfgarchon.utils.dependencies import check_dependency

    # Should return False instead of raising
    result = check_dependency("nonexistent_package_xyz", raise_on_missing=False)
    assert result is False

    # Should return True for available packages
    result = check_dependency("numpy", raise_on_missing=False)
    assert result is True


@pytest.mark.unit
@pytest.mark.fast
def test_check_dependency_custom_install_command():
    """Test check_dependency() uses custom install command."""
    from mfgarchon.utils.dependencies import check_dependency

    try:
        check_dependency("nonexistent_xyz", install_command="conda install nonexistent_xyz")
        pytest.fail("Expected ImportError")
    except ImportError as e:
        error_msg = str(e)
        # Should include custom install command
        assert "conda install nonexistent_xyz" in error_msg


@pytest.mark.unit
@pytest.mark.fast
def test_check_dependency_known_package_error_format():
    """Test error message format for known optional dependencies."""
    from mfgarchon.utils.dependencies import DEPENDENCY_MAP, check_dependency

    # Pick a known optional package (not core) from DEPENDENCY_MAP
    optional_packages = [pkg for pkg, info in DEPENDENCY_MAP.items() if not info.get("required", False)]
    test_package = optional_packages[0] if optional_packages else "cupy"

    # If package is not available, check error message format
    from mfgarchon.utils.dependencies import is_available

    if not is_available(test_package):
        try:
            check_dependency(test_package)
            pytest.fail(f"Expected ImportError for {test_package}")
        except ImportError as e:
            error_msg = str(e)
            # Should contain structured error message
            assert "Install options:" in error_msg
            assert "pip install" in error_msg


# ============================================================================
# Test: require_dependencies() Decorator
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_require_dependencies_decorator_success():
    """Test require_dependencies decorator allows execution when deps available."""
    from mfgarchon.utils.dependencies import require_dependencies

    @require_dependencies("numpy", "scipy")
    def test_function():
        return "success"

    # Should execute successfully
    result = test_function()
    assert result == "success"


@pytest.mark.unit
@pytest.mark.fast
def test_require_dependencies_decorator_failure():
    """Test require_dependencies decorator raises error for missing deps."""
    from mfgarchon.utils.dependencies import require_dependencies

    @require_dependencies("nonexistent_package_xyz")
    def test_function():
        return "should not reach here"

    # Should raise ImportError before executing function
    with pytest.raises(ImportError) as exc_info:
        test_function()

    assert "nonexistent_package_xyz" in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.fast
def test_require_dependencies_with_feature():
    """Test require_dependencies decorator includes feature in error."""
    from mfgarchon.utils.dependencies import require_dependencies

    @require_dependencies("nonexistent_xyz", feature="test functionality")
    def test_function():
        return "should not reach here"

    with pytest.raises(ImportError) as exc_info:
        test_function()

    error_msg = str(exc_info.value)
    assert "test functionality" in error_msg


# ============================================================================
# Test: get_available_features() Function
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_get_available_features_structure():
    """Test get_available_features returns correct structure."""
    from mfgarchon.utils.dependencies import get_available_features

    features = get_available_features()

    # Should return dictionary
    assert isinstance(features, dict)

    # Should contain expected keys
    expected_features = ["pytorch", "jax", "gymnasium", "plotly", "networkx"]
    for feature in expected_features:
        assert feature in features

    # All values should be boolean
    for feature, available in features.items():
        assert isinstance(available, bool), f"{feature} availability should be boolean"


@pytest.mark.unit
@pytest.mark.fast
def test_get_available_features_consistency():
    """Test get_available_features is consistent with is_available."""
    from mfgarchon.utils.dependencies import get_available_features, is_available

    features = get_available_features()

    # PyTorch feature should match torch package
    assert features["pytorch"] == is_available("torch")

    # JAX feature should match jax package
    assert features["jax"] == is_available("jax")


# ============================================================================
# Test: Module-Level Availability Flags
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_module_level_flags_consistency():
    """Test module-level flags match is_available results."""
    from mfgarchon.utils import dependencies
    from mfgarchon.utils.dependencies import is_available

    assert is_available("torch") == dependencies.TORCH_AVAILABLE
    assert is_available("jax") == dependencies.JAX_AVAILABLE
    assert is_available("gymnasium") == dependencies.GYMNASIUM_AVAILABLE
    assert is_available("plotly") == dependencies.PLOTLY_AVAILABLE
    assert is_available("networkx") == dependencies.NETWORKX_AVAILABLE


# ============================================================================
# Test: show_optional_features() Function
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_show_optional_features_executes(capsys):
    """Test show_optional_features executes without error."""
    from mfgarchon.utils.dependencies import show_optional_features

    # Should execute without raising exception
    show_optional_features()

    # Capture output
    captured = capsys.readouterr()

    # Should produce some output
    assert len(captured.out) > 0
    assert "MFGarchon Optional Features" in captured.out


@pytest.mark.unit
@pytest.mark.fast
def test_show_optional_features_format(capsys):
    """Test show_optional_features output format."""
    from mfgarchon.utils.dependencies import show_optional_features

    show_optional_features()
    captured = capsys.readouterr()

    # Should contain section headers
    assert "MFGarchon Optional Features" in captured.out
    assert "Installation Options:" in captured.out

    # Should contain category headers
    assert "Core (always available):" in captured.out or "Neural Methods:" in captured.out

    # Should contain package names
    assert "numpy" in captured.out
    assert "torch" in captured.out or "jax" in captured.out


@pytest.mark.unit
@pytest.mark.fast
def test_get_package_version():
    """Test get_package_version returns correct format."""
    from mfgarchon.utils.dependencies import get_package_version

    # Core packages should have versions
    numpy_version = get_package_version("numpy")
    assert numpy_version is not None
    assert isinstance(numpy_version, str)
    assert len(numpy_version) > 0

    # Nonexistent package should return None
    fake_version = get_package_version("nonexistent_package_xyz")
    assert fake_version is None


# ============================================================================
# Test: Package-Level Function
# ============================================================================


# ============================================================================
# Test: DEPENDENCY_MAP Structure
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_dependency_map_structure():
    """Test DEPENDENCY_MAP has correct structure."""
    from mfgarchon.utils.dependencies import DEPENDENCY_MAP

    # Should be a dictionary
    assert isinstance(DEPENDENCY_MAP, dict)

    # Should contain key packages (both core and optional)
    expected_packages = ["numpy", "scipy", "matplotlib", "torch", "jax", "gymnasium", "plotly", "networkx"]
    for package in expected_packages:
        assert package in DEPENDENCY_MAP

    # Each entry should have required fields
    for _package, info in DEPENDENCY_MAP.items():
        assert "install_group" in info
        assert "install_cmd" in info
        assert "alternative" in info
        assert "description" in info
        assert "required" in info

        # description should be a string
        assert isinstance(info["description"], str)
        assert len(info["description"]) > 0

        # required should be a boolean
        assert isinstance(info["required"], bool)


@pytest.mark.unit
@pytest.mark.fast
def test_dependency_map_install_commands():
    """Test DEPENDENCY_MAP contains valid install commands."""
    from mfgarchon.utils.dependencies import DEPENDENCY_MAP

    for package, info in DEPENDENCY_MAP.items():
        # Install command should reference the package
        assert "pip install" in info["install_cmd"]

        # Alternative should also be pip install
        assert "pip install" in info["alternative"]
        assert package in info["alternative"]
