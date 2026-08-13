#!/usr/bin/env python3
"""
Unit tests for mfgarchon/utils/parameter_migration.py

Tests comprehensive parameter migration system including:
- ParameterMapping dataclass
- MigrationStats dataclass
- ParameterMigrator class
- @migrate_parameters decorator
- Convenience functions (migrate_kwargs, check_deprecated_usage, get_parameter_migration_guide)
"""

import warnings

import pytest

from mfgarchon.utils.parameter_migration import (
    MigrationStats,
    ParameterMapping,
    ParameterMigrator,
    check_deprecated_usage,
    get_parameter_migration_guide,
    global_parameter_migrator,
    migrate_kwargs,
    migrate_parameters,
)

# ===================================================================
# Test ParameterMapping Dataclass
# ===================================================================


@pytest.mark.unit
def test_parameter_mapping_basic():
    """Test ParameterMapping basic initialization."""
    mapping = ParameterMapping(
        old_name="oldParam",
        new_name="newParam",
        deprecation_version="1.0.0",
        removal_version="2.0.0",
    )
    assert mapping.old_name == "oldParam"
    assert mapping.new_name == "newParam"
    assert mapping.deprecation_version == "1.0.0"
    assert mapping.removal_version == "2.0.0"
    assert mapping.transformation is None


@pytest.mark.unit
def test_parameter_mapping_auto_description():
    """Test ParameterMapping auto-generates description."""
    mapping = ParameterMapping(
        old_name="oldParam",
        new_name="newParam",
        deprecation_version="1.0.0",
        removal_version="2.0.0",
    )
    assert mapping.description == "Renamed 'oldParam' to 'newParam'"


@pytest.mark.unit
def test_parameter_mapping_custom_description():
    """Test ParameterMapping with custom description."""
    mapping = ParameterMapping(
        old_name="oldParam",
        new_name="newParam",
        deprecation_version="1.0.0",
        removal_version="2.0.0",
        description="Custom description",
    )
    assert mapping.description == "Custom description"


@pytest.mark.unit
def test_parameter_mapping_with_transformation():
    """Test ParameterMapping with transformation function."""

    def transform(x):
        return x * 2

    mapping = ParameterMapping(
        old_name="oldParam",
        new_name="newParam",
        deprecation_version="1.0.0",
        removal_version="2.0.0",
        transformation=transform,
    )
    assert mapping.transformation is not None
    assert mapping.transformation(5) == 10


# ===================================================================
# Test MigrationStats Dataclass
# ===================================================================


@pytest.mark.unit
def test_migration_stats_initialization():
    """Test MigrationStats initialization with defaults."""
    stats = MigrationStats()
    assert stats.total_parameters == 0
    assert stats.migrated_parameters == 0
    assert stats.warnings_issued == 0
    assert stats.errors_encountered == 0
    assert isinstance(stats.migration_log, list)
    assert len(stats.migration_log) == 0


@pytest.mark.unit
def test_migration_stats_custom_values():
    """Test MigrationStats with custom values."""
    stats = MigrationStats(
        total_parameters=10,
        migrated_parameters=5,
        warnings_issued=3,
        errors_encountered=1,
    )
    assert stats.total_parameters == 10
    assert stats.migrated_parameters == 5
    assert stats.warnings_issued == 3
    assert stats.errors_encountered == 1


@pytest.mark.unit
def test_migration_stats_log():
    """Test MigrationStats log management."""
    stats = MigrationStats()
    stats.migration_log.append("Entry 1")
    stats.migration_log.append("Entry 2")
    assert len(stats.migration_log) == 2
    assert "Entry 1" in stats.migration_log


# ===================================================================
# Test ParameterMigrator Class - Initialization
# ===================================================================


@pytest.mark.unit
def test_parameter_migrator_initialization():
    """Test ParameterMigrator initialization."""
    migrator = ParameterMigrator()
    assert isinstance(migrator.mappings, list)
    assert isinstance(migrator.migration_stats, MigrationStats)
    # Should have standard mappings registered
    assert len(migrator.mappings) > 0


@pytest.mark.unit
def test_parameter_migrator_has_standard_mappings():
    """Test ParameterMigrator registers standard mappings."""
    migrator = ParameterMigrator()

    # Check for known standard mappings
    old_names = [m.old_name for m in migrator.mappings]
    assert "NiterNewton" in old_names
    assert "max_iterations" in old_names
    assert "coefCT" in old_names  # Old name for coupling_coefficient


# ===================================================================
# Test ParameterMigrator Class - add_mapping
# ===================================================================


@pytest.mark.unit
def test_add_mapping_basic():
    """Test adding a custom mapping."""
    migrator = ParameterMigrator()
    initial_count = len(migrator.mappings)

    migrator.add_mapping(
        old_name="testOld",
        new_name="testNew",
        deprecation_version="1.0.0",
        removal_version="2.0.0",
    )

    assert len(migrator.mappings) == initial_count + 1
    last_mapping = migrator.mappings[-1]
    assert last_mapping.old_name == "testOld"
    assert last_mapping.new_name == "testNew"


@pytest.mark.unit
def test_add_mapping_with_transformation():
    """Test adding mapping with transformation function."""
    migrator = ParameterMigrator()

    def transform(x):
        return str(x).upper()

    migrator.add_mapping(
        old_name="testOld",
        new_name="testNew",
        deprecation_version="1.0.0",
        removal_version="2.0.0",
        transformation=transform,
    )

    last_mapping = migrator.mappings[-1]
    assert last_mapping.transformation is not None
    assert last_mapping.transformation("hello") == "HELLO"


# ===================================================================
# Test ParameterMigrator Class - migrate_parameters
# ===================================================================


@pytest.mark.unit
def test_migrate_parameters_empty_dict():
    """Test migrating empty parameter dict."""
    migrator = ParameterMigrator()
    result = migrator.migrate_parameters({})
    assert result == {}


@pytest.mark.unit
def test_migrate_parameters_no_deprecated():
    """Test migrating parameters with no deprecated names."""
    migrator = ParameterMigrator()
    kwargs = {"modern_param": 42, "another_param": "value"}

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Would fail if warnings issued
        result = migrator.migrate_parameters(kwargs)

    assert result == kwargs


@pytest.mark.unit
def test_migrate_parameters_basic_migration():
    """Test basic parameter migration."""
    migrator = ParameterMigrator()

    # Use known standard mapping: NiterNewton → max_newton_iterations
    kwargs = {"NiterNewton": 100}

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = migrator.migrate_parameters(kwargs)

    # Should have migrated
    assert "max_newton_iterations" in result
    assert result["max_newton_iterations"] == 100
    assert "NiterNewton" not in result

    # Should have issued deprecation warning
    assert len(w) == 1
    assert issubclass(w[0].category, DeprecationWarning)
    assert "NiterNewton" in str(w[0].message)


@pytest.mark.unit
def test_migrate_parameters_with_transformation():
    """Test parameter migration with value transformation."""
    migrator = ParameterMigrator()

    # Use known mapping with transformation: returnExtraInfo → return_structured
    kwargs = {"returnExtraInfo": 1}  # Should convert to bool

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = migrator.migrate_parameters(kwargs)

    assert "return_structured" in result
    assert isinstance(result["return_structured"], bool)
    assert result["return_structured"] is True


@pytest.mark.unit
def test_migrate_parameters_conflict_both_specified():
    """Test migration when both old and new parameters specified."""
    migrator = ParameterMigrator()

    kwargs = {"NiterNewton": 100, "max_newton_iterations": 200}

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = migrator.migrate_parameters(kwargs)

    # Should keep new parameter value
    assert result["max_newton_iterations"] == 200
    assert "NiterNewton" not in result

    # Should have issued warnings (deprecation + conflict)
    assert len(w) >= 2
    warning_messages = [str(warn.message) for warn in w]
    assert any("Both" in msg for msg in warning_messages)


@pytest.mark.unit
def test_migrate_parameters_multiple_mappings():
    """Test migrating multiple deprecated parameters at once."""
    migrator = ParameterMigrator()

    kwargs = {
        "NiterNewton": 100,
        "max_iterations": 50,
        "modern_param": "value",
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = migrator.migrate_parameters(kwargs)

    assert "max_newton_iterations" in result
    assert "max_picard_iterations" in result
    assert "modern_param" in result
    assert "NiterNewton" not in result
    assert "max_iterations" not in result


# ===================================================================
# Test ParameterMigrator Class - Utility Methods
# ===================================================================


@pytest.mark.unit
def test_get_deprecated_parameters():
    """Test retrieving all deprecated parameter names."""
    migrator = ParameterMigrator()
    deprecated = migrator.get_deprecated_parameters()

    assert isinstance(deprecated, set)
    assert "NiterNewton" in deprecated
    assert "max_iterations" in deprecated
    assert len(deprecated) > 0


@pytest.mark.unit
def test_get_modern_parameters():
    """Test retrieving all modern parameter names."""
    migrator = ParameterMigrator()
    modern = migrator.get_modern_parameters()

    assert isinstance(modern, set)
    assert "max_newton_iterations" in modern
    assert "max_picard_iterations" in modern
    assert len(modern) > 0


@pytest.mark.unit
def test_check_parameters_no_deprecated():
    """Test check_parameters with no deprecated names."""
    migrator = ParameterMigrator()
    kwargs = {"modern_param": 42}

    deprecated, modern = migrator.check_parameters(kwargs)

    assert len(deprecated) == 0
    assert len(modern) == 0


@pytest.mark.unit
def test_check_parameters_with_deprecated():
    """Test check_parameters finds deprecated names."""
    migrator = ParameterMigrator()
    kwargs = {"NiterNewton": 100, "modern_param": 42}

    deprecated, modern = migrator.check_parameters(kwargs)

    assert "NiterNewton" in deprecated
    assert "max_newton_iterations" in modern
    assert len(deprecated) == len(modern)


@pytest.mark.unit
def test_get_migration_report():
    """Test migration report generation."""
    migrator = ParameterMigrator()

    # Perform some migrations to populate stats
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        migrator.migrate_parameters({"NiterNewton": 100})

    report = migrator.get_migration_report()

    assert isinstance(report, str)
    assert "Parameter Migration Report" in report
    assert "Total Parameters" in report
    assert "Migrated Parameters" in report


# ===================================================================
# Test @migrate_parameters Decorator
# ===================================================================


@pytest.mark.unit
def test_migrate_parameters_decorator_basic():
    """Test @migrate_parameters decorator basic functionality."""
    migrator = ParameterMigrator()

    @migrate_parameters(migrator)
    def test_function(**kwargs):
        return kwargs

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = test_function(NiterNewton=100)

    assert "max_newton_iterations" in result
    assert result["max_newton_iterations"] == 100
    assert "NiterNewton" not in result


@pytest.mark.unit
def test_migrate_parameters_decorator_with_args():
    """Test decorator with positional arguments."""
    migrator = ParameterMigrator()

    @migrate_parameters(migrator)
    def test_function(arg1, arg2, **kwargs):
        return arg1 + arg2, kwargs

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result_sum, result_kwargs = test_function(10, 20, NiterNewton=100)

    assert result_sum == 30
    assert "max_newton_iterations" in result_kwargs


@pytest.mark.unit
def test_migrate_parameters_decorator_uses_global():
    """Test decorator uses global migrator by default."""

    @migrate_parameters()
    def test_function(**kwargs):
        return kwargs

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = test_function(NiterNewton=100)

    # Should use global_parameter_migrator
    assert "max_newton_iterations" in result


# ===================================================================
# Test Convenience Functions
# ===================================================================


@pytest.mark.unit
def test_migrate_kwargs_convenience():
    """Test migrate_kwargs convenience function."""
    kwargs = {"NiterNewton": 100}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = migrate_kwargs(kwargs)

    assert "max_newton_iterations" in result
    assert result["max_newton_iterations"] == 100


@pytest.mark.unit
def test_migrate_kwargs_with_calling_function():
    """Test migrate_kwargs with calling_function parameter."""
    kwargs = {"NiterNewton": 100}

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        migrate_kwargs(kwargs, calling_function="my_function")

    # Warning should mention calling function
    assert len(w) > 0
    assert "my_function" in str(w[0].message)


@pytest.mark.unit
def test_check_deprecated_usage_no_output(capsys):
    """Test check_deprecated_usage with modern parameters."""
    kwargs = {"modern_param": 42}

    check_deprecated_usage(kwargs)

    captured = capsys.readouterr()
    assert "WARNING" not in captured.out


@pytest.mark.unit
def test_check_deprecated_usage_with_deprecated(capsys):
    """Test check_deprecated_usage prints warnings for deprecated."""
    kwargs = {"NiterNewton": 100}

    check_deprecated_usage(kwargs)

    captured = capsys.readouterr()
    assert "WARNING: Deprecated parameters detected" in captured.out
    assert "NiterNewton" in captured.out
    assert "max_newton_iterations" in captured.out


@pytest.mark.unit
def test_get_parameter_migration_guide():
    """Test parameter migration guide generation."""
    guide = get_parameter_migration_guide()

    assert isinstance(guide, str)
    assert "Parameter Migration Guide" in guide
    assert "NiterNewton" in guide
    assert "max_newton_iterations" in guide
    assert "Migration Strategy" in guide


# ===================================================================
# Test Global Migrator
# ===================================================================


@pytest.mark.unit
def test_global_parameter_migrator_has_mappings():
    """Test global migrator has standard mappings."""
    assert len(global_parameter_migrator.mappings) > 0
    old_names = [m.old_name for m in global_parameter_migrator.mappings]
    assert "NiterNewton" in old_names


# ===================================================================
# Test Edge Cases and Error Handling
# ===================================================================


@pytest.mark.unit
def test_migration_with_transformation_error():
    """Test migration handles transformation errors gracefully."""
    migrator = ParameterMigrator()

    def bad_transform(x):
        raise ValueError("Transformation failed")

    migrator.add_mapping(
        old_name="badParam",
        new_name="goodParam",
        deprecation_version="1.0.0",
        removal_version="2.0.0",
        transformation=bad_transform,
    )

    kwargs = {"badParam": "value"}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = migrator.migrate_parameters(kwargs)

    # Should still migrate despite transformation error
    assert "goodParam" in result
    # Should use original value when transformation fails
    assert result["goodParam"] == "value"
    # Should record error
    assert migrator.migration_stats.errors_encountered > 0


@pytest.mark.unit
def test_migration_preserves_original_dict():
    """Test migration doesn't modify original dict."""
    migrator = ParameterMigrator()
    original = {"NiterNewton": 100, "other": "value"}
    original_copy = original.copy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        migrator.migrate_parameters(original)

    # Original dict should be unchanged
    assert original == original_copy


@pytest.mark.unit
def test_migration_report_truncates_log():
    """Test migration report truncates long logs."""
    migrator = ParameterMigrator()

    # Add many log entries
    for i in range(20):
        migrator.migration_stats.migration_log.append(f"Entry {i}")

    report = migrator.get_migration_report()

    # Should only show last 10 entries
    assert "Entry 19" in report
    assert "Entry 0" not in report
    assert "and 10 more entries" in report


# ===================================================================
# Test Module Exports
# ===================================================================
