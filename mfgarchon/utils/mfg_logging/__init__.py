"""
Logging utilities for MFGarchon.

This module provides comprehensive logging capabilities including:
- logger: Core logging functionality with research-oriented features
- decorators: Logging decorators for functions and methods
- analysis: Log file analysis and visualization tools

Usage:
    >>> from mfgarchon.utils.mfg_logging import get_logger, configure_research_logging
    >>> logger = get_logger(__name__)
    >>> configure_research_logging("my_experiment")
    >>> logger.info("Starting computation...")
"""

from __future__ import annotations

# =============================================================================
# Analysis (analysis.py)
# =============================================================================
from .analysis import (
    LogAnalyzer,
    analyze_log_file,
    analyze_recent_logs,
    find_performance_bottlenecks,
)

# =============================================================================
# Decorators (decorators.py)
# =============================================================================
from .decorators import (
    LoggingMixin,
    add_logging_to_class,
    logged_operation,
    logged_solver_method,
    logged_validation,
    performance_logged,
)

# =============================================================================
# Core logging (logger.py)
# =============================================================================
from .logger import (
    # Classes
    LoggedOperation,
    MFGFormatter,
    # Environment-specific configurations
    configure_development_logging,
    # Primary API
    configure_logging,
    configure_performance_logging,
    configure_production_logging,
    configure_research_logging,
    get_logger,
    # Structured logging helpers
    log_convergence_analysis,
    log_mass_conservation,
    log_memory_usage,
    log_performance_metric,
    log_solver_completion,
    log_solver_configuration,
    log_solver_progress,
    log_solver_start,
    log_validation_error,
)

__all__ = [
    # Core logging
    "configure_logging",
    "configure_research_logging",
    "get_logger",
    # Environment configurations
    "configure_development_logging",
    "configure_performance_logging",
    "configure_production_logging",
    # Structured logging helpers
    "log_convergence_analysis",
    "log_mass_conservation",
    "log_memory_usage",
    "log_performance_metric",
    "log_solver_completion",
    "log_solver_configuration",
    "log_solver_progress",
    "log_solver_start",
    "log_validation_error",
    # Classes
    "LoggedOperation",
    "MFGFormatter",
    # Decorators
    "LoggingMixin",
    "add_logging_to_class",
    "logged_operation",
    "logged_solver_method",
    "logged_validation",
    "performance_logged",
    # Analysis
    "LogAnalyzer",
    "analyze_log_file",
    "analyze_recent_logs",
    "find_performance_bottlenecks",
]
