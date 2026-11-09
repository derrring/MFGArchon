"""
Hooks System for MFGarchon

This module provides the hooks pattern for customizing solver behavior
without requiring complex inheritance or configuration.

Basic Usage:
    from mfgarchon.hooks import SolverHooks, DebugHook, PlottingHook

    hooks = DebugHook()
    result = solver.solve(problem, hooks=hooks)

Advanced Usage:
    from mfgarchon.hooks import MultiHook

    combined = MultiHook(DebugHook(), PlottingHook(), CustomHook())
    result = solver.solve(problem, hooks=combined)
"""

from .base import SolverHooks
from .composition import ChainHook, ConditionalHook, FilterHook, MultiHook, PriorityHook, TransformHook
from .control_flow import AdaptiveControlHook, ConditionalStopHook, ControlState, PerformanceControlHook, WatchdogHook
from .debug import ConvergenceAnalysisHook, DebugHook, PerformanceHook, StateInspectionHook
from .extensions import (
    AdaptiveParameterHook,
    AlgorithmExtensionHook,
    CustomConvergenceHook,
    CustomFPHook,
    CustomHJBHook,
    CustomInitializationHook,
    CustomResidualHook,
    MethodSwitchingHook,
    PostprocessingHook,
    PreprocessingHook,
)
from .visualization import AnimationHook, LoggingHook, PlottingHook, ProgressBarHook

__all__ = [
    "ChainHook",
    "ConditionalHook",
    "FilterHook",
    "MultiHook",
    "PriorityHook",
    "SolverHooks",
    "TransformHook",
    "AdaptiveControlHook",
    "ConditionalStopHook",
    "ControlState",
    "PerformanceControlHook",
    "WatchdogHook",
    "ConvergenceAnalysisHook",
    "DebugHook",
    "PerformanceHook",
    "StateInspectionHook",
    "AnimationHook",
    "LoggingHook",
    "PlottingHook",
    "ProgressBarHook",
    "AdaptiveParameterHook",
    "AlgorithmExtensionHook",
    "CustomConvergenceHook",
    "CustomFPHook",
    "CustomHJBHook",
    "CustomInitializationHook",
    "CustomResidualHook",
    "MethodSwitchingHook",
    "PostprocessingHook",
    "PreprocessingHook",
]
