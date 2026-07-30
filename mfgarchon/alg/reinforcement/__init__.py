"""
[FROZEN PROTOTYPE — not under development]

Reinforcement Learning paradigm for MFG problems.

⛔ This paradigm is a DESIGN PROTOTYPE / PLACEHOLDER. It is not under active development and
should not be extended, refactored, or given test coverage without an explicit instruction naming
it. See CLAUDE.md § "FROZEN: alg/neural/ and alg/reinforcement/".

Adding tests here is discouraged for a reason worth stating: coverage reads as a promise that the
behaviour is intended and load-bearing. On a placeholder that promise is false, and it makes later
readers preserve decisions nobody made. Record defects as issues instead — filing is free.


This module contains agent-based approaches for solving Mean Field Games:
- core: Base classes and shared infrastructure for MFG-RL
- algorithms: Specific RL algorithms adapted for MFG (MFRL, Nash-Q, etc.)
- approaches: Mathematical approach categories (value-based, policy-based, actor-critic)

This paradigm bridges multi-agent reinforcement learning with classical MFG theory,
providing agent-based approaches to MFG problems and completing the comprehensive
multi-paradigm architecture.

Note: RL solvers require optional dependencies (gymnasium, stable-baselines3).
"""

from mfgarchon.alg.base_solver import BaseRLSolver

# Check for RL dependencies
try:
    import gymnasium as gym  # noqa: F401

    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False

try:
    import stable_baselines3  # noqa: F401

    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

RL_DEPENDENCIES_AVAILABLE = GYMNASIUM_AVAILABLE and SB3_AVAILABLE

if RL_DEPENDENCIES_AVAILABLE:
    # Import core RL infrastructure
    from .core import (
        BaseMFRLSolver,
        MFGEnvironment,
        PopulationState,
        RLSolverConfig,
        RLSolverResult,
    )

    # Conditional imports for algorithms
    try:
        from .algorithms import (  # noqa: F401
            MeanFieldActorCritic,
            MeanFieldQLearning,
        )

        ALGORITHMS_AVAILABLE = True
    except ImportError:
        ALGORITHMS_AVAILABLE = False

    # Base exports (always available with dependencies)
    __all__ = [
        # Core Infrastructure
        "BaseMFRLSolver",
        "BaseRLSolver",
        "MFGEnvironment",
        "PopulationState",
        "RLSolverConfig",
        "RLSolverResult",
    ]

    # Add algorithm exports if available
    if ALGORITHMS_AVAILABLE:
        __all__.extend(
            [
                "MeanFieldActorCritic",
                # RL Algorithms for MFG
                "MeanFieldQLearning",
            ]
        )

    # Solver categories for factory selection
    VALUE_BASED_RL_SOLVERS = ["MeanFieldQLearning"] if ALGORITHMS_AVAILABLE else []
    ACTOR_CRITIC_RL_SOLVERS = ["MeanFieldActorCritic"] if ALGORITHMS_AVAILABLE else []
    ALL_RL_SOLVERS = VALUE_BASED_RL_SOLVERS + ACTOR_CRITIC_RL_SOLVERS

else:
    import warnings

    missing_deps = []
    if not GYMNASIUM_AVAILABLE:
        missing_deps.append("gymnasium")
    if not SB3_AVAILABLE:
        missing_deps.append("stable-baselines3")

    warnings.warn(
        f"Reinforcement Learning paradigm requires {', '.join(missing_deps)}. "
        f"Install with: pip install mfgarchon[rl] or pip install gymnasium stable-baselines3",
        ImportWarning,
    )

    __all__ = [
        "BaseRLSolver",
    ]

    # Empty solver categories when dependencies unavailable
    VALUE_BASED_RL_SOLVERS = []
    ACTOR_CRITIC_RL_SOLVERS = []
    ALL_RL_SOLVERS = []

# Always export availability info
__all__.extend(
    [
        "GYMNASIUM_AVAILABLE",
        "RL_DEPENDENCIES_AVAILABLE",
        "SB3_AVAILABLE",
    ]
)
