"""
Framework extensions for MFG problems.

Extensions provide optional problem formulations beyond the core MFGProblem:
- Network MFG: Problems on graph topologies
- Multi-population MFG: Heterogeneous K-population games

Import from here for extended functionality:
    from mfgarchon.extensions import NetworkMFGProblem, MultiPopulationMFGProblem

Note: Variational MFG lives in mfgarchon.alg.optimization.variational_problem
(VariationalMFGProblem, VariationalMFGComponents). Its solver stack is
experimental and blocked by Issue #1342.
"""

from .multi_population import MultiPopulationMFGProblem, MultiPopulationMFGProtocol
from .topology import (
    NetworkMFGComponents,
    NetworkMFGProblem,
    create_grid_mfg_problem,
    create_random_mfg_problem,
    create_scale_free_mfg_problem,
)

__all__ = [
    # Multi-population MFG
    "MultiPopulationMFGProblem",
    "MultiPopulationMFGProtocol",
    # Network MFG
    "NetworkMFGComponents",
    "NetworkMFGProblem",
    "create_grid_mfg_problem",
    "create_random_mfg_problem",
    "create_scale_free_mfg_problem",
]
