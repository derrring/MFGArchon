"""#2648 stage 2: declared inputs that no code reads are refused by name, until v0.25.0 (ruling 3).

1. Each member is refused when passed, by keyword, and by position where the class takes it positionally. The
   message names the class or function that refuses it, says the input never had an effect, and gives the reason.
2. Its default is `NEVER_READ`, which the refusal lets through. So each class still constructs, and
   `dataclasses.replace`, which passes every field back in, still works.
3. `SolverResult.ergodic_constant` is removed (#2652): the attribute is gone, and passing the name is refused.
4. A call that does not bind gets the callable's own `TypeError`, with no binding error chained to it (review 2 of
   #2650, S2).
"""

from __future__ import annotations

import dataclasses

import pytest

import numpy as np

from mfgarchon.alg.numerical.hjb_solvers.hjb_gfdm import HJBGFDMSolver
from mfgarchon.alg.optimization.optimal_transport.wasserstein_solver import WassersteinSolverConfig
from mfgarchon.alg.optimization.variational_problem import VariationalMFGComponents
from mfgarchon.core.multi_population import MultiPopulationProblem
from mfgarchon.extensions.topology import NetworkMFGComponents
from mfgarchon.geometry.boundary.applicator_graph import GraphBCConfig, NodeBC
from mfgarchon.geometry.graph.maze_hybrid import AlgorithmSpec, HybridMazeConfig, HybridStrategy
from mfgarchon.hooks.debug import PerformanceHook
from mfgarchon.utils.deprecation import NEVER_READ
from mfgarchon.utils.notebooks.reporting import MFGNotebookReporter
from mfgarchon.utils.numerical.hjb_policy_iteration import LQPolicyIterationHelper, create_lq_policy_problem
from mfgarchon.utils.numerical.particle.mcmc import MCMCConfig
from mfgarchon.utils.numerical.particle.sampling import MCConfig
from mfgarchon.utils.solver_result import SolverResult
from mfgarchon.workflow.parameter_sweep import SweepConfiguration


def _f(*_args):
    return 0.0


_MAZE = {"rows": 10, "cols": 10, "strategy": None, "algorithms": []}

# (callable, its qualified name in the message, positional args, keyword args, the refused name)
REFUSED = [
    (MCMCConfig, "MCMCConfig.__init__", (), {"num_chains": 4}, "num_chains"),
    (MCMCConfig, "MCMCConfig.__init__", (10000, 1000, 4), {}, "num_chains"),
    (MCMCConfig, "MCMCConfig.__init__", (), {"max_tree_depth": 8}, "max_tree_depth"),
    (MCMCConfig, "MCMCConfig.__init__", (), {"max_adapt_iter": 500}, "max_adapt_iter"),
    (MCMCConfig, "MCMCConfig.__init__", (), {"parallel": True}, "parallel"),
    (MCMCConfig, "MCMCConfig.__init__", (), {"compute_rhat": False}, "compute_rhat"),
    (MCMCConfig, "MCMCConfig.__init__", (), {"effective_sample_size": False}, "effective_sample_size"),
    (WassersteinSolverConfig, "WassersteinSolverConfig.__init__", (), {"step_size": 0.5}, "step_size"),
    (MCConfig, "MCConfig.__init__", (), {"adaptive": True}, "adaptive"),
    (MCConfig, "MCConfig.__init__", (), {"max_adaptive_samples": 10}, "max_adaptive_samples"),
    (MCConfig, "MCConfig.__init__", (), {"proposal_distribution": "gaussian"}, "proposal_distribution"),
    (MCConfig, "MCConfig.__init__", (), {"batch_size": 10}, "batch_size"),
    (MCConfig, "MCConfig.__init__", (), {"chunk_size": 10}, "chunk_size"),
    (MCConfig, "MCConfig.__init__", (), {"parallel": False}, "parallel"),
    (
        WassersteinSolverConfig,
        "WassersteinSolverConfig.__init__",
        (),
        {"relative_tolerance": False},
        "relative_tolerance",
    ),
    (WassersteinSolverConfig, "WassersteinSolverConfig.__init__", (), {"history_length": 3}, "history_length"),
    (NetworkMFGComponents, "NetworkMFGComponents.__init__", (), {"velocity_space_dim": 3}, "velocity_space_dim"),
    (NetworkMFGComponents, "NetworkMFGComponents.__init__", (), {"trajectory_cost_func": _f}, "trajectory_cost_func"),
    (NetworkMFGComponents, "NetworkMFGComponents.__init__", (), {"relaxed_control": True}, "relaxed_control"),
    (NetworkMFGComponents, "NetworkMFGComponents.__init__", (), {"edge_cost_func": _f}, "edge_cost_func"),
    (NetworkMFGComponents, "NetworkMFGComponents.__init__", (), {"congestion_func": _f}, "congestion_func"),
    (NetworkMFGComponents, "NetworkMFGComponents.__init__", (), {"edge_interaction_func": _f}, "edge_interaction_func"),
    (NetworkMFGComponents, "NetworkMFGComponents.__init__", (), {"problem_params": {"a": 1}}, "problem_params"),
    (
        VariationalMFGComponents,
        "VariationalMFGComponents.__init__",
        (),
        {"terminal_cost_dx_func": _f},
        "terminal_cost_dx_func",
    ),
    (VariationalMFGComponents, "VariationalMFGComponents.__init__", (), {"potential_func": _f}, "potential_func"),
    (VariationalMFGComponents, "VariationalMFGComponents.__init__", (), {"potential_dx_func": _f}, "potential_dx_func"),
    (VariationalMFGComponents, "VariationalMFGComponents.__init__", (), {"problem_type": "obstacle"}, "problem_type"),
    (
        MultiPopulationProblem,
        "MultiPopulationProblem.__init__",
        (),
        {"populations": [], "cross_coupling": _f},
        "cross_coupling",
    ),
    (GraphBCConfig, "GraphBCConfig.__init__", (), {"default_bc_type": None}, "default_bc_type"),
    (HybridMazeConfig, "HybridMazeConfig.__init__", (), {**_MAZE, "radial_center": (5, 5)}, "radial_center"),
    (SweepConfiguration, "SweepConfiguration.__init__", (), {"parameters": {}, "batch_size": 4}, "batch_size"),
    (HJBGFDMSolver, "HJBGFDMSolver.__init__", (None, None), {"qp_usage_target": 0.1}, "qp_usage_target"),
    (PerformanceHook, "PerformanceHook.__init__", (), {"track_detailed_timing": True}, "track_detailed_timing"),
    (PerformanceHook, "PerformanceHook.__init__", (False, True), {}, "track_detailed_timing"),
    (MFGNotebookReporter, "MFGNotebookReporter.__init__", (), {"enable_latex": False}, "enable_latex"),
    (MFGNotebookReporter, "MFGNotebookReporter.__init__", (), {"template_style": "minimal"}, "template_style"),
    (
        LQPolicyIterationHelper,
        "LQPolicyIterationHelper.__init__",
        (object(),),
        {"potential_func": _f},
        "potential_func",
    ),
    (create_lq_policy_problem, "create_lq_policy_problem", (object(), _f), {}, "potential_func"),
]


@pytest.mark.parametrize(("target", "qualname", "args", "kwargs", "name"), REFUSED)
def test_an_input_that_never_had_an_effect_is_refused_by_name(target, qualname, args, kwargs, name):
    with pytest.raises(
        TypeError,
        match=rf"^{qualname.replace('.', r'\.')}\(\) refuses {name}=: .+; it never had an effect, and the name goes in v0\.25\.0",
    ):
        target(*args, **kwargs)


# (callable, kwargs, phrases reviews 1 and 2 of #2658 found false, which the reason must not carry)
_SECTIONS = ("always written", "same section structure", "fixed by the report method called")
FALSE_REMEDIES = [
    (NetworkMFGComponents, {"trajectory_cost_func": _f}, ("lagrangian_func",)),
    (NetworkMFGComponents, {"congestion_func": _f}, ("pass hamiltonian_func to change it",)),
    (NetworkMFGComponents, {"problem_params": {"a": 1}}, ("parameters=",)),
    (MFGNotebookReporter, {"enable_latex": False}, _SECTIONS),
    (MFGNotebookReporter, {"template_style": "minimal"}, _SECTIONS),
    (MCMCConfig, {"effective_sample_size": False}, ("several runs",)),
]


@pytest.mark.parametrize(("target", "kwargs", "false"), FALSE_REMEDIES)
def test_a_reason_names_no_path_the_code_does_not_take(target, kwargs, false):
    """Reviews 1 and 2 of #2658: no network solve reads `lagrangian_func` or problem parameters, a report's sections
    also follow what is passed to the method, and `effective_sample_size` takes one chain's samples."""
    with pytest.raises(TypeError) as refused:
        target(**kwargs)
    assert not [phrase for phrase in false if phrase in str(refused.value)], str(refused.value)


#: One graph condition, built once, so a round trip through `GraphBCConfig` can be compared by identity (#2292).
_GRAPH_NODE_BCS = (NodeBC(nodes=[0], name="row"),)

DATACLASSES = [
    (MCMCConfig, {}, "num_samples", 7),
    (MCConfig, {}, "num_samples", 7),
    (WassersteinSolverConfig, {}, "max_iterations", 7),
    (NetworkMFGComponents, {}, "node_potential_func", _f),
    (VariationalMFGComponents, {}, "terminal_cost_func", _f),
    # node_bcs, not edge_bcs: an edge condition is refused (#2493). The config keeps its conditions as a tuple, which
    # `tuple()` hands back as the same object, so the round trip below compares identity; and not the empty default,
    # which a config dropping the field would hand back too (#2292).
    (GraphBCConfig, {}, "node_bcs", _GRAPH_NODE_BCS),
    (MultiPopulationProblem, {"populations": [object()]}, "population_names", ["only"]),
    (
        HybridMazeConfig,
        {"rows": 10, "cols": 10, "strategy": HybridStrategy.HIERARCHICAL, "algorithms": [AlgorithmSpec("perfect")]},
        "seed",
        7,
    ),
    (SweepConfiguration, {"parameters": {}}, "max_workers", 7),
]


@pytest.mark.parametrize(("cls", "kwargs", "field", "value"), DATACLASSES)
def test_each_class_constructs_and_replace_passes_the_never_read_default_back(cls, kwargs, field, value):
    obj = cls(**kwargs)
    changed = dataclasses.replace(obj, **{field: value})
    assert getattr(changed, field) is value
    refused = [f.name for f in dataclasses.fields(cls) if f.default is NEVER_READ]
    assert refused, f"{cls.__name__} declares no NEVER_READ field"
    assert all(getattr(changed, name) is NEVER_READ for name in refused)


def test_the_callables_that_take_a_refused_parameter_still_run_without_it():
    """Control: the defaults pass, including the one `create_lq_policy_problem` forwards to its helper."""
    assert PerformanceHook(track_memory=True).track_memory is True
    assert LQPolicyIterationHelper(object()).potential_func is NEVER_READ
    assert create_lq_policy_problem(object()) is not None
    assert MFGNotebookReporter(output_dir="reports").output_dir is not None


def _result(**kwargs):
    return SolverResult(np.zeros((2, 3)), np.ones((2, 3)), 1, np.zeros(1), np.zeros(1), **kwargs)


def test_solver_result_has_no_ergodic_constant():
    """#2652: no solver computed one and there is no stationary mode, so the field is removed and the name refused."""
    result = _result()
    assert not hasattr(result, "ergodic_constant")
    assert "ergodic_constant" not in [f.name for f in dataclasses.fields(SolverResult)]
    assert dataclasses.replace(result, solver_name="x").solver_name == "x"
    with pytest.raises(
        TypeError,
        match=r"^SolverResult\.__init__\(\) refuses ergodic_constant=: the field is removed: .+\(#2652\); "
        r"no library solver ever set it, ",
    ):
        _result(ergodic_constant=0.5)


def test_a_call_that_does_not_bind_gets_the_callables_own_unchained_error():
    with pytest.raises(
        TypeError, match=r"MCMCConfig\.__init__\(\) got an unexpected keyword argument 'bogus'"
    ) as caught:
        MCMCConfig(bogus=1)
    assert caught.value.__context__ is None, caught.value.__context__
