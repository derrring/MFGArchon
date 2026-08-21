"""Both network HJB solvers must REFUSE a continuous-domain problem, and say why (#2045).

They used to fail differently, and only one of them on purpose:

    NetworkHJBSolver                 NotImplementedError, naming node BCs
    NetworkPolicyIterationHJBSolver  AttributeError: 'MFGProblem' object has no attribute 'num_nodes'

The asymmetry is subtle rather than accidental. `NetworkPolicyIterationHJBSolver` sets
`_honors_node_bc`, so it SKIPS the deliberate node-BC refusal its base class raises and fell
through to the first network-only attribute read. An `AttributeError` from inside construction
tells a caller nothing about why the solver is inapplicable, and is indistinguishable from a
defect in the solver itself -- so a capability sweep classifying solvers by how they fail has to
special-case it, and `AttributeError` is the bucket such sweeps tend to read as "harness problem"
rather than "solver refused".

The guard is a try/except around the three network reads rather than `hasattr` on one of them.
Two reasons, and the second is the load-bearing one: the fail-fast policy names try-except as the
replacement for `hasattr` (a `hasattr` here moved the ratchet 107 -> 108), and a single-attribute
guard would wave through a problem that carries `num_nodes` but no adjacency accessor.
"""

from __future__ import annotations

import logging
import warnings

import pytest

from mfgarchon.alg.numerical.network_solvers.hjb_network import (
    NetworkHJBSolver,
    NetworkPolicyIterationHJBSolver,
)
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.geometry.grids import TensorProductGrid


def _grid_problem():
    from tests.integration.test_hjb_with_obstacle import _default_components

    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=no_flux_bc(dimension=1))
    return MFGProblem(geometry=grid, T=0.2, Nt=4, sigma=0.2, components=_default_components())


@pytest.fixture(autouse=True)
def _quiet():
    logging.disable(logging.CRITICAL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield
    logging.disable(logging.NOTSET)


@pytest.mark.parametrize("solver_cls", [NetworkHJBSolver, NetworkPolicyIterationHJBSolver])
def test_a_grid_problem_is_refused_loudly(solver_cls):
    """`NotImplementedError`, not `AttributeError`. The type is the contract here: it is what
    separates "this solver does not apply to your problem" from "this solver is broken"."""
    with pytest.raises(NotImplementedError):
        solver_cls(_grid_problem())


def test_the_policy_iteration_refusal_names_what_it_needs():
    """It skips the base class's node-BC refusal, so its own message is the only one a caller of
    that solver ever sees. A refusal that does not say what would satisfy it is a dead end."""
    with pytest.raises(NotImplementedError) as excinfo:
        NetworkPolicyIterationHJBSolver(_grid_problem())

    message = str(excinfo.value)
    assert "network problem" in message
    assert "num_nodes" in message, "the requirement must be nameable, not just gestured at"
    assert "TensorProductGrid" in message, "and the caller's actual geometry, so they can see the mismatch"


def test_a_problem_with_num_nodes_but_no_adjacency_is_also_refused():
    """Why the guard wraps all three reads instead of testing one attribute.

    A single-attribute guard passes a half-network problem through to the next line, where the
    AttributeError this issue is about reappears one statement later.
    """

    class _HalfNetwork:
        num_nodes = 7

        def __getattr__(self, name):
            if name in {"get_adjacency_matrix", "get_laplacian_matrix"}:
                raise AttributeError(name)
            return getattr(_grid_problem(), name)

    with pytest.raises(NotImplementedError, match="adjacency"):
        NetworkPolicyIterationHJBSolver(_HalfNetwork())


def test_the_guard_does_not_use_hasattr():
    """The fail-fast ratchet counts `hasattr` calls, and this guard moved it 107 -> 108 in its
    first form. Pinned because the obvious rewrite of a try/except is the thing the policy forbids.
    """
    import inspect

    source = inspect.getsource(NetworkHJBSolver.__init__)
    assert "hasattr(problem" not in source, "use try/except AttributeError, per the fail-fast policy"
