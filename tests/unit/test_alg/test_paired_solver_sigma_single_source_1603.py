"""RFC #1574 C14 / Issue #1603: the HJB-FP two-problem sigma-mismatch guard is single-sourced in
`assert_paired_solver_sigma` and wired into EVERY coupling iterator, not just FixedPointIterator.

A coupled HJB-FP pair is an adjoint pair and must share the volatility; if the two paired solvers
were built from problems with different sigma, HJB and FP diffuse at different rates with no warning
and the fixed point is neither problem's MFG. #1603 guarded only FixedPointIterator; the Block,
FictitiousPlay, Newton, and the regime / multi-population / graph list iterators had NO guard (a
silent-wrong gap). The check now lives in one owner (base_mfg.assert_paired_solver_sigma) called by
all of them -- for the list iterators, once per sub-problem pair.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.alg.numerical.coupling.base_mfg import assert_paired_solver_sigma
from mfgarchon.alg.numerical.coupling.block_iterators import BlockIterator
from mfgarchon.alg.numerical.coupling.fictitious_play import FictitiousPlayIterator
from mfgarchon.alg.numerical.coupling.multi_population_iterator import MultiPopulationIterator
from mfgarchon.alg.numerical.coupling.newton_mfg_solver import NewtonMFGSolver
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.core.multi_population import MultiPopulationProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def _make_problem(sigma=0.3):
    components = MFGComponents(
        hamiltonian=SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=1.0), coupling=lambda m: m, coupling_dm=lambda m: 1.0
        ),
        u_terminal=lambda x: 0.0,
        m_initial=lambda x: 1.0,
    )
    return MFGProblem(
        geometry=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[12], boundary_conditions=no_flux_bc(dimension=1)),
        T=0.2,
        Nt=5,
        sigma=sigma,
        components=components,
    )


class _Stub:
    def __init__(self, sigma):
        self.problem = type("P", (), {"sigma": sigma})()


def test_helper_raises_on_scalar_mismatch():
    with pytest.raises(ValueError, match="different sigma"):
        assert_paired_solver_sigma(_Stub(0.3), _Stub(0.5), "ctx")


def test_helper_passes_on_match():
    assert_paired_solver_sigma(_Stub(0.3), _Stub(0.3), "ctx")  # must not raise


def test_helper_skips_non_scalar_sigma():
    """Array / callable / Mock sigma (not a real scalar) is not compared -- no false positive (#1489)."""
    assert_paired_solver_sigma(_Stub(np.array([0.3, 0.4])), _Stub(0.5), "ctx")  # must not raise


@pytest.mark.parametrize("IterCls", [BlockIterator, FictitiousPlayIterator, NewtonMFGSolver])
def test_single_pair_iterator_raises_on_sigma_mismatch(IterCls):
    """Block / FictitiousPlay / Newton now raise on a two-problem sigma mismatch (they had no guard).
    Discriminating: removing the assert_paired_solver_sigma call from an iterator makes it construct
    silently and this fails."""
    hjb = HJBFDMSolver(_make_problem(sigma=0.2))
    fp = FPFDMSolver(_make_problem(sigma=0.8))
    with pytest.raises(ValueError, match="sigma"):
        IterCls(_make_problem(sigma=0.2), hjb, fp)


def test_single_pair_iterator_silent_when_matched():
    """Matched sigma across the two problems -> construction succeeds (guard keys on VALUE)."""
    hjb = HJBFDMSolver(_make_problem(sigma=0.3))
    fp = FPFDMSolver(_make_problem(sigma=0.3))
    BlockIterator(_make_problem(sigma=0.3), hjb, fp)  # must not raise


def test_multipopulation_raises_on_per_pair_sigma_mismatch():
    """The list-based iterators are guarded per sub-problem pair, not just population 0. Population 1's
    HJB (sigma=0.2) and FP (sigma=0.8) mismatch must raise, naming the population. Discriminating:
    dropping the per-pair guard loop makes this construct silently."""
    pop0 = _make_problem(sigma=0.3)
    pop1 = _make_problem(sigma=0.2)
    multi = MultiPopulationProblem(populations=[pop0, pop1])
    hjbs = [HJBFDMSolver(pop0), HJBFDMSolver(pop1)]
    fps = [FPFDMSolver(pop0), FPFDMSolver(_make_problem(sigma=0.8))]  # population-1 FP: different sigma
    with pytest.raises(ValueError, match="population 1"):
        MultiPopulationIterator(multi, hjbs, fps)
