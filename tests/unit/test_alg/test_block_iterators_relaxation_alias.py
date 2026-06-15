"""Equivalence tests for BlockIterator damping_* -> relaxation_* rename (v0.19.3).

Per CLAUDE.md deprecation policy: legacy `damping_factor` / `damping_factor_M`
ctor kwargs must produce behavior identical to the canonical `relaxation` /
`relaxation_M`. Accepts via `@deprecated_parameter` + body redirect on three
classes: `BlockIterator` (base), `BlockJacobiIterator`, `BlockGaussSeidelIterator`.

Silent `@property` aliases keep `solver.damping_factor` attribute access working
without warning flooding.

Removal of legacy kwargs and attribute aliases scheduled for v0.25.0.
"""

from __future__ import annotations

import warnings

import pytest

from mfgarchon.alg.numerical.coupling.block_iterators import (
    BlockGaussSeidelIterator,
    BlockIterator,
    BlockJacobiIterator,
)
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def _make_test_problem():
    H = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: m,
        coupling_dm=lambda m: 1.0,
    )
    components = MFGComponents(
        hamiltonian=H,
        u_terminal=lambda x: 0.0,
        m_initial=lambda x: 1.0,
    )
    return MFGProblem(
        geometry=TensorProductGrid(
            bounds=[(0.0, 1.0)], Nx_points=[11 + 1], boundary_conditions=no_flux_bc(dimension=1)
        ),
        T=0.2,
        Nt=5,
        sigma=0.3,
        components=components,
    )


@pytest.fixture
def solvers():
    problem = _make_test_problem()
    return problem, HJBFDMSolver(problem), FPFDMSolver(problem)


@pytest.mark.parametrize("cls", [BlockIterator, BlockJacobiIterator, BlockGaussSeidelIterator])
class TestEquivalenceAcrossAllThreeClasses:
    """All three block-iterator classes honor the rename consistently."""

    def test_legacy_damping_factor_equivalent_to_relaxation(self, solvers, cls):
        problem, hjb, fp = solvers
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            legacy = cls(problem, hjb, fp, damping_factor=0.7)
        canonical = cls(problem, hjb, fp, relaxation=0.7)
        assert legacy.relaxation == canonical.relaxation == 0.7

    def test_damping_factor_property_reads_relaxation(self, solvers, cls):
        problem, hjb, fp = solvers
        inst = cls(problem, hjb, fp, relaxation=0.8)
        assert inst.damping_factor == 0.8
        assert inst.damping_factor == inst.relaxation

    def test_property_reads_silent(self, solvers, cls):
        problem, hjb, fp = solvers
        inst = cls(problem, hjb, fp, relaxation=0.5)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = inst.damping_factor
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep_warnings) == 0

    def test_canonical_kwarg_emits_no_warning(self, solvers, cls):
        problem, hjb, fp = solvers
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cls(problem, hjb, fp, relaxation=0.5)
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning) and "damping" in str(x.message)]
        assert len(dep_warnings) == 0

    def test_legacy_kwarg_emits_deprecation_warning(self, solvers, cls):
        problem, hjb, fp = solvers
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cls(problem, hjb, fp, damping_factor=0.5)
            dep_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning) and "damping_factor" in str(x.message)
            ]
        assert len(dep_warnings) >= 1


class TestBaseClassBothKwargs:
    """BlockIterator base accepts damping_factor_M as well as damping_factor."""

    def test_both_legacy_kwargs_equivalent_to_both_canonical(self, solvers):
        problem, hjb, fp = solvers
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            legacy = BlockIterator(problem, hjb, fp, damping_factor=0.8, damping_factor_M=0.2)
        canonical = BlockIterator(problem, hjb, fp, relaxation=0.8, relaxation_M=0.2)
        assert legacy.relaxation == canonical.relaxation == 0.8
        assert legacy.relaxation_M == canonical.relaxation_M == 0.2

    def test_damping_factor_M_property_reads_relaxation_M(self, solvers):
        problem, hjb, fp = solvers
        inst = BlockIterator(problem, hjb, fp, relaxation=0.5, relaxation_M=0.2)
        assert inst.damping_factor_M == 0.2
        assert inst.damping_factor_M == inst.relaxation_M


class TestMetadataKey:
    """Result metadata uses canonical `relaxation` key (legacy key removed in v0.19.3)."""

    def test_solve_result_metadata_has_relaxation_key(self, solvers):
        problem, hjb, fp = solvers
        solver = BlockGaussSeidelIterator(problem, hjb, fp, relaxation=0.6)
        result = solver.solve(max_iterations=2, tolerance=1e-4, verbose=False)
        assert "relaxation" in result.metadata
        assert result.metadata["relaxation"] == 0.6
        assert "damping_factor" not in result.metadata
