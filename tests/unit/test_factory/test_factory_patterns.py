#!/usr/bin/env python3
"""
Factory Patterns Test Suite - Simplified API

Tests the simplified factory API following infrastructure cleanup.
Removed: presets, create_fast_solver, create_accurate_solver, create_research_solver.
Remaining: create_solver() with explicit config.
"""

from unittest.mock import Mock, patch

import pytest

from mfgarchon.config import MFGSolverConfig, PicardConfig
from mfgarchon.factory.solver_factory import (
    SolverFactory,
    create_accurate_solver,
    create_basic_solver,
    create_fast_solver,
    create_research_solver,
    create_solver,
)


class MockMFGProblem:
    """Minimal mock MFG problem for testing solver factory."""

    def __init__(
        self,
        T=1.0,
        Nt=50,
        xmin=0.0,
        xmax=1.0,
        Nx=100,
        sigma=0.1,
        coupling_coefficient=0.5,
    ):
        self.T = T
        self.Nt = Nt  # Number of time INTERVALS (not points)
        self.xmin = xmin
        self.xmax = xmax
        self.Nx = Nx  # Number of space INTERVALS (not points)
        # Dx = domain_length / Nx (number of intervals)
        self.Dx = (xmax - xmin) / Nx if Nx > 0 else 0.0
        # Dt = T / Nt (number of intervals)
        self.Dt = T / Nt if Nt > 0 else 0.0
        self.sigma = sigma
        self.diffusion = sigma**2 / 2.0  # PDE coefficient D = sigma^2/2
        self.coupling_coefficient = coupling_coefficient


@pytest.mark.unit
def test_removed_functions_raise_not_implemented():
    """Test that removed convenience functions raise NotImplementedError."""
    removed_functions = [
        ("create_fast_solver", create_fast_solver),
        ("create_accurate_solver", create_accurate_solver),
        ("create_research_solver", create_research_solver),
        ("create_basic_solver", create_basic_solver),
    ]

    for _name, func in removed_functions:
        with pytest.raises(NotImplementedError) as exc_info:
            func()
        assert "has been removed" in str(exc_info.value)
        assert "problem.solve()" in str(exc_info.value) or "create_solver()" in str(exc_info.value)


@pytest.mark.unit
def test_create_solver_with_solvers():
    """Test create_solver with hjb_solver and fp_solver."""
    problem = MockMFGProblem()
    mock_hjb = Mock()
    mock_fp = Mock()

    with patch("mfgarchon.factory.solver_factory.FixedPointIterator") as MockIterator:
        MockIterator.return_value = Mock()

        solver = create_solver(
            problem=problem,
            hjb_solver=mock_hjb,
            fp_solver=mock_fp,
        )

        assert solver is not None
        MockIterator.assert_called_once()
        call_kwargs = MockIterator.call_args[1]
        assert call_kwargs["problem"] == problem
        assert call_kwargs["hjb_solver"] == mock_hjb
        assert call_kwargs["fp_solver"] == mock_fp


@pytest.mark.unit
def test_create_solver_with_custom_config():
    """Test create_solver with custom MFGSolverConfig."""
    problem = MockMFGProblem()
    mock_hjb = Mock()
    mock_fp = Mock()
    # Issue #1766: this was `MFGSolverConfig(convergence_tolerance=1e-8)`. That field
    # belongs to the RL config, not this one, and Pydantic silently dropped it -- so a
    # test named "custom config" was exercising the DEFAULT config with 1e-8 as
    # decoration. `picard.tolerance` is the real knob.
    custom_config = MFGSolverConfig(picard=PicardConfig(tolerance=1e-8))

    with patch("mfgarchon.factory.solver_factory.FixedPointIterator") as MockIterator:
        MockIterator.return_value = Mock()

        solver = create_solver(
            problem=problem,
            hjb_solver=mock_hjb,
            fp_solver=mock_fp,
            config=custom_config,
        )

        assert solver is not None
        call_kwargs = MockIterator.call_args[1]
        assert call_kwargs["config"] is not None


@pytest.mark.unit
def test_solver_factory_class():
    """Test SolverFactory.create_solver directly."""
    problem = MockMFGProblem()
    mock_hjb = Mock()
    mock_fp = Mock()

    with patch("mfgarchon.factory.solver_factory.FixedPointIterator") as MockIterator:
        MockIterator.return_value = Mock()

        solver = SolverFactory.create_solver(
            problem=problem,
            solver_type="fixed_point",
            hjb_solver=mock_hjb,
            fp_solver=mock_fp,
        )

        assert solver is not None
        MockIterator.assert_called_once()


@pytest.mark.unit
def test_create_solver_missing_solvers():
    """Test create_solver raises error when hjb_solver or fp_solver missing."""
    problem = MockMFGProblem()

    with pytest.raises(ValueError) as exc_info:
        SolverFactory.create_solver(problem=problem, solver_type="fixed_point")

    assert "requires both hjb_solver and fp_solver" in str(exc_info.value)


@pytest.mark.unit
def test_create_solver_invalid_solver_type():
    """Test create_solver raises error for invalid solver type."""
    problem = MockMFGProblem()
    mock_hjb = Mock()
    mock_fp = Mock()

    with pytest.raises(ValueError) as exc_info:
        SolverFactory.create_solver(
            problem=problem,
            solver_type="invalid_type",
            hjb_solver=mock_hjb,
            fp_solver=mock_fp,
        )

    assert "Unknown solver type" in str(exc_info.value)


@pytest.mark.unit
def test_create_solver_none_problem():
    """Test create_solver raises error for None problem."""
    with pytest.raises(ValueError) as exc_info:
        SolverFactory.create_solver(problem=None)

    assert "Problem cannot be None" in str(exc_info.value)


def run_comprehensive_test():
    """Run comprehensive factory pattern tests."""
    print("=" * 80)
    print("FACTORY PATTERNS TEST SUITE (Simplified API)")
    print("=" * 80)
    print("NOTE: Convenience functions (create_fast_solver, etc.) have been removed.")
    print("      Use create_solver() or problem.solve() instead.")
    print("=" * 80)

    all_results = {}

    # Test removed functions raise NotImplementedError
    print("\n1. Testing removed functions raise NotImplementedError...")
    removed_functions = [
        ("create_fast_solver", create_fast_solver),
        ("create_accurate_solver", create_accurate_solver),
        ("create_research_solver", create_research_solver),
    ]

    for name, func in removed_functions:
        try:
            func()
            print(f"  X {name}: Did not raise NotImplementedError!")
            all_results[name] = {"success": False, "error": "Did not raise"}
        except NotImplementedError:
            print(f"  OK {name}: Correctly raises NotImplementedError")
            all_results[name] = {"success": True}
        except Exception as e:
            print(f"  X {name}: Wrong exception: {e}")
            all_results[name] = {"success": False, "error": str(e)}

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    passed = sum(1 for r in all_results.values() if r.get("success", False))
    total = len(all_results)
    print(f"Passed: {passed}/{total}")

    return all_results


if __name__ == "__main__":
    run_comprehensive_test()
