#!/usr/bin/env python3
"""
Unit tests for mfgarchon/utils/experiment_manager.py

Tests experiment management utilities including:
- Mass calculation
- Experiment data saving/loading (NPZ format)
- Batch experiment loading from directories
- Comparison plotting functions
- Error handling and validation
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

import numpy as np

from mfgarchon.utils.experiment_manager import (
    calculate_total_mass,
    load_experiment_data,
    load_experiments_from_dir,
    plot_comparison_final_m,
    plot_comparison_initial_U,
    plot_comparison_total_mass,
    plot_comparison_U_slice,
    save_experiment_data,
)

# ===================================================================
# Test Mass Calculation
# ===================================================================


@pytest.mark.unit
def test_calculate_total_mass_1d():
    """Test total mass calculation for 1D density array."""
    M = np.array([[1.0, 2.0, 1.0], [0.5, 1.0, 0.5]])  # (Nt=2, Nx=3)
    Dx = 0.5

    total_mass = calculate_total_mass(M, Dx)

    # Should integrate M * Dx for each time step
    assert total_mass.shape == (2,)
    assert total_mass[0] == pytest.approx(2.0)  # (1.0 + 2.0 + 1.0) * 0.5
    assert total_mass[1] == pytest.approx(1.0)  # (0.5 + 1.0 + 0.5) * 0.5


@pytest.mark.unit
def test_calculate_total_mass_zero_dx():
    """Test total mass calculation when Dx=0 (Nx=1 case)."""
    M = np.array([[1.0], [2.0], [3.0]])  # (Nt=3, Nx=1)
    Dx = 0.0

    total_mass = calculate_total_mass(M, Dx)

    # Should sum without Dx when Dx=0
    assert total_mass.shape == (3,)
    assert total_mass[0] == pytest.approx(1.0)
    assert total_mass[1] == pytest.approx(2.0)
    assert total_mass[2] == pytest.approx(3.0)


@pytest.mark.unit
def test_calculate_total_mass_single_time():
    """Test total mass calculation for 2D array with single time step."""
    M = np.array([[1.0, 2.0, 1.0]])  # (Nt=1, Nx=3)
    Dx = 0.5

    total_mass = calculate_total_mass(M, Dx)

    # Should handle single time step
    assert total_mass.shape == (1,)
    assert total_mass[0] == pytest.approx(2.0)


# ===================================================================
# Test Experiment Data Saving
# ===================================================================


def create_mock_problem():
    """Create mock MFG problem for testing.

    Note: This mock maintains legacy attributes (Nx, Nt, dx, xmin, xmax)
    for backward compatibility with experiment_manager functions which
    directly access these attributes.
    """
    problem = Mock()
    problem.T = 1.0
    problem.Nx = 10  # Number of space INTERVALS
    problem.Nt = 20  # Number of time INTERVALS
    problem.dx = 0.1
    problem.dt = 0.05
    problem.xmin = 0.0
    problem.xmax = 1.0
    problem.sigma = 0.1
    problem.coupling_coefficient = 0.5
    problem.tSpace = np.linspace(0, 1.0, 21)  # 21 time points (Nt+1)
    problem.xSpace = np.linspace(0.0, 1.0, 11)  # 11 space points (Nx+1)

    # Add geometry mock for geometry-first API
    geometry = Mock()
    geometry.get_grid_shape.return_value = (11,)  # Nx+1 points
    geometry.get_grid_spacing.return_value = (0.1,)  # dx
    geometry.get_bounds.return_value = ((0.0,), (1.0,))  # (xmin, xmax)
    geometry.get_spatial_grid.return_value = np.linspace(0.0, 1.0, 11)  # 11 space points
    problem.geometry = geometry

    return problem


@pytest.mark.unit
def test_save_experiment_data_basic():
    """Test basic experiment data saving."""
    problem = create_mock_problem()
    U_solution = np.random.rand(21, 11)
    M_solution = np.random.rand(21, 11)
    rel_dist_U = np.geomspace(1, 1e-5, 10)
    rel_dist_M = np.geomspace(1, 1e-5, 10)
    abs_dist_U = rel_dist_U * 10
    abs_dist_M = rel_dist_M * 5

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = save_experiment_data(
            problem=problem,
            U_solution=U_solution,
            M_solution=M_solution,
            solver_name="TestSolver",
            iterations_run=10,
            rel_dist_U=rel_dist_U,
            rel_dist_M=rel_dist_M,
            abs_dist_U=abs_dist_U,
            abs_dist_M=abs_dist_M,
            execution_time=12.34,
            output_dir_base=tmpdir,
        )

        assert filepath != ""
        assert Path(filepath).exists()
        assert filepath.endswith(".npz")
        assert "TestSolver" in filepath


@pytest.mark.unit
def test_save_experiment_data_with_additional_params():
    """Test saving experiment data with additional parameters."""
    problem = create_mock_problem()
    U_solution = np.random.rand(21, 11)
    M_solution = np.random.rand(21, 11)

    additional_params = {"damping": 0.5, "method": "newton", "adaptive": True}

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = save_experiment_data(
            problem=problem,
            U_solution=U_solution,
            M_solution=M_solution,
            solver_name="CustomSolver",
            iterations_run=5,
            rel_dist_U=np.array([1e-3]),
            rel_dist_M=np.array([1e-3]),
            abs_dist_U=np.array([1e-2]),
            abs_dist_M=np.array([1e-2]),
            execution_time=5.67,
            output_dir_base=tmpdir,
            additional_params=additional_params,
        )

        assert Path(filepath).exists()

        # Load and verify additional params were saved
        loaded_data = load_experiment_data(filepath)
        assert loaded_data is not None
        assert "problem_params" in loaded_data
        params = loaded_data["problem_params"]
        assert params["damping"] == 0.5
        assert params["method"] == "newton"
        assert params["adaptive"] is True


@pytest.mark.unit
def test_save_experiment_data_creates_directories():
    """Test that save_experiment_data creates necessary directories."""
    problem = create_mock_problem()
    U_solution = np.random.rand(21, 11)
    M_solution = np.random.rand(21, 11)

    with tempfile.TemporaryDirectory() as tmpdir:
        nested_dir = Path(tmpdir) / "nested" / "results"
        filepath = save_experiment_data(
            problem=problem,
            U_solution=U_solution,
            M_solution=M_solution,
            solver_name="NestedSolver",
            iterations_run=1,
            rel_dist_U=np.array([1e-3]),
            rel_dist_M=np.array([1e-3]),
            abs_dist_U=np.array([1e-2]),
            abs_dist_M=np.array([1e-2]),
            execution_time=1.0,
            output_dir_base=str(nested_dir),
        )

        # Directory should be created
        assert nested_dir.exists()
        assert Path(filepath).exists()


@pytest.mark.unit
def test_save_experiment_data_filename_format():
    """Test that saved filename contains expected components."""
    problem = create_mock_problem()
    U_solution = np.random.rand(21, 11)
    M_solution = np.random.rand(21, 11)

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = save_experiment_data(
            problem=problem,
            U_solution=U_solution,
            M_solution=M_solution,
            solver_name="FilenameTest",
            iterations_run=1,
            rel_dist_U=np.array([1e-3]),
            rel_dist_M=np.array([1e-3]),
            abs_dist_U=np.array([1e-2]),
            abs_dist_M=np.array([1e-2]),
            execution_time=1.0,
            output_dir_base=tmpdir,
        )

        filename = Path(filepath).name
        # Should contain T, Nx, Nt, sigma, coupling_coefficient
        assert "T1.0" in filename
        assert "Nx10" in filename
        assert "Nt20" in filename
        assert "sig" in filename
        assert "ct" in filename


# ===================================================================
# Test Experiment Data Loading
# ===================================================================


@pytest.mark.unit
def test_load_experiment_data_basic():
    """Test loading experiment data from file."""
    problem = create_mock_problem()
    U_solution = np.random.rand(21, 11)
    M_solution = np.random.rand(21, 11)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save data first
        filepath = save_experiment_data(
            problem=problem,
            U_solution=U_solution,
            M_solution=M_solution,
            solver_name="LoadTest",
            iterations_run=10,
            rel_dist_U=np.geomspace(1, 1e-5, 10),
            rel_dist_M=np.geomspace(1, 1e-5, 10),
            abs_dist_U=np.geomspace(10, 1e-4, 10),
            abs_dist_M=np.geomspace(5, 5e-5, 10),
            execution_time=12.34,
            output_dir_base=tmpdir,
        )

        # Load data
        loaded_data = load_experiment_data(filepath)

        assert loaded_data is not None
        assert "U_solution" in loaded_data
        assert "M_solution" in loaded_data
        assert "total_mass_vs_time" in loaded_data
        assert "solver_name" in loaded_data
        assert "iterations_run" in loaded_data
        assert "execution_time" in loaded_data


@pytest.mark.unit
def test_load_experiment_data_arrays_match():
    """Test that loaded arrays match saved arrays."""
    problem = create_mock_problem()
    U_solution = np.random.rand(21, 11)
    M_solution = np.random.rand(21, 11)

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = save_experiment_data(
            problem=problem,
            U_solution=U_solution,
            M_solution=M_solution,
            solver_name="ArrayTest",
            iterations_run=5,
            rel_dist_U=np.array([1e-3]),
            rel_dist_M=np.array([1e-3]),
            abs_dist_U=np.array([1e-2]),
            abs_dist_M=np.array([1e-2]),
            execution_time=1.0,
            output_dir_base=tmpdir,
        )

        loaded_data = load_experiment_data(filepath)

        assert loaded_data is not None
        np.testing.assert_array_almost_equal(loaded_data["U_solution"], U_solution)
        np.testing.assert_array_almost_equal(loaded_data["M_solution"], M_solution)


@pytest.mark.unit
def test_load_experiment_data_nonexistent_file():
    """Test loading from nonexistent file returns None."""
    loaded_data = load_experiment_data("/nonexistent/path/file.npz")

    assert loaded_data is None


@pytest.mark.unit
def test_load_experiment_data_problem_params():
    """Test that problem parameters are correctly loaded."""
    problem = create_mock_problem()
    U_solution = np.random.rand(21, 11)
    M_solution = np.random.rand(21, 11)

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = save_experiment_data(
            problem=problem,
            U_solution=U_solution,
            M_solution=M_solution,
            solver_name="ParamsTest",
            iterations_run=1,
            rel_dist_U=np.array([1e-3]),
            rel_dist_M=np.array([1e-3]),
            abs_dist_U=np.array([1e-2]),
            abs_dist_M=np.array([1e-2]),
            execution_time=1.0,
            output_dir_base=tmpdir,
        )

        loaded_data = load_experiment_data(filepath)

        assert loaded_data is not None
        params = loaded_data["problem_params"]
        assert params["Nx"] == 10  # Number of space INTERVALS
        assert params["Nt"] == 20  # Number of time INTERVALS
        assert params["T"] == pytest.approx(1.0)
        assert params["sigma"] == pytest.approx(0.1)


# ===================================================================
# Test Batch Loading from Directory
# ===================================================================


@pytest.mark.unit
def test_load_experiments_from_dir_multiple_files():
    """Test loading multiple experiment files from directory."""
    problem = create_mock_problem()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save multiple experiments
        for i in range(3):
            save_experiment_data(
                problem=problem,
                U_solution=np.random.rand(21, 11),
                M_solution=np.random.rand(21, 11),
                solver_name=f"Solver{i}",
                iterations_run=i + 1,
                rel_dist_U=np.array([1e-3]),
                rel_dist_M=np.array([1e-3]),
                abs_dist_U=np.array([1e-2]),
                abs_dist_M=np.array([1e-2]),
                execution_time=float(i),
                output_dir_base=tmpdir,
            )

        # Load all experiments
        solver0_dir = Path(tmpdir) / "Solver0"
        experiments = load_experiments_from_dir(str(solver0_dir))

        assert len(experiments) >= 1  # At least Solver0's file


@pytest.mark.unit
def test_load_experiments_from_dir_nonexistent():
    """Test loading from nonexistent directory."""
    experiments = load_experiments_from_dir("/nonexistent/directory")

    assert experiments == []


@pytest.mark.unit
def test_load_experiments_from_dir_empty():
    """Test loading from empty directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        experiments = load_experiments_from_dir(tmpdir)

        assert experiments == []


@pytest.mark.unit
def test_load_experiments_from_dir_mixed_files():
    """Test loading with mixed file types in directory."""
    problem = create_mock_problem()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save one experiment
        save_experiment_data(
            problem=problem,
            U_solution=np.random.rand(21, 11),
            M_solution=np.random.rand(21, 11),
            solver_name="MixedTest",
            iterations_run=1,
            rel_dist_U=np.array([1e-3]),
            rel_dist_M=np.array([1e-3]),
            abs_dist_U=np.array([1e-2]),
            abs_dist_M=np.array([1e-2]),
            execution_time=1.0,
            output_dir_base=tmpdir,
        )

        # Add non-npz files
        solver_dir = Path(tmpdir) / "MixedTest"
        (solver_dir / "readme.txt").write_text("test")
        (solver_dir / "data.json").write_text("{}")

        # Should only load .npz files
        experiments = load_experiments_from_dir(str(solver_dir))

        assert len(experiments) == 1
        assert experiments[0] is not None


# ===================================================================
# Test Plotting Functions
# ===================================================================


def create_mock_experiment_data():
    """Create mock experiment data for plotting tests."""
    return {
        "solver_name": "TestSolver",
        "timestamp": "20240101-120000",
        "tSpace": np.linspace(0, 1.0, 21),
        "xSpace": np.linspace(0.0, 1.0, 11),
        "total_mass_vs_time": np.ones(21) * 0.95,
        "U_solution": np.random.rand(21, 11),
        "M_solution": np.random.rand(21, 11),
    }


@pytest.mark.unit
def test_plot_comparison_total_mass_no_viz():
    """Test total mass comparison plot without visualization system."""
    exp_data1 = create_mock_experiment_data()
    exp_data2 = create_mock_experiment_data()
    exp_data2["solver_name"] = "TestSolver2"

    with patch("mfgarchon.utils.experiment_manager.VISUALIZATION_AVAILABLE", False), patch("matplotlib.pyplot.show"):
        result = plot_comparison_total_mass([exp_data1, exp_data2])
        assert result is None


@pytest.mark.unit
def test_plot_comparison_total_mass_with_save():
    """Test total mass comparison plot with file saving."""
    exp_data = create_mock_experiment_data()

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "total_mass.png"

        with (
            patch("mfgarchon.utils.experiment_manager.VISUALIZATION_AVAILABLE", False),
            patch("matplotlib.pyplot.show"),
        ):
            plot_comparison_total_mass([exp_data], save_to_file=str(save_path))

        # File should be created
        assert save_path.exists()


@pytest.mark.unit
def test_plot_comparison_final_m_basic():
    """Test final density comparison plot."""
    exp_data = create_mock_experiment_data()

    with patch("mfgarchon.utils.experiment_manager.VISUALIZATION_AVAILABLE", False), patch("matplotlib.pyplot.show"):
        result = plot_comparison_final_m([exp_data])
        assert result is None


@pytest.mark.unit
def test_plot_comparison_final_m_invalid_shape():
    """Test final density plot with invalid M_solution shape."""
    exp_data = create_mock_experiment_data()
    exp_data["M_solution"] = np.array([1.0, 2.0, 3.0])  # Wrong shape (1D)

    with patch("mfgarchon.utils.experiment_manager.VISUALIZATION_AVAILABLE", False), patch("matplotlib.pyplot.show"):
        # Should handle gracefully and print warning
        result = plot_comparison_final_m([exp_data])
        assert result is None


@pytest.mark.unit
def test_plot_comparison_initial_U_basic():
    """Test initial value function comparison plot."""
    exp_data = create_mock_experiment_data()

    with patch("mfgarchon.utils.experiment_manager.VISUALIZATION_AVAILABLE", False), patch("matplotlib.pyplot.show"):
        result = plot_comparison_initial_U([exp_data])
        assert result is None


@pytest.mark.unit
def test_plot_comparison_initial_U_with_save():
    """Test initial U comparison plot with file saving."""
    exp_data = create_mock_experiment_data()

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "initial_U.png"

        with (
            patch("mfgarchon.utils.experiment_manager.VISUALIZATION_AVAILABLE", False),
            patch("matplotlib.pyplot.show"),
        ):
            plot_comparison_initial_U([exp_data], save_to_file=str(save_path))

        assert save_path.exists()


@pytest.mark.unit
def test_plot_comparison_U_slice_basic():
    """Test U slice comparison plot at specific time."""
    exp_data = create_mock_experiment_data()

    with patch("mfgarchon.utils.experiment_manager.VISUALIZATION_AVAILABLE", False), patch("matplotlib.pyplot.show"):
        result = plot_comparison_U_slice([exp_data], time_index=10)
        assert result is None


@pytest.mark.unit
def test_plot_comparison_U_slice_out_of_bounds():
    """Test U slice plot with out-of-bounds time index."""
    exp_data = create_mock_experiment_data()

    with patch("mfgarchon.utils.experiment_manager.VISUALIZATION_AVAILABLE", False), patch("matplotlib.pyplot.show"):
        # Should handle gracefully and print warning
        result = plot_comparison_U_slice([exp_data], time_index=100)
        assert result is None


@pytest.mark.unit
def test_plot_comparison_U_slice_with_save():
    """Test U slice plot with file saving."""
    exp_data = create_mock_experiment_data()

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "U_slice.png"

        with (
            patch("mfgarchon.utils.experiment_manager.VISUALIZATION_AVAILABLE", False),
            patch("matplotlib.pyplot.show"),
        ):
            plot_comparison_U_slice([exp_data], time_index=5, save_to_file=str(save_path))

        assert save_path.exists()


# ===================================================================
# Test Error Handling
# ===================================================================


@pytest.mark.unit
def test_save_experiment_data_handles_exception():
    """Test that save_experiment_data handles exceptions gracefully."""
    problem = create_mock_problem()
    U_solution = np.random.rand(21, 11)
    M_solution = np.random.rand(21, 11)

    with (
        tempfile.TemporaryDirectory() as tmpdir,
        patch("numpy.savez_compressed", side_effect=PermissionError("Cannot write")),
    ):
        filepath = save_experiment_data(
            problem=problem,
            U_solution=U_solution,
            M_solution=M_solution,
            solver_name="ErrorTest",
            iterations_run=1,
            rel_dist_U=np.array([1e-3]),
            rel_dist_M=np.array([1e-3]),
            abs_dist_U=np.array([1e-2]),
            abs_dist_M=np.array([1e-2]),
            execution_time=1.0,
            output_dir_base=tmpdir,
        )

        # Should return empty string on error
        assert filepath == ""


@pytest.mark.unit
def test_load_experiment_data_corrupted_file():
    """Test loading corrupted/invalid npz file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        corrupted_file = Path(tmpdir) / "corrupted.npz"
        corrupted_file.write_text("not a valid npz file")

        loaded_data = load_experiment_data(str(corrupted_file))

        assert loaded_data is None


# ===================================================================
# Test Integration
# ===================================================================


@pytest.mark.unit
def test_save_load_roundtrip():
    """Test complete save and load roundtrip."""
    problem = create_mock_problem()
    U_solution = np.random.rand(21, 11)
    M_solution = np.random.rand(21, 11)
    rel_dist_U = np.geomspace(1, 1e-5, 10)
    rel_dist_M = np.geomspace(1, 1e-5, 10)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save
        filepath = save_experiment_data(
            problem=problem,
            U_solution=U_solution,
            M_solution=M_solution,
            solver_name="RoundtripTest",
            iterations_run=10,
            rel_dist_U=rel_dist_U,
            rel_dist_M=rel_dist_M,
            abs_dist_U=rel_dist_U * 10,
            abs_dist_M=rel_dist_M * 5,
            execution_time=12.34,
            output_dir_base=tmpdir,
            additional_params={"test_param": 42},
        )

        # Load
        loaded_data = load_experiment_data(filepath)

        # Verify all data matches
        assert loaded_data is not None
        np.testing.assert_array_almost_equal(loaded_data["U_solution"], U_solution)
        np.testing.assert_array_almost_equal(loaded_data["M_solution"], M_solution)
        np.testing.assert_array_almost_equal(loaded_data["rel_dist_U"], rel_dist_U)
        np.testing.assert_array_almost_equal(loaded_data["rel_dist_M"], rel_dist_M)
        assert loaded_data["iterations_run"] == 10
        assert loaded_data["execution_time"] == pytest.approx(12.34)
        assert loaded_data["problem_params"]["test_param"] == 42
