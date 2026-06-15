from __future__ import annotations

import os
from datetime import datetime

# Assuming your MFGProblem class is importable for type hinting
# from ..core.mfg_problem import MFGProblem # Adjust path if necessary
from typing import TYPE_CHECKING, Any

import numpy as np

# Modern visualization system
try:
    from mfgarchon.visualization import (  # noqa: F401
        create_visualization_manager,
        modern_plot_convergence,
        plot_convergence,
    )

    VISUALIZATION_AVAILABLE = True
except ImportError:
    # Fallback to matplotlib if visualization system not available

    VISUALIZATION_AVAILABLE = False

if TYPE_CHECKING:
    from mfgarchon.core.mfg_problem import MFGProblem


def calculate_total_mass(M: np.ndarray, dx: float) -> np.ndarray:
    """Calculates the total mass for each time step."""
    if dx == 0:  # Handle single-point case
        return np.sum(M, axis=1) if M.ndim > 1 else np.array([np.sum(M)])
    return np.sum(M * dx, axis=1)


def save_experiment_data(
    problem: MFGProblem,
    U_solution: np.ndarray,
    M_solution: np.ndarray,
    solver_name: str,
    iterations_run: int,
    rel_dist_U: np.ndarray,
    rel_dist_M: np.ndarray,
    abs_dist_U: np.ndarray,  # Assuming you might have absolute distances too
    abs_dist_M: np.ndarray,
    execution_time: float,
    output_dir_base: str = "mfg_results",
    additional_params: dict[str, Any] | None = None,
) -> str:
    """
    Saves the results and parameters of an MFG experiment to an .npz file.

    Args:
        problem (MFGProblem): The instance of the MFGProblem solved.
        U_solution (np.ndarray): The computed value function U(t,x).
        M_solution (np.ndarray): The computed density M(t,x).
        solver_name (str): Name of the solver used (e.g., "HJB-FDM_FP-FDM").
        iterations_run (int): Number of Picard iterations performed.
        rel_dist_U (np.ndarray): Relative L2 distance for U over iterations.
        rel_dist_M (np.ndarray): Relative L2 distance for M over iterations.
        abs_dist_U (np.ndarray): Absolute L2 distance for U over iterations.
        abs_dist_M (np.ndarray): Absolute L2 distance for M over iterations.
        execution_time (float): Total execution time for the solver.
        output_dir_base (str): Base directory to save results.
        additional_params (Optional[Dict[str, Any]]): Any other parameters to save.

    Returns:
        str: The path to the saved data file.
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Create a directory for the specific solver method if it doesn't exist
    solver_output_dir = os.path.join(output_dir_base, solver_name)
    os.makedirs(solver_output_dir, exist_ok=True)

    # Construct filename with key parameters for easy identification
    # (Ensure parameters are sanitized for filenames if they contain special chars)
    grid_shape = problem.geometry.get_grid_shape()
    Nx_intervals = grid_shape[0] - 1
    filename_parts = [
        f"T{problem.T:.1f}",
        f"Nx{Nx_intervals}",
        f"Nt{problem.Nt}",
        f"sig{problem.sigma:.1e}",
        f"ct{problem.coupling_coefficient:.1e}",
        solver_name.replace("/", "-"),  # Replace slashes if any
        timestamp,
    ]
    filename = "_".join(filename_parts) + ".npz"
    filepath = os.path.join(solver_output_dir, filename)

    dx = problem.geometry.get_grid_spacing()[0]
    total_mass_vs_time = calculate_total_mass(M_solution, dx)

    bounds = problem.geometry.get_bounds()
    problem_params_dict = {
        "xmin": bounds[0][0],
        "xmax": bounds[1][0],
        "Nx": Nx_intervals,
        "dx": dx,
        "T": problem.T,
        "Nt": problem.Nt,
        "dt": problem.dt,
        "sigma": problem.sigma,
        "coupling_coefficient": problem.coupling_coefficient,
        # Add other problem-specific parameters if they exist and are relevant
        # e.g., "potential_type": problem.potential_type if hasattr(problem, 'potential_type') else "default"
    }
    if additional_params:
        problem_params_dict.update(additional_params)

    try:
        np.savez_compressed(
            filepath,
            U_solution=U_solution,
            M_solution=M_solution,
            total_mass_vs_time=total_mass_vs_time,
            tSpace=problem.tSpace,
            xSpace=problem.geometry.get_spatial_grid(),
            problem_params=problem_params_dict,  # type: ignore[arg-type]
            solver_name=solver_name,
            timestamp=timestamp,
            iterations_run=iterations_run,
            rel_dist_U=rel_dist_U,
            rel_dist_M=rel_dist_M,
            abs_dist_U=abs_dist_U,
            abs_dist_M=abs_dist_M,
            execution_time=execution_time,
        )
        print(f"Experiment data saved to: {filepath}")
        return filepath
    except Exception as e:
        print(f"Error saving experiment data to {filepath}: {e}")
        return ""


def load_experiment_data(filepath: str) -> dict[str, Any] | None:
    """
    Loads experiment data from an .npz file.

    Args:
        filepath (str): Path to the .npz file.

    Returns:
        Optional[Dict[str, Any]]: A dictionary containing the loaded data,
                                   or None if loading fails.
    """
    try:
        data = np.load(filepath, allow_pickle=True)
        # Convert problem_params back to a standard dict if it's an array object
        loaded_data = {key: data[key] for key in data.files}
        if "problem_params" in loaded_data and isinstance(loaded_data["problem_params"], np.ndarray):
            if loaded_data["problem_params"].size == 1 and isinstance(loaded_data["problem_params"].item(), dict):
                loaded_data["problem_params"] = loaded_data["problem_params"].item()
        print(f"Experiment data loaded from: {filepath}")
        return loaded_data
    except Exception as e:
        print(f"Error loading experiment data from {filepath}: {e}")
        return None


def load_experiments_from_dir(directory_path: str) -> list[dict[str, Any]]:
    """
    Loads all .npz experiment files from a given directory.
    """
    loaded_experiments: list[dict[str, Any]] = []
    if not os.path.isdir(directory_path):
        print(f"Directory not found: {directory_path}")
        return loaded_experiments

    for filename in os.listdir(directory_path):
        if filename.endswith(".npz"):
            filepath = os.path.join(directory_path, filename)
            data = load_experiment_data(filepath)
            if data:
                loaded_experiments.append(data)
    return loaded_experiments


# --- Comparison Plotting Functions ---


def plot_comparison_total_mass(
    experiment_data_list: list[dict[str, Any]],
    title_suffix: str = "",
    save_to_file: str | None = None,
):
    """Plots total mass vs. time for multiple experiments on the same axes."""
    if VISUALIZATION_AVAILABLE:
        # Use modern visualization system
        viz_manager = create_visualization_manager(prefer_plotly=False)  # Use matplotlib for comparison plots

        # Prepare data for plotting
        plot_data = {}
        for exp_data in experiment_data_list:
            label = f"{exp_data.get('solver_name', 'Unknown')}_{exp_data.get('timestamp', '')[-6:]}"
            plot_data[label] = {
                "x": exp_data["tSpace"],
                "y": exp_data["total_mass_vs_time"],
            }

        try:
            fig = viz_manager._create_line_comparison_plot(
                plot_data,
                title=f"Comparison of Total Mass Evolution {title_suffix}",
                xlabel="t",
                ylabel=r"Total Mass $\int m(t) dx$",
                ylim=(0, 1.1),
                save_path=save_to_file,
            )
            return fig
        except (ImportError, AttributeError, TypeError, ValueError):
            # Fallback to matplotlib if Plotly not available or plotting fails
            pass

    # Fallback to matplotlib
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    for exp_data in experiment_data_list:
        label = f"{exp_data.get('solver_name', 'Unknown')}_{exp_data.get('timestamp', '')[-6:]}"
        plt.plot(exp_data["tSpace"], exp_data["total_mass_vs_time"], label=label)

    plt.xlabel("t")
    plt.ylabel(r"Total Mass $\int m(t) dx$")
    plt.title(f"Comparison of Total Mass Evolution {title_suffix}")
    plt.legend(loc="best", fontsize="small")
    plt.grid(True)
    plt.ylim(0, 1.1)  # Adjust as needed, or make dynamic
    if save_to_file:
        plt.savefig(save_to_file)
        print(f"Comparison plot saved to: {save_to_file}")
    plt.show()


def plot_comparison_final_m(
    experiment_data_list: list[dict[str, Any]],
    title_suffix: str = "",
    save_to_file: str | None = None,
):
    """Plots the final density M(T,x) for multiple experiments."""
    if VISUALIZATION_AVAILABLE:
        # Use modern visualization system
        viz_manager = create_visualization_manager(prefer_plotly=False)

        # Prepare data for plotting
        plot_data = {}
        for exp_data in experiment_data_list:
            label = f"{exp_data.get('solver_name', 'Unknown')}_{exp_data.get('timestamp', '')[-6:]}"
            if exp_data["M_solution"].ndim == 2 and exp_data["M_solution"].shape[0] > 0:
                plot_data[label] = {
                    "x": exp_data["xSpace"],
                    "y": exp_data["M_solution"][-1, :],  # M_solution[-1] is M(T,x)
                }
            else:
                print(f"Warning: M_solution for {label} is not in expected 2D shape or is empty.")

        try:
            fig = viz_manager._create_line_comparison_plot(
                plot_data,
                title=f"Comparison of Final Densities m(T,x) {title_suffix}",
                xlabel="x",
                ylabel="m(T,x)",
                save_path=save_to_file,
            )
            return fig
        except (ImportError, AttributeError, TypeError, ValueError):
            # Fallback to matplotlib if Plotly not available or plotting fails
            pass

    # Fallback to matplotlib
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    for exp_data in experiment_data_list:
        label = f"{exp_data.get('solver_name', 'Unknown')}_{exp_data.get('timestamp', '')[-6:]}"
        if exp_data["M_solution"].ndim == 2 and exp_data["M_solution"].shape[0] > 0:
            plt.plot(exp_data["xSpace"], exp_data["M_solution"][-1, :], label=label)  # M_solution[-1] is M(T,x)
        else:
            print(f"Warning: M_solution for {label} is not in expected 2D shape or is empty.")

    plt.xlabel("x")
    plt.ylabel("m(T,x)")
    plt.title(f"Comparison of Final Densities m(T,x) {title_suffix}")
    plt.legend(loc="best", fontsize="small")
    plt.grid(True)
    if save_to_file:
        plt.savefig(save_to_file)
        print(f"Comparison plot saved to: {save_to_file}")
    plt.show()


def plot_comparison_initial_U(
    experiment_data_list: list[dict[str, Any]],
    title_suffix: str = "",
    save_to_file: str | None = None,
):
    """Plots the initial value function U(0,x) for multiple experiments."""
    if VISUALIZATION_AVAILABLE:
        # Use modern visualization system
        viz_manager = create_visualization_manager(prefer_plotly=False)

        # Prepare data for plotting
        plot_data = {}
        for exp_data in experiment_data_list:
            label = f"{exp_data.get('solver_name', 'Unknown')}_{exp_data.get('timestamp', '')[-6:]}"
            if exp_data["U_solution"].ndim == 2 and exp_data["U_solution"].shape[0] > 0:
                plot_data[label] = {
                    "x": exp_data["xSpace"],
                    "y": exp_data["U_solution"][0, :],  # U_solution[0] is U(0,x)
                }
            else:
                print(f"Warning: U_solution for {label} is not in expected 2D shape or is empty.")

        try:
            fig = viz_manager._create_line_comparison_plot(
                plot_data,
                title=f"Comparison of Initial Values U(0,x) {title_suffix}",
                xlabel="x",
                ylabel="U(0,x)",
                save_path=save_to_file,
            )
            return fig
        except (ImportError, AttributeError, TypeError, ValueError):
            # Fallback to matplotlib if Plotly not available or plotting fails
            pass

    # Fallback to matplotlib
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    for exp_data in experiment_data_list:
        label = f"{exp_data.get('solver_name', 'Unknown')}_{exp_data.get('timestamp', '')[-6:]}"
        if exp_data["U_solution"].ndim == 2 and exp_data["U_solution"].shape[0] > 0:
            plt.plot(exp_data["xSpace"], exp_data["U_solution"][0, :], label=label)  # U_solution[0] is U(0,x)
        else:
            print(f"Warning: U_solution for {label} is not in expected 2D shape or is empty.")

    plt.xlabel("x")
    plt.ylabel("U(0,x)")
    plt.title(f"Comparison of Initial Values U(0,x) {title_suffix}")
    plt.legend(loc="best", fontsize="small")
    plt.grid(True)
    if save_to_file:
        plt.savefig(save_to_file)
        print(f"Comparison plot saved to: {save_to_file}")
    plt.show()


# Example of a more detailed comparison plot: U at a specific time slice
def plot_comparison_U_slice(
    experiment_data_list: list[dict[str, Any]],
    time_index: int,
    title_suffix: str = "",
    save_to_file: str | None = None,
):
    """Plots U(t_k, x) for a specific time_index k for multiple experiments."""
    actual_time = -1

    if VISUALIZATION_AVAILABLE:
        # Use modern visualization system
        viz_manager = create_visualization_manager(prefer_plotly=False)

        # Prepare data for plotting
        plot_data = {}
        for exp_data in experiment_data_list:
            if 0 <= time_index < exp_data["U_solution"].shape[0]:
                actual_time = exp_data["tSpace"][time_index]
                label = f"{exp_data.get('solver_name', 'Unknown')}_{exp_data.get('timestamp', '')[-6:]}"
                plot_data[label] = {
                    "x": exp_data["xSpace"],
                    "y": exp_data["U_solution"][time_index, :],
                }
            else:
                print(
                    f"Warning: time_index {time_index} out of bounds for experiment {exp_data.get('solver_name', 'Unknown')}"
                )

        try:
            fig = viz_manager._create_line_comparison_plot(
                plot_data,
                title=f"Comparison of U(t,x) at t={actual_time:.2f} {title_suffix}",
                xlabel="x",
                ylabel=f"U(t={actual_time:.2f}, x)",
                save_path=save_to_file,
            )
            return fig
        except (ImportError, AttributeError, TypeError, ValueError):
            # Fallback to matplotlib if Plotly not available or plotting fails
            pass

    # Fallback to matplotlib
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    for exp_data in experiment_data_list:
        if 0 <= time_index < exp_data["U_solution"].shape[0]:
            actual_time = exp_data["tSpace"][time_index]
            label = f"{exp_data.get('solver_name', 'Unknown')}_{exp_data.get('timestamp', '')[-6:]}"
            plt.plot(exp_data["xSpace"], exp_data["U_solution"][time_index, :], label=label)
        else:
            print(
                f"Warning: time_index {time_index} out of bounds for experiment {exp_data.get('solver_name', 'Unknown')}"
            )

    plt.xlabel("x")
    plt.ylabel(f"U(t={actual_time:.2f}, x)")
    plt.title(f"Comparison of U(t,x) at t={actual_time:.2f} {title_suffix}")
    plt.legend(loc="best", fontsize="small")
    plt.grid(True)
    if save_to_file:
        plt.savefig(save_to_file)
        print(f"Comparison plot saved to: {save_to_file}")
    plt.show()


if __name__ == "__main__":
    # Example usage (normally this would be in a separate script or notebook)
    print("--- Example: Testing Experiment Manager ---")

    # Create a dummy MFGProblem instance (requires MFGProblem to be importable)
    # from mfgarchon.core.mfg_problem import MFGProblem
    # dummy_problem_params = {"Nx": 11, "Nt": 11, "T": 0.1, "sigma":1.0, "coupling_coefficient":0.5}
    # dummy_problem = MFGProblem(**dummy_problem_params)

    # Dummy data
    # U_sol = np.random.rand(dummy_problem.Nt, dummy_problem.Nx)
    # M_sol = np.random.rand(dummy_problem.Nt, dummy_problem.Nx)
    # M_sol[0,:] = dummy_problem.get_initial_m() # Make it somewhat consistent
    # for t in range(dummy_problem.Nt): M_sol[t,:] /= (np.sum(M_sol[t,:])*dummy_problem.dx if dummy_problem.dx >0 else 1)

    # iterations = 10
    # rel_U = np.geomspace(1, 1e-5, iterations)
    # rel_M = np.geomspace(1, 1e-5, iterations)
    # abs_U = rel_U * 10
    # abs_M = rel_M * 5
    # exec_time = 12.34

    # Save dummy data
    # filepath1 = save_experiment_data(
    #     problem=dummy_problem,
    #     U_solution=U_sol,
    #     M_solution=M_sol,
    #     solver_name="TestSolver1",
    #     iterations_run=iterations,
    #     rel_dist_U=rel_U, rel_dist_M=rel_M,
    #     abs_dist_U=abs_U, abs_dist_M=abs_M,
    #     execution_time=exec_time,
    #     output_dir_base="test_results",
    #     additional_params={"damping": 0.5}
    # )
    # U_sol2 = U_sol * 0.9
    # M_sol2 = M_sol * 1.1
    # M_sol2[0,:] = dummy_problem.get_initial_m()
    # for t in range(dummy_problem.Nt): M_sol2[t,:] /= (np.sum(M_sol2[t,:])*dummy_problem.dx if dummy_problem.dx >0 else 1)

    # filepath2 = save_experiment_data(
    #     problem=dummy_problem,
    #     U_solution=U_sol2,
    #     M_solution=M_sol2,
    #     solver_name="TestSolver1", # Same solver, different run
    #     iterations_run=iterations,
    #     rel_dist_U=rel_U*0.8, rel_dist_M=rel_M*0.9,
    #     abs_dist_U=abs_U*0.8, abs_dist_M=abs_M*0.9,
    #     execution_time=15.67,
    #     output_dir_base="test_results",
    #     additional_params={"damping": 0.7}
    # )

    # Load data
    # exp1_data = load_experiment_data(filepath1)
    # exp2_data = load_experiment_data(filepath2)

    # experiments = []
    # if exp1_data: experiments.append(exp1_data)
    # if exp2_data: experiments.append(exp2_data)

    # if experiments:
    #     plot_comparison_total_mass(experiments, title_suffix=" (Test Solvers)")
    #     plot_comparison_final_m(experiments, title_suffix=" (Test Solvers)")
    #     plot_comparison_initial_U(experiments, title_suffix=" (Test Solvers)")
    #     if dummy_problem.Nt > 0:
    #         plot_comparison_U_slice(experiments, time_index=dummy_problem.Nt // 2, title_suffix=" (Mid-Time, Test)")
    # else:
    #     print("No experiment data loaded for plotting.")

    print("Experiment manager module can be tested by uncommenting and running the example usage section.")
    print("Ensure MFGProblem is importable if running this block.")
