"""A package in `[project].dependencies` is required: it is imported plainly, and nothing decides what happens without it.

User rulings, 2026-11-10 and 2026-11-11 (#2930): there is one base install, `pyproject.toml`'s `[project].dependencies`
is its one list, and a missing package is Python's own `ModuleNotFoundError`. A package outside the list may be guarded.

**The population** is read from pyproject, never written here. Each distribution is mapped to the importable top-level
names it installs by `importlib.metadata.packages_distributions()`, and a distribution with none fails the test rather
than shrinking it.

**The structural guard** walks `mfgarchon/`, `examples/`, `benchmarks/` and `tests/` by AST, and counts rows by (file,
enclosing function or class, package, kind):

- ``guard``: an import of a required package, or an `importlib.import_module` or `__import__` of one by a constant
  name, whose nearest enclosing `try` within its own function has a handler that catches `ImportError`,
  `ModuleNotFoundError`, `Exception` or `BaseException`, or catches everything; or that sits in a
  `with contextlib.suppress(...)` of one of those. A handler that ends in a bare `raise`, with no `return`, lets the
  failure through and decides nothing. A `try` around a `def` guards the definition, not the import in its body.
- ``name-catch``: inside `except ... as err`, a comparison of `err.name` with a required package, or with a tuple, list
  or set holding one.

and, in `tests/` only:

- ``importorskip``: `pytest.importorskip` of a required package, by its first positional argument;
- ``sys.modules``: a write to `sys.modules` for one, by subscript or by `setitem(sys.modules, ...)`.

`ALLOWED` lists the rows that #2930's later stages remove, each with its stage and its count, and one row that stays for
another reason. A row found more often than listed fails, and so does a row listed more often than found, so each stage
lowers or deletes its lines in its own diff.

**What it cannot see**, each with an example:

- a package named through a variable: `name = "scipy"; import_module(name)`;
- a guard in another file around a module that imports the package at top level;
- an outer deciding `try` around an inner one that does not decide: `try: (try: import yaml / finally: pass) / except
  ImportError: yaml = None`;
- an exception tuple held in a name: `errors = (ImportError,); try: import scipy / except errors: ...`;
- a name-catch spelled otherwise: `getattr(err, "name")`, `"psutil" in str(err)`;
- other ways a test removes a package: `mock.patch.dict(sys.modules, {"scipy": None})`, `sys.modules.update(...)`,
  `importorskip(modname="scipy")`, an aliased `sys`;
- a fallback reached without a `try`, such as `importlib.util.find_spec("scipy")`, which the census of later stages reads;
- code that is not a `.py` file: fenced code in Markdown, and notebooks such as
  `examples/basic/core_infrastructure/mfg_working_demo.ipynb`.
"""

from __future__ import annotations

import ast
import importlib.metadata
import os
import re
import subprocess
import sys
import tomllib
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CODE_DIRS = ("mfgarchon", "examples", "benchmarks")
CATCHERS = {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"}

#: (file, enclosing scope, package, kind) -> (the stage of #2930 that removes it, or why it stays; how many there are).
ALLOWED: dict[tuple[str, str, str, str], tuple[str, int]] = {
    # #2930 stage B
    ("mfgarchon/alg/numerical/density_estimation.py", "<module>", "scipy", "guard"): ("#2930 stage B", 1),
    ("mfgarchon/alg/numerical/fp_solvers/fp_particle.py", "<module>", "scipy", "guard"): ("#2930 stage B", 1),
    ("mfgarchon/alg/optimization/optimal_transport/__init__.py", "<module>", "scipy", "guard"): ("#2930 stage B", 2),
    (
        "mfgarchon/alg/optimization/optimal_transport/wasserstein_solver.py",
        "WassersteinMFGSolver._check_dependencies",
        "scipy",
        "guard",
    ): ("#2930 stage B", 1),
    ("mfgarchon/alg/optimization/variational_problem.py", "<module>", "scipy", "guard"): ("#2930 stage B", 2),
    ("mfgarchon/alg/optimization/variational_solvers/primal_dual_solver.py", "<module>", "scipy", "guard"): (
        "#2930 stage B",
        1,
    ),
    ("mfgarchon/alg/optimization/variational_solvers/variational_mfg_solver.py", "<module>", "scipy", "guard"): (
        "#2930 stage B",
        1,
    ),
    ("mfgarchon/core/hamiltonian.py", "DualHamiltonian.__call__", "scipy", "guard"): ("#2930 stage B", 1),
    ("mfgarchon/core/hamiltonian.py", "DualHamiltonian._find_optimal_alpha", "scipy", "guard"): ("#2930 stage B", 1),
    ("mfgarchon/core/hamiltonian.py", "DualLagrangian.__call__", "scipy", "guard"): ("#2930 stage B", 1),
    ("mfgarchon/core/hamiltonian.py", "DualLagrangian._find_optimal_p", "scipy", "guard"): ("#2930 stage B", 1),
    ("mfgarchon/geometry/graph/maze_cellular_automata.py", "<module>", "scipy", "guard"): ("#2930 stage B", 1),
    ("mfgarchon/geometry/graph/maze_postprocessing.py", "<module>", "scipy", "guard"): ("#2930 stage B", 1),
    ("mfgarchon/geometry/graph/maze_utils.py", "<module>", "scipy", "guard"): ("#2930 stage B", 1),
    ("mfgarchon/geometry/graph/maze_voronoi.py", "<module>", "scipy", "guard"): ("#2930 stage B", 1),
    (
        "mfgarchon/geometry/grids/tensor_grid.py",
        "TensorProductGrid.get_interpolator.interpolate_linear",
        "scipy",
        "guard",
    ): ("#2930 stage B", 1),
    ("mfgarchon/hooks/visualization.py", "AnimationHook.__init__", "matplotlib", "guard"): ("#2930 stage B", 2),
    ("mfgarchon/hooks/visualization.py", "PlottingHook.__init__", "matplotlib", "guard"): ("#2930 stage B", 2),
    ("mfgarchon/operators/interpolation/projection.py", "_mesh_to_grid_delaunay", "scipy", "guard"): (
        "#2930 stage B",
        2,
    ),
    ("mfgarchon/types/state.py", "ConvergenceInfo.plot_convergence", "matplotlib", "guard"): ("#2930 stage B", 1),
    ("mfgarchon/utils/functional_calculus.py", "create_particle_measure", "scipy", "guard"): ("#2930 stage B", 1),
    ("mfgarchon/utils/numerical/monotonicity_stats.py", "verify_m_matrix_property", "scipy", "guard"): (
        "#2930 stage B",
        1,
    ),
    ("mfgarchon/utils/numerical/particle/kde_boundary.py", "<module>", "scipy", "guard"): ("#2930 stage B", 3),
    ("mfgarchon/utils/numerical/particle/sampling.py", "QuasiMCSampler._init_sequence_generator", "scipy", "guard"): (
        "#2930 stage B",
        1,
    ),
    ("mfgarchon/utils/numpy_compat.py", "<module>", "scipy", "guard"): ("#2930 stage B", 1),
    (
        "mfgarchon/utils/performance/optimization.py",
        "SparseMatrixOptimizer.optimize_matrix_structure",
        "scipy",
        "guard",
    ): ("#2930 stage B", 1),
    ("mfgarchon/visualization/convergence_plots.py", "<module>", "matplotlib", "guard"): ("#2930 stage B", 2),
    ("tests/conftest.py", "pytest_terminal_summary", "scipy", "guard"): ("#2930 stage B", 1),
    ("tests/integration/test_torch_kde.py", "<module>", "scipy", "guard"): ("#2930 stage B", 1),
    ("tests/unit/test_alg/test_density_estimation.py", "<module>", "scipy", "guard"): ("#2930 stage B", 1),
    ("tests/unit/test_core/test_legendre_sign_convention_2375.py", "_no_scipy_optimize", "scipy", "sys.modules"): (
        "#2930 stage B",
        2,
    ),
    ("tests/unit/test_utils/test_monte_carlo.py", "test_monte_carlo_integrate_quasi", "scipy", "importorskip"): (
        "#2930 stage B",
        1,
    ),
    ("tests/unit/test_utils/test_monte_carlo.py", "test_quasi_mc_sampler_custom_domain", "scipy", "importorskip"): (
        "#2930 stage B",
        1,
    ),
    ("tests/unit/test_utils/test_monte_carlo.py", "test_quasi_mc_sampler_halton", "scipy", "importorskip"): (
        "#2930 stage B",
        1,
    ),
    ("tests/unit/test_utils/test_monte_carlo.py", "test_quasi_mc_sampler_latin_hypercube", "scipy", "importorskip"): (
        "#2930 stage B",
        1,
    ),
    # #2930 stage C
    ("benchmarks/particle_kde_gpu_benchmark.py", "<module>", "scipy", "guard"): ("#2930 stage C", 1),
    ("benchmarks/performance_tracker.py", "<module>", "psutil", "guard"): ("#2930 stage C", 1),
    ("benchmarks/run_benchmarks.py", "<module>", "psutil", "guard"): ("#2930 stage C", 1),
    (
        "examples/advanced/geometry_advanced/dual_geometry_fem_mesh.py",
        "register_specialized_fem_projections.mesh_to_grid_delaunay",
        "scipy",
        "guard",
    ): ("#2930 stage C", 1),
    ("examples/advanced/solvers_advanced/semi_lagrangian_2d_enhancements.py", "<module>", "matplotlib", "guard"): (
        "#2930 stage C",
        2,
    ),
    ("examples/advanced/visualization/visualize_2d_density_evolution.py", "<module>", "matplotlib", "guard"): (
        "#2930 stage C",
        2,
    ),
    ("examples/applications/networks/network_mfg_comparison_example.py", "<module>", "plotly", "guard"): (
        "#2930 stage C",
        3,
    ),
    ("examples/basic/core_infrastructure/mfg_working_demo.py", "<module>", "plotly", "guard"): ("#2930 stage C", 2),
    ("examples/basic/solvers/policy_iteration_lq_demo.py", "<module>", "matplotlib", "guard"): ("#2930 stage C", 2),
    ("examples/basic/utilities/solver_result_analysis_demo.py", "demo_create_research_report", "plotly", "guard"): (
        "#2930 stage C",
        1,
    ),
    ("examples/tutorials/01_hello_mfg.py", "<module>", "matplotlib", "guard"): ("#2930 stage C", 1),
    ("examples/tutorials/02_custom_hamiltonian.py", "<module>", "matplotlib", "guard"): ("#2930 stage C", 1),
    ("examples/tutorials/03_2d_geometry.py", "<module>", "matplotlib", "guard"): ("#2930 stage C", 1),
    ("examples/tutorials/04_particle_methods.py", "<module>", "matplotlib", "guard"): ("#2930 stage C", 1),
    ("examples/tutorials/05_config_system.py", "<module>", "matplotlib", "guard"): ("#2930 stage C", 1),
    ("mfgarchon/alg/numerical/feec/discretization.py", "_import_skfem", "skfem", "guard"): ("#2930 stage C", 1),
    ("mfgarchon/alg/numerical/fem/assembly.py", "_import_skfem", "skfem", "guard"): ("#2930 stage C", 1),
    ("mfgarchon/alg/numerical/fem/mesh_adapter.py", "_import_skfem", "skfem", "guard"): ("#2930 stage C", 1),
    ("mfgarchon/geometry/graph/network_backend.py", "IGraphBackend.__init__", "igraph", "guard"): ("#2930 stage C", 1),
    ("mfgarchon/geometry/graph/network_geometry.py", "<module>", "igraph", "guard"): ("#2930 stage C", 1),
    ("mfgarchon/geometry/meshes/mesh_1d.py", "Mesh1D.export_mesh", "meshio", "guard"): ("#2930 stage C", 1),
    ("mfgarchon/geometry/meshes/mesh_2d.py", "Mesh2D.generate_mesh", "meshio", "guard"): ("#2930 stage C", 1),
    ("mfgarchon/geometry/meshes/mesh_3d.py", "Mesh3D.export_mesh", "meshio", "guard"): ("#2930 stage C", 1),
    ("mfgarchon/geometry/meshes/mesh_3d.py", "Mesh3D.generate_mesh", "meshio", "guard"): ("#2930 stage C", 1),
    ("mfgarchon/geometry/meshes/mesh_data.py", "MeshData.to_meshio", "meshio", "guard"): ("#2930 stage C", 1),
    ("mfgarchon/utils/data/polars_integration.py", "benchmark_polars_vs_pandas", "pandas", "guard"): (
        "#2930 stage C",
        1,
    ),
    ("mfgarchon/utils/data/recorder.py", "HDF5Recorder._check_h5py", "h5py", "guard"): ("#2930 stage C", 1),
    ("mfgarchon/utils/data/recorder.py", "_load_hdf5", "h5py", "guard"): ("#2930 stage C", 1),
    ("mfgarchon/utils/io/hdf5_utils.py", "<module>", "h5py", "guard"): ("#2930 stage C", 1),
    ("mfgarchon/utils/notebooks/reporting.py", "<module>", "nbformat", "guard"): ("#2930 stage C", 2),
    ("mfgarchon/utils/notebooks/reporting.py", "<module>", "plotly", "guard"): ("#2930 stage C", 1),
    ("mfgarchon/utils/solver_result.py", "SolverResult._export_notebook", "nbformat", "guard"): ("#2930 stage C", 2),
    ("mfgarchon/visualization/vtk_export.py", "_as_meshdata", "skfem", "guard"): ("#2930 stage C", 1),
    ("mfgarchon/visualization/vtk_export.py", "_import_meshio", "meshio", "guard"): ("#2930 stage C", 1),
    (
        "tests/integration/test_1043_coupler_fp_drift_convention.py",
        "TestFPNetworkSolverPotentialFieldRename.network_problem",
        "igraph",
        "importorskip",
    ): ("#2930 stage C", 1),
    (
        "tests/integration/test_diffusion_magnitude_gate.py",
        "test_weak_form_fem_fp_diffusion_magnitude",
        "skfem",
        "importorskip",
    ): ("#2930 stage C", 1),
    ("tests/integration/test_fem_quad_path.py", "<module>", "skfem", "importorskip"): ("#2930 stage C", 1),
    ("tests/integration/test_fem_robin_bc.py", "<module>", "skfem", "importorskip"): ("#2930 stage C", 1),
    ("tests/integration/test_fem_solver_path.py", "<module>", "skfem", "importorskip"): ("#2930 stage C", 1),
    ("tests/integration/test_phase1_5_validation.py", "<module>", "igraph", "importorskip"): ("#2930 stage C", 1),
    ("tests/integration/test_weak_form_source_mms_2020.py", "_fp_error", "skfem", "importorskip"): ("#2930 stage C", 1),
    ("tests/integration/test_weak_form_source_mms_2020.py", "_hjb_error", "skfem", "importorskip"): (
        "#2930 stage C",
        1,
    ),
    ("tests/unit/test_alg/test_fem_hex_mesh_470.py", "<module>", "skfem", "importorskip"): ("#2930 stage C", 1),
    ("tests/unit/test_alg/test_fem_reads_the_face_owner_2595.py", "<module>", "skfem", "importorskip"): (
        "#2930 stage C",
        1,
    ),
    ("tests/unit/test_alg/test_fp_drift_from_optimal_control_2900.py", "_every_fp_solver", "skfem", "importorskip"): (
        "#2930 stage C",
        1,
    ),
    ("tests/unit/test_alg/test_fp_drift_from_optimal_control_2900.py", "_fem_fp_alone", "skfem", "importorskip"): (
        "#2930 stage C",
        1,
    ),
    ("tests/unit/test_alg/test_fp_network_solver.py", "<module>", "igraph", "importorskip"): ("#2930 stage C", 1),
    ("tests/unit/test_alg/test_hjb_fem_binds_the_coupling_cost_1642.py", "<module>", "skfem", "importorskip"): (
        "#2930 stage C",
        1,
    ),
    ("tests/unit/test_alg/test_network_hamiltonian_pinning.py", "<module>", "igraph", "importorskip"): (
        "#2930 stage C",
        1,
    ),
    ("tests/unit/test_alg/test_one_bc_owner_2530.py", "_geometry_builders.gmsh_mesh.build", "meshio", "importorskip"): (
        "#2930 stage C",
        1,
    ),
    ("tests/unit/test_alg/test_one_bc_owner_2530.py", "_geometry_builders.network.build", "igraph", "importorskip"): (
        "#2930 stage C",
        1,
    ),
    (
        "tests/unit/test_alg/test_one_bc_owner_2530.py",
        "test_a_components_bc_on_a_network_is_refused_where_the_graph_owns_node_bcs",
        "igraph",
        "importorskip",
    ): ("#2930 stage C", 1),
    ("tests/unit/test_alg/test_paired_solvers_share_the_geometry_2775.py", "<module>", "skfem", "importorskip"): (
        "#2930 stage C",
        1,
    ),
    ("tests/unit/test_alg/test_solve_fp_system_first_parameter_2377.py", "_weak_form", "skfem", "importorskip"): (
        "#2930 stage C",
        1,
    ),
    ("tests/unit/test_alg/test_source_term_channel_2020.py", "_fp_fem", "skfem", "importorskip"): ("#2930 stage C", 1),
    ("tests/unit/test_alg/test_source_term_channel_2020.py", "_hjb_fem", "skfem", "importorskip"): ("#2930 stage C", 1),
    (
        "tests/unit/test_alg/test_weak_form_hjb_picard_sign_2023.py",
        "test_the_picard_branch_agrees_with_the_analytic_constant_hamiltonian",
        "skfem",
        "importorskip",
    ): ("#2930 stage C", 1),
    (
        "tests/unit/test_alg/test_weak_form_hjb_picard_sign_2023.py",
        "test_the_two_branches_of_one_method_solve_the_same_equation",
        "skfem",
        "importorskip",
    ): ("#2930 stage C", 1),
    (
        "tests/unit/test_alg/test_weak_form_hjb_picard_sign_2023.py",
        "test_the_weak_form_family_agrees_with_the_reference_fdm_solver",
        "skfem",
        "importorskip",
    ): ("#2930 stage C", 1),
    (
        "tests/unit/test_core/test_coupling_cost_binds_1642.py",
        "test_what_stage_one_does_not_bind_is_refused_and_each_named_route_runs",
        "igraph",
        "importorskip",
    ): ("#2930 stage C", 1),
    (
        "tests/unit/test_core/test_coupling_cost_binds_1642.py",
        "test_what_stage_two_does_not_bind_is_refused",
        "igraph",
        "importorskip",
    ): ("#2930 stage C", 1),
    # stays: #2088's subject
    (
        "mfgarchon/alg/numerical/gfdm_components/grid_collocation_mapper.py",
        "GridCollocationMapper._build_interpolation_matrix",
        "scipy",
        "guard",
    ): ("#2088: retypes every failure of the block as RuntimeError, scipy's absence among them", 3),
}


def _normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def required_distributions(pyproject: Path) -> list[str]:
    """The distribution names in `[project].dependencies`, by the PEP 508 name rule."""
    names = []
    for dep in tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["dependencies"]:
        assert ";" not in dep, (
            f"{dep!r} carries an environment marker, so it is not required on every install. Decide whether it is "
            "required, and teach this test which installs it covers."
        )
        match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", dep)
        assert match, f"cannot read a distribution name from {dep!r}"
        names.append(_normalise(match.group(1)))
    return names


def required_imports(pyproject: Path) -> dict[str, str]:
    """Top-level import name -> the required distribution that installs it."""
    installs: dict[str, list[str]] = {}
    for top, dists in importlib.metadata.packages_distributions().items():
        if top.isidentifier() and not top.startswith("__"):
            for dist in dists:
                installs.setdefault(_normalise(dist), []).append(top)
    names = {}
    for dist in required_distributions(pyproject):
        assert installs.get(dist), (
            f"{dist} is in [project].dependencies, but this environment's metadata names no importable top-level "
            "module for it, so the guard cannot tell which import names it provides. Install the project, from a "
            "distribution that records its modules, before running the suite."
        )
        names.update(dict.fromkeys(installs[dist], dist))
    return names


def _exception_name(node: ast.expr) -> str | None:
    return node.attr if isinstance(node, ast.Attribute) else getattr(node, "id", None)


def _catches(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    nodes = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return any(_exception_name(n) in CATCHERS for n in nodes)


def _ends_in_reraise(handler: ast.ExceptHandler) -> bool:
    """A handler that ends in a bare `raise`, with no `return` on the way, lets the failure through: it may log, but it
    does not decide what happens without the package."""
    last = handler.body[-1]
    returns = any(isinstance(x, ast.Return) for stmt in handler.body for x in ast.walk(stmt))
    return isinstance(last, ast.Raise) and last.exc is None and not returns


def _suppresses(node: ast.With | ast.AsyncWith) -> bool:
    """`with contextlib.suppress(ImportError)`, under any of the names CATCHERS holds."""
    for item in node.items:
        call = item.context_expr
        if isinstance(call, ast.Call) and _exception_name(call.func) == "suppress":
            if any(_exception_name(arg) in CATCHERS for arg in call.args):
                return True
    return False


def _scope(stack: list[tuple[ast.AST, str]]) -> str:
    names = [n.name for n, _ in stack if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)]
    return ".".join(names) or "<module>"


def _top(module: str | None) -> str:
    return (module or "").split(".")[0]


def _constant_top(node: ast.AST | None, required: dict[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str) and _top(node.value) in required:
        return _top(node.value)
    return None


def _guarded(stack: list[tuple[ast.AST, str]]) -> bool:
    """Whether the nearest `try` body or `suppress` block around this point, within its own function, decides what
    happens without the package. A `try` around a `def` guards the definition, not the body that runs later."""
    for parent, field in reversed(stack):
        if isinstance(parent, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda):
            return False
        if isinstance(parent, ast.Try | ast.TryStar) and field == "body":
            return any(_catches(h) and not _ends_in_reraise(h) for h in parent.handlers)
        if isinstance(parent, ast.With | ast.AsyncWith) and field == "body" and _suppresses(parent):
            return True
    return False


def _names_a_required_package(node: ast.expr, required: dict[str, str]) -> str | None:
    """A required package named by a constant, or by a constant inside a tuple, list or set."""
    if top := _constant_top(node, required):
        return top
    if isinstance(node, ast.Tuple | ast.List | ast.Set):
        return next((top for elt in node.elts if (top := _constant_top(elt, required))), None)
    return None


def _is_error_name(node: ast.expr, err: str) -> bool:
    """`err.name`, where `err` is the name the handler binds."""
    return isinstance(node, ast.Attribute) and node.attr == "name" and getattr(node.value, "id", None) == err


def _is_sys_modules(node: ast.AST) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == "modules" and getattr(node.value, "id", None) == "sys"


def scan(path: Path, rel: str, required: dict[str, str], tests: bool) -> Counter[tuple[str, str, str, str]]:
    """Every row in one file: guards and name-catches anywhere, and the ways a test skips or removes a package."""
    rows: Counter[tuple[str, str, str, str]] = Counter()

    def visit(node: ast.AST, stack: list[tuple[ast.AST, str]]) -> None:
        for fieldname, value in ast.iter_fields(node):
            for child in value if isinstance(value, list) else [value]:
                if not isinstance(child, ast.AST):
                    continue
                here = [*stack, (node, fieldname)]
                imported = []
                if isinstance(child, ast.Import):
                    imported = [_top(a.name) for a in child.names]
                elif isinstance(child, ast.ImportFrom) and child.level == 0:
                    imported = [_top(child.module)]
                elif isinstance(child, ast.Call) and _exception_name(child.func) in ("import_module", "__import__"):
                    imported = [_constant_top(child.args[0] if child.args else None, required)]
                for top in imported:
                    if top in required and _guarded(here):
                        rows[(rel, _scope(here), top, "guard")] += 1
                if isinstance(child, ast.ExceptHandler) and child.name:
                    for x in ast.walk(child):
                        if isinstance(x, ast.Compare):
                            sides = [x.left, *x.comparators]
                            if any(_is_error_name(s, child.name) for s in sides):
                                for s in sides:
                                    if top := _names_a_required_package(s, required):
                                        rows[(rel, _scope(here), top, "name-catch")] += 1
                if tests and isinstance(child, ast.Call):
                    called = _exception_name(child.func)
                    if called == "importorskip" and (
                        top := _constant_top(child.args[0] if child.args else None, required)
                    ):
                        rows[(rel, _scope(here), top, "importorskip")] += 1
                    elif called == "setitem" and len(child.args) >= 2 and _is_sys_modules(child.args[0]):
                        if top := _constant_top(child.args[1], required):
                            rows[(rel, _scope(here), top, "sys.modules")] += 1
                if tests and isinstance(child, ast.Assign | ast.AugAssign | ast.AnnAssign):
                    targets = child.targets if isinstance(child, ast.Assign) else [child.target]
                    for target in targets:
                        if isinstance(target, ast.Subscript) and _is_sys_modules(target.value):
                            if top := _constant_top(target.slice, required):
                                rows[(rel, _scope(here), top, "sys.modules")] += 1
                visit(child, here)

    visit(ast.parse(path.read_bytes()), [])
    return rows


def census(root: Path, required: dict[str, str]) -> Counter[tuple[str, str, str, str]]:
    rows: Counter[tuple[str, str, str, str]] = Counter()
    for directory in (*CODE_DIRS, "tests"):
        for path in sorted((root / directory).rglob("*.py")):
            rows += scan(path, path.relative_to(root).as_posix(), required, tests=directory == "tests")
    return rows


def test_no_required_package_is_guarded():
    required = required_imports(REPO / "pyproject.toml")
    found = census(REPO, required)
    unexplained = sorted(
        f"{key}: found {n}, allowed {ALLOWED.get(key, ('', 0))[1]}"
        for key, n in found.items()
        if n > ALLOWED.get(key, ("", 0))[1]
    )
    assert unexplained == [], (
        "A package in [project].dependencies is guarded, skipped or made absent. It is required (#2930): import it "
        "plainly and let a missing one raise Python's own ModuleNotFoundError. Only a package outside "
        "[project].dependencies may be guarded:\n" + "\n".join(unexplained)
    )
    stale = sorted(
        f"{key}: allowed {n}, found {found.get(key, 0)}" for key, (_, n) in ALLOWED.items() if found.get(key, 0) < n
    )
    assert stale == [], "A row of ALLOWED counts more than the tree holds; lower or delete it:\n" + "\n".join(stale)


_PLANTED_CODE = """
import contextlib
import importlib
import torch_like_optional_name
try:
    import scipy.optimize
except ImportError:
    scipy = None
try:
    import scipy.sparse
except (ImportError, OSError):
    scipy = None

def probe_memory():
    try:
        import psutil
    except ModuleNotFoundError as err:
        if err.name != "psutil":
            raise
        return 0.0
    return psutil.Process()

def probe_by_name():
    try:
        module = importlib.import_module("psutil")
    except ImportError as err:
        if err.name in ("psutil", "other"):
            return None
        raise
    return module

with contextlib.suppress(ImportError):
    import numpy.linalg

try:
    import torch
except ImportError:
    torch = None

try:
    import scipy.linalg
except ImportError:
    print("scipy is missing")
    raise

try:
    def later():
        import scipy.integrate
except ImportError:
    later = None

class Backend:
    def check(self):
        try:
            import numpy
        except ValueError as err:
            return self.backend.name != "numpy" or err.args
        return None
"""

_PLANTED_TEST = """
import sys
import pytest

skfem = pytest.importorskip("skfem")

def test_x(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)
    sys.modules["scipy.optimize"] = None
    torch = pytest.importorskip("torch")
    try:
        import pandas
    except ImportError:
        pandas = None
"""


def test_the_census_sees_what_it_claims_and_nothing_else(tmp_path):
    required = required_imports(REPO / "pyproject.toml")
    assert {"scipy", "psutil", "numpy", "skfem", "pandas"} <= set(required), sorted(required)
    assert "torch" not in required, "torch is an optional extra; the controls below assume it"
    (tmp_path / "mfgarchon").mkdir()
    (tmp_path / "mfgarchon" / "planted.py").write_text(_PLANTED_CODE)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_planted.py").write_text(_PLANTED_TEST)
    for directory in ("examples", "benchmarks"):
        (tmp_path / directory).mkdir()
    assert census(tmp_path, required) == Counter(
        {
            ("mfgarchon/planted.py", "<module>", "scipy", "guard"): 2,  # two guards in one scope count twice
            ("mfgarchon/planted.py", "probe_memory", "psutil", "guard"): 1,
            ("mfgarchon/planted.py", "probe_memory", "psutil", "name-catch"): 1,
            ("mfgarchon/planted.py", "probe_by_name", "psutil", "guard"): 1,  # import_module inside a try
            ("mfgarchon/planted.py", "probe_by_name", "psutil", "name-catch"): 1,  # err.name in (...)
            ("mfgarchon/planted.py", "<module>", "numpy", "guard"): 1,  # contextlib.suppress
            ("tests/test_planted.py", "<module>", "skfem", "importorskip"): 1,
            ("tests/test_planted.py", "test_x", "psutil", "sys.modules"): 1,
            ("tests/test_planted.py", "test_x", "scipy", "sys.modules"): 1,
            ("tests/test_planted.py", "test_x", "pandas", "guard"): 1,  # a guard in a test
        }
    )
    # Not reported: torch (optional), a handler that logs and re-raises, a `try` around a `def`, a `ValueError`
    # handler, and a backend's `.name` compared inside a handler that binds another name.


_PROBE = """
import importlib, importlib.abc, importlib.util, sys
sys.path.insert(0, {repo!r})
ABSENT = {absent!r}

class _Absent(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] == ABSENT:
            raise ModuleNotFoundError(f"No module named {{name!r}}", name=name)
        return None

sys.meta_path.insert(0, _Absent())
_find_spec = importlib.util.find_spec
importlib.util.find_spec = lambda name, package=None: None if name.split(".")[0] == ABSENT else _find_spec(name, package)

import mfgarchon
print("TREE:" + mfgarchon.__file__)
"""

#: The required packages whose absence is already Python's own error at `import mfgarchon`. #2930's later stages add
#: theirs as their guards go.
FAILS_AT_IMPORT = ["psutil"]


def _import_without(absent: str) -> subprocess.CompletedProcess:
    env = {**os.environ}
    env.pop("PYTHONSAFEPATH", None)
    code = _PROBE.format(repo=str(REPO), absent=absent)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300, env=env)


@pytest.mark.parametrize("absent", FAILS_AT_IMPORT)
def test_import_mfgarchon_fails_without_a_required_package(absent):
    proc = _import_without(absent)
    assert proc.returncode != 0, f"`import mfgarchon` succeeded without {absent}, which [project].dependencies requires"
    last = proc.stderr.strip().splitlines()[-1]
    assert last.startswith(f"ModuleNotFoundError: No module named '{absent}'"), (
        f"`import mfgarchon` without {absent} must end in Python's own ModuleNotFoundError, not a retyped error:\n"
        f"{proc.stderr[-1500:]}"
    )
    assert str(REPO / "mfgarchon") in proc.stderr, (
        f"the traceback does not run through this tree:\n{proc.stderr[-1500:]}"
    )


def test_the_probe_imports_when_what_it_refuses_is_not_used():
    """The control: refusing a name nothing imports leaves `import mfgarchon` working, from this tree."""
    proc = _import_without("a_package_nothing_imports")
    assert proc.returncode == 0, proc.stderr[-1500:]
    reported = [line[len("TREE:") :] for line in proc.stdout.splitlines() if line.startswith("TREE:")]
    assert reported, proc.stdout
    assert Path(reported[0]).resolve().is_relative_to((REPO / "mfgarchon").resolve()), proc.stdout


def default_install_names(readme: str) -> list[str]:
    lines = [line for line in readme.splitlines() if line.startswith("**Default install** includes:")]
    assert len(lines) == 1, f"README.md should have one '**Default install** includes:' line, found {len(lines)}"
    listed = lines[0].split(":", 1)[1].strip().rstrip(".")
    return [_normalise(name.strip()) for name in listed.split(",")]


def readme_disagreement(readme: str, pyproject: Path) -> tuple[list[str], list[str], list[str]]:
    """(named in README and not required, required and not named in README, named twice in README)."""
    named, required = default_install_names(readme), required_distributions(pyproject)
    twice = sorted({name for name in named if named.count(name) > 1})
    return sorted(set(named) - set(required)), sorted(set(required) - set(named)), twice


def test_readme_lists_exactly_the_required_packages():
    extra, missing, twice = readme_disagreement(
        (REPO / "README.md").read_text(encoding="utf-8"), REPO / "pyproject.toml"
    )
    assert (extra, missing, twice) == ([], [], []), (
        f"README.md's '**Default install** includes:' line disagrees with [project].dependencies: it names {extra} "
        f"that pyproject does not require, leaves out {missing}, and names {twice} twice. pyproject is the owner; "
        "correct the README line."
    )


def test_the_readme_comparison_sees_a_name_too_many_one_too_few_and_one_twice():
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    line = next(line for line in readme.splitlines() if line.startswith("**Default install** includes:"))
    planted = readme.replace(line, line.rstrip(".") + ", a-package-pyproject-never-listed, meshio.")
    planted = planted.replace(", psutil", "")
    assert readme_disagreement(planted, REPO / "pyproject.toml") == (
        ["a-package-pyproject-never-listed"],
        ["psutil"],
        ["meshio"],
    )
