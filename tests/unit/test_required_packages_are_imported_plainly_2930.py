"""A package in `[project].dependencies` is required: it is imported plainly, and nothing decides what happens without it.

User rulings, 2026-11-10 and 2026-11-11 (#2930): there is one install, `pyproject.toml`'s `[project].dependencies` is
its one list, and a missing package is Python's own `ModuleNotFoundError`. Only a package of an optional extra may be
guarded.

**The population** is read from pyproject, never written here. Each distribution is mapped to the top-level names it
installs by `importlib.metadata.packages_distributions()`, so a distribution that is not installed fails the test
rather than shrinking it.

**The structural guard** walks `mfgarchon/`, `examples/` and `benchmarks/` by AST and reports, by (file, enclosing
function or class, package, kind):

- ``guard``: an import of a required package in the body of a `try` with a handler that catches `ImportError`,
  `ModuleNotFoundError`, `Exception` or `BaseException`, or catches everything. A handler whose whole body is a bare
  `raise` does not decide what happens without the package and is not reported.
- ``name-catch``: inside `except ... as err`, a comparison of `err.name` with a required package. Only the name the
  handler binds counts, so a backend's `.name` is not reported.

It walks `tests/` for:

- ``importorskip``: `pytest.importorskip` of a required package;
- ``sys.modules``: a write to `sys.modules` for one, by subscript or by `setitem(sys.modules, ...)`.

`ALLOWED` lists the rows that #2930's later stages remove, each naming its stage, and one row that stays for another
reason. A listed row that no longer exists fails, so each stage removes its lines in its own diff.

**What it cannot see:** a package named through a variable; a guard in another file around a module that imports the
package at top level; fenced code in Markdown; and a fallback reached without a `try`, such as `find_spec` on a required
package, which the census for later stages reads separately.
"""

from __future__ import annotations

import ast
import importlib.metadata
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CODE_DIRS = ("mfgarchon", "examples", "benchmarks")
CATCHERS = {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"}

#: (file, enclosing scope, package, kind) -> the stage of #2930 that removes it, or why it stays.
ALLOWED: dict[tuple[str, str, str, str], str] = {
    # #2930 stage B
    ("mfgarchon/alg/numerical/density_estimation.py", "<module>", "scipy", "guard"): "#2930 stage B",
    ("mfgarchon/alg/numerical/fp_solvers/fp_particle.py", "<module>", "scipy", "guard"): "#2930 stage B",
    ("mfgarchon/alg/optimization/optimal_transport/__init__.py", "<module>", "scipy", "guard"): "#2930 stage B",
    (
        "mfgarchon/alg/optimization/optimal_transport/wasserstein_solver.py",
        "WassersteinMFGSolver._check_dependencies",
        "scipy",
        "guard",
    ): "#2930 stage B",
    ("mfgarchon/alg/optimization/variational_problem.py", "<module>", "scipy", "guard"): "#2930 stage B",
    (
        "mfgarchon/alg/optimization/variational_solvers/primal_dual_solver.py",
        "<module>",
        "scipy",
        "guard",
    ): "#2930 stage B",
    (
        "mfgarchon/alg/optimization/variational_solvers/variational_mfg_solver.py",
        "<module>",
        "scipy",
        "guard",
    ): "#2930 stage B",
    ("mfgarchon/core/hamiltonian.py", "DualHamiltonian.__call__", "scipy", "guard"): "#2930 stage B",
    ("mfgarchon/core/hamiltonian.py", "DualHamiltonian._find_optimal_alpha", "scipy", "guard"): "#2930 stage B",
    ("mfgarchon/core/hamiltonian.py", "DualLagrangian.__call__", "scipy", "guard"): "#2930 stage B",
    ("mfgarchon/core/hamiltonian.py", "DualLagrangian._find_optimal_p", "scipy", "guard"): "#2930 stage B",
    ("mfgarchon/geometry/graph/maze_cellular_automata.py", "<module>", "scipy", "guard"): "#2930 stage B",
    ("mfgarchon/geometry/graph/maze_postprocessing.py", "<module>", "scipy", "guard"): "#2930 stage B",
    ("mfgarchon/geometry/graph/maze_utils.py", "<module>", "scipy", "guard"): "#2930 stage B",
    ("mfgarchon/geometry/graph/maze_voronoi.py", "<module>", "scipy", "guard"): "#2930 stage B",
    (
        "mfgarchon/geometry/grids/tensor_grid.py",
        "TensorProductGrid.get_interpolator.interpolate_linear",
        "scipy",
        "guard",
    ): "#2930 stage B",
    ("mfgarchon/hooks/visualization.py", "AnimationHook.__init__", "matplotlib", "guard"): "#2930 stage B",
    ("mfgarchon/hooks/visualization.py", "PlottingHook.__init__", "matplotlib", "guard"): "#2930 stage B",
    ("mfgarchon/operators/interpolation/projection.py", "_mesh_to_grid_delaunay", "scipy", "guard"): "#2930 stage B",
    ("mfgarchon/types/state.py", "ConvergenceInfo.plot_convergence", "matplotlib", "guard"): "#2930 stage B",
    ("mfgarchon/utils/functional_calculus.py", "create_particle_measure", "scipy", "guard"): "#2930 stage B",
    ("mfgarchon/utils/numerical/monotonicity_stats.py", "verify_m_matrix_property", "scipy", "guard"): "#2930 stage B",
    ("mfgarchon/utils/numerical/particle/kde_boundary.py", "<module>", "scipy", "guard"): "#2930 stage B",
    (
        "mfgarchon/utils/numerical/particle/sampling.py",
        "QuasiMCSampler._init_sequence_generator",
        "scipy",
        "guard",
    ): "#2930 stage B",
    ("mfgarchon/utils/numpy_compat.py", "<module>", "scipy", "guard"): "#2930 stage B",
    (
        "mfgarchon/utils/performance/optimization.py",
        "SparseMatrixOptimizer.optimize_matrix_structure",
        "scipy",
        "guard",
    ): "#2930 stage B",
    ("mfgarchon/visualization/convergence_plots.py", "<module>", "matplotlib", "guard"): "#2930 stage B",
    (
        "tests/unit/test_core/test_legendre_sign_convention_2375.py",
        "_no_scipy_optimize",
        "scipy",
        "sys.modules",
    ): "#2930 stage B",
    (
        "tests/unit/test_utils/test_monte_carlo.py",
        "test_monte_carlo_integrate_quasi",
        "scipy",
        "importorskip",
    ): "#2930 stage B",
    (
        "tests/unit/test_utils/test_monte_carlo.py",
        "test_quasi_mc_sampler_custom_domain",
        "scipy",
        "importorskip",
    ): "#2930 stage B",
    (
        "tests/unit/test_utils/test_monte_carlo.py",
        "test_quasi_mc_sampler_halton",
        "scipy",
        "importorskip",
    ): "#2930 stage B",
    (
        "tests/unit/test_utils/test_monte_carlo.py",
        "test_quasi_mc_sampler_latin_hypercube",
        "scipy",
        "importorskip",
    ): "#2930 stage B",
    # #2930 stage C
    ("benchmarks/particle_kde_gpu_benchmark.py", "<module>", "scipy", "guard"): "#2930 stage C",
    ("benchmarks/performance_tracker.py", "<module>", "psutil", "guard"): "#2930 stage C",
    ("benchmarks/run_benchmarks.py", "<module>", "psutil", "guard"): "#2930 stage C",
    (
        "examples/advanced/geometry_advanced/dual_geometry_fem_mesh.py",
        "register_specialized_fem_projections.mesh_to_grid_delaunay",
        "scipy",
        "guard",
    ): "#2930 stage C",
    (
        "examples/advanced/solvers_advanced/semi_lagrangian_2d_enhancements.py",
        "<module>",
        "matplotlib",
        "guard",
    ): "#2930 stage C",
    (
        "examples/advanced/visualization/visualize_2d_density_evolution.py",
        "<module>",
        "matplotlib",
        "guard",
    ): "#2930 stage C",
    (
        "examples/applications/networks/network_mfg_comparison_example.py",
        "<module>",
        "plotly",
        "guard",
    ): "#2930 stage C",
    ("examples/basic/core_infrastructure/mfg_working_demo.py", "<module>", "plotly", "guard"): "#2930 stage C",
    ("examples/basic/solvers/policy_iteration_lq_demo.py", "<module>", "matplotlib", "guard"): "#2930 stage C",
    (
        "examples/basic/utilities/solver_result_analysis_demo.py",
        "demo_create_research_report",
        "plotly",
        "guard",
    ): "#2930 stage C",
    ("examples/tutorials/01_hello_mfg.py", "<module>", "matplotlib", "guard"): "#2930 stage C",
    ("examples/tutorials/02_custom_hamiltonian.py", "<module>", "matplotlib", "guard"): "#2930 stage C",
    ("examples/tutorials/03_2d_geometry.py", "<module>", "matplotlib", "guard"): "#2930 stage C",
    ("examples/tutorials/04_particle_methods.py", "<module>", "matplotlib", "guard"): "#2930 stage C",
    ("examples/tutorials/05_config_system.py", "<module>", "matplotlib", "guard"): "#2930 stage C",
    ("mfgarchon/alg/numerical/feec/discretization.py", "_import_skfem", "skfem", "guard"): "#2930 stage C",
    ("mfgarchon/alg/numerical/fem/assembly.py", "_import_skfem", "skfem", "guard"): "#2930 stage C",
    ("mfgarchon/alg/numerical/fem/mesh_adapter.py", "_import_skfem", "skfem", "guard"): "#2930 stage C",
    ("mfgarchon/geometry/graph/network_backend.py", "IGraphBackend.__init__", "igraph", "guard"): "#2930 stage C",
    ("mfgarchon/geometry/graph/network_geometry.py", "<module>", "igraph", "guard"): "#2930 stage C",
    ("mfgarchon/geometry/meshes/mesh_1d.py", "Mesh1D.export_mesh", "meshio", "guard"): "#2930 stage C",
    ("mfgarchon/geometry/meshes/mesh_2d.py", "Mesh2D.generate_mesh", "meshio", "guard"): "#2930 stage C",
    ("mfgarchon/geometry/meshes/mesh_3d.py", "Mesh3D.export_mesh", "meshio", "guard"): "#2930 stage C",
    ("mfgarchon/geometry/meshes/mesh_3d.py", "Mesh3D.generate_mesh", "meshio", "guard"): "#2930 stage C",
    ("mfgarchon/geometry/meshes/mesh_data.py", "MeshData.to_meshio", "meshio", "guard"): "#2930 stage C",
    ("mfgarchon/utils/data/polars_integration.py", "benchmark_polars_vs_pandas", "pandas", "guard"): "#2930 stage C",
    ("mfgarchon/utils/data/recorder.py", "HDF5Recorder._check_h5py", "h5py", "guard"): "#2930 stage C",
    ("mfgarchon/utils/data/recorder.py", "_load_hdf5", "h5py", "guard"): "#2930 stage C",
    ("mfgarchon/utils/io/hdf5_utils.py", "<module>", "h5py", "guard"): "#2930 stage C",
    ("mfgarchon/utils/notebooks/reporting.py", "<module>", "nbformat", "guard"): "#2930 stage C",
    ("mfgarchon/utils/notebooks/reporting.py", "<module>", "plotly", "guard"): "#2930 stage C",
    ("mfgarchon/utils/solver_result.py", "SolverResult._export_notebook", "nbformat", "guard"): "#2930 stage C",
    ("mfgarchon/visualization/vtk_export.py", "_as_meshdata", "skfem", "guard"): "#2930 stage C",
    ("mfgarchon/visualization/vtk_export.py", "_import_meshio", "meshio", "guard"): "#2930 stage C",
    (
        "tests/integration/test_1043_coupler_fp_drift_convention.py",
        "TestFPNetworkSolverPotentialFieldRename.network_problem",
        "igraph",
        "importorskip",
    ): "#2930 stage C",
    (
        "tests/integration/test_diffusion_magnitude_gate.py",
        "test_weak_form_fem_fp_diffusion_magnitude",
        "skfem",
        "importorskip",
    ): "#2930 stage C",
    ("tests/integration/test_fem_quad_path.py", "<module>", "skfem", "importorskip"): "#2930 stage C",
    ("tests/integration/test_fem_robin_bc.py", "<module>", "skfem", "importorskip"): "#2930 stage C",
    ("tests/integration/test_fem_solver_path.py", "<module>", "skfem", "importorskip"): "#2930 stage C",
    ("tests/integration/test_phase1_5_validation.py", "<module>", "igraph", "importorskip"): "#2930 stage C",
    ("tests/integration/test_weak_form_source_mms_2020.py", "_fp_error", "skfem", "importorskip"): "#2930 stage C",
    ("tests/integration/test_weak_form_source_mms_2020.py", "_hjb_error", "skfem", "importorskip"): "#2930 stage C",
    ("tests/unit/test_alg/test_fem_hex_mesh_470.py", "<module>", "skfem", "importorskip"): "#2930 stage C",
    ("tests/unit/test_alg/test_fem_reads_the_face_owner_2595.py", "<module>", "skfem", "importorskip"): "#2930 stage C",
    (
        "tests/unit/test_alg/test_fp_drift_from_optimal_control_2900.py",
        "_every_fp_solver",
        "skfem",
        "importorskip",
    ): "#2930 stage C",
    (
        "tests/unit/test_alg/test_fp_drift_from_optimal_control_2900.py",
        "_fem_fp_alone",
        "skfem",
        "importorskip",
    ): "#2930 stage C",
    ("tests/unit/test_alg/test_fp_network_solver.py", "<module>", "igraph", "importorskip"): "#2930 stage C",
    (
        "tests/unit/test_alg/test_hjb_fem_binds_the_coupling_cost_1642.py",
        "<module>",
        "skfem",
        "importorskip",
    ): "#2930 stage C",
    ("tests/unit/test_alg/test_network_hamiltonian_pinning.py", "<module>", "igraph", "importorskip"): "#2930 stage C",
    (
        "tests/unit/test_alg/test_one_bc_owner_2530.py",
        "_geometry_builders.gmsh_mesh.build",
        "meshio",
        "importorskip",
    ): "#2930 stage C",
    (
        "tests/unit/test_alg/test_one_bc_owner_2530.py",
        "_geometry_builders.network.build",
        "igraph",
        "importorskip",
    ): "#2930 stage C",
    (
        "tests/unit/test_alg/test_one_bc_owner_2530.py",
        "test_a_components_bc_on_a_network_is_refused_where_the_graph_owns_node_bcs",
        "igraph",
        "importorskip",
    ): "#2930 stage C",
    (
        "tests/unit/test_alg/test_paired_solvers_share_the_geometry_2775.py",
        "<module>",
        "skfem",
        "importorskip",
    ): "#2930 stage C",
    (
        "tests/unit/test_alg/test_solve_fp_system_first_parameter_2377.py",
        "_weak_form",
        "skfem",
        "importorskip",
    ): "#2930 stage C",
    ("tests/unit/test_alg/test_source_term_channel_2020.py", "_fp_fem", "skfem", "importorskip"): "#2930 stage C",
    ("tests/unit/test_alg/test_source_term_channel_2020.py", "_hjb_fem", "skfem", "importorskip"): "#2930 stage C",
    (
        "tests/unit/test_alg/test_weak_form_hjb_picard_sign_2023.py",
        "test_the_picard_branch_agrees_with_the_analytic_constant_hamiltonian",
        "skfem",
        "importorskip",
    ): "#2930 stage C",
    (
        "tests/unit/test_alg/test_weak_form_hjb_picard_sign_2023.py",
        "test_the_two_branches_of_one_method_solve_the_same_equation",
        "skfem",
        "importorskip",
    ): "#2930 stage C",
    (
        "tests/unit/test_alg/test_weak_form_hjb_picard_sign_2023.py",
        "test_the_weak_form_family_agrees_with_the_reference_fdm_solver",
        "skfem",
        "importorskip",
    ): "#2930 stage C",
    (
        "tests/unit/test_core/test_coupling_cost_binds_1642.py",
        "test_what_stage_one_does_not_bind_is_refused_and_each_named_route_runs",
        "igraph",
        "importorskip",
    ): "#2930 stage C",
    (
        "tests/unit/test_core/test_coupling_cost_binds_1642.py",
        "test_what_stage_two_does_not_bind_is_refused",
        "igraph",
        "importorskip",
    ): "#2930 stage C",
    # stays: #2088's subject
    (
        "mfgarchon/alg/numerical/gfdm_components/grid_collocation_mapper.py",
        "GridCollocationMapper._build_interpolation_matrix",
        "scipy",
        "guard",
    ): "#2088: retypes every failure of the block as RuntimeError, scipy's absence among them",
}


def _normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def required_distributions(pyproject: Path) -> list[str]:
    deps = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["dependencies"]
    return [_normalise(re.split(r"[<>=!~;\[ ]", d, maxsplit=1)[0]) for d in deps]


def required_imports(pyproject: Path) -> dict[str, str]:
    """Top-level import name -> the required distribution that installs it."""
    installs: dict[str, list[str]] = {}
    for top, dists in importlib.metadata.packages_distributions().items():
        for dist in dists:
            installs.setdefault(_normalise(dist), []).append(top)
    names = {}
    for dist in required_distributions(pyproject):
        assert installs.get(dist), (
            f"{dist} is in [project].dependencies but this environment has not installed it, so the guard cannot "
            "tell which import names it provides. Install the project before running the suite."
        )
        names.update(
            dict.fromkeys((top for top in installs[dist] if top.isidentifier() and not top.startswith("__")), dist)
        )
    return names


def _catches(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    nodes = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return any((n.attr if isinstance(n, ast.Attribute) else getattr(n, "id", None)) in CATCHERS for n in nodes)


def _reraises_only(handler: ast.ExceptHandler) -> bool:
    return len(handler.body) == 1 and isinstance(handler.body[0], ast.Raise) and handler.body[0].exc is None


def _scope(stack: list[tuple[ast.AST, str]]) -> str:
    names = [n.name for n, _ in stack if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)]
    return ".".join(names) or "<module>"


def _top(module: str | None) -> str:
    return (module or "").split(".")[0]


def scan_code(path: Path, rel: str, required: dict[str, str]) -> set[tuple[str, str, str, str]]:
    rows = set()

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
                for top in imported:
                    if top not in required:
                        continue
                    for parent, field in reversed(here):
                        if isinstance(parent, ast.Try | ast.TryStar) and field == "body":
                            deciding = [h for h in parent.handlers if _catches(h) and not _reraises_only(h)]
                            if deciding:
                                rows.add((rel, _scope(here), top, "guard"))
                            break
                if isinstance(child, ast.ExceptHandler) and child.name:
                    for x in ast.walk(child):
                        if not isinstance(x, ast.Compare):
                            continue
                        sides = [x.left, *x.comparators]
                        named = any(
                            isinstance(s, ast.Attribute)
                            and s.attr == "name"
                            and isinstance(s.value, ast.Name)
                            and s.value.id == child.name
                            for s in sides
                        )
                        for s in sides:
                            if named and isinstance(s, ast.Constant) and isinstance(s.value, str):
                                if _top(s.value) in required:
                                    rows.add((rel, _scope(here), _top(s.value), "name-catch"))
                visit(child, here)

    visit(ast.parse(path.read_text(encoding="utf-8")), [])
    return rows


def scan_test(path: Path, rel: str, required: dict[str, str]) -> set[tuple[str, str, str, str]]:
    rows = set()

    def constant_top(node: ast.AST) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and _top(node.value) in required:
            return _top(node.value)
        return None

    def is_sys_modules(node: ast.AST) -> bool:
        return isinstance(node, ast.Attribute) and node.attr == "modules" and getattr(node.value, "id", None) == "sys"

    def visit(node: ast.AST, stack: list[tuple[ast.AST, str]]) -> None:
        for fieldname, value in ast.iter_fields(node):
            for child in value if isinstance(value, list) else [value]:
                if not isinstance(child, ast.AST):
                    continue
                here = [*stack, (node, fieldname)]
                if isinstance(child, ast.Call):
                    func = child.func
                    called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                    if called == "importorskip" and child.args and (top := constant_top(child.args[0])):
                        rows.add((rel, _scope(here), top, "importorskip"))
                    if called == "setitem" and len(child.args) >= 2 and is_sys_modules(child.args[0]):
                        if top := constant_top(child.args[1]):
                            rows.add((rel, _scope(here), top, "sys.modules"))
                if isinstance(child, ast.Assign | ast.AugAssign | ast.AnnAssign):
                    targets = child.targets if isinstance(child, ast.Assign) else [child.target]
                    for target in targets:
                        if isinstance(target, ast.Subscript) and is_sys_modules(target.value):
                            if top := constant_top(target.slice):
                                rows.add((rel, _scope(here), top, "sys.modules"))
                visit(child, here)

    visit(ast.parse(path.read_text(encoding="utf-8")), [])
    return rows


def census(root: Path, required: dict[str, str]) -> set[tuple[str, str, str, str]]:
    rows = set()
    for directory in CODE_DIRS:
        for path in sorted((root / directory).rglob("*.py")):
            rows |= scan_code(path, path.relative_to(root).as_posix(), required)
    for path in sorted((root / "tests").rglob("*.py")):
        rows |= scan_test(path, path.relative_to(root).as_posix(), required)
    return rows


def test_no_required_package_is_guarded():
    required = required_imports(REPO / "pyproject.toml")
    found = census(REPO, required)
    unexplained = sorted(found - set(ALLOWED))
    assert unexplained == [], (
        "A package in [project].dependencies is guarded, skipped or made absent. It is required (#2930): import it "
        "plainly and let a missing one raise Python's own ModuleNotFoundError. Only a package of an optional extra may "
        "be guarded:\n" + "\n".join(map(str, unexplained))
    )
    stale = sorted(set(ALLOWED) - found)
    assert stale == [], "A row of ALLOWED no longer exists; delete its line:\n" + "\n".join(map(str, stale))


_PLANTED_CODE = """
import torch_like_optional_name
try:
    import scipy.optimize
except ImportError:
    scipy = None

def probe_memory():
    try:
        import psutil
    except ModuleNotFoundError as err:
        if err.name != "psutil":
            raise
        return 0.0
    return psutil.Process()

try:
    import torch
except ImportError:
    torch = None

try:
    import scipy.linalg
except ImportError:
    raise

class Backend:
    def check(self):
        try:
            import numpy
        except ValueError:
            return None
        return self.backend.name != "numpy"
"""

_PLANTED_TEST = """
import sys
import pytest

skfem = pytest.importorskip("skfem")

def test_x(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)
    sys.modules["scipy.optimize"] = None
    torch = pytest.importorskip("torch")
"""


def test_the_census_sees_what_it_claims_and_nothing_else(tmp_path):
    required = required_imports(REPO / "pyproject.toml")
    assert {"scipy", "psutil", "numpy", "skfem", "yaml"} <= set(required), sorted(required)
    assert "torch" not in required, "torch is an optional extra; the controls below assume it"
    (tmp_path / "mfgarchon").mkdir()
    (tmp_path / "mfgarchon" / "planted.py").write_text(_PLANTED_CODE)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_planted.py").write_text(_PLANTED_TEST)
    for directory in ("examples", "benchmarks"):
        (tmp_path / directory).mkdir()
    assert census(tmp_path, required) == {
        ("mfgarchon/planted.py", "<module>", "scipy", "guard"),
        ("mfgarchon/planted.py", "probe_memory", "psutil", "guard"),
        ("mfgarchon/planted.py", "probe_memory", "psutil", "name-catch"),
        ("tests/test_planted.py", "<module>", "skfem", "importorskip"),
        ("tests/test_planted.py", "test_x", "psutil", "sys.modules"),
        ("tests/test_planted.py", "test_x", "scipy", "sys.modules"),
    }


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
    assert f"ModuleNotFoundError: No module named '{absent}" in proc.stderr, proc.stderr[-1500:]
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


def readme_disagreement(readme: str, pyproject: Path) -> tuple[list[str], list[str]]:
    """(named in README and not required, required and not named in README)."""
    named, required = default_install_names(readme), required_distributions(pyproject)
    return sorted(set(named) - set(required)), sorted(set(required) - set(named))


def test_readme_lists_exactly_the_required_packages():
    extra, missing = readme_disagreement((REPO / "README.md").read_text(encoding="utf-8"), REPO / "pyproject.toml")
    assert (extra, missing) == ([], []), (
        f"README.md's '**Default install** includes:' line disagrees with [project].dependencies: it names {extra} "
        f"that pyproject does not require, and leaves out {missing}. pyproject is the owner; correct the README line."
    )


def test_the_readme_comparison_sees_a_name_too_many_and_one_too_few():
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    line = next(line for line in readme.splitlines() if line.startswith("**Default install** includes:"))
    planted = readme.replace(line, line.rstrip(".") + ", a-package-pyproject-never-listed.").replace(", psutil", "")
    assert readme_disagreement(planted, REPO / "pyproject.toml") == (["a-package-pyproject-never-listed"], ["psutil"])
