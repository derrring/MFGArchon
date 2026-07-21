"""Issue #1362: rendering-free extraction of plot-ready data from solve results."""

import ast
import inspect

import pytest

import numpy as np

from mfgarchon.visualization import extract as ex


def _result():
    from mfgarchon.utils.solver_result import SolverResult

    U = np.arange(12, dtype=float).reshape(3, 4)
    M = np.full((3, 4), 0.25)  # uniform density
    return SolverResult(
        U=U,
        M=M,
        iterations=5,
        error_history_U=np.array([1.0, 0.1, 0.01, 0.001, 1e-4]),
        error_history_M=np.array([2.0, 0.2, 0.02, 0.002, 2e-4]),
    )


def _graph_result():
    from mfgarchon.alg.numerical.coupling.graph_mfg_solver import GraphMFGResult

    v = [np.arange(6, dtype=float).reshape(3, 2), np.ones((3, 2))]
    d = [np.full((3, 2), 0.5), np.full((3, 2), 0.25)]
    return GraphMFGResult(values=v, densities=d, converged=True, iterations=3, error_history=[1.0, 0.1], n_nodes=2)


def test_convergence_history():
    df = ex.extract_convergence_history(_result())
    assert list(df.columns) == ["iteration", "error_U", "error_M"]
    assert len(df) == 5
    assert df["error_U"].iloc[0] == 1.0
    assert df["error_M"].iloc[-1] == pytest.approx(2e-4)


def test_mass_history_integrates_over_x():
    x = np.linspace(0.0, 1.0, 4)
    df = ex.extract_mass_history(_result(), x)
    assert list(df.columns) == ["step", "total_mass"]
    assert len(df) == 3
    np.testing.assert_allclose(df["total_mass"], np.trapezoid(np.full(4, 0.25), x))


def test_density_slices_tidy_and_time_selection():
    r = _result()
    x = np.linspace(0.0, 1.0, 4)
    t = np.array([0.0, 0.5, 1.0])
    all_df = ex.extract_density_slices(r, x, t)
    assert list(all_df.columns) == ["t", "x", "value"]
    assert len(all_df) == 3 * 4
    sel = ex.extract_density_slices(r, x, t, times=[0.5])
    assert set(sel["t"]) == {0.5}
    assert len(sel) == 4
    # value_slices reads U
    v = ex.extract_value_slices(r, x, t, times=[1.0])
    assert v["value"].tolist() == r.U[2].tolist()


def test_slices_fail_loud():
    r = _result()
    x = np.linspace(0.0, 1.0, 4)
    t = np.array([0.0, 0.5, 1.0])
    with pytest.raises(ValueError, match="not in t_grid"):
        ex.extract_density_slices(r, x, t, times=[0.3])  # not a grid time -> no silent snap
    with pytest.raises(ValueError, match="shape"):
        ex.extract_density_slices(r, x, np.array([0.0, 0.5]), None)  # wrong t_grid length


def test_graph_trajectories():
    gr = _graph_result()
    df = ex.extract_graph_trajectories(gr, field="density")
    assert list(df.columns) == ["node", "step", "x", "value"]
    assert set(df["node"]) == {0, 1}
    assert len(df) == 2 * 3 * 2  # nodes * steps * x
    assert df[(df.node == 0) & (df.step == 0)]["value"].tolist() == [0.5, 0.5]
    with pytest.raises(ValueError, match="field"):
        ex.extract_graph_trajectories(gr, field="bogus")


def test_extract_module_imports_no_renderer():
    """#1362 acceptance: extract.py itself imports no rendering backend. Checked at source level —
    the package __init__ pulls meshio (via vtk_export), so a sys.modules check would be confounded;
    this verifies the extract module's own import surface."""
    tree = ast.parse(inspect.getsource(ex))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    bad = imported & {"matplotlib", "plotly", "pyvista", "meshio"}
    assert not bad, f"extract.py imports renderers: {bad}"
