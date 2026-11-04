"""FP-FEM and the meshless FP keep a negative value and report it, and a non-finite value function ends the solve (#2845).

The oracles: a step's mass is the row sums of its mass matrix on the values it computed, so a clip's gain shows there; the
report is rebuilt from the array the result returns; and the guard's line names the site and the minimum of what it read.
"""

import logging
import math
import warnings

import pytest
import skfem

import numpy as np
from scipy.spatial import Delaunay

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.coupling.block_iterators import BlockIterator
from mfgarchon.alg.numerical.coupling.fictitious_play import FictitiousPlayIterator
from mfgarchon.alg.numerical.coupling.fixed_point_iterator import FixedPointIterator
from mfgarchon.alg.numerical.coupling.graph_coupling import AdjacencyCoupling
from mfgarchon.alg.numerical.coupling.graph_mfg_solver import GraphMFGSolver
from mfgarchon.alg.numerical.coupling.multi_population_iterator import MultiPopulationIterator
from mfgarchon.alg.numerical.coupling.newton_mfg_solver import NewtonMFGSolver
from mfgarchon.alg.numerical.coupling.regime_switching_iterator import RegimeSwitchingIterator
from mfgarchon.alg.numerical.fem import HJBFEMSolver
from mfgarchon.alg.numerical.fem.fp_fem_solver import FPFEMSolver
from mfgarchon.alg.numerical.fem.mesh_adapter import skfem_to_meshdata
from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver
from mfgarchon.alg.numerical.meshless_galerkin.fp_solver import MeshlessGalerkinFPSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.measure import ParticleMeasure
from mfgarchon.core.multi_population import MultiPopulationProblem
from mfgarchon.core.regime_switching import RegimeSwitchingConfig
from mfgarchon.factory.scheme_factory import create_paired_solvers
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, no_flux_bc
from mfgarchon.geometry.meshes.mesh_1d import Mesh1D
from mfgarchon.geometry.meshes.mesh_2d import Mesh2D
from mfgarchon.geometry.meshes.mesh_3d import Mesh3D
from mfgarchon.types import NumericalScheme
from mfgarchon.utils.numerical.mass_fabrication_gate import negative_density_report

_FACES = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
_LOGGER = "mfgarchon.alg.numerical.weak_form_fp_solver"


def _line(n):
    mesh = Mesh1D(bounds=(0.0, 1.0), num_elements=n)
    mesh.generate_mesh()
    return mesh


def _on_square(triangles):
    mesh = Mesh2D(domain_type="rectangle", bounds=(0.0, 1.0, 0.0, 1.0))
    mesh.mesh_data = skfem_to_meshdata(triangles)
    return mesh


def _on_cube(tets):
    mesh = Mesh3D(domain_type="box", bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0))
    mesh.mesh_data = skfem_to_meshdata(tets)
    return mesh


def _tensor(n):
    axis = np.linspace(0.0, 1.0, n + 1)
    return _on_square(skfem.MeshTri.init_tensor(axis, axis))


def _quads(n):
    axis = np.linspace(0.0, 1.0, n + 1)
    return _on_square(skfem.MeshQuad.init_tensor(axis, axis))


def _delaunay_tets():
    rng = np.random.default_rng(1)
    axis = np.linspace(0.0, 1.0, 6)
    grid = np.array(np.meshgrid(axis, axis, axis, indexing="ij")).reshape(3, -1).T
    points = np.vstack([grid[np.any((grid == 0) | (grid == 1), axis=1)], rng.uniform(0.02, 0.98, size=(400, 3))])
    tets = Delaunay(points).simplices
    volume = np.abs(np.linalg.det(points[tets[:, 1:]] - points[tets[:, :1]])) / 6
    return _on_cube(skfem.MeshTet(points.T, tets[volume > 1e-9].T))


def _uncoupled(mesh, D, dt, nt):
    dimension = 1 if isinstance(mesh, Mesh1D) else 2 if isinstance(mesh, Mesh2D) else 3
    level = mesh.num_elements / (mesh.num_elements + 1) if dimension == 1 else 1.0
    mesh.boundary_conditions = BoundaryConditions(
        dimension=dimension,
        segments=[BCSegment(name=f, bc_type=BCType.NO_FLUX, boundary=f) for f in _FACES[: 2 * dimension]],
    )
    return MFGProblem(
        model=Model(
            hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0)),
            volatility=float(np.sqrt(2 * D)),
        ),
        domain=mesh,
        conditions=Conditions(m_initial=lambda x: level, u_terminal=lambda x: 0.0, T=nt * dt),
        Nt=nt,
    )


def _meshless(dim, points, D, dt, nt, streamline=0.0):
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0)] * dim, Nx_points=[points] * dim, boundary_conditions=no_flux_bc(dimension=dim)
    )
    problem = MFGProblem(
        model=Model(
            hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0)),
            volatility=float(np.sqrt(2 * D)),
        ),
        domain=grid,
        conditions=Conditions(m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0, T=nt * dt),
        Nt=nt,
    )
    axis = np.linspace(0.0, 1.0, points)
    cloud = np.array(np.meshgrid(*[axis] * dim, indexing="ij")).reshape(dim, -1).T
    return MeshlessGalerkinFPSolver(
        problem, collocation_points=cloud, delta=0.12 if dim == 1 else 0.25, streamline_diffusion_scale=streamline
    )


def _gaussian(x):
    return np.exp(-np.sum((x - 0.5) ** 2, axis=1) / (2 * 0.1**2))


def _step(x):
    return (x[:, 0] < 0.5).astype(float)


def _points(solver):
    return np.asarray(solver._disc.dof_coordinates, dtype=float)


def _solve(solver, m0, v=0.0):
    x = _points(solver)
    U = np.tile(-v * x[:, 0], (solver.problem.Nt + 1, 1))
    return np.asarray(solver.solve_fp_system(M_initial=m0(x), potential_field=U))


def _mass(solver, density):
    return np.asarray(solver._M.sum(axis=1)).ravel() @ density


def _fem_row():
    return FPFEMSolver(_uncoupled(_line(32), 0.01, 1e-2, 20), order=1)


def _meshless_row():
    return _meshless(1, 31, 0.01, 1e-2, 20)


@pytest.mark.parametrize("build", [_fem_row, _meshless_row], ids=["FP-FEM", "meshless"])
def test_a_forward_step_keeps_its_negative_and_adds_nothing_to_the_mass(build):
    """A clip's gain would show in the mass, on the step it clipped and each step after."""
    solver = build()
    M = _solve(solver, _gaussian, 3.2 if isinstance(solver, FPFEMSolver) else 1.0)
    assert M.min() < -1e-8
    mass = np.array([_mass(solver, level) for level in M])
    assert np.max(np.abs(mass / mass[0] - 1.0)) < 1e-12, mass


@pytest.mark.parametrize("build", [_fem_row, _meshless_row], ids=["FP-FEM", "meshless"])
def test_an_adjoint_step_keeps_its_negative_as_the_forward_step_does(build):
    solver = build()
    x = _points(solver)
    v = 3.2 if isinstance(solver, FPFEMSolver) else 30.0
    advection = solver._build_advection(-v * x[:, 0], 0.01)
    m0 = _gaussian(x)
    stepped = solver.solve_fp_step_adjoint_mode(m0, advection, volatility=np.sqrt(0.02), time=0.25)
    assert stepped.min() < -1e-8
    assert solver.last_negative_density["lowest"] == stepped.min()
    assert solver.last_negative_density["levels"] == 1


def _coupled(refine, sigma=0.3, coupling=None, coupling_dm=None, **model):
    mesh = _on_square(skfem.MeshTri.init_sqsymmetric().refined(refine))
    mesh.boundary_conditions = no_flux_bc(dimension=2)
    terms = {} if coupling is None else {"coupling": coupling, "coupling_dm": coupling_dm}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # the density's mass on the mesh is not these rows' subject
        return MFGProblem(
            model=Model(
                hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0), **terms),
                volatility=sigma,
                **model,
            ),
            domain=mesh,
            conditions=Conditions(
                u_terminal=lambda x: float(np.sum((np.asarray(x) - 0.7) ** 2)),
                m_initial=lambda x: float(np.exp(-10 * np.sum((np.asarray(x) - 0.3) ** 2))),
                T=0.5,
            ),
            Nt=8,
        )


def test_a_coupled_solve_reports_the_negative_it_returns_and_one_warning(mfg_caplog):
    """``problem.solve()`` at its default sweeps: 81 nodes converges with a negative, mass exact; 289 nodes holds none. The
    report is rebuilt from the array the result returns, level by level, which is what the user holds."""
    with mfg_caplog.at_level(logging.WARNING, logger=_LOGGER):
        coarse = _coupled(2).solve(verbose=False)
    warned = [m for m in mfg_caplog.messages if "time levels hold a negative value" in m]
    assert len(warned) == 1, warned
    assert warned[0].endswith(
        "The values are kept: nothing is added to the mass. FP-FEM's Galerkin step can go negative at a high cell Peclet "
        "number |v|h/(2D), and on rough data or a coarse mesh at a lower one; a negative source, load, boundary value or "
        "initial density can do it too. What has cleared one: a finer mesh, given diffusion; P1 on a simplex mesh; P1 on "
        "skfem's MeshTri.init_tensor or MeshTet.init_tensor."
    )
    assert coarse.converged
    M = np.asarray(coarse.M)
    assert M.min() < -1e-3
    weights = np.asarray(FPFEMSolver(_coupled(2), order=1)._M.sum(axis=1)).ravel()
    assert abs((weights @ M[-1]) / (weights @ M[0]) - 1.0) < 1e-12
    report = coarse.metadata["fp_negative_density"]
    negative = np.where(M < 0, -M, 0.0) @ weights
    positive = np.where(M > 0, M, 0.0) @ weights
    assert np.allclose(report["fraction"], negative / positive, rtol=1e-12)
    assert np.allclose(report["minimum"], M.min(axis=1), rtol=1e-12)
    assert report["lowest"] == M.min()
    assert report["lowest_level"] == int(np.argmin(M.min(axis=1)))
    assert report["levels_negative"] == int(np.sum(M.min(axis=1) < 0))
    assert report["levels"] == M.shape[0]
    assert report["cumulative"] == pytest.approx(float(np.sum(negative / positive)))
    assert report["largest"] == pytest.approx(float(np.max(negative / positive)))
    assert "output_validation" not in coarse.metadata

    mfg_caplog.clear()
    with mfg_caplog.at_level(logging.WARNING, logger=_LOGGER):
        fine = _coupled(3).solve(verbose=False)
    assert fine.converged
    assert fine.metadata["fp_negative_density"]["levels_negative"] == 0
    assert not [m for m in mfg_caplog.messages if "time levels hold a negative value" in m]


def _with_an_input(kind, sign, n=32, D=0.01, dt=1e-2, nt=50):
    """A Gaussian on a line, with the input named by ``kind`` at ``sign`` times a size past the gate."""
    faces = {"x_min": BCSegment(name="x_min", bc_type=BCType.NO_FLUX, boundary="x_min")}
    if kind == "boundary value":
        faces["x_min"] = BCSegment(name="x_min", bc_type=BCType.DIRICHLET, value=sign * 0.01, boundary="x_min")
    if kind == "load":
        faces["x_min"] = BCSegment(
            name="x_min", bc_type=BCType.ROBIN, alpha=1.0, beta=1.0, value=sign * 0.01, boundary="x_min"
        )
    wall = BCSegment(name="x_max", bc_type=BCType.NO_FLUX, boundary="x_max")
    solver = FPFEMSolver(
        _uncoupled(_line(n), D, dt, nt),
        order=1,
        boundary_conditions=BoundaryConditions(dimension=1, segments=[faces["x_min"], wall]),
    )
    x = _points(solver).reshape(-1)
    m0 = np.exp(-((x - 0.5) ** 2) / (2 * 0.05**2)) + (sign * 1e-3 if kind == "initial density" else 0.0)
    source = (lambda t, x: np.full(np.shape(x)[0], sign * 1e-3)) if kind == "source" else None
    solver.solve_fp_system(M_initial=m0, potential_field=np.zeros((nt + 1, len(x))), source_term=source)
    return solver.last_negative_density["largest"]


@pytest.mark.parametrize("kind", ["source", "load", "boundary value", "initial density"])
def test_a_negative_input_can_take_the_density_negative_and_its_sign_flipped_does_not(kind):
    """The note's clause, input by input, at a size where this row is past the gate: it says "can", and refining is not
    claimed, since a small enough input does clear on a finer mesh."""
    assert _with_an_input(kind, -1.0) > 1e-8
    assert _with_an_input(kind, 1.0) <= 1e-8


def test_the_result_reports_the_density_it_returns_not_the_last_sweeps_step():
    """A coupler damps, so the array it returns is not the last FP solve's. The last solve's own record is the control."""
    problem = _coupled(2, coupling=lambda m: m, coupling_dm=lambda m: 1.0)
    fp = FPFEMSolver(problem)
    result = BlockIterator(problem, HJBFEMSolver(problem), fp, relaxation=0.5).solve(
        max_iterations=4, tolerance=1e-12, verbose=False
    )
    report = result.metadata["fp_negative_density"]
    assert report["minimum"] == fp.negative_density_of(result.M)["minimum"]
    assert report["minimum"] != fp.last_negative_density["minimum"]


def test_a_solver_alone_warns_and_holds_its_report(mfg_caplog):
    solver = _fem_row()
    with mfg_caplog.at_level(logging.WARNING, logger=_LOGGER):
        M = _solve(solver, _gaussian, 3.2)
    assert len([m for m in mfg_caplog.messages if "time levels hold a negative value" in m]) == 1
    assert solver.last_negative_density["lowest"] == M.min()
    with pytest.raises(ValueError):
        solver.solve_fp_system(np.ones(3), show_progress=False)
    assert solver.last_negative_density is None, "a solve that raises leaves no earlier solve's report behind"


def test_the_meshless_report_says_these_are_coefficients(mfg_caplog):
    with mfg_caplog.at_level(logging.WARNING, logger=_LOGGER):
        _solve(_meshless_row(), _gaussian, 1.0)
    assert any(
        m.endswith(
            "These are MLS coefficients, not the density at a point (#2672). What has cleared one: "
            "streamline_diffusion_scale = 1.0 on a line at D = 0.01, v = 1, dt = 1e-2; in 2-D at the same values it has not."
        )
        for m in mfg_caplog.messages
    )


def test_a_polynomial_coupling_reads_the_negative_and_converges():
    result = _coupled(2, coupling=lambda m: m**2, coupling_dm=lambda m: 2 * m).solve(verbose=False)
    assert result.converged
    assert np.asarray(result.M).min() < 0.0


def _fem_pair(problem):
    return HJBFEMSolver(problem), FPFEMSolver(problem)


def _meshless_pair(problem):
    return create_paired_solvers(
        problem,
        NumericalScheme.MESHLESS_GALERKIN,
        hjb_config={"collocation_points": np.linspace(0.0, 1.0, 21)[:, None], "delta": 0.18},
        fp_config={},
    )


def _on_meshless(coupling, coupling_dm):
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # the density's mass on the grid is not these rows' subject
        return MFGProblem(
            model=Model(
                hamiltonian=SeparableHamiltonian(
                    control_cost=QuadraticControlCost(control_cost=1.0), coupling=coupling, coupling_dm=coupling_dm
                ),
                volatility=0.1,
            ),
            domain=grid,
            conditions=Conditions(
                u_terminal=lambda x: float((np.ravel(x)[0] - 0.7) ** 2),
                m_initial=lambda x: float(np.exp(-10 * (np.ravel(x)[0] - 0.3) ** 2)),
                T=0.5,
            ),
            Nt=10,
        )


def _fixed_point(f, df):
    problem = _coupled(2, coupling=f, coupling_dm=df)
    hjb, fp = _fem_pair(problem)
    return FixedPointIterator(problem, hjb, fp, relaxation=0.5).solve(
        max_iterations=30, tolerance=1e-6, verbose=False
    ), [fp]


def _block(f, df):
    problem = _coupled(2, coupling=f, coupling_dm=df)
    hjb, fp = _fem_pair(problem)
    return BlockIterator(problem, hjb, fp).solve(max_iterations=12, tolerance=1e-6, verbose=False), [fp]


def _fictitious(f, df):
    problem = _on_meshless(f, df)
    hjb, fp = _meshless_pair(problem)
    return FictitiousPlayIterator(problem, hjb, fp).solve(max_iterations=2, tolerance=1e-6, verbose=False), [fp]


def _newton(f, df):
    # 25 nodes, which still hold a negative, since its finite-difference Jacobian is the row's whole cost.
    problem = _coupled(1, coupling=f, coupling_dm=df)
    hjb, fp = _fem_pair(problem)
    return NewtonMFGSolver(problem, hjb, fp, picard_warmup=1).solve(max_iterations=1, tolerance=1e-6, verbose=False), [
        fp
    ]


def _multi_population(f, df):
    problem = _coupled(2, coupling=f, coupling_dm=df)
    hjb, fp = _fem_pair(problem)
    populations = MultiPopulationProblem(populations=[problem], population_names=["a"])
    return MultiPopulationIterator(populations, [hjb], [fp]).solve(max_iterations=12, tolerance=1e-6), [fp]


def _regime(f, df):
    a, b = _coupled(2, coupling=f, coupling_dm=df), _coupled(2, coupling=f, coupling_dm=df)
    (ha, fa), (hb, fb) = _fem_pair(a), _fem_pair(b)
    config = RegimeSwitchingConfig(transition_matrix=np.array([[-0.5, 0.5], [0.5, -0.5]]))
    return RegimeSwitchingIterator([a, b], config, [ha, hb], [fa, fb], max_iterations=6, tolerance=1e-5).solve(), [
        fa,
        fb,
    ]


def _graph(f, df):
    a, b = _coupled(2, coupling=f, coupling_dm=df), _coupled(2, coupling=f, coupling_dm=df)
    (ha, fa), (hb, fb) = _fem_pair(a), _fem_pair(b)
    coupling = AdjacencyCoupling(np.array([[0.0, 1.0], [1.0, 0.0]]), alpha=0.1, beta=0.05)
    return GraphMFGSolver([a, b], coupling, [ha, hb], [fa, fb], max_iterations=6, tolerance=1e-5).solve(), [fa, fb]


_COUPLERS = {
    "FixedPointIterator": _fixed_point,
    "BlockIterator": _block,
    "FictitiousPlayIterator": _fictitious,
    "NewtonMFGSolver": _newton,
    "MultiPopulationIterator": _multi_population,
    "RegimeSwitchingIterator": _regime,
    "GraphMFGSolver": _graph,
}


def _density_and_record(result):
    if isinstance(result, tuple):  # Newton's (U, M, info)
        return result[1], result[2]
    densities = getattr(result, "densities", None)
    return (result.M if densities is None else densities), result.metadata


@pytest.mark.parametrize("coupler", list(_COUPLERS))
def test_each_coupler_records_the_density_it_returns_and_warns_per_solver(coupler, mfg_caplog):
    """Newton's record is in its ``info``; on several FP solvers the record is one per solver."""
    with mfg_caplog.at_level(logging.WARNING, logger=_LOGGER):
        result, solvers = _COUPLERS[coupler](lambda m: m, lambda m: 1.0)
    density, record = _density_and_record(result)
    per_solver = [density] if len(solvers) == 1 else list(density)
    reports = record["fp_negative_density"]
    reports = [reports] if len(solvers) == 1 else reports
    for solver, one, report in zip(solvers, per_solver, reports, strict=True):
        assert report["minimum"] == solver.negative_density_of(one)["minimum"]
        assert report["fraction"] == solver.negative_density_of(one)["fraction"]
        assert solver.on_negative_density == "warn"
    negative = sum(1 for report in reports if report["largest"] > 1e-8)
    assert negative >= 1, "the fixture must hold a negative for this row to say anything"
    assert len([m for m in mfg_caplog.messages if "time levels hold a negative value" in m]) == negative


@pytest.mark.parametrize("coupler", list(_COUPLERS))
def test_a_non_finite_value_function_ends_the_solve_and_names_what_it_read(coupler, mfg_caplog):
    """The stop is the guard all seven share: it reports the array, the site and the density's minimum, and it does not raise."""
    with (
        mfg_caplog.at_level(logging.WARNING, logger="mfgarchon.alg.numerical.coupling.fixed_point_utils"),
        np.errstate(invalid="ignore", divide="ignore"),
    ):
        result, _ = _COUPLERS[coupler](lambda m: np.log(m), lambda m: 1 / m)
    if isinstance(result, tuple):
        assert not result[2]["converged"]
    else:
        assert not result.converged
    lines = [m for m in mfg_caplog.messages if "NaN/Inf in the value function" in m]
    assert lines, "the guard must fire on this row"
    assert coupler in lines[0]
    assert "source:" not in lines[0]
    # What the solve read is what it returns, since FP did not run on this iterate.
    density, _ = _density_and_record(result)
    # A regime's HJB reads its own density, and other regimes' only through their values; a node's source reads them all.
    held = (
        [density]
        if isinstance(density, np.ndarray)
        else [density[0]]
        if coupler == "RegimeSwitchingIterator"
        else density
    )
    read = min(float(np.min(one)) for one in held)
    assert f"has a minimum of {read:.3e}" in lines[0]
    assert f"at iteration {1 if coupler == 'NewtonMFGSolver' else 2}," in lines[0]


def test_newton_warm_up_states_a_negative_the_solve_read(mfg_caplog):
    """The row above has Newton stop at its initial guess, whose minimum is 0, so an absolute value would pass there."""
    with (
        mfg_caplog.at_level(logging.WARNING, logger="mfgarchon.alg.numerical.coupling.fixed_point_utils"),
        np.errstate(invalid="ignore", divide="ignore"),
    ):
        problem = _coupled(2, coupling=lambda m: np.log(m + 1e-4), coupling_dm=lambda m: 1 / (m + 1e-4))
        hjb, fp = _fem_pair(problem)
        NewtonMFGSolver(problem, hjb, fp).solve(max_iterations=5, tolerance=1e-6, verbose=False)
    line = next(m for m in mfg_caplog.messages if "NaN/Inf in the value function" in m)
    assert "at iteration 2," in line
    assert "has a minimum of -" in line


def test_the_sweep_guard_names_the_array_and_the_density_the_sweep_read(mfg_caplog):
    mesh = _on_square(skfem.MeshTri.init_sqsymmetric().refined(2))
    mesh.boundary_conditions = no_flux_bc(dimension=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        problem = MFGProblem(
            model=Model(
                hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)), volatility=0.3
            ),
            source_term_fp=lambda t, x, v, m: 0.0 * np.log(m),
            domain=mesh,
            conditions=Conditions(
                u_terminal=lambda x: float(np.sum((np.asarray(x) - 0.7) ** 2)),
                m_initial=lambda x: float(np.exp(-10 * np.sum((np.asarray(x) - 0.3) ** 2))),
                T=0.5,
            ),
            Nt=8,
        )
    hjb, fp = _fem_pair(problem)
    with (
        mfg_caplog.at_level(logging.WARNING, logger="mfgarchon.alg.numerical.coupling.fixed_point_iterator"),
        np.errstate(invalid="ignore", divide="ignore"),
    ):
        result = FixedPointIterator(problem, hjb, fp, relaxation=0.5).solve(
            max_iterations=30, tolerance=1e-6, verbose=False
        )
    assert not result.converged
    line = next(m for m in mfg_caplog.messages if "NaN/Inf in the density in iteration" in m)
    assert "source:" not in line
    assert "has a minimum of -" in line


def test_a_graph_nodes_line_states_the_lowest_density_any_node_holds(mfg_caplog):
    """A node's source reads its neighbours' densities, so its own would state a minimum that solve did not stop at."""
    read = {}

    def reads_its_neighbour(i, j, v_j, m_j, t):
        read[j] = min(read.get(j, np.inf), float(np.nanmin(m_j)))
        return np.log(m_j + 1e-9)

    high, low = _coupled(2, sigma=2.0), _coupled(2, sigma=0.05)
    (ha, fa), (hb, fb) = _fem_pair(high), _fem_pair(low)
    coupling = AdjacencyCoupling(
        np.array([[0.0, 1.0], [0.0, 0.0]]), coupling_hjb=reads_its_neighbour, alpha=0.1, beta=0.0
    )
    with (
        mfg_caplog.at_level(logging.WARNING, logger="mfgarchon.alg.numerical.coupling.fixed_point_utils"),
        np.errstate(invalid="ignore", divide="ignore"),
    ):
        result = GraphMFGSolver([high, low], coupling, [ha, hb], [fa, fb], max_iterations=6, tolerance=1e-5).solve()
    line = next(m for m in mfg_caplog.messages if "NaN/Inf in the value function" in m)
    assert read[1] < 0.0, "the node must read a negative for this row to say anything"
    lowest = min(float(np.nanmin(one)) for one in result.densities)
    assert f"has a minimum of {lowest:.3e}" in line


@pytest.mark.parametrize(
    ("density", "kept"),
    [(np.array([0.0, 1.0, 3.0]), 2), (np.array([-1e-14, 1.0, 3.0]), 2)],
    ids=["clean", "a negative at round-off counts as zero"],
)
def test_a_measure_reads_a_density_with_no_negative_past_the_gate(density, kept):
    measure = ParticleMeasure.from_density(density, np.array([0.0, 0.5, 1.0]))
    assert measure.weights.size == kept


@pytest.mark.parametrize(
    ("density", "says"),
    [
        (np.array([-0.5, 1.0, 3.0]), r"density went to -5\.000e-01"),
        (np.array([np.nan, 1.0, 3.0]), r"1 of 3 density values are not finite"),
        (np.array([np.inf, 1.0, 3.0]), r"1 of 3 density values are not finite"),
    ],
    ids=["a negative past the gate", "NaN", "inf"],
)
def test_a_measure_refuses_what_it_would_have_dropped(density, says):
    with pytest.raises(ValueError, match=says):
        ParticleMeasure.from_density(density, np.array([0.0, 0.5, 1.0]))


def _tracked_meshless(sigma):
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1))
    problem = MFGProblem(
        model=Model(
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0), coupling=lambda m: m, coupling_dm=lambda m: 1.0
            ),
            volatility=sigma,
        ),
        domain=grid,
        conditions=Conditions(
            u_terminal=lambda x: float((np.ravel(x)[0] - 0.7) ** 2),
            m_initial=lambda x: float(np.exp(-10 * (np.ravel(x)[0] - 0.3) ** 2)),
            T=0.5,
        ),
        Nt=10,
    )
    hjb, fp = create_paired_solvers(
        problem,
        NumericalScheme.MESHLESS_GALERKIN,
        hjb_config={"collocation_points": np.linspace(0.0, 1.0, 21)[:, None], "delta": 0.18},
        fp_config={},
    )
    return FixedPointIterator(problem, hjb, fp, relaxation=0.5)


def test_tracking_a_solve_refuses_the_negative_and_the_advice_runs():
    """The refusal's advice is that the same solve without tracking has run, and its result reports the negative."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match=r"density went to .*track_measure_field"):
            _tracked_meshless(0.1).solve(max_iterations=60, tolerance=1e-6, verbose=False, track_measure_field=True)
        result = _tracked_meshless(0.1).solve(max_iterations=60, tolerance=1e-6, verbose=False)
    assert result.metadata["fp_negative_density"]["lowest"] < 0.0


def test_a_tracked_sweep_the_guard_ends_records_no_snapshot(mfg_caplog):
    """The guard names a NaN density, tracked or not: the snapshot follows it, so `from_density` does not meet the NaN."""
    iterator = _tracked_meshless(0.3)
    solve = iterator.fp_solver.solve_fp_system

    def goes_non_finite(*args, **kwargs):
        M = solve(*args, **kwargs)
        M[-1, 3] = np.nan
        return M

    iterator.fp_solver.solve_fp_system = goes_non_finite
    with (
        mfg_caplog.at_level(logging.WARNING, logger="mfgarchon.alg.numerical.coupling.fixed_point_iterator"),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore")
        result = iterator.solve(max_iterations=5, tolerance=1e-6, verbose=False, track_measure_field=True)
    assert (
        result.convergence_reason == "diverged_nan" if hasattr(result, "convergence_reason") else not result.converged
    )
    assert any("NaN/Inf in the density in iteration 1" in m for m in mfg_caplog.messages)


def _flag_after_the_first_solve():
    """A coupling whose body raises once ``flag["on"]``, before any FP solve of the second solve runs."""
    flag = {"on": False}

    def body(m):
        if flag["on"]:
            raise RuntimeError("this body raises")
        return m

    return flag, body


def test_a_raise_keeps_its_exception_restores_the_policy_and_reports_no_earlier_solve(mfg_caplog):
    """A reused iterator: the first solve leaves a negative, and the second's HJB raises before any FP step runs."""
    flag, body = _flag_after_the_first_solve()
    problem = _coupled(2, coupling=body, coupling_dm=lambda m: 1.0)
    hjb, fp = _fem_pair(problem)
    iterator = FixedPointIterator(problem, hjb, fp, relaxation=0.5)
    with mfg_caplog.at_level(logging.WARNING, logger=_LOGGER):
        iterator.solve(max_iterations=30, tolerance=1e-6, verbose=False)
    assert any("time levels hold a negative value" in m for m in mfg_caplog.messages)
    assert fp.on_negative_density == "warn"

    mfg_caplog.clear()
    flag["on"] = True
    with mfg_caplog.at_level(logging.WARNING, logger=_LOGGER), pytest.raises(RuntimeError, match=r"^this body raises$"):
        iterator.solve(max_iterations=30, tolerance=1e-6, verbose=False)
    assert not [m for m in mfg_caplog.messages if "time levels hold a negative value" in m]
    assert fp.last_negative_density is None
    assert fp.on_negative_density == "warn"


def test_a_body_that_raises_on_a_negative_keeps_its_exception_and_the_last_solve_is_reported(mfg_caplog):
    """``math.log`` meets a negative on a later sweep: its own ``ValueError``, and one warning from the last FP solve."""
    with (
        mfg_caplog.at_level(logging.WARNING, logger=_LOGGER),
        pytest.raises(ValueError, match=r"^math domain error$") as raised,
    ):
        _coupled(2, coupling=lambda m: math.log(m), coupling_dm=lambda m: 1 / m).solve(verbose=False)
    assert raised.value.__cause__ is None
    assert len([m for m in mfg_caplog.messages if "time levels hold a negative value" in m]) == 1


@pytest.mark.parametrize(("build", "checked"), [(_fem_pair, False), (None, True)], ids=["FP-FEM", "FP-FDM"])
def test_the_validators_density_check_is_off_only_where_the_solver_reports_it(build, checked, monkeypatch):
    """On FP-FEM the solver's report says the same thing, and
    `test_a_coupled_solve_reports_the_negative_it_returns_and_one_warning` goes red if it is lost. Elsewhere it stays."""
    import mfgarchon.utils.validation.runtime as runtime

    seen = []
    validate = runtime.validate_solver_output

    def recording(*args, **kwargs):
        seen.append(kwargs["check_density_positive"])
        return validate(*args, **kwargs)

    monkeypatch.setattr(runtime, "validate_solver_output", recording)
    if build is None:
        grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # the density's mass on the grid is not this row's subject
            problem = MFGProblem(
                model=Model(
                    hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
                    volatility=0.3,
                ),
                domain=grid,
                conditions=Conditions(
                    u_terminal=lambda x: 0.0,
                    m_initial=lambda x: float(np.exp(-10 * (np.ravel(x)[0] - 0.3) ** 2)),
                    T=0.5,
                ),
                Nt=10,
            )
        hjb, fp = HJBFDMSolver(problem), FPFDMSolver(problem)
    else:
        problem = _coupled(3)
        hjb, fp = build(problem)
    FixedPointIterator(problem, hjb, fp, relaxation=0.5).solve(max_iterations=3, tolerance=1e-6, verbose=False)
    assert seen == [checked]


def test_the_warning_states_the_reports_numbers(mfg_caplog):
    """Each number by its place: a swap of the largest and the sum, or of two levels, reads differently here."""
    solver = _fem_row()
    n = solver.n_dof
    density = np.ones((3, n))
    density[1, 5:9] = -0.25  # the larger negative part, on level 1
    density[2, 5] = -0.5  # the lower minimum, on level 2
    report = solver.negative_density_of(density)
    assert report["levels_negative"] == 2
    assert report["lowest"] == -0.5
    assert report["lowest_level"] == 2
    assert report["largest_level"] == 1
    with mfg_caplog.at_level(logging.WARNING, logger=_LOGGER):
        solver.warn_negative_density(report)
    line = next(m for m in mfg_caplog.messages if "time levels hold a negative value" in m)
    assert line.startswith(
        f"FPFEMSolver: 2 of 3 time levels hold a negative value, down to {report['lowest']:.3e} at level 2, at most "
        f"{report['largest']:.3e} of a level's positive mass (sum over levels {report['cumulative']:.3e})."
    )
    assert report["cumulative"] == pytest.approx(report["fraction"][1] + report["fraction"][2])


def test_a_nan_in_a_level_does_not_hide_its_negative():
    """Each reads the rest of the level. A level that is all NaN holds no negative, and the report says NaN there."""
    report = negative_density_report(np.array([[1.0, -0.5, 1.0], [np.nan, np.nan, np.nan]]))
    assert report["lowest"] == -0.5
    assert report["levels_negative"] == 1
    assert np.isnan(report["minimum"][1])
    assert negative_density_report(np.array([[1.0, np.nan, -0.5]]))["levels_negative"] == 1
    assert negative_density_report(np.array([[np.nan, np.nan]]))["levels_negative"] == 0
    assert negative_density_report(np.array([[np.nan, np.nan]]))["largest"] == 0.0


# (mesh with a negative past the gate, mesh that holds none, D, v, dt, nt, data, element order on each)
_CLEARED = {
    "a finer mesh on a line": (lambda: _line(32), lambda: _line(256), 0.01, 3.2, 1e-2, 20, _gaussian, 1, 1),
    "a finer mesh on init_tensor": (lambda: _tensor(4), lambda: _tensor(16), 0.8839, 1.0, 1e-4, 5, _gaussian, 1, 1),
    "P1 on a simplex mesh, from Q1": (lambda: _quads(32), lambda: _tensor(32), 0.18, 0.0, 1e-4, 5, _step, 1, 1),
    "P1 on a simplex mesh, from P2": (lambda: _line(32), lambda: _line(32), 0.18, 0.0, 1e-4, 5, _step, 2, 1),
    "P1 on init_tensor, from Delaunay": (
        _delaunay_tets,
        lambda: _on_cube(skfem.MeshTet.init_tensor(*[np.linspace(0.0, 1.0, 9)] * 3)),
        0.18,
        0.0,
        1e-3,
        10,
        _gaussian,
        1,
        1,
    ),
}


@pytest.mark.parametrize("row", list(_CLEARED))
def test_each_thing_the_report_says_has_cleared_one_clears_one(row):
    """ "Cleared" is what the warning fires on: no level past the gate's 1e-8 of its mass."""
    given, finer, D, v, dt, nt, data, order, cleared_order = _CLEARED[row]
    solver = FPFEMSolver(_uncoupled(given(), D, dt, nt), order=order)
    _solve(solver, data, v)
    assert solver.last_negative_density["largest"] > 1e-8
    cleared = FPFEMSolver(_uncoupled(finer(), D, dt, nt), order=cleared_order)
    _solve(cleared, data, v)
    assert cleared.last_negative_density["largest"] <= 1e-8


def test_streamline_diffusion_cleared_the_line_and_not_the_plane():
    """The meshless note's own record, both halves."""
    line = _meshless(1, 31, 0.01, 1e-2, 20, streamline=1.0)
    _solve(line, _gaussian, 1.0)
    assert line.last_negative_density["levels_negative"] == 0
    plane = _meshless(2, 13, 0.01, 1e-2, 20, streamline=1.0)
    _solve(plane, _gaussian, 1.0)
    assert plane.last_negative_density["lowest"] < -1e-5


class _ClipsWhatItReturns(FPFEMSolver):
    """The documented way in: a caller who wants a clip writes it, and it is their change of scheme."""

    def solve_fp_system(self, M_initial=None, potential_field=None, volatility=None, source_term=None, **kwargs):
        solved = super().solve_fp_system(
            M_initial=M_initial,
            potential_field=potential_field,
            volatility=volatility,
            source_term=source_term,
            **kwargs,
        )
        return np.maximum(solved, 0.0)


@pytest.mark.parametrize("refine", [2, 3])
def test_a_subclass_may_clip_what_it_returns(refine):
    """``log(max(m, 1e-10))``, 81 and 289 nodes: the solver alone blows up, and the subclass's run converges."""
    floor = 1e-10

    def run(fp_type):
        problem = _coupled(
            refine, coupling=lambda m: np.log(np.maximum(m, floor)), coupling_dm=lambda m: 1 / np.maximum(m, floor)
        )
        return FixedPointIterator(problem, HJBFEMSolver(problem), fp_type(problem), relaxation=0.5).solve(
            max_iterations=100, tolerance=1e-6, verbose=False
        )

    assert not run(FPFEMSolver).converged
    assert run(_ClipsWhatItReturns).converged
