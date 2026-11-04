"""FP-FEM and the meshless FP keep a negative value and report it, and a density read that cannot take one stops (#2845).

The oracles: a step's mass is the row sums of its mass matrix on the values it computed, so a clip's gain shows there; the
report is rebuilt from the array the result returns; and a stop names its read, its time and the minimum of ``m``.
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
from mfgarchon.alg.numerical.fem import HJBFEMSolver
from mfgarchon.alg.numerical.fem.fp_fem_solver import FPFEMSolver
from mfgarchon.alg.numerical.fem.mesh_adapter import skfem_to_meshdata
from mfgarchon.alg.numerical.meshless_galerkin.fp_solver import MeshlessGalerkinFPSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, no_flux_bc
from mfgarchon.geometry.meshes.mesh_1d import Mesh1D
from mfgarchon.geometry.meshes.mesh_2d import Mesh2D
from mfgarchon.geometry.meshes.mesh_3d import Mesh3D
from mfgarchon.operators.interaction import PotentialCost

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


def test_the_meshless_report_says_these_are_coefficients(mfg_caplog):
    with mfg_caplog.at_level(logging.WARNING, logger=_LOGGER):
        _solve(_meshless_row(), _gaussian, 1.0)
    assert any("MLS coefficients, not the density at a point" in m for m in mfg_caplog.messages)


@pytest.mark.parametrize(
    ("coupling", "dm"),
    [(np.log, lambda m: 1 / m), (np.sqrt, lambda m: 0.5 / np.sqrt(m))],
    ids=["log", "sqrt"],
)
def test_a_density_read_that_cannot_take_a_negative_stops_and_names_it(coupling, dm):
    """On the coupled row where the clip converged, naming the read, its time and the minimum of m."""
    with (
        pytest.raises(ValueError, match=r"coupling returned a non-finite value at t=[\d.]+: the minimum of m is -"),
        np.errstate(invalid="ignore", divide="ignore"),
    ):
        _coupled(2, coupling=coupling, coupling_dm=dm).solve(verbose=False)


def test_a_polynomial_coupling_reads_the_negative_and_converges():
    result = _coupled(2, coupling=lambda m: m**2, coupling_dm=lambda m: 2 * m).solve(verbose=False)
    assert result.converged
    assert np.asarray(result.M).min() < 0.0


def _hamiltonian(coupling):
    return SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0), coupling=coupling)


_POINT = (0.25, np.zeros(1), np.zeros(1))
_BATCH = (0.25, np.zeros((3, 1)), np.zeros((3, 1)))
_WITH_A_NEGATIVE = np.array([0.5, -0.1, 0.2])


def _refuses_a_negative(m):
    if float(m) < 0:
        raise ValueError("this body takes no negative")
    return float(m)


@pytest.mark.parametrize(
    ("coupling", "message", "cause"),
    [
        (lambda m: math.log(m), r"raised ValueError: math domain error, at t=0\.25", ValueError),
        (lambda m: float(m) ** 0.5, r"raised TypeError: .*complex.* at t=0\.25", TypeError),
        (_refuses_a_negative, r"raised ValueError: this body takes no negative, at t=0\.25", ValueError),
    ],
    ids=["math.log", "complex", "own raise"],
)
def test_a_body_that_raises_on_a_negative_stops_with_its_own_message(coupling, message, cause):
    for args in [(*_BATCH, _WITH_A_NEGATIVE), (*_POINT, -0.1)]:
        with pytest.raises(ValueError, match=message) as stopped:
            _hamiltonian(coupling)(*args)
        assert "the minimum of m is -1.000e-01" in str(stopped.value)
        assert isinstance(stopped.value.__cause__, cause)


def test_a_body_that_raises_with_no_negative_is_left_alone():
    def always(m):
        raise ValueError("not about a negative")

    with pytest.raises(ValueError, match=r"^not about a negative$"):
        _hamiltonian(always)(*_POINT, 0.3)
    with pytest.raises(ZeroDivisionError):
        _hamiltonian(lambda m: 1 / m)(*_POINT, 0.0)


def test_an_exact_zero_stops_too():
    with pytest.raises(ValueError, match=r"the minimum of m is 0\.000e\+00"), np.errstate(divide="ignore"):
        _hamiltonian(np.log)(*_POINT, 0.0)


class _Logarithmic:
    """A user's own cost: its derivative reads log(m), which the shipped costs do not."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def flat_derivative(self, m, *, t=None):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.log(np.asarray(m, dtype=float))


class _LogarithmicCost(PotentialCost):
    def __init__(self):
        super().__init__(lambda x: 0.0)

    def bind_on(self, measure):
        return _Logarithmic(super().bind_on(measure))


def test_a_users_coupling_cost_that_cannot_take_a_negative_stops_at_its_read():
    with pytest.raises(
        ValueError, match=r"coupling_cost returned a non-finite value at t=[\d.]+: the minimum of m is -"
    ):
        _coupled(2, coupling_cost=_LogarithmicCost()).solve(verbose=False)


def test_a_users_terminal_coupling_that_cannot_take_a_negative_stops_at_its_read():
    """Directly, since on the coupled row a first sweep's blow-up ends the solve before a negative m(T) reaches this read."""
    problem = _coupled(2, terminal_coupling=_LogarithmicCost())
    hjb = HJBFEMSolver(problem)
    m_terminal = np.full(problem.get_u_terminal().size, 0.1)
    assert problem.terminal_value(m_terminal, binding=hjb.terminal_binding()).size == m_terminal.size
    m_terminal[0] = -0.1
    with pytest.raises(
        ValueError, match=r"terminal_coupling returned a non-finite value at t=0\.5: the minimum of m is -"
    ):
        problem.terminal_value(m_terminal, binding=hjb.terminal_binding())


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
    from mfgarchon.alg.numerical.coupling.fixed_point_iterator import FixedPointIterator

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
