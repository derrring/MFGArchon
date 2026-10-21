"""#1642 stage 1b-2: HJB-FEM P1 binds ``Model(coupling_cost=...)`` on its own point measure, its dofs with the consistent
mass matrix's row sums, and the couplers hand that binding to the HJB source.

- The measure: the dofs, the row sums and the mesh's measure (`LumpedMassPointMeasure`).
- The plumbing: a linear F = int V m, bound by ``PotentialCost(V)``, against the same V as the Hamiltonian's potential,
  through each call site that passes the binding.
- An exact shift: a constant kernel c adds ``c * mass * (T - t)`` to u, the mass the P1 interpolant's integral.
- The quadrature: ``K * 1`` against the exact box integral, second order in h.
- What is refused, and by whom: P2, a cost that cannot bind on a point measure, the FFT path, a periodic condition, the
  meshless solver, and the readers that pass no binding.
"""

from __future__ import annotations

import inspect
import logging
import warnings

import pytest

import numpy as np
from scipy.special import erf

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.coupling.base_mfg import hjb_solve_kwargs
from mfgarchon.alg.numerical.coupling.source_composition import compose_hjb_source
from mfgarchon.config import MFGSolverConfig, PicardConfig
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.operators.interaction import InteractionCost, PotentialCost
from mfgarchon.operators.interaction.kernels import GaussianKernel, RadialKernel

skfem = pytest.importorskip("skfem", reason="scikit-fem required for FEM tests")

from mfgarchon.alg.numerical.fem import HJBFEMSolver  # noqa: E402
from mfgarchon.alg.numerical.fem.fp_fem_solver import FPFEMSolver  # noqa: E402
from mfgarchon.alg.numerical.fem.mesh_adapter import skfem_to_meshdata  # noqa: E402


def _V(x):
    x = np.asarray(x, dtype=float)
    return 0.8 * np.sin(2 * np.pi * x[..., 0]) + 0.5 * x[..., 1]


def _mesh(refine=2, stretch=(1.0, 1.0), bc=None, d=2):
    """A gmsh-free mesh: ``[0, a] x [0, b]`` triangulated by skfem's ``init_sqsymmetric``, a polygon, so the sum of
    its element areas is the domain's area; or, with ``d`` 1 or 3, a ``Mesh1D`` of 16 elements or ``[0, 1]^3`` in
    tetrahedra."""
    if d == 1:
        from mfgarchon.geometry.meshes.mesh_1d import Mesh1D

        geometry = Mesh1D(bounds=(0.0, 1.0), num_elements=16)
        geometry.generate_mesh()
    elif d == 3:
        from mfgarchon.geometry.meshes.mesh_3d import Mesh3D

        geometry = Mesh3D(domain_type="box", bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0))
        geometry.mesh_data = skfem_to_meshdata(skfem.MeshTet.init_tensor(*[np.linspace(0.0, 1.0, 4)] * 3))
    else:
        from mfgarchon.geometry.meshes.mesh_2d import Mesh2D

        mesh = skfem.MeshTri.init_sqsymmetric().refined(refine)
        mesh = skfem.MeshTri(mesh.p * np.asarray(stretch)[:, None], mesh.t)
        geometry = Mesh2D(domain_type="rectangle", bounds=(0.0, stretch[0], 0.0, stretch[1]))
        geometry.mesh_data = skfem_to_meshdata(mesh)
    geometry.boundary_conditions = bc if bc is not None else no_flux_bc(dimension=d)
    return geometry


def _problem(where=None, cost=None, nt=8, **mesh):
    """V as the Hamiltonian's potential (``"H"``), as ``PotentialCost(V)`` (``"F"``), or ``cost``."""
    if where == "F":
        cost = PotentialCost(lambda x: float(_V(x)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # the density's mass on the mesh is not these pins' subject
        return MFGProblem(
            model=Model(
                hamiltonian=SeparableHamiltonian(
                    control_cost=QuadraticControlCost(control_cost=1.0),
                    potential=(lambda t, x: _V(x)) if where == "H" else None,
                ),
                volatility=0.3,
                coupling_cost=cost,
            ),
            domain=_mesh(**mesh),
            conditions=Conditions(
                u_terminal=lambda x: float(np.sum((np.asarray(x) - 0.7) ** 2)),
                m_initial=lambda x: float(np.exp(-10 * np.sum((np.asarray(x) - 0.3) ** 2))),
                T=0.5,
            ),
            Nt=nt,
        )


def _frozen(hjb, problem):
    """A density and an iterate on the solver's dofs, the same for every solve that compares them."""
    x = hjb._disc.dof_coordinates
    M = np.tile(np.exp(-10 * np.sum((x - 0.3) ** 2, axis=1)), (problem.Nt + 1, 1))
    U = np.tile(np.sum((x - 0.7) ** 2, axis=1), (problem.Nt + 1, 1))
    return M, U


_PICARD = MFGSolverConfig(picard=PicardConfig(max_iterations=5, tolerance=1e-12, relaxation=0.5, verbose=False))


def test_the_p1_measure_is_the_dofs_with_the_lumped_mass():
    """The binding's points are the P1 dofs, which are the mesh's vertices in the order the problem evaluates
    ``m_initial`` in; its weights are the consistent mass matrix's row sums; its domain measure is the sum of the
    element areas, read from the mesh and not from the weights, so element volumes 1% too large are refused by the
    measure's contract. The arrays are read-only copies, sharing no memory with the basis."""
    problem = _problem(cost=PotentialCost(lambda x: 0.0))
    hjb = HJBFEMSolver(problem)
    measure = hjb._coupling_point_measure()
    vertices = np.asarray(problem.geometry.get_spatial_grid(), dtype=float)
    assert np.array_equal(measure.points, vertices)
    assert np.array_equal(hjb.coupling_binding().nodes, vertices)
    assert np.array_equal(measure.weights, np.asarray(hjb._M.sum(axis=1)).ravel())
    assert measure.domain_measure == float(np.sum(problem.geometry.mesh_data.compute_element_volumes()))
    assert np.isclose(measure.weights.sum(), 1.0, rtol=1e-13, atol=0.0)
    for array in (measure.points, measure.weights):
        with pytest.raises(ValueError, match="read-only"):
            array[0] = 0.0
    assert not np.shares_memory(measure.points, hjb.basis.doflocs)
    mesh_data = problem.geometry.mesh_data
    mesh_data.element_volumes = 1.01 * mesh_data.compute_element_volumes()
    with pytest.raises(ValueError, match=r"weights summing to .* not to the domain's measure"):
        HJBFEMSolver(problem)


@pytest.mark.parametrize("route", ["problem.solve()", "MultiPopulationIterator", "solve_hjb_system"])
def test_a_linear_coupling_cost_solves_as_the_same_potential_on_hjb_fem(route):
    """F = int V m has ``delta F / delta m = V``, so ``PotentialCost(V)`` and V as H's potential pose one game, and on
    HJB-FEM P1 the source the binding gives is V at the dofs, which is where H reads its potential. One row per place
    the binding is handed over: ``BaseCouplingIterator._build_hjb_kwargs`` (``problem.solve()`` builds a
    ``FixedPointIterator`` with HJB-FEM and FP-FEM here), ``MultiPopulationIterator``'s own call, and the standalone
    route the solvers' refusal names, ``compose_hjb_source(..., binding=solver.coupling_binding())``. Five Picard sweeps,
    and the standalone solve, agree to 2.2e-16, 3.3e-16 and 2.2e-16 in that order; without V, U moves by 0.42, 0.40 and
    0.49."""

    def solve(where):
        problem = _problem(where)
        if route == "problem.solve()":
            return np.asarray(problem.solve(config=_PICARD).U)
        hjb = HJBFEMSolver(problem)
        if route == "MultiPopulationIterator":
            from mfgarchon.alg.numerical.coupling.multi_population_iterator import MultiPopulationIterator
            from mfgarchon.core.multi_population import MultiPopulationProblem

            result = MultiPopulationIterator(
                MultiPopulationProblem(populations=[problem], population_names=["a"]), [hjb], [FPFEMSolver(problem)]
            ).solve(max_iterations=5, tolerance=1e-12)
            return np.asarray(result.U[0])
        M, U = _frozen(hjb, problem)
        source = compose_hjb_source(problem, M, U, binding=hjb.coupling_binding())
        return np.asarray(hjb.solve_hjb_system(M, U[-1], U, source_term=source))

    logging.disable(logging.WARNING)  # the FP's positivity-clip notices are not this pin's subject
    try:
        gap = np.max(np.abs(solve("H") - solve("F")))
        dropped = np.max(np.abs(solve("H") - solve(None)))
    finally:
        logging.disable(logging.NOTSET)
    assert gap <= 1e-14, f"{route}: PotentialCost(V) and V in H differ by {gap:.3e} on HJB-FEM P1"
    assert dropped > 0.1, f"the fixture does not see V: dropping it moves U by {dropped:.3e}"


def _grid_problem(model):
    from mfgarchon.geometry import TensorProductGrid

    return MFGProblem(
        model=model,
        domain=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1)),
        conditions=Conditions(
            u_terminal=lambda x: (np.asarray(x, dtype=float) - 0.7) ** 2,
            m_initial=lambda x: np.exp(-40 * (np.asarray(x, dtype=float) - 0.3) ** 2),
            T=0.3,
        ),
        Nt=6,
    )


def _cost_model(V):
    return Model(
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
        volatility=0.3,
        coupling_cost=None if V is None else PotentialCost(V),
    )


# The fixtures' unnormalised densities and the HJB solvers' tolerance notices are not this pin's subject.
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    "case", ["grid, another cost", "grid, swapped populations", "mesh, a rebuild", "mesh, no cost solved"]
)
def test_an_hjb_solver_built_from_another_problem_reads_the_solved_problems_cost(case):
    """A coupler may pair the problem it solves with an HJB solver built from another problem whose volatility agrees
    (#2378). The coupling part is the solved problem's F, as at e6c9df6d, and not the solver's problem's:

    - on a grid it is the solved problem's own binding, whatever the solver, with ``FixedPointIterator`` and with
      ``MultiPopulationIterator``'s solvers swapped: equal to the solve with each solver built from its own problem,
      to 0.0. At f591dddf, where the solver's own problem's binding won, they differed by 3.0e-01 and
      3.4e-01;
    - on a mesh HJB-FEM's binding is read when it was bound from the solved problem's cost: a rebuild with other
      conditions shares it, and solves equal to 0.0;
    - a solved problem with no cost reads none, whatever the solver binds.
    """
    from mfgarchon.alg.numerical.coupling import FixedPointIterator
    from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
    from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver

    def V1(x):
        return 0.8 * np.sin(2 * np.pi * np.asarray(x, dtype=float))

    def V2(x):
        return -1.5 * np.cos(3 * np.pi * np.asarray(x, dtype=float))

    def fpi(solved, hjb, fp):
        U, *_ = FixedPointIterator(solved, hjb, fp, relaxation=0.5).solve(max_iterations=3, tolerance=1e-12)
        return np.asarray(U)

    logging.disable(logging.WARNING)
    try:
        if case == "grid, another cost":
            built, solved = _grid_problem(_cost_model(V1)), _grid_problem(_cost_model(V2))
            got = fpi(solved, HJBFDMSolver(built), FPFDMSolver(solved))
            want = fpi(solved, HJBFDMSolver(solved), FPFDMSolver(solved))
            other = fpi(built, HJBFDMSolver(built), FPFDMSolver(built))
        elif case == "grid, swapped populations":
            from mfgarchon.alg.numerical.coupling.multi_population_iterator import MultiPopulationIterator
            from mfgarchon.core.multi_population import MultiPopulationProblem

            ps = [_grid_problem(_cost_model(V1)), _grid_problem(_cost_model(V2))]

            def solve(hjbs):
                result = MultiPopulationIterator(
                    MultiPopulationProblem(populations=ps, population_names=["a", "b"]),
                    hjbs,
                    [FPFDMSolver(p) for p in ps],
                ).solve(max_iterations=3, tolerance=1e-12)
                return np.asarray(result.U)

            got = solve([HJBFDMSolver(ps[1]), HJBFDMSolver(ps[0])])
            want = solve([HJBFDMSolver(p) for p in ps])
            other = None
        elif case == "mesh, a rebuild":
            built = _problem("F")
            solved = built.with_conditions(
                Conditions(
                    u_terminal=lambda x: float(np.sum(np.asarray(x) ** 2)),
                    m_initial=lambda x: float(np.exp(-10 * np.sum((np.asarray(x) - 0.6) ** 2))),
                    T=0.5,
                )
            )
            assert solved.declared_coupling_cost is built.declared_coupling_cost
            got = fpi(solved, HJBFEMSolver(built), FPFEMSolver(solved))
            want = fpi(solved, HJBFEMSolver(solved), FPFEMSolver(solved))
            other = fpi(built, HJBFEMSolver(built), FPFEMSolver(built))
        else:
            built, solved = _problem("F"), _problem(None)
            got = fpi(solved, HJBFEMSolver(built), FPFEMSolver(solved))
            want = fpi(solved, HJBFEMSolver(solved), FPFEMSolver(solved))
            other = fpi(built, HJBFEMSolver(built), FPFEMSolver(built))
    finally:
        logging.disable(logging.NOTSET)
    assert np.array_equal(got, want), f"{case}: the solve reads another F, {np.max(np.abs(got - want)):.3e} off"
    if other is not None:
        assert np.max(np.abs(other - want)) > 0.1, f"{case}: the two problems do not tell their costs apart"


@pytest.mark.parametrize("built_from", ["another cost", "no cost"])
def test_hjb_fem_built_from_another_problem_is_refused_on_a_mesh(built_from):
    """Off a grid the solved problem binds no F, and HJB-FEM's binding is of the problem it was built from. Built from
    one with another cost, or with none, it is refused by name, before the first evaluation, and the message says that
    a solver binds the cost of the problem it was built from. At f591dddf the first solved with the other F, in
    silence."""
    from mfgarchon.alg.numerical.coupling import FixedPointIterator

    built = _problem(
        cost=InteractionCost(GaussianKernel(amplitude=1.0, length_scale=0.3)) if built_from == "another cost" else None
    )
    solved = _problem("F")
    loop = FixedPointIterator(solved, HJBFEMSolver(built), FPFEMSolver(solved))
    refusal = (
        r"HJBFEMSolver was handed a binding of another problem's coupling cost.*InteractionCost.*PotentialCost"
        if built_from == "another cost"
        else r"HJBFEMSolver has no binding of Model\(coupling_cost=\.\.\.\) on this Mesh2D.*a solver binds the cost "
        r"of the problem it was built from"
    )
    with pytest.raises(NotImplementedError, match=refusal):
        loop.solve(max_iterations=1, tolerance=1e-12)


class _Constant(RadialKernel):
    """``K = c``: ``delta F / delta m = c * mass(m)``, the same at every point."""

    def __init__(self, c):
        self.c = c

    def profile(self, r):
        return self.c * np.ones_like(np.asarray(r, dtype=float))

    @property
    def is_repulsive(self):
        return self.c > 0


@pytest.mark.parametrize("nt", [4, 8])
def test_a_constant_kernel_shifts_u_by_c_times_the_mass(nt):
    """With K = c the source is ``c * mass(m_t)``, a constant in x. P1's no-flux operator and H's gradient both
    annihilate a constant, so on a density of fixed mass the source shifts u by exactly ``c * mass * (T - t_n)`` at
    level n, implicit Euler integrating a constant exactly. The mass is the P1 interpolant's integral, assembled by
    scikit-fem's own quadrature rather than read from the binding's weights. Measured: 2.5e-16 off at Nt = 4 and 3.0e-16
    at Nt = 8, for a shift of 0.090 at t = 0."""
    c = 0.7
    with_cost, without = _problem(cost=InteractionCost(_Constant(c)), nt=nt), _problem(nt=nt)
    hjb, plain = HJBFEMSolver(with_cost), HJBFEMSolver(without)
    M, U = _frozen(hjb, with_cost)
    params = {"M_density", "U_terminal", "U_coupling_prev", "source_term", "volatility"}
    kwargs = hjb_solve_kwargs(with_cost, params, "HJBFEMSolver", M=M, U=U, binding=hjb.coupling_binding())
    shift = np.asarray(hjb.solve_hjb_system(M, U[-1], U, **kwargs)) - np.asarray(plain.solve_hjb_system(M, U[-1], U))
    mass = skfem.Functional(lambda w: w["m"]).assemble(hjb.basis, m=hjb.basis.interpolate(M[0]))
    t = np.arange(with_cost.Nt + 1) * with_cost.dt
    exact = c * mass * (with_cost.T - t)[:, None] * np.ones_like(shift)
    assert np.max(np.abs(shift - exact)) <= 1e-13 * np.max(np.abs(exact)), (
        f"the constant kernel shifts u by {shift[0].mean():.6f} at t = 0, not c * mass * T = {exact[0, 0]:.6f}"
    )


def _box(x, a, b, length):
    """``int_a^b exp(-(x - y)^2 / (2 l^2)) dy``."""
    return length * np.sqrt(np.pi / 2) * (erf((b - x) / (np.sqrt(2) * length)) - erf((a - x) / (np.sqrt(2) * length)))


def test_the_p1_binding_integrates_an_interaction_at_second_order():
    """``InteractionCost(K)`` bound on the P1 measure gives ``flat_derivative(1) = sum_j K(x - y_j) w_j``, the lumped
    quadrature of ``(K * 1)(x)``. On ``[0, 2] x [0, 1]``, a polygon whose area the mesh holds exactly, it converges at
    second order to the exact box integral of a Gaussian of length 0.15: the max error is 2.91e-3, 7.17e-4 and 1.79e-4
    at refinements 3 to 5, slopes 2.02 and 2.00."""
    length, errors = 0.15, []
    for refine in (3, 4, 5):
        problem = _problem(
            cost=InteractionCost(GaussianKernel(amplitude=1.0, length_scale=length)),
            nt=2,
            refine=refine,
            stretch=(2.0, 1.0),
        )
        binding = HJBFEMSolver(problem).coupling_binding()
        x = binding.nodes
        exact = _box(x[:, 0], 0.0, 2.0, length) * _box(x[:, 1], 0.0, 1.0, length)
        errors.append(np.max(np.abs(binding.energy.flat_derivative(np.ones(len(x))) - exact)))
    slopes = np.log2(np.asarray(errors[:-1]) / np.asarray(errors[1:]))
    assert np.all(slopes > 1.9), f"K * 1 converges at slopes {slopes}, errors {errors}, not second order"
    assert errors[-1] < 2.5e-4, f"K * 1 is {errors[-1]:.3e} off at refinement 5"


def test_what_hjb_fem_p1_does_not_bind_is_refused_by_name():
    """Each refusal by its owner and its reason:

    - P2, out of this stage; on triangles its vertex row sums are 0 and on tetrahedra negative, which the message
      says, and on a line, where they are positive (Simpson's weights), it says only the stage;
    - a quad or hexahedron mesh, at either order, which holds no element volumes to read the domain's measure from;
    - a user cost with ``bind`` alone, by ``bind_on_measure``; ``use_fft=True``, by ``InteractionCost``;
    - a PERIODIC condition, by the BC gate (#1456) before any binding is read;
    - a weak-form subclass whose measure is not its dofs in their order;
    - the meshless solver on an implicit domain, which binds none of its own.
    """
    from mfgarchon.geometry import Hyperrectangle
    from mfgarchon.geometry.boundary.conditions import BoundaryConditions
    from mfgarchon.geometry.boundary.types import BCSegment, BCType

    cost = PotentialCost(lambda x: 0.0)
    for d, reason in ((1, r"stage 1b-2\)\. Solve with order=1"), (2, "zero at the vertices"), (3, "negative at the")):
        with pytest.raises(NotImplementedError, match=rf"HJBFEMSolver\(order=2\) binds .* P1 only.*{reason}"):
            HJBFEMSolver(_problem(cost=cost, d=d), order=2)
    for element, d, mesh in (
        ("quad", 2, skfem.MeshQuad.init_tensor(*[np.linspace(0.0, 1.0, 5)] * 2)),
        ("hexahedron", 3, skfem.MeshHex.init_tensor(*[np.linspace(0.0, 1.0, 3)] * 3)),
    ):
        problem = _problem(cost=cost, d=d)
        problem.geometry.mesh_data = skfem_to_meshdata(mesh)
        for order in (1, 2):
            with pytest.raises(NotImplementedError, match=rf"this mesh of {element} elements has none"):
                HJBFEMSolver(problem, order=order)

    class BindOnly:
        def bind(self, problem):
            return PotentialCost(lambda x: 0.0).bind(problem)

    with pytest.raises(NotImplementedError, match=r"BindOnly has no bind_on\(measure\)"):
        HJBFEMSolver(_problem(cost=BindOnly()))
    fft = InteractionCost(GaussianKernel(amplitude=1.0, length_scale=0.3), use_fft=True)
    with pytest.raises(ValueError, match=r"use_fft=True\) binds on LumpedMassPointMeasure"):
        HJBFEMSolver(_problem(cost=fft))
    periodic = BoundaryConditions(
        dimension=2,
        segments=[BCSegment(name="x", bc_type=BCType.PERIODIC, boundary="x_min")],
        default_bc=BCType.NO_FLUX,
    )
    with pytest.raises(NotImplementedError, match=r"does not support boundary condition type\(s\) \['PERIODIC'\]"):
        HJBFEMSolver(_problem(cost=cost, bc=periodic))

    from mfgarchon.alg.numerical.fem.discretization import LumpedMassPointMeasure

    class Reversed(HJBFEMSolver):
        def _coupling_point_measure(self):
            measure = super()._coupling_point_measure()
            return LumpedMassPointMeasure(measure.points[::-1], measure.weights[::-1], measure.domain_measure)

    with pytest.raises(ValueError, match=r"whose points are not this solver's dofs in their order"):
        Reversed(_problem(cost=cost))

    from mfgarchon.alg.numerical.meshless_galerkin import MeshlessGalerkinHJBSolver

    points = np.stack(np.meshgrid(np.linspace(0, 1, 8), np.linspace(0, 1, 8)), -1).reshape(-1, 2)
    on_box = MFGProblem(
        model=Model(
            hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
            volatility=0.3,
            coupling_cost=cost,
        ),
        domain=Hyperrectangle(np.array([[0.0, 1.0], [0.0, 1.0]])),
        conditions=Conditions(u_terminal=lambda x: 0.0, m_initial=lambda x: 1.0, T=0.2),
        Nt=3,
    )
    on_box.components.boundary_conditions = no_flux_bc(dimension=2)
    with pytest.raises(
        NotImplementedError, match=r"MeshlessGalerkinHJBSolver solves on its own discretization and binds no"
    ):
        MeshlessGalerkinHJBSolver(on_box, points, delta=0.35)


@pytest.mark.parametrize("reader", ["MFGResidual", "RegimeSwitchingIterator", "GraphMFGSolver", "HJBGFDMSolver"])
def test_a_reader_that_passes_no_binding_refuses_by_its_name(reader):
    """Off a grid the problem binds no coupling cost, so a reader that hands the source no solver binding refuses it,
    naming itself, before the first evaluation (for the graph loop, when it composes a node's source): Newton's
    residual, the regime-switching and graph loops on HJB-FEM P1,
    and HJB-GFDM, which binds none of its own, through a coupler on an implicit domain (its binding is #2761). Each of
    these constructed nothing at e6c9df6d: the problem refused the cost."""
    from mfgarchon.geometry import Hyperrectangle

    refusal = rf"{reader} has no binding of Model\(coupling_cost=\.\.\.\) on this"
    cost = InteractionCost(GaussianKernel(amplitude=1.0, length_scale=0.3))
    if reader == "HJBGFDMSolver":
        from mfgarchon.alg.numerical.coupling.base_mfg import hjb_coupling_binding
        from mfgarchon.alg.numerical.hjb_solvers import HJBGFDMSolver

        problem = MFGProblem(
            model=Model(
                hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
                volatility=0.3,
                coupling_cost=cost,
            ),
            domain=Hyperrectangle(np.array([[0.0, 1.0], [0.0, 1.0]])),
            conditions=Conditions(u_terminal=lambda x: 0.0, m_initial=lambda x: 1.0, T=0.2),
            Nt=3,
        )
        problem.components.boundary_conditions = no_flux_bc(dimension=2)
        points = np.stack(np.meshgrid(np.linspace(0, 1, 8), np.linspace(0, 1, 8)), -1).reshape(-1, 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            hjb = HJBGFDMSolver(problem, collocation_points=points)
        with pytest.raises(NotImplementedError, match=rf"{refusal} Hyperrectangle.*#2761"):
            hjb_solve_kwargs(
                problem,
                {"source_term"},
                reader,
                M=np.ones((4, 64)),
                U=np.zeros((4, 64)),
                binding=hjb_coupling_binding(hjb),
            )
        return
    problems = [_problem(cost=cost, nt=3), _problem(cost=cost, nt=3)]
    pairs = [(HJBFEMSolver(p), FPFEMSolver(p)) for p in problems]
    if reader == "MFGResidual":
        from mfgarchon.alg.numerical.coupling import NewtonMFGSolver

        loop = NewtonMFGSolver(problems[0], *pairs[0], picard_warmup=0, newton_max_iterations=1)

        def solve():
            return loop.solve(max_iterations=1, tolerance=1e-14, verbose=False)
    elif reader == "RegimeSwitchingIterator":
        from mfgarchon.alg.numerical.coupling.regime_switching_iterator import RegimeSwitchingIterator
        from mfgarchon.core.regime_switching import RegimeSwitchingConfig

        solve = RegimeSwitchingIterator(
            problems,
            RegimeSwitchingConfig(transition_matrix=np.array([[-0.1, 0.1], [0.2, -0.2]])),
            [h for h, _ in pairs],
            [f for _, f in pairs],
            max_iterations=2,
            tolerance=1e-12,
        ).solve
    else:
        from mfgarchon.alg.numerical.coupling.graph_coupling import AdjacencyCoupling
        from mfgarchon.alg.numerical.coupling.graph_mfg_solver import GraphMFGSolver

        graph = GraphMFGSolver(
            problems,
            AdjacencyCoupling(np.ones((2, 2)) - np.eye(2), alpha=0.05, beta=0.02),
            [h for h, _ in pairs],
            [f for _, f in pairs],
            max_iterations=2,
            tolerance=1e-12,
        )
        M = np.ones((problems[0].Nt + 1, pairs[0][0].n_dof))

        def solve():
            # Composing node 0's source is where the graph solver refuses: before any evaluation, so a refusal deferred
            # to the source's first call would return a callable here.
            return graph._compose_hjb_source(0, [M, M], [M, M], lambda t, x: np.zeros(len(x)))

    with pytest.raises(NotImplementedError, match=rf"{refusal} Mesh2D"):
        solve()


# --- The terminal coupling G, bound on the same measure (#1642, stage 1b's G line) ---------------------------------


def _VT(x):
    """A fifth of ``_V``: the terminal term the G pins use, under which every route's solve stays bounded (at ``_V``
    itself the density reaches 1.9e4 through ``problem.solve()`` and 8.3e19 through a ``BlockIterator``)."""
    return 0.2 * _V(x)


def _g_problem(where=None, g=None, nt=8, **mesh):
    """``_VT`` added to ``u_terminal`` (``"u"``), ``PotentialCost(_VT)`` as G (``"G"``), or ``g`` as G."""
    if where == "G":
        g = PotentialCost(lambda x: float(_VT(x)))

    def terminal(x):
        return float(np.sum((np.asarray(x) - 0.7) ** 2))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # the density's mass on the mesh is not these pins' subject
        return MFGProblem(
            model=Model(
                hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
                volatility=0.3,
                terminal_coupling=g,
            ),
            domain=_mesh(**mesh),
            conditions=Conditions(
                u_terminal=(lambda x: terminal(x) + float(_VT(x))) if where == "u" else terminal,
                m_initial=lambda x: float(np.exp(-10 * np.sum((np.asarray(x) - 0.3) ** 2))),
                T=0.5,
            ),
            Nt=nt,
        )


@pytest.mark.parametrize("route", ["problem.solve()", "BlockIterator"])
def test_a_linear_terminal_coupling_solves_as_the_same_terminal_cost_on_hjb_fem(route):
    """G = int V m(T) has ``delta G / delta m = V``, so ``PotentialCost(V)`` as G and V added to ``u_terminal`` pose one
    game. On HJB-FEM P1 the binding gives V at the dofs, the points ``u_terminal`` is read at, so u(T) is g + V either
    way, and a solve through any route that hands the solver's binding to ``terminal_value`` on a mesh is bitwise the
    same. One row per such route: ``FixedPointIterator`` (``problem.solve()`` builds one with HJB-FEM and FP-FEM here)
    and ``BlockIterator``. ``FictitiousPlayIterator`` and the solver's own ``solve()`` hand it over too, and run on no
    mesh: the first takes a ``CartesianGrid`` only, and HJB-FEM refuses a standalone Picard solve (#2642)."""
    from mfgarchon.alg.numerical.coupling.block_iterators import BlockIterator

    def solve(where):
        problem = _g_problem(where)
        if route == "problem.solve()":
            return np.asarray(problem.solve(config=_PICARD).U)
        loop = BlockIterator(problem, HJBFEMSolver(problem), FPFEMSolver(problem))
        result = loop.solve(max_iterations=5, tolerance=1e-12, verbose=False)
        return np.asarray(result.U)

    logging.disable(logging.WARNING)  # the FP's positivity-clip notices are not this pin's subject
    try:
        gap = np.max(np.abs(solve("u") - solve("G")))
        dropped = np.max(np.abs(solve("u") - solve(None)))
    finally:
        logging.disable(logging.NOTSET)
    assert gap == 0.0, f"{route}: PotentialCost(V) as G and V in u_terminal differ by {gap:.3e} on HJB-FEM P1"
    assert dropped > 0.1, f"the fixture does not see V: dropping it moves U by {dropped:.3e}"


def test_a_constant_kernel_as_g_shifts_u_at_t_by_c_times_the_mass():
    """With K = c, G = (1/2) int int c m m has ``delta G / delta m = c * mass(m(T))``, the same at every dof: u(T) is g
    shifted by exactly that, the mass the P1 interpolant's integral, assembled by scikit-fem's own quadrature rather
    than read from the binding's weights."""
    c = 0.7
    problem = _g_problem(g=InteractionCost(_Constant(c)))
    hjb = HJBFEMSolver(problem)
    m_T = _frozen(hjb, problem)[0][-1]
    shift = problem.terminal_value(m_T, binding=hjb.terminal_binding()) - problem.get_u_terminal()
    mass = skfem.Functional(lambda w: w["m"]).assemble(hjb.basis, m=hjb.basis.interpolate(m_T))
    assert np.max(np.abs(shift - c * mass)) <= 1e-13 * c * mass, (
        f"the constant kernel shifts u(T) by {shift.mean():.6f}, not c * mass = {c * mass:.6f}"
    )


def _g_refused(case):
    """The call ``test_a_terminal_coupling_that_cannot_be_read_is_refused_by_name`` refuses, for ``case``."""
    from mfgarchon.alg.numerical.coupling.source_composition import CouplingBinding

    if case == "HJB-FEM P2":
        return lambda: HJBFEMSolver(_g_problem("G"), order=2)
    if case == "the meshless solver":
        from mfgarchon.alg.numerical.meshless_galerkin import MeshlessGalerkinHJBSolver

        problem = _g_problem("G")
        points = np.asarray(problem.geometry.get_spatial_grid(), dtype=float)
        return lambda: MeshlessGalerkinHJBSolver(problem, collocation_points=points, delta=0.3)
    if case in ("GraphMFGSolver", "RegimeSwitchingIterator", "MFGResidual"):
        problems = [_g_problem("G"), _g_problem("G")]
        hjbs, fps = [HJBFEMSolver(p) for p in problems], [FPFEMSolver(p) for p in problems]
        if case == "GraphMFGSolver":
            from mfgarchon.alg.numerical.coupling.graph_coupling import AdjacencyCoupling
            from mfgarchon.alg.numerical.coupling.graph_mfg_solver import GraphMFGSolver

            coupling = AdjacencyCoupling(np.array([[0.0, 1.0], [1.0, 0.0]]))
            return lambda: GraphMFGSolver(problems, coupling, hjbs, fps, max_iterations=2).solve()
        if case == "RegimeSwitchingIterator":
            from mfgarchon.alg.numerical.coupling.regime_switching_iterator import RegimeSwitchingIterator
            from mfgarchon.core.regime_switching import RegimeSwitchingConfig

            config = RegimeSwitchingConfig(transition_matrix=np.array([[-1.0, 1.0], [1.0, -1.0]]))
            return lambda: RegimeSwitchingIterator(problems, config, hjbs, fps, max_iterations=2).solve()
        from mfgarchon.alg.numerical.coupling.mfg_residual import MFGResidual

        return lambda: MFGResidual(problems[0], hjbs[0], fps[0])
    if case == "MultiPopulationIterator":
        from mfgarchon.alg.numerical.coupling.multi_population_iterator import MultiPopulationIterator
        from mfgarchon.core.multi_population import MultiPopulationProblem

        problem = _g_problem("G")
        loop = MultiPopulationIterator(
            MultiPopulationProblem(populations=[problem], population_names=["a"]),
            [HJBFEMSolver(problem)],
            [FPFEMSolver(problem)],
        )
        return lambda: loop.solve(max_iterations=2, tolerance=1e-12)
    problem = _g_problem("G")
    m_T = np.ones(problem.get_u_terminal().size)
    binding = HJBFEMSolver(problem).terminal_binding()
    if case == "no binding":
        return lambda: problem.terminal_value(m_T)
    if case == "a binding on other points":
        moved = CouplingBinding(binding.energy, binding.nodes + 0.01, binding.cost)
        return lambda: problem.terminal_value(m_T, binding=moved)
    if case == "another problem's G":
        return lambda: problem.terminal_value(m_T, binding=HJBFEMSolver(_g_problem("G")).terminal_binding())
    if case == "two bindings of one G":
        from mfgarchon.geometry import TensorProductGrid

        grid = TensorProductGrid(bounds=[(0.0, 1.0)] * 2, Nx_points=[9, 9], boundary_conditions=no_flux_bc(dimension=2))
        on_grid = MFGProblem(
            model=Model(
                hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
                volatility=0.3,
                terminal_coupling=problem.declared_terminal_coupling,
            ),
            domain=grid,
            conditions=Conditions(u_terminal=lambda x: 0.0, m_initial=lambda x: 1.0, T=0.5),
            Nt=4,
        )
        return lambda: on_grid.terminal_value(np.ones(81), binding=binding)
    raise AssertionError(case)


_G_REFUSALS = {
    "HJB-FEM P2": r"HJBFEMSolver\(order=2\) binds .*Model\(terminal_coupling=\.\.\.\) on P1 only",
    "the meshless solver": r"MeshlessGalerkinHJBSolver solves on its own discretization and binds no "
    r"Model\(terminal_coupling=\.\.\.\)",
    "GraphMFGSolver": r"GraphMFGSolver has no binding of Model\(terminal_coupling=\.\.\.\) on this Mesh2D",
    "RegimeSwitchingIterator": r"RegimeSwitchingIterator has no binding of Model\(terminal_coupling=\.\.\.\) on this",
    "MFGResidual": r"MFGResidual has no binding of Model\(terminal_coupling=\.\.\.\) on this Mesh2D",
    "MultiPopulationIterator": r"population\(s\) \[0\] set Model\(terminal_coupling=\.\.\.\)",
    "no binding": r"MFGProblem.terminal_value has no binding of Model\(terminal_coupling=\.\.\.\) on this Mesh2D",
    "a binding on other points": r"on 81 points that are not this problem's 81, in their order, compared bitwise",
    "another problem's G": r"a binding of another problem's terminal coupling",
    "two bindings of one G": r"for a problem that binds its own on its grid: one G with two bindings",
}


@pytest.mark.parametrize("case", list(_G_REFUSALS))
def test_a_terminal_coupling_that_cannot_be_read_is_refused_by_name(case):
    """Off a grid the problem declares G and binds none, and every reader that cannot read it refuses it by name (#1642,
    stage 1b's G line): a solver with no point measure, P2 and the meshless solver, at construction; the routes that
    hand ``terminal_value`` no binding (graph, regime switching, the Newton residual), as they refuse F; the
    multi-population iterator, where G is single-population (the user's ruling 5); and ``terminal_value`` itself, for
    no binding, a binding on other points, a binding of another problem's G, and a binding beside the problem's own on
    a grid, one G with two bindings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        call = _g_refused(case)
        with pytest.raises((NotImplementedError, ValueError), match=_G_REFUSALS[case]):
            call()


def test_a_network_problem_refuses_a_terminal_binding():
    """``NetworkMFGProblem.terminal_value``, the protocol's second implementer, takes ``binding`` and ``reader`` as
    ``MFGProblem.terminal_value`` does, since ``FixedPointIterator`` hands both on the network route, and refuses a
    binding, since a network binds no terminal coupling (#1642, #2568)."""
    from mfgarchon.extensions.topology import NetworkMFGProblem

    terminal_value = NetworkMFGProblem.terminal_value
    assert list(inspect.signature(terminal_value).parameters) == ["self", "m_terminal", "binding", "reader"]
    with pytest.raises(ValueError, match=r"FixedPointIterator was handed a binding of Model\(terminal_coupling"):
        terminal_value(object.__new__(NetworkMFGProblem), None, binding=object(), reader="FixedPointIterator")
