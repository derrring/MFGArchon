"""One owner decides which BC object is "the BC" of a problem (#2530, #2512 row B9).

`MFGProblem.get_boundary_conditions` holds the order among the channels that can carry the shared BC -- the
geometry's and ``components.boundary_conditions`` -- and what happens when none does. A solver handed its own BC
uses that one; every other reader asks the owner. Before this, five readers each had their own order: FP-FDM and
FP-FVM read the components BC first, the HJB solvers, FP-SL and the particle solver read the geometry's, so on
#2530's legacy route the FP absorbed at an exit the HJB never imposed.

Rulings of the audit session (2026-10-13), plan at #2530 comment 6133804095:
- D1: a components BC and a geometry BC both given are two declarations of one BC: refused when they differ, the
  geometry's when they agree (`BoundaryConditions.declares_the_same_as`).
- D2: the ``spatial_bounds=`` route builds its grid with the components BC, and the owner returns that grid's object.
- D4: no BC on any channel is refused by name (#2589); a torus answers ``periodic_bc``; a graph geometry's node BCs
  are its own (#1471).
"""

from __future__ import annotations

import logging

import pytest

import numpy as np

from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
from mfgarchon.alg.numerical.fp_solvers.fp_fvm import FPFVMSolver
from mfgarchon.alg.numerical.fp_solvers.fp_particle import FPParticleSolver
from mfgarchon.alg.numerical.fp_solvers.fp_semi_lagrangian_adjoint import FPSLSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.alg.numerical.hjb_solvers.hjb_gfdm import HJBGFDMSolver
from mfgarchon.alg.numerical.hjb_solvers.hjb_semi_lagrangian import HJBSemiLagrangianSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import (
    BCSegment,
    BCType,
    BoundaryConditions,
    PeriodicGridConvention,
    dirichlet_bc,
    no_flux_bc,
    periodic_bc,
)
from mfgarchon.operators.differential.gradient import value_gradient

N, NT, T, SIGMA = 21, 10, 0.5, 0.4
TWICE = "declares its boundary conditions twice"


def _components(bc=None):
    kw = {
        "hamiltonian": SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
        "m_initial": lambda x: 1.0,
        "u_terminal": lambda x: 0.0,
    }
    if bc is not None:
        kw["boundary_conditions"] = bc
    return MFGComponents(**kw)


def _legacy(bc):
    """#2530's route: ``spatial_bounds=``, the BC in components."""
    return MFGProblem(
        spatial_bounds=[(0.0, 1.0)],
        spatial_discretization=[N - 1],
        components=_components(bc),
        T=T,
        Nt=NT,
        volatility=SIGMA,
    )


def _on_grid(grid_bc, components_bc=None):
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[N], boundary_conditions=grid_bc)
    return MFGProblem(geometry=grid, components=_components(components_bc), T=T, Nt=NT, volatility=SIGMA)


# --- D1: two declarations of one BC -----------------------------------------------------------------------------


def test_a_components_bc_that_differs_from_the_grids_is_refused_by_name():
    problem = _on_grid(no_flux_bc(dimension=1), dirichlet_bc(dimension=1, value=0.7))
    with pytest.raises(ValueError, match=TWICE):
        problem.get_boundary_conditions()
    for build in (FPFDMSolver, FPFVMSolver, lambda p: FPParticleSolver(p, num_particles=100, seed=0)):
        with pytest.raises(ValueError, match=TWICE):
            build(problem)


def test_the_same_bc_on_both_channels_is_the_grids():
    bc = dirichlet_bc(dimension=1, value=0.7)
    problem = _on_grid(bc, bc)
    assert problem.get_boundary_conditions() is problem.geometry.get_boundary_conditions()


def test_a_bc_declared_without_a_dimension_agrees_with_its_bound_copy():
    """A grid stores a bound copy, with its dimension filled in: content, not identity, is the test."""
    unbound = no_flux_bc()
    problem = _on_grid(unbound, unbound)
    stored = problem.geometry.get_boundary_conditions()
    assert stored is not unbound
    assert unbound.dimension is None
    assert stored.dimension == 1
    assert problem.get_boundary_conditions() is stored


@pytest.mark.parametrize(
    ("field", "other"),
    [
        ("default_bc", BCType.NO_FLUX),
        ("default_value", 0.3),
        ("domain_bounds", np.array([[0.0, 2.0]])),
        ("domain_sdf", lambda x: np.asarray(x)[..., 0] - 0.5),
        ("corner_strategy", "average"),
        ("corner_mollification_radius", 0.2),
        ("periodic_convention", PeriodicGridConvention.ENDPOINT_INCLUSIVE),
    ],
)
def test_declares_the_same_as_refuses_a_difference_in_each_declared_field(field, other):
    """Review 1 of #2638: each declared field outside the segments is compared, so a components BC differing from the
    grid's only in, say, its fall-through ``default_bc`` is not taken for the same declaration and dropped."""
    from dataclasses import replace

    base = BoundaryConditions(
        dimension=1,
        segments=[BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=0.7, boundary="x_max")],
        default_bc=BCType.DIRICHLET,
        default_value=0.0,
        domain_bounds=np.array([[0.0, 1.0]]),
        periodic_convention=PeriodicGridConvention.ENDPOINT_EXCLUSIVE,
    )
    assert base.declares_the_same_as(replace(base))
    assert not base.declares_the_same_as(replace(base, **{field: other}))


def test_declares_the_same_as_compares_content_field_by_field():
    exit_ = dirichlet_bc(dimension=1, value=0.7)
    assert exit_.declares_the_same_as(dirichlet_bc(dimension=1, value=0.7))
    assert not exit_.declares_the_same_as(dirichlet_bc(dimension=1, value=0.6))
    assert not exit_.declares_the_same_as(dirichlet_bc(dimension=2, value=0.7))
    assert exit_.declares_the_same_as(dirichlet_bc(value=0.7))  # an unset dimension agrees


def test_a_mesh_with_a_bc_attached_declares_it_as_a_grid_does():
    """A mesh's attached BC is the geometry's declaration (`UnstructuredMesh.has_explicit_boundary_conditions`):
    a components BC beside it is the second declaration D1 compares, where it used to outrank the mesh's."""
    from mfgarchon.geometry.meshes import Mesh1D

    mesh = Mesh1D(bounds=(0.0, 1.0), num_elements=10)
    mesh.generate_mesh()
    assert not mesh.has_explicit_boundary_conditions()
    mesh.boundary_conditions = no_flux_bc(dimension=1)
    assert mesh.has_explicit_boundary_conditions()
    problem = MFGProblem(geometry=mesh, components=_components(dirichlet_bc(dimension=1, value=0.7)), T=T, Nt=NT)
    with pytest.raises(ValueError, match=TWICE):
        problem.get_boundary_conditions()


# --- D2: the legacy route --------------------------------------------------------------------------------------


def test_the_legacy_route_builds_its_grid_with_the_components_bc():
    exit_ = dirichlet_bc(dimension=1, value=0.7)
    problem = _legacy(exit_)
    stored = problem.geometry.get_boundary_conditions()
    assert problem.get_boundary_conditions() is stored
    assert stored.declares_the_same_as(exit_)
    assert _legacy(None).get_boundary_conditions().declares_the_same_as(no_flux_bc(dimension=1))


def test_inside_using_resolved_bc_the_components_bc_is_resolved_with_the_grids():
    """The context sets the provider-resolved BC on the grid; the components BC, the same declaration, is resolved
    with it, or the owner would compare resolved against unresolved and refuse (#2530 D1)."""
    from mfgarchon.geometry.boundary import BaseBCValueProvider

    class _Half(BaseBCValueProvider):
        def compute(self, state):
            return 0.5

    with_provider = BoundaryConditions(
        dimension=1, segments=[BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=_Half())]
    )
    problem = _legacy(with_provider)
    with problem.using_resolved_bc({"geometry": problem.geometry}):
        inside = problem.get_boundary_conditions()
        assert inside.segments[0].value == 0.5
    after = problem.get_boundary_conditions()
    assert isinstance(after.segments[0].value, _Half)
    assert problem.components.boundary_conditions is with_provider


# --- D4: no BC ---------------------------------------------------------------------------------------------------


def test_a_mesh_with_no_bc_on_any_channel_is_refused_by_name_and_the_advice_runs():
    """#2589's probe: a mesh with no BC reached `has_providers()` on None from inside the Picard loop."""
    from mfgarchon.geometry.meshes import Mesh1D

    def problem_on(mesh):
        return MFGProblem(geometry=mesh, components=_components(), T=0.5, Nt=5, volatility=SIGMA)

    mesh = Mesh1D(bounds=(0.0, 1.0), num_elements=10)
    mesh.generate_mesh()
    with pytest.raises(ValueError, match=r"no channel declares boundary conditions.*Mesh1D.*#2589"):
        problem_on(mesh).solve(max_iterations=3, verbose=False)
    mesh.boundary_conditions = no_flux_bc(dimension=1)  # the message's advice: mesh.boundary_conditions = bc
    assert problem_on(mesh).solve(max_iterations=3, verbose=False) is not None


def test_a_torus_answers_periodic_bc():
    from mfgarchon.geometry.implicit import Hyperrectangle

    torus = Hyperrectangle(np.array([[0.0, 1.0], [0.0, 1.0]]), periodic_dims=(0, 1))
    bc = MFGProblem(geometry=torus, components=_components(), T=T, Nt=NT).get_boundary_conditions()
    assert bc.type == "periodic"
    assert bc.dimension == 2
    box = Hyperrectangle(np.array([[0.0, 1.0], [0.0, 1.0]]), periodic_dims=(0,))
    with pytest.raises(ValueError, match=r"no channel declares boundary conditions.*on a geometry that holds none"):
        MFGProblem(geometry=box, components=_components(), T=T, Nt=NT).get_boundary_conditions()
    # The refusal's advice for a geometry that holds none: the components BC is then the BC.
    declared = no_flux_bc(dimension=2)
    assert MFGProblem(geometry=box, components=_components(declared), T=T, Nt=NT).get_boundary_conditions() is declared


def test_a_point_cloud_declares_none_so_the_components_bc_is_the_bc():
    """Review 1 of #2638: a `GeometryProtocol` outside the `Geometry` hierarchy has no
    ``has_explicit_boundary_conditions``; its accessor answers None when it declares none, and the owner reads that
    instead of raising AttributeError."""
    from mfgarchon.geometry.implicit.point_cloud import PointCloudGeometry

    cloud = PointCloudGeometry(np.linspace(0.0, 1.0, N).reshape(-1, 1))
    declared = no_flux_bc(dimension=1)
    assert (
        MFGProblem(geometry=cloud, components=_components(declared), T=T, Nt=NT).get_boundary_conditions() is declared
    )
    with pytest.raises(ValueError, match=r"no channel declares boundary conditions.*PointCloudGeometry"):
        MFGProblem(geometry=cloud, components=_components(), T=T, Nt=NT).get_boundary_conditions()


def test_a_components_bc_on_a_network_is_refused_where_the_graph_owns_node_bcs():
    """A graph geometry's node BCs are its own (#1471): the owner returns them, None included (pinned in
    test_network_hamiltonian_pinning.py), and a continuum BC in components is refused rather than dropped."""
    pytest.importorskip("igraph")
    from mfgarchon.extensions.topology import NetworkMFGProblem
    from mfgarchon.geometry.graph.network_geometry import GridNetwork

    net = GridNetwork(width=3, height=3)
    net.create_network()
    problem = NetworkMFGProblem(geometry=net, T=0.5, Nt=4)
    assert problem.get_boundary_conditions() is None
    problem.components.boundary_conditions = no_flux_bc(dimension=1)
    with pytest.raises(ValueError, match="#1471"):
        problem.get_boundary_conditions()


# --- Form (b): a solver's operators are built on the BC it resolved ----------------------------------------------


def _x_field(shape):
    """u = x along axis 0: a mirror ghost and a periodic wrap give different wall gradients."""
    return np.broadcast_to(np.linspace(0.0, 1.0, shape[0]).reshape(-1, *([1] * (len(shape) - 1))), shape).copy()


@pytest.mark.parametrize("scheme", ["gradient_centered", "gradient_upwind"])
def test_hjb_fdm_builds_its_gradients_on_the_bc_it_resolved(scheme):
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[9, 9], boundary_conditions=no_flux_bc(dimension=2)
    )
    problem = MFGProblem(geometry=grid, components=_components(), T=T, Nt=NT, volatility=SIGMA)
    solver = HJBFDMSolver(problem, advection_scheme=scheme)
    solver._boundary_conditions = periodic_bc(dimension=2)  # a BC handed to the solver, which outranks the grid's
    U = _x_field((9, 9))
    if scheme == "gradient_centered":
        got = solver._compute_gradients_nd(U, time=0.0)[0]
        on_own = grid.get_gradient_operator(scheme="central", time=0.0, bc=solver.get_boundary_conditions())[0](U)
        on_grid = grid.get_gradient_operator(scheme="central", time=0.0)[0](U)
    else:
        got = solver._one_sided_differences_nd(U, time=0.0)[0][0]
        on_own = grid.get_gradient_operator(scheme="backward", time=0.0, bc=solver.get_boundary_conditions())[0](U)
        on_grid = grid.get_gradient_operator(scheme="backward", time=0.0)[0](U)
    assert not np.allclose(on_own, on_grid)
    np.testing.assert_allclose(got, on_own)


def test_hjb_sl_builds_its_gradient_on_the_bc_it_resolved():
    problem = _on_grid(no_flux_bc(dimension=1))
    solver = HJBSemiLagrangianSolver(problem)
    solver._boundary_conditions = periodic_bc(dimension=1)
    U = np.linspace(0.0, 1.0, N)
    on_own = value_gradient(problem.geometry, U, 0.0, bc=solver.get_boundary_conditions())[0]
    on_grid = value_gradient(problem.geometry, U, 0.0)[0]
    assert not np.allclose(on_own, on_grid)
    np.testing.assert_allclose(solver._compute_gradient(U, time=0.0), on_own)


def test_fp_sl_builds_its_gradient_of_u_on_the_bc_it_resolved():
    problem = _on_grid(no_flux_bc(dimension=1))
    solver = FPSLSolver(problem, boundary_conditions=periodic_bc(dimension=1))
    U = np.linspace(0.0, 1.0, N)
    on_own = value_gradient(problem.geometry, U, 0.0, bc=solver.get_boundary_conditions())[0]
    on_grid = value_gradient(problem.geometry, U, 0.0)[0]
    assert not np.allclose(on_own, on_grid)
    np.testing.assert_allclose(solver._gradient(U, 0.0)[0], on_own)


# --- #2530's oracle: each solver imposes the BC the owner returns -------------------------------------------------


def _hjb(make, problem, U_T):
    return np.asarray(make(problem).solve_hjb_system(np.ones((NT + 1, N)), U_T, np.tile(U_T, (NT + 1, 1))))


_X = np.linspace(0.0, 1.0, N)


def _fp(make, problem, kind):
    """m0 = 1 + x: data that tells periodic from no-flux, which uniform data does not."""
    solver = make(problem)
    m0 = 1.0 + _X
    if kind == "sl":
        return np.asarray(solver.solve_fp_system(m0, potential_field=np.zeros((NT + 1, N))))
    return np.asarray(solver.solve_fp_system(m0, drift_field=np.zeros((NT + 1, N))))


_EXIT = dirichlet_bc(dimension=1, value=0.7)
_SOLVERS = {
    "HJB-FDM": (lambda p: _hjb(HJBFDMSolver, p, _X.copy()), _EXIT),
    "HJB-SL": (lambda p: _hjb(HJBSemiLagrangianSolver, p, _X.copy()), periodic_bc(dimension=1)),
    "FP-FDM": (lambda p: _fp(FPFDMSolver, p, "fdm"), _EXIT),
    "FP-FVM": (lambda p: _fp(FPFVMSolver, p, "fvm"), periodic_bc(dimension=1)),
    "FP-SL": (lambda p: _fp(FPSLSolver, p, "sl"), periodic_bc(dimension=1)),
    "FP-particle": (lambda p: _fp(lambda q: FPParticleSolver(q, num_particles=2000, seed=2530), p, "particle"), _EXIT),
}


@pytest.mark.parametrize("name", list(_SOLVERS))
def test_on_the_legacy_route_each_solver_solves_the_bc_the_owner_returns(name):
    """Route A of the plan: the components BC on ``spatial_bounds=``. Each solver's solution equals its solution on a
    grid that declares the same BC, and differs from the no-flux grid's -- which is what the legacy route used to
    solve for every solver but FP-FDM. Solvers that refuse DIRICHLET (HJB-SL, FP-SL, FP-FVM) are read with
    periodic against no-flux. Route B, a grid BC and a different components BC, is refused (D1)."""
    run, bc = _SOLVERS[name]
    legacy = run(_legacy(bc))
    np.testing.assert_allclose(legacy, run(_on_grid(bc)))
    assert not np.allclose(legacy, run(_on_grid(no_flux_bc(dimension=1))))
    with pytest.raises(ValueError, match=TWICE):
        run(_on_grid(no_flux_bc(dimension=1), bc))


def test_hjb_gfdm_solves_the_bc_the_owner_returns_once_newton_converges(mfg_caplog):
    """HJB-GFDM's default Newton does not converge against a wall datum that jumps from U_T (#2610), and a wall read
    from an unconverged solve says nothing about which BC was resolved. So the datum is U_T = 0.7 + x(1 - x), which
    equals the exit's 0.7 at both walls, and the solver's own verdict -- no "did not converge" record -- is asserted
    before the walls are read. Its logger does not propagate, so the records are counted on it, and the absence is
    paired with a presence on the same logger: #2610's jumping datum, U_T = 0 against the exit, which does not
    converge."""
    x = np.linspace(0.0, 1.0, N)
    U_T = 0.7 + x * (1.0 - x)

    def run(problem):
        return _hjb(lambda p: HJBGFDMSolver(p, collocation_points=p.geometry.get_spatial_grid()), problem, U_T)

    with mfg_caplog.at_level(logging.WARNING, logger="mfgarchon.alg.numerical.hjb_solvers.hjb_gfdm"):
        legacy, exit_grid, no_flux = run(_legacy(_EXIT)), run(_on_grid(_EXIT)), run(_on_grid(no_flux_bc(dimension=1)))
    unconverged = [r for r in mfg_caplog.records if "did not converge" in r.getMessage()]
    assert unconverged == [], f"{len(unconverged)} unconverged steps: the walls below would mean nothing"
    with mfg_caplog.at_level(logging.WARNING, logger="mfgarchon.alg.numerical.hjb_solvers.hjb_gfdm"):
        _hjb(lambda p: HJBGFDMSolver(p, collocation_points=p.geometry.get_spatial_grid()), _on_grid(_EXIT), np.zeros(N))
    assert any("did not converge" in r.getMessage() for r in mfg_caplog.records), "the instrument saw no record"
    np.testing.assert_allclose(legacy[0, [0, -1]], [0.7, 0.7], atol=1e-10)
    np.testing.assert_allclose(legacy, exit_grid)
    assert abs(no_flux[0, 0] - 0.7) > 0.05
    assert abs(no_flux[0, -1] - 0.7) > 0.05
    with pytest.raises(ValueError, match=TWICE):
        run(_on_grid(no_flux_bc(dimension=1), _EXIT))


# --- Every geometry type x every channel state (review 1 of #2638: B1's class) ------------------------------------
#
# The owner branches on the geometry's kind (a graph, a `Geometry`, another `GeometryProtocol`), on whether it holds a
# BC, whether it answers a default of its own, and whether it is periodic on every axis; and on the components BC.
# Every concrete `GeometryProtocol` implementer in `mfgarchon.geometry` gets a builder here or a reason it has none,
# and `test_every_geometry_type_has_a_builder_or_a_reason` fails on a new type until it does. Each cell's expected
# answer is derived from the rulings and from what the type does: whether it holds a BC is measured by attaching
# one through its own channel and reading it back. Blind spot: a protocol implementer that keeps its members as
# instance attributes fails the class-level check below; `git grep "def get_boundary_conditions"` is the other net.

_STATES = ("no BC anywhere", "components only", "geometry only", "both, the same", "both, differing")


def _concrete_geometry_types():
    import importlib
    import inspect
    import pkgutil

    import mfgarchon.geometry as package
    from mfgarchon.geometry.base import Geometry
    from mfgarchon.geometry.protocol import GeometryProtocol

    for info in pkgutil.walk_packages(package.__path__, "mfgarchon.geometry."):
        importlib.import_module(info.name)
    members = GeometryProtocol.__protocol_attrs__
    found = {}
    for cls in _all_subclasses(object):
        module = vars(cls).get("__module__")
        if not isinstance(module, str) or not module.startswith("mfgarchon.geometry") or inspect.isabstract(cls):
            continue
        if vars(cls).get("_is_protocol", False):  # GeometryProtocol and its kin describe types; they are none
            continue
        if issubclass(cls, Geometry) or all(hasattr(cls, name) for name in members):
            found[f"{cls.__module__}.{cls.__qualname__}"] = cls
    return found


def _all_subclasses(root):
    seen, stack = set(), list(type.__subclasses__(root))
    while stack:
        cls = stack.pop()
        if cls not in seen:
            seen.add(cls)
            stack.extend(type.__subclasses__(cls))
    return seen


def _box():
    from mfgarchon.geometry.implicit import Hyperrectangle

    return Hyperrectangle(np.array([[0.0, 1.0], [0.0, 1.0]]))


def _disc():
    from mfgarchon.geometry.implicit import Hypersphere

    return Hypersphere(center=[0.5, 0.5], radius=0.3)


def _generated(mesh):
    mesh.generate_mesh()
    return mesh


def _network(net):
    net.create_network()
    return net


def _cloud_that_holds_a_bc():
    """A `GeometryProtocol` outside the `Geometry` hierarchy whose accessor answers a BC it stores. No library type
    does; it pins the owner's arm for one (review 2 of #2638)."""
    from mfgarchon.geometry.implicit.point_cloud import PointCloudGeometry

    class CloudThatHoldsABC(PointCloudGeometry):
        def __init__(self, points):
            super().__init__(points)
            self.stored = None

        def set_boundary_conditions(self, bc):
            self.stored = bc

        def get_boundary_conditions(self):
            return self.stored

    return CloudThatHoldsABC(np.linspace(0.0, 1.0, 11).reshape(-1, 1))


def _attach(geometry, bc):
    """Put ``bc`` on the geometry's own channel -- its setter, else its ``boundary_conditions`` attribute -- and say
    whether the geometry then answers it, which is what holding a BC means here."""
    if getattr(type(geometry), "set_boundary_conditions", None) is not None:
        geometry.set_boundary_conditions(bc)
    else:
        geometry.boundary_conditions = bc
    answer = geometry.get_boundary_conditions()
    if bc is None or answer is None:
        return False
    if answer is bc:
        return True
    return (
        isinstance(answer, BoundaryConditions)
        and isinstance(bc, BoundaryConditions)
        and answer.declares_the_same_as(bc)
    )


def _geometry_builders():
    """type -> {variant: builder}, or the reason no problem can be built on it."""
    from mfgarchon.geometry.implicit import Hyperrectangle
    from mfgarchon.geometry.implicit.csg_operations import (
        ComplementDomain,
        DifferenceDomain,
        IntersectionDomain,
        UnionDomain,
    )
    from mfgarchon.geometry.implicit.point_cloud import PointCloudGeometry
    from mfgarchon.geometry.meshes import Mesh1D, Mesh2D, Mesh3D

    def complement():
        domain = ComplementDomain(_disc())
        domain.set_bounding_box(np.array([[0.0, 1.0], [0.0, 1.0]]))
        return domain

    def network(kind, **kw):
        def build():
            pytest.importorskip("igraph")
            from mfgarchon.geometry.graph import network_geometry

            return _network(getattr(network_geometry, kind)(**kw))

        return build

    def gmsh_mesh(cls, **kw):
        def build():
            pytest.importorskip("gmsh")
            pytest.importorskip("meshio")
            return _generated(cls(**kw))

        return build

    square = np.array([[0.0, 1.0], [0.0, 1.0]])
    return {
        "mfgarchon.geometry.grids.tensor_grid.TensorProductGrid": {
            "": lambda: TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=no_flux_bc(dimension=1)
            )
        },
        "mfgarchon.geometry.meshes.mesh_1d.Mesh1D": {
            "": lambda: _generated(Mesh1D(bounds=(0.0, 1.0), num_elements=10))
        },
        "mfgarchon.geometry.meshes.mesh_2d.Mesh2D": {
            "": gmsh_mesh(Mesh2D, domain_type="rectangle", bounds=(0.0, 1.0, 0.0, 1.0), mesh_size=0.5)
        },
        "mfgarchon.geometry.meshes.mesh_3d.Mesh3D": {
            "": gmsh_mesh(Mesh3D, domain_type="box", bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0), mesh_size=0.5)
        },
        "mfgarchon.geometry.implicit.hyperrectangle.Hyperrectangle": {
            "a box": _box,
            "periodic on one axis": lambda: Hyperrectangle(square, periodic_dims=(0,)),
            "a torus": lambda: Hyperrectangle(square, periodic_dims=(0, 1)),
        },
        "mfgarchon.geometry.implicit.hypersphere.Hypersphere": {"": _disc},
        "mfgarchon.geometry.implicit.csg_operations.UnionDomain": {"": lambda: UnionDomain([_box(), _disc()])},
        "mfgarchon.geometry.implicit.csg_operations.IntersectionDomain": {
            "": lambda: IntersectionDomain([_box(), _disc()])
        },
        "mfgarchon.geometry.implicit.csg_operations.DifferenceDomain": {"": lambda: DifferenceDomain(_box(), _disc())},
        "mfgarchon.geometry.implicit.csg_operations.ComplementDomain": {"": complement},
        "mfgarchon.geometry.implicit.point_cloud.PointCloudGeometry": {
            "": lambda: PointCloudGeometry(np.linspace(0.0, 1.0, 11).reshape(-1, 1))
        },
        "mfgarchon.geometry.graph.network_geometry.GridNetwork": {"": network("GridNetwork", width=3, height=3)},
        "mfgarchon.geometry.graph.network_geometry.RandomNetwork": {
            "": network("RandomNetwork", num_nodes=6, connection_prob=0.6)
        },
        "mfgarchon.geometry.graph.network_geometry.ScaleFreeNetwork": {"": network("ScaleFreeNetwork", num_nodes=6)},
        "mfgarchon.geometry.graph.network_geometry.CustomNetwork": {
            "": network("CustomNetwork", adjacency_matrix=np.array([[0.0, 1.0], [1.0, 0.0]]))
        },
        "tests: a GeometryProtocol outside Geometry that holds a BC": {"": _cloud_that_holds_a_bc},
        "mfgarchon.geometry.graph.maze_generator.MazeGeometry": (
            "NetworkMFGProblem cannot be built on it: AttributeError, no 'network_data', at 79fae339 as here (#2573)"
        ),
    }


def test_every_geometry_type_has_a_builder_or_a_reason():
    found = _concrete_geometry_types()
    assert "mfgarchon.geometry.grids.tensor_grid.TensorProductGrid" in found  # control: a Geometry subclass
    assert "mfgarchon.geometry.implicit.point_cloud.PointCloudGeometry" in found  # control: protocol-only
    assert sorted(set(found) - set(_geometry_builders())) == []


def _cells():
    for type_name, variants in _geometry_builders().items():
        if isinstance(variants, str):
            continue
        for variant, build in variants.items():
            for state in _STATES:
                yield pytest.param(
                    build, state, id=f"{type_name.rsplit('.', 1)[-1]}{' ' + variant if variant else ''}, {state}"
                )


@pytest.mark.parametrize(("build", "state"), list(_cells()))
def test_the_owner_answers_every_geometry_type_in_every_channel_state(build, state):
    """Each cell is one of the rulings: the declared BC; D1 (two declarations that differ are refused, the same one
    is the geometry's); D4 (none on any channel is refused by name, a torus answers periodic_bc, a graph its own
    node config, None included). A geometry that holds no BC has no "geometry" states. Never an unnamed exception.
    One cell answers a default the library declares for the user -- a grid after ``set_boundary_conditions(None)``
    falls back to its parent class's no-flux -- and is kept by the audit session's ruling; that family is #2639."""
    from mfgarchon.extensions.topology import NetworkMFGProblem
    from mfgarchon.geometry.base import GraphGeometry
    from mfgarchon.geometry.boundary.applicator_graph import GraphBCConfig

    geometry = build()
    dimension = geometry.dimension
    is_graph = isinstance(geometry, GraphGeometry)  # a graph's BC is a node config, not a BoundaryConditions (#1471)
    on_geometry = GraphBCConfig() if is_graph else dirichlet_bc(dimension=dimension, value=0.7)
    if state in _STATES[2:] and not _attach(build(), on_geometry):
        pytest.skip("this geometry holds no BC")
    # The geometry states attach a BC through the type's own channel; the others clear it (a grid always declares
    # one, and set_boundary_conditions(None) is how it holds none).
    _attach(geometry, on_geometry if state in _STATES[2:] else None)
    stored = geometry.get_boundary_conditions()
    declared = {
        _STATES[1]: no_flux_bc(dimension=dimension),
        _STATES[3]: on_geometry,
        _STATES[4]: no_flux_bc(dimension=dimension),
    }.get(state)
    if is_graph:
        problem = NetworkMFGProblem(geometry=geometry, T=T, Nt=4)
        problem.components.boundary_conditions = declared
    else:
        problem = MFGProblem(geometry=geometry, components=_components(declared), T=T, Nt=4)

    if is_graph:
        if declared is not None:
            with pytest.raises(ValueError, match="#1471"):
                problem.get_boundary_conditions()
        else:
            assert problem.get_boundary_conditions() is stored
    elif state == _STATES[4]:
        with pytest.raises(ValueError, match=TWICE):
            problem.get_boundary_conditions()
    elif state in (_STATES[2], _STATES[3]):
        assert problem.get_boundary_conditions() is stored
    elif state == _STATES[1]:
        assert problem.get_boundary_conditions() is declared
    elif stored is not None:  # the geometry answers a default of its own, a new object per call (#2639)
        assert problem.get_boundary_conditions().declares_the_same_as(stored)
    elif len(getattr(geometry, "periodic_dimensions", ())) == dimension:
        assert problem.get_boundary_conditions().type == "periodic"
    else:
        with pytest.raises(ValueError, match="no channel declares boundary conditions"):
            problem.get_boundary_conditions()


def test_a_new_geometry_type_without_a_builder_is_found():
    """The coverage check is live: a geometry type nobody has given a builder or a reason is reported."""
    import gc

    from mfgarchon.geometry.implicit import Hyperrectangle

    planted = type("PlantedDomain", (Hyperrectangle,), {"__module__": "mfgarchon.geometry.implicit.planted"})
    try:
        assert "mfgarchon.geometry.implicit.planted.PlantedDomain" in set(_concrete_geometry_types()) - set(
            _geometry_builders()
        )
    finally:
        del planted
        gc.collect()
    assert "mfgarchon.geometry.implicit.planted.PlantedDomain" not in _concrete_geometry_types()


def test_a_solver_on_an_implicit_domain_solves_the_components_bc():
    """The D4 refusal's remedy for a geometry that holds none, run as a solve (#2638): ``problem.solve()`` refuses an
    implicit domain before any BC (FixedPointIterator takes a grid or a mesh), so HJB-GFDM is run on its own, on a
    Hyperrectangle whose only BC is the components one."""
    from mfgarchon.geometry.implicit import Hyperrectangle

    box = Hyperrectangle(np.array([[0.0, 1.0]]))
    declared = no_flux_bc(dimension=1)
    problem = MFGProblem(geometry=box, components=_components(declared), T=T, Nt=4, volatility=SIGMA)
    points = np.linspace(0.0, 1.0, 11).reshape(-1, 1)
    solver = HJBGFDMSolver(problem, collocation_points=points, boundary_indices=np.array([0, 10]))
    assert solver.get_boundary_conditions() is declared
    U = np.asarray(solver.solve_hjb_system(np.ones((5, 11)), points[:, 0] ** 2, np.tile(points[:, 0] ** 2, (5, 1))))
    assert U.shape == (5, 11)
    assert np.all(np.isfinite(U))
    with pytest.raises(ValueError, match="no channel declares boundary conditions"):
        HJBGFDMSolver(
            MFGProblem(geometry=box, components=_components(), T=T, Nt=4, volatility=SIGMA),
            collocation_points=points,
            boundary_indices=np.array([0, 10]),
        )
