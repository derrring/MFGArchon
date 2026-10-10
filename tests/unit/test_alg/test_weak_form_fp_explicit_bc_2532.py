"""FP-FEM and meshless Galerkin take a BC of their own, and the shared one is still read as before (#2532).

`WeakFormFPSolver` takes ``boundary_conditions=``. A BC handed to it is the FP's own and is used as given, a
NEUMANN refused (`BaseFPSolver._fp_own_bc`); without one the solver reads the problem's shared BC through
`fp_view_of_shared_bc`, where a DIRICHLET is an absorbing exit whose value is the HJB's (#2512, row B3).

- **FP-FEM** gains a prescribed density: an explicit DIRICHLET(g) holds m = g at the wall, and its value
  enters the solve through the condensation's lift, not only the post-solve write.
- **Meshless Galerkin** gains no wall (#2581, A-4: no oracle checks a meshless density). Its explicit route
  solves NO_FLUX and an absorbing DIRICHLET(0) as the shared route does, and refuses a DIRICHLET with a
  value. The route exists so that a shared NEUMANN(g) or ROBIN can be run by giving the FP its own no-flux BC.
"""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, no_flux_bc
from mfgarchon.geometry.boundary.providers import ConstantProvider

SIGMA, T, NT = 0.4, 0.5, 10
EXIT_VALUE = 0.7


def _model() -> Model:
    return Model(
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)), volatility=SIGMA
    )


def _problem(domain, bc: BoundaryConditions, horizon: float = T) -> MFGProblem:
    """``domain(bc)`` builds the domain carrying ``bc`` as the shared BC."""
    from mfgarchon.geometry.meshes.mesh_1d import Mesh1D

    geometry = domain(bc)
    # Unit mass on the domain's own measure: Mesh1D's uniform-cell measure gives a constant c the mass c (n+1)/n.
    c = 16 / 17 if isinstance(geometry, Mesh1D) else 1.0
    return MFGProblem(
        model=_model(),
        domain=geometry,
        conditions=Conditions(
            m_initial=lambda x: c + 0.0 * np.asarray(x, dtype=float)[..., 0], u_terminal=lambda x: 0.0, T=horizon
        ),
        Nt=NT,
    )


def _mesh(dim: int):
    """A builder of the unit interval (16 cells) or square (8 x 8), carrying the BC it is given."""

    def build(bc: BoundaryConditions):
        if dim == 1:
            from mfgarchon.geometry.meshes.mesh_1d import Mesh1D

            geometry = Mesh1D(bounds=(0.0, 1.0), num_elements=16)
            geometry.generate_mesh()
        else:
            import skfem

            from mfgarchon.alg.numerical.fem.mesh_adapter import skfem_to_meshdata
            from mfgarchon.geometry.meshes.mesh_2d import Mesh2D

            geometry = Mesh2D(domain_type="rectangle", bounds=(0.0, 1.0, 0.0, 1.0))
            xs = np.linspace(0.0, 1.0, 9)
            geometry.mesh_data = skfem_to_meshdata(skfem.MeshTri.init_tensor(xs, xs))
        geometry.boundary_conditions = bc
        return geometry

    return build


def _grid(bc: BoundaryConditions):
    return TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=bc)


def _exit(dim: int, value: float) -> BoundaryConditions:
    """DIRICHLET(value) on x_max, NO_FLUX on every other face."""
    walls = ["x_min"] if dim == 1 else ["x_min", "y_min", "y_max"]
    return BoundaryConditions(
        dimension=dim,
        segments=[
            BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=value, boundary="x_max"),
            *(BCSegment(name=face, bc_type=BCType.NO_FLUX, value=0.0, boundary=face) for face in walls),
        ],
    )


def _fem(problem: MFGProblem, bc: BoundaryConditions | None = None):
    from mfgarchon.alg.numerical.fem.fp_fem_solver import FPFEMSolver

    return FPFEMSolver(problem, order=1, boundary_conditions=bc)


def _meshless(problem: MFGProblem, bc: BoundaryConditions | None = None):
    from mfgarchon.alg.numerical.meshless_galerkin.fp_solver import MeshlessGalerkinFPSolver

    return MeshlessGalerkinFPSolver(
        problem, np.linspace(0.0, 1.0, 21).reshape(-1, 1), delta=0.35, boundary_conditions=bc
    )


def _drift_into_x_max(solver, m0: float) -> np.ndarray:
    x = solver._disc.dof_coordinates[:, 0]
    return np.asarray(
        solver.solve_fp_system(
            M_initial=np.full(x.size, m0), potential_field=np.broadcast_to(-x, (NT + 1, x.size)).copy()
        )
    )


@pytest.mark.parametrize("dim", [1, 2])
def test_fem_holds_an_explicit_density_and_reads_a_shared_dirichlet_as_an_exit(dim):
    """One problem, its shared BC DIRICHLET(0.7) on x_max: the HJB's exit cost.

    Read from the problem, the FP absorbs: from m = 0 with no drift nothing ever enters, so the density stays
    exactly 0. Handed the same BC as its own, the FP holds m = 0.7 at the wall, and with no drift and a no-flux
    wall opposite the exact steady state is m = 0.7 everywhere, which a long horizon (T = 100, dt = 10)
    reaches: measured max|m(T) - 0.7| = 1.6e-5 in 1-D and 2-D. A route that wrote 0.7 after a solve lifted at
    0 (the write-only rival of `test_fp_fem_bc_oracles_2512.py`, review 1) misses that state by 0.699 in 1-D
    and 0.698 in 2-D, its density falling to 6e-4 and 1.9e-3.
    """
    problem = _problem(_mesh(dim), _exit(dim, EXIT_VALUE), horizon=100.0)
    shared, own = _fem(problem), _fem(problem, _exit(dim, EXIT_VALUE))
    x = own._disc.dof_coordinates[:, 0]
    at_exit = np.isclose(x, 1.0)

    M_shared = np.asarray(shared.solve_fp_system(M_initial=np.zeros(x.size), potential_field=None))
    M_own = np.asarray(own.solve_fp_system(M_initial=np.zeros(x.size), potential_field=None))

    np.testing.assert_array_equal(M_shared, 0.0)
    np.testing.assert_array_equal(M_own[1:, at_exit], EXIT_VALUE)
    assert np.abs(M_own[-1] - EXIT_VALUE).max() < 1e-3


@pytest.mark.parametrize(
    ("build", "domain", "dim"),
    [(_fem, _mesh(1), 1), (_fem, _mesh(2), 2), (_meshless, _grid, 1)],
    ids=["fem-1d", "fem-2d", "meshless-1d"],
)
def test_an_explicit_no_flux_solves_as_the_shared_one(build, domain, dim):
    """The route changes where the BC comes from, not what a no-flux wall is, under a drift into x_max."""
    problem = _problem(domain, no_flux_bc(dimension=dim))
    np.testing.assert_array_equal(
        _drift_into_x_max(build(problem), 1.0), _drift_into_x_max(build(problem, no_flux_bc(dimension=dim)), 1.0)
    )


def test_meshless_reads_an_explicit_absorbing_wall_as_the_shared_exit():
    """Meshless gains no wall: its own DIRICHLET(0) is the exit it already had from a shared DIRICHLET(0.7)."""
    problem = _problem(_grid, _exit(1, EXIT_VALUE))
    np.testing.assert_array_equal(
        _drift_into_x_max(_meshless(problem), 1.0), _drift_into_x_max(_meshless(problem, _exit(1, 0.0)), 1.0)
    )


@pytest.mark.parametrize("value", [EXIT_VALUE, lambda x: EXIT_VALUE], ids=["number", "callable"])
def test_meshless_refuses_an_explicit_density(value):
    """A DIRICHLET with a value not provably zero, handed to meshless, would be m = g: refused (#2581, A-4)."""
    problem = _problem(_grid, no_flux_bc(dimension=1))
    with pytest.raises(NotImplementedError, match="would be a prescribed density m = g"):
        _meshless(problem, _exit(1, value))


# ------------------------------------------------------------------------- the FEM adapter's Dirichlet value
# The explicit route hands FP-FEM a Dirichlet value the shared view used to zero first, and the FEM adapter read
# any value but a plain int or float as 0: np.float32(0.7) held m = 0 (#2593, review 1). HJB-FEM reads the same
# adapter from the shared BC, so it held u = 0 there on main. A value is now a callable g(x), a finite real number
# or a finite 0-d real array, and anything else is refused at construction.
_READABLE = [0.7, np.float32(0.7), np.int64(1), np.array(0.7), Fraction(7, 10), lambda x: 0.7]
_READABLE_IDS = ["float", "float32", "int64", "0-d", "Fraction", "callable"]
_UNREADABLE = [
    ConstantProvider(0.7),
    "0.7",
    np.array([0.7]),
    Decimal("0.7"),
    float("nan"),
    np.inf,
    np.array(0.7 + 0j),
    True,
    np.True_,
    np.array(True),
    False,
    np.False_,
]
_UNREADABLE_IDS = [
    "provider",
    "str",
    "1-d",
    "Decimal",
    "nan",
    "inf",
    "complex",
    "bool",
    "np.bool_",
    "0-d bool",
    "False",
    "np.False_",
]


def _as_number(value) -> float:
    return float(value(np.zeros(1))) if callable(value) else float(value)


@pytest.mark.parametrize("value", _READABLE, ids=_READABLE_IDS)
def test_fem_holds_an_explicit_density_of_every_readable_value(value):
    """The steady state of the density test, m = g everywhere, for each value class."""
    own = _fem(_problem(_mesh(1), _exit(1, EXIT_VALUE), horizon=100.0), _exit(1, value))
    x = own._disc.dof_coordinates[:, 0]
    M = np.asarray(own.solve_fp_system(M_initial=np.zeros(x.size), potential_field=None))
    assert np.abs(M[-1] - _as_number(value)).max() < 1e-3


@pytest.mark.parametrize("value", _READABLE, ids=_READABLE_IDS)
def test_hjb_fem_holds_a_shared_dirichlet_of_every_readable_value(value):
    """The HJB's u = g on the exit at every level the solve writes, for each value class."""
    from mfgarchon.alg.numerical.fem.hjb_fem_solver import HJBFEMSolver

    solver = HJBFEMSolver(_problem(_mesh(1), _exit(1, value)))
    x = solver._disc.dof_coordinates[:, 0]
    U = np.asarray(solver.solve_hjb_system(M_density=np.ones((NT + 1, x.size)), U_terminal=np.zeros(x.size)))
    np.testing.assert_allclose(U[:-1, np.isclose(x, 1.0)], _as_number(value), rtol=1e-6)


@pytest.mark.parametrize("value", _UNREADABLE, ids=_UNREADABLE_IDS)
@pytest.mark.parametrize("route", ["fp-explicit", "hjb-shared"])
def test_fem_refuses_a_dirichlet_value_it_cannot_read(route, value):
    from mfgarchon.alg.numerical.fem.hjb_fem_solver import HJBFEMSolver

    if route == "fp-explicit":
        problem, own = _problem(_mesh(1), _exit(1, EXIT_VALUE)), _exit(1, value)
        build = lambda: _fem(problem, own)  # noqa: E731
    else:
        problem = _problem(_mesh(1), _exit(1, value))
        build = lambda: HJBFEMSolver(problem)  # noqa: E731
    with pytest.raises(NotImplementedError, match=r"Dirichlet segment 'exit' has a (value of type|non-finite|boolean)"):
        build()


def _build(route: str, bc: BoundaryConditions, domain=None):
    """FP-FEM given ``bc`` as its own BC, or HJB-FEM given ``bc`` as the shared BC, on ``domain`` (default the unit
    interval). The FP's shared BC is the exit on the interval, and no-flux on any other domain."""
    from mfgarchon.alg.numerical.fem.hjb_fem_solver import HJBFEMSolver

    if route == "fp-explicit":
        if domain is None:
            return _fem(_problem(_mesh(1), _exit(1, EXIT_VALUE)), bc)
        return _fem(_problem(domain, no_flux_bc(dimension=bc.dimension)), bc)
    return HJBFEMSolver(_problem(_mesh(1) if domain is None else domain, bc))


def _wall_and_exit(route: str, bc: BoundaryConditions) -> tuple[float, float]:
    """The density (FP, from m = 0) at T or the value (HJB, terminal 0) at t = 0, at the x_min wall and the x_max
    exit."""
    solver = _build(route, bc)
    x = solver._disc.dof_coordinates[:, 0]
    if route == "fp-explicit":
        field = np.asarray(solver.solve_fp_system(M_initial=np.zeros(x.size), potential_field=None))[-1]
    else:
        field = np.asarray(solver.solve_hjb_system(M_density=np.ones((NT + 1, x.size)), U_terminal=np.zeros(x.size)))[0]
    return float(field[np.isclose(x, 0.0)][0]), float(field[np.isclose(x, 1.0)][0])


_WALL = BCSegment(name="wall", bc_type=BCType.NO_FLUX, value=0.0, boundary="x_min")
_EXIT_SEGMENT = BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=EXIT_VALUE, boundary="x_max")


def _with_dirichlet_default(*segments: BCSegment) -> BoundaryConditions:
    return BoundaryConditions(
        dimension=1, segments=list(segments), default_bc=BCType.DIRICHLET, default_value=EXIT_VALUE
    )


@pytest.mark.parametrize("route", ["fp-explicit", "hjb-shared"])
def test_fem_refuses_a_dirichlet_default_no_segment_names(route):
    """The FEM adapter imposes conditions only from segments, so a DIRICHLET default_bc behind a segment set that
    leaves a face unnamed got the natural condition there. That BC is refused, and its advice -- give each
    Dirichlet face its own segment -- is executed: the wall stays free and the exit holds the value (measured
    0.0013 at the wall).

    It does not advise boundary=None any more. The FEM adapter ignores segment precedence, so a whole-boundary
    DIRICHLET beside the NO_FLUX wall condensed the wall too, to 0.7 (#2593, review 2)."""
    from mfgarchon.geometry.boundary import dirichlet_bc

    with pytest.raises(NotImplementedError, match=r"default_bc=DIRICHLET governs 1 boundary facet") as excinfo:
        _build(route, _with_dirichlet_default(_WALL))
    assert "boundary=None" not in str(excinfo.value)
    assert f"value={EXIT_VALUE!r}" in str(excinfo.value)
    wall, exit_value = _wall_and_exit(route, _with_dirichlet_default(_WALL, _EXIT_SEGMENT))
    assert wall < 0.1
    assert exit_value == pytest.approx(EXIT_VALUE, abs=1e-12)
    _wall_and_exit(route, dirichlet_bc(value=EXIT_VALUE, dimension=1))


@pytest.mark.parametrize("route", ["fp-explicit", "hjb-shared"])
def test_a_segment_named_by_an_alias_is_refused_and_the_advice_runs(route):
    """Under a DIRICHLET default every segment's facets are resolved by the mesh's tags, so a wall named by a library
    alias ("left") is refused, though both FEM solvers solved that BC correctly on main: HJB-FEM held u = 0.0013 at
    the wall and 0.7 at the exit (#2593, review 2; FEM's naming rule against the BC layer's is #2512's row B1). The
    advice names the mesh's tags; renaming the wall to the tag it means solves with it intact."""
    alias = BCSegment(name="wall", bc_type=BCType.NO_FLUX, value=0.0, boundary="left")
    with pytest.raises(
        ValueError,
        match=r"no such tagged boundary \(available: \[.*'x_min'.*\]\).*Name one of the mesh's tagged boundaries",
    ) as excinfo:
        _build(route, _with_dirichlet_default(alias, _EXIT_SEGMENT))
    assert "The BC layer reads 'left' as 'x_min'" in str(excinfo.value)
    wall, exit_value = _wall_and_exit(route, _with_dirichlet_default(_WALL, _EXIT_SEGMENT))
    assert wall < 0.1
    assert exit_value == pytest.approx(EXIT_VALUE, abs=1e-12)


def _disc(tag_exit: bool):
    """A builder of the unit disc (``MeshTri.init_circle(3)``), whose axis tags, placed by bounding box, hold no
    facets. With ``tag_exit`` the boundary faces with x > 0.9 at both ends carry MeshData tag 1, as ``region_1``."""

    def build(bc: BoundaryConditions):
        import skfem

        from mfgarchon.alg.numerical.fem.mesh_adapter import skfem_to_meshdata
        from mfgarchon.geometry.meshes.mesh_2d import Mesh2D

        mesh_data = skfem_to_meshdata(skfem.MeshTri.init_circle(3))
        if tag_exit:
            x = mesh_data.vertices[mesh_data.boundary_faces][..., 0]
            mesh_data.boundary_tags = np.where(x.min(axis=1) > 0.9, 1, 0)
        geometry = Mesh2D(domain_type="circle", bounds=(-1.0, 1.0, -1.0, 1.0))
        geometry.mesh_data = mesh_data
        geometry.boundary_conditions = bc
        return geometry

    return build


def _disc_exit(name: str) -> BoundaryConditions:
    segment = BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=EXIT_VALUE, boundary=name)
    return BoundaryConditions(dimension=2, segments=[segment], default_bc=BCType.NO_FLUX)


def _solve_on_disc(route: str, name: str, tag_exit: bool):
    """Build and solve on the disc with the exit named ``name``: a Dirichlet segment's name is resolved with its
    DOFs, at solve time. Returns the solver and the FP density at T or the HJB value at t = 0."""
    solver = _build(route, _disc_exit(name), domain=_disc(tag_exit))
    n = solver._disc.dof_coordinates.shape[0]
    if route == "fp-explicit":
        return solver, np.asarray(solver.solve_fp_system(M_initial=np.zeros(n), potential_field=None))[-1]
    return solver, np.asarray(solver.solve_hjb_system(M_density=np.ones((NT + 1, n)), U_terminal=np.zeros(n)))[0]


@pytest.mark.parametrize("route", ["fp-explicit", "hjb-shared"])
def test_a_tag_holding_no_facet_is_refused_and_the_advice_runs(route):
    """On the disc the axis tags exist and hold no facets, so an exit named "x_max" condensed nothing and the
    face stayed a wall, on main too; the alias refusal's advice pointed there (#2593, review 3). An empty tag is
    refused as a missing one. With no tag holding a facet, the advice is to tag the mesh, and it is executed: the
    exit faces tagged 1 and named "region_1" hold the value. Once tagged, "x_max" is still refused, listing it.

    The no-tag message offers no whole-boundary segment: beside a named wall that would condense the wall too
    (#2595 rows 1 and 10). And the tagged exit is not spread over the boundary: the 5 boundary DOFs at x < -0.9
    hold at most 1.21e-9 (FP) and 1.09e-9 (HJB), measured, so 1e-6 separates them from the exit's 0.7."""
    with pytest.raises(
        ValueError, match=r"none of its tags holds a facet.*Tag the mesh before naming a face"
    ) as excinfo:
        _solve_on_disc(route, "x_max", tag_exit=False)
    assert "available" not in str(excinfo.value)
    assert "boundary=None" not in str(excinfo.value)
    assert "whole-boundary" not in str(excinfo.value)
    with pytest.raises(ValueError, match=r"no such tagged boundary \(available: \['region_1'\]\)") as excinfo:
        _solve_on_disc(route, "x_max", tag_exit=True)
    assert "The mesh carries the tag 'x_max', but no facet is in it." in str(excinfo.value)

    solver, field = _solve_on_disc(route, "region_1", tag_exit=True)
    exit_dofs = solver._basis.get_dofs(solver._skfem_mesh.boundaries["region_1"]).flatten()
    assert exit_dofs.size == 5
    np.testing.assert_allclose(field[exit_dofs], EXIT_VALUE, atol=1e-12)
    boundary_dofs = solver._basis.get_dofs(solver._skfem_mesh.boundary_facets()).flatten()
    far_wall = boundary_dofs[solver._basis.doflocs[0, boundary_dofs] < -0.9]
    assert far_wall.size == 5
    assert np.max(np.abs(field[far_wall])) < 1e-6


@pytest.mark.parametrize("value", [None, np.array([0.0]), np.zeros(3)], ids=["None", "1-d", "3-vector"])
@pytest.mark.parametrize("route", ["fp-explicit", "hjb-shared"])
def test_fem_reads_a_verifiably_zero_value_as_zero(route, value):
    """``None`` and an all-zero array are verifiably zero data to the library's one owner (`_describe_bc_value`), as
    on main and as the FP's view of a shared BC says; the adapter asks that owner first, so it no longer refuses the
    array."""
    from mfgarchon.alg.numerical.fem.hjb_fem_solver import HJBFEMSolver

    if route == "fp-explicit":
        solver = _fem(_problem(_mesh(1), no_flux_bc(dimension=1)), _exit(1, value))
        x = solver._disc.dof_coordinates[:, 0]
        field = np.asarray(solver.solve_fp_system(M_initial=np.ones(x.size), potential_field=None))[1:]
    else:
        solver = HJBFEMSolver(_problem(_mesh(1), _exit(1, value)))
        x = solver._disc.dof_coordinates[:, 0]
        field = np.asarray(solver.solve_hjb_system(M_density=np.ones((NT + 1, x.size)), U_terminal=np.ones(x.size)))[
            :-1
        ]
    np.testing.assert_array_equal(field[:, np.isclose(x, 1.0)], 0.0)
