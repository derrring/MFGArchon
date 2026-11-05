"""A user callable reads what it names, ``x`` is a coordinate tuple, and its batch and point reads agree (#2875, #2872, #2481).

The oracle is the contract's own words (`CONVENTIONS.md` § *Callable signatures*): a parameter named for a slot is given
that slot, by name; ``x`` holds one entry per coordinate, each a scalar or an array; and a body the library may read
either way must read both the same, or it is refused where it is first read on a batch.
"""

from __future__ import annotations

import pickle
from types import SimpleNamespace

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
from mfgarchon.alg.numerical.fp_solvers.fp_particle import FPParticleSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver, HJBGFDMSolver, HJBSemiLagrangianSolver
from mfgarchon.alg.numerical.pde_solvers.implicit_heat import ImplicitHeatSolver
from mfgarchon.core.hamiltonian import (
    DualHamiltonian,
    HamiltonianBase,
    QuadraticControlCost,
    SeparableHamiltonian,
    refuse_a_hamiltonian_whose_reads_disagree,
)
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.geometry.boundary.nitsche_1d import Nitsche1DPoissonSolver
from mfgarchon.geometry.level_set.core import LevelSetEvolver
from mfgarchon.types import callable_protocols as cp
from mfgarchon.utils.pde_coefficients import CoefficientField, resolve_diffusion_source

A, B = np.array([0.1, 0.35, 0.8, 0.95, 0.2]), np.array([0.6, 0.15, 0.4, 0.05, 0.9])


def _grid(dimension=1, n=11):
    return TensorProductGrid(
        bounds=[(0.0, 1.0)] * dimension, Nx_points=[n] * dimension, boundary_conditions=no_flux_bc(dimension=dimension)
    )


def _hamiltonian(**kwargs):
    return SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0), coupling=lambda m: m, **kwargs)


def _problem(*, dimension=1, n=11, hamiltonian=None, u_terminal=lambda x: 0.0, m_initial=lambda x: 1.0, **kwargs):
    return MFGProblem(
        model=Model(hamiltonian=hamiltonian or _hamiltonian(), volatility=0.3),
        domain=_grid(dimension, n),
        conditions=Conditions(u_terminal=u_terminal, m_initial=m_initial, T=0.2),
        Nt=4,
        **kwargs,
    )


def _read(body, x=(A, B), slots=cp.REGION_SLOTS, **slot_values):
    return cp.bind_user_callable(body, slots, role="r")(x=x, **slot_values)


# ---------------------------------------------------------------------------------------------- what a body is given


@pytest.mark.parametrize(
    ("condition", "x_first"),
    [
        (lambda c=0.7, x=0.0: c + x[0], lambda x: 0.7 + x[0]),
        (lambda t, x: t + x[0], lambda x: 0.2 + x[0]),
        (lambda x, gain=2.0: gain * x[0], lambda x: 2.0 * x[0]),
    ],
    ids=["a default ahead of x", "time first", "a default behind x"],
)
def test_a_parameter_named_for_a_slot_is_given_that_slot(condition, x_first):
    """#2872: ``lambda c=0.7, x=0.0`` got the point in ``c``, and ``problem.solve()`` returned with no error."""
    np.testing.assert_array_equal(
        _problem(u_terminal=condition).get_u_terminal(), _problem(u_terminal=x_first).get_u_terminal()
    )


@pytest.mark.parametrize(
    "body",
    [lambda x, y: x + y, lambda z: z[0], lambda *x: x[0], lambda **k: k["x"], lambda x, /: x[0]],
    ids=["expanded coordinates", "a name no slot has", "*args", "**kwargs", "positional-only"],
)
def test_what_cannot_be_read_by_name_is_refused_when_it_is_bound(body):
    with pytest.raises(TypeError, match="#2875"):
        cp.bind_user_callable(body, cp.REGION_SLOTS, role="r")


@pytest.mark.parametrize("handed", [A, [A, B], None, 0.5], ids=["an array", "a list", "None", "a number"])
def test_a_door_that_hands_x_in_any_other_form_is_named_as_the_defect(handed):
    with pytest.raises(TypeError, match=r"the library handed .* x as .*a defect in the door"):
        cp.bind_user_callable(lambda x: 0.0, cp.REGION_SLOTS, role="r")(x=handed)


# ------------------------------------------------------------------------------------------------ what a body may do


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (lambda x: x[0], A),
        (lambda x: x[0] + 2.0 * x[1], A + 2.0 * B),
        (lambda x: np.hypot(*x), np.hypot(A, B)),
        (lambda x: sum(xi**2 for xi in x), A**2 + B**2),
        (lambda x: np.linalg.norm(x, axis=0), np.hypot(A, B)),
        (lambda x: 3.0, 3.0),
    ],
    ids=["x[0]", "x[0] and x[1]", "np.hypot", "a squared norm", "np.linalg.norm, axis=0", "a constant"],
)
def test_a_body_that_reads_both_ways_the_same_is_accepted_and_returns_what_it_read(body, expected):
    np.testing.assert_allclose(_read(body), expected, rtol=0, atol=1e-14)


@pytest.mark.parametrize(
    "body",
    [
        lambda x: x**2,
        lambda x: x - 0.5,
        lambda x: x[:, 0],
        lambda x: x[..., 0],
        lambda x: np.asarray(x)[:, 0],
        lambda x: np.asarray(x)[..., 0],
        lambda x: x[0][0],
        lambda x: float(x[0]),
        lambda x: np.linalg.norm(x),
        lambda x: np.sum(x),
        lambda x: np.max(x[0]),
    ],
    ids=[
        "x**2",
        "x - 0.5",
        "x[:, 0]",
        "x[..., 0]",
        "an array's column",
        "an array's last axis",
        "node 0",
        "float",
        "np.linalg.norm, no axis",
        "np.sum, no axis",
        "np.max of a coordinate",
    ],
)
def test_a_body_that_reads_a_batch_and_a_point_differently_is_refused(body):
    with pytest.raises(ValueError, match="on a batch of 5 points"):
        _read(body)


def test_a_misread_is_seen_on_a_uniform_grid_where_a_periodic_term_vanishes_at_the_ends_and_the_middle():
    """``sin(2 pi x)`` is 0 at the first, middle and last nodes, so samples there read a body frozen at node 0 the same both
    ways. The control states that: at those three nodes its two reads agree."""
    nodes = np.linspace(0.0, 1.0, 21)
    misread = lambda t, x, p, m: 0.5 * np.sum(p**2, axis=-1) + np.sin(2 * np.pi * np.asarray(x)[..., 0])  # noqa: E731

    class _Misread(HamiltonianBase):
        def __call__(self, t, x, p, m):
            return misread(t, x, np.asarray(p, dtype=float), m)

    at_old_samples = np.sin(2 * np.pi * nodes[[0, 10, 20]])
    assert np.max(np.abs(at_old_samples - at_old_samples[0])) < 1e-14, "control: these three read the same"
    with pytest.raises(ValueError, match="evaluate_H, at 3 samples, reads differently"):
        refuse_a_hamiltonian_whose_reads_disagree(_Misread(), nodes)


@pytest.mark.parametrize(("count", "samples"), [(2, 3), (3, 3), (4, 4), (5, 3), (21, 3), (81, 3), (1000, 4)])
def test_the_samples_are_inside_the_batch_and_miss_the_symmetric_nodes(count, samples):
    picked = cp.spread_samples(count, samples)
    assert picked.min() >= 0
    assert picked.max() < count
    assert len(set(picked.tolist())) == min(count, samples)
    if count % 2 == 1 and count > samples:
        assert not {0, count // 2, count - 1} <= set(picked.tolist())


def test_a_body_that_writes_into_its_coordinates_changes_neither_read():
    def writes(x):
        coordinate = x[0]
        coordinate += 0.3
        return coordinate

    handed = A.copy()
    np.testing.assert_allclose(_read(writes, x=(handed,)), A + 0.3, rtol=0, atol=1e-14)


# ------------------------------------------------------------------------------------------- the forms a return takes


@pytest.mark.parametrize(
    ("body", "x"),
    [
        (lambda x: 2.0, (A,)),
        (lambda x: 2.0 * x[0], (A,)),
        (lambda x: np.stack([x[0], x[1]], axis=-1), (A, B)),
        (lambda x: np.stack([x[0], x[1]], axis=0), (A, B)),
        (lambda x: np.array([[0.3, 0.1], [0.1, 0.2]]), (A, B)),
    ],
    ids=[
        "a scalar",
        "one entry per point",
        "per coordinate, the axis last",
        "per coordinate, the axis first",
        "one array",
    ],
)
def test_each_form_a_return_takes_is_compared_and_accepted(body, x):
    _read(body, x=x)


def test_a_grid_shaped_batch_is_compared_at_multi_indices():
    x, y = np.meshgrid(np.linspace(0.0, 1.0, 5), np.linspace(0.0, 2.0, 4), indexing="ij")
    np.testing.assert_allclose(_read(lambda x: x[0] + x[1], x=(x, y)), x + y, rtol=0, atol=1e-14)
    with pytest.raises(ValueError, match="on a batch of 20 points"):
        _read(lambda x: x[0][0, 0] + x[1], x=(x, y))


@pytest.mark.parametrize(
    "body",
    [
        lambda x: np.zeros((1, 2, 2)) if np.ndim(x[0]) == 0 else np.zeros((5, 2, 2)),
        lambda x: np.zeros(3) if np.ndim(x[0]) == 0 else np.zeros(7),
    ],
    ids=["a point's own extra axis", "shapes that share nothing"],
)
def test_a_return_with_no_entry_to_compare_is_refused_by_name(body):
    with pytest.raises(ValueError, match=r"has no entry|no entry per point"):
        _read(body)


def test_a_fixed_array_with_the_wrong_values_at_a_point_is_refused():
    with pytest.raises(ValueError, match="reads differently"):
        _read(lambda x: np.array([1.0, 2.0, 3.0]) * (1.0 if np.ndim(x[0]) else x[0]))


# ---------------------------------------------------------------------------------------- what the slots beside x hold


def test_a_field_door_hands_a_point_read_the_field_whole():
    """A body may read a field whole, so `np.max(m)` is accepted. The limit: at a field door `m[0]` reads node 0 at every
    point, and that cannot be told from `np.max(m)`."""
    m = np.array([0.5, 2.0, 1.0, 0.25, 0.75])
    values = _read(lambda x, m: x[0] * np.max(m), x=(A,), slots=cp.COEFFICIENT_SLOTS, m=m, t=0.0)
    np.testing.assert_allclose(values, A * 2.0, rtol=0, atol=1e-14)
    _read(lambda x, m: x[0] * m[0], x=(A,), slots=cp.COEFFICIENT_SLOTS, m=m, t=0.0)


def test_a_field_door_refuses_a_body_written_to_x_as_a_field():
    m = np.array([0.5, 2.0, 1.0, 0.25, 0.75])
    with pytest.raises(ValueError, match="on a batch of 5 points"):
        _read(lambda x, m: np.gradient(m, x[0]), x=(np.linspace(0.0, 1.0, 5),), slots=cp.COEFFICIENT_SLOTS, m=m, t=0.0)


def test_a_per_point_slot_is_cut_with_x():
    p = np.column_stack([A, B])
    m = np.array([0.5, 2.0, 1.0, 0.25, 0.75])
    body = lambda x, p, m: x[0] + np.sum(p**2, axis=-1) * m  # noqa: E731
    expected = A + np.sum(p**2, axis=-1) * m
    np.testing.assert_allclose(
        _read(body, x=(A,), slots=cp.HAMILTONIAN_SLOTS, p=p, m=m, t=0.0), expected, rtol=0, atol=1e-14
    )
    with pytest.raises(ValueError, match="on a batch of 5 points"):
        _read(lambda x, p, m: x[0] + p[0, 0], x=(A,), slots=cp.HAMILTONIAN_SLOTS, p=p, m=m, t=0.0)


# ------------------------------------------------------------------------------------------------- batches too small


@pytest.mark.parametrize(
    ("x", "count"),
    [
        ((np.array([0.3]),), "one point in 1-D"),
        ((np.array([0.3]), np.array([0.6])), "one point in 2-D"),
        ((np.array([0.3, 0.7]), np.array([0.6, 0.1])), "two points in 2-D, as many as d"),
        ((np.array([0.3, 0.7, 0.2]), np.array([0.6, 0.1, 0.5]), np.array([0.4, 0.9, 0.8])), "three points in 3-D"),
    ],
    ids=["N=1, d=1", "N=1, d=2", "N=d=2", "N=d=3"],
)
def test_a_batch_too_small_to_compare_is_cycled_so_none_is_returned_unchecked(x, count):
    with pytest.raises(ValueError, match="on a batch of"):
        _read(lambda x: np.linalg.norm(x), x=x)
    np.testing.assert_allclose(_read(lambda x: sum(x), x=x), sum(x), rtol=0, atol=1e-14)


def test_a_door_that_hands_one_point_has_nothing_to_compare_and_the_first_batch_is_still_checked():
    bound = cp.bind_user_callable(lambda x: np.sum(x), cp.REGION_SLOTS, role="r")
    assert bound(x=(0.3, 0.6)) == pytest.approx(0.9)
    assert not bound.checked
    with pytest.raises(ValueError, match="on a batch of 5 points"):
        bound(x=(A, B))


def test_the_check_runs_once():
    calls = []

    def counted(x):
        calls.append(np.ndim(x[0]))
        return x[0]

    bound = cp.bind_user_callable(counted, cp.REGION_SLOTS, role="r")
    bound(x=(A,))
    first = len(calls)
    assert first > 1, "the first batch is read together and at its samples"
    bound(x=(A,))
    assert len(calls) == first + 1


# --------------------------------------------------------------------------------- a first batch can be a narrow probe


def test_a_first_batch_inside_a_dead_zone_cannot_see_the_misread_and_that_is_the_limit():
    """The check reads what the door's first batch holds. A particle drift's first batch is the initial cloud: inside the
    dead zone a body reading the cloud's largest coordinate reads the same both ways, and it is accepted. Outside it the
    same body is refused, so the limit is the first batch and not the body. The Hamiltonian does not have this limit, since
    it is read on wide synthetic p and m."""
    dead_zone = lambda t, x, m: np.maximum(np.abs(np.max(x[0])) - 0.5, 0.0) * np.ones_like(x[0])  # noqa: E731
    problem = _problem()
    narrow = FPParticleSolver(problem, num_particles=40, seed=0)
    narrow.solve_fp_system(
        initial_particles=np.random.default_rng(0).uniform(0.1, 0.4, (40, 1)),
        drift_field=dead_zone,
        drift_needs_density=False,
        show_progress=False,
    )
    wide = FPParticleSolver(problem, num_particles=40, seed=0)
    with pytest.raises(ValueError, match="reads differently"):
        wide.solve_fp_system(
            initial_particles=np.random.default_rng(0).uniform(0.1, 0.95, (40, 1)),
            drift_field=dead_zone,
            drift_needs_density=False,
            show_progress=False,
        )


# -------------------------------------------------------------------------------- bodies written to per-axis vectors


@pytest.mark.parametrize(
    "body",
    [
        lambda t, x, m: np.zeros((2, *np.shape(x[0][:, None]))),
        lambda t, x, m: np.stack(np.meshgrid(*x, indexing="ij")) * 0.0,
    ],
    ids=["x[0][:, None]", "np.meshgrid(*x)"],
)
def test_a_body_written_to_per_axis_vectors_is_refused_at_fp_fdm(body):
    problem = _problem(dimension=2, n=6)
    with pytest.raises((ValueError, IndexError), match=r"on a batch of|no entry|too many indices"):
        FPFDMSolver(problem).solve_fp_system(np.ones((6, 6)) / 36.0, drift_field=body, show_progress=False)


# --------------------------------------------------------------------------------------------------- library closures


def test_a_library_closure_is_handed_x_as_it_is_and_read_once():
    """A closure the library wrote is not a user input: the user's callable it wraps has its own door."""
    calls = []

    def closure(t, x):
        calls.append(np.shape(x))
        return np.zeros(np.shape(x)[0])

    closure.__module__ = "mfgarchon.alg.numerical.coupling.source_composition"
    points = np.linspace(0.0, 1.0, 7).reshape(-1, 1)
    cp.evaluate_solver_source(closure, t=0.0, x=points, dimension=1)
    assert calls == [(7, 1)]


def test_a_users_source_is_read_as_a_tuple_and_refused_at_its_own_door():
    points = np.linspace(0.0, 1.0, 7).reshape(-1, 1)
    np.testing.assert_allclose(
        cp.evaluate_solver_source(lambda t, x: x[0] + t, t=0.5, x=points, dimension=1),
        points.ravel() + 0.5,
        rtol=0,
        atol=1e-14,
    )
    with pytest.raises(ValueError, match=r"source_term .*, on a batch of 7 points"):
        cp.evaluate_solver_source(lambda t, x: np.max(x[0]), t=0.5, x=points, dimension=1)


# ------------------------------------------------------------------------------------- doors no other test file reaches


def _heat(forcing):
    grid = _grid(1, 11)
    return ImplicitHeatSolver(grid, alpha=0.1, bc=no_flux_bc(dimension=1), forcing=forcing).solve_step(
        np.ones(11), 0.01
    )


def _level_set(velocity):
    grid = _grid(2, 21)
    x, y = grid.meshgrid()
    phi = np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2) - 0.25
    return LevelSetEvolver(grid).evolve_step(phi, velocity=velocity, dt=0.001)


def _nitsche_load(f):
    return Nitsche1DPoissonSolver(8).assemble_load_vector(f)


def _nitsche_error(u_exact):
    solver = Nitsche1DPoissonSolver(8)
    return solver.compute_l2_error(np.zeros(9), u_exact)


@pytest.mark.parametrize(
    ("door", "reads", "misreads"),
    [
        (_heat, lambda t, x: np.sin(x[0]), lambda t, x: x**2),
        (_level_set, lambda x: 0.5 + x[0], lambda x: x[:, 0]),
        (_nitsche_load, lambda x: 2.0 + x[0], lambda x: x**2),
        (_nitsche_error, lambda x: x[0] * (1.0 - x[0]), lambda x: x * (1.0 - x)),
    ],
    ids=["forcing", "velocity", "Nitsche's f", "Nitsche's u_exact"],
)
def test_a_door_reads_a_coordinate_tuple_and_refuses_what_cannot(door, reads, misreads):
    assert np.all(np.isfinite(door(reads)))
    with pytest.raises((TypeError, ValueError)):
        door(misreads)


def test_an_implicit_domain_reads_a_region_as_a_grid_does():
    """`mark_region` is one method: a body written for a grid means the same on an implicit domain."""
    from mfgarchon.geometry.implicit import Hypersphere

    points = np.array([[0.5, 0.7], [0.7, 0.5], [0.7, 0.7]])
    disk = Hypersphere(center=[0.5, 0.5], radius=0.3)
    disk.mark_region("top", predicate=lambda x: x[1] > 0.6)
    disk.mark_region("right", predicate=lambda x: x[0] > 0.6)
    disk.mark_region("signed", sdf_region=lambda x: 0.6 - x[1])
    assert disk.get_region_mask("top", points).tolist() == [True, False, True]
    assert disk.get_region_mask("signed", points).tolist() == [True, False, True]
    assert disk.intersect_regions("top", "right")(points).tolist() == [False, False, True]
    assert disk.get_region_predicate("top")(points).tolist() == [True, False, True]

    disk.mark_region("misread", predicate=lambda x: x[:, 1] > 0.6)
    with pytest.raises(ValueError, match=r"predicate .*, on a batch of 3 points"):
        disk.get_region_mask("misread", points)
    disk.mark_region("a geometry's own", sdf_region=Hypersphere(center=[0.5, 0.7], radius=0.1).signed_distance)
    assert disk.get_region_mask("a geometry's own", points).tolist() == [True, False, False]


def test_the_meshless_domain_masks_by_the_tuple_and_keeps_the_librarys_own_predicates():
    from mfgarchon.alg.numerical.meshless_galerkin.discretization import discretization_from_cloud
    from mfgarchon.geometry.predicates import sdf_region

    axis = np.linspace(0.0, 1.0, 6)
    cloud = np.stack([c.ravel() for c in np.meshgrid(axis, axis, indexing="ij")], axis=1)
    half = discretization_from_cloud(cloud, delta=0.45, degree=1, domain=lambda x: x[0] < 0.5)
    assert half._w.sum() == pytest.approx(0.5, abs=0.1)
    same = discretization_from_cloud(cloud, delta=0.45, degree=1, domain=sdf_region(lambda x: x[0] - 0.5))
    np.testing.assert_array_equal(same._w, half._w)
    with pytest.raises(ValueError, match=r"domain .*, on a batch of"):
        discretization_from_cloud(cloud, delta=0.45, degree=1, domain=lambda x: x[:, 0] < 0.5)


def test_the_diffusion_source_reads_one_point_as_a_tuple():
    """One point, so no batch to compare: a body that reads a single point is legal here."""
    points = np.array([[0.1, 0.2], [0.3, 0.4]])
    assert resolve_diffusion_source(lambda x: x[0] + 2.0 * x[1], index=1, points=points) == pytest.approx(1.1)
    assert resolve_diffusion_source(lambda x: x[0] + 2.0 * x[1], points=points) == pytest.approx(0.8)
    assert resolve_diffusion_source(lambda x: x[0], index=1, points=np.array([0.25, 0.75])) == pytest.approx(0.75)
    assert resolve_diffusion_source(lambda x: float(x[0]), index=0, points=points) == pytest.approx(0.1)
    with pytest.raises(TypeError, match=r"names no slot"):
        resolve_diffusion_source(lambda point: point[0], index=0, points=points)


def test_a_dict_bcs_callable_reaches_the_door_of_a_boundary_value():
    from mfgarchon.geometry.boundary.types import boundary_value

    def read(values):
        bc = HJBGFDMSolver._boundary_conditions_from_dict({"type": "dirichlet", "values": values}, dimension=1)
        return boundary_value(bc.segments[0].value, kind="dirichlet", t=0.0, x=np.array([0.5]), where="the row")

    assert float(read(lambda x: x[0] + 1.0)) == pytest.approx(1.5)
    with pytest.raises(TypeError, match=r"names no slot"):
        read(lambda point: point[0])


def test_a_refinement_function_reads_a_coordinate_tuple():
    from mfgarchon.geometry.meshes.mesh_1d import Mesh1D

    vertices = (
        Mesh1D((0.0, 1.0), 4, "adaptive", refinement_function=lambda x: 1.0 + x[0]).generate_mesh().vertices.ravel()
    )
    np.testing.assert_allclose(np.diff(vertices)[:2], [0.25, 0.25 * 1.25], rtol=0, atol=1e-14)
    assert len(Mesh1D((0.0, 1.0), 4, "adaptive").generate_mesh().vertices) > 2
    with pytest.raises(TypeError, match=r"names no slot"):
        Mesh1D((0.0, 1.0), 4, "adaptive", refinement_function=lambda s: 1.0 + s).generate_mesh()


def test_a_network_condition_is_read_by_node():
    from mfgarchon.extensions.topology import NetworkMFGComponents, NetworkMFGProblem
    from mfgarchon.geometry.graph.network_geometry import GridNetwork

    network = GridNetwork(width=2, height=2)
    network.create_network()
    components = NetworkMFGComponents(
        initial_node_density_func=lambda node: node + 1.0, terminal_node_value_func=lambda t, node: t * node
    )
    problem = NetworkMFGProblem(geometry=network, T=0.5, Nt=4, components=components)
    np.testing.assert_allclose(problem.get_initial_density(), [1.0, 2.0, 3.0, 4.0], rtol=0, atol=0)
    np.testing.assert_allclose(problem.get_terminal_value(), [0.0, 0.5, 1.0, 1.5], rtol=0, atol=0)
    problem.components.initial_node_density_func = lambda i: i
    with pytest.raises(TypeError, match=r"names no slot"):
        problem.get_initial_density()


def test_a_graph_coupling_is_read_by_its_names_in_any_order():
    from mfgarchon.alg.numerical.coupling.graph_coupling import AdjacencyCoupling

    adjacency = np.array([[0.0, 1.0], [1.0, 0.0]])
    values = [np.full((3, 5), 1.0), np.full((3, 5), 2.0)]
    densities = [np.full((3, 5), 3.0), np.full((3, 5), 4.0)]
    x = np.linspace(0.0, 1.0, 5)
    coupling = AdjacencyCoupling(adjacency, coupling_hjb=lambda t, m_j, v_j, j, i: 10.0 * i + j + v_j + m_j + t)
    source = coupling.compute_hjb_source(0, values, densities, dt=0.5)
    np.testing.assert_allclose(source(0.0, x), 1.0 + 2.0 + 4.0, rtol=0, atol=1e-14)
    with pytest.raises(TypeError, match=r"names no slot"):
        AdjacencyCoupling(adjacency, coupling_fp=lambda a, b: a + b)


def test_the_gradient_of_an_sdf_reads_it_as_a_tuple():
    from mfgarchon.geometry.implicit import Hypersphere
    from mfgarchon.geometry.implicit.sdf_utils import sdf_gradient

    points = np.array([[0.6, 0.0], [0.0, 0.5], [0.3, 0.4]])
    expected = points / np.linalg.norm(points, axis=1, keepdims=True)
    np.testing.assert_allclose(sdf_gradient(points, lambda x: np.hypot(x[0], x[1]) - 1.0), expected, rtol=0, atol=1e-6)
    disk = Hypersphere(center=[0.0, 0.0], radius=1.0)
    np.testing.assert_allclose(sdf_gradient(points, disk.signed_distance), expected, rtol=0, atol=1e-6)
    with pytest.raises(ValueError, match=r"sdf_func .*, on a batch of 3 points"):
        sdf_gradient(points, lambda x: np.linalg.norm(x, axis=1) - 1.0)


def _unbuilt(solver_class, **held):
    """These solvers refuse construction until #1342. What they read stays a door, so each row sets what its read uses."""
    solver = solver_class.__new__(solver_class)
    vars(solver).update(held)
    return solver


def test_a_blocked_solvers_constraints_and_conditions_are_read_by_name():
    from mfgarchon.alg.optimization.optimal_transport.sinkhorn_solver import SinkhornMFGSolver
    from mfgarchon.alg.optimization.optimal_transport.wasserstein_solver import WassersteinMFGSolver
    from mfgarchon.alg.optimization.variational_solvers.primal_dual_solver import PrimalDualMFGSolver

    grid = np.linspace(0.0, 1.0, 5)
    shared = {"x_grid": grid, "t_grid": np.array([0.0, 0.5]), "Nt": 2}
    components = SimpleNamespace(state_constraints=[lambda x, t: x[0] + t - 1.0], velocity_constraints=None)
    solver = _unbuilt(PrimalDualMFGSolver, problem=SimpleNamespace(components=components), **shared)
    density = np.ones((2, 5))
    assert solver._evaluate_all_constraints(density)["state_constraints"] == pytest.approx(0.5)
    components.state_constraints, components.velocity_constraints = [lambda position: 0.0], None
    with pytest.raises(TypeError, match=r"names no slot"):
        solver._evaluate_all_constraints(density)

    problem = SimpleNamespace(
        geometry=SimpleNamespace(get_bounds=lambda: ([0.0], [1.0])),
        initial_density=lambda x: 1.0 + x[0],
        running_cost=lambda m, x, t: m * x[0] + t,
    )
    held = {"problem": problem, "spatial_grid": grid, "time_grid": np.array([0.0, 1.0]), "dx": 0.25, "dt": 1.0}
    for solver_class in (SinkhornMFGSolver, WassersteinMFGSolver):
        densities = _unbuilt(solver_class, **held)._initialize_densities()
        np.testing.assert_allclose(densities[0], (1.0 + grid) / 1.5, rtol=0, atol=1e-14)  # normalised
    wasserstein = _unbuilt(WassersteinMFGSolver, ground_cost=np.zeros((5, 5)), **held)
    assert wasserstein._compute_total_cost(np.ones((1, 5)), []) == pytest.approx(0.25 * grid.sum())


def test_a_variational_terminal_cost_reads_a_coordinate_tuple():
    from mfgarchon.alg.optimization.variational_problem import VariationalMFGComponents, VariationalMFGProblem

    components = VariationalMFGComponents(
        lagrangian_func=lambda t, x, v, m: 0.5 * v**2, terminal_cost_func=lambda x: 2.0 * x[0], m_initial=lambda x: 1.0
    )
    problem = VariationalMFGProblem(xmin=0.0, xmax=1.0, Nx=10, T=0.2, Nt=4, components=components)
    np.testing.assert_allclose(problem.evaluate_terminal_cost(np.array([0.25, 0.5])), [0.5, 1.0], rtol=0, atol=1e-14)
    with pytest.raises((TypeError, ValueError)):
        VariationalMFGProblem(
            xmin=0.0,
            xmax=1.0,
            Nx=10,
            T=0.2,
            Nt=4,
            components=VariationalMFGComponents(
                lagrangian_func=lambda t, x, v, m: 0.5 * v**2,
                terminal_cost_func=lambda x: x**2,
                m_initial=lambda x: 1.0,
            ),
        ).evaluate_terminal_cost(np.array([0.25, 0.5]))


def test_the_drift_helpers_read_what_a_drift_names():
    from mfgarchon.utils.drift_helpers import composite_drift, prescribed_drift

    wind = prescribed_drift(lambda t, x: 2.0 * x[0])
    pull = lambda x, m: -x[0] * m  # noqa: E731
    total = composite_drift([wind, pull])
    m = np.array([1.0, 0.5, 2.0, 0.25, 0.75])
    np.testing.assert_allclose(total(0.0, (A,), m), 2.0 * A - A * m, rtol=0, atol=1e-14)
    with pytest.raises(TypeError, match=r"names no slot"):
        prescribed_drift(lambda t, position: position)
    with pytest.raises(TypeError, match=r"names no slot"):
        composite_drift([lambda t, position: position])


def test_howards_running_cost_is_read_by_the_name_of_its_slot():
    bound = cp.bind_user_callable(lambda t_idx: 2.0 * t_idx, cp.HOWARD_RUNNING_COST_SLOTS, role="running_cost")
    assert bound(t_idx=3) == 6.0
    with pytest.raises(TypeError, match=r"names no slot"):
        cp.bind_user_callable(lambda n: n, cp.HOWARD_RUNNING_COST_SLOTS, role="running_cost")


# ----------------------------------------------------------------------------------- the Hamiltonian, on each batch path


class _RowWise(HamiltonianBase):
    def __call__(self, t, x, p, m):
        return 0.5 * np.sum(np.asarray(p, dtype=float) ** 2, axis=-1) + 0.1 * np.asarray(m)

    def dp(self, t, x, p, m):
        return np.asarray(p, dtype=float)

    def dm(self, t, x, p, m):
        return 0.1 * np.ones_like(np.asarray(m, dtype=float))


class _PointOnly(_RowWise):
    def __call__(self, t, x, p, m):
        return 0.5 * np.asarray(p, dtype=float)[0] ** 2 + 0.1 * np.asarray(m)

    def dp(self, t, x, p, m):
        along_the_first_axis = np.zeros_like(np.asarray(p, dtype=float))
        along_the_first_axis[..., 0] = np.asarray(p, dtype=float)[..., 0]
        return along_the_first_axis


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_a_problem_refuses_a_point_only_hamiltonian_whichever_path_would_read_it(dimension):
    """#2481: HJB-FDM in d >= 2, the GFDM batch path and the semi-Lagrangian solver never read a check. It is one check now,
    where a problem takes the Hamiltonian, so none of them can be reached with one that disagrees."""
    with pytest.raises(ValueError, match=r"_PointOnly.evaluate_H, at"):
        _problem(dimension=dimension, n=5, hamiltonian=_PointOnly())


@pytest.mark.parametrize(
    "solver",
    [
        lambda problem: HJBFDMSolver(problem),
        lambda problem: HJBSemiLagrangianSolver(problem),
        lambda problem: HJBGFDMSolver(problem, collocation_points=problem.geometry.get_spatial_grid(), delta=0.6),
    ],
    ids=["HJB-FDM", "HJB-SL", "HJB-GFDM"],
)
def test_each_batch_path_solves_with_the_row_wise_form_of_it(solver):
    problem = _problem(dimension=2, n=5, hamiltonian=_RowWise(), u_terminal=lambda x: x[0] * 0.1)
    n_points = problem.geometry.get_spatial_grid().reshape(-1, 2).shape[0]
    shape = (5, 5) if not isinstance(solver(problem), HJBGFDMSolver) else (n_points,)
    terminal = np.zeros(shape)
    density = np.ones((problem.Nt + 1, *shape))
    result = solver(problem).solve_hjb_system(
        M_density=density, U_terminal=terminal, U_coupling_prev=np.zeros_like(density)
    )
    assert np.all(np.isfinite(result))


def test_a_hamiltonian_that_never_reads_a_batch_is_not_read_on_one():
    dual = DualHamiltonian(lambda t, x, alpha, m: 0.5 * float(np.sum(np.asarray(alpha) ** 2)) + float(np.ravel(m)[0]))
    assert dual.reads_a_batch is False
    refuse_a_hamiltonian_whose_reads_disagree(dual, np.linspace(0.0, 1.0, 11))


# ----------------------------------------------------------------------------------- constants and fixed returns at doors


def test_a_constant_noise_matrix_reaches_the_door_that_judges_it():
    grid = _grid(2, 5)
    S = np.array([[0.3, 0.1], [0.1, 0.2]])

    def build(**kind):
        return MFGProblem(
            model=Model(hamiltonian=_hamiltonian(), volatility=lambda t, x, m: S, **kind),
            domain=grid,
            conditions=Conditions(u_terminal=lambda x: 0.0, m_initial=lambda x: 1.0, T=0.2),
            Nt=2,
        )

    at = tuple(np.zeros((2, 4)))
    with pytest.raises(ValueError, match="volatility_kind='tensor'"):
        build().diffusion(0.0, at, None)
    np.testing.assert_allclose(
        build(volatility_kind="tensor").diffusion(0.0, at, None), 0.5 * S @ S.T, rtol=0, atol=1e-14
    )


def test_a_coefficient_of_the_wrong_length_reaches_the_door_that_judges_it():
    field = CoefficientField(lambda t, x, m: np.zeros(7), 0.1, "volatility")
    with pytest.raises(ValueError, match="returned array with shape"):
        field.evaluate_at(0, np.linspace(0.0, 1.0, 5), np.ones(5), 0.0)


# ---------------------------------------------------------------------------------------- what holds the checked state


def _module_level_potential(t, x):
    return 0.5 * x[0] + t


def test_an_object_holding_a_binding_pickles_and_reads_the_same_after():
    bound = cp.bind_user_callable(_module_level_potential, cp.POTENTIAL_SLOTS, role="potential")
    before = bound(t=0.25, x=(A,))
    assert bound.checked
    again = pickle.loads(pickle.dumps(bound))
    assert again.checked
    np.testing.assert_array_equal(again(t=0.25, x=(A,)), before)


def test_a_hamiltonian_holding_a_user_potential_pickles_and_reads_the_same_after():
    hamiltonian = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0), potential=_module_level_potential
    )
    p, m = np.zeros((5, 1)), np.ones(5)
    before = hamiltonian(t=0.25, x=(A,), p=p, m=m)
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(hamiltonian))(t=0.25, x=(A,), p=p, m=m), before)
