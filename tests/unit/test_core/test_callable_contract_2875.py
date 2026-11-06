"""A user callable reads what it names, ``x`` is a coordinate tuple, and its batch and point reads agree (#2875, #2872, #2481).

The oracle is the contract's own words (`CONVENTIONS.md` § *Callable signatures*): a parameter named for a slot is given
that slot, by name; ``x`` holds one entry per coordinate, each a scalar or an array; and a body the library may read
either way must read both the same, or it is refused where it is first read on a batch.
"""

from __future__ import annotations

import pickle
import re

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
from mfgarchon.alg.numerical.fp_solvers.fp_particle import FPParticleSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver, HJBGFDMSolver, HJBSemiLagrangianSolver, HJBWENOSolver
from mfgarchon.alg.numerical.hjb_solvers.hjb_howard import HJBHowardSolver
from mfgarchon.alg.numerical.pde_solvers.implicit_heat import ImplicitHeatSolver
from mfgarchon.core.hamiltonian import (
    CongestionHamiltonian,
    DualHamiltonian,
    HamiltonianBase,
    LagrangianBase,
    QuadraticControlCost,
    SeparableHamiltonian,
    refuse_a_hamiltonian_whose_reads_disagree,
)
from mfgarchon.core.measure import ParticleMeasure
from mfgarchon.core.measure_field import FunctionalMeasureField
from mfgarchon.factory import lq_mfg_initial_density, lq_mfg_terminal_cost
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.geometry.boundary.nitsche_1d import Nitsche1DPoissonSolver
from mfgarchon.geometry.implicit import ImplicitDomain
from mfgarchon.geometry.level_set.core import LevelSetEvolver
from mfgarchon.operators.interaction import PotentialCost, PotentialEnergy
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


def test_a_body_that_writes_into_its_field_sees_the_callers_field_once():
    """The comparison reads a field door's body several times, and read it on the caller's own ``m``: ``m *= 2`` doubled the
    density at each read, 32 for 2 after one call, and moved a solve (review 1). The comparison reads copies, so the
    caller's field is written as often as the body is called on it, once."""

    def doubles(t, x, m):
        m *= 2.0
        return x[0] + m

    density = np.ones(5)
    read = cp.bind_user_callable(doubles, cp.COEFFICIENT_SLOTS, role="drift")(t=0.0, x=(A,), m=density)
    np.testing.assert_array_equal(density, np.full(5, 2.0))
    np.testing.assert_allclose(read, A + 2.0, rtol=0, atol=1e-15)


def test_a_body_that_reads_an_attribute_the_tuple_does_not_have_is_refused_by_name():
    """``x.shape`` raised a bare AttributeError from the first batch, with no word of the tuple (review 1)."""
    with pytest.raises(ValueError, match="raised AttributeError"):
        _read(lambda x: np.zeros(x.shape[:1]))


# ------------------------------------------------------------------------------------------- the forms a return takes


@pytest.mark.parametrize(
    ("body", "x"),
    [
        (lambda x: 2.0, (A,)),
        (lambda x: 2.0 * x[0], (A,)),
        (lambda x: np.stack([x[0], x[1]], axis=-1), (A, B)),
        (lambda x: np.stack([x[0], x[1]], axis=0), (A, B)),
    ],
    ids=[
        "a scalar",
        "one entry per point",
        "per coordinate, the axis last",
        "per coordinate, the axis first",
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


def test_a_door_that_reads_a_fixed_array_refuses_one_that_changes_shape_at_a_point():
    with pytest.raises(ValueError, match=r"a return of shape \(7,\) has no entry per point"):
        _read(lambda x: np.zeros(3) if np.ndim(x[0]) == 0 else np.zeros(7), slots=cp.VOLATILITY_SLOTS)


def test_a_fixed_array_with_the_wrong_values_at_a_point_is_refused():
    with pytest.raises(ValueError, match="reads differently"):
        _read(lambda x: np.array([1.0, 2.0, 3.0]) * (1.0 if np.ndim(x[0]) else x[0]), slots=cp.VOLATILITY_SLOTS)


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


def test_a_coordinate_axis_cannot_pass_as_a_batch_of_exactly_d_points():
    """On this symmetric batch a body reading across the coordinate axis returns what a right one does, entry by entry,
    so only a count other than d can tell them apart."""
    x = (np.array([0.2, 0.7]), np.array([0.7, 0.4]))
    np.testing.assert_allclose(np.linalg.norm(x, axis=-1), np.hypot(*x), rtol=0, atol=1e-14)
    with pytest.raises(ValueError, match=r"a return of shape \(2,\) has no entry per point"):
        _read(lambda x: np.linalg.norm(x, axis=-1), x=x)
    np.testing.assert_allclose(_read(lambda x: np.linalg.norm(x, axis=0), x=x), np.hypot(*x), rtol=0, atol=1e-14)


@pytest.mark.parametrize(
    ("points", "dimension"), [(np.zeros((3, 2)), 3), (np.array(1.0), 2)], ids=["(3, 2) in 3-D", "0-D"]
)
def test_points_with_no_coordinate_axis_are_refused(points, dimension):
    with pytest.raises(ValueError, match=r"hold no coordinate axis of length"):
        cp.coordinate_tuple(points, dimension)


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


def _dead_zone():
    return lambda t, x, m: np.maximum(np.abs(np.max(x[0])) - 0.5, 0.0) * np.ones_like(x[0])


def _particle_solve(drift, high):
    FPParticleSolver(_problem(), num_particles=40, seed=0).solve_fp_system(
        initial_particles=np.random.default_rng(0).uniform(0.1, high, (40, 1)),
        drift_field=drift,
        drift_needs_density=False,
        show_progress=False,
    )


def test_a_first_batch_inside_a_dead_zone_cannot_see_the_misread_and_that_is_the_limit():
    """The check reads what the door's first batch holds. A particle drift's first batch is the initial cloud: inside the
    dead zone a body reading the cloud's largest coordinate reads the same both ways, and it is accepted. Outside it the
    same body, written again, is refused, so the limit is the first batch and not the body. The Hamiltonian does not have
    this limit, since it is read on wide synthetic p and m."""
    _particle_solve(_dead_zone(), high=0.4)
    with pytest.raises(ValueError, match="reads differently"):
        _particle_solve(_dead_zone(), high=0.95)


def test_a_body_compared_in_one_solve_is_not_compared_in_the_next():
    """What the comparison found is kept with the body, so the body accepted inside the dead zone is read with no
    comparison in a second solve, whose first batch would refuse it: the limit holds past the first solve. A fresh binding
    per solve would compare it again and refuse it."""
    accepted = _dead_zone()
    _particle_solve(accepted, high=0.4)
    _particle_solve(accepted, high=0.95)


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


def test_a_closure_the_library_marks_is_the_librarys_in_a_module_run_as_a_script():
    """``python -m`` runs a library module as ``__main__``, so the particle solver's ``navigable`` closure, whose module was
    then ``__main__``, was refused as a user's in its own ``__main__`` block. The library marks the closure it calls itself;
    a body written in that block, as a user writes one, is still a user's and is handed the tuple."""

    def made_in_main(body):
        body.__module__ = "__main__"
        return body

    assert cp._defined_by_the_library(cp._held_by_the_library(made_in_main(lambda points: points)))
    assert not cp._defined_by_the_library(made_in_main(lambda x: x[0]))


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


@pytest.mark.parametrize(
    ("build", "term"),
    [
        (
            lambda **kw: SeparableHamiltonian(
                QuadraticControlCost(control_cost=1.0), coupling=kw["f"], coupling_dm=kw["df"]
            ),
            "coupling",
        ),
        (
            lambda **kw: CongestionHamiltonian(
                QuadraticControlCost(control_cost=1.0), kw["f"], kw["df"], coupling=lambda m: 0.0 * m
            ),
            "congestion_factor",
        ),
    ],
    ids=["separable", "congestion"],
)
def test_a_density_term_is_read_by_name_on_a_batch(build, term):
    """No read per point: a body that cannot read a batch raises, and a constant holds for every point."""
    x = (np.linspace(0.0, 1.0, 3),)
    p = np.ones((3, 1))
    m = np.array([1.0, 2.0, 4.0])
    good = build(f=lambda m: 1.0 + m, df=lambda m: 1.0)
    assert np.all(np.isfinite(good(x=x, m=m, p=p, t=0.0)))
    assert np.all(np.isfinite(good.dm(x=x, m=m, p=p, t=0.0)))
    with pytest.raises(TypeError, match=rf"{term} .*names no slot"):
        build(f=lambda rho: 1.0 + rho, df=lambda m: 1.0)
    with pytest.raises(TypeError, match=r"only 0-dimensional|only length-1"):
        build(f=lambda m: float(m), df=lambda m: 1.0)(x=x, m=m, p=p, t=0.0)


def test_a_coupling_returns_a_value_per_point_and_a_constant_holds_for_all():
    H = SeparableHamiltonian(QuadraticControlCost(control_cost=1.0), coupling=lambda m: m**2, coupling_dm=lambda m: 3.0)
    x = (np.linspace(0.0, 1.0, 3),)
    p, m = np.zeros((3, 1)), np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(H(x=x, m=m, p=p, t=0.0), -(m**2))
    np.testing.assert_array_equal(H.dm(x=x, m=m, p=p, t=0.0), [-3.0, -3.0, -3.0])
    assert H(x=(0.5,), m=2.0, p=np.zeros(1), t=0.0) == pytest.approx(-4.0)
    named = SeparableHamiltonian(QuadraticControlCost(control_cost=1.0), coupling=lambda scale=2.0, m=None: scale * m)
    np.testing.assert_array_equal(named(x=x, m=m, p=p, t=0.0), -2.0 * m)
    assert named(x=(0.5,), m=2.0, p=np.zeros(1), t=0.0) == pytest.approx(-4.0)


@pytest.mark.parametrize("population", [0, 1])
def test_a_stacked_density_still_reads_the_first_populations_entries(population):
    """Not a contract: what #2335 will change, pinned so that this PR moves no value on a multi-population solve. A body
    that reads the stack itself, by its own defaults, is left to."""
    x = (np.linspace(0.0, 1.0, 4),)
    p = np.zeros((4, 1))
    first, second = np.array([1.0, 2.0, 3.0, 4.0]), np.array([10.0, 20.0, 30.0, 40.0])
    stacked = np.concatenate([first, second])
    cost = QuadraticControlCost(control_cost=1.0)
    H = SeparableHamiltonian(cost, coupling=lambda m: m, population_index=population)
    np.testing.assert_array_equal(H(x=x, m=stacked, p=p, t=0.0), -first)
    own = SeparableHamiltonian(cost, coupling=lambda m, k=population: m[4 * k : 4 * k + 4], population_index=population)
    np.testing.assert_array_equal(own(x=x, m=stacked, p=p, t=0.0), -(second if population else first))


def test_a_node_rule_is_read_by_name():
    from mfgarchon.geometry.boundary.applicator_graph import NodeBC

    assert NodeBC(nodes=lambda num_nodes: [num_nodes - 1]).get_nodes(5) == [4]
    assert NodeBC(nodes=lambda offset=2, num_nodes=None: [num_nodes - offset]).get_nodes(5) == [3]
    with pytest.raises(TypeError, match=r"nodes .*names no slot"):
        NodeBC(nodes=lambda count: [count - 1]).get_nodes(5)


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


class _ADiscAUserWrote(ImplicitDomain):
    """Its method is the `ImplicitGeometry` contract: points as an array, though its module is not the library's."""

    dimension = 2

    def signed_distance(self, x):
        return np.linalg.norm(np.atleast_2d(np.asarray(x, dtype=float)), axis=1) - 0.5

    def get_bounding_box(self):
        return np.array([[-0.5, 0.5], [-0.5, 0.5]])


def test_a_geometrys_own_signed_distance_keeps_its_array_on_a_subclass_a_user_wrote():
    points = np.array([[0.5, 0.0], [0.0, 0.5], [0.3, 0.4]])
    normals = _ADiscAUserWrote().get_boundary_normal(points)
    np.testing.assert_allclose(normals, points / 0.5, rtol=0, atol=1e-6)
    from mfgarchon.geometry.collocation import ImplicitDomainCollocation

    from_collocation = ImplicitDomainCollocation(_ADiscAUserWrote())._compute_normals(points)
    np.testing.assert_allclose(from_collocation, points / 0.5, rtol=0, atol=1e-6)


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
        initial_node_density_func=lambda t, node: t + node + 1.0, terminal_node_value_func=lambda t, node: t * node
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


def test_a_variational_terminal_cost_reads_a_coordinate_tuple():
    from mfgarchon.alg.optimization.variational_problem import VariationalMFGComponents, VariationalMFGProblem

    components = VariationalMFGComponents(
        lagrangian_func=lambda t, x, v, m: 0.5 * v**2, terminal_cost_func=lambda x: 2.0 * x[0], m_initial=lambda x: 1.0
    )
    problem = VariationalMFGProblem(xmin=0.0, xmax=1.0, Nx=10, T=0.2, Nt=4, components=components)
    np.testing.assert_allclose(problem.evaluate_terminal_cost(np.array([0.25, 0.5])), [0.5, 1.0], rtol=0, atol=1e-14)
    problem.use_jax = False  # the route with no JAX, where the door is the method's own
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


class _DpAcrossTheBatch(_RowWise):
    """Its value reads row by row, and its dp subtracts the batch's mean: alone, a point is its own mean."""

    def dp(self, t, x, p, m):
        p = np.asarray(p, dtype=float)
        return p - p.mean(axis=0, keepdims=True)


def test_the_check_reads_a_hamiltonians_dp_as_well_as_its_value():
    with pytest.raises(ValueError, match=r"_DpAcrossTheBatch.evaluate_dp, at"):
        refuse_a_hamiltonian_whose_reads_disagree(_DpAcrossTheBatch(), np.linspace(0.0, 1.0, 11))


def test_a_lagrangians_proximal_hands_its_stand_in_x_as_a_tuple():
    """With no ``x`` given, the search still calls the Lagrangian, and that call reads the contract's form."""
    seen = []

    class _Recording(LagrangianBase):
        def __call__(self, t, x, alpha, m):
            seen.append(type(x))
            return 0.5 * float(np.sum(np.asarray(alpha) ** 2))

    _Recording().proximal(0.5, np.array([0.4]))
    assert seen
    assert set(seen) == {tuple}


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
        lambda problem: HJBGFDMSolver(
            problem, collocation_points=problem.geometry.get_spatial_grid(), delta=0.6, monotonicity_scheme="none"
        ),
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


_HELD = np.linspace(0.1, 0.9, 15)


def _five_by_three():
    return TensorProductGrid(bounds=[(0.0, 1.0)] * 2, Nx_points=[5, 3], boundary_conditions=no_flux_bc(dimension=2))


def _on_five_by_three(**doors):
    conditions = {
        "u_terminal": doors.pop("u_terminal", lambda x: 0.0),
        "m_initial": doors.pop("m_initial", lambda x: 1.0),
    }
    return MFGProblem(
        model=Model(hamiltonian=_hamiltonian(), volatility=0.3),
        domain=_five_by_three(),
        conditions=Conditions(**conditions, T=0.2),
        Nt=2,
        **doors,
    )


@pytest.mark.parametrize("door", ["u_terminal", "m_initial"])
@pytest.mark.parametrize("shape", [(5, 3), (3, 5), (15,), (5, 1), (2,)])
def test_a_condition_holding_an_array_is_refused_whatever_its_shape(door, shape):
    """A condition is taken by its size, so a transposed array would be read in C order with no error, and `Conditions`
    holds a function of the continuum, not a projection onto one grid."""
    held = np.resize(_HELD, shape)
    with pytest.raises(ValueError, match="no entry per point"):
        _on_five_by_three(**{door: lambda x: held})


@pytest.mark.parametrize("door", ["source_term_hjb", "source_term_fp"])
@pytest.mark.parametrize("shape", [(3, 5), (15,), (5, 1), (2,)])
def test_a_problem_source_returning_no_entry_per_point_is_refused(door, shape):
    held = np.resize(_HELD, shape)
    with pytest.raises(ValueError, match="no entry per point"):
        _on_five_by_three(**{door: lambda t, x: held}).solve(max_iterations=1, tolerance=1e-3, verbose=False)


@pytest.mark.parametrize("door", ["source_term_hjb", "source_term_fp"])
def test_a_field_door_reads_the_entries_of_a_return_with_the_grids_shape(door):
    """Why such a return is read at all: a body may read the field whole, as ``lambda m: m`` does. It must read the same
    as a body that computes each point, on a grid whose axes cannot be swapped unseen."""
    x, y = np.meshgrid(np.linspace(0.0, 1.0, 5), np.linspace(0.0, 1.0, 3), indexing="ij")
    held = x + 2.0 * y

    def solve(body):
        return _on_five_by_three(**{door: body}).solve(max_iterations=1, tolerance=1e-3, verbose=False)

    read = solve(lambda t, x: held)
    written = solve(lambda t, x: x[0] + 2.0 * x[1])
    np.testing.assert_allclose(read.U, written.U, rtol=0, atol=1e-13)
    np.testing.assert_allclose(read.M, written.M, rtol=0, atol=1e-13)
    assert np.ptp(read.U[0] if door == "source_term_hjb" else read.M[-1]) > 1e-3  # the source reached the solve


_NO_FIELD = [cp.CONDITION_SLOTS, cp.POTENTIAL_SLOTS, cp.REGION_SLOTS, cp.SOLVER_SOURCE_SLOTS, cp.EXACT_SOLUTION_SLOTS]


@pytest.mark.parametrize("slots", _NO_FIELD, ids=lambda slots: "+".join(slots.order))
def test_a_door_with_no_field_reads_no_entry_from_a_batch_shaped_return(slots):
    x = (np.linspace(0.0, 1.0, 7),)
    held = np.linspace(0.1, 0.9, 7)
    with pytest.raises(ValueError, match="no entry per point"):
        cp.bind_user_callable(lambda x: held, slots, role="r")(x=x, t=0.0)


@pytest.mark.parametrize(
    "slots",
    [*_NO_FIELD, cp.COEFFICIENT_SLOTS, cp.SOURCE_TERM_SLOTS, cp.MEASURE_FIELD_SLOTS, cp.BOUNDARY_VALUE_SLOTS],
    ids=lambda slots: "+".join(slots.order) + (" with a field" if slots.fields else ""),
)
def test_a_door_that_does_not_say_so_reads_no_fixed_array(slots):
    x = (np.linspace(0.0, 1.0, 7), np.linspace(1.0, 2.0, 7))
    held = np.array([[0.3, 0.1], [0.1, 0.2]])
    with pytest.raises(ValueError, match=r"a return of shape \(2, 2\) has no entry per point"):
        cp.bind_user_callable(lambda x: held, slots, role="r")(x=x, t=0.0)


def test_a_constant_vector_drift_is_refused_by_the_check():
    x = (np.linspace(0.0, 1.0, 7), np.linspace(1.0, 2.0, 7))
    with pytest.raises(ValueError, match=r"a return of shape \(2,\) has no entry per point"):
        cp.bind_user_callable(lambda t, x, m: np.array([0.1, 0.2]), cp.COEFFICIENT_SLOTS, role="drift_field")(
            x=x, t=0.0, m=np.ones(7)
        )


def test_only_the_noise_matrix_door_says_it_reads_a_fixed_array():
    stated = {name for name, value in vars(cp).items() if isinstance(value, cp.Slots) and value.fixed_return}
    assert stated == {"VOLATILITY_SLOTS"}


@pytest.mark.parametrize("dimension", [1, 2])
@pytest.mark.parametrize("door", ["source_term_hjb", "source_term_fp"])
def test_a_constant_source_reads_as_the_same_value_at_every_point(door, dimension):
    """``lambda t, x: 0.5`` raised a bare AttributeError from the HJB's read of the composed source, at d = 1 and d = 2."""

    def solve(body):
        return _problem(dimension=dimension, n=5, **{door: body}).solve(max_iterations=1, tolerance=1e-3, verbose=False)

    constant = solve(lambda t, x: 0.5)
    per_point = solve(lambda t, x: 0.5 + 0.0 * x[0])
    np.testing.assert_array_equal(constant.U, per_point.U)
    np.testing.assert_array_equal(constant.M, per_point.M)


@pytest.mark.parametrize("dimension", [1, 2])
def test_a_constant_source_handed_to_a_solver_reads_at_every_point(dimension):
    problem = _problem(dimension=dimension, n=5)
    points = np.asarray(problem.geometry.get_spatial_grid(), dtype=float)
    value = cp.evaluate_solver_source(lambda t, x: 0.5, t=0.0, x=points, dimension=dimension)
    np.testing.assert_array_equal(value, np.full(points.shape[0], 0.5))
    shape = problem.geometry.get_grid_shape()
    density = np.ones((5, *shape)) / points.shape[0]
    terminal, previous = np.zeros(shape), np.zeros((5, *shape))
    constant = HJBFDMSolver(problem).solve_hjb_system(density, terminal, previous, source_term=lambda t, x: 0.5)
    per_point = HJBFDMSolver(problem).solve_hjb_system(
        density, terminal, previous, source_term=lambda t, x: 0.5 + 0.0 * x[0]
    )
    np.testing.assert_array_equal(constant, per_point)


@pytest.mark.filterwarnings("ignore:initial density mass")  # the advice's Gaussian is not of mass 1, which is reported
@pytest.mark.parametrize("door", ["m_initial", "u_terminal"])
def test_the_advice_conditions_give_on_an_array_runs_as_written(door):
    """Advice is a claim: the function the refusal writes out is accepted, and reads the point it names."""
    with pytest.raises(TypeError) as refused:
        Conditions(**{door: np.ones(3)})
    written = re.search(rf"{door}=(lambda x: [^ ]+)", str(refused.value)).group(1).rstrip(".")
    body = eval(written, {"np": np})
    assert float(body((0.5,))) == pytest.approx(1.0 if door == "m_initial" else 0.0)
    _on_five_by_three(**{door: body})


@pytest.mark.parametrize(("dimension", "nodes"), [(1, 21), (2, 21), (3, 11)])
def test_the_constant_the_refusal_says_to_compute_runs_on_a_tensor_grid(dimension, nodes):
    with pytest.raises(TypeError) as refused:
        Conditions(m_initial=np.ones(3))
    message = str(refused.value)
    statement = re.search(r"`(c = [^`]+)`", message).group(1)
    f = eval(re.search(r"m_initial=(lambda x: [^ ]+)", message).group(1).rstrip("."), {"np": np})
    domain = _grid(dimension, nodes)
    scope = {"domain": domain, "f": f}
    exec(statement, scope)
    # the closed form, one Gaussian per the first coordinate, against a fine quadrature on the same box
    fine = np.linspace(0.0, 1.0, 20001)
    exact = float(np.trapezoid(np.exp(-5 * (fine - 0.5) ** 2), fine))
    assert scope["c"] == pytest.approx(exact, rel=1e-2)


# ------------------------------------------------------------------------------------ doors a wider census found


def test_a_potential_energy_reads_v_by_name_with_x_a_tuple():
    """At n = d an array's first axis passes for a coordinate, so ``x[1]`` read the second point with no error."""
    points = np.array([[0.1, 0.2], [0.3, 0.4]])

    def read(V):
        return PotentialEnergy(np.zeros(2), 1.0, V=V).flat_derivative_at(np.ones(2), points)

    np.testing.assert_allclose(read(lambda x: x[1]), [0.2, 0.4], rtol=0, atol=1e-14)
    np.testing.assert_allclose(read(lambda x: 2.0), [2.0, 2.0], rtol=0, atol=1e-14)
    with pytest.raises((TypeError, ValueError)):
        read(lambda x: x[:, 1])
    with pytest.raises(TypeError, match="names no slot"):
        PotentialEnergy(np.zeros(2), 1.0, V=lambda y: y[0])
    with pytest.raises(TypeError, match="V takes t"):
        PotentialEnergy(np.zeros(2), 1.0, V=lambda t, x: x[0])


def test_a_potential_cost_binds_its_v_once_and_evaluates_it_elsewhere():
    calls = []

    def V(x):
        calls.append(np.shape(x[0]))
        return x[0] + 2.0 * x[1]

    problem = _problem(dimension=2, n=5)
    energy = PotentialCost(V).bind(problem)
    points = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    np.testing.assert_allclose(energy.flat_derivative_at(np.ones(25) / 25, points), [0.5, 1.1, 1.7], rtol=0, atol=1e-14)
    assert len([shape for shape in calls if shape == ()]) == 3  # one comparison, of three samples


def _howard_solve(control_lagrangian):
    points = np.linspace(0.0, 1.0, 21).reshape(-1, 1)
    problem = _problem(n=21, u_terminal=lambda x: (x[0] - 0.5) ** 2)
    howard = HJBHowardSolver(
        problem,
        stencil_provider=HJBGFDMSolver(problem, collocation_points=points, monotonicity_scheme="none"),
        alpha_star=lambda t_idx, x, p, m: -p,
        control_lagrangian=control_lagrangian,
    )
    return np.asarray(howard.solve_hjb_system(M_density=None, U_terminal=(points.ravel() - 0.5) ** 2))


@pytest.mark.filterwarnings("ignore:HJBHowardSolver. running on non-SOCP")  # this row is the Lagrangian's read
def test_a_control_lagrangian_is_read_by_the_name_it_declares():
    named_second = _howard_solve(lambda alpha, c=3.0: np.sum(c * alpha * alpha, axis=1) / 2)
    named_first = _howard_solve(lambda c=3.0, alpha=0.0: np.sum(c * alpha * alpha, axis=1) / 2)
    np.testing.assert_array_equal(named_first, named_second)
    assert np.abs(named_second - _howard_solve(lambda alpha, c=1.0: np.sum(c * alpha * alpha, axis=1) / 2)).max() > 1e-3
    with pytest.raises(TypeError, match="names no slot"):
        _howard_solve(lambda a: np.sum(a * a, axis=1) / 2)


@pytest.mark.filterwarnings("ignore:initial density mass")  # the library's bimodal density is not of mass 1
def test_the_conditions_the_library_writes_read_a_coordinate_tuple():
    u_terminal, m_initial = lq_mfg_terminal_cost(Lx=2.0), lq_mfg_initial_density()
    at = np.linspace(0.0, 1.0, 5)
    np.testing.assert_allclose(
        u_terminal((at,)), 5 * (np.cos(at * np.pi) + 0.4 * np.sin(at * 2 * np.pi)), rtol=0, atol=1e-14
    )
    np.testing.assert_allclose(
        m_initial((at,)), 2 * np.exp(-200 * (at - 0.2) ** 2) + np.exp(-200 * (at - 0.8) ** 2), rtol=0, atol=1e-14
    )
    assert _problem(u_terminal=u_terminal, m_initial=m_initial, n=21).get_u_terminal().size == 21


@pytest.mark.parametrize("dimension", [1, 2])
def test_the_documented_measure_field_body_reads_a_coordinate_tuple(dimension):
    measure = ParticleMeasure(np.random.default_rng(0).uniform(0.2, 0.8, (40, dimension)), np.full(40, 1 / 40))
    at = np.linspace(0.3, 0.7, 5).reshape(-1, 1) * np.ones((5, dimension))

    def v_fn(t, x, mu):
        m = mu.to_density(np.stack(x, axis=-1).reshape(-1, len(x)))
        return -np.log(np.maximum(m, 1e-15))

    np.testing.assert_allclose(
        FunctionalMeasureField(v_fn).evaluate(0.1, at, measure),
        -np.log(np.maximum(measure.to_density(at), 1e-15)),
        rtol=1e-12,
        atol=1e-12,
    )


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


# ------------------------------------------------- what review 1 of #2896 measured, each pinned at the door it was measured


def _misread(x):
    """Reads across the batch: the largest first coordinate, at every point, and no points on no points."""
    return np.max(x[0], initial=-np.inf) + 0.0 * x[0]


_BATCH = np.linspace(0.1, 0.9, 6).reshape(-1, 1)
_SAME = np.full((6, 1), 0.5)  # a batch every body reads the same both ways


def _doors_that_keep_a_body():
    """Doors that read one body many times, each as a function of the points, by the routes review 1 measured."""
    from mfgarchon.utils.drift_helpers import composite_drift
    from mfgarchon.utils.manufactured import ManufacturedPair

    bound = cp.bind_user_callable(_misread, cp.REGION_SLOTS, role="r")
    source = lambda t, x: _misread(x)  # noqa: E731
    region = lambda x: _misread(x)  # noqa: E731
    field = FunctionalMeasureField(lambda t, x, mu: _misread(x))
    zero = lambda t, x: 0.0 * x[0]  # noqa: E731
    pair = ManufacturedPair(
        u=lambda t, x: _misread(x), u_t=zero, grad_u=zero, hess_u=zero, m=zero, m_t=zero, grad_m=zero, hess_m=zero
    )
    drift = composite_drift([lambda t, x, m: _misread(x)])
    return {
        "binder": lambda points: bound(x=(points[:, 0],)),
        "solver source": lambda points: cp.evaluate_solver_source(source, t=0.0, x=points, dimension=1),
        "region": lambda points: cp.read_region(region, points),
        "measure field": lambda points: field.evaluate(0.0, points, None),
        "manufactured pair": lambda points: pair.at("u", 0.1, points),
        "composite drift": lambda points: drift(0.0, (points[:, 0],), np.ones(len(points))),
    }


@pytest.mark.parametrize("door", sorted(_doors_that_keep_a_body()))
def test_a_refused_body_is_compared_again_at_its_next_read(door):
    """Run again in a notebook cell, or in a sweep's try/except, a refused body was read on its second batch: the body had
    been marked compared before the comparison ran (review 1). A refusal is not remembered, so the next read compares
    again: on the same batch it is refused again, and on a batch it reads the same both ways it is read (CONVENTIONS § 6,
    "What this does not catch")."""
    read = _doors_that_keep_a_body()[door]
    for _ in range(2):
        with pytest.raises(ValueError, match="reads differently"):
            read(_BATCH)
    np.testing.assert_allclose(np.ravel(read(_SAME)), 0.5, rtol=0, atol=1e-15)


def test_a_batch_whose_compared_rows_are_one_point_leaves_the_body_uncompared():
    """The comparison reads a few rows of a batch. When every row it reads is the same point it has compared one point
    with itself, so the body stays not yet compared, however many distinct points the rest of the batch holds, and the
    next batch is compared (review 2)."""
    bound = cp.bind_user_callable(
        lambda x: x[0] + (np.max(x[0]) - np.min(x[0])) * (x[0] > 0.7), cp.REGION_SLOTS, role="r"
    )
    first = np.r_[np.full(20, 0.5), 0.9]
    assert set(first[cp.spread_samples(first.size, 3)]) == {0.5}, "the rows the comparison reads"
    bound(x=(first,))
    assert not bound.checked
    with pytest.raises(ValueError, match="reads differently"):
        bound(x=(np.linspace(0.1, 0.9, 6),))


def test_the_comparison_reads_copies_of_every_array_it_hands():
    """A body that writes into an array it is handed sees the caller's array once per read it makes, whatever slot holds
    it: the comparison's reads are of copies (review 2: an array ``mu`` was written at every one of them)."""

    def writes(t, x, mu):
        mu += 1.0
        return x[0]

    held = np.zeros(3)
    FunctionalMeasureField(writes).evaluate(0.0, _BATCH, held)
    np.testing.assert_array_equal(held, np.ones(3))


def test_a_mock_does_not_pass_for_a_closure_the_library_marked():
    from unittest.mock import MagicMock

    assert not cp._defined_by_the_library(MagicMock())


class _HandedError(Exception):
    pass


def test_the_particle_solvers_navigable_is_the_librarys_in_a_module_run_as_a_script(monkeypatch):
    """``python -m`` runs fp_particle as ``__main__``, where its ``navigable`` closure, read as a user's, was refused in its
    own ``__main__`` block: the solver marks the closure it hands its sampler, and the mark holds whatever its module."""
    import mfgarchon.alg.numerical.fp_solvers.fp_particle as fp_particle_module

    handed = {}

    def capture(**kwargs):
        handed.update(kwargs)
        raise _HandedError

    monkeypatch.setattr(fp_particle_module, "sample_from_density", capture)
    with pytest.raises(_HandedError):
        FPParticleSolver(_problem(dimension=2, n=5), num_particles=10, seed=0).solve_fp_system(
            np.ones((5, 5)) / 25, drift_field=lambda t, x, m: 0.0, show_progress=False
        )
    navigable = handed["navigable"]
    navigable.__module__ = "__main__"
    assert cp._defined_by_the_library(navigable)


# ------------------------------------------------------------------- the doors that read one point, recorded at d = 1


def test_the_upwind_probe_reads_a_hamiltonian_at_one_point_as_a_number_at_d_1():
    """The HJB-FDM upwind probe evaluates H at one point. Handed an array of one there, it read a Hamiltonian that the
    problem's check accepts on another branch, and refused it (review 2)."""

    class ArrayOfOneBranch(HamiltonianBase):
        def __call__(self, t, x, p, m):
            pa = np.asarray(p, dtype=float)
            base = 0.5 * np.sum(pa**2, axis=-1)
            return base + 0.1 * pa[..., 0] if np.ndim(x[0]) == 1 and np.size(x[0]) == 1 else base

    HJBFDMSolver(_problem(hamiltonian=ArrayOfOneBranch()), advection_scheme="gradient_upwind")


def test_the_semi_lagrangian_control_search_hands_its_lagrangian_one_point_as_numbers_at_d_1():
    """The semi-Lagrangian solver's search over controls reads the Lagrangian at one point; it handed x an array of one,
    so a potential inside it was compared at every read (review 2)."""
    from mfgarchon.core.hamiltonian import L1ControlCost

    forms = set()

    def potential(t, x):
        forms.add(tuple(np.shape(entry) for entry in x))
        return 0.4 * np.asarray(x[0]) ** 2

    problem = MFGProblem(
        model=Model(
            hamiltonian=SeparableHamiltonian(control_cost=L1ControlCost(lambda_=1.0), potential=potential),
            volatility=0.15,
        ),
        domain=_grid(n=11),
        conditions=Conditions(u_terminal=lambda x: (x[0] - 0.5) ** 2, m_initial=lambda x: 1.0, T=0.2),
        Nt=4,
    )
    forms.clear()
    nodes = np.linspace(0.0, 1.0, 11)
    solver = HJBSemiLagrangianSolver(problem)
    assert solver._use_dpp, "the control search runs"
    solver.solve_hjb_system(np.ones((5, 11)), (nodes - 0.5) ** 2, U_coupling_prev=np.zeros((5, 11)))
    assert ((),) in forms
    assert ((1,),) not in forms, forms


def test_the_meshfree_particle_handler_reads_its_sdf_at_one_point_as_numbers():
    from mfgarchon.geometry.boundary.applicator_meshfree import SDFParticleBCHandler

    forms = set()

    def disc(x):
        forms.add(tuple(np.shape(entry) for entry in x))
        return np.hypot(x[0] - 0.5, x[1] - 0.5) - 0.3

    SDFParticleBCHandler(disc, dimension=2).apply_bc(
        np.array([[0.5, 0.5], [0.7, 0.5]]), np.array([[0.55, 0.5], [0.9, 0.52]])
    )
    assert ((), ()) in forms
    assert ((1,), (1,)) not in forms, forms


@pytest.mark.parametrize("door", sorted(_doors_that_keep_a_body()))
def test_a_batch_of_one_point_leaves_a_body_uncompared_and_the_next_batch_is_compared(door):
    read = _doors_that_keep_a_body()[door]
    read(_BATCH[:1])
    with pytest.raises(ValueError, match="reads differently"):
        read(_BATCH)


@pytest.mark.parametrize("door", sorted(_doors_that_keep_a_body()))
def test_an_empty_batch_reads_empty_and_the_next_batch_is_compared(door):
    read = _doors_that_keep_a_body()[door]
    assert np.size(read(np.zeros((0, 1)))) == 0
    with pytest.raises(ValueError, match="reads differently"):
        read(_BATCH)


def test_a_region_predicate_reads_no_points_as_no_points():
    from mfgarchon.geometry.predicates import sdf_region

    assert sdf_region(lambda x: np.hypot(x[0], x[1]) - 0.5)(np.zeros((0, 2))).shape == (0,)


@pytest.mark.parametrize(
    "door",
    [
        lambda f, points: cp.evaluate_solver_source(f, t=0.0, x=points, dimension=1),
        lambda f, points: cp.read_region(f, points),
    ],
    ids=["solver source", "region"],
)
def test_a_door_that_binds_at_every_read_compares_a_body_once(door):
    calls = []

    def counted(x):
        calls.append(1)
        return x[0]

    door(counted, _BATCH)
    first = len(calls)
    assert first > 1, "the first batch is read together and at its samples"
    door(counted, _BATCH)
    door(counted, _BATCH)
    assert len(calls) == first + 2


def test_a_volatility_is_compared_once_in_a_solve(monkeypatch):
    """A solver reads its volatility at every step, and compared it at every step: 950 body calls against 190 (review 1).
    What the comparison found is kept with the body, so it is compared once."""
    compared = []
    original = cp._compare_reads

    def counted(read, values, slots, name, guidance):
        compared.append(name)
        return original(read, values, slots, name, guidance)

    monkeypatch.setattr(cp, "_compare_reads", counted)
    volatility = lambda t, x, m: 0.2 + 0.1 * x[0]  # noqa: E731
    problem = MFGProblem(
        model=Model(hamiltonian=_hamiltonian(), volatility=volatility),
        domain=_grid(),
        conditions=Conditions(u_terminal=lambda x: 0.0, m_initial=lambda x: 1.0, T=0.2),
        Nt=4,
    )
    problem.solve(max_iterations=2, tolerance=1e-3, verbose=False)
    assert sum(name.startswith("volatility ") for name in compared) == 1, compared


def test_a_problems_diffusion_reads_the_volatility_the_problem_holds():
    """``diffusion`` read a binding made at construction, so a volatility replaced afterwards left it stale: 0.005 for
    0.045 (review 1). It reads the volatility the problem holds when it is called."""
    problem = MFGProblem(
        model=Model(hamiltonian=_hamiltonian(), volatility=lambda t, x, m: 0.1 + 0.0 * x[0]),
        domain=_grid(),
        conditions=Conditions(u_terminal=lambda x: 0.0, m_initial=lambda x: 1.0, T=0.2),
        Nt=4,
    )
    problem._volatility = lambda t, x, m: 0.3 + 0.0 * x[0]
    diffusion = problem.diffusion(t=0.0, x=(np.linspace(0.0, 1.0, 3),), m=np.ones(3))
    np.testing.assert_allclose(diffusion, np.full(3, 0.045), rtol=0, atol=1e-15)


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_pure_diffusion_of_particles_reads_a_constant_callable_volatility_as_the_constant(dimension):
    """With no drift the particle solver's own zero drift is read on the tuple; it read it as an array and raised in every
    dimension (review 1). A constant callable volatility then moves the particles as the same constant held in an array,
    read on the same path. A scalar volatility takes another path, which returns one time level fewer."""
    n = 5
    shape = (n,) * dimension
    density = np.ones(shape) / n**dimension
    problem = _problem(dimension=dimension, n=n)

    def solve(volatility):
        return FPParticleSolver(problem, num_particles=200, seed=3).solve_fp_system(
            density.copy(), drift_field=None, volatility=volatility, volatility_kind="field", show_progress=False
        )

    np.testing.assert_allclose(solve(lambda t, x, m: 0.1), solve(np.full(shape, 0.1)), rtol=0, atol=1e-12)


def test_a_domain_subclass_that_overrides_contains_is_solved_on():
    """The particle solver reads its domain's ``contains`` itself, so a subclass a user wrote is not a callable it was handed:
    it was refused as a method, with advice to wrap what the user never passed (review 1)."""
    from mfgarchon.geometry.implicit import DifferenceDomain, Hyperrectangle, Hypersphere

    class Pierced(DifferenceDomain):
        def contains(self, x, tol=0.0):
            return super().contains(x, tol)

    box = Hyperrectangle(np.array([[0.0, 1.0], [0.0, 1.0]]))
    hole = Hypersphere(center=[0.5, 0.5], radius=0.3)
    density = np.zeros((21, 21))
    density[1:3, 1:3] = 1.0
    density /= density.sum()

    def solve(domain):
        return FPParticleSolver(
            _problem(dimension=2, n=21), num_particles=200, seed=3, implicit_domain=domain
        ).solve_fp_system(density.copy(), drift_field=lambda t, x, m: 0.0, show_progress=False)

    np.testing.assert_array_equal(solve(Pierced(box, hole)), solve(DifferenceDomain(box, hole)))


def test_a_stochastic_problems_own_terminal_cost_method_is_read_by_its_names():
    """A subclass's ``terminal_cost`` is read before the components' ``u_terminal``, by the names it declares past ``self``;
    a bound method was refused as not a plain function (review 1)."""
    from mfgarchon.core.mfg_components import MFGComponents
    from mfgarchon.core.stochastic.stochastic_problem import StochasticMFGProblem

    class Quadratic(StochasticMFGProblem):
        def terminal_cost(self, x):
            return 0.5 * x[0] ** 2

    components = MFGComponents(hamiltonian=_hamiltonian(), u_terminal=lambda x: 0.0, m_initial=lambda x: 1.0)
    problem = Quadratic(geometry=_grid(), T=0.2, Nt=4, volatility=0.3, components=components)
    np.testing.assert_allclose(problem.g_conditional(np.array([[0.2], [0.6]]), 0.0), [0.02, 0.18], rtol=0, atol=1e-15)


# --------------------------------------------------------------------- one point, handed as a tuple of scalars at d = 1


_A_POINT_AT_D1 = lambda x: float(x[0]) - 0.5  # noqa: E731  # float() reads one scalar, and raises on an array of one


def _one_point_doors():
    from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions
    from mfgarchon.geometry.boundary.ghost import compute_normal_from_sdf
    from mfgarchon.geometry.boundary.types import sdf_region_holds
    from mfgarchon.geometry.implicit.sdf_utils import sdf_gradient
    from mfgarchon.operators.differential.function_gradient import outward_normal_from_sdf

    point = np.array([0.5])
    conditions = BoundaryConditions(
        segments=[BCSegment(name="wall", bc_type=BCType.NEUMANN, value=0.0)], dimension=1, domain_sdf=_A_POINT_AT_D1
    )
    segment = BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=0.0, sdf_region=_A_POINT_AT_D1)
    return {
        "is_on_boundary": (lambda: conditions.is_on_boundary(point), True),
        "get_outward_normal": (lambda: conditions.get_outward_normal(point), [1.0]),
        "sdf_region_holds": (lambda: sdf_region_holds(segment, np.array([[0.2], [0.8]]), 0.0), [True, False]),
        "compute_normal_from_sdf": (lambda: compute_normal_from_sdf(point, _A_POINT_AT_D1), [1.0]),
        "outward_normal_from_sdf": (lambda: outward_normal_from_sdf(_A_POINT_AT_D1, point), [1.0]),
        "sdf_gradient": (lambda: sdf_gradient(point, _A_POINT_AT_D1), [1.0]),
    }


@pytest.mark.parametrize("door", sorted(_one_point_doors()))
def test_a_door_that_reads_one_point_hands_a_tuple_of_scalars_at_d_1(door):
    read, expected = _one_point_doors()[door]
    np.testing.assert_allclose(np.asarray(read(), dtype=float), np.asarray(expected, dtype=float), rtol=0, atol=1e-6)


class _RawH(HamiltonianBase):
    def __call__(self, t, x, p, m):
        return 0.5 * np.sum(np.asarray(p) ** 2, axis=-1) + 0.3 * np.asarray(m) + 0.5 * np.asarray(x[0])


@pytest.mark.parametrize(
    "solver",
    [lambda problem: HJBFDMSolver(problem, analytic_jacobian=False), HJBSemiLagrangianSolver, HJBWENOSolver],
    ids=["FDM per point", "semi-Lagrangian", "WENO"],
)
def test_a_per_point_path_reads_a_raw_hamiltonian_as_its_separable_twin(solver):
    """The per-point paths hand H one point; at d = 1 they handed x an array of one, and each raised (review 1)."""
    twin = SeparableHamiltonian(
        QuadraticControlCost(control_cost=1.0),
        potential=lambda t, x: -0.5 * x[0],
        coupling=lambda m: -0.3 * m,
        coupling_dm=lambda m: -0.3 + 0.0 * m,
    )
    nodes = np.linspace(0.0, 1.0, 21)
    u_terminal = np.cos(np.pi * nodes)
    density = np.tile(1.0 + 0.5 * np.sin(np.pi * nodes), (6, 1))

    def solve(hamiltonian):
        problem = MFGProblem(
            model=Model(hamiltonian=hamiltonian, volatility=0.3),
            domain=_grid(n=21),
            conditions=Conditions(u_terminal=lambda x: np.cos(np.pi * x[0]), m_initial=lambda x: 1.0, T=0.5),
            Nt=5,
        )
        return np.asarray(solver(problem).solve_hjb_system(density, u_terminal, np.tile(u_terminal, (6, 1))))

    np.testing.assert_allclose(solve(_RawH()), solve(twin), rtol=0, atol=1e-10)


# -------------------------------------------------------------------------- a region reads one point or rows of points


def test_a_region_refuses_a_1d_array_by_name_and_reads_a_column_as_points_on_a_line():
    from mfgarchon.geometry.predicates import sdf_region

    inside = sdf_region(lambda x: x[0] - 0.5)
    np.testing.assert_array_equal(inside(np.linspace(0.0, 1.0, 5)[:, None]), [True, True, True, False, False])
    with pytest.raises(ValueError, match=r"does not say whether it is N points on a line or one point in N dimensions"):
        inside(np.linspace(0.0, 1.0, 5))


def test_a_constant_level_set_velocity_moves_the_front_as_a_velocity_field_does():
    evolver = LevelSetEvolver(_grid(n=41))
    phi = np.linspace(0.0, 1.0, 41) - 0.3
    np.testing.assert_array_equal(
        evolver.evolve_step(phi, velocity=lambda x: 0.5, dt=0.01),
        evolver.evolve_step(phi, velocity=lambda x: 0.5 + 0.0 * x[0], dt=0.01),
    )


# ------------------------------------------------------------------------------- a check at each door the mutants found


def _measure_field_gradient(points):
    field = FunctionalMeasureField(lambda t, x, mu: 0.0 * x[0], gradient_fn=lambda t, x, mu: _misread(x))
    return field.spatial_gradient(0.0, points, None)


def _hamiltonians_potential(points):
    return _hamiltonian(potential=lambda x: _misread(x))(
        t=0.0, x=(points[:, 0],), p=np.zeros((len(points), 1)), m=np.ones(len(points))
    )


def _implicit_heat_forcing(points):
    grid = _grid(n=21)
    heat = ImplicitHeatSolver(grid, alpha=0.1, bc=grid.boundary_conditions, forcing=lambda t, x: _misread(x))
    return heat.solve_step(np.ones(21), dt=0.01)


def _volatility_in_a_solve(points):
    problem = MFGProblem(
        model=Model(hamiltonian=_hamiltonian(), volatility=lambda t, x, m: 0.2 + 0.1 * _misread(x)),
        domain=_grid(),
        conditions=Conditions(u_terminal=lambda x: 0.0, m_initial=lambda x: 1.0, T=0.2),
        Nt=4,
    )
    return problem.solve(max_iterations=2, tolerance=1e-3, verbose=False)


def _state_penalty(points):
    return _problem(state_penalty=lambda x: _misread(x))


@pytest.mark.parametrize(
    "door",
    [_measure_field_gradient, _hamiltonians_potential, _implicit_heat_forcing, _volatility_in_a_solve, _state_penalty],
    ids=[
        "measure field gradient",
        "a Hamiltonian's potential off a problem",
        "ImplicitHeat forcing",
        "volatility",
        "state_penalty",
    ],
)
def test_each_door_that_compares_at_its_first_batch_refuses_a_misread(door):
    """Each of these doors' comparisons could be switched off with every test passing (review 1's survivors D03, D07, D11,
    D12, D13); the measure field, the manufactured pair and the drift are pinned with the doors that keep a body."""
    with pytest.raises(ValueError, match="reads differently") as refused:
        door(_BATCH)
    if door is _state_penalty:
        # Composed into the potential, it is refused inside the Hamiltonian's check, whose message wraps the
        # state_penalty's own refusal: the inner one must name it, or the penalty's comparison was not the one that ran.
        assert str(refused.value).count("state_penalty '") == 1, refused.value


def test_a_difference_far_below_a_misread_is_still_refused():
    """The comparison's tolerance is round-off: a relative difference of 1e-5 is refused (review 1's survivor K17)."""
    with pytest.raises(ValueError, match="reads differently"):
        _read(lambda x: x[0] * (1.0 + 1e-5 * (np.ndim(x[0]) > 0)), x=(np.linspace(1.0, 2.0, 7),))


def test_a_solvers_coordinates_on_a_square_grid_are_its_axes_in_order():
    """On a square grid a transposed ``x`` has the right shape and reads the other axis: FP-FDM's volatility read through
    ``x`` must match the same field written as an array (review 1's survivor X05)."""
    grid = _grid(dimension=2, n=9)
    X, Y = grid.meshgrid()
    density = np.exp(-20.0 * ((X - 0.3) ** 2 + (Y - 0.6) ** 2))
    density /= density.sum()
    problem = _problem(dimension=2, n=9)

    def solve(volatility):
        return FPFDMSolver(problem).solve_fp_system(
            density, volatility=volatility, volatility_kind="field", show_progress=False
        )

    np.testing.assert_allclose(solve(lambda t, x, m: 0.05 + 0.6 * x[0]), solve(0.05 + 0.6 * X), rtol=0, atol=1e-12)


@pytest.mark.parametrize("dimension", [1, 2])
def test_the_check_reads_a_hamiltonian_at_a_node_alone_as_the_per_point_paths_do(dimension):
    """The per-point solve paths hand H one point: ``x`` a tuple of scalars, ``p`` of shape ``(d,)``, ``m`` a scalar,
    measured on FDM per point, semi-Lagrangian and WENO at d = 1 and 2. The check read a node alone as a batch of one
    instead, a form no solve uses."""
    seen = []

    class Recording(HamiltonianBase):
        def __call__(self, t, x, p, m):
            seen.append((tuple(np.shape(entry) for entry in x), np.shape(p), np.shape(m)))
            return 0.5 * np.sum(np.asarray(p) ** 2, axis=-1) + 0.3 * np.asarray(m) + 0.5 * np.asarray(x[0])

    refuse_a_hamiltonian_whose_reads_disagree(Recording(), _grid(dimension, n=5).get_spatial_grid())
    alone = {read for read in seen if len(read[1]) < 2}
    assert alone == {(((),) * dimension, (dimension,), ())}, seen
    assert all(read[1][0] > 1 for read in seen if len(read[1]) == 2), seen


def test_a_batch_of_exactly_d_points_is_cycled_with_its_field():
    """A batch of exactly d points is cycled to more, since a coordinate axis passes for it; a field door's field is cycled
    with x, or a right body is refused for the broadcast (review 1's survivor K21)."""
    read = cp.bind_user_callable(lambda t, x, m: x[0] + m, cp.COEFFICIENT_SLOTS, role="drift")
    np.testing.assert_allclose(
        read(t=0.0, x=(np.array([0.1, 0.2]), np.array([0.3, 0.4])), m=np.array([0.5, 0.6])),
        [0.6, 0.8],
        rtol=0,
        atol=1e-15,
    )
