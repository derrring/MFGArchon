"""The one owner of a boundary value (#2512 row B2): `types.segment_value` and `types.boundary_value`.

A callable is ``g(t, x)``, time first (#2378 phase 5, #2375 ruling 8; user ruling for boundary values). It is bound
by its parameter names when the segment is built, and a callable whose parameters are not named after the slots,
or that puts ``x`` before ``t``, is refused there -- never swapped. ``x`` is one point of shape ``(d,)``. A reader
with no point refuses a callable that declares ``x``, at first use, naming itself and the face.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions
from mfgarchon.geometry.boundary.types import segment_value
from mfgarchon.types.callable_protocols import POTENTIAL_SLOTS, SOLVER_SOURCE_SLOTS, bind_user_callable

_OUT_OF_ORDER = r"out of order\. Since #2378 phase 5 \(#2375 ruling 8\) a boundary value takes \(t, x\), time first"
_NOT_ALL_NAMED = r"whose parameters are not all named after the slots: a boundary value names them t, x, time first"
_REFUSED = {
    "unnamed one-arg": (lambda s: s, r"cannot be matched to \(t, x\)\. Name its parameters t, x, time first"),
    "unnamed two-arg": (lambda a, b: a, r"none of its parameters is named after a slot"),
    "(x, t)": (lambda x, t: t, _OUT_OF_ORDER),
    "(point, time)": (lambda point, time: time, _OUT_OF_ORDER),
    "(p, t)": (lambda p, t: t, _OUT_OF_ORDER),
    "varargs": (lambda *args, **kw: 0.0, r"cannot be matched to \(t, x\)"),
    # Time first but a parameter named otherwise: the shared binder would bind these positionally, which the
    # ruling for boundary values does not allow.
    "(t, pos)": (lambda t, pos: t, _NOT_ALL_NAMED),
    "(time, pos)": (lambda time, pos: time, _NOT_ALL_NAMED),
    "(s, x)": (lambda s, x: s, _NOT_ALL_NAMED),
}
_BOUND = {
    "(t, x)": (lambda t, x: t + x[0], 1.4),
    "(x)": (lambda x: x[0], 0.9),
    "(t)": (lambda t: t, 0.5),
    "(time)": (lambda time: time, 0.5),
}


def _segment(value, boundary="x_max") -> BCSegment:
    return BCSegment(name="d", bc_type=BCType.DIRICHLET, value=value, boundary=boundary)


@pytest.mark.parametrize("form", sorted(_REFUSED))
def test_a_callable_not_named_time_first_is_refused_when_the_segment_is_built(form):
    fn, message = _REFUSED[form]
    with pytest.raises(TypeError, match=message):
        _segment(fn)


@pytest.mark.parametrize("form", sorted(_BOUND))
def test_a_callable_named_after_its_slots_is_called_by_keyword_with_what_it_declares(form):
    """At t = 0.5 and the point x = [0.9]: each form receives what it names, and nothing else."""
    fn, expected = _BOUND[form]
    assert segment_value(_segment(fn), t=0.5, x=np.array([0.9]), where="test") == pytest.approx(expected, abs=1e-15)


def test_the_unnamed_refusal_offers_no_positional_binding_to_a_reordered_role():
    """The binder's advice for an unnamed callable used to end "or take exactly 2 positionally in that order", which
    under a reordered role leads into the next refusal. Pinned on a second reordered role, the potential, and with a
    control: a role ruling 8 did not reorder still binds positionally, so the advice stays there, and holds."""
    with pytest.raises(TypeError) as excinfo:
        bind_user_callable(lambda s: s, POTENTIAL_SLOTS, role="potential")
    assert "positionally" not in str(excinfo.value)
    assert "time first (#2378 phase 5, #2375 ruling 8)" in str(excinfo.value)
    with pytest.raises(TypeError, match="or take exactly 2 positionally in that order"):
        bind_user_callable(lambda s: s, SOLVER_SOURCE_SLOTS, role="source term")
    assert bind_user_callable(lambda a, b: a, SOLVER_SOURCE_SLOTS, role="source term")(t=0.5, x=0.9) == 0.5


def test_a_value_reassigned_after_construction_is_bound_afresh():
    """The binding is keyed on the value's identity: a stale one would read the new callable through the old
    signature, here as g(t) = 0.5 instead of 2 x[0] = 1.8."""
    segment = _segment(lambda t: t)
    segment.value = lambda x: 2.0 * x[0]
    assert segment_value(segment, t=0.5, x=np.array([0.9]), where="test") == pytest.approx(1.8)
    segment.value = lambda x, t: t
    with pytest.raises(TypeError, match=_OUT_OF_ORDER):
        segment_value(segment, t=0.5, x=np.array([0.9]), where="test")


def test_the_accessor_passes_its_point_as_x_of_shape_d():
    """`BoundaryConditions.get_bc_value_at_boundary` called the value as (point, time); it now reads g(t, x)."""
    bc = BoundaryConditions(dimension=1, segments=[_segment(lambda t, x: t + 10.0 * x[0])], default_bc=BCType.NO_FLUX)
    assert bc.get_bc_value_at_boundary("x_max", time=0.5, point=np.array([0.9])) == pytest.approx(9.5)


# `BCSegment.get_value(point=[0.9], t=0.5)`, HJB-GFDM's Dirichlet read: the callable, the value b10b9e65 read, and
# the value now (None: refused when the segment is built). b10b9e65 tried four positional signatures.
_GET_VALUE = {
    "(x, t)": (lambda x, t: t + 10.0 * x[0], 9.5, None),  # the one form it read right
    "(t, x)": (lambda t, x: t + 10.0 * x[0], 0.0, 9.5),  # every try failed: 0.0, with a warning
    "(t, x) unindexed": (lambda t, x: t + 10.0 * x, 5.9, 9.5),  # called (point, t): swapped
    "(x)": (lambda x: 10.0 * x[0], 9.0, 9.0),
    "(t)": (lambda t: t, 0.9, 0.5),  # the position read as time
}


@pytest.mark.parametrize("form", sorted(_GET_VALUE))
def test_get_value_passes_its_point_as_x_and_its_time_as_t(form):
    fn, before, expected = _GET_VALUE[form]
    if expected is None:
        with pytest.raises(TypeError, match=_OUT_OF_ORDER):
            _segment(fn)
        return
    got = _segment(fn).get_value(np.array([0.9]), t=0.5)
    assert got == pytest.approx(expected, abs=1e-15), f"{form}: read {got}; b10b9e65 read {before}"


def test_a_callable_returning_more_than_one_number_at_a_point_is_refused():
    with pytest.raises(NotImplementedError, match=r"returned 2 numbers, shape \(2,\), at one point"):
        _segment(lambda t, x: np.array([t, t])).get_value(np.array([0.9]), t=0.5)


def _fdm_ghost_wall(bc):
    from mfgarchon.geometry.boundary.applicator_fdm import pad_array_with_ghosts

    padded = pad_array_with_ghosts(np.ones(5), bc, time=0.5, spacing=(0.25,))
    return (padded[-1] + padded[-2]) / 2  # the value the ghost imposes at the face


def _fdm_wall_row(bc):
    from mfgarchon.geometry.boundary.applicator_fdm import dirichlet_wall_rows

    mask, values = dirichlet_wall_rows(bc, (5,), time=0.5)
    assert mask[-1]
    return values[-1]


def _hjb_1d(bc):
    from mfgarchon.alg.numerical.hjb_solvers.base_hjb import _get_bc_info_1d

    return _get_bc_info_1d(bc, "right", time=0.5)[1]


@pytest.mark.parametrize("reader", [_fdm_ghost_wall, _fdm_wall_row, _hjb_1d], ids=["ghost", "wall_row", "hjb_1d"])
def test_each_fdm_reader_receives_the_owners_value_at_its_time(reader):
    """Row B2's families whose numbers B2a does not change (#2512 comment 6104689754, section 5): FDM reads one value
    per face, at the time it is given. ``0.25 + t`` at t = 0.5 is 0.75."""
    bc = BoundaryConditions(
        dimension=1,
        segments=[_segment(lambda t: 0.25 + t), BCSegment(name="w", bc_type=BCType.NO_FLUX, boundary="x_min")],
    )
    assert reader(bc) == pytest.approx(0.75, abs=1e-12)


def test_nitsche_receives_the_owners_value_at_each_point():
    """Meshless Galerkin's Nitsche assembly reads g(x) per boundary point, the whole point: ``x[0] + 10 x[1]``."""
    from mfgarchon.alg.numerical.meshless_galerkin.nitsche import _evaluate_g

    g = _evaluate_g(_segment(lambda x: x[0] + 10.0 * x[1]), np.array([[1.0, 0.0], [1.0, 0.5]]))
    np.testing.assert_allclose(g, [1.0, 6.0], rtol=0, atol=1e-12)


def test_fem_passes_each_dof_as_x_of_shape_d():
    """HJB-FEM's Dirichlet DOFs read g(x) per DOF: at the 1-D exit x = 1, ``0.5 + x[0]`` holds 1.5."""
    from mfgarchon import Conditions, MFGProblem, Model
    from mfgarchon.alg.numerical.fem.hjb_fem_solver import HJBFEMSolver
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
    from mfgarchon.geometry.meshes.mesh_1d import Mesh1D

    mesh = Mesh1D(bounds=(0.0, 1.0), num_elements=8)
    mesh.generate_mesh()
    mesh.boundary_conditions = BoundaryConditions(
        dimension=1,
        segments=[_segment(lambda x: 0.5 + x[0]), BCSegment(name="w", bc_type=BCType.NO_FLUX, boundary="x_min")],
    )
    model = Model(hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)), volatility=0.4)
    conditions = Conditions(m_initial=lambda x: 8 / 9 + 0.0 * x[..., 0], u_terminal=lambda x: 0.0, T=0.5)
    solver = HJBFEMSolver(MFGProblem(model=model, domain=mesh, conditions=conditions, Nt=4))
    x = solver._disc.dof_coordinates[:, 0]
    U = np.asarray(solver.solve_hjb_system(M_density=np.ones((5, x.size)), U_terminal=np.zeros(x.size)))
    np.testing.assert_allclose(U[:-1, np.isclose(x, 1.0)], 1.5, rtol=1e-12)


def test_an_fdm_ghost_refuses_a_value_that_declares_x_at_first_use_naming_itself_and_the_face():
    """FDM reads one value per face and has no point to pass: the segment is built, then refused when read."""
    from mfgarchon.geometry.boundary.applicator_fdm import pad_array_with_ghosts

    bc = BoundaryConditions(
        dimension=1,
        segments=[_segment(lambda t, x: x[0]), BCSegment(name="w", bc_type=BCType.NO_FLUX, boundary="x_min")],
    )
    with pytest.raises(NotImplementedError, match=r"declares x, but the FDM .* face x_max"):
        pad_array_with_ghosts(np.ones(5), bc, time=0.5, spacing=(0.25,))
