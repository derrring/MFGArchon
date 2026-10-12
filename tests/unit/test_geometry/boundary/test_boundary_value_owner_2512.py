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


def test_an_accessor_given_no_time_or_no_point_refuses_a_value_that_declares_it():
    """`get_bc_value_at_boundary` and `BCSegment.get_value` read a time-dependent value at t = 0 when given no time,
    which is how FP-FDM read every step at the initial time (#2535). With none, they now refuse it, and the advice
    runs: the same call with a time reads g at it. A constant, or g(x), needs no time. Given no point, the accessor
    refuses g(x) and reads it once given one (#2512 row B2)."""
    bc = BoundaryConditions(dimension=1, segments=[_segment(lambda t: 0.25 + t)], default_bc=BCType.NO_FLUX)
    with pytest.raises(
        NotImplementedError,
        match=r"declares t, but BoundaryConditions.get_bc_value_at_boundary.* given no time; pass time= to read it",
    ):
        bc.get_bc_value_at_boundary("x_max")
    assert bc.get_bc_value_at_boundary("x_max", time=0.5) == pytest.approx(0.75)
    segment = bc.segments[0]
    with pytest.raises(
        NotImplementedError,
        match=r"declares t, but BCSegment.get_value was given no time; pass t= to read it at a time",
    ):
        segment.get_value(np.array([1.0]))
    assert segment.get_value(np.array([1.0]), t=0.5) == pytest.approx(0.75)
    for value, expected in ((0.3, 0.3), (lambda x: 2.0 * x[0], 2.0)):
        no_time = BoundaryConditions(dimension=1, segments=[_segment(value)], default_bc=BCType.NO_FLUX)
        assert no_time.get_bc_value_at_boundary("x_max", point=np.array([1.0])) == pytest.approx(expected)
    with pytest.raises(NotImplementedError, match=r"declares x, but BoundaryConditions.get_bc_value_at_boundary"):
        no_time.get_bc_value_at_boundary("x_max", time=0.5)


def _timed_readers():
    """The reachable sites that name their reader's time keyword, one case per branch, each a callable taking the
    reader's keywords. A call with a time of 0.5 returns values holding g(0.5) = 0.75, g(t) = 0.25 + t, or None for the
    ghost fill, which returns nothing to read. Four more sites (D1's segment branch, `BCSegment.get_value`, FEM's and
    Nitsche's assemblies) are pinned by the accessor, FEM and Nitsche tests. The implicit applicator's, the n-D ghost
    values' and the FDM ghost check's became reachable when their readers' time defaulted to None (2b-iv)."""
    import warnings
    from types import SimpleNamespace

    from mfgarchon.alg.base_solver import BaseNeuralSolver
    from mfgarchon.alg.numerical.hjb_solvers.base_hjb import _get_bc_info_1d
    from mfgarchon.geometry import TensorProductGrid
    from mfgarchon.geometry.boundary import no_flux_bc
    from mfgarchon.geometry.boundary._compat import get_ghost_values_nd
    from mfgarchon.geometry.boundary.applicator_fdm import FDMApplicator, PreallocatedGhostBuffer, dirichlet_wall_rows
    from mfgarchon.geometry.boundary.applicator_implicit import ImplicitApplicator
    from mfgarchon.geometry.boundary.applicator_meshfree import MeshfreeApplicator
    from mfgarchon.geometry.boundary.bc_utils import _ghost_effect
    from mfgarchon.geometry.implicit.hypersphere import Hypersphere

    def g(t):
        return 0.25 + t

    def face(value, name, kind, boundary, **kw):
        return BCSegment(name=name, bc_type=kind, value=value, boundary=boundary, **kw)

    uniform = BoundaryConditions(dimension=1, segments=[_segment(g, boundary=None)])
    mixed = BoundaryConditions(
        dimension=1,
        segments=[face(g, "L", BCType.NEUMANN, "x_min"), face(0.0, "R", BCType.NEUMANN, "x_max")],
    )
    on_x_min = BoundaryConditions(
        dimension=1, segments=[face(g, "L", BCType.DIRICHLET, "x_min")], default_bc=BCType.NO_FLUX
    )
    default_g = BoundaryConditions(
        dimension=1,
        segments=[face(0.0, "R", BCType.DIRICHLET, "x_max")],
        default_bc=BCType.DIRICHLET,
        default_value=g,
    )
    dirichlet_default = BoundaryConditions(dimension=1, segments=[], default_bc=BCType.DIRICHLET, default_value=g)
    robin_default = BoundaryConditions(
        dimension=1,
        segments=[face(0.0, "r", BCType.ROBIN, "x_max", alpha=1.0, beta=0.0)],
        default_bc=BCType.ROBIN,
        default_value=g,
    )
    bounds = np.array([[0.0, 1.0]])
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[5], boundary_conditions=no_flux_bc(dimension=1))
    points = np.linspace(0.0, 1.0, 5).reshape(-1, 1)

    sphere = Hypersphere(center=np.array([0.5, 0.5]), radius=0.4)
    theta = np.linspace(0.0, 2.0 * np.pi, 13)[:-1]
    on_sphere = np.column_stack([0.5 + 0.4 * np.cos(theta), 0.5 + 0.4 * np.sin(theta)])
    implicit_uniform = BoundaryConditions(
        dimension=2, segments=[face(g, "d", BCType.DIRICHLET, None)], default_bc=BCType.DIRICHLET
    )
    implicit_default = BoundaryConditions(
        dimension=2,
        segments=[face(0.0, "L", BCType.DIRICHLET, "x_min"), face(0.0, "R", BCType.DIRICHLET, "x_max")],
        default_bc=BCType.DIRICHLET,
        default_value=g,
    )
    held = np.full(5, 0.75)  # g(0.5): the field a Dirichlet ghost reproduces, so the ghost holds g(0.5) either way

    def implicit(bc, **kw):
        return ImplicitApplicator(geometry=sphere).apply(np.zeros(len(on_sphere)), bc, on_sphere, **kw)

    def ghost_values(bc, **kw):
        with warnings.catch_warnings():  # deprecated, and still reachable with no time
            warnings.simplefilter("ignore", DeprecationWarning)
            ghosts = get_ghost_values_nd(held, bc, (0.25,), **kw)
        return np.concatenate([np.ravel(v) for v in ghosts.values()])

    def neural(**kw):
        solver = SimpleNamespace(get_boundary_conditions=lambda: SimpleNamespace(value=g))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return BaseNeuralSolver.get_boundary_target_values(solver, np.array([[0.0], [1.0]]), **kw)

    return {
        "update_ghosts": lambda **kw: PreallocatedGhostBuffer((5,), uniform, domain_bounds=bounds).update_ghosts(**kw),
        "update_ghosts, mixed": lambda **kw: PreallocatedGhostBuffer((5,), mixed, domain_bounds=bounds).update_ghosts(
            **kw
        ),
        "enforce_values": lambda **kw: FDMApplicator(dimension=1).enforce_values(np.ones(5), uniform, (0.25,), **kw),
        "dirichlet_wall_rows": lambda **kw: dirichlet_wall_rows(uniform, (5,), **kw)[1],
        "_get_bc_info_1d, a face segment": lambda **kw: _get_bc_info_1d(on_x_min, "left", **kw)[1],
        "get_bc_value_at_boundary, the default value": lambda **kw: default_g.get_bc_value_at_boundary("x_min", **kw),
        "MeshfreeApplicator.apply, Dirichlet": lambda **kw: MeshfreeApplicator(grid).apply(
            np.zeros(5), dirichlet_default, points, **kw
        ),
        "MeshfreeApplicator.apply, Robin": lambda **kw: MeshfreeApplicator(grid).apply(
            np.zeros(5), robin_default, points, **kw
        ),
        "get_boundary_target_values": neural,
        "ImplicitApplicator.apply, a uniform segment": lambda **kw: implicit(implicit_uniform, **kw),
        "ImplicitApplicator.apply, the default value": lambda **kw: implicit(implicit_default, **kw),
        "get_ghost_values_nd, uniform": lambda **kw: ghost_values(uniform, **kw),
        "get_ghost_values_nd, a face segment": lambda **kw: ghost_values(on_x_min, **kw),
        "get_ghost_values_nd, the default value": lambda **kw: ghost_values(default_g, **kw),
        # `refuse_unplaceable_segments(time=None)` hands its time to the check as is.
        "the FDM ghost check, Dirichlet": lambda time=None: _ghost_effect(face(g, "s", BCType.DIRICHLET, None), time)[
            1
        ],
        "the FDM ghost check, Neumann": lambda time=None: _ghost_effect(face(g, "s", BCType.NEUMANN, None), time)[1],
        "the FDM ghost check, Robin": lambda time=None: _ghost_effect(
            face(g, "s", BCType.ROBIN, None, alpha=1.0, beta=1.0), time
        )[3],
    }


@pytest.mark.parametrize(
    "reader",
    [
        "update_ghosts",
        "update_ghosts, mixed",
        "enforce_values",
        "dirichlet_wall_rows",
        "_get_bc_info_1d, a face segment",
        "get_bc_value_at_boundary, the default value",
        "MeshfreeApplicator.apply, Dirichlet",
        "MeshfreeApplicator.apply, Robin",
        "get_boundary_target_values",
        "ImplicitApplicator.apply, a uniform segment",
        "ImplicitApplicator.apply, the default value",
        "get_ghost_values_nd, uniform",
        "get_ghost_values_nd, a face segment",
        "get_ghost_values_nd, the default value",
        "the FDM ghost check, Dirichlet",
        "the FDM ghost check, Neumann",
        "the FDM ghost check, Robin",
    ],
)
def test_a_reader_given_no_time_says_to_pass_its_time(reader):
    """A reader that takes a time refuses a value that declares t when given none, and names its own time parameter,
    which the advice then runs with (#2512 row B2, #2605). Dropping the keyword at any site `_timed_readers` holds turns
    its case red."""
    call = _timed_readers()[reader]
    with pytest.raises(NotImplementedError, match=r"declares t, but .* was given no time; pass time= to read it"):
        call()
    result = call(time=0.5)
    if result is not None:
        assert np.isclose(np.ravel(np.asarray(result, dtype=float)), 0.75).any(), result


@pytest.mark.parametrize("entry", ["laplacian_with_bc", "LaplacianOperator"])
def test_an_operator_given_no_time_refuses_and_its_advice_runs(entry):
    """`laplacian_with_bc`, and an operator that stores its time, read a value that declares t at the time they are
    given. With none, the default since 2b-iv, they refuse at the first read and say to pass `time=`. With `time=0.5`
    they give what g(0.5) = 0.75 as a constant gives. At 375122e5 both read g(0) with no time, in silence."""
    from mfgarchon.geometry.boundary import dirichlet_bc
    from mfgarchon.operators.differential.laplacian import LaplacianOperator
    from mfgarchon.operators.stencils.finite_difference import laplacian_with_bc

    u = np.linspace(0.0, 1.0, 9) ** 2
    timed, held = dirichlet_bc(dimension=1, value=lambda t: 0.25 + t), dirichlet_bc(dimension=1, value=0.75)
    if entry == "laplacian_with_bc":
        apply = lambda bc, **kw: laplacian_with_bc(u, [0.125], bc, **kw)  # noqa: E731
    else:
        apply = lambda bc, **kw: LaplacianOperator([0.125], (9,), bc=bc, **kw)(u)  # noqa: E731
    with pytest.raises(NotImplementedError, match=r"declares t, but .* was given no time; pass time= to read it"):
        apply(timed)
    np.testing.assert_array_equal(apply(timed, time=0.5), apply(held))
    assert not np.array_equal(apply(timed, time=0.5), apply(timed, time=0.0)), "g(t) does not reach the result"


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


def _fdm_bc(value) -> BoundaryConditions:
    return BoundaryConditions(
        dimension=1, segments=[_segment(value), BCSegment(name="w", bc_type=BCType.NO_FLUX, boundary="x_min")]
    )


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
    assert reader(_fdm_bc(lambda t: 0.25 + t)) == pytest.approx(0.75, abs=1e-12)


@pytest.mark.parametrize(
    ("returned", "message"),
    [
        (float("nan"), r"\(what its callable returned\) has a non-finite value nan"),
        (True, r"\(what its callable returned\) has a boolean value"),
        (np.complex128(1j), r"\(what its callable returned\) has a complex value"),
        (None, r"its value returned None at one point, .* does the function lack a `return`"),
        (
            np.array([True], dtype=object),
            r"\(what its callable returned\) has a value of type ndarray with dtype object",
        ),
        (
            np.array([None], dtype=object),
            r"\(what its callable returned\) has a value of type ndarray with dtype object",
        ),
    ],
    ids=["nan", "True", "imaginary", "None", "object-True", "object-None"],
)
def test_what_a_callable_returns_is_read_as_a_constant_is(returned, message):
    """One reading of a value, whether given or returned. At this FDM wall b10b9e65 imposed NaN, 1.0 and 0.0 (with a
    ComplexWarning) for the first three, and None raised a bare TypeError; ``None`` is refused rather than read as 0,
    since a function with no ``return`` gives it. A one-element array is unwrapped before any rule reads it (review 1
    of #2598), so an object array holding True or None is read as True or None."""
    with pytest.raises(NotImplementedError, match=message):
        _fdm_ghost_wall(_fdm_bc(lambda t: returned))


def test_an_object_array_is_refused_at_the_container():
    """Review 1 of #2598: the boolean test ran before the unwrap, so 7d382a7f read object [True] as 1.0 at every reader.
    An object array is now refused whatever it holds (audit session ruling on review 1 of #2601)."""
    with pytest.raises(NotImplementedError, match="has a value of type ndarray with dtype object"):
        _fdm_ghost_wall(_fdm_bc(np.array([True], dtype=object)))


@pytest.mark.parametrize(
    ("value", "expected"),
    [(np.ma.masked_all(1), 0.0), (np.ma.array([0.7]), 0.7)],
    ids=["fully-masked", "unmasked"],
)
def test_a_one_element_masked_array_is_its_element(value, expected):
    """Ruling (a) holds for a masked array too: its one element is read, and a masked element is zero."""
    assert _fdm_ghost_wall(_fdm_bc(value)) == pytest.approx(expected, abs=1e-12)


def test_a_default_does_not_make_x_optional_where_a_reader_has_no_point():
    """A default for x reads one fixed position, not the face: at the right wall x = 1, ``x=0`` would be a silent
    misread, so FDM refuses it as it refuses any x (audit session ruling, 2026-10-11)."""
    with pytest.raises(NotImplementedError, match=r"declares x, with a default, but the FDM .* write it as g\(t\)"):
        _fdm_ghost_wall(_fdm_bc(lambda t, x=0.0: t))


def _cubic_spline():
    from scipy.interpolate import CubicSpline

    return CubicSpline([0.0, 0.5, 1.0], [0.0, 1.0, 4.0])


def _interp1d():
    from scipy.interpolate import interp1d

    return interp1d([0.0, 0.5, 1.0], [0.0, 1.0, 4.0])


class _CallableInstance:
    def __call__(self, t, x):
        return t


@pytest.mark.parametrize(
    "make",
    [_cubic_spline, _interp1d, _CallableInstance, lambda: __import__("math").sin, lambda: np.sin],
    ids=["CubicSpline", "interp1d", "instance", "builtin", "ufunc"],
)
def test_a_callable_that_is_not_a_function_you_write_is_refused_when_the_segment_is_built(make):
    """User ruling, 2026-10-11 (rule C): a boundary value is a def or a lambda, or a partial of one. Another callable's
    parameter names were not written against the slots: a SciPy interpolant takes ``(x, ...)``, so one built over time
    data would be read as g(x), time data at the position. b10b9e65 called FDM's value as ``value(time)`` and read it."""
    with pytest.raises(
        TypeError,
        match=r"not a plain def or lambda: a boundary value is a function whose parameter names declare what it reads",
    ):
        _segment(make())


def test_the_advised_wrap_of_an_interpolant_over_time_reads_it_at_the_time():
    """The wrap the refusal advises: ``lambda t: s(t)`` reads the spline through (0, 0), (0.5, 1), (1, 4) at t = 0.5."""
    spline = _cubic_spline()
    assert _fdm_ghost_wall(_fdm_bc(lambda t: spline(t))) == pytest.approx(1.0, abs=1e-12)


def test_a_partial_of_a_function_is_a_function_you_write():
    import functools

    assert _fdm_ghost_wall(_fdm_bc(functools.partial(lambda a, t: a + t, a=0.25))) == pytest.approx(0.75, abs=1e-12)


def test_a_positional_only_function_is_refused_and_its_advised_wrap_reads():
    """``def f(t, /)`` is a function but cannot be called by name; the refusal advises a wrap, and the wrap of a builtin
    reads sin(0.5) at the FDM wall."""
    import math

    with pytest.raises(TypeError, match=r"takes \(t, /\), with positional-only parameters.*`lambda t: f\(t\)`"):
        _segment(eval("lambda t, /: t"))
    assert _fdm_ghost_wall(_fdm_bc(lambda t: math.sin(t))) == pytest.approx(math.sin(0.5), abs=1e-12)


def test_a_reader_with_neither_time_nor_point_advises_a_constant_and_reads_one():
    """Review 2 of #2598 (M1): the resolver's passthrough has neither t nor x, so advising g(x) for a g(t), or g(t) for
    a g(x), led into a second refusal. Its advice is a constant, and a constant reads there."""
    from mfgarchon.geometry.boundary.resolution import _resolve_passthrough

    with pytest.raises(NotImplementedError, match=r"declares t, but the BC resolver's .* write it as a constant"):
        _resolve_passthrough(_segment(lambda t: t))
    with pytest.raises(NotImplementedError, match=r"declares x, but the BC resolver's .* write it as a constant"):
        _resolve_passthrough(_segment(lambda x: x[0]))
    assert _resolve_passthrough(_segment(0.75)).value == pytest.approx(0.75, abs=1e-15)


def test_fdm_advises_g_of_t_and_reads_it():
    """FDM has a time and no point: it advises g(t), and g(t) reads at its wall."""
    with pytest.raises(NotImplementedError, match=r"declares x, but the FDM .* write it as g\(t\)"):
        _fdm_ghost_wall(_fdm_bc(lambda x: x[0]))
    assert _fdm_ghost_wall(_fdm_bc(lambda t: 0.25 + t)) == pytest.approx(0.75, abs=1e-12)


def test_fem_dirichlet_reads_its_level_time_and_says_to_pass_one_without_it():
    """FEM's Dirichlet assembly reads a value at the DOF x = 1 and at the time of the level it solves for. Given no
    time, it refuses a value that declares t and says to pass `t=`. The advice runs: `get_dirichlet_dofs_and_values`
    with `t=0.5` reads g(0.5) (#2512 row B2)."""
    import skfem

    from mfgarchon.alg.numerical.fem.bc_adapter import _evaluate_segment_values, get_dirichlet_dofs_and_values

    basis = skfem.Basis(skfem.MeshLine(np.linspace(0.0, 1.0, 5)), skfem.ElementLineP1())
    dof = int(np.argmax(basis.doflocs[0]))
    with pytest.raises(
        NotImplementedError, match=r"declares t, but the FEM Dirichlet assembly .* given no time; pass t= to read it"
    ):
        _evaluate_segment_values(_segment(lambda t: t), basis, [dof], None)
    bc = BoundaryConditions(dimension=1, segments=[_segment(lambda t: 0.25 + t, boundary=None)])
    with pytest.raises(NotImplementedError, match=r"pass t= to read it at a time"):
        get_dirichlet_dofs_and_values(basis, bc)
    _, values = get_dirichlet_dofs_and_values(basis, bc, t=0.5)
    np.testing.assert_allclose(values, 0.75, rtol=0, atol=1e-12)
    assert _evaluate_segment_values(_segment(lambda x: 0.5 + x[0]), basis, [dof], None) == pytest.approx(
        [1.5], abs=1e-12
    )
    assert _evaluate_segment_values(_segment(lambda t, x: t + x[0]), basis, [dof], 0.5) == pytest.approx(
        [1.5], abs=1e-12
    )


def test_nitsche_receives_the_owners_value_at_each_point():
    """Meshless Galerkin's Nitsche assembly reads g per boundary point, the whole point, ``x[0] + 10 x[1]``, and at
    the time it is given. Given none, it refuses a value that declares t and says to pass `t=`."""
    from mfgarchon.alg.numerical.meshless_galerkin.nitsche import _evaluate_g

    points = np.array([[1.0, 0.0], [1.0, 0.5]])
    g = _evaluate_g(_segment(lambda x: x[0] + 10.0 * x[1]), points, None)
    np.testing.assert_allclose(g, [1.0, 6.0], rtol=0, atol=1e-12)
    with pytest.raises(
        NotImplementedError, match=r"declares t, but the meshless Nitsche assembly .* given no time; pass t= to read it"
    ):
        _evaluate_g(_segment(lambda t, x: t + x[0] + 10.0 * x[1]), points, None)
    g = _evaluate_g(_segment(lambda t, x: t + x[0] + 10.0 * x[1]), points, 0.5)
    np.testing.assert_allclose(g, [1.5, 6.5], rtol=0, atol=1e-12)


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


def _nested(value):
    """A one-element object array holding ``value``, which may itself be one."""
    array = np.empty(1, dtype=object)
    array[0] = value
    return array


_BOOLEAN, _COMPLEX = r"has a boolean value", r"has a complex value"
_NONE = r"returned None at one point, .* does the function lack a `return`"
_OBJECT = r"has a value of type \w+ with dtype object"


def _self_referential():
    array = np.empty(1, dtype=object)
    array[0] = array
    return array


# The layered contract (#2512 row B2's second half), with literal expectations: a number is what reads, a pattern is
# the refusal. Rows are named by the review finding they pin, or by the layer outcome they show. ``returned`` reads the
# value as a callable's return.
_CONTRACT = [
    # Layer 1, dtype: an array must be of integer or floating dtype, refused at the container otherwise.
    ("True", True, False, _BOOLEAN),
    ("1j", 1j, False, _COMPLEX),
    ("review-1 blocker: object [True]", _nested(True), False, _OBJECT),
    ("M3a: fully masked [True]", np.ma.array([True], mask=[True]), False, _BOOLEAN),
    ("M3a: fully masked [1j]", np.ma.array([1j], mask=[True]), False, _COMPLEX),
    ("M3b: doubly nested False", _nested(_nested(False)), False, _OBJECT),
    ("M3b: doubly nested 0j", _nested(_nested(0j)), False, _OBJECT),
    ("M4: doubly nested 0.7, constant", _nested(_nested(0.7)), False, _OBJECT),
    ("M4: doubly nested 0.7, returned", _nested(_nested(0.7)), True, _OBJECT),
    ("object [0.7], read as 0.7 at 07dfd27b", _nested(0.7), False, _OBJECT),
    ("review 1 of #2601, M1: object np.matrix [[0.7]]", np.matrix([[0.7]], dtype=object), False, _OBJECT),
    ("review 1 of #2601, M1: a self-referential object array", _self_referential(), False, _OBJECT),
    ("review 1: object [None], returned", _nested(None), True, _OBJECT),
    ("a fully masked string array", np.ma.array(["0.7"], mask=[True]), False, r"with dtype <U3"),
    ("a timedelta64 array", np.array([1], dtype="timedelta64[s]"), False, r"with dtype timedelta64\[s\]"),
    # Layer 2, one unwrap.
    ("one element of shape (1, 1)", np.array([[0.7]]), False, 0.7),
    ("np.matrix [[0.7]]", np.matrix([[0.7]]), False, 0.7),
    ("masked [0.7], unmasked", np.ma.array([0.7]), False, 0.7),
    ("a return of two numbers", np.array([1.0, 2.0]), True, r"returned 2 numbers, shape \(2,\), at one point"),
    # Layer 3, the value.
    ("None, constant", None, False, 0.0),
    ("None, returned", None, True, _NONE),
    ("fully masked, one element", np.ma.masked_all(1), False, 0.0),
    ("fully masked [nan]", np.ma.array([np.nan], mask=[True]), False, 0.0),
    ("fully masked, three elements", np.ma.masked_all(3), False, 0.0),
    ("all zero, three elements", np.zeros(3), False, 0.0),
    ('"0"', "0", False, 0.0),
    ("float32 0.5", np.float32(0.5), False, 0.5),
    ('"0.7"', "0.7", False, r"has a value of type str"),
    ("nan", float("nan"), False, r"has a non-finite value nan"),
]


@pytest.mark.parametrize(
    ("value", "returned", "expected"), [row[1:] for row in _CONTRACT], ids=[row[0] for row in _CONTRACT]
)
def test_the_layered_contract(value, returned, expected):
    """One function reads a constant and a callable's return (#2512 row B2): the dtype, one unwrap, the value.
    Each review of #2598 found the next container corner of a one-layer reading; these rows pin each of them."""
    from mfgarchon.geometry.boundary.types import boundary_value

    read = (lambda t: value) if returned else value
    if isinstance(expected, str):
        with pytest.raises(NotImplementedError, match=expected):
            boundary_value(read, kind="k", t=0.5, x=None, where="w")
    else:
        assert boundary_value(read, kind="k", t=0.5, x=None, where="w") == pytest.approx(expected, abs=1e-7)


def test_a_provider_returns_one_number_read_as_a_return_is():
    """A provider's result is read as a callable's return (#2512 row B2): one number. A value per node is #2600."""
    from mfgarchon.geometry.boundary.providers import BaseBCValueProvider
    from mfgarchon.geometry.boundary.types import boundary_value

    class _Provider(BaseBCValueProvider):
        def __init__(self, result):
            self.result = result

        def compute(self, state):
            return self.result

    assert boundary_value(_Provider(np.float64(0.3)), kind="k", t=0.5, x=None, where="w", state={}) == 0.3
    for result, refusal in ((np.array([0.1, 0.2]), r"returned 2 numbers.*#2600"), (True, _BOOLEAN), (None, _NONE)):
        with pytest.raises(NotImplementedError, match=refusal):
            boundary_value(_Provider(result), kind="k", t=0.5, x=None, where="w", state={})


def test_a_robin_coefficient_is_read_by_one_reader_and_a_callable_or_a_field_is_refused():
    """`segment_robin_coefficients` (#2512 row B2): a constant each, by the constant contract. A callable is refused
    naming #1237, and a field from a provider naming #2600, the per-node capability no reader has yet."""
    from mfgarchon.geometry.boundary.types import segment_robin_coefficients

    def robin(alpha, beta):
        return BCSegment(name="r", bc_type=BCType.ROBIN, value=0.0, alpha=alpha, beta=beta, boundary="x_max")

    assert segment_robin_coefficients(robin(np.float32(2.0), np.array([0.5])), where="w") == (2.0, 0.5)
    for alpha, refusal in (
        (lambda t: 1.0, r"is a callable; a Robin coefficient is one constant per segment.*\(#1237\)"),
        (np.array([1.0, 2.0]), r"is a field of 2 values along the wall.*#2600"),
        (
            True,
            r"has a boolean value, which is not a boundary datum: give a real number, or a real array of one element: a Robin",
        ),
        (1j, r"has a complex value"),
        (float("nan"), r"has a non-finite value"),
        (None, r"'s alpha is None"),
    ):
        with pytest.raises(NotImplementedError, match=refusal):
            segment_robin_coefficients(robin(alpha, 1.0), where="w")


def test_the_fem_robin_load_refuses_a_callable_naming_1237():
    """FEM's Robin and Neumann loads are one constant per segment (#1237; the user's coverage-not-every-axis
    ruling): a callable g declares what the load cannot pass, and the refusal names #1237."""
    import skfem

    from mfgarchon.alg.numerical.fem.bc_adapter import assemble_robin_terms

    mesh = skfem.MeshLine(np.linspace(0.0, 1.0, 5)).with_boundaries({"x_max": lambda x: np.isclose(x[0], 1.0)})
    basis = skfem.Basis(mesh, skfem.ElementLineP1())
    bc = BoundaryConditions(
        dimension=1,
        segments=[BCSegment(name="r", bc_type=BCType.ROBIN, value=lambda t: t, alpha=1.0, beta=1.0, boundary="x_max")],
        default_bc=BCType.NO_FLUX,
    )
    with pytest.raises(NotImplementedError, match=r"declares t, but the FEM Robin load .*\(#1237\).* a constant"):
        assemble_robin_terms(basis, bc, 0.125, natural_bc="gradient")


@pytest.mark.parametrize(
    "segment",
    [
        BCSegment(name="d", bc_type=BCType.DIRICHLET, value=0.7, alpha=None, boundary="x_max"),
        BCSegment(name="n", bc_type=BCType.NEUMANN, value=0.0, alpha=lambda t: 1.0, boundary="x_max"),
        BCSegment(name="f", bc_type=BCType.NO_FLUX, alpha=True, boundary="x_max"),
    ],
    ids=["dirichlet-alpha-None", "neumann-alpha-callable", "no-flux-alpha-True"],
)
def test_a_segment_that_is_not_robin_has_its_coefficients_left_unread(segment):
    """Review 1 of #2601 (M3): the uniform FDM ghosts, the resolver's passthrough and the 1-D HJB read use alpha and
    beta in their Robin arm alone, so a non-Robin segment's fields are not read, and what 07dfd27b solved still solves:
    the Dirichlet wall at 0.7, the Neumann and no-flux walls at the interior value 1."""
    from mfgarchon.alg.numerical.hjb_solvers.base_hjb import _get_bc_info_1d
    from mfgarchon.geometry.boundary.resolution import _resolve_passthrough

    bc = BoundaryConditions(dimension=1, segments=[segment], default_bc=BCType.NO_FLUX)
    expected = 0.7 if segment.bc_type == BCType.DIRICHLET else 1.0
    assert _fdm_ghost_wall(bc) == pytest.approx(expected, abs=1e-12)
    assert _get_bc_info_1d(bc, "right", time=0.5)[0] == segment.bc_type
    if segment.bc_type != BCType.NO_FLUX:  # NO_FLUX resolves per equation, not by the passthrough
        assert _resolve_passthrough(segment).value == pytest.approx(0.7 if segment.bc_type == BCType.DIRICHLET else 0.0)


def test_the_fdm_uniform_ghosts_leave_a_non_robin_segments_coefficients_unread():
    """Review 1 of #2601 (M3), at the site `pad_array_with_ghosts` does not reach: `PreallocatedGhostBuffer`'s uniform
    update. A Dirichlet segment whose alpha is None gives exactly the ghosts of one whose alpha is the default."""
    from mfgarchon.geometry.boundary.applicator_fdm import PreallocatedGhostBuffer

    def ghosts(segment):
        bc = BoundaryConditions(dimension=1, segments=[segment])
        buf = PreallocatedGhostBuffer(interior_shape=(5,), boundary_conditions=bc, domain_bounds=np.array([[0.0, 1.0]]))
        buf.interior[:] = np.linspace(0.0, 1.0, 5)
        buf.update_ghosts(time=0.5)
        return buf.padded.copy()

    without = ghosts(BCSegment(name="d", bc_type=BCType.DIRICHLET, value=0.7, alpha=None))
    np.testing.assert_array_equal(without, ghosts(BCSegment(name="d", bc_type=BCType.DIRICHLET, value=0.7)))


@pytest.mark.parametrize("resolver", ["HJBResolver", "FPResolver"])
def test_a_provider_valued_robin_segment_is_resolved_with_the_solver_state(resolver):
    """Review 1 of #2601 (C6), a regression #2598 introduced: `resolve_bc` with `HJBResolver` resolved a provider on a
    Robin segment's value with the solver state (b10b9e65: 0.37). #2598's passthrough read it with no state and
    refused it, so the resolver's own provider branch never ran. The passthrough now passes the state to the owner,
    for both resolvers: `FPResolver`, whose passthrough read the value as 0.0 at b10b9e65, too (review 2 of #2601)."""
    from mfgarchon.geometry.boundary import resolution
    from mfgarchon.geometry.boundary.providers import ConstantProvider

    segment = BCSegment(
        name="r", bc_type=BCType.ROBIN, value=ConstantProvider(0.37), alpha=2.0, beta=1.0, boundary="x_max"
    )
    bc = BoundaryConditions(dimension=1, segments=[segment], default_bc=BCType.NO_FLUX)
    (resolved,) = resolution.resolve_bc(bc, getattr(resolution, resolver)(), {"iteration": 0})
    assert (resolved.value, resolved.alpha, resolved.beta) == (0.37, 2.0, 1.0)
