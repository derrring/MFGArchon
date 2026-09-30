"""#2378 phase 4 part 2a: a per-solve volatility override is read by its declared kind.

`volatility_field=`, `volatility_matrix=`, `tensor_diffusion_field=` and `diffusion_field=` are
retired (ruling 11). `volatility=` replaces all four; an array declares `volatility_kind="field"`
or `"tensor"` (ruling 12), and one owner, `resolve_volatility_override`, reads it for every
solver. The census on #2378 (2026-09-28) found six defects in the old, per-solver reading. Pinned
here: the refusal (and defect 5), its removal schedule, the owner's own refusals, and defects 1-3.
Defect 4 -- HJB-FDM and FP-FDM reading one array two ways -- is pinned by the d x d grid tests and
the override-route tests in test_volatility_is_never_collapsed_2376.py.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from types import SimpleNamespace

import pytest

import numpy as np

import mfgarchon
from mfgarchon.utils.pde_coefficients import RETIRED_VOLATILITY_KEYWORDS, resolve_volatility_override

_ENTRY_POINTS = ("solve_fp_system", "solve_hjb_system", "__init__")


def _override_takers() -> dict[str, tuple[object, bool]]:
    """Every solver entry point in `mfgarchon.alg` that could take a volatility override: name -> (function, is_method).

    The population must not be defined by the property audited (the refusal), nor by what the part 2a
    conversion added (`volatility_kind`): a function it missed would then be absent from the census and
    from its own completeness check (#2420 review). Nor by the bare name `volatility`, which part 2b
    gave to helpers that never took an override. So it is keyed on the entry points by name: every
    concrete `solve_fp_system` / `solve_hjb_system`; every hand-written `__init__` in
    `mfgarchon.alg.numerical` (not a dataclass record) that names `volatility` or a retired keyword; and
    every module function `solve_fp_...system` / `solve_hjb_...system`. Found by walking the package.
    """
    import dataclasses
    import re

    found: dict[str, tuple[object, bool]] = {}
    volatility_names = {"volatility", *RETIRED_VOLATILITY_KEYWORDS}
    module_entry = re.compile(r"^solve_(fp|hjb)_\w*system")
    for info in pkgutil.walk_packages(mfgarchon.alg.__path__, "mfgarchon.alg."):
        module = importlib.import_module(info.name)
        for obj in vars(module).values():
            if inspect.isclass(obj) and obj.__module__ == module.__name__:
                members = [
                    (f"{obj.__qualname__}.{k}", v, True, obj) for k, v in vars(obj).items() if k in _ENTRY_POINTS
                ]
            elif inspect.isfunction(obj) and obj.__module__ == module.__name__ and module_entry.match(obj.__name__):
                members = [(obj.__qualname__, obj, False, None)]
            else:
                continue
            for qualname, func, is_method, owner in members:
                if not inspect.isfunction(func) or getattr(func, "__isabstractmethod__", False):
                    continue
                names = set(inspect.signature(func).parameters)
                if is_method and func.__name__ in ("solve_fp_system", "solve_hjb_system"):
                    found[qualname] = (func, is_method)
                elif is_method:  # __init__
                    hand_written = not dataclasses.is_dataclass(owner)
                    if (
                        hand_written
                        and module.__name__.startswith("mfgarchon.alg.numerical")
                        and volatility_names & names
                    ):
                        found[qualname] = (func, is_method)
                elif volatility_names & names:
                    found[qualname] = (func, is_method)
    return found


def test_every_entry_point_taking_the_override_refuses_the_retired_names():
    """Ruling 11: each retired name raises a TypeError that names what replaces it.

    Python's own "unexpected keyword argument" names no replacement, and a signature with
    `**kwargs` -- the weak-form pair, the meshless HJB, FixedPointIterator -- accepted the old name
    and ignored it (#2419). Before part 2a FixedPointIterator took `diffusion_field=` with only a
    warning (census defect 5). The decorator raises before the arguments bind, so each function is
    called with a placeholder for `self`.
    """
    takers = _override_takers()
    # Reach control: known members, one per kind of entry point, and a floor on the population.
    for known in (
        "FPFDMSolver.solve_fp_system",
        "HJBFDMSolver.solve_hjb_system",
        "WeakFormFPSolver.solve_fp_system",
        "FixedPointIterator.__init__",
        "solve_hjb_system_backward",
    ):
        assert known in takers, f"{known} is missing from the population: the walk no longer reaches it"
    assert len(takers) >= 20, f"expected every solver entry point, found {len(takers)}"

    wrong = []
    for qualname, (func, is_method) in sorted(takers.items()):
        for name in RETIRED_VOLATILITY_KEYWORDS:
            args = (None,) if is_method else ()
            try:
                func(*args, **{name: 0.3})
            except TypeError as exc:
                if f"no longer takes {name}=; pass volatility=" in str(exc) and "#2378" in str(exc):
                    continue
                wrong.append(f"{qualname}({name}=): {exc}")
            except Exception as exc:  # any other outcome is the finding
                wrong.append(f"{qualname}({name}=): {type(exc).__name__}: {exc}")
            else:
                wrong.append(f"{qualname}({name}=): accepted")
    assert not wrong, "entry points that do not refuse a retired name by name:\n" + "\n".join(wrong)


def test_the_refusal_is_scheduled_and_the_audit_reads_the_schedule():
    """The refusal goes at v0.25.0, except where a `**kwargs` would then swallow the name (#2419).

    `retired_parameters` records the date and the blocker, and the deprecation audit reads them,
    so a v0.25.0 written into a message alone is not what schedules it -- that was #2270, a removal
    five minors overdue because nothing read its date.
    """
    from mfgarchon.utils.deprecation import audit_all_deprecations

    def retired(report, bucket):
        return {e["name"]: e for e in report[bucket] if e["type"] == "retired_parameters"}

    before = audit_all_deprecations(mfgarchon, current_version="v0.24.0")
    assert "FPFDMSolver.solve_fp_system" in retired(before, "active")
    assert not retired(before, "ready")
    assert not retired(before, "not_ready")

    due = audit_all_deprecations(mfgarchon, current_version="v0.25.0")
    ready = retired(due, "ready")
    assert "FPFDMSolver.solve_fp_system" in ready, "a closed signature is removable at v0.25.0"
    # #2419 closed the weak-form **kwargs, so their refusal is removable on schedule too.
    assert "WeakFormFPSolver.solve_fp_system" in ready
    assert "WeakFormHJBSolver.solve_hjb_system" in ready
    # The blocker is read from the signature, so a function that still takes **kwargs stays blocked.
    # For FixedPointIterator that is a known false positive: its **kwargs now raises on an unknown
    # name (#2419) rather than swallowing it. It is held until that **kwargs goes with the damping_*
    # names it exists to explain, due at the same v0.25.0.
    blocked = retired(due, "not_ready")
    assert blocked["FixedPointIterator.__init__"]["remaining_blockers"] == ["var_keyword"]


_OWNER_PROBLEM = SimpleNamespace(volatility=0.3, volatility_kind=None, dimension=2)


@pytest.mark.parametrize(
    ("volatility", "kind", "message"),
    [
        (np.full((4, 4), 0.3), None, r"needs volatility_kind='field'"),
        (0.3, "field", r"read by nothing"),
        (np.array([0.3, 0.2]), "tensor", r"per-axis vector.*np\.diag\(v\) with volatility_kind='tensor'"),
        (np.full((4, 4), 0.3), "diagonal", r"must be 'field' or 'tensor'"),
        (None, "field", r"without a volatility= override"),
        (-0.1, None, r"non-negative"),
    ],
    ids=["array-without-kind", "kind-on-a-scalar", "per-axis-vector", "unknown-kind", "kind-alone", "negative"],
)
def test_the_owner_refuses_what_it_cannot_read_unambiguously(volatility, kind, message):
    """Rulings 12 and 14. The library does not read the kind from the shape: a (d, d) tensor and a
    field on a d x d grid are the same array, and before part 2a FP-FDM read one such array as a
    tensor or as a field depending on the drift type (census defect 2)."""
    with pytest.raises(ValueError, match=message):
        resolve_volatility_override(volatility, kind, problem=_OWNER_PROBLEM, consumer="probe")


def test_the_owner_returns_the_problems_own_volatility_and_passes_a_declared_override_through():
    field = np.full((4, 4), 0.3)
    problem = SimpleNamespace(volatility=field, volatility_kind="field", dimension=2)
    value, kind = resolve_volatility_override(None, None, problem=problem, consumer="probe")
    assert value is field
    assert kind == "field"

    def sigma(t, x, m):
        return np.eye(2)

    assert resolve_volatility_override(sigma, "tensor", problem=_OWNER_PROBLEM, consumer="probe") == (sigma, "tensor")
    assert resolve_volatility_override(sigma, None, problem=_OWNER_PROBLEM, consumer="probe") == (sigma, None)
    # Ruling 14's refusal is the per-axis READING; a length-d field declared as a field is read as one.
    vector_field = np.array([0.3, 0.2])
    assert resolve_volatility_override(vector_field, "field", problem=_OWNER_PROBLEM, consumer="probe")[1] == "field"


def _problem_2d(n=9, volatility=0.3):
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
    from mfgarchon.core.mfg_problem import MFGProblem
    from mfgarchon.core.model import Conditions, Model
    from mfgarchon.geometry import TensorProductGrid
    from mfgarchon.geometry.boundary import no_flux_bc

    grid = TensorProductGrid(bounds=[(0.0, 1.0)] * 2, Nx_points=[n, n], boundary_conditions=no_flux_bc(dimension=2))
    x = grid.get_spatial_grid()
    bump = np.exp(-10 * np.sum((np.asarray(x, float) - 0.45) ** 2, axis=-1)).reshape(n, n)
    mass = grid.integrate(bump)
    hamiltonian = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0), coupling=lambda m: -m, coupling_dm=lambda m: -1.0
    )
    problem = MFGProblem(
        model=Model(hamiltonian=hamiltonian, volatility=volatility),
        domain=grid,
        conditions=Conditions(
            m_initial=lambda z: (
                np.exp(-10 * np.sum((np.atleast_2d(z) - 0.45) ** 2, axis=-1)).reshape(np.shape(z)[:-1] or ()) / mass
            ),
            u_terminal=lambda z: 0.0,
            T=0.2,
        ),
        Nt=6,
    )
    return problem, bump / mass


def test_fp_fdm_callable_drift_reads_a_tensor_override():
    """Census defects 1 and 2, on the callable-drift route of FP-FDM.

    Defect 1: a tensor-valued callable was ignored there, and the solve ran at the problem's 0.3.
    Defect 2: a constant (2, 2) tensor was read as a spatial field and refused for not matching the
    9 x 9 grid, while the potential route read the same array as a tensor. The oracle: a callable
    returning a constant Sigma is that constant Sigma, so the two solves must agree exactly, and
    both must differ from the problem's own volatility or the override was never read.
    """
    from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver

    problem, m0 = _problem_2d()
    sigma = np.array([[0.4, 0.1], [0.1, 0.3]])

    def drift(t, x, m):
        return np.full((2, *np.shape(m)), -0.05)

    def solve(**override):
        return FPFDMSolver(problem).solve_fp_system(m0, drift_field=drift, **override)

    constant = solve(volatility=sigma, volatility_kind="tensor")
    called = solve(volatility=lambda t, x, m: sigma, volatility_kind="tensor")
    own = solve()

    np.testing.assert_array_equal(called, constant)
    assert np.max(np.abs(constant - own)) > 1e-3, "the tensor override did not reach the callable-drift solve"


def test_fvm_refuses_an_all_equal_tensor_override_and_reads_a_constant_field_as_its_scalar():
    """Census defect 3: FVM read an all-equal (2, 2) override as the scalar it repeats.

    The array passes FVM's constancy check, so only its declared kind can refuse it. A constant
    FIELD is the other half: it is the scalar it repeats, and must solve bit for bit like it.
    """
    from mfgarchon.alg.numerical.fp_solvers import FPFVMSolver

    problem, m0 = _problem_2d()
    U = np.zeros((problem.Nt + 1, 9, 9))

    with pytest.raises(NotImplementedError, match=r"this one is a tensor \(volatility_kind='tensor'\)"):
        FPFVMSolver(problem).solve_fp_system(
            m0, potential_field=U, volatility=np.full((2, 2), 0.45), volatility_kind="tensor"
        )

    field = FPFVMSolver(problem).solve_fp_system(
        m0, potential_field=U, volatility=np.full((9, 9), 0.45), volatility_kind="field"
    )
    scalar = FPFVMSolver(problem).solve_fp_system(m0, potential_field=U, volatility=0.45)
    np.testing.assert_array_equal(field, scalar)


def _loops():
    from mfgarchon.alg.numerical.coupling.block_iterators import BlockIterator
    from mfgarchon.alg.numerical.coupling.fictitious_play import FictitiousPlayIterator
    from mfgarchon.alg.numerical.coupling.fixed_point_iterator import FixedPointIterator
    from mfgarchon.alg.numerical.coupling.newton_mfg_solver import NewtonMFGSolver

    return {
        "FixedPointIterator": FixedPointIterator,
        "FictitiousPlayIterator": FictitiousPlayIterator,
        "BlockIterator": BlockIterator,
        "NewtonMFGSolver": NewtonMFGSolver,
    }


def test_expert_mode_refuses_a_pair_built_for_another_volatility():
    """#2420 review, blocker 1: a pair of solvers built for another problem's volatility.

    Before part 2a, `MFGProblem.solve` forwarded its volatility as an override, so Expert Mode with a
    pair built from a problem of volatility 0.3 solved at the calling problem's 0.6. With no forwarding
    each solver reads its own problem, and the same call returned the 0.3 solve while the boundary
    providers were handed 0.6: the review measured, on a 1-D pair, the other problem's solve returned
    bit for bit. Now the coupling loop refuses the mismatch by name.
    """
    from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
    from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver

    built_for, _ = _problem_2d(volatility=0.3)
    solved, _ = _problem_2d(volatility=0.6)
    with pytest.raises(ValueError, match=r"differs from the problem being solved \(solved=0\.6, solver's=0\.3"):
        solved.solve(hjb_solver=HJBFDMSolver(built_for), fp_solver=FPFDMSolver(built_for), max_iterations=1)


@pytest.mark.parametrize("loop", ["FixedPointIterator", "FictitiousPlayIterator", "BlockIterator", "NewtonMFGSolver"])
def test_every_coupling_loop_refuses_a_pair_built_for_another_volatility(loop):
    """The same guard in each loop that takes an override, which an override still satisfies.

    And for callables: two distinct callable volatilities cannot be compared, so they count as
    different, while a pair sharing the solved problem's callable is accepted (#2420 review, round 2).
    """
    from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
    from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver

    cls = _loops()[loop]
    built_for, _ = _problem_2d(volatility=0.3)
    solved, _ = _problem_2d(volatility=0.6)
    with pytest.raises(ValueError, match=r"HJB solver was built from a problem whose volatility differs") as refusal:
        cls(solved, HJBFDMSolver(built_for), FPFDMSolver(built_for))
    assert "callables" not in str(refusal.value), "a scalar mismatch must not be explained as a callable one"
    cls(solved, HJBFDMSolver(built_for), FPFDMSolver(built_for), volatility=0.6)

    def sigma_low(t, x, m):
        return 0.3 + 0.0 * np.asarray(m)

    def sigma_high(t, x, m):
        return 0.6 + 0.0 * np.asarray(m)

    callable_built_for, _ = _problem_2d(volatility=sigma_low)
    callable_solved, _ = _problem_2d(volatility=sigma_high)
    with pytest.raises(ValueError, match=r"two distinct callables count as different"):
        cls(callable_solved, HJBFDMSolver(callable_built_for), FPFDMSolver(callable_built_for))
    sharing, _ = _problem_2d(volatility=sigma_high)
    cls(callable_solved, HJBFDMSolver(sharing), FPFDMSolver(sharing))


def test_the_fp_solvers_problem_is_compared_too():
    """An HJB solver without a `problem` is not compared, so the FP solver's must be (#2420 review, round 2)."""
    from mfgarchon.alg.numerical.coupling.base_mfg import assert_paired_solver_sigma
    from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver

    built_for, _ = _problem_2d(volatility=0.3)
    solved, _ = _problem_2d(volatility=0.6)
    fp_for_another = FPFDMSolver(built_for)
    with pytest.raises(ValueError, match=r"FP solver was built from a problem whose volatility differs"):
        assert_paired_solver_sigma(SimpleNamespace(), fp_for_another, "probe", problem=solved)
