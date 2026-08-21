"""Issue #2020: `source_term` must be honoured or refused, never silently discarded.

WHY THIS IS NOT A SIGNATURE AUDIT
---------------------------------
`inspect.signature` answers "does this callable name the parameter", which is a coarser question
than "does a source reach the discretisation". Two ways it diverges, both live in this repo:

- `HJBWENOSolver` NAMES `source_term` and still refuses it in multi-D (`hjb_weno.py`), because the
  multi-D path is a dimensional split and a source added per axis sweep would be applied `d` times.
  A signature census counts it as reachable.
- Four solvers declare `**kwargs` and swallow the parameter whole. A census keyed on "does not
  raise TypeError" counts THOSE as reachable too.

So the population predicate has to be behavioural: pass a source, and see whether the answer moves.
That is the shape `test_weno_mms_order_1991.py` already uses for the solvers that do thread it.

WHAT IS AND IS NOT AT RISK
--------------------------
The coupling layer is NOT the hazard. `resolve_volatility_kwarg` in `coupling/base_mfg.py` treats
`**kwargs` as *not* accepting a parameter and raises; the `source_term` branch beside it has done the
same since #1424, and `test_kwarg_gate_var_keyword_1783.py` pins that end-to-end on both the Picard
and the Newton path. Anything going through `FixedPointIterator` or `MFGResidual` gets a loud
refusal.

The hazard is the DIRECT call, which bypasses that gate. Surveyed at the time of writing, no test in
the suite drives a manufactured solution through a swallowing solver -- the six direct
`source_term=` call sites are all on `HJBFDMSolver`, `FPFDMSolver`, `HJBGFDMSolver`,
`HJBWENOSolver` and `PenaltyHJBSolver`. This file exists so that stays true by failure rather than
by nobody having tried.

STATUS: `_SWALLOWERS` is empty -- nothing silently discards a source any more, which is what #2020
was about. ~~every solver here now HONOURS a source~~ [CORRECTED 2026-08-21] that was true of the
six rows this file had; it covered 6 of 21 concrete solvers. Eight more are now rows, and five of
them REFUSE -- `HJBSemiLagrangianSolver`, `FPGFDMSolver`, `FPParticleSolver`, `FPSLSolver`,
`FPSLJacobianSolver`. Refusing is a correct outcome and not a defect, but "every solver honours a
source" and "every solver in my list honours a source" are different claims, and the list was the
smaller half of the population.

`test_every_concrete_solver_is_covered_or_named` now holds the parametrisation against discovery,
so the gap between the two cannot reopen quietly. Seven solvers still have no row and are named in
`_UNCOVERED` with the reason.

What the file does that the class-definition gate cannot is check BEHAVIOUR: the gate reads a
signature, and this repository has twice now had a solver whose signature and behaviour disagreed -- `HJBWENOSolver` names
`source_term` and refuses it in multi-D, and six solvers named nothing and swallowed it. A signature
gate cannot tell those apart; passing a real source and watching the answer can.
"""

from __future__ import annotations

import inspect
import pkgutil
from importlib import import_module

import pytest

import numpy as np

import mfgarchon.alg as _alg
from mfgarchon.alg.numerical.fp_solvers.base_fp import BaseFPSolver
from mfgarchon.alg.numerical.hjb_solvers.base_hjb import BaseHJBSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import Mesh1D, TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

_N = 21

# Classes whose resolved solve_*_system accepts **kwargs and discards `source_term`. RECORDED
# DEFECT, not a contract: when one of these starts honouring or refusing the parameter, the
# behavioural test below fails on that row and the failure message says to move it.
# Empty, and the point is that it stays empty. `BaseHJBSolver.__init_subclass__` /
# `BaseFPSolver.__init_subclass__` now refuse a `**kwargs` override that does not NAME
# `source_term`, so a new swallower cannot be defined at all -- this set is the behavioural witness
# that the structural gate is doing what it claims, which a class-definition gate cannot check about
# itself. It held these six until #2020: HJBFEMSolver, MeshlessGalerkinHJBSolver, WeakFormHJBSolver,
# FPFEMSolver, MeshlessGalerkinFPSolver, WeakFormFPSolver.
_SWALLOWERS: set[str] = set()


def _components():
    return MFGComponents(
        m_initial=lambda x: np.exp(-10 * (np.asarray(x) - 0.5) ** 2),
        u_terminal=lambda x: np.asarray(x) * 0.0,
        hamiltonian=SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=1.0),
            coupling=lambda m: np.asarray(m) * 0.0,
            coupling_dm=lambda m: np.asarray(m) * 0.0,
        ),
    )


def _grid_problem():
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[_N], boundary_conditions=no_flux_bc(dimension=1))
    return MFGProblem(geometry=grid, T=0.2, Nt=5, sigma=0.3, components=_components(), coupling_coefficient=0.0)


def _mesh_problem():
    mesh = Mesh1D(bounds=(0.0, 1.0), num_elements=_N - 1)
    mesh.generate_mesh()
    mesh.boundary_conditions = no_flux_bc(dimension=1)
    return MFGProblem(geometry=mesh, T=0.2, Nt=5, sigma=0.3, components=_components(), coupling_coefficient=0.0)


class _Source:
    """A source that records whether it was ever called. A constant 5.0 is deliberate: it is large
    against every field in this fixture, so a solver that reaches it cannot produce a zero delta."""

    def __init__(self):
        self.calls = 0

    def __call__(self, _t, x):
        self.calls += 1
        a = np.asarray(x, dtype=float)
        return np.full(a.shape[0] if a.ndim == 2 else a.size, 5.0)


def _hjb_fdm():
    from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver

    p = _grid_problem()
    m = np.tile(np.ones(_N) / _N, (p.Nt + 1, 1))
    u = np.zeros((p.Nt + 1, _N))
    s = HJBFDMSolver(p)
    return lambda f: s.solve_hjb_system(
        M_density=m, U_terminal=np.zeros(_N), U_coupling_prev=u, **({"source_term": f} if f else {})
    )


def _fp_fdm():
    from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver

    p = _grid_problem()
    u = np.zeros((p.Nt + 1, _N))
    s = FPFDMSolver(p)
    return lambda f: s.solve_fp_system(np.ones(_N) / _N, potential_field=u, **({"source_term": f} if f else {}))


def _meshless_hjb():
    from mfgarchon.alg.numerical.meshless_galerkin.hjb_solver import MeshlessGalerkinHJBSolver

    p = _grid_problem()
    pts = np.linspace(0.0, 1.0, _N).reshape(-1, 1)
    m = np.tile(np.ones(_N) / _N, (p.Nt + 1, 1))
    u = np.zeros((p.Nt + 1, _N))
    s = MeshlessGalerkinHJBSolver(p, pts, delta=2.6 / np.sqrt(_N))
    return lambda f: s.solve_hjb_system(m, np.zeros(_N), u, **({"source_term": f} if f else {}))


def _meshless_fp():
    from mfgarchon.alg.numerical.meshless_galerkin.fp_solver import MeshlessGalerkinFPSolver

    p = _grid_problem()
    pts = np.linspace(0.0, 1.0, _N).reshape(-1, 1)
    u = np.zeros((p.Nt + 1, _N))
    s = MeshlessGalerkinFPSolver(p, pts, delta=2.6 / np.sqrt(_N))
    return lambda f: s.solve_fp_system(np.ones(_N) / _N, potential_field=u, **({"source_term": f} if f else {}))


def _hjb_fem():
    pytest.importorskip("skfem", reason="scikit-fem required for the weak-form FEM solvers")
    from mfgarchon.alg.numerical.fem.hjb_fem_solver import HJBFEMSolver

    s = HJBFEMSolver(_mesh_problem(), order=1)
    return lambda f: s.solve_hjb_system(
        M_density=None, U_terminal=None, U_coupling_prev=None, **({"source_term": f} if f else {})
    )


def _fp_fem():
    pytest.importorskip("skfem", reason="scikit-fem required for the weak-form FEM solvers")
    from mfgarchon.alg.numerical.fem.fp_fem_solver import FPFEMSolver

    s = FPFEMSolver(_mesh_problem(), order=1)
    x = s._disc.dof_coordinates[:, 0]
    m0 = np.ones_like(x) / len(x)
    return lambda f: s.solve_fp_system(
        m0.copy(), potential_field=None, volatility_field=0.3, **({"source_term": f} if f else {})
    )


def _hjb_gfdm():
    from mfgarchon.alg.numerical.hjb_solvers.hjb_gfdm import HJBGFDMSolver

    p = _grid_problem()
    m = np.tile(np.ones(_N) / _N, (p.Nt + 1, 1))
    u = np.zeros((p.Nt + 1, _N))
    s = HJBGFDMSolver(p, collocation_points=np.linspace(0.0, 1.0, _N).reshape(-1, 1))
    return lambda f: s.solve_hjb_system(m, np.zeros(_N), u, **({"source_term": f} if f else {}))


def _hjb_weno():
    from mfgarchon.alg.numerical.hjb_solvers.hjb_weno import HJBWENOSolver

    p = _grid_problem()
    m = np.tile(np.ones(_N) / _N, (p.Nt + 1, 1))
    u = np.zeros((p.Nt + 1, _N))
    s = HJBWENOSolver(p)
    # 1-D ONLY. `hjb_weno.py` refuses `source_term` for every d >= 2, deliberately: the multi-D
    # path is a dimensional split, so a source added per axis sweep would be applied d times.
    # This row therefore says nothing about WENO in 2-D, where the correct verdict is "refuses".
    return lambda f: s.solve_hjb_system(m, np.zeros(_N), u, **({"source_term": f} if f else {}))


def _fp_fvm():
    from mfgarchon.alg.numerical.fp_solvers.fp_fvm import FPFVMSolver

    p = _grid_problem()
    u = np.zeros((p.Nt + 1, _N))
    s = FPFVMSolver(p)
    return lambda f: s.solve_fp_system(np.ones(_N) / _N, potential_field=u, **({"source_term": f} if f else {}))


def _hjb_semi_lagrangian():
    from mfgarchon.alg.numerical.hjb_solvers.hjb_semi_lagrangian import HJBSemiLagrangianSolver

    p = _grid_problem()
    m = np.tile(np.ones(_N) / _N, (p.Nt + 1, 1))
    u = np.zeros((p.Nt + 1, _N))
    s = HJBSemiLagrangianSolver(p)
    return lambda f: s.solve_hjb_system(m, np.zeros(_N), u, **({"source_term": f} if f else {}))


def _fp_gfdm():
    from mfgarchon.alg.numerical.fp_solvers.fp_gfdm import FPGFDMSolver

    p = _grid_problem()
    u = np.zeros((p.Nt + 1, _N))
    s = FPGFDMSolver(p, collocation_points=np.linspace(0.0, 1.0, _N).reshape(-1, 1))
    return lambda f: s.solve_fp_system(np.ones(_N) / _N, drift_field=u, **({"source_term": f} if f else {}))


def _fp_particle():
    from mfgarchon.alg.numerical.fp_solvers.fp_particle import FPParticleSolver

    p = _grid_problem()
    u = np.zeros((p.Nt + 1, _N))
    s = FPParticleSolver(p)
    return lambda f: s.solve_fp_system(np.ones(_N) / _N, potential_field=u, **({"source_term": f} if f else {}))


def _fp_sl():
    from mfgarchon.alg.numerical.fp_solvers.fp_semi_lagrangian_adjoint import FPSLSolver

    p = _grid_problem()
    u = np.zeros((p.Nt + 1, _N))
    s = FPSLSolver(p)
    return lambda f: s.solve_fp_system(np.ones(_N) / _N, potential_field=u, **({"source_term": f} if f else {}))


def _fp_sl_jacobian():
    from mfgarchon.alg.numerical.fp_solvers.fp_semi_lagrangian import FPSLJacobianSolver

    p = _grid_problem()
    u = np.zeros((p.Nt + 1, _N))
    s = FPSLJacobianSolver(p)
    return lambda f: s.solve_fp_system(np.ones(_N) / _N, potential_field=u, **({"source_term": f} if f else {}))


# (label, factory, expected). "honours" rows are the positive control WITHOUT which the
# "swallows" rows prove nothing -- a harness that never delivers a source would report every
# solver as swallowing it.
_CASES = [
    ("HJBFDMSolver", _hjb_fdm, "honours"),
    ("FPFDMSolver", _fp_fdm, "honours"),
    ("MeshlessGalerkinHJBSolver", _meshless_hjb, "honours"),
    ("MeshlessGalerkinFPSolver", _meshless_fp, "honours"),
    ("HJBFEMSolver", _hjb_fem, "honours"),
    ("FPFEMSolver", _fp_fem, "honours"),
    # Added 2026-08-21. These eight were in the population and outside _CASES, so the file's
    # headline covered 6 of 21 concrete solvers while reading as though it covered the channel.
    ("HJBGFDMSolver", _hjb_gfdm, "honours"),
    ("HJBWENOSolver", _hjb_weno, "honours"),  # 1-D only -- see the factory
    ("FPFVMSolver", _fp_fvm, "honours"),
    ("HJBSemiLagrangianSolver", _hjb_semi_lagrangian, "refuses"),
    ("FPGFDMSolver", _fp_gfdm, "refuses"),
    ("FPParticleSolver", _fp_particle, "refuses"),
    ("FPSLSolver", _fp_sl, "refuses"),
    ("FPSLJacobianSolver", _fp_sl_jacobian, "refuses"),
]


#: Concrete solvers with no row above, and why. NAMED rather than inferred: a curated
#: parametrisation is a claim about scope, and the claim is only auditable if its gaps are written
#: down. `test_every_concrete_solver_is_covered_or_named` compares this against discovery, so a
#: solver added to the package cannot slip past the list in silence.
_UNCOVERED: dict[str, str] = {
    "FPSLAdjointSolver": "alias -- overrides only __init__ and shares FPSLSolver.solve_fp_system, "
    "so a row here would measure the same method object twice",
    "NetworkHJBSolver": "needs a network problem; raises NotImplementedError about node BCs on a grid",
    "NetworkPolicyIterationHJBSolver": "needs a network problem; reads problem.num_nodes unguarded "
    "and raises AttributeError on a grid",
    "FPNetworkSolver": "needs a network problem",
    "WeakFormHJBSolver": "intended-abstract base -- constructs given a discretization, then its BC "
    "hooks raise NotImplementedError; the concrete subclasses are the FEM rows above",
    "WeakFormFPSolver": "intended-abstract base -- same",
    "PenaltyHJBSolver": "wrapper -- delegates solve_hjb_system to an inner solver, so it measures "
    "whichever row that inner solver already has",
}


def _concrete_solvers() -> set[str]:
    """Every concrete BaseHJBSolver / BaseFPSolver subclass under `mfgarchon.alg.numerical`.

    Discovery must not be keyed on the property under audit, so no signature and no membership of
    `_CASES` takes part in choosing the population.
    """
    import importlib
    import pkgutil

    import mfgarchon.alg.numerical as _numerical

    found: set[str] = set()
    for info in pkgutil.walk_packages(_numerical.__path__, _numerical.__name__ + "."):
        try:
            module = importlib.import_module(info.name)
        except Exception:
            continue
        for obj in vars(module).values():
            if (
                inspect.isclass(obj)
                and issubclass(obj, (BaseHJBSolver, BaseFPSolver))
                and obj not in (BaseHJBSolver, BaseFPSolver)
            ):
                found.add(obj.__name__)
    return found


def test_every_concrete_solver_is_covered_or_named():
    """The parametrisation must account for the whole population, row by row or by name.

    Without this the file reads as "the source_term channel is tested" while covering whichever
    solvers someone thought of. It covered 6 of 21 until 2026-08-21 -- every FDM, FEM and meshless
    path, and none of GFDM, WENO, FVM, semi-Lagrangian or particle -- and nothing said so.
    """
    population = _concrete_solvers()
    assert len(population) == 21, sorted(population)

    covered = {c[0] for c in _CASES}
    assert covered - population == set(), f"rows for classes that are not in the population: {covered - population}"

    unaccounted = population - covered - set(_UNCOVERED)
    assert unaccounted == set(), (
        f"concrete solvers with neither a row nor an entry in _UNCOVERED: {sorted(unaccounted)}. "
        f"Add a row if the fixture can drive it, or name the reason it cannot."
    )

    stale = set(_UNCOVERED) - population
    assert stale == set(), f"_UNCOVERED names classes that no longer exist: {sorted(stale)}"

    overlap = set(_UNCOVERED) & covered
    assert overlap == set(), f"listed as both covered and uncovered: {sorted(overlap)}"


@pytest.mark.parametrize(("label", "factory", "expected"), _CASES, ids=[c[0] for c in _CASES])
def test_a_source_term_is_honoured_or_refused_never_discarded(label, factory, expected):
    """Pass a source; the answer must move, or the call must raise. Silence is the defect."""
    call = factory()
    baseline = np.asarray(call(None), dtype=float)

    src = _Source()
    try:
        forced = np.asarray(call(src), dtype=float)
    except (TypeError, NotImplementedError):
        # A refusal is a correct outcome -- the third option the contract allows, and what the
        # coupling layer produces for every solver here. It is recorded per row rather than
        # accepted anywhere, so that a solver which STARTS honouring the parameter fails this
        # test and gets its row updated deliberately (that is #2020 progress, not a regression).
        assert expected in ("swallows", "refuses"), f"{label} refuses source_term but is listed as {expected!r}"
        return

    delta = float(np.abs(forced - baseline).max())

    if expected == "refuses":
        raise AssertionError(
            f"{label} is listed as refusing source_term but accepted it "
            f"(called {src.calls} times, max|diff| = {float(np.abs(forced - baseline).max()):.3e}). "
            f"If it now honours the parameter that is #2020 progress -- change the row to "
            f"'honours' and say so."
        )

    if expected == "honours":
        assert src.calls > 0, f"{label} accepted source_term without ever calling it (delta {delta:.3e})"
        assert delta > 1e-9, (
            f"{label} called the source {src.calls} times and the answer did not move "
            f"(max|diff| = {delta:.3e}). The channel is wired but inert."
        )
        return

    # RECORDED DEFECT, not a contract. `source_term` is swallowed by a `**kwargs` signature and
    # never reaches the discretisation. Fixing it -- by honouring the source or by refusing it --
    # trips one of these two lines, and that failure is the instruction.
    raise AssertionError(  # pragma: no cover - unreachable while _CASES has no "swallows" rows
        f"{label} is listed as swallowing source_term, but every solver in _CASES now honours or "
        f"refuses it (#2020). Remove the row or the classification."
    )


def test_the_swallowing_set_has_not_grown():
    """A new solver inheriting a `**kwargs` solve method joins the defect silently otherwise.

    Classification is by MRO-resolved signature, not by each class's own ``__dict__``: three of the
    six define nothing themselves and resolve to a CONCRETE intermediate that already dropped the
    parameter (``FPFEMSolver`` -> ``WeakFormFPSolver``, ``MeshlessGalerkinFPSolver`` ->
    ``WeakFormFPSolver``, ``HJBFEMSolver`` -> ``WeakFormHJBSolver``). A ``__dict__``-only census
    reads "not overridden" as "inherits the base, therefore accepts", which is how #1991's table
    came to carry a row that was wrong when written.
    """
    found: dict[str, type] = {}
    failed = []
    for mod in pkgutil.walk_packages(_alg.__path__, _alg.__name__ + "."):
        try:
            module = import_module(mod.name)
        except Exception:  # a module that cannot import cannot contribute a solver
            failed.append(mod.name)
            continue
        for obj in vars(module).values():
            if not inspect.isclass(obj):
                continue
            for base, method in ((BaseHJBSolver, "solve_hjb_system"), (BaseFPSolver, "solve_fp_system")):
                if issubclass(obj, base) and obj is not base:
                    found[obj.__name__] = getattr(obj, method)

    assert not failed, f"modules failed to import, so the population is short: {failed}"
    assert "HJBFDMSolver" in found, "the walk did not find a solver known to exist -- the query is wrong"
    assert len(found) >= 15, f"expected the full solver population, found {len(found)}"

    swallow = {
        name
        for name, fn in found.items()
        if "source_term" not in inspect.signature(fn).parameters
        and any(p.kind is inspect.Parameter.VAR_KEYWORD for p in inspect.signature(fn).parameters.values())
    }
    assert swallow == _SWALLOWERS, (
        f"the set of solvers that swallow `source_term` through **kwargs changed.\n"
        f"  added:   {sorted(swallow - _SWALLOWERS)}\n"
        f"  removed: {sorted(_SWALLOWERS - swallow)}\n"
        f"Added means a new solver joined the #2020 defect. Removed means one was fixed -- update "
        f"_SWALLOWERS and the 'swallows' rows in _CASES."
    )
