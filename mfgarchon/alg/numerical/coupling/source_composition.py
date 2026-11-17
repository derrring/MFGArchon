"""Single-source composition of problem-level source/nonlocal/obstacle terms.

Issue #1361: lift ``_compose_hjb_source`` / ``_compose_fp_source`` out of
``FixedPointIterator`` so the Picard coupler (``FixedPointIterator``) and the
coupled-Newton path (``MFGResidual`` behind ``NewtonMFGSolver``) consume **one**
copy of the convention. [The consumer set above is #1361's and is no longer current.
``base_mfg.hjb_solve_kwargs`` / ``fp_solve_kwargs`` compose from the iterates rather than
receiving a composed source (#2207); ``FixedPointIterator``, ``FictitiousPlayIterator`` and
``BlockIterator`` reach them through ``_build_hjb_kwargs`` / ``_build_fp_kwargs``, and
``MultiPopulationIterator`` calls them per population (#2666). ``MFGResidual`` (behind
``NewtonMFGSolver``) and ``RegimeSwitchingIterator``, which adds its own terms to the problem's
source, call this module directly; ``BaseHJBSolver.solve`` asks only whether it would compose
anything, and refuses. ``GraphMFGSolver`` composes in its own ``_compose_hjb_source`` /
``_compose_fp_source`` from this module's primitives. The reason for one copy is unchanged.] A second private copy is the "parallel physics paths
with private convention copies" bug class that already produced two silent
divergences in the Picard copy alone:

- #1259 — ``nonlocal_operator`` ``J[v]`` computed but never applied.
- #1285 — time-dependent source used the terminal density slice ``m[-1]`` for
  every backward time step instead of the time-``t`` slice.

Both functions read the extended PDE fields off ``MFGProblem`` and return a
solver-level ``(t, x) -> array`` closure compatible with
``BaseHJBSolver.solve_hjb_system(source_term=...)`` /
``BaseFPSolver.solve_fp_system(source_term=...)``, or ``None`` when no relevant
field is active.

Conventions (mirrored verbatim from the prior ``FixedPointIterator`` copy):

- **Time-``t`` slicing** of the bound density / value iterates via
  :func:`graph_coupling._get_time_slice` (round-to-nearest index), so a
  time-dependent source sees ``m[k]`` / ``v[k]``, not the full ``(Nt+1, Nx)``
  array (Issue #1285).
- **Nonlocal term** applied as ``s += nonlocal_operator @ v_t`` with ``v_t`` the
  time-``t`` slice of the value function (Issue #1259), matching
  ``graph_mfg_solver``'s sign convention.
- **Obstacle** — the branch and its whole account are GONE (#2002 resolved, 2026-08-21). It
  computed ``(1/eps) * max(0, psi)``, a term with no ``v`` in it: positive wherever ``Psi > 0``
  whether or not ``v >= Psi`` held, and byte-identical at a node satisfying the constraint and
  one violating it. It penalised POSITION, not VIOLATION, so it was never an approximation to
  the variational inequality it was documented as — it was a different term.

  It resolved as a **soft wall**, which is what that term always was. A cost that is ``alpha``-free
  and ``u``-free is a POTENTIAL, so it now lives where potentials live: ``problem.state_penalty``,
  composed into the Hamiltonian's ``V`` at problem construction. It never reaches this closure and
  there is nothing here to compose. ``problem.obstacle`` raises and names both successors.

  Deleted rather than left beside the replacement: this module exists to stop parallel physics
  paths keeping private copies of a convention, and a composition that only ADDS an owner has
  consolidated nothing.

  The **variational inequality** is a separate, reserved slot — ``ObstacleConstraint`` with
  ``HJBFDMSolver(constraint=...)`` (#591), deliberately unfinished: it projects rather than
  solving the VI, and only that one solver carries it (#2046). It was never this channel's
  to provide, since a constraint penalty needs ``v`` and ``source_term`` is ``(t, x) -> array``.

- The HJB source passes the **value-function slice** ``v_t`` to
  ``source_term_hjb`` (Issue #1382), matching the documented
  ``Callable(t, x, v, m)`` contract (``mfg_problem.py``: "source_term_hjb/fp"),
  the FP source (which already binds ``v_t``), and ``graph_mfg_solver``. The
  prior copy passed ``v = 0`` here — a latent divergence from the graph coupler
  for any ``v``-dependent HJB source. Both couplers now build these terms through
  the single :func:`_problem_hjb_source_terms` primitive so the fork cannot
  re-open silently.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np

from mfgarchon.core.mfg_problem import (
    CouplingBinding,
    _call_problem_source,
    _coupling_part,
    problem_coupling_binding,
)

from .graph_coupling import _get_time_slice

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from mfgarchon.core.mfg_problem import MFGProblem


def hjb_source_fields(problem: MFGProblem) -> list[str]:
    """The fields that declare a problem-level HJB source, of those this problem sets: the one owner of that
    predicate (#2666, #1642 stage 1).

    ``"source_term_hjb"``, ``"nonlocal_operator"``, and ``"coupling_cost"`` wherever the model declares a
    ``coupling_cost``, bound by the problem on its grid (``problem.coupling_energy``) or left to the HJB solver off a
    grid (stage 1b-2). Every consumer reads this rather than the fields, so a part added here is composed, and refused
    where it cannot be, everywhere at once: the grid couplers (`compose_hjb_source`), ``GraphMFGSolver``, and
    ``BaseHJBSolver.solve``'s refusal.
    """
    fields = []
    if problem.source_term_hjb is not None:
        fields.append("source_term_hjb")
    if problem.nonlocal_operator is not None:
        fields.append("nonlocal_operator")
    if problem.declared_coupling_cost is not None:
        fields.append("coupling_cost")
    return fields


def hjb_source_is_active(problem: MFGProblem) -> bool:
    """Whether this problem declares any HJB source, without composing one (`hjb_source_fields`)."""
    return bool(hjb_source_fields(problem))


def coupling_binding_for(
    problem: MFGProblem, binding: CouplingBinding | None = None, reader: str | None = None
) -> CouplingBinding:
    """The binding a declared ``Model.coupling_cost`` is read through (#1642 stage 1b-2).

    On a grid, the problem's own (`problem_coupling_binding`), whatever HJB solver evaluates it. Off a grid the problem
    binds none, and the binding is ``binding``, the HJB solver's (``BaseHJBSolver.coupling_binding``), when it was bound
    from this problem's cost: a solver binds the cost of the problem it was built from, and a coupler may pair it with
    another problem. Otherwise it is refused by name, naming ``reader``.
    """
    own = problem_coupling_binding(problem)
    if own is not None:
        return own
    who = reader or "compose_hjb_source"
    if binding is not None and binding.cost is problem.declared_coupling_cost:
        return binding
    if binding is not None:
        raise NotImplementedError(
            f"{who} was handed a binding of another problem's coupling cost: the HJB solver was built from a problem "
            f"whose Model(coupling_cost=...), a {type(binding.cost).__name__}, is not the same object as the "
            f"{type(problem.declared_coupling_cost).__name__} the problem solved declares (#1642). The costs are "
            "compared as objects, so an equal cost built again, or a copy, is another. A solver binds the cost of the "
            "problem it was built from, so build the HJB solver from the problem it solves."
        )
    raise NotImplementedError(
        f"{who} has no binding of Model(coupling_cost=...) on this {type(problem.geometry).__name__}: off a grid the "
        "problem binds none, since a coupling cost is bound where it is put to use, by the HJB solver that evaluates "
        "it, on its own points and weights, and a solver binds the cost of the problem it was built from (#1642). "
        "HJB-FEM P1 on a mesh binds it, built from this problem, through a coupler that passes its binding: "
        "problem.solve(), FixedPointIterator, a BlockIterator, or MultiPopulationIterator with one population. On a "
        "point cloud or an implicit domain, HJB-GFDM's binding at its collocation points is #2761."
    )


def _problem_hjb_source_terms(
    problem: MFGProblem,
    m_current: NDArray,
    u_current: NDArray,
    t: float,
    x: NDArray,
    dt: float,
    binding: CouplingBinding | None = None,
    reader: str | None = None,
) -> dict[str, NDArray]:
    """The convention-bearing HJB source terms shared by every coupler (Issue #1382).

    Returns a dict with the present problem-level HJB source contributions evaluated
    at the time-``t`` slices: ``"source"`` = ``source_term_hjb(t, x, v_t, m_t)`` with
    ``v_t`` the value-function slice (NOT zero), ``"nonlocal"`` =
    ``nonlocal_operator @ v_t``, and ``"coupling"`` = the bound coupling cost's
    ``delta F / delta m`` at ``m_t`` (`_coupling_part`). This is the single source of the ``v_t`` convention
    for both the grid couplers (``compose_hjb_source``) and ``graph_mfg_solver``, so a
    ``v``-dependent source cannot silently diverge between paths.

    Named (dict) so a consumer can tell the parts apart; their insertion order, source then coupling then
    nonlocal, is the one order every consumer sums them in (``compose_hjb_source``, ``GraphMFGSolver``).

    ``dt`` is passed explicitly (rather than read from ``problem.dt``) so the graph
    coupler — which carries its own ``self._dt`` — slices identically.

    ``binding`` is the coupling cost's, from the HJB solver the source is for (`CouplingBinding`). On a grid the
    problem's own binding is read instead; off a grid ``binding`` is read when it was bound from this problem's cost. A
    declared cost with neither, or with another problem's, is refused by name, naming ``reader``
    (`coupling_binding_for`).
    """
    out: dict[str, NDArray] = {}
    fields = hjb_source_fields(problem)
    needs_v = "source_term_hjb" in fields or "nonlocal_operator" in fields
    v_t = _get_time_slice(u_current, t, dt) if needs_v else None
    needs_m = "source_term_hjb" in fields or "coupling_cost" in fields
    m_t = _get_time_slice(m_current, t, dt) if needs_m else None
    if "source_term_hjb" in fields:
        out["source"] = _call_problem_source(problem, "source_term_hjb", t=t, x=x, m=m_t, v=v_t)
    if "coupling_cost" in fields:
        bound = coupling_binding_for(problem, binding, reader)
        out["coupling"] = _coupling_part(bound.energy, bound.nodes, m_t, x, t)
    if "nonlocal_operator" in fields:
        out["nonlocal"] = problem.nonlocal_operator @ v_t
    return out


def compose_hjb_source(
    problem: MFGProblem,
    m_current: NDArray,
    u_current: NDArray,
    binding: CouplingBinding | None = None,
    reader: str | None = None,
) -> Callable[[float, NDArray], NDArray] | None:
    """Compose problem-level HJB source terms into a solver-level callable.

    Composes the parts `hjb_source_fields` names -- ``source_term_hjb``, the bound
    coupling cost, ``nonlocal_operator`` -- binds the current density ``m_current``
    and value function ``u_current``, and returns a ``(t, x) -> array`` closure.

    Args:
        problem: MFG problem definition carrying the extended PDE fields.
        m_current: Density iterate ``(Nt+1, Nx)`` bound for time-``t`` slicing.
        u_current: Value-function iterate ``(Nt+1, Nx)`` bound for the nonlocal
            term's time-``t`` slicing.
        binding: The coupling cost's binding from the HJB solver the source is for (`CouplingBinding`). On a grid
            the problem's own binding is read instead. Off a grid it is read when it was bound from this problem's
            cost; a declared cost with no such binding is refused by name here, before any evaluation
            (`coupling_binding_for`).
        reader: Who reads the source, for that refusal's message.

    Returns:
        Callable, or ``None`` if the problem declares no HJB source (`hjb_source_is_active`).
    """

    # The obstacle branch that stood here is GONE (#2002), not moved. `problem.obstacle` is
    # retired and raises; a soft wall is `problem.state_penalty`, composed into the Hamiltonian's
    # potential at problem construction, because a term that is `alpha`-free and `u`-free is a
    # potential and belongs where V lives. Deleted in the same change that added the replacement:
    # a composition that only ADDS an owner has not consolidated anything.
    if not hjb_source_is_active(problem):
        return None
    if "coupling_cost" in hjb_source_fields(problem):
        binding = coupling_binding_for(problem, binding, reader)
    return ComposedHJBSource(problem, m_current, u_current, binding, reader)


class ComposedHJBSource:
    """The HJB source a problem declares, composed for a solver (`compose_hjb_source`).

    Called at ``(t, x)`` it is the sum of its parts, in the one order every consumer sums them in. `parts` is the same
    parts by the problem's field names, read through the same function (`_problem_hjb_source_terms`), so a reader that
    meets a non-finite sum names the part that made it (#2975).
    """

    _FIELDS: ClassVar[dict[str, str]] = {
        "source": "source_term_hjb",
        "coupling": "coupling_cost",
        "nonlocal": "nonlocal_operator",
    }

    def __init__(
        self,
        problem: MFGProblem,
        m_current: NDArray,
        u_current: NDArray,
        binding: CouplingBinding | None,
        reader: str | None,
    ) -> None:
        self._problem, self._m, self._u, self._binding, self._reader = problem, m_current, u_current, binding, reader

    def parts(self, t: float, x: NDArray) -> dict[str, NDArray]:
        """The parts at ``(t, x)``, by the problem's field name: ``source_term_hjb``, ``coupling_cost``,
        ``nonlocal_operator``, those the problem declares."""
        raw = _problem_hjb_source_terms(
            self._problem, self._m, self._u, t, x, self._problem.dt, self._binding, self._reader
        )
        return {self._FIELDS[key]: value for key, value in raw.items()}

    def __call__(self, t: float, x: NDArray) -> NDArray:
        # Source + nonlocal via the shared single-source primitive (Issue #1382). Order
        # [source, nonlocal] preserves byte-for-byte float-sum associativity with the pre-#1382
        # closure; the obstacle term that used to sit between them contributed
        # `(1/eps) * max(0, psi)` with `eps = 1e6`, i.e. 5e-07 at psi = 0.5, so its removal is
        # visible in the last bits of any problem that set it -- and every such problem now
        # raises at construction instead.
        # The parts in the primitive's own order, source -> coupling -> nonlocal, the one order every consumer sums
        # in: the coupling part (#1642) sits after the source, so a problem without it sums the other two in their
        # pre-#1642 order, bit for bit.
        parts = _problem_hjb_source_terms(
            self._problem, self._m, self._u, t, x, self._problem.dt, self._binding, self._reader
        )
        terms = list(parts.values())
        return sum(terms) if terms else np.zeros(x.shape[0])


def fp_source_is_active(problem: MFGProblem) -> bool:
    """Whether this problem has an FP source at all, without composing one.

    Exists so a caller that only needs the PREDICATE does not restate it. The strict-adjoint FP
    path in ``block_iterators.py`` needs exactly that -- it refuses a source it has no route for
    (#2207) -- and its first version restated ``problem.source_term_fp is not None`` inline, which
    put a second owner on the predicate inside the change whose whole thesis is that composition
    has one. The risk is not hypothetical: the sibling :func:`compose_hjb_source` read
    ``obstacle`` until #2002 and reads ``coupling_cost`` since #1642, so a second copy drifts the
    day this one grows a field.

    Calling :func:`compose_fp_source` instead would work and says the wrong thing: it reads as an
    evaluation, and it would bind an ``M_initial`` that ``_get_time_slice`` freezes in time.
    """
    return problem.source_term_fp is not None


def compose_fp_source(
    problem: MFGProblem,
    m_current: NDArray,
    v_current: NDArray,
) -> Callable[[float, NDArray], NDArray] | None:
    """Compose problem-level FP source terms into a solver-level callable.

    Reads ``source_term_fp`` from the problem, binds the current density and
    value-function iterates, and returns a ``(t, x) -> array`` closure that
    evaluates the source at the time-``t`` slices of ``m`` and ``v``.

    Args:
        problem: MFG problem definition carrying the extended PDE fields.
        m_current: Density iterate ``(Nt+1, Nx)`` bound for time-``t`` slicing.
        v_current: Value-function iterate ``(Nt+1, Nx)`` bound for time-``t``
            slicing.

    Returns:
        Callable, or ``None`` if ``source_term_fp`` is not set.
    """
    has_source = problem.source_term_fp is not None

    if not has_source:
        return None

    def composed(t: float, x: NDArray) -> NDArray:
        m_t = _get_time_slice(m_current, t, problem.dt)
        v_t = _get_time_slice(v_current, t, problem.dt)
        return _call_problem_source(problem, "source_term_fp", t=t, x=x, m=m_t, v=v_t)

    return composed
