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

from typing import TYPE_CHECKING

import numpy as np

from mfgarchon.types.callable_protocols import SOURCE_TERM_SLOTS, bound_attribute

from .graph_coupling import _get_time_slice

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from mfgarchon.core.mfg_problem import MFGProblem
    from mfgarchon.operators.interaction.energy_functionals import EnergyFunctional


def _call_problem_source(problem: MFGProblem, name: str, *, t: float, x: NDArray, m: NDArray, v: NDArray) -> NDArray:
    """``problem.<name>`` at one time slice, through its binding (#2375 ruling 8).

    The one place a problem's ``source_term_hjb`` or ``source_term_fp`` is invoked.
    """
    return bound_attribute(problem, name, SOURCE_TERM_SLOTS, role=name)(x=x, m=m, v=v, t=t)


def hjb_source_fields(problem: MFGProblem) -> list[str]:
    """The fields that declare a problem-level HJB source, of those this problem sets: the one owner of that
    predicate (#2666, #1642 stage 1).

    ``"source_term_hjb"``, ``"nonlocal_operator"``, and ``"coupling_cost"`` for a bound ``Model.coupling_cost``
    (``problem.coupling_energy``). Every consumer reads this rather than the fields, so a part added here is composed,
    and refused where it cannot be, everywhere at once: the grid couplers (`compose_hjb_source`), ``GraphMFGSolver``,
    and ``BaseHJBSolver.solve``'s refusal.
    """
    fields = []
    if problem.source_term_hjb is not None:
        fields.append("source_term_hjb")
    if problem.nonlocal_operator is not None:
        fields.append("nonlocal_operator")
    if problem.coupling_energy is not None:
        fields.append("coupling_cost")
    return fields


def hjb_source_is_active(problem: MFGProblem) -> bool:
    """Whether this problem declares any HJB source, without composing one (`hjb_source_fields`)."""
    return bool(hjb_source_fields(problem))


def _coupling_part(energy: EnergyFunctional, nodes: NDArray, m_t: NDArray, x: NDArray, t: float) -> NDArray:
    """The coupling cost's part of the HJB source: ``delta F / delta m`` at the density slice ``m_t``, at the nodes F
    was bound on (#1642 stage 1).

    ``nodes`` are the grid's, the rows of ``geometry.get_spatial_grid()`` in the grid's C order, which F was bound in,
    and ``m_t`` is flattened in that order. The part evaluates only at those nodes: an ``x`` that is not them -- other
    points, or the same points in another order -- is refused rather than given the nodes' values. Evaluating it at a
    solver's own points, such as HJB-GFDM's collocation points where they are not the grid's nodes in their order, is
    #2737.
    """
    flat = np.asarray(m_t, dtype=float).ravel()
    nodes = np.asarray(nodes, dtype=float).reshape(np.shape(energy.weights)[0], -1)
    points = np.asarray(x, dtype=float).reshape(np.shape(x)[0], -1) if np.ndim(x) else np.zeros((0, 1))
    axes = [np.unique(column) for column in nodes.T]
    spacing = min((float(np.min(np.diff(a))) for a in axes if a.size > 1), default=1.0)
    if (
        points.shape != nodes.shape
        or flat.size != len(nodes)
        or not np.allclose(points, nodes, rtol=0, atol=1e-9 * spacing)
    ):
        raise ValueError(
            f"The coupling cost was bound on this problem's {len(nodes)} grid nodes, and this HJB evaluation passes x "
            f"with {points.shape[0]} rows that are not those nodes in their order, and a density slice of {flat.size} "
            "values. The coupling cost evaluates only at the nodes it was bound on (#1642): a solver that evaluates "
            "its source at its own points, such as HJB-GFDM's collocation points where they are not the grid's "
            "nodes in their order, cannot carry it until #2737. Solve "
            "with a grid solver -- HJB-FDM, HJB-SL, or HJB-WENO in 1-D -- or, for a linear F = integral V m, give V "
            "to the Hamiltonian as its potential."
        )
    return np.asarray(energy.flat_derivative(flat, t=t), dtype=float)


def _problem_hjb_source_terms(
    problem: MFGProblem,
    m_current: NDArray,
    u_current: NDArray,
    t: float,
    x: NDArray,
    dt: float,
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
        out["coupling"] = _coupling_part(
            problem.coupling_energy, np.asarray(problem.geometry.get_spatial_grid()), m_t, x, t
        )
    if "nonlocal_operator" in fields:
        out["nonlocal"] = problem.nonlocal_operator @ v_t
    return out


def compose_hjb_source(
    problem: MFGProblem,
    m_current: NDArray,
    u_current: NDArray,
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

    def composed(t: float, x: NDArray) -> NDArray:
        # Source + nonlocal via the shared single-source primitive (Issue #1382). Order
        # [source, nonlocal] preserves byte-for-byte float-sum associativity with the pre-#1382
        # closure; the obstacle term that used to sit between them contributed
        # `(1/eps) * max(0, psi)` with `eps = 1e6`, i.e. 5e-07 at psi = 0.5, so its removal is
        # visible in the last bits of any problem that set it -- and every such problem now
        # raises at construction instead.
        # The parts in the primitive's own order, source -> coupling -> nonlocal, the one order every consumer sums
        # in: the coupling part (#1642) sits after the source, so a problem without it sums the other two in their
        # pre-#1642 order, bit for bit.
        parts = _problem_hjb_source_terms(problem, m_current, u_current, t, x, problem.dt)
        terms = list(parts.values())
        return sum(terms) if terms else np.zeros(x.shape[0])

    return composed


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
