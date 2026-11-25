"""A coupled solve's result, built in one place for every coupled route (#2997, rulings 4 and 7, and stage 1c's plan).

Every coupling route returns one `SolverResult`. The fields every route has are decided here, from what the route
hands over:

- `SweepRecord`: each sweep's change, per component, as `sweep_change` measured it. The iteration count, the absolute
  histories, the relative histories in the metadata, and each component's own histories are read from it.
- `mass_conservation_error`: the drift of the mass the route's model conserves, per component and at the top level.
- `refuse_components_off_one_grid`: a route with a component axis pairs its components' levels step for step and
  holds one array per field, so its components share one time grid and one shape.
- `coupled_result`: the `SolverResult`, U and M with a leading component axis on a route that has one.

What only one route decides, such as Newton's counters or FixedPoint's Anderson flag, it hands over as metadata of its
own.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from mfgarchon.utils.non_finite_read import refuse_a_non_finite_read
from mfgarchon.utils.solver_result import SolverResult

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mfgarchon.config.damping import DampingPolicy

_KEYS = ("l2distu_abs", "l2distm_abs", "l2distu_rel", "l2distm_rel")


class MassInvariant(Enum):
    """Which mass a route's model conserves, which its `mass_conservation_error` measures the drift of."""

    EACH = "each"
    """Each component's own mass: the components exchange none, as MultiPopulation's populations. The top level is the
    worst component's drift, since a total can hide two drifts that cancel."""

    TOTAL = "total"
    """The total over components: the components exchange mass, as Graph's nodes through a Laplacian coupling and
    RegimeSwitching's regimes through Q, whose rows sum to zero. A component's own drift is then partly the exchange."""


class SweepRecord:
    """Each sweep's change, per component, as `sweep_change` measured it: what a coupled result's iteration count and
    histories are read from.

    A route records every sweep it runs: `measured` with the sweep's changes, or `diverged` for a sweep that stopped
    before its change was measured. The diverged sweep counts, and its entries are NaN, so a diverged solve's final
    change is not the last finite sweep's (#1672's shape).
    """

    def __init__(self, components: int) -> None:
        self._components = components
        self._sweeps: list[list[dict[str, float]]] = []

    def measured(self, per_component: Sequence[Mapping[str, float]]) -> None:
        if len(per_component) != self._components:
            raise ValueError(f"SweepRecord: a sweep of {len(per_component)} components, of {self._components}")
        self._sweeps.append([{key: float(change[key]) for key in _KEYS} for change in per_component])

    def diverged(self) -> None:
        self._sweeps.append([dict.fromkeys(_KEYS, float("nan")) for _ in range(self._components)])

    @property
    def iterations(self) -> int:
        return len(self._sweeps)

    def history(self, key: str) -> np.ndarray:
        """Per sweep, the worst component's ``key``: the change the verdict reads (`worst_sweep_change`)."""
        if not self._sweeps:
            return np.zeros(0)
        return np.max(np.array([[change[key] for change in sweep] for sweep in self._sweeps]), axis=1)

    def per_component(self) -> list[dict[str, np.ndarray]]:
        """Each component's own histories."""
        return [
            {key: np.array([sweep[k][key] for sweep in self._sweeps]) for key in _KEYS} for k in range(self._components)
        ]


def _mass_history(M: Any, geometry: Any, reader: str) -> np.ndarray | None:
    """The total mass at each time level, through the geometry's own integral, or None where it has none (#2145, #2157).

    `integrate` refuses a geometry with no measure, a single-node axis being the case that exists, and that is no
    measurement either. A non-positive initial mass is refused: the drift from it is undefined, and the solve that
    produced it is already wrong.
    """
    integrate = getattr(geometry, "integrate", None)
    if not callable(integrate):
        return None
    try:
        mass = np.asarray(integrate(M), dtype=float)
    except (ValueError, NotImplementedError):
        return None
    initial_mass = float(mass[0])
    refuse_a_non_finite_read(reader, "the initial density's total mass", {"the initial mass": initial_mass})
    if initial_mass <= 0.0:
        raise ValueError(
            f"initial density has non-positive total mass ({initial_mass!r}); "
            "mass conservation is undefined and the solve that produced it is already wrong"
        )
    return mass


def _override(fp_solver: Any) -> float | None:
    """The FP solver's own number, `BaseFPSolver.mass_conservation_error_override` (#2188), or None: the coupling layer
    duck-types its solvers, and one without the seam reports nothing solver-specific, as the base class's does."""
    method = getattr(fp_solver, "mass_conservation_error_override", None)
    return method() if callable(method) else None


def _drift(mass: np.ndarray) -> float:
    return float(np.max(np.abs(mass / float(mass[0]) - 1.0)))


def mass_conservation_error(
    Ms: Sequence[Any],
    geometries: Sequence[Any],
    fp_solvers: Sequence[Any],
    *,
    invariant: MassInvariant,
    measured: bool,
    reader: str,
) -> tuple[float | None, list[float | None]]:
    """``(top level, per component)``: max over t of ``|mass(t) / mass(0) - 1|``, the drift from the initial mass.

    None means not measured (#1672), and None and 0.0 are different statements:
    - ``measured`` is False when no FP step of this solve completed: M is then the cold start or a warm start, not this
      solve's density (#1717);
    - a geometry with no measure has no total mass.

    An FP solver whose state holds what the grid density discards, the particles still alive, reports its own number
    (`mass_conservation_error_override`, #2188). A total over components cannot combine such a number with the others,
    so under `MassInvariant.TOTAL` it leaves the top level None, and each component's number stands.
    """
    if not measured:
        return None, [None] * len(Ms)
    overrides = [_override(solver) for solver in fp_solvers]
    histories = [
        None if override is not None else _mass_history(M, geometry, reader)
        for M, geometry, override in zip(Ms, geometries, overrides, strict=True)
    ]
    per_component = [
        override if override is not None else (None if mass is None else _drift(mass))
        for override, mass in zip(overrides, histories, strict=True)
    ]
    if invariant is MassInvariant.EACH:
        if any(drift is None for drift in per_component):
            return None, per_component
        return max(per_component), per_component
    if any(mass is None for mass in histories):
        return None, per_component
    total = np.sum(histories, axis=0)
    initial_total = float(total[0])
    refuse_a_non_finite_read(reader, "the initial density's total mass", {"the initial mass": initial_total})
    return _drift(total), per_component


def refuse_components_off_one_grid(
    problems: Sequence[Any], shapes: Sequence[tuple[int, ...]], *, reader: str, component: str
) -> None:
    """Refuse, by name and before the first sweep, components that do not share one time grid and one iterate shape
    (#2997, the user's rulings of 2026-11-25).

    - **The time grid, (T, Nt), compared exactly.** A route reads every other component's level at the step it is
      solving, by index: RegimeSwitching's sources read regime j's level ``round(t / dt_k)``, so regimes with different
      time steps were coupled at different times, with no error.
    - **The shape.** The result holds every component's U and M in one array each, of shape ``(K, Nt + 1, *shape)``.
    """
    times = [(problem.T, problem.Nt) for problem in problems]
    late = [k for k in range(1, len(times)) if times[k] != times[0]]
    if late:
        raise ValueError(
            f"{reader}: {component}(s) {late} step through time other than {component} 0's: (T, Nt) "
            f"{[times[k] for k in late]} against {times[0]}, compared exactly. Every {component}'s level is read beside "
            f"the others' at the same step, so the {component}s of one solve share one time grid (#2997)."
        )
    differ = [k for k in range(1, len(shapes)) if shapes[k] != shapes[0]]
    if differ:
        raise ValueError(
            f"{reader}: {component}(s) {differ} have an iterate of shape {[shapes[k] for k in differ]} against "
            f"{component} 0's {shapes[0]}. The result holds every {component}'s U and M in one array each, of shape "
            f"(K, Nt + 1, *shape), so the {component}s of one solve share one grid size (#2997)."
        )


def coupled_result(
    *,
    U: Any,
    M: Any,
    record: SweepRecord,
    converged: bool,
    reason: str,
    solver_name: str,
    damping: DampingPolicy,
    problems: Sequence[Any],
    fp_solvers: Sequence[Any],
    fp_output_available: bool,
    component_names: Sequence[str] | None = None,
    invariant: MassInvariant | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SolverResult:
    """The `SolverResult` of a coupled solve.

    A route with a component axis passes ``component_names``, its ``invariant``, and U and M as one array per
    component; its result's U and M are those arrays stacked, of shape ``(K, Nt + 1, *shape)``, and ``components`` is
    K. A single-pair route passes U and M as arrays, and its ``components`` is None.

    ``reason`` is the stop the route's verdict gave: `check_convergence_criteria`'s reason, a divergence, or the
    route's own stop. A refusal over failed inner HJB steps replaces it afterwards, in `verdict_reads_inner_failures`.
    """
    components = None if component_names is None else len(component_names)
    if components is None:
        Us, Ms = [U], [M]
        invariant = MassInvariant.EACH
    else:
        if invariant is None:
            raise ValueError(f"{solver_name}: a route with a component axis declares the mass its model conserves")
        Us, Ms = list(U), list(M)
    top, per_component_mass = mass_conservation_error(
        Ms,
        [problem.geometry for problem in problems],
        fp_solvers,
        invariant=invariant,
        measured=fp_output_available,
        reader=solver_name,
    )
    result_metadata: dict[str, Any] = {
        "convergence_reason": reason,
        "l2distu_rel": record.history("l2distu_rel"),
        "l2distm_rel": record.history("l2distm_rel"),
        "damping": damping,
    }
    if components is not None:
        result_metadata["component_names"] = list(component_names)
        result_metadata["per_component"] = [
            {**histories, "mass_conservation_error": drift}
            for histories, drift in zip(record.per_component(), per_component_mass, strict=True)
        ]
    for key, value in (metadata or {}).items():
        if key in result_metadata:
            raise ValueError(f"{solver_name}: metadata key {key!r} is the result owner's")
        result_metadata[key] = value
    return SolverResult(
        U=Us[0] if components is None else np.stack(Us),
        M=Ms[0] if components is None else np.stack(Ms),
        iterations=record.iterations,
        error_history_U=record.history("l2distu_abs"),
        error_history_M=record.history("l2distm_abs"),
        solver_name=solver_name,
        converged=converged,
        metadata=result_metadata,
        mass_conservation_error=top,
        components=components,
    )
