"""
Multi-population Picard iterator for K-population MFG.

Issue #910 Phase 2: Coordinates K single-population solves in the
Picard fixed-point loop. Each iteration:
  1. Solve K HJB equations (each sees all current densities)
  2. Extract K drift fields via H_k.optimal_control()
  3. Solve K FP equations
  4. Damp each population's U and M (its `DampingPolicy`, #2997)

Reuses standard HJB/FP solvers — this class only coordinates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mfgarchon.alg.numerical.coupling.fixed_point_utils import verdict_reads_inner_failures
from mfgarchon.config.damping import DampingPolicy
from mfgarchon.core.hamiltonian import _terms_reading_every_population, _why_not_as_it_ships
from mfgarchon.utils.mfg_logging import get_logger

from .base_mfg import (
    assert_paired_solver_geometry,
    assert_paired_solver_sigma,
    damping_argument,
    fp_solve_kwargs,
    handed,
    hjb_coupling_binding,
    hjb_solve_kwargs,
    hjb_terminal_binding,
    refuse_a_capped_exit_in_a_coupled_solve,
    solver_signature,
)
from .coupled_result import MassInvariant, SweepRecord, coupled_result, refuse_components_off_one_grid
from .damping_run import DampingRun
from .fixed_point_utils import BUDGET_SPENT, diverged_value_function, fp_output_stop
from .iteration_state import IterationState

if TYPE_CHECKING:
    from mfgarchon.alg.numerical.fp_solvers.base_fp import BaseFPSolver, DriftConvention
    from mfgarchon.alg.numerical.hjb_solvers.base_hjb import BaseHJBSolver
    from mfgarchon.core.multi_population import MultiPopulationProblem
    from mfgarchon.utils.solver_result import SolverResult

logger = get_logger(__name__)


def refuse_what_the_doors_cannot_pair(multi_problem: MultiPopulationProblem) -> None:
    """Refuse, before the first sweep, what the doors cannot read row for row (#2335).

    The iterator stacks one row of nodes per population and step, and its doors (the 1-D HJB residual and Jacobian, and
    the 1-D FP velocity) hand a density term that declares ``cross_density`` every population's row at the same points
    and step. So at two or more populations it refuses a grid of two or more dimensions; nodes that differ, compared
    exactly; and a Hamiltonian with a term that declares the slot which is not a
    ``SeparableHamiltonian`` or ``CongestionHamiltonian`` as the library ships it (`_why_not_as_it_ships`): the
    library bodies of those two hand a declaring term the rows (`_takes_the_rows_path`). A time grid that differs is
    refused by the solve, as on every route with a component axis (`refuse_components_off_one_grid`, #2997).
    """
    K = multi_problem.K
    if K < 2:
        return
    populations = [multi_problem.get_population(k) for k in range(K)]
    flat = [k for k, population in enumerate(populations) if population.dimension != 1]
    if flat:
        raise NotImplementedError(
            f"MultiPopulationIterator: population(s) {flat} sit on a grid of dimension "
            f"{[populations[k].dimension for k in flat]}. The iterator stacks one row of nodes per population, and its "
            "doors read the rows on a 1-D grid only (#2335)."
        )
    nodes = [np.asarray(population.geometry.get_spatial_grid(), dtype=float) for population in populations]
    apart = {
        k: f"{nodes[k].shape[0]} nodes against {nodes[0].shape[0]}"
        if nodes[k].shape != nodes[0].shape
        else f"coordinates up to {np.abs(nodes[k] - nodes[0]).max():.3e} apart"
        for k in range(1, K)
        if nodes[k].shape != nodes[0].shape or not np.array_equal(nodes[k], nodes[0])
    }
    if apart:
        raise ValueError(
            f"MultiPopulationIterator: population(s) {list(apart)} sit on a grid other than population 0's ("
            + "; ".join(f"population {k}: {gap}" for k, gap in apart.items())
            + ", compared exactly). Each population's density is read at the points of the door that reads it, and "
            "every population's is handed beside it as cross_density at those same points, so the populations of one "
            "solve share one grid (#2335)."
        )
    for k, population in enumerate(populations):
        hamiltonian = population.hamiltonian_class
        terms = _terms_reading_every_population(hamiltonian)
        why = _why_not_as_it_ships(hamiltonian) if terms else None
        if why:
            raise NotImplementedError(
                f"MultiPopulationIterator: population {k}'s {type(hamiltonian).__name__} has term(s) {terms} that "
                f"declare cross_density, and is not a SeparableHamiltonian or CongestionHamiltonian as the library "
                f"ships it: {why}. Only the library bodies of a SeparableHamiltonian or CongestionHamiltonian as the "
                "library ships it hand a term every population's rows; this population's Hamiltonian is not one, so "
                "the solve was not run (#2335)."
            )


class MultiPopulationIterator:
    """Picard iteration for K-population MFG.

    Parameters
    ----------
    multi_problem : MultiPopulationProblem
        Container holding K single-population problems.
    hjb_solvers : list[BaseHJBSolver]
        One HJB solver per population.
    fp_solvers : list[BaseFPSolver]
        One FP solver per population.
    damping : DampingPolicy | None
        Each population's damping of U and M, a `DampingPolicy` (#2997), applied per population: every population's
        fields adapt on their own history. None is ``DampingPolicy.constant(0.5)``: both fields damped from the first
        sweep.

    Examples
    --------
    >>> iterator = MultiPopulationIterator(
    ...     multi_problem=multi,
    ...     hjb_solvers=[hjb_A, hjb_B],
    ...     fp_solvers=[fp_A, fp_B],
    ... )
    >>> result = iterator.solve(max_iterations=50, tolerance=1e-6)
    >>> result.U  # K value functions, of shape (K, Nt+1, *shape)
    >>> result.M  # K densities, of the same shape
    """

    def __init__(
        self,
        multi_problem: MultiPopulationProblem,
        hjb_solvers: list[BaseHJBSolver],
        fp_solvers: list[BaseFPSolver],
        damping: DampingPolicy | None = None,
    ):
        self.multi_problem = multi_problem
        self.hjb_solvers = hjb_solvers
        self.fp_solvers = fp_solvers
        self.damping = damping_argument(damping, "MultiPopulationIterator") or DampingPolicy.constant(0.5)
        K = multi_problem.K

        if len(hjb_solvers) != K:
            raise ValueError(f"Need {K} HJB solvers, got {len(hjb_solvers)}")
        if len(fp_solvers) != K:
            raise ValueError(f"Need {K} FP solvers, got {len(fp_solvers)}")
        # RFC #1574 C14 / Issue #1603: each population HJB-FP pair is an adjoint pair; guard every
        # pair, not just population 0, against silent sigma desync.
        for _k in range(K):
            assert_paired_solver_sigma(hjb_solvers[_k], fp_solvers[_k], f"MultiPopulationIterator[population {_k}]")
            assert_paired_solver_geometry(
                hjb_solvers[_k],
                fp_solvers[_k],
                f"MultiPopulationIterator[population {_k}]",
                problem=multi_problem.get_population(_k),
            )

        # Issue #1043: each FP solver's solve_fp_system signature, so the drift/potential convention is resolved via the
        # shared resolve_fp_drift_kwargs (same as single-pop); and #2666's HJB twin, for the population's source. Both
        # through the one reader every coupled route builds with, which refuses one it cannot read (#2209).
        self._fp_sig_params: list[set[str]] = [solver_signature(fp, "FP") for fp in fp_solvers]
        self._hjb_sig_params: list[set[str]] = [solver_signature(h, "HJB") for h in hjb_solvers]

        # Issue #1489 (S1): per-population FP drift-input convention, parallel to _fp_sig_params, so
        # resolve_fp_drift_kwargs routes each population by convention rather than param presence.
        self._fp_drift_convention: list[DriftConvention | None] = [
            getattr(fp, "_drift_convention", None) for fp in fp_solvers
        ]

    @verdict_reads_inner_failures(lambda self: list(self.hjb_solvers), fp_solvers_of=lambda self: list(self.fp_solvers))
    def solve(
        self,
        max_iterations: int = 50,
        tolerance: float = 1e-6,
        absolute_tolerance: float | None = None,
    ) -> SolverResult:
        """Run Picard iteration over K populations.

        ``tolerance`` bounds the largest relative L2 change of a sweep, over populations and fields, each
        population's in its own problem's measure (docs/user/CONVENTIONS.md § 9). ``absolute_tolerance``,
        if given, must also bound the largest absolute change (#2555).

        Returns
        -------
        SolverResult
            U and M of shape (K, Nt+1, *shape), population k's at ``U[k]`` and ``M[k]``, and ``components`` K (#2997).
            Its ``mass_conservation_error`` is the worst population's drift, since the populations exchange no mass; each
            population's own is in ``metadata["per_component"]``. On a diverged HJB, each population the sweep solved
            holds that HJB output as its U, the diverged population's with its NaN kept (#1718); the later populations'
            U, and every M, are the iterate after the last finite sweep.
        """
        refuse_a_capped_exit_in_a_coupled_solve(self.fp_solvers, "MultiPopulationIterator.solve")
        K = self.multi_problem.K
        # The terminal coupling is single-population (#1642, the user's ruling 5): with more than one population this
        # loop couples them through the stacked densities it hands each HJB as its cross-density.
        coupled = [k for k in range(K) if self.multi_problem.get_population(k).declared_terminal_coupling is not None]
        if coupled:
            raise NotImplementedError(
                f"MultiPopulationIterator: population(s) {coupled} set Model(terminal_coupling=...). A terminal coupling "
                "is single-population (#1642): with more than one population this iterator couples them through the "
                "stacked densities it hands each population's HJB, and a G read at them is not defined. Solve a single "
                "population alone, with its own MFGProblem.solve(), or give each population's terminal cost through "
                "Conditions(u_terminal=...)."
            )
        refuse_what_the_doors_cannot_pair(self.multi_problem)

        # Each population's iterate from its one owner (#2997): its layout, the grid's shape or, for one population on a
        # mesh, the pair's DOF count, and its cold start, u(T) at m(0) in every row of U and m(0) with the FP's
        # prescribed values in M (#2991). Its u(T) is the terminal datum's one owner, read through the population's HJB
        # solver's binding (#1642); no population carries a terminal coupling (refused above).
        states = [
            IterationState(
                self.multi_problem.get_population(k),
                self.hjb_solvers[k],
                self.fp_solvers[k],
                reader=f"MultiPopulationIterator[population {k}]",
                terminal_binding=hjb_terminal_binding(self.hjb_solvers[k]),
                unstructured_mesh=True,
            )
            for k in range(K)
        ]
        refuse_components_off_one_grid(
            [self.multi_problem.get_population(k) for k in range(K)],
            [state.solution_shape for state in states],
            reader="MultiPopulationIterator",
            component="population",
        )
        U, M = [], []
        for state in states:
            U_k, M_k = state.start()
            U.append(U_k)
            M.append(M_k)
        # One damping run per population: each population's fields adapt on its own history (#2997 rulings 5 and 9).
        runs = [DampingRun(self.damping) for state in states]

        from mfgarchon.utils.convergence import sweep_change, worst_sweep_change

        from .fixed_point_utils import check_convergence_criteria

        populations = [self.multi_problem.get_population(k) for k in range(K)]
        integrates = [p.spatial_measure().integrate for p in populations]
        dts = [p.dt for p in populations]

        # Picard iteration. Every sweep run is recorded, a diverged one as NaN, so a diverged solve's final change is not
        # the last finite sweep's: the #1672 shape, the best-looking number attached to the worst solve (#2997).
        converged = False
        record = SweepRecord(K)
        fp_output_available = False
        # The reason a spent budget reports, on every route (#3002, ruling 2)
        reason = BUDGET_SPENT
        for iteration in range(max_iterations):
            # The owner's refusals answer at every sweep, so a BC changed during the solve is refused at the next one
            # (#3005; #3002 stage 2a, ruling A). Stage 2b moves this call into the coupled driver.
            for state in states:
                state.read_owner()
            M_old = [m.copy() for m in M]
            # The value function is half of the coupled unknown and was never captured, so the
            # convergence test below could not see it (#1684 item 5). A Picard sweep can settle m
            # while u is still moving -- #1914 records exactly that shape.
            U_old = [u.copy() for u in U]

            # Validate all populations have hamiltonian_class
            for k in range(K):
                if self.multi_problem.get_population(k).hamiltonian_class is None:
                    raise ValueError(
                        f"Population {k} ({self.multi_problem.population_names[k]}) "
                        "has no hamiltonian_class. Cannot compute drift velocity."
                    )

            # Build per-timestep stacked density for cross-coupling.
            # m_all_per_t[n] = concat(M[0][n], M[1][n], ..., M[K-1][n])
            m_all = np.concatenate(M, axis=-1)  # (Nt+1, K*Nx)

            # Step 1: Solve K HJB equations, each population's into its own slot: the map's output, before damping.
            # `break` inside the loop below leaves only the population loop; this carries the stop
            # out to the Picard loop (#1718).
            diverged_population = None
            U_map: list = [None] * K
            terminals: list = [None] * K
            for k in range(K):
                solver_k = self.hjb_solvers[k]
                # u(T) at the density this population's HJB solve uses, g, since a terminal coupling is refused above:
                # the solve and the write-back after the update both read it.
                U_terminal_k = terminals[k] = states[k].terminal_at(M[k][-1])

                # Issue #1071 (lock-faithful): pass the stacked cross-density trajectory ``m_all``
                # directly. The HJB solver indexes it at each integer timestep and hands it to the
                # population's own Hamiltonian beside ``M[k]``, its own density, which is what ``m``
                # holds there (#2335); a term reads the stack by declaring ``cross_density``. No
                # BoundHamiltonian wrapper and no round(t/dt). Only
                # HJBFDMSolver threads this today (Issue #1157); other backends would silently
                # decouple (their solve reads the uncoupled ``problem.hamiltonian_class``), so fail
                # loud for K>1. K==1 has no cross-coupling: a standalone solve is byte-identical,
                # so no cross-density is sent regardless of backend.
                honors_cross_density = getattr(solver_k, "_honors_multipop_cross_density", False)
                if K > 1 and not honors_cross_density:
                    raise NotImplementedError(
                        f"Multi-population cross-coupling requires HJBFDMSolver for the HJB step "
                        f"(Issue #1071); population {k} uses {type(solver_k).__name__}, which does "
                        "not thread the cross-density trajectory into solve_hjb_system and "
                        "would silently decouple. Use HJBFDMSolver for multi-population MFG."
                    )
                # Issue #2666: the population's own source, composed from the iterates this sweep starts
                # from -- the previous U, as every single-population loop binds it (#1259).
                hjb_kwargs = hjb_solve_kwargs(
                    populations[k],
                    self._hjb_sig_params[k],
                    type(solver_k).__name__,
                    M=M[k],
                    U=U[k],
                    binding=hjb_coupling_binding(solver_k),
                )
                # Every array handed to a solver is copied (`handed`, #3002 class line 7)
                if honors_cross_density:
                    args, own = handed(
                        M[k], U_terminal_k, U[k], cross_density=(None if K == 1 else m_all), **hjb_kwargs
                    )
                else:
                    args, own = handed(M[k], U_terminal_k, U[k], **hjb_kwargs)
                U_map[k] = solver_k.solve_hjb_system(*args, **own)

                # Issue #1718: check before the NEXT population is solved, not after the loop. Every
                # population's HJB is solved before any FP runs, so one population's NaN would reach
                # every other population's coupling source through `m_all` before anything noticed.
                # This sweep's map output is published for the populations solved so far, the diverged one's as the
                # guard returns it; the later populations keep the previous iterate.
                diverged = diverged_value_function(
                    U_map[k],
                    U_terminal_k,
                    site=f"MultiPopulationIterator[population {k}]",
                    density=m_all,
                    iteration=iteration,
                )
                if diverged is not None:
                    U[:k] = U_map[:k]
                    U[k] = diverged
                    diverged_population = k
                    break

            if diverged_population is not None:
                converged = False
                reason = "diverged_nan"
                record.diverged()
                break

            # Step 2: Solve K FP equations. The FP drift/potential convention is single-sourced
            # through resolve_fp_drift_kwargs (Issue #1043), identical to the single-population
            # FixedPointIterator — so a K=1 multi-population solve matches single-pop. Issue #1071
            # (lock-faithful): the population's OWN (unbound) Hamiltonian is passed as h_class plus,
            # at two or more populations, the stacked cross-density trajectory m_all, as at the HJB
            # step; the velocity (non-smooth H) reads its own density as m and every population's as
            # cross_density, at the faces (#2335) — no
            # BoundHamiltonian wrapper, no round(t/dt). Smooth-separable H takes the potential_field=U
            # path (cross-coupling enters via the HJB only). Network problems keep the U-as-drift
            # path (FPNetworkSolver extracts rates internally).
            from .fixed_point_utils import resolve_fp_drift_kwargs

            # One slot per population for the FP solver's own output, before damping (#1684 item 6/7).
            M_map: list = [None] * K

            for k in range(K):
                prob_k = self.multi_problem.get_population(k)
                m0_k = states[k].m0
                H_k = prob_k.hamiltonian_class
                fp_k = self.fp_solvers[k]
                # Issue #2666: the population's FP source, composed from the density this solve replaces
                # and the value function it consumes -- the new U, as FixedPointIterator binds it.
                fp_kwargs = fp_solve_kwargs(prob_k, self._fp_sig_params[k], type(fp_k).__name__, M=M[k], U=U_map[k])

                if getattr(prob_k, "spatial_dimension", None) == 0:
                    args, own = handed(m0_k, potential_field=U_map[k], show_progress=False, **fp_kwargs)
                    M_new_k = fp_k.solve_fp_system(*args, **own)
                else:
                    drift_kwargs, use_positional_U = resolve_fp_drift_kwargs(
                        prob_k,
                        self._fp_sig_params[k],
                        None,
                        U_map[k],
                        M[k],
                        h_class=H_k,
                        cross_density=None if K == 1 else m_all,
                        drift_convention=self._fp_drift_convention[k],
                        fp_solver=fp_k,
                    )
                    if use_positional_U:
                        args, own = handed(m0_k, U_map[k], show_progress=False, **fp_kwargs)
                    else:
                        args, own = handed(m0_k, show_progress=False, **drift_kwargs, **fp_kwargs)
                    M_new_k = fp_k.solve_fp_system(*args, **own)

                # The UNDAMPED map output is what the convergence test needs; see the block below.
                M_map[k] = M_new_k

            # Check convergence on BOTH fields, and on the MAP's residual rather than the damped
            # step.
            #
            # The map's output against the sweep's input, `U_map[k] - U_old[k]` and `M_map[k] - M_old[k]`, NOT the
            # damped step: the damped update is `theta * map + (1 - theta) * old`, so the damped step is identically
            # `theta` times the map's and the reported error would scale with the damping. That is #1684 items 6 and
            # 7 -- "turning damping down makes anything converge". An earlier version of this block measured the damped
            # step and so reintroduced it here: measured on the cross-coupled 2-population fixture at relaxation 0.01,
            # it reported converged=True at sweep 2 with a true residual 45-61x the tolerance.
            #
            # #2555: each population's change is its relative L2 change in its OWN problem's measure,
            # and the criterion is the single-population one, fed the max over populations -- so u and
            # m no longer meet one tolerance in their own units.
            # The FP-output rule every coupled route reads, on each population's own mass, MultiPopulation's invariant
            # (#2997; #3002 stage 2a, rulings 3(a) and A), before the change is measured, which refuses a non-finite one
            # (#2578). A stop publishes the sweep's HJB and FP outputs (ruling E). Stage 2b moves this call into the
            # coupled driver.
            stop = fp_output_stop(
                M_map,
                M_old,
                integrates,
                each_component=True,
                site="MultiPopulationIterator",
                iteration=iteration,
            )
            if stop is not None:
                U[:], M[:] = U_map, M_map
                fp_output_available = True
                reason = stop
                record.diverged()
                break

            per_population = [
                sweep_change(U_map[k], U_old[k], M_map[k], M_old[k], integrates[k], dts[k]) for k in range(K)
            ]
            worst = worst_sweep_change(per_population, "population")
            record.measured(per_population)

            # Damping, per population and field (the policy, #2997), then the write-back: level 0 to m(0), and U's
            # terminal row to the u(T) this sweep's HJB solve was given.
            for k in range(K):
                U_k, M_k = runs[k].update(iteration, U_old[k], U_map[k], M_old[k], M_map[k])
                U[k], M[k] = states[k].write_back(U_k, M_k, terminals[k])
                for message in runs[k].observe(iteration, per_population[k]):
                    logger.warning(f"MultiPopulationIterator[population {k}]: {message}")
            fp_output_available = True

            per_pop_m = [f"{change['l2distm_rel']:.2e}" for change in per_population]
            per_pop_u = [f"{change['l2distu_rel']:.2e}" for change in per_population]
            logger.info(
                f"Multi-pop iter {iteration + 1}/{max_iterations}: "
                f"max_err={max(worst['l2distm_rel'], worst['l2distu_rel']):.4e}, "
                f"per_pop_m={per_pop_m}, per_pop_u={per_pop_u}"
            )

            met, criteria_reason = check_convergence_criteria(
                worst["l2distu_rel"],
                worst["l2distm_rel"],
                worst["l2distu_abs"],
                worst["l2distm_abs"],
                tolerance,
                absolute_tolerance,
            )
            if met:
                converged, reason = True, criteria_reason
                break

        return coupled_result(
            U=U,
            M=M,
            record=record,
            converged=converged,
            reason=reason,
            solver_name="MultiPopulationIterator",
            damping=self.damping,
            problems=[self.multi_problem.get_population(k) for k in range(K)],
            fp_solvers=self.fp_solvers,
            fp_output_available=fp_output_available,
            component_names=list(self.multi_problem.population_names),
            invariant=MassInvariant.EACH,
        )
