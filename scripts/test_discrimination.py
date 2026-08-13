#!/usr/bin/env python3
"""Measure which tests discriminate, by mutating load-bearing conventions.

#1714 measured 47 of 85 tests in one campaign as guard-echo, and #1715 found two
agreement tests that passed under a 2x diffusion error. Both were established by
mutation, not by reading. This generalises that: perturb each convention the library
single-sources, run the suite, and record which tests notice.

A test killed by no mutation is not necessarily worthless -- it may guard a surface
none of these conventions reach. It is a test whose discrimination on the load-bearing
paths is **unmeasured**, which is the population #1701 asks for.

Why not select the population by name. Counting `*_agree` / `*_matches` / `*_equals`
gives 51, 114 or 156 depending on the pattern, and the name has no reliable relation to
whether the test compares two paths. Behaviour under mutation does.

TWO TRAPS, both live in this repo, both handled here:

- **#1677 -- an editable install pins imports to the main checkout.** Mutating a copy
  or a worktree and running pytest there can leave the *original* module imported, so
  every mutation reports zero kills and reads as "nothing discriminates". This script
  mutates the main checkout in place, restores each file from the text it read before
  writing (held in memory, under try/finally), and refuses to start with modified
  tracked files. It also asserts at run time that the imported `mfgarchon` is the tree
  it mutated. SIGINT is safe -- the original text is captured before the write and
  restored by the outer finally. SIGKILL is not: recover with `git checkout --`, which
  the dirty-tree guard forces before a re-run.
- **A mutation that kills nothing is ambiguous.** Either every test is blind to that
  convention, or the mutation never executed. Each mutation therefore carries `verify`:
  an expression run against the mutated tree that is true only if the perturbation is
  observably live. Three outcomes, not two:

      live + kills > 0   the kills are data
      live + kills == 0  UNCOVERED -- no test exercises this convention. A finding.
      not live           INEFFECTIVE -- a harness fault. Its zeros mean nothing.

  The control has to sit on the thing in doubt. An earlier version pinned each mutation
  to "a named test that must die", which conflated a wrong guess about the test's path
  with a genuinely untested convention: four mutations killing 113, 17, 5 and 5 tests
  were all reported INEFFECTIVE because the test *file* had been guessed wrong, while
  the one mutation that genuinely killed nothing looked identical to them.

Usage:
    python scripts/test_discrimination.py                  # all mutations
    python scripts/test_discrimination.py --only diffusion_scalar_2x
    python scripts/test_discrimination.py --paths tests/unit --json out.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# The CI marker set. ONE owner, read by scripts/local_ci.sh and by this script, so that a kill
# here means a kill there. It was two byte-identical string literals bound by a comment until
# #1909, and a comment is not a mechanism: diverging them silently measures kill counts against a
# different population than the gate runs, and nothing reports it.
MARKERS = (REPO / "scripts" / "ci_markers.txt").read_text().strip()

# The instrument must not be inside its own population.
#
# `tests/unit/test_discrimination_ratchet.py` asserts that each mutation's literal
# source anchor matches exactly once. Under a mutation it does not -- that is the whole
# point of the anchor test -- so it fails under EVERY mutation and adds exactly +1 to
# every kill count. Measured: 129/34/19/5/5/0 became 130/35/20/6/6/1.
#
# The +1 on the last one is the damage. `bc_default_reads_as_reflect` is UNCOVERED at 0,
# and a self-referential +1 reads as "one test now covers it" -- the instrument erasing
# the finding it exists to produce, while every other count moves just enough to look
# like a real improvement rather than an artifact.
SELF_TESTS = "tests/unit/test_discrimination_ratchet.py"

# The population #1715 asks about, as a regex over pytest node ids. Owned here and
# emitted into the kill matrix, because a claim of the form "193 of 308 are inert" is
# unrecoverable without it -- and hand-adding it to the JSON does not survive the next
# --json, which is how it was lost once already.
AGREEMENT_SHAPED = r"::[^:]*?(_agree|_agrees|_matches|_equals|_single_source|_identical|_consistent)"


@dataclass
class Mutation:
    """A single-site perturbation of a convention the library single-sources."""

    name: str
    path: str
    old: str
    new: str
    owner: str  # what convention this is, and the issue that made it single-source
    verify: str  # expression, true ONLY under the mutation -- the positive control


MUTATIONS: list[Mutation] = [
    Mutation(
        name="diffusion_scalar_2x",
        path="mfgarchon/utils/pde_coefficients.py",
        old="        return 0.5 * float(arr) ** 2  # scalar isotropic: unambiguous, kind not needed",
        new="        return float(arr) ** 2  # MUTATED: 2x diffusion",
        owner="D = sigma^2/2 for scalar sigma (#811)",
        verify="diffusion_from_volatility(2.0) == 4.0",
    ),
    Mutation(
        name="diffusion_field_2x",
        path="mfgarchon/utils/pde_coefficients.py",
        old='    if kind == "field":\n        return 0.5 * arr**2',
        new='    if kind == "field":\n        return arr**2  # MUTATED: 2x diffusion',
        owner="D = sigma^2/2 elementwise for a volatility field (#811)",
        verify="float(diffusion_from_volatility(np.array([2.0]), kind='field')[0]) == 4.0",
    ),
    Mutation(
        name="drift_coefficient_2x",
        path="mfgarchon/utils/pde_coefficients.py",
        old="        return 1.0 / h_class.control_cost.lambda_",
        new="        return 2.0 / h_class.control_cost.lambda_  # MUTATED",
        owner="FP drift c = 1/control_cost (#1420 / G-017)",
        verify="fp_drift_coefficient(_stub_problem(control_cost=2.0)) == 1.0",
    ),
    Mutation(
        name="optimal_control_sign",
        path="mfgarchon/core/hamiltonian.py",
        old="        return -self.sign * p / self._lambda",
        new="        return self.sign * p / self._lambda  # MUTATED: sign flipped",
        owner="QuadraticControlCost alpha* = -sign*p/lambda (#1649)",
        verify="float(QuadraticControlCost(control_cost=1.0).optimal_control(np.array([1.0]))[0]) > 0",
    ),
    Mutation(
        name="bc_noflux_reads_as_clamp",
        path="mfgarchon/geometry/boundary/bc_utils.py",
        old='    elif bc_type_lower in ("neumann", "no_flux", "robin"):\n        return "reflect"',
        new='    elif bc_type_lower in ("neumann", "no_flux", "robin"):\n        return "clamp"  # MUTATED',
        owner="no_flux/neumann/robin -> reflect (#1698)",
        verify="bc_type_to_geometric_operation('no_flux') == 'clamp'",
    ),
    Mutation(
        name="bc_default_reads_as_reflect",
        path="mfgarchon/geometry/boundary/bc_utils.py",
        old='    if bc_type is None:\n        return "clamp"  # Default: absorbing',
        new='    if bc_type is None:\n        return "reflect"  # MUTATED',
        owner="absent BC defaults to clamp/absorbing (#1698)",
        verify="bc_type_to_geometric_operation(None) == 'reflect'",
    ),
    Mutation(
        name="ghost_spacing_ignored",
        path="mfgarchon/geometry/boundary/applicator_fdm.py",
        old="            self._grid_spacing = values",
        new="            self._grid_spacing = None  # MUTATED: explicit spacing dropped, dx = 1.0 fallback",
        owner="the ghost buffer uses the caller's spacing, not dx = 1.0 (#1904)",
        verify=(
            "pad_array_with_ghosts(np.array([1.0, 2.0, 3.0]), neumann_bc(dimension=1, value=2.0),"
            " ghost_depth=1, spacing=0.05)[0] == 3.0"
        ),
    ),
    Mutation(
        name="grid_spacing_uses_point_count",
        path="mfgarchon/geometry/grids/tensor_grid.py",
        old="                (bounds[i][1] - bounds[i][0]) / (self._Nx_points[i] - 1) if self._Nx_points[i] > 1 else 0.0",
        new="                (bounds[i][1] - bounds[i][0]) / self._Nx_points[i] if self._Nx_points[i] > 1 else 0.0  # MUTATED: L/n instead of L/(n-1)",
        owner='uniform grid spacing is L/(node count - 1), not L/(node count) -- the L/n vs L/(n-1) convention. `self.spacing` is the single owner: get_grid_spacing() returns it verbatim (tensor_grid.py:823 `return self.spacing`), get_spacing() indexes it (:607), legacy_1d_attrs["Dx"] reads it (:484), cell volume ',
        verify="TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=no_flux_bc(dimension=1)).get_grid_spacing()[0] == 1.0 / 11",
    ),
    Mutation(
        name="grid_interval_count_reads_as_points",
        path="mfgarchon/geometry/grids/tensor_grid.py",
        old="            self._Nx_points: list[int] = [n + 1 for n in Nx]",
        new="            self._Nx_points: list[int] = [n for n in Nx]  # MUTATED: Nx read as node count, not interval count",
        owner='`Nx` names the INTERVAL count, so node count = Nx + 1 -- the #1889 interval-vs-node ambiguity at the grid\'s own entry point. Stated in the constructor docstring at tensor_grid.py:181-182: "- Nx (intervals): Nx_points = Nx + 1" / "- Nx_points (points): Nx = Nx_points - 1". Line 220-221 normalises a s',
        verify="TensorProductGrid(bounds=[(0.0, 1.0)], Nx=[10], boundary_conditions=no_flux_bc(dimension=1)).Nx_points == [10]",
    ),
    Mutation(
        name="periodic_endpoint_convention_inverted",
        path="mfgarchon/geometry/grids/tensor_grid.py",
        old="        if abs(coords[-1] - hi) > 0.5 * abs(coords[-1] - coords[-2]):\n            return PeriodicGridConvention.ENDPOINT_EXCLUSIVE\n        return PeriodicGridConvention.ENDPOINT_INCLUSIVE",
        new="        if abs(coords[-1] - hi) > 0.5 * abs(coords[-1] - coords[-2]):  # MUTATED: verdicts swapped\n            return PeriodicGridConvention.ENDPOINT_INCLUSIVE\n        return PeriodicGridConvention.ENDPOINT_EXCLUSIVE",
        owner='where the last node sits, MEASURED from the coordinates the grid built (#1822). `_axis_convention` is the single owner: `periodic_convention` calls it (tensor_grid.py:415) and `_first_axis_with_convention` calls it (:385). The property docstring states why it is derived rather than asserted: "Derive',
        verify="TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=no_flux_bc(dimension=1)).periodic_convention is PeriodicGridConvention.ENDPOINT_EXCLUSIVE",
    ),
    Mutation(
        name="m_initial_1d_counting_measure",
        path="mfgarchon/core/mfg_problem.py",
        old="            # 1D normalization (original)\n            dx = self._get_spacing() or 1.0",
        new="            # 1D normalization (original)\n            dx = 1.0  # MUTATED: normaliser ignores the grid spacing (counting measure)",
        owner="m(0,.) integrates to 1 under the geometry's own volume element, not under the counting measure -- the 1-D branch of the four-way normalisation dispatch at mfg_problem.py:1996-2042. Owner established by #1888 (`tests/unit/test_core/test_initial_density_mass_1888.py`), whose module docstring records t",
        verify="abs(float(np.sum(MFGProblem(geometry=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1)), Nt=4, T=0.2, sigma=1.0, components=MFGComponents(m_initial=lambda x: np.exp(-10 * (np.asarray(x) - 0.5) ** 2).squeeze(), u_terminal=lambda x: 0.0, hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)))).m_initial)) - 1.0) < 1e-9",
    ),
    Mutation(
        name="mass_drift_reported_as_deviation_from_one",
        path="mfgarchon/alg/numerical/coupling/fixed_point_iterator.py",
        old="                mass_conservation_error = float(np.max(np.abs(mass_per_step / initial_mass - 1.0)))",
        new="                mass_conservation_error = float(np.max(np.abs(mass_per_step - 1.0)))  # MUTATED: absolute deviation from 1.0",
        owner='`SolverResult.mass_conservation_error` is DRIFT from the initial mass, `max|mass(t)/mass(0) - 1|`, not deviation from a 1.0 target (#1672). Documented at `mfgarchon/utils/solver_result.py:41` -- "mass_conservation_error: max|mass(t)/mass(0) - 1| over time steps -- the drift from the" -- and argued a',
        verify="MFGProblem(geometry=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1)), Nt=4, T=0.2, sigma=1.0, components=MFGComponents(m_initial=lambda x: np.exp(-10 * (np.asarray(x) - 0.5) ** 2).squeeze(), u_terminal=lambda x: 0.0, hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)))).solve(scheme=NumericalScheme.FDM_UPWIND, max_iterations=2, verbose=False).mass_conservation_error > 0.5",
    ),
    Mutation(
        name="particle_mass_counting_measure",
        path="mfgarchon/alg/numerical/fp_solvers/fp_particle.py",
        old="            dV = float(spacing) if spacing > 1e-14 else 1.0",
        new="            dV = 1.0  # MUTATED: scalar spacing ignored, mass measured with the counting measure",
        owner="The particle FP step measures total mass as `sum(density) * dV` with dV the grid volume element, and the KDE normalisation divides by it so the density on the grid integrates to 1 (`FPParticleSolver._compute_total_mass`, the scalar-spacing branch; consumed by `_normalize_density` at :823 and by the ",
        verify="FPParticleSolver(MFGProblem(geometry=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1)), Nt=4, T=0.2, sigma=1.0, components=MFGComponents(m_initial=lambda x: np.exp(-10 * (np.asarray(x) - 0.5) ** 2).squeeze(), u_terminal=lambda x: 0.0, hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)))), num_particles=10)._compute_total_mass(np.ones(4), 0.25) == 4.0",
    ),
    Mutation(
        name="picard_criterion_reads_as_or",
        path="mfgarchon/alg/numerical/coupling/fixed_point_utils.py",
        old="    if max_rel_err < tol_picard and max_abs_err < tol_picard:",
        new="    if max_rel_err < tol_picard or max_abs_err < tol_picard:  # MUTATED: conjunction -> disjunction",
        owner="Picard convergence requires BOTH the relative AND the absolute L2 error below tolerance. Owner: check_convergence_criteria, fixed_point_utils.py:192-229, whose docstring states 'Convergence criteria (both must be satisfied)'. Single-sourced by three call sites routing to it: fixed_point_iterator.py:",
        verify="check_convergence_criteria(1e-9, 1e-9, 1.0, 1.0, 1e-6)[0]",
    ),
    Mutation(
        name="newton_residual_grid_scaling_dropped",
        path="mfgarchon/alg/numerical/hjb_solvers/base_hjb.py",
        old="    return float(np.linalg.norm(residual) * np.sqrt(dx))",
        new="    return float(np.linalg.norm(residual))  # MUTATED: sqrt(dx) grid scaling dropped",
        owner="The HJB Newton residual is the grid-scaled discrete L2 norm ||F||_2 * sqrt(dx). Owner: hjb_residual_norm, base_hjb.py:1362, whose docstring says verbatim 'The one owner of \"how large is this HJB residual\".' and 'It is a function rather than an inline expression because the ``sqrt(dx)`` is load-beari",
        verify="hjb_residual_norm(np.array([1.0, 0.0]), 0.25) == 1.0",
    ),
    Mutation(
        name="godunov_branch_swap",
        path="mfgarchon/operators/stencils/finite_difference.py",
        old="    return xp.where(grad_central >= 0, grad_backward, grad_forward)",
        new="    return xp.where(grad_central >= 0, grad_forward, grad_backward)  # MUTATED: Godunov branches swapped",
        owner="Godunov upwind takes the BACKWARD difference where the central gradient is >= 0 -- the library's one statement of the upwind selection rule, consumed by the HJB residual, the HJB Jacobian, GradientOperator and AdvectionOperator (#1896 items 3-4 turn on it)",
        verify="float(gradient_upwind(np.array([0.0, 1.0, 3.0]), axis=0, h=1.0)[1]) == 2.0",
    ),
    Mutation(
        name="upwind_jacobian_tiebreak_inverted",
        path="mfgarchon/alg/numerical/hjb_solvers/base_hjb.py",
        old="    took_backward = np.where(value_decides, np.abs(g_up - backward) <= np.abs(g_up - forward), g_c >= 0)",
        new="    took_backward = np.where(value_decides, np.abs(g_up - backward) <= np.abs(g_up - forward), g_c < 0)  # MUTATED: locally-linear tie-break inverted",
        owner="on rows where forward and backward agree in VALUE, the upwind Jacobian's row is decided by sign(central), matching the residual's own selector -- the one place #1896 left the rule restated rather than measured (#1896 item 3)",
        verify="float(_advection_bands(np.zeros(5), 0.25, None, 0.0, True, _bc_laplacian_bands(5, 0.25, None, 0.0))[1][2]) == -4.0",
    ),
    Mutation(
        name="fv_upwind_donor_cell_swapped",
        path="mfgarchon/operators/differential/advection.py",
        old="            flux = np.where(v_face >= 0.0, v_face * md[..., :-1], v_face * md[..., 1:])",
        new="            flux = np.where(v_face >= 0.0, v_face * md[..., 1:], v_face * md[..., :-1])  # MUTATED: donor cell swapped",
        owner="the conservative FV upwind flux takes the UPSTREAM cell: F_{i+1/2} = v_{i+1/2} m_i when v_{i+1/2} >= 0, else v_{i+1/2} m_{i+1} (#1184 / #1428)",
        verify="float(AdvectionOperator(velocity_field=np.ones((1, 3)), spacings=[1.0], field_shape=(3,), scheme='upwind', form='divergence', bc=no_flux_bc(dimension=1), mass_conservative=True)(np.array([1.0, 2.0, 3.0]))[0]) == 2.0",
    ),
    Mutation(
        name="periodic_wrap_endpoint_inverted",
        path="mfgarchon/geometry/boundary/types.py",
        old="    return 1 if convention is PeriodicGridConvention.ENDPOINT_INCLUSIVE else 0",
        new="    return 0 if convention is PeriodicGridConvention.ENDPOINT_INCLUSIVE else 1  # MUTATED: wrap convention swapped",
        owner='how many trailing nodes of a periodic axis repeat a node the array already holds -- the ONE number both the ghost skip and the modular span derive from (#1822). types.py:126-130 states the ownership verbatim: "Every periodic wrap in the package is one of two expressions of this single number, which ',
        verify="pad_array_with_ghosts(np.array([1.0, 2.0, 3.0]), periodic_bc(dimension=1), ghost_depth=1)[0] == 2.0",
    ),
    Mutation(
        name="bc_uniform_dispatch_reads_as_mixed",
        path="mfgarchon/geometry/boundary/applicator_fdm.py",
        old="        if bc.is_uniform:",
        new="        if False:  # MUTATED: uniform BC reads as mixed -- every BC takes the per-face path",
        owner="PreallocatedGhostBuffer.update_ghosts routes a uniform BC to the single-segment path and a mixed BC to the per-face path (#577 Phase 3 for the mixed rewrite, #1255 (C) for the alpha/beta forwarding the uniform branch carries). The two paths are NOT equivalent: the uniform branch applies the inhomoge",
        verify="pad_array_with_ghosts(np.array([1.0, 2.0, 3.0]), neumann_bc(dimension=1, value=2.0), ghost_depth=1, spacing=0.05)[0] == 1.0",
    ),
    Mutation(
        name="neumann_low_wall_flux_sign",
        path="mfgarchon/geometry/boundary/applicator_fdm.py",
        old="                        buf[tuple(lo_ghost)] += dx * v  # Issue #1262: was -= (du/dx sign), now += (du/dn sign)",
        new="                        buf[tuple(lo_ghost)] -= dx * v  # MUTATED: low wall reads du/dx instead of du/dn",
        owner='inhomogeneous Neumann is prescribed on the OUTWARD normal, so at the low wall (outward normal -x) du/dx = -v and ghost = interior + dx*v, the same expression as the high wall (#1262). The line\'s own comment states the convention it replaced: "Issue #1262: was -= (du/dx sign), now += (du/dn sign)".',
        verify="pad_array_with_ghosts(np.array([1.0, 2.0, 3.0]), neumann_bc(dimension=1, value=2.0), ghost_depth=1, spacing=0.05)[0] == 0.9",
    ),
    Mutation(
        name="terminal_condition_pinned_at_initial_index",
        path="mfgarchon/alg/numerical/coupling/fixed_point_utils.py",
        old="    U[-1] = U_terminal",
        new="    U[0] = U_terminal  # MUTATED: terminal condition re-imposed at t=0 instead of t=T",
        owner="u is boundary-data at the TERMINAL time and m at the INITIAL one: preserve_terminal_condition re-imposes u(T)=g on the LAST time row after damping/Anderson. Single owner, three caller families -- fixed_point_iterator.py:644 and :724 (Picard), fictitious_play.py:432, block_iterators.py:583. No issue ",
        verify="preserve_terminal_condition(np.zeros((3, 4)), np.full(4, 7.0))[0, 0] == 7.0",
    ),
    Mutation(
        name="hjb_marches_forward_in_time",
        path="mfgarchon/alg/numerical/hjb_solvers/base_hjb.py",
        old="    for n_idx_hjb in range(Nt - 2, -1, -1):  # Solves for U_solution_this_picard_iter at t_idx_n = n_idx_hjb",
        new="    for n_idx_hjb in range(0, Nt - 1):  # MUTATED: HJB marched forward from t=0, not backward from T",
        owner='The HJB marches BACKWARD in time, from the terminal row down to t=0, while the FP marches forward -- the adjoint pairing. Single site: base_hjb.py:1780, the only time loop in solve_hjb_system_backward. The M-indexing that rides on it is documented in the same loop body at base_hjb.py:1790: "# BUG #7',
        verify="float(solve_hjb_system_backward(np.ones((3, 6)), np.full(6, 3.0), np.zeros((3, 6)), MFGProblem(model=Model(hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0), coupling=lambda m: 0.0 * m, coupling_dm=lambda m: 0.0), sigma=0.3), domain=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[6], boundary_conditions=no_flux_bc(dimension=1)), conditions=Conditions(u_terminal=lambda x: 3.0 + 0.0 * x, m_initial=lambda x: 1.0 + 0.0 * x, T=0.1), Nt=2))[0, 0]) < 1.0",
    ),
    Mutation(
        name="fp_initial_condition_written_at_final_index",
        path="mfgarchon/alg/numerical/fp_solvers/fp_fdm_time_stepping.py",
        old="    M_solution[0] = m_initial_condition.copy()",
        new="    M_solution[-1] = m_initial_condition.copy()  # MUTATED: forward march anchored at t=T, not t=0",
        owner='The FP marches FORWARD from t=0 with m_0 written at time row 0 -- the initial-condition half of the HJB/FP boundary-data pair. Single site in the live FDM FP time loop, fp_fdm_time_stepping.py:797, whose own docstring states the convention at :681: "- Forward time evolution: k=0 -> Nt-1". No issue n',
        verify="float(np.sum(solve_fp_nd_full_system(np.ones(6), np.zeros((3, 6)), MFGProblem(model=Model(hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0), coupling=lambda m: 0.0 * m, coupling_dm=lambda m: 0.0), sigma=0.3), domain=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[6], boundary_conditions=no_flux_bc(dimension=1)), conditions=Conditions(u_terminal=lambda x: 3.0 + 0.0 * x, m_initial=lambda x: 1.0 + 0.0 * x, T=0.1), Nt=2))[0])) == 0.0",
    ),
]

# Node IDs, not the first whitespace-delimited token. A parametrisation label may contain
# spaces -- `test_x[a-V+f(m), lambda=2]` -- and `\S+?` cut it at the comma, so two committed
# killer IDs were already truncated. Harmless while both sides truncated identically; not
# harmless once these strings are compared as identities, because three params sharing a
# prefix collapse into one set member. Found by review (#1903); the two affected entries were
# repaired mechanically (each truncation resolved to exactly one collected node ID).
_FAILED = re.compile(r"^(?:FAILED|ERROR) (.+?)(?: - |$)", re.MULTILINE)


@dataclass
class Run:
    failed: set[str] = field(default_factory=set)
    returncode: int = 0
    collected: int = 0
    seconds: float = 0.0
    output: str = ""


def _pytest(paths: list[str], timeout: int = 3600) -> Run:
    import time

    t0 = time.perf_counter()
    proc = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "pytest",
            *paths,
            "-n",
            "auto",
            "-q",
            "--color=no",
            "-p",
            "no:cacheprovider",
            "-m",
            MARKERS,
            f"--ignore={SELF_TESTS}",
            "--timeout=900",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**_env()},
    )
    return Run(
        failed=set(_FAILED.findall(proc.stdout)),
        returncode=proc.returncode,
        collected=_collected(proc.stdout),
        seconds=round(time.perf_counter() - t0, 1),
        output=proc.stdout,
    )


def _failure_excerpt(output: str, limit: int = 120) -> str:
    """The part of a pytest run that says WHY it failed, bounded.

    Both refusal paths below must print this. A weekly red often does not reproduce
    locally, and then the runner's own output is the only evidence there is.
    """
    banner = re.search(r"^=+ FAILURES =+$", output, re.MULTILINE)
    excerpt = output[banner.start() :] if banner else output
    lines = excerpt.splitlines()
    if len(lines) <= limit:
        return "\n".join(lines)
    # Keep the tail: with -q the short summary lands last, and it names every failure
    # even when a single traceback is long enough to push the others out.
    return "\n".join([f"... {len(lines) - limit} earlier lines omitted ...", *lines[-limit:]])


_COLLECTED = re.compile(r"(\d+) (?:passed|failed|error)")


def _collected(stdout: str) -> int:
    """Tests that actually ran. Zero means the run did not happen, not that it passed."""
    counts = re.findall(r"(\d+) (?:passed|failed|xfailed|xpassed|skipped)", stdout)
    return sum(int(c) for c in counts)


def _env() -> dict:
    import os

    return {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}


def _assert_mutations_restored(selected: list[Mutation]) -> None:
    """The end-of-run guarantee: every file this run mutated is back as it was.

    NOT a whole-tree check. That is right at startup -- the run must begin from a state
    it can prove it restored -- but wrong at the end, where the script has by then
    written its own `--json` output into the repo. Implemented as a tree check, it
    refused to write the baseline in the same invocation that produced the matrix, and
    the refusal message blamed the operator for a file the script itself had created.
    """
    paths = sorted({m.path for m in selected})
    out = subprocess.run(
        ["git", "status", "--porcelain", "--", *paths], cwd=REPO, capture_output=True, text=True
    ).stdout
    dirty = [ln for ln in out.splitlines() if not ln.startswith("??")]
    if dirty:
        sys.exit(
            "MUTATION NOT RESTORED -- the tree still carries a perturbation. Recover with "
            "`git checkout --` on the paths below before doing anything else.\n" + "\n".join(dirty)
        )


def _assert_clean_tree() -> None:
    """No MODIFIED tracked files.

    Untracked files are ignored on purpose: the restore only rewrites the tracked files
    it mutated, so an untracked file cannot be clobbered by it and its presence says
    nothing about whether the restore worked. Blocking on `??` would only train the
    operator to reach for --force.
    """
    out = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True).stdout
    modified = [ln for ln in out.splitlines() if not ln.startswith("??")]
    if modified:
        sys.exit(
            "Refusing to run: tracked files are modified. This script edits the checkout in "
            "place (see the #1677 note in the module docstring) and restores what it wrote, "
            "so it must start from a state it can prove it restored.\n" + "\n".join(modified)
        )


def _assert_import_is_the_mutated_tree() -> None:
    """#1677's control: prove the process under measurement imports what we mutate."""
    proc = subprocess.run(
        [sys.executable, "-B", "-c", "import mfgarchon, pathlib; print(pathlib.Path(mfgarchon.__file__).resolve())"],
        cwd=REPO,
        capture_output=True,
        text=True,
        env=_env(),
    )
    imported = Path(proc.stdout.strip()) if proc.stdout.strip() else None
    expected = REPO / "mfgarchon" / "__init__.py"
    if imported != expected:
        sys.exit(
            f"Refusing to run: `import mfgarchon` resolves to {imported}, not {expected}. "
            f"Mutations applied here would not reach the code under test -- every mutation "
            f"would report zero kills and read as 'nothing discriminates' (Issue #1677)."
        )


_VERIFY_PRELUDE = """
import numpy as np
from mfgarchon.utils.pde_coefficients import diffusion_from_volatility, fp_drift_coefficient
from mfgarchon.geometry.boundary.bc_utils import bc_type_to_geometric_operation
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry.boundary import neumann_bc
from mfgarchon.geometry.boundary.applicator_fdm import pad_array_with_ghosts
from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon import MFGProblem
from mfgarchon.alg.numerical.coupling.fixed_point_utils import check_convergence_criteria
from mfgarchon.alg.numerical.coupling.fixed_point_utils import preserve_terminal_condition
from mfgarchon.alg.numerical.fp_solvers.fp_fdm_time_stepping import solve_fp_nd_full_system
from mfgarchon.alg.numerical.fp_solvers.fp_particle import FPParticleSolver
from mfgarchon.alg.numerical.hjb_solvers.base_hjb import _advection_bands, _bc_laplacian_bands
from mfgarchon.alg.numerical.hjb_solvers.base_hjb import hjb_residual_norm
from mfgarchon.alg.numerical.hjb_solvers.base_hjb import solve_hjb_system_backward
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import dirichlet_bc, periodic_bc, robin_bc
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.geometry.boundary import periodic_bc   # _VERIFY_PRELUDE line 301 currently imports only `neumann_bc` from this module; it must become `from mfgarchon.geometry.boundary import neumann_bc, periodic_bc`
from mfgarchon.geometry.boundary.types import PeriodicGridConvention
from mfgarchon.operators.differential.advection import AdvectionOperator
from mfgarchon.operators.stencils.finite_difference import gradient_upwind
from mfgarchon.types import NumericalScheme

def _stub_problem(control_cost):
    class P:
        hamiltonian_class = SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=control_cost),
            coupling=lambda m: m, coupling_dm=lambda m: 1.0)
    return P()
"""


def _mutation_is_live(mut: Mutation) -> bool:
    """Is the perturbation observable from outside? The control, on the thing in doubt.

    Evaluated against the mutated tree in a fresh interpreter. Only when this is true
    does a kill count of zero mean "no test covers this convention" rather than "the
    mutation never ran".
    """
    proc = subprocess.run(
        [sys.executable, "-B", "-c", f"{_VERIFY_PRELUDE}\nassert ({mut.verify}), {mut.verify!r}"],
        cwd=REPO,
        capture_output=True,
        text=True,
        env=_env(),
    )
    if proc.returncode != 0:
        tail = (proc.stderr.strip().splitlines() or ["(no stderr)"])[-1]
        print(f"  verify FAILED: {mut.verify}\n    {tail}", flush=True)
    return proc.returncode == 0


def apply_mutation(mut: Mutation, backups: dict[str, str]) -> None:
    target = REPO / mut.path
    text = target.read_text()
    occurrences = text.count(mut.old)
    if occurrences != 1:
        raise SystemExit(
            f"mutation {mut.name!r}: expected its anchor exactly once in {mut.path}, found "
            f"{occurrences}. The source moved; fix the mutation rather than skipping it -- a "
            f"silently unapplied mutation reports zero kills."
        )
    backups.setdefault(mut.path, text)
    target.write_text(text.replace(mut.old, mut.new))


def restore(backups: dict[str, str]) -> None:
    for rel, text in backups.items():
        (REPO / rel).write_text(text)
    backups.clear()


def _head_sha() -> str:
    """Stamped by the same call that produces the counts, not looked up later."""
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True)
    return out.stdout.strip() or "unknown"


def _write_baseline(path: Path, results: dict, *, paths: list[str], collected: int) -> None:
    payload = {
        "_comment": (
            "Kill counts per mutated convention. --check-baseline fails when a count DROPS "
            "(discrimination lost) and when one RISES (record the gain in the same change). "
            "Counts, not test names: this population cannot be selected by name -- the "
            "agreement-shaped patterns give 51, 114 or 156 depending on the regex."
        ),
        "_measured_at": {
            "commit": _head_sha(),
            "paths": paths,
            "markers": MARKERS,
            "collected": collected,
            "excluded": SELF_TESTS,
        },
        "mutations": {
            name: {
                "owner": res["owner"],
                "status": res["status"],
                "kill_count": res["kill_count"],
            }
            for name, res in sorted(results.items())
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def compare_to_baseline(results: dict, baseline: dict, matrix: dict | None = None) -> list[str]:
    """Every way discrimination can degrade. Empty list means it did not.

    Two ratchets, both unforgeable by renaming -- which matters, because the
    population this measures cannot be selected by name at all:

    - **Per-mutation kill counts must not drop.** A convention that 129 tests noticed
      and 120 notice now has lost coverage, whatever the test names are. Deleting a
      discriminating test trips this, which is correct: it is a real loss.
    - **The UNCOVERED set must not grow.** A convention going from watched to
      unwatched is the defect this tool exists to find.
    - **No killer may silently leave**, which a count cannot see. `drift_coefficient_2x`
      held 19 -> 19 across a one-for-one swap of killers: one test stopped noticing the
      convention and a different one started, and the gate reported no change. Measured
      2026-08-12 (#1901), where it ranked as the reason a kill *count* is a weaker
      instrument than it reads. The killer sets live in the sibling kill matrix, which
      until now was committed as evidence that nothing read.

    Improvements trip it too, and must be recorded in the same change -- otherwise the
    next baseline encodes the gain as if it had always held, and the ratchet loses the
    ability to say when anything got better. Same rule as the capability matrix.
    """
    problems = []
    base_muts = baseline["mutations"]
    for name, was in sorted(base_muts.items()):
        now = results.get(name)
        if now is None:
            problems.append(f"  {name}: mutation DISAPPEARED (baseline killed {was['kill_count']})")
            continue
        if now["status"] == "INEFFECTIVE":
            problems.append(f"  {name}: became INEFFECTIVE -- the mutation no longer applies; fix it")
            continue
        if now["kill_count"] < was["kill_count"]:
            problems.append(f"  {name}: {was['kill_count']} -> {now['kill_count']} killed  [DISCRIMINATION LOST]")
        elif now["kill_count"] > was["kill_count"]:
            problems.append(
                f"  {name}: {was['kill_count']} -> {now['kill_count']} killed  [IMPROVED -- record it in the baseline]"
            )
    if matrix is not None:
        problems.extend(_compare_killers(results, matrix))
    for name in sorted(set(results) - set(base_muts)):
        problems.append(f"  {name}: NEW mutation, not in baseline")
    return problems


def _compare_killers(results: dict, matrix: dict) -> list[str]:
    """Which tests kill each mutation, not how many -- the half a count cannot express.

    A departure is a regression even when the count holds: that test stopped noticing the
    convention, and whatever replaced it is a different assertion about a different thing.
    Renames land here too, and that is the intended cost: the fix is to regenerate the
    baseline in the same commit, exactly as a count change requires. Silence would be the
    alternative, and silence is what this whole tool exists to remove.

    Two details are deliberately unpinned because they change no verdict, only how it reads:
    the held/moved word and the "+N more" truncation.

    ~~Everything that decides pass or fail is mutation-covered~~ [CORRECTED 2026-08-13] --
    that sentence was written from inside this function and was false about the one thing
    outside it: whether `main()` hands the matrix over at all. Dropping that argument left
    all 41 tests green while `--check-baseline` printed "counts and killer sets" over a
    comparison that never saw a killer set. Found by independent review (#1903), now pinned
    by `test_without_a_matrix_the_comparison_itself_is_silent`. The claim is a statement
    about the test file, so read it there: `tests/unit/test_discrimination_ratchet.py`.
    """
    problems: list[str] = []
    base = matrix.get("mutations", {})
    compared = 0
    for name, was in sorted(base.items()):
        now = results.get(name)
        if now is None or now.get("status") == "INEFFECTIVE":
            continue  # already reported by the count ratchet, and its zeros mean nothing
        before = set(was.get("killed", ()))
        # `main()` writes this under "killed" (see the results dict it builds). Reading
        # "failed" here made `after` empty on every real run, so all 220 baseline killers
        # reported as departed on an unchanged tree -- and the new tests could not see it,
        # because their fixture fabricated "failed" too. Found by review (#1903).
        after = set(now["killed"])
        compared += 1
        departed = sorted(before - after)
        if departed:
            shown = ", ".join(t.rsplit("::", 1)[-1] for t in departed[:3])
            more = f" (+{len(departed) - 3} more)" if len(departed) > 3 else ""
            problems.append(
                f"  {name}: {len(departed)} test(s) STOPPED killing it while the count "
                f"{'held' if len(before) == len(after) else 'moved'}  [DISCRIMINATION LOST]: {shown}{more}"
            )
        arrived = sorted(after - before)
        if arrived:
            shown = ", ".join(t.rsplit("::", 1)[-1] for t in arrived[:3])
            more = f" (+{len(arrived) - 3} more)" if len(arrived) > 3 else ""
            problems.append(
                f"  {name}: {len(arrived)} new killer(s)  "
                f"[{'a rename? regenerate' if departed else 'IMPROVED -- record it in the baseline'}]: {shown}{more}"
            )
    unchecked = sorted(set(results) - set(base))
    if unchecked:
        problems.append(
            f"  killer sets compared for {compared} of {len(results)} mutation(s); "
            f"NO matrix entry for: {', '.join(unchecked)} -- regenerate the matrix alongside the baseline"
        )
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", action="append", help="Run only these mutations (repeatable)")
    parser.add_argument("--paths", default="tests", help="Test paths to run (default: tests)")
    parser.add_argument("--json", metavar="FILE", help="Write the full kill matrix to FILE")
    parser.add_argument("--write-baseline", metavar="FILE", help="Write the ratchet baseline to FILE")
    parser.add_argument("--check-baseline", metavar="FILE", help="Fail if discrimination degraded vs FILE")
    args = parser.parse_args()

    _assert_clean_tree()
    _assert_import_is_the_mutated_tree()

    selected = [m for m in MUTATIONS if not args.only or m.name in args.only]
    if not selected:
        sys.exit(f"No mutation matched {args.only}. Known: {[m.name for m in MUTATIONS]}")

    paths = args.paths.split()
    print(f"Baseline: pytest {' '.join(paths)} (excluding {SELF_TESTS}) ...", flush=True)
    base = _pytest(paths)
    if base.failed:
        sys.exit(
            f"Refusing to run: {len(base.failed)} tests already fail before any mutation, so a "
            f"kill could not be attributed.\n  "
            + "\n  ".join(sorted(base.failed)[:10])
            + f"\n\n--- baseline pytest output ---\n{_failure_excerpt(base.output)}"
        )
    # pytest exits 5 on "no tests collected" and 2/3/4 on usage or internal errors, and in
    # every one of those the FAILED set is empty. Without this, a --paths typo produces a
    # clean baseline and then six UNCOVERED "findings" -- the exact ambiguity the
    # three-way verdict removes, entering through the door `verify` does not watch:
    # `verify` proves the MUTATION took effect, nothing proved that PYTEST RAN.
    if base.returncode != 0 or base.collected == 0:
        sys.exit(
            f"Refusing to run: the baseline pytest exited {base.returncode} having run "
            f"{base.collected} tests. Nothing below would be a measurement."
            f"\n\n--- baseline pytest output ---\n{_failure_excerpt(base.output)}"
        )
    print(f"  clean, {base.collected} ran, {base.seconds}s\n", flush=True)

    results: dict[str, dict] = {}
    backups: dict[str, str] = {}
    try:
        for mut in selected:
            print(f"[{mut.name}] {mut.owner}", flush=True)
            apply_mutation(mut, backups)
            try:
                live = _mutation_is_live(mut)
                run = _pytest(paths) if live else Run()
            finally:
                restore(backups)

            shrank = live and run.collected < base.collected * 0.99
            if not live:
                status, note = "INEFFECTIVE", "  <-- mutation not observable; its zeros mean nothing"
            elif run.returncode not in (0, 1) or shrank:
                # A mutation that breaks collection silently shrinks the population, so
                # every survivor "survived" a run it was never in.
                status = "INEFFECTIVE"
                note = f"  <-- pytest exited {run.returncode} having run {run.collected} (baseline {base.collected})"
            elif run.failed:
                status, note = "ok", ""
            else:
                status, note = "UNCOVERED", "  <-- mutation IS live and no test noticed"
            results[mut.name] = {
                "owner": mut.owner,
                "status": status,
                "killed": sorted(run.failed),
                "kill_count": len(run.failed),
                "seconds": run.seconds,
                "verify": mut.verify,
            }
            print(f"  killed {len(run.failed):4d}  [{status}]  {run.seconds}s{note}\n", flush=True)
    finally:
        restore(backups)

    effective = {k: v for k, v in results.items() if v["status"] == "ok"}
    uncovered = sorted(k for k, v in results.items() if v["status"] == "UNCOVERED")
    ineffective = sorted(k for k, v in results.items() if v["status"] == "INEFFECTIVE")

    killed_by: dict[str, list[str]] = {}
    for name, res in effective.items():
        for test in res["killed"]:
            killed_by.setdefault(test, []).append(name)

    print("=" * 78)
    print(f"{len(effective)} of {len(results)} mutations were live and killed at least one test.")
    if uncovered:
        print(f"\nUNCOVERED -- live, and no test noticed: {', '.join(uncovered)}")
        for name in uncovered:
            print(f"  {name}: {results[name]['owner']}")
        print("  These are findings, not harness faults: the convention is single-sourced")
        print("  and nothing in the selected paths asserts anything about it.")
    if ineffective:
        print(f"\nINEFFECTIVE, excluded from the verdict: {', '.join(ineffective)}")
        print("  The mutation was not observable, so its zeros prove nothing. Fix the")
        print("  mutation or its verify expression before reading them.")
    print(f"\n{len(killed_by)} distinct tests were killed by at least one effective mutation.")
    for name, res in sorted(effective.items()):
        print(f"  {name:<30} {res['kill_count']:>4} killed")

    payload = {
        "_measured_at": {
            "commit": _head_sha(),
            "paths": paths,
            "markers": MARKERS,
            "collected": base.collected,
            "excluded": SELF_TESTS,
        },
        "_selection_regex_for_agreement_shaped": AGREEMENT_SHAPED,
        "uncovered": uncovered,
        "markers": MARKERS,
        "paths": paths,
        "baseline_seconds": base.seconds,
        "mutations": results,
        "killed_by": killed_by,
    }
    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"\nKill matrix written to {args.json}")

    _assert_mutations_restored(selected)
    print("\nEvery mutated file verified restored.")

    if args.write_baseline:
        _write_baseline(Path(args.write_baseline), results, paths=paths, collected=base.collected)
        print(f"Baseline written to {args.write_baseline}")
        sys.exit(0)

    if args.check_baseline:
        baseline_path = Path(args.check_baseline)
        baseline = json.loads(baseline_path.read_text())
        # The killer sets are the half a count cannot express, and they live beside the
        # baseline rather than inside it -- the matrix is the richer artifact and a test
        # already pins the two to the same run. Absent, the gate degrades to counts and
        # SAYS so, because a silently weaker gate is the failure mode this tool is for.
        matrix_path = baseline_path.parent / "discrimination_killmatrix.json"
        matrix = json.loads(matrix_path.read_text()) if matrix_path.exists() else None
        if matrix is None:
            print(
                f"CANNOT MEASURE: {matrix_path.name} is absent, so killer sets cannot be compared "
                f"and this gate would silently degrade to counts. Exit 2 is 'could not measure', "
                f"distinct from 0 (matches) and 1 (degraded) -- a green run with half the gate off "
                f"is invisible in a three-hour log."
            )
            sys.exit(2)
        problems = compare_to_baseline(results, baseline, matrix)
        if problems:
            print("\nDiscrimination baseline mismatch:")
            print("\n".join(problems))
            print(
                "\nIf intended, regenerate BOTH in one sweep and commit them together:\n"
                "  python scripts/test_discrimination.py --json scripts/discrimination_killmatrix.json "
                "--write-baseline scripts/discrimination_baseline.json\n"
                "--write-baseline alone leaves the matrix at the old run, which reddens "
                "test_the_kill_matrix_is_committed_beside_the_baseline and costs a second sweep."
            )
            sys.exit(1)
        print(f"Discrimination matches baseline ({len(baseline['mutations'])} mutations, counts and killer sets).")
        sys.exit(0)

    sys.exit(0 if effective else 1)


if __name__ == "__main__":
    main()
