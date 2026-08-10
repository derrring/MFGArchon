#!/usr/bin/env python3
"""Execute the public solve surface and record what it can actually do.

Counts of tests, issues, or fail-fast violations all move for reasons unrelated to
whether the library solves anything. This records the other quantity: for a fixed set
of configurations driven through the public API, does the result satisfy an oracle
that lives outside the code under test.

Three properties make a cell hard to satisfy the cheap way:

- The oracle is a number (mass drift, relative L2 against a second discretisation),
  not an exception. ``pytest.raises`` cannot turn a cell green.
- Where a cell claims independence, the second reading comes from a genuinely
  different discretisation, so consolidating two paths does not make it tautological
  the way an agreement test does. Independence is claimed **per axis, in the cell's
  own docstring**, not globally -- ``fvm_vs_fdm`` is independent on M and not on U,
  because both arms share ``HJBFDMSolver``. Read the cell before relying on it.
- ``--check-baseline`` fails in BOTH directions. A cell that recovers fails the check
  until the baseline records the recovery, so improvements cannot land silently and a
  baseline cannot be lowered without doing the work. Same structure as
  ``check_fail_fast.py`` / ``fail_fast_baseline.json``.

A cell status is one of:

  PASS         solve returned and the oracle held
  FAIL         solve returned and the oracle was violated (measured value recorded)
  UNSUPPORTED  the path refused to run (exception type + message head recorded)
  ERROR        the harness itself broke

ERROR is not a status like the others: it is never written to a baseline and never
compared against one. Any ERROR exits 2 before the baseline is read. Treating it as
comparable was a real hole -- an ERROR baselined as ERROR matched, so a harness broken
during a regeneration would have stayed green forever, the matrix silently no longer
measuring anything and reporting success for it.

What ``--check-baseline`` compares is STATUS, not the recorded artifacts. The artifact
blocks in the baseline are a record for a human diffing a PR; a cell can degrade well
within its own tolerance and the gate stays green.

An artifact may carry ``library_said``: what the library warned about while that cell ran,
folded by category and message with numbers collapsed, and absent entirely from a cell that
warned about nothing. It records capability the oracles cannot see -- `fdm_upwind` passes its
mass oracle while saying 39 times that the value function it returned is not a root of the
discrete HJB. Warnings that describe the machine rather than the solve are excluded: import
and deprecation by category, the JAX-autodiff fallback by message. That exclusion is not
tidiness. An entry that appears only where an optional package is installed makes this
committed baseline machine-dependent, and a regeneration elsewhere both fakes a diff and
drops the ``intended`` note of every cell whose artifact moved.

Every cell but one drives the public API. ``regime_switching/non_negativity`` reaches
through the deep module path because ``RegimeSwitchingIterator`` is exported from no
package ``__init__`` -- recorded in that cell rather than treated as a reason to leave
a measured defect (#1681) unwatched.

Cells deliberately NOT in this set, so the omission is on the record rather than
implied absent:

- 2-D meshless-Galerkin + Nitsche refinement (#1679). Needs a refinement sweep, so it
  is minutes, not seconds; it belongs in a nightly-tier matrix.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import os
import re
import sys
import time
import traceback
import warnings

import numpy as np

# No blanket `warnings.filterwarnings("ignore")` here. There was one until #1879, and it
# silenced the library while this file measured it -- including the warning base_hjb raises
# 39 times in the fdm_upwind cell, saying in as many words that the value function it
# returned "is not a root of the discrete HJB, and the outer iteration will consume it as
# if it were". That is a statement about capability, made by the code under test, to the
# instrument that exists to record capability, and it went to /dev/null for the life of
# this file. What replaces it is `_warning_summary`: each cell runs under `catch_warnings`,
# and what the library said lands in the artifact as `library_said`, where a baseline diff
# can show it.

# Set by --self-test. Applied to every density this module measures, so a mass oracle
# that does not read the density it claims to read stays green and the self-test
# reports the cell as inert.
#
# The mutation is a RAMP along the time axis, not a constant factor. A constant factor
# was tried first and left all three drift cells PASS -- correctly, because
# max|mass(t) - mass(0)| is invariant under a uniform rescaling of every slice. A
# control has to break the invariant the oracle measures; picking one that cannot is
# the same error as a test that passes either way.
_DENSITY_MUTATION: float | None = None


def _apply_mutation(M: np.ndarray) -> np.ndarray:
    """Inject ``_DENSITY_MUTATION`` relative drift, linear in time, zero at t=0."""
    if _DENSITY_MUTATION is None:
        return M
    nt = M.shape[0]
    ramp = 1.0 + _DENSITY_MUTATION * (np.arange(nt) / max(nt - 1, 1))
    return M * ramp.reshape((-1,) + (1,) * (M.ndim - 1))


_STATUSES = frozenset({"PASS", "FAIL", "UNSUPPORTED", "ERROR"})

MASS_RTOL = 1e-9  # matches tests/integration/test_three_mode_api.py
AGREEMENT_RTOL = 0.07  # matches tests/integration/test_fvm_hjb_coupling.py:175


# --------------------------------------------------------------------------------
# Shared problem builders. Both mirror an existing test fixture rather than inventing
# a configuration, so a cell here and the corresponding test disagree only if the
# product changed.
# --------------------------------------------------------------------------------


def _smoke_problem():
    """The tests/integration/test_three_mode_api.py fixture."""
    from mfgarchon import MFGProblem
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
    from mfgarchon.core.mfg_components import MFGComponents
    from mfgarchon.geometry import TensorProductGrid
    from mfgarchon.geometry.boundary import no_flux_bc

    return MFGProblem(
        geometry=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1)),
        Nt=10,
        T=1.0,
        components=MFGComponents(
            m_initial=lambda x: np.exp(-10 * (np.asarray(x) - 0.5) ** 2),
            u_terminal=lambda x: 0.0,
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0),
                coupling=lambda m: m,
                coupling_dm=lambda m: 1.0,
            ),
        ),
    )


# The 2-D smoke fixture's coupling scale, and why it is not 1.
#
# `coupling=lambda m: m` reads the density directly, so its strength at the crowd is set by the
# peak of the normalised density -- and a probability density on a 2-D grid is intrinsically
# peakier than the 1-D one carrying the same mass. Measured: 1.8180 in 1-D against 9.5495 in 2-D.
# With the same nominal coupling the 2-D cell therefore applies 5.3x the interaction its 1-D
# sibling does, and the cell has been reporting that difference as an effect of dimension.
#
# The scale below restores comparability by the one rule this file can state and check: make
# `f(m)` at the crowd equal. It is DERIVED, 1.8180 / 9.5495, not chosen -- picking a number
# because it turns cells green is lowering the baseline, which `--check-baseline` exists to stop.
# It is computed from the two fixtures rather than written down: a literal would have to be
# maintained against both of them, and the first version of it, `1.8180 / 9.5495`, was already
# wrong in the seventh digit against the real peaks. A constant with one derivation cannot drift,
# and a test asserting it equals its own derivation would be a tautology -- what
# `test_capability_2d_coupling_scale.py` pins instead is that the scale is APPLIED.
#
# What this deliberately does NOT cover: the aggregating regime. At scale 1.0 this fixture sits
# where `f` rewards density, Lasry-Lions monotonicity fails, uniqueness is not guaranteed and the
# Picard map is not a contraction -- so "can this configuration solve at all" has no answer to
# certify, and the cell was red for a reason no code change could address. That is a research
# question and it lives in #1865, not in a fixed-size instrument. It may earn a cell later, if the
# library gains an outer solver that makes the regime answerable.
def _smoke_problem_2d_at(scale: float):
    """The 2-D smoke problem at an explicit coupling scale. Only `_coupling_scale_2d` passes 1.0."""
    from mfgarchon import MFGProblem
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
    from mfgarchon.core.mfg_components import MFGComponents
    from mfgarchon.geometry import TensorProductGrid
    from mfgarchon.geometry.boundary import no_flux_bc

    return MFGProblem(
        geometry=TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[11, 11], boundary_conditions=no_flux_bc(dimension=2)
        ),
        Nt=6,
        T=0.2,
        sigma=0.4,
        components=MFGComponents(
            m_initial=lambda x: np.exp(-30 * np.sum((np.asarray(x) - 0.5) ** 2, axis=-1)),
            u_terminal=lambda x: 0.0,
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0),
                coupling=lambda m: scale * np.asarray(m, dtype=float),
                coupling_dm=lambda m: scale * np.ones_like(np.asarray(m, dtype=float)),
            ),
        ),
    )


@functools.cache
def _coupling_scale_2d() -> float:
    """peak of the 1-D crowd / peak of the 2-D crowd, from the fixtures themselves.

    `m_initial` is normalised independently of the Hamiltonian, so reading the 2-D peak off a
    scale-1.0 instance is not circular.
    """
    peak_1d = float(np.asarray(_smoke_problem().m_initial, dtype=float).max())
    peak_2d = float(np.asarray(_smoke_problem_2d_at(1.0).m_initial, dtype=float).max())
    return peak_1d / peak_2d


def _smoke_problem_2d():
    """The 1-D smoke fixture lifted to 2-D: same Gaussian family, same no-flux boundaries, and a
    coupling rescaled so its strength at the crowd matches.

    Honest about what is NOT shared, because the previous docstring claimed "unchanged in
    everything but dimension" and four things were:

    ==================  ====================  ====================
    quantity            1-D (`_smoke_problem`)  here
    ==================  ====================  ====================
    Gaussian            ``exp(-10 r^2)``      ``exp(-30 r^2)``
    ``T``               1.0                   0.2
    ``Nt``              10                    6
    ``sigma``           0.0                   0.4
    ==================  ====================  ====================

    Those four stay, and the reasons are worth stating rather than quietly fixing. The horizon and
    step count keep this cell inside its runtime budget, which a faithful lift does not -- measured,
    a full lift runs 3.2 s against this fixture's 3.9 s and still fails, so matching them buys
    honesty and no answer. `sigma` is the load-bearing one: the 1-D fixture runs at **sigma = 0**,
    pure transport, which is the degenerate regime where the Godunov residual is non-smooth and
    Newton can reach a non-viscosity branch (#1878). Lifting that into 2-D would make this cell a
    second instance of that problem rather than a test of dimension.

    So the cell certifies: this configuration, at a coupling strength comparable to its 1-D
    sibling's, solves and conserves mass. It does not certify the aggregating regime -- see
    `_COUPLING_SCALE_2D` above and #1865.
    """
    return _smoke_problem_2d_at(_coupling_scale_2d())


def _lq_problem_1d():
    """The tests/integration/test_fvm_hjb_coupling.py 1-D LQ fixture."""
    from mfgarchon import MFGProblem
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
    from mfgarchon.core.mfg_components import MFGComponents
    from mfgarchon.geometry import TensorProductGrid
    from mfgarchon.geometry.boundary import no_flux_bc

    coupling = 0.3
    return MFGProblem(
        geometry=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[25], boundary_conditions=no_flux_bc(dimension=1)),
        T=0.3,
        Nt=12,
        sigma=0.4,
        coupling_coefficient=coupling,
        components=MFGComponents(
            m_initial=lambda x: np.exp(-((np.asarray(x) - 0.4) ** 2) / (2 * 0.13**2)),
            u_terminal=lambda x: 0.2 * (np.asarray(x) - 0.6) ** 2,
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0 / coupling),
                coupling=lambda m: m,
                coupling_dm=lambda m: 1.0,
            ),
        ),
    )


def _density(result) -> np.ndarray:
    return _apply_mutation(np.asarray(result.M, dtype=float))


def _mass_drift(result, problem) -> dict:
    """Total mass per time slice, and its maximum excursion from t=0."""
    M = _density(result)
    bounds = problem.geometry.get_bounds()
    dx = (bounds[1][0] - bounds[0][0]) / (M.shape[1] - 1)
    mass = M.sum(axis=1) * dx
    return {
        "mass_t0": float(mass[0]),
        "mass_max": float(mass.max()),
        "max_drift": float(np.abs(mass - mass[0]).max()),
        "min_density": float(M.min()),
        "all_finite": bool(np.isfinite(M).all()),
    }


def _picard_verdict(result, prefix: str = "") -> dict:
    """What the coupled iteration said about itself, recorded beside the oracle (#1871).

    Deliberately NOT part of any cell's ``ok``. Measured when this was added: three of the
    five PASS cells run exactly to their iteration budget without converging --
    ``fdm_upwind`` and ``sl_linear`` at 5, ``sl_linear_2d`` at 3 -- so gating on it today
    turns three long-standing greens red. Two of those still do not converge at 100 sweeps,
    and for different reasons, which is why the gate is not a one-line follow-up:

    - ``fdm_upwind``: the value function's Picard error rises from 0.496 to 287, oscillates
      (13.5 at sweep 10, back to 235 at sweep 19) and ends at 2.3e-04. The density gets to
      9.5e-10 by sweep 85 but rises on 35 of 99 steps and ends 40x higher, at 3.9e-08.
    - ``sl_linear``: BOTH are above tolerance, so neither fix alone would turn it green.
      The density is the worse of the two and is not descending -- never below 0.357,
      ending at 8.2e-01, O(1) after 100 sweeps -- while the value function ends at 5.9e-02
      and is still coming down.

    ``sl_linear_2d`` is the one a budget fixes -- it converges at 35 sweeps in ~3s.

    A mass oracle cannot see any of this, and not because its tolerance is loose: mass
    conservation is a property of the FP time-stepping, which holds on whatever drift it is
    handed, converged or not. So this field is what makes the difference visible in the
    baseline and diffable across changes, before anyone argues about the gate. The
    non-convergence itself is #1873.

    ``picard_iterations`` is the count of sweeps performed, in both result types -- but the
    two reach it differently, and the asymmetry runs opposite to the one you would guess.
    ``RegimeSwitchingIterator`` exits early only on convergence, so a non-converged run has
    always executed the full budget and its count equals it by construction.
    ``FixedPointIterator`` has three non-converged breaks besides, so its count is the one
    that can fall short of the budget without having converged.
    """
    return {
        f"{prefix}picard_converged": bool(result.converged),
        f"{prefix}picard_iterations": int(result.iterations),
    }


def _rel_l2(a: np.ndarray, b: np.ndarray, dx: float) -> float:
    return float(np.sqrt(dx * np.sum((a - b) ** 2)) / np.sqrt(dx * np.sum(b**2)))


class HarnessError(Exception):
    """The measurement apparatus failed. Never the code under test.

    A cell has three phases and only the middle one may report UNSUPPORTED:

      construct  build the fixture      -- any failure here is mine
      solve      call the library       -- a refusal here is the measurement
      measure    compute the oracle     -- any failure here is mine

    Classifying by exception type instead was the wrong shape, and shipped two
    holes: a `ValueError` from a mistyped fixture argument, and an `IndexError`
    from `M.shape[1]` on a malformed result, both read as "the library does not
    support this". Phase, not type, is what separates the two.

    **What goes inside the wrapper: the fixture, never the object under test.**
    Constructing the thing being measured is part of the measurement, because its
    ``__init__`` runs the library's own validation -- ``RegimeSwitchingIterator``
    calls ``assert_paired_solver_sigma`` (#1603 / RFC #1574 C14), and refusing
    there is a capability statement, not a broken harness. Wrapping it would turn
    that refusal into an ERROR, and ERROR is never baselined, so the refusal could
    never be recorded at all. The fixtures -- geometry, components, Hamiltonian,
    the problem list -- are the harness's own choices and do go inside.
    """


def _construct(what: str, thunk):
    """Build a fixture. Any exception is a harness fault, not a library refusal.

    Takes a zero-argument callable, deliberately. An earlier signature was
    ``_construct(what, fn, *args, **kwargs)``, which does not do what it looks
    like: Python evaluates argument expressions in the CALLER's frame, so
    ``_construct("iterator", Cls, solvers=[Solver(p) for p in problems])`` runs
    every one of those constructors before ``_construct`` is entered, and their
    failures bypass the wrapper entirely. A thunk is the only form where the
    wrapper actually covers what it appears to cover.
    """
    try:
        return thunk()
    except Exception as exc:
        raise HarnessError(f"constructing {what}: {type(exc).__name__}: {exc}") from exc


def _measure(what: str, thunk):
    """Compute the verdict from a returned result. Any exception is a harness fault.

    The thunk returns the whole ``(status, artifact)`` pair, not just the artifact.
    Returning the artifact alone left the comparison against it -- ``art["worst"] <
    AGREEMENT_RTOL`` -- one line outside the wrapper, so a ``KeyError`` from a
    partial artifact still read as a library refusal.

    The return shape is CHECKED, not merely documented. A cell that goes back to
    returning a bare artifact puts its verdict outside the wrapper again, and that
    reverts silently -- the tests all pass thunks, so they hold under either shape.
    Making the old shape raise is the only version of this that cannot rot.
    """
    try:
        out = thunk()
    except Exception as exc:
        raise HarnessError(f"measuring {what}: {type(exc).__name__}: {exc}") from exc
    if not (isinstance(out, tuple) and len(out) == 2 and out[0] in _STATUSES):
        raise HarnessError(
            f"measuring {what}: oracle must return (status, artifact) with status in "
            f"{sorted(_STATUSES)}, got {type(out).__name__} {repr(out)[:80]}. "
            f"Returning the artifact alone puts the verdict outside this wrapper."
        )
    return out


# --------------------------------------------------------------------------------
# Cells
# --------------------------------------------------------------------------------


def _mass_conservation_cell(scheme_name: str):
    """Public problem.solve(scheme=...) must not move total mass."""

    def run():
        from mfgarchon.types import NumericalScheme

        problem = _construct("smoke problem", _smoke_problem)
        result = problem.solve(scheme=getattr(NumericalScheme, scheme_name), max_iterations=5, verbose=False)

        def verdict():
            art = _mass_drift(result, problem)
            art |= _picard_verdict(result)
            art["tolerance"] = MASS_RTOL * max(abs(art["mass_t0"]), 1.0)
            ok = art["all_finite"] and art["max_drift"] <= art["tolerance"]
            return ("PASS" if ok else "FAIL"), art

        return _measure("mass drift", verdict)

    return run


def _mass_conservation_2d_cell(scheme_name: str):
    """Same oracle as the 1-D cell, in 2-D (#1745).

    Every cell in this file was 1-D until now, so a scheme could conserve mass to
    2.2e-16 in one dimension and not run at all in two with the matrix reporting
    nothing. Measured when these were added: SL_LINEAR passes at 3.67e-16 while
    FDM_UPWIND, FDM_CENTERED and FVM_MUSCL all raise the same ConvergenceError from
    the same Newton solve -- they share `HJBFDMSolver`, and SL_LINEAR does not. One
    defect, three dead schemes.
    """

    def run():
        from mfgarchon.types import NumericalScheme

        problem = _construct("2-D smoke problem", _smoke_problem_2d)
        result = problem.solve(scheme=getattr(NumericalScheme, scheme_name), max_iterations=3, verbose=False)

        def verdict():
            M = _apply_mutation(np.asarray(result.M, dtype=float))
            dv = (1.0 / 10) ** 2  # 11 points per axis on the unit square
            mass = M.reshape(M.shape[0], -1).sum(axis=1) * dv
            art = {
                "mass_t0": float(mass[0]),
                "max_rel_drift": float(np.abs(mass - mass[0]).max() / abs(mass[0])),
                "min_density": float(M.min()),
                "all_finite": bool(np.isfinite(M).all()),
                "tolerance": 1e-9,
            }
            art |= _picard_verdict(result)
            ok = art["all_finite"] and art["min_density"] >= -1e-12 and art["max_rel_drift"] <= 1e-9
            return ("PASS" if ok else "FAIL"), art

        return _measure("2-D mass drift", verdict)

    return run


def _gfdm_rbf_cell():
    """HJBGFDMSolver(derivative_method='rbf') -- a declared constructor argument.

    The taylor sibling is constructed first and must succeed. Without it, a harness
    bug -- a renamed argument, a changed signature -- reports UNSUPPORTED and reads as
    the RBF defect. Same shape as the ``_Bare`` object in #1447, which made a
    failure-mode test pass without ever reaching the branch it named.
    """

    def run():
        from mfgarchon.alg.numerical.hjb_solvers import HJBGFDMSolver

        problem = _construct("smoke problem", _smoke_problem)
        pts = np.linspace(0.0, 1.0, 21).reshape(-1, 1)

        # Control: the same call with the supported method must construct, or the rbf
        # verdict below is meaningless. The CONTROL is a fixture, so it goes inside
        # _construct; the rbf construction one line below is the object under test and
        # deliberately does not.
        _construct(
            "gfdm taylor control",
            lambda: HJBGFDMSolver(problem=problem, collocation_points=pts, derivative_method="taylor"),
        )

        HJBGFDMSolver(problem=problem, collocation_points=pts, derivative_method="rbf")
        return "PASS", {"constructed": True}

    return run


def _fvm_fdm_agreement_cell():
    """FVM_MUSCL and FDM_UPWIND on one 1-D LQ problem must agree to a few percent.

    **What is and is not independent here.** ``create_paired_solvers`` gives
    ``FVM_MUSCL`` the pair (``HJBFDMSolver``, ``FPFVMSolver``) and ``FDM_UPWIND`` the
    pair (``HJBFDMSolver``, ``FPFDMSolver``) -- FVM has no HJB partner of its own yet.
    So the **M** axis is a genuine second discretisation and does not go tautological
    when the FP drift or the Hamiltonian is single-sourced; the **U** axis is the same
    HJB class on both arms and is a consistency check, not an independent comparison.
    The headline number is driven by the M axis (measured 4.891% M vs 2.723% U).

    An earlier version of this docstring claimed independence for the whole cell. That
    was wrong, and would have mattered: it is the claim someone would rely on when
    deciding this cell still bites after a consolidation.
    """

    def run():
        from mfgarchon.types import NumericalScheme

        p_fvm = _construct("FVM problem", _lq_problem_1d)
        dx = p_fvm.geometry.get_grid_spacing()[0]
        r_fvm = p_fvm.solve(scheme=NumericalScheme.FVM_MUSCL, max_iterations=40, tolerance=1e-4, verbose=False)

        p_fdm = _construct("FDM problem", _lq_problem_1d)
        r_fdm = p_fdm.solve(scheme=NumericalScheme.FDM_UPWIND, max_iterations=40, tolerance=1e-4, verbose=False)

        def verdict():
            M_fvm, M_fdm = _density(r_fvm), np.asarray(r_fdm.M, dtype=float)
            U_fvm, U_fdm = np.asarray(r_fvm.U, dtype=float), np.asarray(r_fdm.U, dtype=float)
            # Checked before the norms. `max()` drops a NaN in any non-leading
            # position -- max(0.0, nan, 0.0) is 0.0 -- so an all-NaN value function
            # would otherwise pass, and would also make --write-baseline emit a bare
            # NaN token, which is not valid JSON.
            art = {
                "all_finite": bool(
                    np.isfinite(M_fvm).all()
                    and np.isfinite(M_fdm).all()
                    and np.isfinite(U_fvm).all()
                    and np.isfinite(U_fdm).all()
                ),
                "tolerance": AGREEMENT_RTOL,
            }
            art |= _picard_verdict(r_fvm, "fvm_") | _picard_verdict(r_fdm, "fdm_")
            if not art["all_finite"]:
                art["worst"] = None
                return "FAIL", art
            art["rel_l2_M"] = _rel_l2(M_fvm, M_fdm, dx)
            art["rel_l2_U"] = _rel_l2(U_fvm, U_fdm, dx)
            art["rel_l2_M_terminal"] = _rel_l2(M_fvm[-1], M_fdm[-1], dx)
            art["worst"] = max(art["rel_l2_M"], art["rel_l2_U"], art["rel_l2_M_terminal"])
            return ("PASS" if art["worst"] < AGREEMENT_RTOL else "FAIL"), art

        return _measure("FVM/FDM agreement", verdict)

    return run


def _fvm_mass_cell():
    def run():
        from mfgarchon.types import NumericalScheme

        problem = _construct("FVM problem", _lq_problem_1d)
        result = problem.solve(scheme=NumericalScheme.FVM_MUSCL, max_iterations=40, tolerance=1e-4, verbose=False)

        def verdict():
            M = _density(result)
            dx = problem.geometry.get_grid_spacing()[0]
            mass = M.sum(axis=1) * dx
            art = {
                "mass_t0": float(mass[0]),
                "max_rel_drift": float(np.abs(mass - mass[0]).max() / abs(mass[0])),
                "min_density": float(M.min()),
                "all_finite": bool(np.isfinite(M).all()),
                "tolerance": 1e-6,
            }
            art |= _picard_verdict(result)
            ok = art["all_finite"] and art["min_density"] >= -1e-12 and art["max_rel_drift"] <= 1e-6
            return ("PASS" if ok else "FAIL"), art

        return _measure("FVM mass drift", verdict)

    return run


def _regime_switching_cell():
    """Two-regime Markov-switching MFG: every regime density stays non-negative.

    Mirrors what was the #1681 reproducer
    (tests/integration/test_phase1_5_validation.py::test_regime_switching_iterator_runs)
    at its original NT=10.

    [SUPERSEDED 2026-08-02] This docstring used to add: "That fixture is deliberately not
    retuned: the run maximum decays only first-order in dt, so clearing the guard threshold
    extrapolates to NT ~ 3e6 and refining would hide the defect rather than fix it." That
    described the negative density #1681 recorded, which is fixed -- the diagonal Markov
    outflow is no longer passed to the FP solver as a lagged source, and both regime
    densities are now strictly positive (min 8.4e-04), stable under dt-refinement. The
    fixture stays at NT=10 for continuity of the record, not to preserve a defect.
    SUPERSEDED-BY: the cell's `intended` note in capability_baseline.json, and #1798 --
    what fails here now is this cell's own per-regime mass oracle, not the solve.

    Reached through the deep module path. ``RegimeSwitchingIterator`` is exported from
    no package ``__init__``, so unlike every other cell here this one is not on the
    public surface -- which is itself part of what the cell records.
    """

    def run():
        from mfgarchon import MFGProblem
        from mfgarchon.alg.numerical.coupling.regime_switching_iterator import (
            RegimeSwitchingIterator,
        )
        from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
        from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
        from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
        from mfgarchon.core.mfg_components import MFGComponents
        from mfgarchon.core.regime_switching import RegimeSwitchingConfig
        from mfgarchon.geometry import TensorProductGrid
        from mfgarchon.geometry.boundary import no_flux_bc

        def lq(coupling_coeff):
            return MFGProblem(
                geometry=TensorProductGrid(
                    bounds=[(0.0, 1.0)],
                    Nx_points=[31],
                    boundary_conditions=no_flux_bc(dimension=1),
                ),
                Nt=10,
                T=1.0,
                sigma=0.1,
                components=MFGComponents(
                    m_initial=lambda x: np.exp(-10 * (np.asarray(x) - 0.5) ** 2),
                    u_terminal=lambda x: 0.0,
                    hamiltonian=SeparableHamiltonian(
                        control_cost=QuadraticControlCost(control_cost=1.0),
                        coupling=lambda m: coupling_coeff * m,
                        coupling_dm=lambda m: coupling_coeff,
                    ),
                ),
            )

        problems = _construct("regime problems", lambda: [lq(c) for c in (1.0, 0.5)])

        # NOT wrapped, deliberately. Solver and iterator construction is the object
        # under test: RegimeSwitchingIterator.__init__ runs assert_paired_solver_sigma
        # (#1603 / RFC #1574 C14) and three other library validations, so a refusal
        # here is a capability statement. Wrapping it would make that ERROR, and ERROR
        # is never baselined -- the refusal could then never be recorded at all.
        iterator = RegimeSwitchingIterator(
            problems=problems,
            regime_config=RegimeSwitchingConfig(transition_matrix=np.array([[-0.1, 0.1], [0.2, -0.2]])),
            hjb_solvers=[HJBFDMSolver(p) for p in problems],
            fp_solvers=[FPFDMSolver(p) for p in problems],
            max_iterations=3,
            tolerance=1e-4,
            damping=0.5,
        )
        result = iterator.solve()

        def verdict():
            dx = problems[0].geometry.get_grid_spacing()[0]
            per_regime = []
            for k, dens in enumerate(result.densities):
                M = _apply_mutation(np.asarray(dens, dtype=float))
                mass = M.sum(axis=1) * dx
                per_regime.append(
                    {
                        "regime": k,
                        "min_density": float(M.min()),
                        "max_rel_drift": float(np.abs(mass - mass[0]).max() / abs(mass[0])),
                        "all_finite": bool(np.isfinite(M).all()),
                    }
                )
            art = {
                "regimes": per_regime,
                "min_density": min(r["min_density"] for r in per_regime),
                "max_rel_drift": max(r["max_rel_drift"] for r in per_regime),
                "all_finite": all(r["all_finite"] for r in per_regime),
                "tolerance": 1e-6,
            }
            art |= _picard_verdict(result)
            ok = art["all_finite"] and art["min_density"] >= -1e-12 and art["max_rel_drift"] <= 1e-6
            return ("PASS" if ok else "FAIL"), art

        return _measure("regime mass/non-negativity", verdict)

    return run


CELLS = {
    "fdm_upwind/mass_conservation": _mass_conservation_cell("FDM_UPWIND"),
    "sl_linear/mass_conservation": _mass_conservation_cell("SL_LINEAR"),
    "fdm_centered/mass_conservation": _mass_conservation_cell("FDM_CENTERED"),
    "fvm_muscl/mass_conservation": _fvm_mass_cell(),
    "fvm_vs_fdm/agreement": _fvm_fdm_agreement_cell(),
    "sl_linear_2d/mass_conservation": _mass_conservation_2d_cell("SL_LINEAR"),
    "fdm_upwind_2d/mass_conservation": _mass_conservation_2d_cell("FDM_UPWIND"),
    "fdm_centered_2d/mass_conservation": _mass_conservation_2d_cell("FDM_CENTERED"),
    "fvm_muscl_2d/mass_conservation": _mass_conservation_2d_cell("FVM_MUSCL"),
    "regime_switching/non_negativity": _regime_switching_cell(),
    "gfdm_rbf/construction": _gfdm_rbf_cell(),
}

# Cells whose oracle is the density this module measures. --self-test perturbs that
# density and requires every one of them to leave PASS; one that does not is inert.
# Cells that are not PASS today are still listed: the self-test skips them now and
# picks them up automatically if they recover, so a recovery cannot arrive with an
# unproven oracle.
MASS_ORACLE_CELLS = {
    "fdm_upwind/mass_conservation",
    "sl_linear/mass_conservation",
    "fdm_centered/mass_conservation",
    "fvm_muscl/mass_conservation",
    "fvm_vs_fdm/agreement",
    "regime_switching/non_negativity",
    "sl_linear_2d/mass_conservation",
    "fdm_upwind_2d/mass_conservation",
    "fdm_centered_2d/mass_conservation",
    "fvm_muscl_2d/mass_conservation",
}


# Substrings that identify a warning as a report about the machine rather than about whether
# the configuration solves. Category alone does not separate these: the JAX-autodiff fallback
# is a RuntimeWarning, and it fires only where JAX is importable.
_ENVIRONMENT_MARKERS = (
    "JAX autodiff failed",
    "Falling back to finite-difference Jacobian",
)


def _warning_summary(caught) -> dict:
    """Fold a cell's warnings into something a baseline can carry and a diff can show.

    Keyed by category and a normalised prefix -- digits collapsed -- because the useful
    signal is "this cell emitted 39 non-convergence warnings", not 39 near-identical
    strings differing in a time index and a residual. Counts are stable against Python's
    once-per-location registry, because `simplefilter("always")` defeats it; a library that
    keeps its own warn-once set (`backends/__init__.py` has one) is not covered by that
    argument, and no cell here selects a configuration that reaches it.
    """
    seen: dict[str, int] = {}
    for w in caught:
        # Environment, not capability. These say what is installed on the machine that ran
        # the harness, so recording them makes the committed baseline machine-dependent:
        # a regeneration elsewhere produces a false diff, and -- worse -- `_note_still_applies`
        # drops the `intended` note attached to every cell whose artifact moved. Measured on
        # the JAX-autodiff fallback: forcing `_JAX_AVAILABLE = False` changes no number in
        # three 2-D cells (the numeric half is byte-identical) yet destroys three notes,
        # including the ~1500-character #1865 investigation record. A statement that holds
        # with and without a package installed is not a statement about the package.
        if issubclass(w.category, (ImportWarning, DeprecationWarning, PendingDeprecationWarning)):
            continue
        text = " ".join(str(w.message).split())
        if any(marker in text for marker in _ENVIRONMENT_MARKERS):
            continue
        # Collapse numbers so near-identical warnings fold, but only tokens that START with a
        # digit or a signed digit: `[-+0-9][0-9.eE+-]*` also ate the hyphen in ordinary words
        # ("non-negativity" -> "nonNnegativity"), which is not what "digits collapsed" means.
        norm = re.sub(r"(?<![\w.])[-+]?\d[\d.eE]*(?:[-+]\d+)?", "N", text)[:110]
        seen[f"{w.category.__name__}: {norm}"] = seen.get(f"{w.category.__name__}: {norm}", 0) + 1
    return dict(sorted(seen.items()))


def evaluate(only: list[str] | None = None) -> dict:
    results = {}
    for name, run in CELLS.items():
        if only and name not in only:
            continue
        t0 = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                status, artifact = run()
            # A HarnessError names its own phase: construct or measure, never the solve.
            except HarnessError as exc:
                status = "ERROR"
                artifact = {
                    "exception": "HarnessError",
                    "message": " ".join(str(exc).split())[:200],
                    "traceback_tail": traceback.format_exc().strip().splitlines()[-1],
                }
            # Broad by design: classifying the refusal IS the measurement here. Anything
            # reaching this point came out of the solve phase.
            except Exception as exc:
                status = "UNSUPPORTED"
                artifact = {
                    "exception": type(exc).__name__,
                    "message": " ".join(str(exc).split())[:200],
                }
                # Second net, for exception classes that cannot come from a library
                # refusal no matter which phase raised them.
                if type(exc).__name__ in {
                    "ImportError",
                    "ModuleNotFoundError",
                    "AttributeError",
                    "TypeError",
                    "AssertionError",
                    "KeyError",
                }:
                    status = "ERROR"
                    artifact["traceback_tail"] = traceback.format_exc().strip().splitlines()[-1]
        # Recorded, not gated (#1879). A cell that emits 39 "not a root of the discrete
        # HJB" warnings has said something about its capability; the status alone cannot
        # carry it, and `--check-baseline` compares status only, so this shows up where a
        # human diffing a regeneration will see it.
        said = _warning_summary(caught)
        if said:
            artifact["library_said"] = said
        results[name] = {
            "status": status,
            "artifact": artifact,
            "seconds": round(time.perf_counter() - t0, 2),
        }
    return results


def _statuses(results: dict) -> dict:
    return {k: v["status"] for k, v in results.items()}


def errored(results: dict) -> list[str]:
    """Cells whose apparatus failed. Never comparable, never baselined."""
    return sorted(k for k, v in results.items() if v["status"] == "ERROR")


def compare_to_baseline(current: dict[str, str], baseline: dict[str, dict]) -> list[str]:
    """Every status difference, in either direction, as a human-readable line.

    Bidirectional on purpose. A one-directional check (fail only on regression)
    lets a recovery land unrecorded, and the next run's baseline then encodes the
    recovery as if it had always held -- which is how a gate stops being able to
    say when something was fixed. Empty list means the tree matches the baseline.
    """
    problems = []
    for name, was in sorted(baseline.items()):
        now = current.get(name)
        if now is None:
            problems.append(f"  {name}: cell DISAPPEARED (baseline {was['status']})")
        elif now != was["status"]:
            kind = (
                "REGRESSION"
                if was["status"] == "PASS"
                else "RECOVERED -- record it in the baseline"
                if now == "PASS"
                else "SHIFT"
            )
            problems.append(f"  {name}: {was['status']} -> {now}  [{kind}]")
    for name in sorted(set(current) - set(baseline)):
        problems.append(f"  {name}: NEW cell, not in baseline")
    return problems


def _summarise(art: dict) -> str:
    """One line per artifact. Never raises: a reporting crash would lose the run.

    A cell whose oracle short-circuits (the NaN guard returns before the norms are
    computed) hands back a partial artifact, and a KeyError here would take down a
    report whose whole job is to show that something went wrong. The branches below
    cover every shape a production cell emits; the catch-all covers the rest, since
    "never raises" has to hold for shapes nobody anticipated -- that is the point.
    """
    try:
        return _summarise_known(art)
    except Exception:
        return f"UNSUMMARISABLE artifact: {repr(art)[:110]}"


def _summarise_known(art: dict) -> str:
    if "exception" in art:
        return f"{art['exception']}: {str(art.get('message', ''))[:70]}"
    if art.get("all_finite") is False:
        return "NON-FINITE values in the measured arrays"
    if art.get("worst") is not None:
        tol = art.get("tolerance")
        tol_s = f" (tol {tol:.0%})" if isinstance(tol, float) else ""
        return f"worst rel L2 {art['worst']:.3%}{tol_s}"
    if art.get("max_drift") is not None:
        return f"mass drift {art['max_drift']:.3e}, min M {art.get('min_density', float('nan')):.3e}"
    if art.get("max_rel_drift") is not None:
        return f"rel mass drift {art['max_rel_drift']:.3e}, min M {art.get('min_density', float('nan')):.3e}"
    return json.dumps(art, default=str)


def print_report(results: dict) -> None:
    width = max(len(k) for k in results)
    print(f"\n{'cell':<{width}}  {'status':<12} {'s':>6}  artifact")
    print("-" * (width + 60))
    for name, r in sorted(results.items()):
        print(f"{name:<{width}}  {r['status']:<12} {r['seconds']:>6.2f}  {_summarise(r['artifact'])}")
    tally = {}
    for r in results.values():
        tally[r["status"]] = tally.get(r["status"], 0) + 1
    print("\n" + "  ".join(f"{k}={v}" for k, v in sorted(tally.items())))


def self_test() -> int:
    """Prove each mass-oracle cell is load-bearing on the density it reports.

    Injects 10% relative mass drift, linear in time, and requires each such cell to
    leave PASS. This proves the oracle reads the density it reports; it does NOT prove
    the solver correct.
    """
    global _DENSITY_MUTATION

    baseline = evaluate(only=sorted(MASS_ORACLE_CELLS))
    broken = errored(baseline)
    if broken:
        print(f"SELF-TEST ABORTED: harness broken in {', '.join(broken)}.")
        return 2
    passing = [k for k, v in baseline.items() if v["status"] == "PASS"]
    if not passing:
        print("SELF-TEST INCONCLUSIVE: no mass-oracle cell is PASS, nothing to mutate.")
        return 1

    _DENSITY_MUTATION = 0.10
    try:
        mutated = evaluate(only=passing)
    finally:
        _DENSITY_MUTATION = None

    # ERROR under mutation is NOT evidence the oracle bites. Scoring on "not PASS"
    # counted a harness that broke under mutation as a working control -- the same
    # shape as treating ERROR as a comparable status, surviving in the one place the
    # earlier fix did not reach, and it matters more here because this IS the control.
    mutated_broken = errored(mutated)
    if mutated_broken:
        print(f"SELF-TEST ABORTED: harness broke under mutation in {', '.join(mutated_broken)}.")
        return 2

    inert = [k for k in passing if mutated[k]["status"] == "PASS"]
    for k in passing:
        verdict = "INERT" if k in inert else "discriminates"
        print(f"  {k:<34} PASS -> {mutated[k]['status']:<12} {verdict}")
    if inert:
        print(f"\nSELF-TEST FAILED: {len(inert)} cell(s) do not read the density they report.")
        return 1
    print(f"\nSELF-TEST PASSED: {len(passing)} mass-oracle cell(s) go red under 10% injected drift.")
    return 0


def _note_still_applies(prior_cell: dict | None, current: dict) -> bool:
    """Whether a carried-forward ``intended`` note still describes what the cell now does.

    Carry-forward previously required only that the status stayed non-PASS, so a note survived an
    arbitrary change of failure: swap the exception for an unrelated one and the note is preserved
    verbatim while the run still reports "0 unexplained". The note is an unchanged line, so it does
    not appear in the reviewer's diff at all -- only the artifact does. That is how an annotation
    outlives the evidence it cites, which is exactly the defect this file's own #1745 note had.

    Requiring the whole artifact to match means a changed failure drops the note, the cell reads as
    unexplained, and someone has to look at it. Comparing only the exception TYPE was the first
    attempt and was blind to the motivating case -- a note citing residual 2.42e-01 beside an
    artifact that had become 1.17e-05, ConvergenceError on both sides.

    Consequence worth knowing before annotating a FAIL cell: a FAIL artifact is a measurement, and
    measurements move, so its note will be dropped on the next regeneration. That is the safe
    direction (the cell reads as unexplained and gets looked at) but it makes `intended` awkward
    for FAIL cells. Every non-PASS cell today is exception-shaped, so this is latent.
    """
    if prior_cell is None or "intended" not in prior_cell or current["status"] == "PASS":
        return False
    # The WHOLE artifact, not just the exception type. Comparing types alone was blind to the
    # case this gate was written for: the #1745 note cited residual 2.42e-01 beside an artifact
    # that had become 1.17e-05, and both sides are ConvergenceError -- so the stale note carried
    # forward and the run still reported zero unexplained cells.
    return (prior_cell.get("artifact") or {}) == (current.get("artifact") or {})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="Emit full results as JSON")
    parser.add_argument("--write-baseline", metavar="FILE", help="Write current statuses to FILE and exit")
    parser.add_argument(
        "--check-baseline",
        metavar="FILE",
        help="Fail if any cell's status differs from FILE, in either direction",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Mutate the measured density and require every mass-oracle cell to go red",
    )
    args = parser.parse_args()

    if args.self_test:
        sys.exit(self_test())

    # --json must emit JSON and nothing else. The cells run real coupled solves, and those
    # write to stdout from three independent places -- library INFO logging, Rich progress bars,
    # and any print added later -- so `--json` never actually produced parseable output. Suppress
    # per-source and the next source added breaks it again; redirect the whole evaluation instead,
    # to stderr, so the diagnostics are still visible when a human is watching.
    if args.json:
        with contextlib.redirect_stdout(sys.stderr):
            results = evaluate()
    else:
        results = evaluate()

    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
        sys.stdout.flush()
        # Everything after the blob goes to stderr. Routing prints one at a time is how this
        # broke twice: the fix said "both remaining prints" when there were four, and the two it
        # missed fire exactly when the new unexplained-cell report has something to say. One
        # redirect cannot be defeated by a print added later.
        sys.stdout = sys.stderr
    else:
        print_report(results)

    # ERROR means the apparatus failed, so no verdict below is trustworthy. It is
    # checked before the baseline is even read, and it is never comparable: an ERROR
    # baselined as ERROR would otherwise match, and a harness broken during a
    # regeneration would be green forever -- the matrix silently stops measuring and
    # reports success for doing so.
    broken = errored(results)
    if broken:
        # stderr, not stdout: --json writes the blob to stdout above, and appending
        # a human block after it makes the output unparseable.
        print("\nHarness is broken; no capability verdict is trustworthy:", file=sys.stderr)
        for name in broken:
            art = results[name]["artifact"]
            print(f"  {name}: {art.get('exception')}: {art.get('message', '')[:150]}", file=sys.stderr)
        print("\nFix the harness. ERROR is never baselined and never compared.", file=sys.stderr)
        sys.exit(2)

    if args.write_baseline:
        # Carry forward any `intended` annotations. A regeneration that silently dropped them
        # would retire the distinction on its first use, which is how this kind of mechanism
        # usually dies. A cell that has RECOVERED loses its annotation on purpose: the claim
        # "this configuration is no longer refused" is exactly the one that needs review.
        prior: dict = {}
        if os.path.exists(args.write_baseline):
            with open(args.write_baseline) as fh:
                prior = json.load(fh).get("cells", {})
        elif any(v["status"] != "PASS" for v in results.values()):
            # Writing to a path with no prior file. In-place regeneration carries annotations
            # forward; writing somewhere new cannot, and silently emitting an unannotated
            # baseline is how the distinction would be lost the first time someone redirects
            # the output. Say so.
            print(
                f"\nNote: {args.write_baseline} does not exist, so no `intended` notes were "
                "carried forward. Non-PASS cells in the new file will read as unexplained. "
                "Regenerate in place to preserve them.",
                file=sys.stderr,
            )
        payload = {
            "_comment": (
                "Executed capability of the solve surface. --check-baseline compares STATUS "
                "only, in BOTH directions: a recovered cell must be recorded here in the same "
                "change that recovers it. The artifact blocks are a record for diffing by a "
                "human reviewer, NOT a gate -- a cell can degrade within its own tolerance "
                "(fvm_vs_fdm at 4.891% could reach 6.99%) and the status check stays green. "
                "Regenerate with --write-baseline. A non-PASS cell may carry an `intended` "
                "note saying WHY it is not PASS; cells without one are unexplained and are the "
                "actual backlog."
                " A cell's artifact may also carry `library_said`: the warnings the library"
                " emitted while that cell ran, folded by category and message with numbers"
                " collapsed. Warnings about the machine rather than about whether the"
                " configuration solves (import, deprecation, and the JAX-autodiff fallback) are"
                " excluded by design -- recording them makes this file machine-dependent and"
                " silently drops the `intended` notes of every cell whose artifact moves. A cell"
                " that emitted nothing carries no field at all."
                " The three 2-D cells read PASS from 2026-08-11 because the fixture was"
                " restated at a coupling strength comparable to its 1-D sibling (#1865), not"
                " because a solver improved. The configuration they used to fail at --"
                " `coupling=m` unscaled, the aggregating regime -- is no longer covered here and is tracked on #1865."
            ),
            "cells": {
                k: (
                    {"status": v["status"], "artifact": v["artifact"], "intended": prior[k]["intended"]}
                    if _note_still_applies(prior.get(k), v)
                    else {"status": v["status"], "artifact": v["artifact"]}
                )
                for k, v in results.items()
            },
        }
        # allow_nan=False: Python would otherwise emit a bare `NaN` token, which is
        # not JSON and which every non-Python reader rejects. A non-finite artifact
        # also means a cell's oracle produced one, which is a defect in the cell --
        # refuse rather than record it.
        try:
            body = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
        except ValueError as exc:
            print(f"\nRefusing to write a baseline with a non-finite artifact: {exc}")
            print("A cell's oracle returned NaN or Infinity. Fix the cell.")
            sys.exit(1)
        with open(args.write_baseline, "w") as fh:
            fh.write(body + "\n")
        # stderr under --json so the two flags together still emit parseable output.
        print(f"\nBaseline written to {args.write_baseline}", file=sys.stderr if args.json else sys.stdout)
        sys.exit(0)

    if args.check_baseline:
        with open(args.check_baseline) as fh:
            baseline = json.load(fh)["cells"]
        problems = compare_to_baseline(_statuses(results), baseline)
        # A non-PASS cell without an `intended` note is unexplained. This is the number that
        # matters, not the count of UNSUPPORTED: "drive UNSUPPORTED to 0" is satisfiable by
        # loosening a guard or by quietly picking a configuration the guard does not refuse,
        # and neither is visible in a status count. An `intended` note makes the difference
        # between "the gate correctly refuses this" and "this does not work" a written claim.
        unexplained = [
            f"  {name}: {info['status']} with no `intended` note"
            for name, info in sorted(baseline.items())
            if info["status"] != "PASS" and "intended" not in info
        ]
        if unexplained:
            print(f"\n{len(unexplained)} non-PASS cell(s) with no recorded reason:")
            print("\n".join(unexplained))
            print('  Add "intended": "<why>" to each in the baseline, or fix the cell.')
        if problems:
            print("\nCapability baseline mismatch:")
            print("\n".join(problems))
            print("\nIf the change is intended, regenerate with --write-baseline in the same commit.")
            sys.exit(1)
        explained = sum(1 for i in baseline.values() if i["status"] != "PASS" and "intended" in i)
        non_pass = sum(1 for i in baseline.values() if i["status"] != "PASS")
        print(
            f"\nCapability matches baseline ({len(baseline)} cells; "
            f"{non_pass - explained} of {non_pass} non-PASS cells unexplained).",
            file=sys.stderr if args.json else sys.stdout,
        )
        sys.exit(0)

    sys.exit(0)


if __name__ == "__main__":
    main()
