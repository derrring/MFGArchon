#!/usr/bin/env python3
"""Execute the public solve surface and record what it can actually do.

Counts of tests, issues, or fail-fast violations all move for reasons unrelated to
whether the library solves anything. This records the other quantity: for a fixed set
of configurations driven through the public API, does the result satisfy an oracle
that lives outside the code under test.

Three properties make a cell hard to satisfy the cheap way:

- The oracle is a number (mass drift, relative L2 against a second discretisation),
  not an exception. ``pytest.raises`` cannot turn a cell green.
- The comparison is against a closed form or a second independent discretisation,
  never against another call into the same owner -- so consolidating two paths does
  not make a cell tautological the way an agreement test becomes tautological.
- ``--check-baseline`` fails in BOTH directions. A cell that recovers fails the check
  until the baseline records the recovery, so improvements cannot land silently and a
  baseline cannot be lowered without doing the work. Same structure as
  ``check_fail_fast.py`` / ``fail_fast_baseline.json``.

A cell status is one of:

  PASS         solve returned and the oracle held
  FAIL         solve returned and the oracle was violated (measured value recorded)
  UNSUPPORTED  the path refused to run (exception type + message head recorded)
  ERROR        the harness itself broke -- always fails --check-baseline

Cells deliberately NOT in this first set, so the omission is on the record rather than
implied absent:

- regime-switching non-negativity (#1681). ``RegimeSwitchingIterator`` is not exported
  from any package ``__init__``; reaching it needs the deep module path plus a
  per-regime problem list. It belongs here, it is not here yet.
- 2-D meshless-Galerkin + Nitsche refinement (#1679). Needs a refinement sweep, so it
  is minutes, not seconds; it belongs in a nightly-tier matrix.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
import warnings

import numpy as np

warnings.filterwarnings("ignore")

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


def _rel_l2(a: np.ndarray, b: np.ndarray, dx: float) -> float:
    return float(np.sqrt(dx * np.sum((a - b) ** 2)) / np.sqrt(dx * np.sum(b**2)))


# --------------------------------------------------------------------------------
# Cells
# --------------------------------------------------------------------------------


def _mass_conservation_cell(scheme_name: str):
    """Public problem.solve(scheme=...) must not move total mass."""

    def run():
        from mfgarchon.types import NumericalScheme

        problem = _smoke_problem()
        result = problem.solve(scheme=getattr(NumericalScheme, scheme_name), max_iterations=5, verbose=False)
        art = _mass_drift(result, problem)
        tol = MASS_RTOL * max(abs(art["mass_t0"]), 1.0)
        art["tolerance"] = tol
        if not art["all_finite"]:
            return "FAIL", art
        if art["max_drift"] > tol:
            return "FAIL", art
        return "PASS", art

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

        problem = _smoke_problem()
        pts = np.linspace(0.0, 1.0, 21).reshape(-1, 1)

        # Control: the same call with the supported method must construct. If this
        # raises, the cell reports ERROR (harness broken), never UNSUPPORTED.
        try:
            HJBGFDMSolver(problem=problem, collocation_points=pts, derivative_method="taylor")
        except Exception as exc:
            raise AssertionError(
                f"harness control failed: derivative_method='taylor' did not construct "
                f"({type(exc).__name__}: {' '.join(str(exc).split())[:120]}); "
                f"the rbf verdict below would be meaningless"
            ) from exc

        HJBGFDMSolver(problem=problem, collocation_points=pts, derivative_method="rbf")
        return "PASS", {"constructed": True}

    return run


def _fvm_fdm_agreement_cell():
    """Two independent discretisations of one PDE must agree to a few percent.

    The oracle is a second discretisation, not a second call into a shared owner, so
    single-sourcing the drift or the Hamiltonian does not make this tautological.
    """

    def run():
        from mfgarchon.types import NumericalScheme

        p_fvm = _lq_problem_1d()
        dx = p_fvm.geometry.get_grid_spacing()[0]
        r_fvm = p_fvm.solve(scheme=NumericalScheme.FVM_MUSCL, max_iterations=40, tolerance=1e-4)

        p_fdm = _lq_problem_1d()
        r_fdm = p_fdm.solve(scheme=NumericalScheme.FDM_UPWIND, max_iterations=40, tolerance=1e-4)

        M_fvm, M_fdm = _density(r_fvm), np.asarray(r_fdm.M, dtype=float)
        U_fvm, U_fdm = np.asarray(r_fvm.U, dtype=float), np.asarray(r_fdm.U, dtype=float)

        art = {
            "rel_l2_M": _rel_l2(M_fvm, M_fdm, dx),
            "rel_l2_U": _rel_l2(U_fvm, U_fdm, dx),
            "rel_l2_M_terminal": _rel_l2(M_fvm[-1], M_fdm[-1], dx),
            "tolerance": AGREEMENT_RTOL,
        }
        worst = max(art["rel_l2_M"], art["rel_l2_U"], art["rel_l2_M_terminal"])
        art["worst"] = worst
        return ("PASS" if worst < AGREEMENT_RTOL else "FAIL"), art

    return run


def _fvm_mass_cell():
    def run():
        from mfgarchon.types import NumericalScheme

        problem = _lq_problem_1d()
        result = problem.solve(scheme=NumericalScheme.FVM_MUSCL, max_iterations=40, tolerance=1e-4)
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
        ok = art["all_finite"] and art["min_density"] >= -1e-12 and art["max_rel_drift"] <= 1e-6
        return ("PASS" if ok else "FAIL"), art

    return run


CELLS = {
    "fdm_upwind/mass_conservation": _mass_conservation_cell("FDM_UPWIND"),
    "sl_linear/mass_conservation": _mass_conservation_cell("SL_LINEAR"),
    "fdm_centered/mass_conservation": _mass_conservation_cell("FDM_CENTERED"),
    "fvm_muscl/mass_conservation": _fvm_mass_cell(),
    "fvm_vs_fdm/agreement": _fvm_fdm_agreement_cell(),
    "gfdm_rbf/construction": _gfdm_rbf_cell(),
}

# Cells whose oracle is the density this module measures. --self-test scales that
# density and requires every one of them to leave PASS; one that does not is inert.
MASS_ORACLE_CELLS = {
    "fdm_upwind/mass_conservation",
    "sl_linear/mass_conservation",
    "fdm_centered/mass_conservation",
    "fvm_muscl/mass_conservation",
    "fvm_vs_fdm/agreement",
}


def evaluate(only: list[str] | None = None) -> dict:
    results = {}
    for name, run in CELLS.items():
        if only and name not in only:
            continue
        t0 = time.perf_counter()
        try:
            status, artifact = run()
        # Broad by design: classifying the refusal IS the measurement here. The
        # narrowing happens below, on the exception type.
        except Exception as exc:
            status = "UNSUPPORTED"
            head = " ".join(str(exc).split())[:200]
            artifact = {"exception": type(exc).__name__, "message": head}
            # A cell reports UNSUPPORTED only for a refusal by the code under test.
            # A broken harness -- bad import, wrong signature, failed control -- must
            # never be readable as "the library does not support this".
            if type(exc).__name__ in {
                "ImportError",
                "ModuleNotFoundError",
                "AttributeError",
                "TypeError",
                "AssertionError",
            }:
                status = "ERROR"
                artifact["traceback_tail"] = traceback.format_exc().strip().splitlines()[-1]
        results[name] = {
            "status": status,
            "artifact": artifact,
            "seconds": round(time.perf_counter() - t0, 2),
        }
    return results


def _statuses(results: dict) -> dict:
    return {k: v["status"] for k, v in results.items()}


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


def print_report(results: dict) -> None:
    width = max(len(k) for k in results)
    print(f"\n{'cell':<{width}}  {'status':<12} {'s':>6}  artifact")
    print("-" * (width + 60))
    for name, r in sorted(results.items()):
        art = r["artifact"]
        if "exception" in art:
            summary = f"{art['exception']}: {art['message'][:70]}"
        elif "worst" in art:
            summary = f"worst rel L2 {art['worst']:.3%} (tol {art['tolerance']:.0%})"
        elif "max_drift" in art:
            summary = f"mass drift {art['max_drift']:.3e}, min M {art['min_density']:.3e}"
        elif "max_rel_drift" in art:
            summary = f"rel mass drift {art['max_rel_drift']:.3e}, min M {art['min_density']:.3e}"
        else:
            summary = json.dumps(art)
        print(f"{name:<{width}}  {r['status']:<12} {r['seconds']:>6.2f}  {summary}")
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
    passing = [k for k, v in baseline.items() if v["status"] == "PASS"]
    if not passing:
        print("SELF-TEST INCONCLUSIVE: no mass-oracle cell is PASS, nothing to mutate.")
        return 1

    _DENSITY_MUTATION = 0.10
    try:
        mutated = evaluate(only=passing)
    finally:
        _DENSITY_MUTATION = None

    inert = [k for k in passing if mutated[k]["status"] == "PASS"]
    for k in passing:
        verdict = "INERT" if k in inert else "discriminates"
        print(f"  {k:<34} PASS -> {mutated[k]['status']:<12} {verdict}")
    if inert:
        print(f"\nSELF-TEST FAILED: {len(inert)} cell(s) do not read the density they report.")
        return 1
    print(f"\nSELF-TEST PASSED: {len(passing)} mass-oracle cell(s) go red under 10% injected drift.")
    return 0


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

    results = evaluate()

    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
    else:
        print_report(results)

    if args.write_baseline:
        payload = {
            "_comment": (
                "Executed capability of the public solve surface. --check-baseline fails in "
                "BOTH directions: a recovered cell must be recorded here in the same change "
                "that recovers it. Regenerate with --write-baseline."
            ),
            "cells": {k: {"status": v["status"], "artifact": v["artifact"]} for k, v in results.items()},
        }
        with open(args.write_baseline, "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print(f"\nBaseline written to {args.write_baseline}")
        sys.exit(0)

    if args.check_baseline:
        with open(args.check_baseline) as fh:
            baseline = json.load(fh)["cells"]
        problems = compare_to_baseline(_statuses(results), baseline)
        if problems:
            print("\nCapability baseline mismatch:")
            print("\n".join(problems))
            print("\nIf the change is intended, regenerate with --write-baseline in the same commit.")
            sys.exit(1)
        print(f"\nCapability matches baseline ({len(baseline)} cells).")
        sys.exit(0)

    sys.exit(1 if any(r["status"] == "ERROR" for r in results.values()) else 0)


if __name__ == "__main__":
    main()
