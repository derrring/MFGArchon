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
  mutates the main checkout in place, restores from a pristine copy under try/finally,
  and refuses to start on a dirty tree. It also asserts at run time that the imported
  `mfgarchon` is the tree it mutated.
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
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# The CI marker set, matching scripts/local_ci.sh so a kill here means a kill there.
MARKERS = "not slow and not benchmark and not experimental and not optional_torch and not environment"

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
]

_FAILED = re.compile(r"^(?:FAILED|ERROR) (\S+?)(?:\s|$)", re.MULTILINE)


@dataclass
class Run:
    failed: set[str] = field(default_factory=set)
    returncode: int = 0
    seconds: float = 0.0


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
        seconds=round(time.perf_counter() - t0, 1),
    )


def _env() -> dict:
    import os

    return {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}


def _assert_clean_tree() -> None:
    """No MODIFIED tracked files.

    Untracked files are ignored on purpose: the restore rewrites tracked files from a
    pristine copy, so an untracked file cannot be clobbered by it and its presence says
    nothing about whether the restore worked. Blocking on `??` would only train the
    operator to reach for --force.
    """
    out = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True).stdout
    modified = [ln for ln in out.splitlines() if not ln.startswith("??")]
    if modified:
        sys.exit(
            "Refusing to run: tracked files are modified. This script edits the checkout in "
            "place (see the #1677 note in the module docstring) and restores from a pristine "
            "copy, so it must start from a state it can prove it restored.\n" + "\n".join(modified)
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


def _write_baseline(path: Path, results: dict) -> None:
    payload = {
        "_comment": (
            "Kill counts per mutated convention. --check-baseline fails when a count DROPS "
            "(discrimination lost) and when one RISES (record the gain in the same change). "
            "Counts, not test names: this population cannot be selected by name -- the "
            "agreement-shaped patterns give 51, 114 or 156 depending on the regex."
        ),
        "_measured_at": _head_sha(),
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


def compare_to_baseline(results: dict, baseline: dict) -> list[str]:
    """Every way discrimination can degrade. Empty list means it did not.

    Two ratchets, both unforgeable by renaming -- which matters, because the
    population this measures cannot be selected by name at all:

    - **Per-mutation kill counts must not drop.** A convention that 129 tests noticed
      and 120 notice now has lost coverage, whatever the test names are. Deleting a
      discriminating test trips this, which is correct: it is a real loss.
    - **The UNCOVERED set must not grow.** A convention going from watched to
      unwatched is the defect this tool exists to find.

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
    for name in sorted(set(results) - set(base_muts)):
        problems.append(f"  {name}: NEW mutation, not in baseline")
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
            f"kill could not be attributed.\n  " + "\n  ".join(sorted(base.failed)[:10])
        )
    print(f"  clean, {base.seconds}s\n", flush=True)

    results: dict[str, dict] = {}
    backups: dict[str, str] = {}
    pristine = Path(tempfile.mkdtemp(prefix="discrim-"))
    try:
        for mut in selected:
            shutil.copy2(REPO / mut.path, pristine / Path(mut.path).name)
            print(f"[{mut.name}] {mut.owner}", flush=True)
            apply_mutation(mut, backups)
            try:
                live = _mutation_is_live(mut)
                run = _pytest(paths) if live else Run()
            finally:
                restore(backups)

            if not live:
                status, note = "INEFFECTIVE", "  <-- mutation not observable; its zeros mean nothing"
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
        shutil.rmtree(pristine, ignore_errors=True)

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

    _assert_clean_tree()
    print("\nWorking tree restored and verified clean.")

    if args.write_baseline:
        _write_baseline(Path(args.write_baseline), results)
        print(f"Baseline written to {args.write_baseline}")
        sys.exit(0)

    if args.check_baseline:
        baseline = json.loads(Path(args.check_baseline).read_text())
        problems = compare_to_baseline(results, baseline)
        if problems:
            print("\nDiscrimination baseline mismatch:")
            print("\n".join(problems))
            print("\nIf intended, regenerate with --write-baseline in the same commit.")
            sys.exit(1)
        print(f"Discrimination matches baseline ({len(baseline['mutations'])} mutations).")
        sys.exit(0)

    sys.exit(0 if effective else 1)


if __name__ == "__main__":
    main()
