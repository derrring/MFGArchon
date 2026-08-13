#!/usr/bin/env python3
"""Count the deprecations that actually exist, and refuse to let the count drift unnoticed.

Two checks, and the first one is why this file was rewritten (Issue #1713).

**The count.** The previous version discovered deprecated symbols by walking the AST for
``@deprecated`` decorators and calling ``_extract_metadata``, which returned ``None`` unless the
decorator literally passed ``removal_blockers=``. That kwarg is optional and **no call site in the
package passes it**, so the detector found nothing by construction: it printed ``Total deprecated
symbols: 0`` and exited 0 while ``audit_all_deprecations`` reported 72 live in the same tree, and
``.github/workflows/deprecation-check.yml`` was green over it.

This now asks the **runtime registry** instead. ``audit_all_deprecations`` reads what the decorators
actually registered, so it cannot under-count the way a syntax guess can -- the same class of error
that made ``check_fail_fast``'s ``broad_except`` read 11 against a true 115 for 32 days, because
``except\\s+Exception\\s*:`` cannot match ``except Exception as e:`` (#1706).

**Boundaries, and they are real.** Three, each measured rather than supposed:

- This covers deprecations *declared through the decorators*. Something retired without one is
  invisible here too. Asking the registry removes the under-counting that came from guessing
  syntax; it does not make the count a census of everything ever deprecated.
- A new deprecation whose ``(name, type, since)`` collides with an existing one does not move the
  count, because that triple is `audit_all_deprecations`' dedup key. Constructed: a second
  ``@deprecated(since="v0.17.0") def create_solver()`` reads 63 and passes. Predates this ratchet.
- Scope is decided from the modules a symbol was *found* in. A deprecated method on a class that
  has no module-level binding anywhere in scope -- built by a factory and bound only inside
  ``alg/neural/__init__.py`` -- records only the frozen site and is excluded. No module-based
  policy can see it; the registry holds no evidence of an in-scope site.

**Why the count is a property of the tree and not of the runner.** The registry is populated by
importing, so the first version of this check compared a number that moved with the environment
against a committed baseline: 72 locally, **41** on the GitHub runner, and the run was red for a
reason that said nothing about the tree. Two things fixed that, and both are needed:

- ``scan_deprecated`` no longer lets one unimportable subpackage end the walk. Without torch,
  ``alg/reinforcement/multi_population`` raises ``AttributeError``, which ``pkgutil.walk_packages``
  re-raises, and the walk stopped at **160 of 428** modules -- so 29 of the 31 missing deprecations
  were not torch's, they were in ``geometry/``, ``utils/`` and ``operators/``, never reached.
- The census is scoped to the **live** library. ``alg/neural`` and ``alg/reinforcement`` are frozen
  prototypes and out of scope for repo-wide campaigns (CLAUDE.md), and measured, they are also
  exactly where every torch-dependent module lives. Scoped that way the count is **63 with torch
  and 63 without**, which is what makes a committed baseline meaningful.

That is a claim about this environment, so it is checked rather than assumed: any module *in scope*
that cannot be imported makes this exit 2 and name it, because a smaller number and a number
measured over less tree are indistinguishable from the outside.

**The usage check.** The original purpose: production code must not call a deprecated symbol whose
``internal_usage`` blocker has been cleared. It is kept, and now runs over a real symbol list. Note
it is vacuous today by construction -- 0 of 63 have that blocker cleared -- so it starts doing work
the day someone clears one, and not before. Reporting the count is what carries information now.

The baseline fails in BOTH directions, like ``fail_fast_baseline.json`` and
``capability_baseline.json``: a count that drops must be recorded too, so an improvement cannot land
silently and a baseline cannot be lowered without doing the work.

Usage:
    python scripts/check_internal_deprecation.py                    # check against the shipped baseline
    python scripts/check_internal_deprecation.py --write-baseline   # record the current counts
    python scripts/check_internal_deprecation.py --show             # list every live deprecation
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
BASELINE = Path(__file__).resolve().parent / "deprecation_baseline.json"

# The two frozen prototype paradigms (CLAUDE.md): out of scope for repo-wide campaigns, and the
# only place in the package where a module needs torch to import. Excluding them is what makes the
# count identical with and without it.
# The frozen paradigms this excluded were deleted (see the changelog for this change), so the
# tuple is empty rather than removed: the exclusion mechanism is still the right shape if a
# package is ever frozen again, and an empty tuple keeps `startswith(FROZEN)` well defined.
FROZEN: tuple[str, ...] = ()


def is_frozen(module: str) -> bool:
    """Exact package containment, not a string prefix.

    `startswith(FROZEN)` would also swallow a future `mfgarchon.alg.neural_ops`, and swallow it
    silently, which is the whole failure mode this check exists to stop.
    """
    return any(module == prefix or module.startswith(prefix + ".") for prefix in FROZEN)


class EnvironmentIncompleteError(RuntimeError):
    """The tree could not be read in full HERE, so nothing was measured about it."""

    def __init__(self, unimportable: dict[str, str]) -> None:
        self.unimportable = unimportable
        super().__init__(f"{len(unimportable)} in-scope module(s) could not be imported")


def in_scope(audited: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Split the census by scope, judging each symbol by EVERY module it was found in.

    A symbol is out of scope only when the frozen paradigms are the *only* place it appears.
    Judging `entry["module"]` alone judges an accident of walk order instead: that field holds
    whichever copy the dedup happened to keep, and `mfgarchon.alg.neural` sorts before every other
    top-level subpackage. Measured -- adding a deprecation under `mfgarchon/utils/` took the count
    63 -> 64 and went red correctly, and one `from mfgarchon.utils.<mod> import <symbol>` appended
    to `alg/neural/__init__.py` put it back to 63 and green over a symbol that had just been added.
    """
    live = [entry for entry in audited if not all(is_frozen(mod) for mod in entry["modules"])]
    return live, len(audited) - len(live)


def live_deprecations() -> tuple[list[dict[str, Any]], int]:
    """The live library's deprecations, and how many the frozen-paradigm scope dropped.

    The second number is returned rather than swallowed so the scope is visible in the output: a
    census that silently covers less than its name suggests is the failure this check exists for.

    Raises:
        EnvironmentIncompleteError: an in-scope module could not be imported. Distinct from a count
            that moved: one is a verdict about the tree, the other about the runner, and reporting
            the second as the first is what made this check red on a tree nobody had changed.
    """
    import logging

    logging.disable(logging.INFO)  # importing the package emits workflow setup chatter
    import mfgarchon
    from mfgarchon.utils.deprecation import IncompleteScanError, audit_all_deprecations

    try:
        buckets = audit_all_deprecations(mfgarchon)
    except IncompleteScanError as exc:
        blocking = {mod: why for mod, why in exc.unimportable.items() if not is_frozen(mod)}
        if blocking:
            raise EnvironmentIncompleteError(blocking) from exc
        # Only the frozen paradigms are unreadable, which the census excludes anyway. The second
        # walk is warm (measured 0.07 s against 0.62 s cold) because sys.modules is populated.
        buckets = audit_all_deprecations(mfgarchon, allow_incomplete=True)

    return in_scope([entry for bucket in buckets.values() for entry in bucket])


def counts(entries: list[dict[str, Any]]) -> dict[str, int]:
    """The numbers the baseline pins.

    Coarse on purpose: a per-symbol baseline would churn on every rename and stop being read.
    """
    by_type: dict[str, int] = {}
    for entry in entries:
        by_type[entry["type"]] = by_type.get(entry["type"], 0) + 1
    cleared = sum(1 for e in entries if not _blocked_on_internal_usage(e))
    return {
        "total": len(entries),
        "internal_usage_cleared": cleared,
        **{f"type:{k}": v for k, v in sorted(by_type.items())},
    }


def _blocked_on_internal_usage(entry: dict[str, Any]) -> bool:
    blockers = entry.get("remaining_blockers") or entry.get("removal_blockers") or []
    return "internal_usage" in blockers


def production_uses(symbols: set[str]) -> list[str]:
    """Call sites in ``mfgarchon/`` naming one of ``symbols``, excluding the places that must.

    ``deprecation.py`` defines the decorators and ``compat/`` exists to hold backward-compatible
    wrappers. A hit anywhere else is production code depending on something declared ready to go.
    """
    if not symbols:
        return []
    hits = []
    for path in sorted((REPO / "mfgarchon").rglob("*.py")):
        if path.name == "deprecation.py" or "compat" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            name = node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else None
            if name in symbols:
                hits.append(f"{path.relative_to(REPO)}:{node.lineno}: {name}")
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write-baseline", action="store_true", help="record current counts and exit")
    parser.add_argument("--show", action="store_true", help="list every live deprecation")
    args = parser.parse_args()

    try:
        entries, out_of_scope = live_deprecations()
    except EnvironmentIncompleteError as exc:
        print("This environment cannot read the whole in-scope tree, so NOTHING was measured:", file=sys.stderr)
        for module, why in sorted(exc.unimportable.items()):
            print(f"  {module}: {why}", file=sys.stderr)
        print(
            "\nThat is an environment failure, not a verdict on the code: install what those "
            "modules need and re-run. Do NOT regenerate the baseline here -- a count taken over "
            "less tree than the baseline covers is a different quantity wearing the same name, "
            "and recording it is how the smaller number becomes the official one.",
            file=sys.stderr,
        )
        return 2

    current = counts(entries)

    if args.show:
        for entry in sorted(entries, key=lambda x: (x["type"], x["name"])):
            blockers = ",".join(entry.get("remaining_blockers") or []) or "-"
            print(
                f"  {entry['type']:10s} {entry['name']:58s} since {entry.get('since')!s:9s} "
                f"removal {entry.get('removal')!s:9s} [{blockers}]"
            )
        print()

    print(f"Live deprecations (runtime registry, excluding the frozen paradigms): {current['total']}")
    for key, value in current.items():
        if key != "total":
            print(f"  {key}: {value}")
    print(f"  (out of scope, in alg/neural + alg/reinforcement: {out_of_scope})")

    if args.write_baseline:
        BASELINE.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
        print(f"\nBaseline written to {BASELINE.relative_to(REPO)}")
        return 0

    if not BASELINE.exists():
        print(f"\nNo baseline at {BASELINE.relative_to(REPO)}; write one with --write-baseline.", file=sys.stderr)
        return 2

    baseline = json.loads(BASELINE.read_text())
    drift = [
        f"  {key}: {baseline.get(key, 0)} -> {current.get(key, 0)}"
        for key in sorted(set(baseline) | set(current))
        if baseline.get(key, 0) != current.get(key, 0)
    ]
    if drift:
        print("\nDeprecation counts moved against the baseline:")
        print("\n".join(drift))
        print(
            "\nBoth directions fail. A count that ROSE means a deprecation was added without "
            "recording it; a count that FELL means one was retired while the baseline still claims "
            "it exists. Regenerate with --write-baseline in the commit that does the work."
        )
        return 1

    ready = {e["name"].split(".")[-1] for e in entries if not _blocked_on_internal_usage(e)}
    violations = production_uses(ready)
    if violations:
        print(f"\nProduction code calls {len(violations)} deprecated symbol(s) with internal_usage cleared:")
        for violation in violations[:40]:
            print(f"  {violation}")
        return 1

    print(f"\nMatches baseline. {len(ready)} symbol(s) have internal_usage cleared; no production call sites.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
