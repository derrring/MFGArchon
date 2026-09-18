#!/usr/bin/env python3
"""Every keyword argument in a docstring example must be a real parameter of the callee it names.

#674 removed a positional constructor and left 36 of 37 docstring examples dead, 7 of them raising
`TypeError` on the removed form, invisible because nothing executed them. #2341 repeated it at one
line. The deprecation-removal programme (#2343) is that operation at much larger scale, so the rot
needs an instrument that does not require the examples to be executable -- and most of them are not:
measured at 408415d4, running the package's examples gives 1348 failures of 2439, of which 1153 are
`NameError` from examples written as narrative fragments that assume names from a neighbouring
docstring. Stock per-docstring isolation cannot supply those names; a collector that shares globals
across a module recovers 154 of the 1153, which still leaves execution a documentation programme
rather than a check (review of #2351 measured the 154). This is the half that can be checked today.

What it does NOT catch, stated so the number is not read as coverage: a positional argument that
moved, a keyword whose meaning changed while the name survived, a callee it cannot resolve (a local
name, a method on an instance built earlier in the example), anything in a `**kwargs` signature,
keywords hidden behind `f(**mapping)` since those names are not in the AST, and the 19 blocks of 643
whose source does not parse. Those 19 are excluded from the reported block count rather than counted
as analysed -- the count below is what was ANALYSED, not what was seen.

An aliased import is NOT a miss; it resolves and fires. That was listed as a limit here before the
review of #2351 measured it.

It resolves the callee in the example's own module namespace first, because a bare-name map resolved
`MFGComponents` to the wrong class and manufactured 7 findings that do not exist. An ATTRIBUTE call
follows its receiver only when the receiver names a module this package owns: reducing
`np.gradient(..., axis=0)` to the bare `gradient` resolved it to this package's own `gradient` and
put a false positive in the shipped baseline, which is the same wrong-callee class one path over.

Ratchets against a baseline: the count may fall freely, and a rise names the example.
"""

from __future__ import annotations

import argparse
import ast
import doctest
import importlib
import inspect
import io
import json
import pkgutil
import sys
import warnings
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE = REPO / "scripts" / "docstring_kwargs_baseline.json"


def _import_all(package) -> list:
    """Every importable submodule, with import noise suppressed; an unimportable one is skipped."""
    modules = []
    for info in pkgutil.walk_packages(package.__path__, prefix=f"{package.__name__}."):
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                modules.append(importlib.import_module(info.name))
        except BaseException:  # an unimportable module is not this check's subject
            continue
    return modules


def _fallback_names(modules) -> dict:
    """Bare-name map, used only when a name is absent from the example's own module."""
    names: dict = {}
    for mod in modules:
        for attr, obj in vars(mod).items():
            if (inspect.isclass(obj) or inspect.isfunction(obj)) and attr not in names:
                names[attr] = obj
    return names


def _check_call(node, target, test, name: str, findings: list[str], counters: dict) -> None:
    """One call's keywords against one resolved callee. Shared so the attribute path cannot drift."""
    if not (inspect.isclass(target) or inspect.isfunction(target)):
        return
    try:
        signature = inspect.signature(target)
    except (ValueError, TypeError):
        return
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return  # **kwargs accepts anything; nothing to check
    for keyword in node.keywords:
        if keyword.arg is None:
            continue  # `f(**mapping)`: the names are not in the AST, so this check cannot see them
        counters["checked"] += 1
        if keyword.arg not in signature.parameters:
            findings.append(f"{test.name}: {name}({keyword.arg}=...)")


def scan(package_name: str = "mfgarchon") -> tuple[list[str], int, int]:
    """Returns (findings, keyword arguments checked, example blocks seen)."""
    warnings.simplefilter("ignore")
    package = importlib.import_module(package_name)
    modules = _import_all(package)
    fallback = _fallback_names(modules)
    findings: list[str] = []
    counters = {"checked": 0}
    blocks = unparsed = 0
    for mod in modules:
        try:
            tests = [t for t in doctest.DocTestFinder().find(mod) if t.examples]
        except Exception:  # a docstring the finder cannot parse is not the subject
            continue
        for test in tests:
            blocks += 1
            try:
                tree = ast.parse("".join(example.source for example in test.examples))
            except SyntaxError:
                unparsed += 1  # an illustrative fragment, not python -- counted, not silently dropped
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not node.keywords:
                    continue
                if isinstance(node.func, ast.Name):
                    name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    # `np.gradient(..., axis=0)` is numpy's, not this package's. Reducing it to the bare
                    # `gradient` resolved it to `utils.numerical.tensor_calculus.gradient` and shipped a FALSE
                    # POSITIVE in the baseline -- the same wrong-callee class the module-namespace fix was for,
                    # left on the attribute path (review of #2351). A receiver is only followed when it names a
                    # module this package owns; anything else is skipped rather than guessed.
                    receiver = ast.unparse(node.func.value)
                    owner = vars(mod).get(receiver) or test.globs.get(receiver)
                    if owner is None or not getattr(owner, "__name__", "").startswith(package_name):
                        continue
                    name = node.func.attr
                    target = getattr(owner, name, None)
                    _check_call(node, target, test, name, findings, counters)
                    continue
                else:
                    continue
                target = vars(mod).get(name) or test.globs.get(name) or fallback.get(name)
                _check_call(node, target, test, name, findings, counters)
    # deduplicated: a finding is an (example, keyword) pair, and the same pair can appear twice in one
    # docstring. Without this the printed count was 24 while `--check-baseline`, which compares sets, said
    # 22 -- the instrument reporting one population and enforcing another. `checked` is NOT deduplicated
    # and is not meant to be: it counts occurrences, so the ratio is unique-defects over occurrences.
    return sorted(set(findings)), counters["checked"], blocks - unparsed


def self_test() -> int:
    """Run the PRODUCTION `scan()` against a synthetic package, two-sided.

    The first version of this re-implemented the predicate inline and never called `scan()`. Measured in the review
    of #2351: with `scan()` stubbed to return nothing it still printed OK and exited 0, and with `scan()` restricted
    to 11 of 182 modules both this and `--check-baseline` exited 0 while the population silently fell from 1029
    keyword arguments to 92. That is the shape `check_fail_fast.py`'s own self-test docstring warns about -- "a
    ratchet whose checks have gone inert reports a stable count and reads like success" -- and the siblings all call
    their production scanner. This one now does too.
    """
    import tempfile
    import textwrap

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        package = root / "docstring_kwargs_probe"
        package.mkdir()
        (package / "__init__.py").write_text("")
        (package / "carrier.py").write_text(
            textwrap.dedent(
                '''
                """Probe module."""


                def target(alpha=1, beta=2):
                    """A callee with two parameters.

                    >>> target(alpha=1)
                    >>> target(gamma=1)
                    """


                def flexible(**kwargs):
                    """Anything goes here, so nothing is checkable.

                    >>> flexible(whatever=1)
                    """
                '''
            )
        )
        sys.path.insert(0, str(root))
        try:
            findings, checked, blocks = scan("docstring_kwargs_probe")
        finally:
            sys.path.remove(str(root))
            for name in [n for n in sys.modules if n.startswith("docstring_kwargs_probe")]:
                del sys.modules[name]

    bad = [f for f in findings if "gamma" in f]
    good = [f for f in findings if "alpha" in f or "whatever" in f]
    if len(bad) != 1 or good or checked < 2 or blocks < 1:
        print(
            f"self-test FAILED: findings={findings}, checked={checked}, blocks={blocks}; "
            f"want exactly one finding naming `gamma`, none for `alpha` or the **kwargs callee"
        )
        return 1
    print(f"self-test OK: scan() flagged {bad[0].split(': ')[-1]} and left the real parameter and **kwargs alone")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--write-baseline", metavar="FILE", nargs="?", const=str(DEFAULT_BASELINE))
    parser.add_argument("--check-baseline", metavar="FILE", nargs="?", const=str(DEFAULT_BASELINE))
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    findings, checked, blocks = scan()
    print(
        f"docstring keyword arguments: {len(findings)} of {checked} checked name a parameter the callee "
        f"does not have, over {blocks} example blocks"
    )
    if args.list:
        for finding in findings:
            print(f"  {finding}")

    if args.write_baseline:
        Path(args.write_baseline).write_text(
            json.dumps(
                {
                    "_comment": __doc__.strip().splitlines()[0],
                    # the POPULATION, not only the verdict: a scanner that goes blind reports a stable finding
                    # list and a collapsed population, and without these the ratchet reads that as a pass
                    # (review of #2351 shrank the scan to 11 of 182 modules and both checks still exited 0)
                    "checked": checked,
                    "blocks": blocks,
                    "findings": findings,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"baseline written to {args.write_baseline}: {len(findings)} findings")
        return 0

    if args.check_baseline:
        path = Path(args.check_baseline)
        if not path.exists():
            print(f"CANNOT RUN: no baseline at {path}. Write one with --write-baseline.")
            return 2
        record = json.loads(path.read_text())
        recorded = set(record["findings"])
        current = set(findings)
        floor_checked, floor_blocks = record.get("checked", 0), record.get("blocks", 0)
        if checked < floor_checked * 0.9 or blocks < floor_blocks * 0.9:
            print(
                f"FAIL: the scan shrank -- {checked} keyword arguments over {blocks} blocks, against a recorded "
                f"{floor_checked} over {floor_blocks}. A smaller population is not a smaller problem: an "
                f"unimportable module is skipped at the `except BaseException` in `_import_all`, so a missing "
                f"optional dependency reads as a clean tree. Find what stopped importing before re-recording."
            )
            return 1
        new, gone = sorted(current - recorded), sorted(recorded - current)
        if new:
            print(f"FAIL: {len(new)} docstring example(s) newly pass a keyword the callee does not have:")
            for finding in new:
                print(f"  {finding}")
            print(
                f"\nFix the example, or -- if intended -- record it:\n"
                f"  python scripts/check_docstring_kwargs.py --write-baseline {path}"
            )
            return 1
        if gone:
            print(f"{len(gone)} finding(s) fixed; re-record the baseline so the gain is not encoded as always held:")
            for finding in gone:
                print(f"  {finding}")
            return 1
        print(f"docstring kwargs ratchet OK: {len(recorded)} known findings, none new")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
