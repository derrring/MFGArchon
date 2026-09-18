#!/usr/bin/env python3
"""Every keyword argument in a docstring example must be a real parameter of the callee it names.

#674 removed a positional constructor and left 36 of 37 docstring examples dead, 7 of them raising
`TypeError` on the removed form, invisible because nothing executed them. #2341 repeated it at one
line. The deprecation-removal programme (#2343) is that operation at much larger scale, so the rot
needs an instrument that does not require the examples to be executable -- and most of them are not:
measured at 408415d4, running the package's examples gives 1348 failures of 2439, of which 1153 are
`NameError` from examples written as narrative fragments that assume names from a neighbouring
docstring. Those can never pass under any runner, `--doctest-modules` included, so execution is a
documentation programme rather than a check. This is the half that can be checked today.

What it does NOT catch, stated so the number is not read as coverage: a positional argument that
moved, a keyword whose meaning changed while the name survived, a callee it cannot resolve (a local
name, a method on an instance built earlier in the example), and anything in a `**kwargs` signature.
It resolves the callee in the example's own module namespace first, because a bare-name map resolved
`MFGComponents` to the wrong class and manufactured 7 findings that do not exist.

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


def scan(package_name: str = "mfgarchon") -> tuple[list[str], int, int]:
    """Returns (findings, keyword arguments checked, example blocks seen)."""
    warnings.simplefilter("ignore")
    package = importlib.import_module(package_name)
    modules = _import_all(package)
    fallback = _fallback_names(modules)
    findings: list[str] = []
    checked = blocks = 0
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
                continue  # an illustrative fragment, not python
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not node.keywords:
                    continue
                name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", None)
                if name is None:
                    continue
                target = vars(mod).get(name) or test.globs.get(name) or fallback.get(name)
                if not (inspect.isclass(target) or inspect.isfunction(target)):
                    continue
                try:
                    signature = inspect.signature(target)
                except (ValueError, TypeError):
                    continue
                if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
                    continue  # **kwargs accepts anything; nothing to check
                for keyword in node.keywords:
                    if keyword.arg is None:
                        continue
                    checked += 1
                    if keyword.arg not in signature.parameters:
                        findings.append(f"{test.name}: {name}({keyword.arg}=...)")
    # deduplicated: a finding is an (example, keyword) pair, and the same pair can appear twice in one
    # docstring. Without this the printed count was 24 while `--check-baseline`, which compares sets, said
    # 22 -- the instrument reporting one population and enforcing another.
    return sorted(set(findings)), checked, blocks


def self_test() -> int:
    """The check must fire on a keyword that is not a parameter and stay silent on one that is."""

    def target(alpha: int = 1, beta: int = 2) -> None: ...

    def probe(source: str) -> int:
        hits = 0
        tree = ast.parse(
            "".join(e.source for e in doctest.DocTestParser().get_doctest(source, {}, "p", "p", 0).examples)
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and node.keywords:
                signature = inspect.signature(target)
                hits += sum(1 for kw in node.keywords if kw.arg not in signature.parameters)
        return hits

    good, bad = probe(">>> target(alpha=1)\n"), probe(">>> target(gamma=1)\n")
    if good != 0 or bad != 1:
        print(f"self-test FAILED: a real parameter scored {good} (want 0), an absent one {bad} (want 1)")
        return 1
    print("self-test OK: fires on a keyword that is not a parameter, silent on one that is")
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
            json.dumps({"_comment": __doc__.strip().splitlines()[0], "findings": findings}, indent=2) + "\n"
        )
        print(f"baseline written to {args.write_baseline}: {len(findings)} findings")
        return 0

    if args.check_baseline:
        path = Path(args.check_baseline)
        if not path.exists():
            print(f"CANNOT RUN: no baseline at {path}. Write one with --write-baseline.")
            return 2
        recorded = set(json.loads(path.read_text())["findings"])
        current = set(findings)
        new, gone = sorted(current - recorded), sorted(recorded - current)
        if new:
            print(f"FAIL: {len(new)} docstring example(s) newly pass a keyword the callee does not have:")
            for finding in new:
                print(f"  {finding}")
            print(f"\nFix the example, or -- if intended -- record it:\n  python {path.name} --write-baseline")
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
