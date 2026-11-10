"""psutil is required: imported plainly, and never guarded, skipped or removed; and README lists what pyproject requires.

User rulings, 2026-11-10 and 2026-11-11 (#2930), and AGENTS.md's package line: `[project].dependencies` is the one list
of what an install guarantees; a package on it is imported plainly, and a missing one is Python's own
`ModuleNotFoundError`. This stage of #2930 covers psutil; its stage B brings the same rule to every required package,
and until then the library still guards the others (#2930's census, by stage).

**The rule for psutil is lexical.** It reads no handler, so no spelling of a handler can get round it. It matches the
name "psutil" and any "psutil.<something>".

- L1. In `mfgarchon/` and in `tests/`, no `import psutil...` or `from psutil... import` sits inside a `try` or a `with`, at any depth:
  its body, a handler, its `else`, its `finally`, a nested block or a class body. The body of a `def`, `async def` or
  `lambda` is a new start, since a `try` around a `def` does not guard what the function imports when it is called.
- L2. In `mfgarchon/`, no string constant equals "psutil" or begins with "psutil.": that is every dynamic import,
  presence check and name comparison at once. A docstring that mentions psutil in prose is not equal to it.
- T1. In `tests/`, no `importorskip` call has an argument or keyword equal to "psutil".
- T2. In `tests/`, no statement both mentions `sys.modules` and holds the literal "psutil". A statement's own
  expressions count, not the statements inside it, so a `with` header counts and its body is judged on its own.

It over-reports by design, and has no exceptions. At this stage's head it reports nothing. On main before #2930
it reports the five function guards (L1) and `utils/__init__.py`'s two `err.name != "psutil"` (L2). T1 and T2 have no
row on main either, so their evidence is the planted cases below, each of which fails the rule with its own message.

**What it does not cover, and who does:**
- a guard in another module around an internal module that imports psutil at its top, such as
  `try: from .memory_management import ...`: `test_no_import_guard_hides_a_missing_name_2088.py` reports that, and its
  `ALLOWED` is empty;
- a conditional import that is not a `try` or a `with`, such as `if sys.platform == "linux": import psutil`: by design,
  since it does not decide what happens without the package;
- psutil reached through a computed name, such as `importlib.import_module("ps" + "util")`;
- an aliased `sys` in a test, such as `import sys as s; s.modules["psutil"] = None`;
- code that is not a `.py` file, and `benchmarks/` and `examples/`, which #2930's stage C covers.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
NAME = "psutil"


def _normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def required_distributions(pyproject: Path) -> list[str]:
    """The distribution names in `[project].dependencies`, by the PEP 508 name rule."""
    names = []
    for dep in tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["dependencies"]:
        assert ";" not in dep, (
            f"{dep!r} carries an environment marker, so it is not required on every install. Decide whether it is "
            "required, and teach this test which installs it covers."
        )
        match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", dep)
        assert match, f"cannot read a distribution name from {dep!r}"
        names.append(_normalise(match.group(1)))
    return names


def _is_psutil(value: object) -> bool:
    return isinstance(value, str) and (value == NAME or value.startswith(NAME + "."))


def _imports_psutil(node: ast.AST) -> bool:
    if isinstance(node, ast.Import):
        return any(_is_psutil(alias.name) for alias in node.names)
    return isinstance(node, ast.ImportFrom) and node.level == 0 and _is_psutil(node.module)


def library_rows(source: str, rel: str) -> list[tuple[str, int, str]]:
    """(file, line, rule) for L1 and L2 in one library file."""
    rows = []

    def visit(node: ast.AST, guarded: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if guarded and _imports_psutil(child):
                rows.append((rel, child.lineno, "L1"))
            elif isinstance(child, ast.Constant) and _is_psutil(child.value):
                rows.append((rel, child.lineno, "L2"))
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
                visit(child, False)
            else:
                visit(child, guarded or isinstance(child, ast.Try | ast.TryStar | ast.With | ast.AsyncWith))

    visit(ast.parse(source), False)
    return rows


def _own_parts(statement: ast.stmt) -> list[ast.AST]:
    """A statement's own expressions, without the statements nested in it."""
    return [
        part
        for fieldname, value in ast.iter_fields(statement)
        if fieldname not in ("body", "orelse", "finalbody", "handlers", "cases")
        for part in (value if isinstance(value, list) else [value])
        if isinstance(part, ast.AST)
    ]


def test_rows(source: str, rel: str) -> list[tuple[str, int, str]]:
    """(file, line, rule) for T1, T2 and L1 in one test file. L2 does not apply: tests name psutil in strings."""
    rows = [row for row in library_rows(source, rel) if row[2] == "L1"]
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            func = node.func
            called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            values = [*node.args, *(k.value for k in node.keywords)]
            if called == "importorskip" and any(isinstance(v, ast.Constant) and _is_psutil(v.value) for v in values):
                rows.append((rel, node.lineno, "T1"))
        if isinstance(node, ast.stmt):
            parts = [x for part in _own_parts(node) for x in ast.walk(part)]
            names_modules = any(
                isinstance(x, ast.Attribute) and x.attr == "modules" and getattr(x.value, "id", None) == "sys"
                for x in parts
            )
            if names_modules and any(isinstance(x, ast.Constant) and _is_psutil(x.value) for x in parts):
                rows.append((rel, node.lineno, "T2"))
    return rows


test_rows.__test__ = False  # a helper whose name pytest would otherwise collect

RULES = {
    "L1": "imports psutil inside a `try` or a `with`",
    "L2": "names psutil in a string, the shape of a dynamic import, a presence check or a name comparison",
    "T1": "skips a test without psutil",
    "T2": "makes psutil absent through `sys.modules`",
}


def rule_rows(root: Path) -> list[tuple[str, int, str]]:
    rows = []
    for path in sorted((root / "mfgarchon").rglob("*.py")):
        rows += library_rows(path.read_text(encoding="utf-8"), path.relative_to(root).as_posix())
    for path in sorted((root / "tests").rglob("*.py")):
        rows += test_rows(path.read_text(encoding="utf-8"), path.relative_to(root).as_posix())
    return rows


def message(rows: list[tuple[str, int, str]]) -> str:
    return "psutil is required (#2930): import it plainly, and do not decide what happens without it.\n" + "\n".join(
        f"  {rel}:{line}: {rule}, {RULES[rule]}" for rel, line, rule in rows
    )


def test_psutil_is_imported_plainly_and_never_removed():
    rows = rule_rows(REPO)
    assert rows == [], message(rows)


#: case -> (the directory the source would sit in, the source, the rule that must report it, at that line). The re-adds are the
#: reviewers' own mutants of #2943, copied from their mutant files with only the indentation changed.
PLANTED = {
    "review 1, suppress_watchdog": (
        "mfgarchon",
        "def _check_memory_usage(self) -> float:\n    import contextlib\n\n    with contextlib.suppress(ImportError):\n"
        "        import psutil\n\n        return psutil.Process().memory_info().rss / (1024**3)\n    return 0.0\n",
        ("L1", 5),
    ),
    "review 1, import_module_debug": (
        "mfgarchon",
        "def _get_memory_usage_mb(self) -> float:\n    import importlib\n\n    try:\n"
        '        psutil = importlib.import_module("psutil")\n    except ImportError:\n        return 0.0\n'
        "    return psutil.Process().memory_info().rss / (1024 * 1024)\n",
        ("L2", 5),
    ),
    "review 1, errs_name_numpy_backend": (
        "mfgarchon",
        "def memory_usage(self) -> dict:\n    _ERRS = (ImportError, OSError)\n    try:\n        import psutil\n"
        "    except _ERRS:\n        return None\n\n    process = psutil.Process()\n",
        ("L1", 4),
    ),
    "review 1, test_patch_dict": (
        "tests",
        "import sys\nfrom unittest import mock\n\ndef test_x():\n    with mock.patch.dict(sys.modules, {'psutil': None}):\n"
        "        pass\n",
        ("T2", 5),
    ),
    "review 2, R12_break_out_of_reraise_handler": (
        "mfgarchon",
        "def _check_memory_usage(self) -> float:\n    for _attempt in (1,):\n        try:\n            import psutil\n"
        "        except ImportError:\n            if _attempt:\n                break\n            raise\n"
        "        return psutil.Process().memory_info().rss / (1024**3)\n    return 0.0\n",
        ("L1", 4),
    ),
    "review 2, R9_class_body_guard_psutil": (
        "mfgarchon",
        "try:\n\n    class _Probe:\n        import psutil\n\nexcept ImportError:\n    _Probe = None\n",
        ("L1", 4),
    ),
    "review 1, test_side_try_guard": (
        "tests",
        "import pytest\ntry:\n    import psutil\n    HAS_PSUTIL = True\nexcept ImportError:\n    HAS_PSUTIL = False\n\n"
        "@pytest.mark.skipif(not HAS_PSUTIL, reason='psutil not installed')\ndef test_x():\n    assert psutil.Process()\n",
        ("L1", 3),
    ),
    "an importorskip in a test": (
        "tests",
        'import pytest\n\npsutil = pytest.importorskip("psutil")\n',
        ("T1", 3),
    ),
    "a from-import of psutil under a try": (
        "mfgarchon",
        "try:\n    from psutil import Process\nexcept ImportError:\n    Process = None\n",
        ("L1", 2),
    ),
}

#: Sources the rule must not report: an import in a def defined under a try, a docstring that names psutil, a guarded
#: package that is not psutil, and a local module that happens to be named psutil.
NOT_REPORTED = {
    "an import inside a def that a try defines": "try:\n    def later():\n        import psutil\nexcept ImportError:\n"
    "    later = None\n",
    "a docstring that names psutil": 'def f():\n    """Measure memory with psutil."""\n',
    "a guarded optional package": "try:\n    import torch\nexcept ImportError:\n    torch = None\n",
    "a relative import of a local module named psutil": "try:\n    from .psutil import x\nexcept ImportError:\n    x = None\n",
}


@pytest.mark.parametrize("case", sorted(PLANTED))
def test_each_planted_re_add_fails_the_rule(case):
    where, source, (rule, line) = PLANTED[case]
    rows = (library_rows if where == "mfgarchon" else test_rows)(source, f"{where}/planted.py")
    assert (f"{where}/planted.py", line, rule) in rows, (
        f"{case}: the rule reports {rows}, and should report {rule} at {line}"
    )
    assert f"{where}/planted.py:{line}: {rule}, {RULES[rule]}" in message(rows)


@pytest.mark.parametrize("case", sorted(NOT_REPORTED))
def test_the_rule_does_not_report(case):
    assert library_rows(NOT_REPORTED[case], "mfgarchon/planted.py") == []


def test_a_dependency_with_a_marker_is_refused(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "x"\ndependencies = ["numpy; python_version >= \'3.12\'"]\n')
    with pytest.raises(AssertionError, match="environment marker"):
        required_distributions(pyproject)


_PROBE = """
import importlib, importlib.abc, importlib.util, sys
sys.path.insert(0, {repo!r})
ABSENT = {absent!r}

class _Absent(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] == ABSENT:
            raise ModuleNotFoundError(f"No module named {{name!r}}", name=name)
        return None

sys.meta_path.insert(0, _Absent())
_find_spec = importlib.util.find_spec
importlib.util.find_spec = lambda name, package=None: None if name.split(".")[0] == ABSENT else _find_spec(name, package)

import mfgarchon
print("TREE:" + mfgarchon.__file__)
"""

#: The required packages whose absence is already Python's own error at `import mfgarchon`. #2930's later stages add
#: theirs as their guards go.
FAILS_AT_IMPORT = ["psutil"]


def _import_without(absent: str) -> subprocess.CompletedProcess:
    env = {**os.environ}
    env.pop("PYTHONSAFEPATH", None)
    code = _PROBE.format(repo=str(REPO), absent=absent)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300, env=env)


@pytest.mark.parametrize("absent", FAILS_AT_IMPORT)
def test_import_mfgarchon_fails_without_a_required_package(absent):
    proc = _import_without(absent)
    assert proc.returncode != 0, f"`import mfgarchon` succeeded without {absent}, which [project].dependencies requires"
    last = proc.stderr.strip().splitlines()[-1]
    assert last.startswith(f"ModuleNotFoundError: No module named '{absent}'"), (
        f"`import mfgarchon` without {absent} must end in Python's own ModuleNotFoundError, not a retyped error:\n"
        f"{proc.stderr[-1500:]}"
    )
    assert str(REPO / "mfgarchon") in proc.stderr, (
        f"the traceback does not run through this tree:\n{proc.stderr[-1500:]}"
    )


def test_the_probe_imports_when_what_it_refuses_is_not_used():
    """The control: refusing a name nothing imports leaves `import mfgarchon` working, from this tree."""
    proc = _import_without("a_package_nothing_imports")
    assert proc.returncode == 0, proc.stderr[-1500:]
    reported = [line[len("TREE:") :] for line in proc.stdout.splitlines() if line.startswith("TREE:")]
    assert reported, proc.stdout
    assert Path(reported[0]).resolve().is_relative_to((REPO / "mfgarchon").resolve()), proc.stdout


def default_install_names(readme: str) -> list[str]:
    lines = [line for line in readme.splitlines() if line.startswith("**Default install** includes:")]
    assert len(lines) == 1, f"README.md should have one '**Default install** includes:' line, found {len(lines)}"
    listed = lines[0].split(":", 1)[1].strip().rstrip(".")
    return [_normalise(name.strip()) for name in listed.split(",")]


def readme_disagreement(readme: str, pyproject: Path) -> tuple[list[str], list[str], list[str]]:
    """(named in README and not required, required and not named in README, named twice in README)."""
    named, required = default_install_names(readme), required_distributions(pyproject)
    twice = sorted({name for name in named if named.count(name) > 1})
    return sorted(set(named) - set(required)), sorted(set(required) - set(named)), twice


def test_readme_lists_exactly_the_required_packages():
    extra, missing, twice = readme_disagreement(
        (REPO / "README.md").read_text(encoding="utf-8"), REPO / "pyproject.toml"
    )
    assert (extra, missing, twice) == ([], [], []), (
        f"README.md's '**Default install** includes:' line disagrees with [project].dependencies: it names {extra} "
        f"that pyproject does not require, leaves out {missing}, and names {twice} twice. pyproject is the owner; "
        "correct the README line."
    )


def test_the_readme_comparison_sees_a_name_too_many_one_too_few_and_one_twice():
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    line = next(line for line in readme.splitlines() if line.startswith("**Default install** includes:"))
    planted = readme.replace(line, line.rstrip(".") + ", a-package-pyproject-never-listed, meshio.")
    planted = planted.replace(", psutil", "")
    assert readme_disagreement(planted, REPO / "pyproject.toml") == (
        ["a-package-pyproject-never-listed"],
        ["psutil"],
        ["meshio"],
    )
