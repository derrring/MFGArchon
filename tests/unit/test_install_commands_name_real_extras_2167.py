"""Every extra and dependency-group named by an install command must exist (#2167, #1658).

This defect has already shipped here. `nightly.yml` referenced a `numerical` extra for three and a
half months while `pyproject.toml` declared no such extra; pip installs an unknown extra with a
warning and a zero exit, so the SOCP gate failed on `ImportError` every night and burned one of ten
`--maxfail` slots before anyone read the warning (#1658). Nothing has guarded it since.

#2167 makes the class wider, not narrower: development tooling moved out of
`[project.optional-dependencies]` into PEP 735 `[dependency-groups]`, so `-e .[dev]` is now exactly
that failure -- an extra that does not exist, installed with a warning nobody reads. Eleven call
sites were rewritten. This is what stops the twelfth.

**The population is the thing that can be wrong here.** A guard that scans the wrong set of files
reports zero violations, which reads exactly like a clean repository, so
`test_the_scan_finds_the_known_call_sites` is the sentinel: if the globs stop selecting files, it
fails rather than the repository silently going unchecked.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

#: `pip install mfgarchon[a,b]`, `pip install -e ".[a,b]"`, `uv pip install -e .[a]`.
EXTRAS = re.compile(r"""(?:mfgarchon|-e\s+["']?\.)["']?\[([A-Za-z0-9_,.\- ]+)\]""")
#: `--group dev`, `--only-group docs`, `uv sync --group dev`.
GROUPS = re.compile(r"--(?:only-)?group[= ]+([A-Za-z0-9_.\-]+)")

#: Where an install command can live. Extension-free entries are deliberate: `Makefile` has no
#: suffix, and restricting to a suffix list is how a population predicate silently loses a file.
ROOTS = (".github", "scripts", "docs")
FILES = ("Makefile", "README.md", "pyproject.toml", "CONTRIBUTING.md")


def _declared() -> tuple[set[str], set[str]]:
    data = tomllib.loads((REPO / "pyproject.toml").read_text())
    return (
        set(data["project"].get("optional-dependencies", {})),
        set(data.get("dependency-groups", {})),
    )


def _scan() -> tuple[int, list[tuple[str, int, str, str]]]:
    """Every (file, line, kind, name) an install command names. Kind is `extra` or `group`."""
    paths: list[Path] = []
    for root in ROOTS:
        base = REPO / root
        if base.is_dir():
            paths += [p for p in sorted(base.rglob("*")) if p.is_file()]
    paths += [REPO / name for name in FILES if (REPO / name).is_file()]
    paths += sorted((REPO / "mfgarchon").rglob("*.py"))

    found = []
    for path in paths:
        try:
            text = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        rel = str(path.relative_to(REPO))
        for n, line in enumerate(text.splitlines(), 1):
            for match in EXTRAS.finditer(line):
                for name in match.group(1).split(","):
                    if name.strip():
                        found.append((rel, n, "extra", name.strip()))
            for match in GROUPS.finditer(line):
                found.append((rel, n, "group", match.group(1)))
    return len(paths), found


def test_the_scan_finds_the_known_call_sites():
    """Sentinel. A glob that stops selecting files reports zero violations, which reads as clean."""
    count, found = _scan()
    assert count > 20, f"the scan selected only {count} files; the globs are wrong, not the repo"
    files = {rel for rel, _, _, _ in found}
    assert any(f.startswith(".github/workflows/") for f in files), "no workflow install command seen"
    assert "pyproject.toml" in files, "pyproject's own documented install commands not seen"
    groups = {name for _, _, kind, name in found if kind == "group"}
    assert "dev" in groups, "the `dev` group is installed by CI and the scan did not see it"


#: Sites that already named an undeclared extra before this guard existed, pinned as an exact set so
#: a new one fails immediately and fixing one requires deleting its entry. These are user-facing
#: `install_cmd` strings printed when an optional dependency is missing -- `pip install
#: mfgarchon[neural]` warns and exits zero, so the user follows the advice, sees success, and still
#: cannot import torch. Tracked in #2170, which decides between declaring the extras and pointing at
#: the ones that exist.
#:
#: Keyed by (file, name) and not by line: a line number in a durable artifact expires on the next
#: edit above it, and this set has to survive unrelated changes to those files.
KNOWN_UNDECLARED = frozenset(
    {
        ("mfgarchon/alg/optimization/optimal_transport/__init__.py", "optimization"),
        ("mfgarchon/alg/optimization/optimal_transport/sinkhorn_solver.py", "optimization"),
        ("mfgarchon/alg/optimization/optimal_transport/wasserstein_solver.py", "optimization"),
        ("mfgarchon/backends/__init__.py", "jax"),
        ("mfgarchon/backends/jax_backend.py", "jax"),
        ("mfgarchon/core/mfg_problem.py", "geometry"),
        ("mfgarchon/utils/dependencies.py", "neural"),
        ("mfgarchon/utils/dependencies.py", "performance"),
        ("mfgarchon/utils/dependencies.py", "reinforcement"),
        ("mfgarchon/utils/dependencies.py", "visualization"),
        ("mfgarchon/utils/dependencies.py", "gpu"),
    }
)


def _undeclared() -> list[tuple[str, int, str, str]]:
    extras, groups = _declared()
    _, found = _scan()
    return [(rel, n, kind, name) for rel, n, kind, name in found if name not in (extras if kind == "extra" else groups)]


def test_the_known_set_is_exact():
    """A ratchet, not a floor. Fixing one of #2170's sites must delete its entry here.

    Asserted as equality in both directions: a floor would let the set rot into a list of things
    that were fixed years ago, and nothing would say so.
    """
    seen = {(rel, name) for rel, _, kind, name in _undeclared() if kind == "extra"}
    assert seen == set(KNOWN_UNDECLARED), (
        "the known-undeclared set no longer matches the tree.\n"
        f"    gone (delete from KNOWN_UNDECLARED): {sorted(set(KNOWN_UNDECLARED) - seen)}\n"
        f"    new  (fix it, or add it with an issue): {sorted(seen - set(KNOWN_UNDECLARED))}"
    )


def test_every_named_extra_and_group_exists():
    extras, groups = _declared()
    bad = [(rel, n, kind, name) for rel, n, kind, name in _undeclared() if (rel, name) not in KNOWN_UNDECLARED]
    assert not bad, "install commands naming something pyproject.toml does not declare:\n" + "\n".join(
        f"    {rel}:{n}  {kind} {name!r}  (declared {kind}s: {sorted(extras if kind == 'extra' else groups)})"
        for rel, n, kind, name in bad
    )


@pytest.mark.parametrize(
    ("line", "kind", "name"),
    [
        ("        pip install -e . --group dev", "group", "dev"),
        ('          pip install -e ".[all]"', "extra", "all"),
        ("pip install mfgarchon[nn]", "extra", "nn"),
        ("    uv sync --group=docs", "group", "docs"),
        ('    "install_cmd": "pip install mfgarchon[numerical]",', "extra", "numerical"),
    ],
)
def test_the_extractors_read_the_forms_that_are_actually_used(line, kind, name):
    """Control on the regexes themselves.

    Both assertions above are satisfied by an extractor that finds nothing: one asserts an empty
    difference, the other asserts membership over an empty set. These are the spellings in this
    repository today, one per shape.
    """
    got = [n for n in EXTRAS.findall(line) for n in n.split(",")] if kind == "extra" else GROUPS.findall(line)
    assert name in [g.strip() for g in got], f"{kind} extractor missed {line!r}"


def test_a_nonexistent_name_is_caught():
    """The failing direction, on the shape that actually shipped: #1658's phantom extra."""
    extras, _ = _declared()
    assert "definitely-not-an-extra" not in extras
    line = 'pip install -e ".[definitely-not-an-extra]"'
    assert EXTRAS.findall(line) == ["definitely-not-an-extra"]


def test_dev_tooling_is_a_group_and_backends_are_extras():
    """The distinction #2167 turns on, and getting it backwards makes the backends uninstallable.

    A group is read from the source tree and never reaches the built distribution's metadata, so a
    downstream `pip install mfgarchon[nn]` can only be satisfied by an extra.
    """
    extras, groups = _declared()
    assert {"dev", "docs"} <= groups, "development tooling must be a dependency-group"
    assert not ({"dev", "docs"} & extras), "development tooling must not also be an extra"
    assert {"nn", "all"} <= extras, "compute backends must stay installable as extras"
    assert not ({"nn", "all"} & groups), "a backend in a group cannot be installed by a user"


def test_the_python_floor_for_the_group_flag_is_documented():
    """`--group` is pip >= 25.1. Every workflow upgrades pip first; that is load-bearing now."""
    for name in ("ci.yml", "nightly.yml", "security.yml", "python-compat.yml"):
        path = REPO / ".github" / "workflows" / name
        if not path.is_file():
            continue
        text = path.read_text()
        if "--group" not in text:
            continue
        assert "install --upgrade pip" in text, (
            f"{name} installs a dependency-group but never upgrades pip; `--group` needs pip >= 25.1 "
            "and the runner's bundled pip may be older"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
