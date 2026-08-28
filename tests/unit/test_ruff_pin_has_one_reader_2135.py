"""#2135: one reader and one writer for the ruff version pinned in `.pre-commit-config.yaml`.

Six sites read that version through five different expressions, and two wrote it. They disagreed
on what a ruff block may look like -- a comment between `repo:` and `rev:` is valid YAML that the
script's expression spans and every `grep -A1` misses -- so the writer could land a bump the
verifiers then read as absent, and #2123 had to be fixed once on each writer.

**The first version of this file pinned a spelling, not the property.** It searched three named
files for `grep -A\\s?1.*ruff-pre-commit`, which is exactly the four expressions the fix deleted.
Independent review wrote twelve second implementations that all passed it: `grep -A2`,
`grep --after-context=1`, `awk`, `yq`, `perl -pi`, `python -c`, a `sed -i` reached through a
variable, `sed ... > tmp && mv tmp config`, and the same expressions placed in `nightly.yml`, in a
new `scripts/*.sh`, or in a `.yaml` rather than `.yml`. A test that can only fail on the instance
already fixed is the inert kind `AGENTS.md` names.

What replaces it is a population and a predicate. The population is every file under `scripts/` and
`.github/`; the predicate is "names where the pin lives AND pipes it through a text tool or edits
it in place". That cannot be dodged by changing tool, flag spelling, or file, because a reader has
to name the thing it reads. Exceptions are an explicit allowlist with reasons, not regex holes, and
`test_the_scan_can_see_the_owners_call_sites` is the sentinel: a glob that stops selecting files
reports zero violations, which reads exactly like clean code.
"""

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "update_ruff_version.py"
OWNER_REL = "scripts/update_ruff_version.py"

# Where the pin lives. A reader or writer must name one of these; there is no way around it.
MENTIONS = re.compile(r"\.pre-commit-config\.yaml|ruff-pre-commit")
# Pulling a value out of a text file, by any of the tools anyone would reach for.
EXTRACTS = re.compile(r"\b(grep|awk|sed|yq|jq|perl|cut|tr|head|tail)\b|python3?\s+-c|\brev:")
# Editing it in place, including the two-step forms that are not `sed -i`.
WRITES = re.compile(
    r"\bsed\s+-i|\bperl\s+-pi\b|\byq\s+-i\b|\bwrite_text\b|\btee\b"
    r"|open\([^)]*[\"']w[\"']|\.write\(|\bmv\s+\S+\s+\S*pre-commit-config"
)
GOES_THROUGH_OWNER = re.compile(r"update_ruff_version")

# Every exception is a line and a reason. Adding one should take an argument, not a regex tweak.
ALLOWED = [
    (
        "git diff --numstat",
        "counts changed lines; its `awk` consumes numstat output, not the YAML",
    ),
]

SCANNED_SUFFIXES = {".py", ".sh", ".yml", ".yaml", ".toml", ""}


def _candidate_lines():
    """Every non-comment line under scripts/ and .github/ that touches the pin, minus the owner."""
    found = []
    files = 0
    for root in (REPO / "scripts", REPO / ".github"):
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in SCANNED_SUFFIXES:
                continue
            rel = str(path.relative_to(REPO))
            files += 1
            try:
                lines = path.read_text().splitlines()
            except (UnicodeDecodeError, OSError):
                continue
            for n, line in enumerate(lines, 1):
                if line.lstrip().startswith("#") or not MENTIONS.search(line):
                    continue
                if not (EXTRACTS.search(line) or WRITES.search(line)):
                    continue
                found.append((rel, n, line.strip()))
    return files, found


# A lock file records the ruff it resolved. That is a version of the pin living outside the owner,
# and unlike the call sites above it cannot be caught by scanning `scripts/` and `.github/`.
_LOCK_RUFF = re.compile(r'^name = "ruff"\nversion = "([^"]+)"', re.M)

_LOCK_WITH_RUFF = """version = 1
requires-python = ">=3.12"

[[package]]
name = "ruff"
version = "0.13.1"
source = { registry = "https://pypi.org/simple" }
"""


def _pinned_in_pre_commit() -> str:
    block = (REPO / ".pre-commit-config.yaml").read_text()
    match = re.search(r"ruff-pre-commit(?:\.git)?[^\n]*\n(?:\s*#[^\n]*\n)*\s*rev:\s*v?([0-9][^\s#]*)", block)
    assert match, "no ruff rev found in .pre-commit-config.yaml"
    return match.group(1)


def test_a_tracked_lock_does_not_become_a_second_pin():
    """#2138/#2147: `uv.lock` recorded ruff 0.13.1 against this file's 0.16.0.

    An interpreter carrying that toolchain went `GATE RED` on a two-file documentation diff, and the
    only line naming the cause was a WARN in ~800 lines of output. The lock is untracked as of #2138;
    #2167 will generate a current one, and this is what stops it coming back as a fourth spelling of
    the pin.

    The control is not optional. With no lock present the assertion below is vacuous, and a vacuous
    assertion passes just as loudly when the extractor is broken -- so the extractor is first made to
    find a version in a lock that has one.
    """
    assert _LOCK_RUFF.search(_LOCK_WITH_RUFF).group(1) == "0.13.1"

    lock = REPO / "uv.lock"
    if not lock.is_file():
        return
    found = _LOCK_RUFF.search(lock.read_text())
    if not found:
        return
    pin = _pinned_in_pre_commit()
    assert found.group(1) == pin, (
        f"uv.lock resolves ruff {found.group(1)} against the {pin} pinned in "
        ".pre-commit-config.yaml. Two pins for one tool; the gate runs one and warns about the "
        "other. Regenerate the lock, or see #2147."
    )


def test_the_scan_can_see_the_owners_call_sites():
    """The sentinel. Without it every assertion below passes on an empty population.

    `scripts/check_single_source.py` in this repository raises rather than reporting a count when
    its globs select nothing, and says why in its docstring: a broken search pattern returns 0 hits,
    and 0 hits reads exactly like clean code. The same hazard is here -- one wrong `rglob` and the
    two tests below go green over any number of second implementations.
    """
    files, _ = _candidate_lines()
    assert files > 20, f"the scan selected {files} files under scripts/ and .github/ -- glob broken"

    calls = [
        (str(p.relative_to(REPO)), n)
        for root in (REPO / "scripts", REPO / ".github")
        for p in root.rglob("*")
        if p.is_file() and p.suffix in SCANNED_SUFFIXES
        for n, line in enumerate(p.read_text(errors="ignore").splitlines(), 1)
        if "--print-current" in line and not line.lstrip().startswith("#")
    ]
    assert len(calls) >= 3, (
        f"only {len(calls)} call sites of `--print-current` are visible to this scan; "
        "the owner has four, so the scan is not reading what it thinks it is"
    )


def test_nothing_outside_the_owner_reads_or_writes_the_pin_directly():
    _, found = _candidate_lines()
    violations = [
        f"{rel}:{n}  {line[:100]}"
        for rel, n, line in found
        if rel != OWNER_REL and not GOES_THROUGH_OWNER.search(line) and not any(a in line for a, _ in ALLOWED)
    ]
    assert not violations, (
        "these reach into .pre-commit-config.yaml instead of calling "
        f"`{OWNER_REL} --print-current` / `--force`:\n  " + "\n  ".join(violations)
    )


def test_every_reader_survives_a_comment_in_the_block(tmp_path):
    """The behavioural half: the shape the two bumpers disagreed on (#2123).

    A comment between `repo:` and `rev:` is valid YAML. The script's expression spans it; every
    `grep -A1` returns empty on it. Whichever ran decided whether a bump landed.
    """
    (tmp_path / ".pre-commit-config.yaml").write_text(
        "repos:\n"
        "  - repo: https://github.com/astral-sh/ruff-pre-commit\n"
        "    # kept in step with the version in uv.lock\n"
        "    rev: v0.16.0\n"
        "    hooks:\n"
        "      - id: ruff\n"
    )
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--print-current"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "0.16.0"


def test_a_failure_leaves_stdout_empty(tmp_path):
    """stdout is the data channel; a diagnostic on it is captured as the value.

    All four call sites read this command inside `$(...)`. Three pair it with a fallback outside the
    substitution (`$(cmd) || VAR=""`), which clears the capture on a non-zero exit, so for them this
    is prophylactic. `check-ruff-updates.yml`'s post-bump read is the one that is not: `$(cmd ||
    true)` captures the stdout of a failing run, and a diagnostic there would arrive as a version.
    """
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--print-current"],
        cwd=tmp_path,  # no .pre-commit-config.yaml here
        capture_output=True,
        text=True,
    )
    assert out.returncode != 0, "a missing config must fail, not return something usable"
    assert out.stdout.strip() == "", f"stdout must stay empty on failure, got {out.stdout.strip()!r}"
    assert "pre-commit-config" in out.stderr, "the diagnostic must still reach stderr"


def test_no_call_site_puts_the_fallback_inside_the_substitution():
    """The regression this PR introduced and fixed mid-flight, which nothing else pins.

    `RUFF_VERSION=$(cmd || VAR="")` cannot see the exit status: the `||` runs inside the subshell,
    so the assignment captures whatever the failing command left on stdout. Non-empty, so a `-z`
    guard downstream stays quiet and the text travels on as a version -- #2134's shape, reached from
    the caller's side. The correct form puts the `||` outside: `VAR=$(cmd) || VAR=""`.
    """
    bad = re.compile(r"\$\([^)]*\|\|[^)]*=[^)]*\)")
    offenders = []
    for path in [
        *sorted((REPO / ".github" / "workflows").glob("*.yml")),
        REPO / "scripts" / "local_ci.sh",
    ]:
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if "update_ruff_version" in line and bad.search(line):
                offenders.append(f"{path.relative_to(REPO)}:{n}  {line.strip()[:100]}")
    assert not offenders, (
        "the `||` fallback is inside the command substitution, where it cannot see the exit "
        "status and captures stdout instead:\n  " + "\n  ".join(offenders)
    )
