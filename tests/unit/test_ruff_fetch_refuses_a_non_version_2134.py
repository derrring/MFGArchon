"""#2134: the ruff-update workflow shipped an unauthenticated `curl` to api.github.com.

Unauthenticated, that endpoint allows 60 requests/hour per IP and GitHub-hosted runners share
addresses, so a rate-limited response is routine rather than exotic. Its body carries no
`tag_name`, `jq -r` prints the string "null", and "null" compares unequal to the pinned version --
so the workflow concluded an update was available and ran `pip install ruff==null`. The job failed
naming pip, monthly, on a schedule with no reader.

Authentication makes it rarer; the shape guard is what makes it legible when it happens anyway (a
token expires, the limit moves, the API changes). This exercises the guard, not the token: the
`case` is located by content in the parsed YAML rather than by line number, so it survives edits
above it, and it is run under a real bash.
"""

import re
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "check-ruff-updates.yml"


def _the_guard() -> str:
    """The version-shape `case`, taken from the parsed workflow so line moves cannot break this."""
    doc = yaml.safe_load(WORKFLOW.read_text())
    bodies = [step["run"] for job in doc["jobs"].values() for step in job["steps"] if isinstance(step.get("run"), str)]
    guards = [
        m.group(0) for b in bodies if (m := re.search(r'^\s*case "\$LATEST_VERSION" in.*?^\s*esac', b, re.S | re.M))
    ]
    assert len(guards) == 1, f"expected exactly one version-shape guard, found {len(guards)}"
    return guards[0]


@pytest.mark.parametrize(
    ("value", "accepted"),
    [
        ("0.16.0", True),
        ("0.17.4", True),
        ("1.0.0", True),
        # The observed failure. Not hypothetical: this is what `jq -r .tag_name` prints when the
        # body is a rate-limit error, and it is what reached `pip install ruff==`.
        ("null", False),
        ("", False),
        ("v0.16.0", False),  # the `sed 's/v//'` upstream failed to strip
        ("Not Found", False),
        ("API rate limit exceeded", False),
    ],
)
def test_the_guard_admits_versions_and_refuses_everything_else(value, accepted):
    script = f"LATEST_VERSION={value!r}\n{_the_guard()}\n"
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    if accepted:
        assert out.returncode == 0, f"{value!r} is a version but the guard refused it: {out.stdout}"
    else:
        assert out.returncode != 0, f"{value!r} is not a version but the guard let it through"
        assert "::error::" in out.stdout, "the refusal must be a GitHub annotation, not a bare exit"


def test_the_fetch_is_authenticated():
    """The guard is the net; authentication is what keeps the workflow off it in normal months."""
    doc = yaml.safe_load(WORKFLOW.read_text())
    fetches = [
        step
        for job in doc["jobs"].values()
        for step in job["steps"]
        if isinstance(step.get("run"), str) and "api.github.com" in step["run"]
    ]
    assert fetches, "no step fetches api.github.com -- has this workflow's shape changed?"
    for step in fetches:
        assert "Authorization" in step["run"], (
            f"step {step.get('name')!r} calls api.github.com without an Authorization header; "
            "unauthenticated is 60 req/hour per IP, shared across runners"
        )
        assert step.get("env", {}).get("GH_TOKEN"), (
            f"step {step.get('name')!r} sends an Authorization header but no GH_TOKEN is in scope"
        )
