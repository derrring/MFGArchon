"""#2134: the ruff-update workflow read the releases API unauthenticated and used the reply unchecked.

Unauthenticated, `api.github.com` allows 60 requests/hour per IP and GitHub-hosted runners share
addresses, so a rate-limited response is routine. Its body carries no `tag_name`, `jq -r` prints the
string "null", and "null" compares unequal to the pinned version -- so the workflow concluded an
update was available and the next step ran `pip install ruff==null`. The job failed monthly, on a
schedule with no reader, naming pip rather than the API.

Authentication makes it rare; the shape check is what makes it legible when it happens anyway (a
token expires, the limit moves, a prerelease is mismarked). Note that `-e` does not help here: a
`run:` block with no `shell:` key runs under `bash -e {0}`, without `pipefail`, so the exit status
of `curl … | jq … | sed …` is `sed`'s and is 0 however badly `curl` failed. The check is the only
thing between the API and `$GITHUB_OUTPUT`.

**This runs the step, rather than inspecting it.** An earlier version located the guard with a
regex for `case "$LATEST_VERSION" in` and asserted exactly one existed. Independent review broke it
four ways: replacing the `case` with a *stricter* `grep -qE` turned eight tests red, a `uses:` job
anywhere in the file raised `KeyError: 'steps'`, moving the guard *after* `$GITHUB_OUTPUT` stayed
green while contradicting the docstring, and a duplicated guard inside one block stayed green
because the count was over blocks. Executing the block with `curl` and `jq` stubbed pins the
behaviour the issue is about and survives any refactor that keeps it.
"""

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "check-ruff-updates.yml"


def _fetch_step() -> dict:
    """The step that calls the releases API, located by what it does, not by name or position."""
    doc = yaml.safe_load(WORKFLOW.read_text())
    steps = [
        step
        for job in doc["jobs"].values()
        for step in job.get("steps", [])  # a `uses:`-only job has no `steps`
        if isinstance(step.get("run"), str) and "api.github.com" in step["run"]
    ]
    assert len(steps) == 1, f"expected one step fetching api.github.com, found {len(steps)}"
    return steps[0]


def _run_step(tmp_path: Path, api_body: str):
    """Execute the step's shell under `bash -e`, with `curl` stubbed to return `api_body`."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "curl"
    stub.write_text("#!/bin/sh\ncat <<'BODY'\n" + api_body + "\nBODY\n")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)

    output = tmp_path / "github_output"
    output.write_text("")
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "GH_TOKEN": "stub-token",
        "GITHUB_OUTPUT": str(output),
    }
    done = subprocess.run(
        ["bash", "-e", "-c", _fetch_step()["run"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    return done, output.read_text()


@pytest.mark.skipif(shutil.which("jq") is None, reason="the step pipes through jq")
@pytest.mark.parametrize(
    ("body", "accepted", "why"),
    [
        ('{"tag_name": "v0.17.0"}', True, "an ordinary release"),
        ('{"tag_name": "0.17.0"}', True, "already unprefixed"),
        # The observed failure: this is what the API returns when the limit is hit.
        ('{"message": "API rate limit exceeded", "documentation_url": "…"}', False, "rate limited"),
        ('{"message": "Bad credentials"}', False, "the token expired"),
        ("", False, "an empty body -- a network failure"),
        ("<html>502 Bad Gateway</html>", False, "an HTML error page"),
        # A prerelease the maintainer forgot to mark. The `case` glob this replaced admitted it,
        # and `--force` then wrote `rev: v0.17.0rc1` to disk before raising over it.
        ('{"tag_name": "v0.17.0rc1"}', False, "a mismarked prerelease"),
        ('{"tag_name": "v0.17.0-beta.1"}', False, "a tagged beta"),
        # `sed 's/v//'` unanchored turned this into `0.14.0-preiew`, which any loose check admits.
        ('{"tag_name": "v0.14.0-preview"}', False, "a preview tag, and the `v` inside the word"),
        ('{"tag_name": "1.2.3.4.5.6"}', False, "not three components"),
    ],
)
def test_the_step_refuses_anything_that_is_not_a_version(tmp_path, body, accepted, why):
    done, written = _run_step(tmp_path, body)
    if accepted:
        assert done.returncode == 0, f"{why}: {done.stdout}{done.stderr}"
        assert "version=" in written, f"{why}: nothing reached GITHUB_OUTPUT"
    else:
        assert done.returncode != 0, f"{why}: the step accepted {body!r}"
        assert "::error::" in done.stdout, f"{why}: the refusal must be a GitHub annotation"
        # The ordering assertion, which an inspection test could not make: a refused value must
        # never reach the output, because every downstream step reads it from there.
        assert written == "", f"{why}: a refused value reached GITHUB_OUTPUT as {written!r}"


def test_the_fetch_is_authenticated():
    """The guard is the net; authentication is what keeps the workflow off it in normal months.

    Deliberately loose about *how*: an `Authorization` header and a token in scope, whether the
    token sits on the step or on the job, and `gh api` counts as authenticated because it reads
    `GH_TOKEN` itself. An earlier version required the header on the step and `env` on the step, and
    went red on both of those legitimate alternatives.
    """
    doc = yaml.safe_load(WORKFLOW.read_text())
    for job in doc["jobs"].values():
        job_env = job.get("env", {}) or {}
        for step in job.get("steps", []):
            run = step.get("run")
            if not isinstance(run, str) or "api.github.com" not in run:
                continue
            step_env = step.get("env", {}) or {}
            in_scope = {**job_env, **step_env}
            authenticated = "Authorization" in run or "gh api" in run
            assert authenticated, (
                f"step {step.get('name')!r} calls api.github.com without authentication; "
                "unauthenticated is 60 requests/hour per IP, shared across runners"
            )
            assert any(k.endswith("TOKEN") for k in in_scope), (
                f"step {step.get('name')!r} authenticates but no token is in scope "
                f"(step env: {sorted(step_env)}, job env: {sorted(job_env)})"
            )
