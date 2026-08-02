"""The nightly notifier must have both edges, and exactly one must fire per run (#1806).

`notify-on-failure` had an opening edge and no closing one: it filed or commented when a run
was not green and did nothing when one was, so every issue it created outlived its cause.
#1703 stayed open at `priority: high` through eight consecutive green nightlies.

The condition pair is the part worth pinning. Two ways to get it wrong, both silent:

- a gap -- some result combination fires neither job, so a failure goes unreported or a
  green run leaves the issue open;
- an overlap -- some combination fires both, so a run files an issue and closes it.

`resolve-on-success` must also require both dependencies to be literally `'success'`. Using
`success()` or "not failure" would close the issue on a `cancelled` shard, which is the state
`notify-on-failure`'s own comment records for 98 of 108 historical runs -- precisely the runs
that most need it open.
"""

from __future__ import annotations

import itertools
import pathlib
import re

import pytest
import yaml

WORKFLOW = pathlib.Path(__file__).resolve().parents[2] / ".github" / "workflows" / "nightly.yml"

# Every state GitHub can report for a needed job.
RESULTS = ("success", "failure", "cancelled", "skipped")

NOTIFY = "notify-on-failure"
RESOLVE = "resolve-on-success"
DEPS = ("full-test-suite", "capability-matrix")


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text())


def _condition(job_name: str) -> str:
    """The job's `if:`, stripped of the ${{ }} wrapper and normalised to one line."""
    raw = _workflow()["jobs"][job_name]["if"]
    inner = re.sub(r"^\s*\$\{\{\s*|\s*\}\}\s*$", "", raw.strip(), flags=re.S)
    return " ".join(inner.split())


def _fires(condition: str, results: dict[str, str]) -> bool:
    """Evaluate a GitHub `if:` expression for the subset of syntax these two jobs use.

    Deliberately NOT a general evaluator: it handles `always()`, `needs.X.result`, string
    literals, `==`, `!=`, `&&`, `||`, and parentheses, and raises on anything else so a
    future condition using unsupported syntax fails loudly instead of being mis-evaluated.
    """
    expr = condition
    for dep, value in results.items():
        expr = expr.replace(f"needs.{dep}.result", repr(value))
    expr = expr.replace("always()", "True").replace("&&", " and ").replace("||", " or ")
    unresolved = re.findall(r"\b[a-z_]+\(\)|needs\.[\w.-]+", expr)
    if unresolved:
        raise AssertionError(f"unsupported syntax {unresolved} in {condition!r}")
    return bool(eval(expr))


def test_both_edges_exist():
    jobs = _workflow()["jobs"]
    assert NOTIFY in jobs, "the opening edge is gone"
    assert RESOLVE in jobs, "the closing edge is missing: an automated issue would outlive its cause, which is #1806"


@pytest.mark.parametrize(
    "results", [dict(zip(DEPS, combo, strict=True)) for combo in itertools.product(RESULTS, repeat=len(DEPS))]
)
def test_exactly_one_edge_fires_for_every_result_combination(results):
    notify = _fires(_condition(NOTIFY), results)
    resolve = _fires(_condition(RESOLVE), results)
    all_green = all(v == "success" for v in results.values())

    assert notify != resolve, (
        f"{results}: notify={notify} resolve={resolve} -- a gap leaves the run unreported or the "
        "issue stale; an overlap files and closes in the same run"
    )
    assert resolve is all_green, f"{results}: the closing edge must fire exactly when both deps are 'success'"


def test_closing_edge_does_not_treat_cancelled_as_green():
    """The specific state that made 98 of 108 historical runs report nothing."""
    assert not _fires(_condition(RESOLVE), {"full-test-suite": "cancelled", "capability-matrix": "success"})
    assert _fires(_condition(NOTIFY), {"full-test-suite": "cancelled", "capability-matrix": "success"})


def test_closing_edge_matches_the_automated_label_not_the_title_alone():
    """A human-filed issue sharing the title must survive: it carries no `automated` label."""
    script = _workflow()["jobs"][RESOLVE]["steps"][0]["with"]["script"]
    assert "labels: 'automated'" in script, "closing edge must filter on the `automated` label"
    assert "i.title === title" in script, "closing edge must also match the title"


def test_closing_edge_and_notifier_agree_on_the_title():
    """Two copies of a string that must match, so pin that they do."""
    jobs = _workflow()["jobs"]
    titles = {
        name: re.search(r"const title = '([^']+)'", jobs[name]["steps"][0]["with"]["script"]).group(1)
        for name in (NOTIFY, RESOLVE)
    }
    assert titles[NOTIFY] == titles[RESOLVE], f"the closing edge would never find the issue: {titles}"
