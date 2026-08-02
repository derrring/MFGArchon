"""The nightly notifier must have both edges, and exactly one must fire per run (#1806).

`notify-on-failure` had an opening edge and no closing one: it filed or commented when a run
was not green and did nothing when one was, so every issue it created outlived its cause.
#1703 stayed open at `priority: high` through eight consecutive green nightlies.

The condition pair is the part worth pinning. Two ways to get it wrong, both silent:

- a gap -- some result combination fires neither job, so a failure goes unreported or a
  green run leaves the issue open;
- an overlap -- some combination fires both, so a run files an issue and closes it.

`resolve-on-success` must also require both dependencies to be literally `'success'`.
(The evaluator models GitHub's implicit `success()`; see `_fires`. Its unsupported-syntax
guard catches tokens that are present and unmodelled -- never semantics that are absent.) Using
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


STATUS_CHECKS = ("success()", "always()", "cancelled()", "failure()")


def _fires(condition: str, results: dict[str, str]) -> bool:
    """Evaluate a GitHub `if:` expression for the subset of syntax these two jobs use.

    Models GitHub's **implicit status check**, which is the part that bites: per
    `expressions.md`, *"A default status check of `success()` is applied unless you include
    one of these functions"* -- `success()`, `always()`, `cancelled()`, `failure()`. So a
    condition with no status-check function is silently ANDed with "every needed job
    succeeded".

    An earlier version of this evaluator omitted that rule, and the omission was not inert:
    deleting `always()` from `notify-on-failure` -- one token -- kept all 20 tests green
    while the opening edge would have fired 0 of 16 on real GitHub, reinstating exactly the
    #1658 silent-nightly state. Nothing raised, because the mis-model was in the ABSENCE of
    a token; there was no unsupported syntax to detect.

    Still deliberately NOT a general evaluator. It handles the status-check functions above,
    `needs.X.result`, string literals, `==`, `!=`, `&&`, `||` and parentheses, and raises on
    anything else. Note what that guard can and cannot do: it catches syntax that is
    PRESENT and unmodelled, never semantics that are absent.
    """
    expr = condition
    for dep, value in results.items():
        expr = expr.replace(f"needs.{dep}.result", repr(value))

    implicit_success = not any(fn in condition for fn in STATUS_CHECKS)
    for fn, py in (
        ("always()", "True"),
        ("success()", "_ALL_OK"),
        ("failure()", "_ANY_BAD"),
        ("cancelled()", "_ANY_CANCELLED"),
    ):
        expr = expr.replace(fn, py)
    expr = expr.replace("&&", " and ").replace("||", " or ")

    unresolved = re.findall(r"\b[a-z_]+\(\)|needs\.[\w.-]+", expr)
    if unresolved:
        raise AssertionError(f"unsupported syntax {unresolved} in {condition!r}")

    env = {
        "_ALL_OK": all(v == "success" for v in results.values()),
        "_ANY_BAD": any(v == "failure" for v in results.values()),
        "_ANY_CANCELLED": any(v == "cancelled" for v in results.values()),
    }
    fired = bool(eval(expr, {"__builtins__": {}}, env))  # noqa: S307 - the repo's own workflow file
    return fired and (env["_ALL_OK"] if implicit_success else True)


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


def test_notifier_keeps_a_status_check_function():
    """Without one, GitHub ANDs the condition with `success()` and the opening edge dies.

    `notify-on-failure` fires precisely when a dependency is NOT 'success', so an implicit
    `success()` makes its condition unsatisfiable -- 0 of 16 combinations -- and nothing
    would ever report a red nightly again. That is #1658's state, reachable by deleting one
    token. Asserted on the text because the consequence is invisible in the truth table
    unless the evaluator models the implicit check.
    """
    cond = _condition(NOTIFY)
    assert any(fn in cond for fn in STATUS_CHECKS), (
        f"{NOTIFY} has no status-check function, so GitHub applies success() and the job can "
        f"never fire on a red run (#1658). Condition: {cond!r}"
    )


def test_closing_edge_actually_closes():
    """#1806 is 'it comments and does not close'. Pin the close, not just the selection.

    Deleting the `issues.update` call left every other assertion green while the job posted
    'Nightly is green ...; closing.' and closed nothing -- the defect restored, with the
    comment now also false.
    """
    script = _workflow()["jobs"][RESOLVE]["steps"][0]["with"]["script"]
    assert "github.rest.issues.update" in script, "the closing edge does not call issues.update"
    assert "state: 'closed'" in script, "issues.update is called but not with state: 'closed'"
    assert "state_reason: 'completed'" in script, "closed without a reason"


def test_each_job_declares_every_dependency_its_condition_reads():
    """A `needs.X.result` reference with no matching `needs:` entry silently reads null.

    GitHub's `needs` context holds only DIRECT dependencies, so dropping a job from `needs:`
    while the condition still names it makes that comparison false forever -- the closing
    edge would stop firing with nothing to report it.
    """
    jobs = _workflow()["jobs"]
    for name in (NOTIFY, RESOLVE):
        referenced = set(re.findall(r"needs\.([\w.-]+)\.result", _condition(name)))
        declared = set(jobs[name]["needs"])
        assert referenced <= declared, (
            f"{name}: condition reads {sorted(referenced - declared)} which is not in needs="
            f"{sorted(declared)}; that reference evaluates to null, not to a job result"
        )
