#!/usr/bin/env python3
"""The backlog family map's two questions, answered from LIVE labels.

The backlog census established that the open issues are about far fewer distinct things, and that
fixing per-instance has a yield of exactly 1:1 and never converges -- the count falls only when the
fix lands at the layer the family points to.

**Membership is a label, not a file, and that is the whole design.** An earlier version of this
script read a mirrored `{family: [issue numbers]}` snapshot; that snapshot went stale at exactly the
rate of the analysis it mirrored, which is the defect the family map exists to name, one level down.
A `family: <slug>` label lives on the issue, is maintained where the classification happens, is
current by construction, and shows up in the issue list -- which is where someone picks work, so the
family is visible before any tool runs. ORDER is a separate `blocks: <slug>` label on the issues that
stand in a family's way; the order file only declares that such a dependency exists.

Two modes:

1. ``--branch <issue>`` -- which family does this issue belong to, how many siblings are open, and
   is some other family the head of the queue? Run by hand when picking work; nothing triggers it.
2. ``--closed <issue>`` -- what did closing this issue change? Run on ``issues: closed``. Two events,
   and **silent unless one of them happened**:

   - **a blocker cleared**: the issue carried a `blocks:` label and no open issue carries it now.
     That is what changes the head of the queue, and it is the only event that reports what it
     unblocks. The one in the record: #1991 took the last `blocks: no-convergence-obligation` on
     2026-09-13, and nothing noticed for eight days.
   - **a family emptied**: the issue carried a `family:` label and no open issue carries it now.
     Grouping, not order: it unblocks nothing by itself. No family has emptied yet (measured
     2026-09-24: none of the 28 `family:` labels has a closed member).

**Why not a notice on every close.** Measured 2026-09-24 over issues created or closed 2026-08-22
to 2026-09-21 (search API): 207 opened and 140 closed, peaking at 45 events on 2026-09-05. A line on
every close is scenery -- a signal that fires identically every time carries no information. The
sub-events above happened once in that window.

**Why there is no ``--opened`` mode.** Which family a new issue belongs to cannot be computed: a
deterministic route (citation-graph clustering) measured recall 0.40 and precision 0.17-0.22 against
a hand-traced control, because a family is defined by MECHANISM and its members routinely share no
vocabulary. Nothing useful fires on open; what changes is the unclassified count, which this script
derives on demand rather than storing.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

HERE = pathlib.Path(__file__).resolve().parent
ORDER = HERE / "backlog_family_order.json"
PREFIX = "family: "
NOTICE_LABELS = "automated,priority: medium,area: testing,size: small,type: chore"


def _gh(args: list[str]) -> str:
    """Run `gh`, raising on failure rather than returning an empty result.

    An empty list would make every label look cleared and every issue look unclassified, so every
    caller must see the failure instead of a plausible answer.
    """
    r = subprocess.run(["gh", *args], capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args[:4])} failed: {r.stderr.strip()[:200]}")
    return r.stdout


def rest_issues(state: str, label: str | None = None) -> list[dict]:
    """Issues (not pull requests) in `state`, optionally carrying `label`, from the REST list endpoint.

    Not `gh issue list`, which is served by the search index. That index is updated asynchronously,
    and this runs seconds after a close, so the issue just closed could still read as open and the
    run would be silent (review of #2395). `--paginate` also removes the page-size question: a
    `--limit N` returns N and says nothing about there being more.
    """
    args = [
        "api",
        "-X",
        "GET",
        "repos/{owner}/{repo}/issues",
        "--paginate",
        "-f",
        f"state={state}",
        "-f",
        "per_page=100",
    ]
    if label:
        args += ["-f", f"labels={label}"]
    args += ["--jq", ".[] | select(.pull_request | not) | {number, title, labels: [.labels[].name]}"]
    return [json.loads(line) for line in _gh(args).splitlines() if line.strip()]


def open_numbers(label: str) -> list[int]:
    return sorted(r["number"] for r in rest_issues("open", label))


def labels_of(issue: int) -> list[str]:
    return [lb["name"] for lb in json.loads(_gh(["issue", "view", str(issue), "--json", "labels"]))["labels"]]


def family_labels(labels: list[str]) -> list[str]:
    return sorted(name for name in labels if name.startswith(PREFIX))


def load_order() -> dict:
    return json.loads(ORDER.read_text())


def close_events(
    issue: int, labels: list[str], open_with: Callable[[str], list[int]], dependencies: list[dict]
) -> list[dict]:
    """What closing `issue` changed. Pure given `open_with`, so the self-test drives this decision.

    `issue` is excluded from every open list. The trigger runs after the close, but a lookup can lag
    it, and a replay by dispatch evaluates an issue that may still be open, "as if it had just been
    closed".

    UNBLOCKS belongs to a blocker label the closed issue CARRIED and nothing else. Printed for every
    unblocked dependency, it attached a dependency cleared weeks earlier to every family that emptied
    since (review of #2395, B1).
    """
    carried = set(labels)
    events = []
    for dep in dependencies:
        blocker = dep["blocker_label"]
        if blocker in carried and not [n for n in open_with(blocker) if n != issue]:
            dependents = [n for n in open_with(PREFIX + dep["unblocks"]) if n != issue]
            events.append(
                {
                    "kind": "blocker",
                    "label": blocker,
                    "unblocks": dep["unblocks"],
                    "open": dependents,
                    "why": dep["why"],
                }
            )
    for fam in family_labels(labels):
        if not [n for n in open_with(fam) if n != issue]:
            events.append({"kind": "family", "label": fam})
    return events


def render(issue: int, event: dict) -> tuple[str, str]:
    """Title and body of the notice for one event. One title per event, so no title spans two lines."""
    if event["kind"] == "blocker":
        members = " ".join(f"#{n}" for n in event["open"]) or "none"
        title = f"A backlog blocker cleared: {event['label']}"
        body = "\n".join(
            [
                f"After #{issue} closed, no open issue carries `{event['label']}`.",
                "",
                f"That label marked what stood in the way of `{PREFIX}{event['unblocks']}`, which has "
                f"{len(event['open'])} open member(s): {members}",
                "",
                f"Why the dependency exists: {event['why']}",
                "",
                "**What to do with it.** Decide whether that family is now the head of the queue, against",
                "whatever else is in flight. The trigger's job is to make someone look; it does not decide.",
                "Read the members in the private ledger, not here: this notice carries issue numbers because",
                "they are public, and the reasoning is internal analysis.",
            ]
        )
    else:
        title = f"A backlog family emptied: {event['label']}"
        body = "\n".join(
            [
                f"After #{issue} closed, no open issue carries `{event['label']}`.",
                "",
                "A family is a grouping, not an order: this unblocks nothing by itself. Order lives on",
                "`blocks:` labels, and a cleared blocker opens its own notice.",
            ]
        )
    return title, body + "\n\nOpened by `.github/workflows/family-emptied.yml`.\n"


def post(issue: int, events: list[dict]) -> None:
    """Open one notice per event, or comment on the open one with the same title.

    One pull request closing k members of a family runs this k times, seconds apart, and each run
    sees the family empty. Dedupe is on title plus the `automated` label, which is how the other
    notifying workflows here find their own issues. There is no `concurrency:` group: GitHub keeps one
    pending run per group and cancels the one before it (documented behaviour, not measured here), so
    a group would drop events, and a dropped run whose issue carried a `blocks:` label would lose the
    one notice that matters. Two runs that race past the dedupe both create; the later one then finds
    the earlier and closes itself as a duplicate.
    """
    for event in events:
        title, body = render(issue, event)
        same = sorted(r["number"] for r in rest_issues("open", "automated") if r["title"] == title)
        if same:
            _gh(["issue", "comment", str(same[0]), "--body", f"#{issue} closed too, and the state above still holds."])
            continue
        url = _gh(["issue", "create", "--title", title, "--body", body, "--label", NOTICE_LABELS]).strip()
        mine = int(url.rsplit("/", 1)[-1])
        same = sorted(r["number"] for r in rest_issues("open", "automated") if r["title"] == title)
        if same and same[0] != mine:
            _gh(
                [
                    "issue",
                    "close",
                    str(mine),
                    "--reason",
                    "not planned",
                    "--comment",
                    f"Duplicate of #{same[0]}: two runs raced.",
                ]
            )
            _gh(["issue", "comment", str(same[0]), "--body", f"#{issue} closed too, and the state above still holds."])


def report_closed(issue: int, *, do_post: bool) -> int:
    """Print the notices closing `issue` warrants, and open them with `do_post`. Silent otherwise."""
    events = close_events(issue, labels_of(issue), open_numbers, load_order().get("dependencies", []))
    for event in events:
        title, body = render(issue, event)
        print(f"== {title}\n{body}")
    if do_post and events:
        post(issue, events)
    return 0


def unblocked() -> list[dict]:
    """Dependency pairs whose prerequisite has no open members while the dependent still has."""
    out = []
    for dep in load_order().get("dependencies", []):
        # The blocker is its OWN label, not the prerequisite family. Keyed to a family, a single
        # member whose mechanism matches but whose oracle certifies something else -- a performance
        # benchmark among correctness oracles -- holds the queue head shut forever. Grouping and
        # order do not share a grain.
        blocker = dep["blocker_label"]
        # "Nothing open carries it" is NOT enough, and a stress test is what showed it: a blocker
        # label that exists and was never applied to anything reads exactly like one whose issues
        # have all closed. The order file declaring that a dependency EXISTS does not separate them
        # either -- it says the relation is real, not that anyone ever identified what blocks it.
        # So the evidence for "unblocked" is that something CLOSED carries the label: the blockers
        # were identified, and they are done.
        if not rest_issues("all", blocker):
            out.append(
                {
                    **dep,
                    "open": [],
                    "unverified": "no issue has ever carried this label, so the blocker was never identified",
                }
            )
            continue
        if not open_numbers(blocker) and (still := open_numbers(PREFIX + dep["unblocks"])):
            out.append({**dep, "open": still})
    return out


def report_branch(issue: int) -> int:
    fams = family_labels(labels_of(issue))
    if not fams:
        total = len(rest_issues("open"))
        print(f"  #{issue} carries no family label. {total} issues are open; an unlabelled one is")
        print("  the common case, not a judgement that it stands alone.")
    for fam in fams:
        siblings = [n for n in open_numbers(fam) if n != issue]
        print(f"  #{issue} is in '{fam}' -- {len(siblings)} other members still open")
        if siblings:
            print(f"    {' '.join('#' + str(n) for n in siblings)}")
            print("    Fixing this one alone has a yield of 1:1; the family is the unit that")
            print("    makes the count fall.")
            # A label good enough to diagnose from is good enough to stop you reading the source,
            # and this one is not good enough to diagnose from. Membership was established from what
            # each ISSUE says, not from the instrument it describes. Measured cost of skipping this,
            # 2026-09-21: an incident matched 'ratchet-counts-not-sets' so exactly that it was
            # written into a PR body as the diagnosis, and the source said the opposite -- that
            # checker pins identities as well as counts, and a comment above the function describes
            # the failure that was being attributed to it. The defect was elsewhere.
            print("    Membership is a HYPOTHESIS about a mechanism, taken from what the issue says")
            print("    and not from the code it describes. Read the source before you diagnose from it.")
    for u in unblocked():
        if u.get("unverified"):
            print(f"  DEPENDENCY UNVERIFIED: '{u['blocker_label']}' -- {u['unverified']}.")
            print("    This is not 'unblocked'. Label the issues that block it, or drop the dependency.")
            continue
        print(f"  QUEUE HEAD: nothing open carries '{u['blocker_label']}', which unblocks")
        print(f"    '{u['unblocks']}' -- {len(u['open'])} open: {' '.join('#' + str(n) for n in u['open'])}")
        print(f"    {u['why']}")
    return 0


def self_test(online: bool = False) -> int:
    """Controls on the decision, the rendering and the order file. Offline by default, deliberately.

    The failure this guards is the silent one: a trigger that never fires reads exactly like "nothing
    changed". So the offline half drives `close_events` itself -- not a copy of its filter -- over
    fabricated labels and open lists, with positive and negative controls. The review of #2395
    found the earlier self-test copied the label filter inline, and stayed green with
    `families_of` returning [] while the trigger went silent.

    **`--online` is separated out because the local gate does not touch the network.** Measured:
    `scripts/local_ci.sh` contains zero `gh`/`curl` invocations, and putting a network call in its
    self-test loop would trade that property for one check. So the gate runs the offline half, and
    the workflow -- which has a token anyway -- runs `--self-test --online`, which additionally
    asserts that the order file's labels exist. That assertion matters: an order file naming a label
    nobody created makes the queue head silent forever, in a way no offline check can see.
    """
    failures = []
    dep = {"blocker_label": "blocks: b", "unblocks": "f", "why": "w"}
    open_lists = {"blocks: b": [7], "family: f": [11, 12], "family: g": [7], "family: h": [8, 9]}

    def open_with(label: str) -> list[int]:
        return open_lists.get(label, [])

    def kinds(issue: int, labels: list[str]) -> list[tuple[str, str]]:
        return [(e["kind"], e["label"]) for e in close_events(issue, labels, open_with, [dep])]

    # the last open carrier of a blocker, still listed open (a lagging lookup, or a dispatch replay)
    got = kinds(7, ["blocks: b", "family: g", "type: bug"])
    if got != [("blocker", "blocks: b"), ("family", "family: g")]:
        failures.append(f"closing the last carrier of a blocker and the last member of a family gave {got}")
    unblocks = [e for e in close_events(7, ["blocks: b"], open_with, [dep]) if e["kind"] == "blocker"]
    if not unblocks or unblocks[0]["open"] != [11, 12]:
        failures.append(f"a cleared blocker did not report its dependent family's open members: {unblocks}")
    # negative controls
    if kinds(8, ["family: h"]):
        failures.append("a family with another open member was reported empty")
    if kinds(9, ["type: bug", "area: core"]):
        failures.append("an issue with no family or blocker label produced an event")
    # the #2224 case: the blocker cleared long ago, and an unrelated family empties now
    cleared = {"blocks: b": [], "family: f": [11, 12], "family: k": [20]}
    got = [(e["kind"], e["label"]) for e in close_events(20, ["family: k"], lambda lb: cleared.get(lb, []), [dep])]
    if got != [("family", "family: k")]:
        failures.append(f"a family emptying after its blocker cleared elsewhere gave {got} (review of #2395, B1)")
    for event in close_events(7, ["blocks: b", "family: g"], open_with, [dep]):
        title, _ = render(7, event)
        if "\n" in title:
            failures.append(f"a notice title spans lines: {title!r}")

    try:
        order = load_order()
        if not order.get("dependencies"):
            failures.append("the order file declares no dependencies, so no blocker can ever clear")
        for d in order.get("dependencies", []):
            for key in ("blocker_label", "unblocks"):
                if not d.get(key):
                    failures.append(f"a dependency has no '{key}'")
            if not d.get("why"):
                failures.append(
                    f"dependency on {d.get('blocker_label')} states no reason, "
                    "so the notice it opens cannot explain itself"
                )
    except Exception as exc:  # the self-test reports; it does not handle
        failures.append(f"the order file does not load: {exc}")

    if online:
        try:
            live = set(_gh(["api", "repos/{owner}/{repo}/labels", "--paginate", "--jq", ".[].name"]).splitlines())
            for d in load_order().get("dependencies", []):
                for name in (d["blocker_label"], PREFIX + d["unblocks"]):
                    if name not in live:
                        failures.append(
                            f"order names '{name}', which is not a label in this repository -- the dependency is inert"
                        )
            if not any(n.startswith(PREFIX) for n in live):
                failures.append("no family labels exist at all, so neither event can fire")
        except Exception as exc:  # reported as a self-test failure, not handled
            failures.append(f"online label check failed: {exc}")

    for f in failures:
        print(f"SELF-TEST FAIL: {f}")
    scope = "online" if online else "offline"
    print(f"family_queue self-test ({scope}): {'OK' if not failures else str(len(failures)) + ' failures'}")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--branch", type=int, metavar="ISSUE")
    g.add_argument("--closed", type=int, metavar="ISSUE")
    g.add_argument("--self-test", action="store_true")
    ap.add_argument(
        "--online", action="store_true", help="add the checks that need the network; the local gate does not use this"
    )
    ap.add_argument("--post", action="store_true", help="with --closed: open the notices instead of only printing them")
    a = ap.parse_args()
    if a.self_test:
        return self_test(online=a.online)
    if a.branch:
        return report_branch(a.branch)
    return report_closed(a.closed, do_post=a.post)


if __name__ == "__main__":
    sys.exit(main())
