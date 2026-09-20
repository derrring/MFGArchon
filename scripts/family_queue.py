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
family is visible before any tool runs. The only thing left in a file is the ORDER (below), because
a dependency between two families is a claim about oracles and not a property of any issue.

Two modes, one per observable moment:

1. ``--branch <issue>`` -- which family does this issue belong to, how many siblings are open, and
   is some other family the head of the queue? Run when a branch is created, which is the only
   artifact that task SELECTION produces.
2. ``--emptied <issue>`` -- did closing this issue take the last open member of a family? Run on
   ``issues: closed``. **Silent unless it did**, which is the point.

**Why (2) is not a notice on every close.** Measured over the 30 days to 2026-09-21: 203 issues
opened, 138 closed, 11.4 events/day, peaking at 45 in one day. A line on every close is scenery -- a
signal that fires identically every time carries no information. A close that EMPTIES a family fired
**once** in the same window, on 2026-09-13, and that once is the event nobody noticed. 138 against 1
is why the trigger sits on the sub-event rather than the event.

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

HERE = pathlib.Path(__file__).resolve().parent
ORDER = HERE / "backlog_family_order.json"
PREFIX = "family: "


def gh_json(args: list[str]) -> list | dict:
    """Run `gh` and parse JSON, raising on failure rather than returning an empty result.

    An empty list would make every family look emptied and every issue look unclassified, so both
    callers must see the failure instead of a plausible answer.
    """
    r = subprocess.run(["gh", *args], capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args[:3])} failed: {r.stderr.strip()[:200]}")
    return json.loads(r.stdout)


def families_of(issue: int) -> list[str]:
    labels = gh_json(["issue", "view", str(issue), "--json", "labels"])["labels"]
    return sorted(lb["name"] for lb in labels if lb["name"].startswith(PREFIX))


LIMIT = 400


def issues_with(label: str, state: str = "open") -> list[int]:
    """Issue numbers carrying `label`. Raises if the page size was reached rather than truncating.

    `gh --limit N` returns N and says nothing about there being more -- a limit is a page size
    wearing a number. A truncated list here would under-report a family's open members and, at the
    limit, could report a populated family as empty.
    """
    rows = gh_json(["issue", "list", "--label", label, "--state", state, "--limit", str(LIMIT), "--json", "number"])
    if len(rows) >= LIMIT:
        raise RuntimeError(f"'{label}' returned {len(rows)} at the page size; raise LIMIT rather than trusting this")
    return sorted(r["number"] for r in rows)


def open_in(family: str) -> list[int]:
    return issues_with(family, "open")


def load_order() -> dict:
    return json.loads(ORDER.read_text())


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
        ever = issues_with(blocker, "all")
        if not ever:
            out.append({**dep, "open": [], "unverified": "no issue has ever carried this label, so the blocker was never identified"})
            continue
        if not open_in(blocker) and (still := open_in(PREFIX + dep["unblocks"])):
            out.append({**dep, "open": still})
    return out


def report_branch(issue: int) -> int:
    fams = families_of(issue)
    if not fams:
        total = len(gh_json(["issue", "list", "--state", "open", "--limit", "600", "--json", "number"]))
        print(f"  #{issue} carries no family label. {total} issues are open; an unlabelled one is")
        print("  the common case, not a judgement that it stands alone.")
    for fam in fams:
        siblings = [n for n in open_in(fam) if n != issue]
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


def report_emptied(issue: int) -> int:
    """Print nothing unless closing `issue` left one of its families with no open members."""
    for fam in families_of(issue):
        if open_in(fam):
            continue
        print(f"FAMILY_EMPTIED={fam}")
        print(f"CLOSED_BY=#{issue}")
        for u in unblocked():
            if not u.get("unverified"):
                print(f"UNBLOCKS={u['unblocks']}")
                print(f"UNBLOCKS_OPEN={' '.join('#' + str(n) for n in u['open'])}")
                print(f"WHY={u['why']}")
    return 0


def self_test(online: bool = False) -> int:
    """Controls on the parsing and the order file. Offline by default, and that is deliberate.

    The failure this guards is the silent one: a label prefix that matches nothing would make
    `report_emptied` silent for every input, which reads exactly like "no family emptied". The
    negative control is therefore mandatory.

    **`--online` is separated out because the local gate does not touch the network.** Measured:
    `scripts/local_ci.sh` contains zero `gh`/`curl` invocations, and putting a network call in its
    self-test loop would trade that property for one check. So the gate runs the offline half, and
    the workflow -- which has a token anyway -- runs `--self-test --online`, which additionally
    asserts that the order file's family names are labels that actually exist. That assertion
    matters: an order file naming a label nobody created makes the queue head silent forever, in a
    way no offline check can see.
    """
    failures = []

    fake = {"labels": [{"name": "family: ghost-value-unowned"}, {"name": "type: bug"}, {"name": "area: core"}]}
    got = sorted(lb["name"] for lb in fake["labels"] if lb["name"].startswith(PREFIX))
    if got != ["family: ghost-value-unowned"]:
        failures.append(f"label filter picked {got}")
    none = [lb["name"] for lb in fake["labels"][1:] if lb["name"].startswith(PREFIX)]
    if none:
        failures.append(f"label filter matched a non-family label: {none}")

    try:
        order = load_order()
        if not order.get("dependencies"):
            failures.append("the order file declares no dependencies, so the queue head can never fire")
        for dep in order.get("dependencies", []):
            for key in ("blocker_label", "unblocks"):
                if not dep.get(key):
                    failures.append(f"a dependency has no '{key}'")
            if not dep.get("why"):
                failures.append(
                    f"dependency on {dep.get('blocker_label')} states no reason, "
                    "so the notice it fires cannot explain itself"
                )
    except Exception as exc:  # noqa: BLE001 - the self-test reports; it does not handle
        failures.append(f"the order file does not load: {exc}")

    if online:
        try:
            live = {lb["name"] for lb in gh_json(["label", "list", "--limit", "120", "--json", "name"])}
            for dep in load_order().get("dependencies", []):
                for name in (dep["blocker_label"], PREFIX + dep["unblocks"]):
                    if name not in live:
                        failures.append(f"order names '{name}', which is not a label in this repository -- the dependency is inert")
            if not any(n.startswith(PREFIX) for n in live):
                failures.append("no family labels exist at all, so both modes are inert")
        except Exception as exc:  # noqa: BLE001
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
    g.add_argument("--emptied", type=int, metavar="ISSUE")
    g.add_argument("--self-test", action="store_true")
    ap.add_argument("--online", action="store_true", help="add the checks that need the network; the local gate does not use this")
    a = ap.parse_args()
    if a.self_test:
        return self_test(online=a.online)
    return report_branch(a.branch) if a.branch else report_emptied(a.emptied)


if __name__ == "__main__":
    sys.exit(main())
