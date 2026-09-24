The backlog family map gets a trigger, and membership stops being a snapshot.

Membership is a `family: <slug>` label on the issue. It is maintained where the classification happens and is visible in the issue list, which is where work is picked. An earlier draft mirrored the membership into a JSON file; that snapshot went stale at exactly the rate of the analysis it mirrored, which is the defect the family map exists to name, one level down.

Order is a separate `blocks: <slug>` label on the issues that stand in a family's way, because grouping and order do not share a grain. Keyed to the family, one member whose mechanism matches while its oracle certifies something else (a GPU performance benchmark among correctness oracles) holds the queue head shut forever.

`.github/workflows/family-emptied.yml` runs `scripts/family_queue.py --closed` on `issues: closed` and is silent unless the close did one of two things:

- **took the last open carrier of a `blocks:` label.** The notice names the family that label stood in front of, and its open members. This is the event that changes the head of the queue. The one in the record: #1991 took the last `blocks: no-convergence-obligation` on 2026-09-13, and nothing noticed for eight days.
- **took the last open member of a `family:` label.** The notice says the family is empty and claims nothing about order.

Not a notice on every close: over issues created or closed 2026-08-22 to 2026-09-21, 207 opened and 140 closed, peaking at 45 events on 2026-09-05, while the blocker event above happened once. Notices carry the `automated` label, and a second close with the same effect comments on the open notice instead of opening another. `--branch <issue>` answers which family an issue is in and what heads the queue; it is run by hand.
