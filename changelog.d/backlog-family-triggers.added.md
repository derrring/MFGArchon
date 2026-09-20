The backlog family map gets two triggers and stops being a snapshot.

Membership is a `family: <slug>` label on the issue, maintained where the classification happens and
visible in the issue list, which is where work is picked. An earlier draft mirrored the membership
into a JSON file; that snapshot went stale at exactly the rate of the analysis it mirrored, which is
the defect the family map exists to name, one level down.

Order is a separate `blocks: <slug>` label, because grouping and order do not share a grain. Keyed
to the family, one member whose mechanism matches while its oracle certifies something else — a GPU
performance benchmark among correctness oracles — holds the queue head shut forever. Measured the
same afternoon the family-keyed version was written.

`scripts/family_queue.py --emptied` runs on `issues: closed` and is silent unless the close took the
last open member. Not a notice on every close: over the 30 days to 2026-09-21, 203 issues opened and
138 closed, 11.4 events a day peaking at 45, while a close that emptied a family fired **once** — on
2026-09-13, the event nobody noticed for eight days.
