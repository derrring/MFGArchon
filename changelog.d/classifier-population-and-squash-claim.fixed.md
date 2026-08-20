Two corrections to the branch classifier, both found by pointing it at the tree it was written for.

**Its population excluded where the branches were.** It enumerated `refs/heads/` only, so on
2026-08-21 it reported "0 branch(es) show evidence of having landed" on a repository carrying
**sixteen** remote branches whose PRs were merged. They had never been local. That is the same shape
as the defects this repository has been chasing — a check that is not where the thing it checks lives
— and it is invisible in the way that matters: the output is a clean report, not an error. Both
scopes are now pooled, and the remedy line names `git push origin --delete` alongside
`git branch -D`, since the local command does not touch origin.

**The squash-merge claim was overstated.** The comment said absorption by `merge-tree --write-tree`
"holds under a squash merge". It holds only while `main` has not since modified the same regions;
after that the three-way merge sees both sides changing one region and conflicts, so the branch reads
unmerged again. Measured: of the sixteen merged branches above, **eight** classify as unmerged this
way — `fix/1986-sense-sign-one-owner` (#1987) conflicts in `problem_factories.py`, which main has
touched twice since that branch point.

The error remains a **false negative** — a landed branch kept, never an unlanded one offered for
deletion — which is why the merged-PR name signal runs first and is the primary evidence, with this
predicate corroborating. The limitation is pinned by a test carrying its own retirement condition, so
the flat claim cannot be re-asserted without a failure saying otherwise.
