`scripts/prune_local_branches.sh` classifies local branches by whether their work reached `main`
and prints the evidence. It does **not** delete: adversarial review found that each of its three
signals fails in a way that destroys unmerged work if deletion is automatic — the content check
reverse-applies against the working tree rather than `main` (running it from `scripts/` marked 9
live branches disposable, including two whose issue is open), the merged-PR check matches head-ref
names that are reused after a merge, and the digit run it extracts from a branch name may be a grid
size. Deletion stays a manual step with the sha recorded first.

The absorption predicate is `git merge-tree --write-tree`: merge the branch into `origin/main` and
compare the resulting tree with main's own. Equal means the branch contributes nothing, which is what
"absorbed" means and which holds under a squash merge — where no commit of the branch is an ancestor
of main and every commit-graph predicate (`rev-list --count`, `branch --merged`) says unmerged.

It replaces a reverse-apply of the branch diff (`git apply --check --reverse`), which answers a
different question and is context-sensitive. The discriminating case, found by running the two
against each other rather than by reading the code: the branch adds a line, main squash-merges it and
then edits a *different* line inside the diff's context window. Reverse-apply reports NOT absorbed;
the branch's content is demonstrably all in main.

Stated in the direction it actually runs: that error is a **false negative**. The old predicate keeps
a prunable branch and never proposes deleting an unmerged one, so this is an accuracy and noise fix
rather than a safety hole being closed.

Six discrimination tests build a real throwaway repository each — including one that pins the verdict
does not depend on which branch is checked out, which an even earlier predicate got wrong by
evaluating against the working tree. **Five of the six pass under both predicates**; only the
nearby-edit one separates them, and that is recorded in the test rather than left for a reader to
discover.

A startup guard aborts if `git merge-tree --write-tree` is unavailable (needs git ≥ 2.38): without
it every branch would classify as not-absorbed, which reads as "nothing to prune" rather than as a
broken check.
