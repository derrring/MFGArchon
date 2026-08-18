`scripts/prune_local_branches.sh` classifies local branches by whether their work reached `main`
and prints the evidence. It does **not** delete: adversarial review found that each of its three
signals fails in a way that destroys unmerged work if deletion is automatic — the content check
reverse-applies against the working tree rather than `main` (running it from `scripts/` marked 9
live branches disposable, including two whose issue is open), the merged-PR check matches head-ref
names that are reused after a merge, and the digit run it extracts from a branch name may be a grid
size. Deletion stays a manual step with the sha recorded first.
