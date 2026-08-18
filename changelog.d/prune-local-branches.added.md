`scripts/prune_local_branches.sh` classifies local branches by whether their work reached `main`
and deletes the ones that did, dry-run by default. Derivation is three-tier — merged PR head ref,
reverse-apply of the branch diff against `main`, then a MERGED number in the branch name — because
branch names record no fate of their own. `--delete` writes a name-and-sha recovery manifest first,
since iteration branches can sit several commits ahead of `main` and the reflog is not a plan. It
refuses to run if the merged-PR fetch looks truncated, which would silently reclassify live
branches as orphans.
