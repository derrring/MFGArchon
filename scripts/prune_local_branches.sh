#!/usr/bin/env bash
# Report what is known about each local branch's fate. This tool does not delete.
#
# Branch names record nothing about whether their work landed, and squash merges make
# `git branch --merged` useless, so the evidence has to be derived. Deriving it costs a
# full merged-PR fetch plus a per-branch content test; doing that once per invocation
# rather than by hand each time is the whole value here. The judgement stays with you.
#
# Each line reports what was OBSERVED, not what to do. The distinction is not cosmetic:
# an earlier version printed DELETE under a column headed VERDICT, which `grep DELETE |
# awk '{print "git branch -D " $1}'` turns into destructive commands with the caveats
# forty lines out of band.
set -euo pipefail

# Sourcing this file defines `absorbed` and runs nothing, so a test can exercise the content
# check itself. Re-implementing it in a test would pin a copy and pass over a broken original --
# which is precisely the inversion that shipped here and went unnoticed.
_PRUNE_SOURCED=0
[[ "${BASH_SOURCE[0]}" != "${0}" ]] && _PRUNE_SOURCED=1

if [[ "$_PRUNE_SOURCED" -eq 0 && -n "${1:-}" ]]; then
  echo "This tool takes no arguments and does not delete. It prints evidence; you decide." >&2
  exit 2
fi

# The content check below runs `git apply`, which SKIPS patch paths outside the current
# directory -- silently, checking zero hunks and exiting 0. Run from `scripts/`, every
# branch therefore looked absorbed. Anchor to the toplevel before anything else.
if [[ "$_PRUNE_SOURCED" -eq 0 ]]; then
  cd "$(git rev-parse --show-toplevel)"
fi

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

# CONTENT, not text. An earlier version reverse-applied the branch's diff against a temp index
# (`git apply --cached --check --reverse`). That answers "does this diff reverse-apply", which is a
# different question and wrong in both directions: a branch whose changes landed and were then
# FURTHER edited fails to reverse and reads as unmerged, while a diff that happens to reverse for
# unrelated reasons reads as absorbed. (An even earlier one applied against the WORKING TREE, where
# it was anti-correlated with the property it claims to measure -- it called an unmerged branch
# absorbed and a merged one not, simultaneously, whenever you sat on a third branch.)
#
# `merge-tree --write-tree` answers it directly: merge the branch into origin/main and compare the
# resulting tree with origin/main's own. Equal means the branch contributes NOTHING to main, which
# is exactly "absorbed" -- and it holds under a squash merge, where no commit of the branch is an
# ancestor of main and every commit-graph predicate (`rev-list --count`, `branch --merged`) says
# unmerged. A conflict exits non-zero and is, by construction, not absorbed.
#
# Requires git >= 2.38 for `--write-tree`; checked once at startup rather than per branch, because a
# missing subcommand would otherwise return 1 for every branch and read as "nothing is absorbed".
absorbed() {
  local mb tree main_tree
  mb=$(git merge-base origin/main "$1" 2>/dev/null) || return 1
  git diff --quiet "$mb" "$1" 2>/dev/null && return 2   # 2 = no content difference, distinct from absorbed
  main_tree=$(git rev-parse "origin/main^{tree}" 2>/dev/null) || return 1
  tree=$(git merge-tree --write-tree origin/main "$1" 2>/dev/null) || return 1
  [[ "$tree" == "$main_tree" ]]
}

if [[ "$_PRUNE_SOURCED" -eq 1 ]]; then
  return 0 2>/dev/null || true
fi
if ! git merge-tree --write-tree HEAD HEAD >/dev/null 2>&1; then
  echo "ABORT: this git has no \`merge-tree --write-tree\` (needs >= 2.38). Without it every branch" >&2
  echo "       would classify as not-absorbed, which reads as 'nothing to prune' rather than as a" >&2
  echo "       broken check. Found: $(git --version)" >&2
  exit 1
fi

REPO_SLUG=$(gh repo view --json nameWithOwner -q .nameWithOwner)

# A branch held by a worktree is one `git status` will not report; `git worktree list` is
# the observable. `--format='%(refname:lstrip=2)'` rather than `%(refname:short)`, which
# emits `heads/v1.0` when a tag of the same name exists and would defeat the match below.
git worktree list --porcelain | awk '/^branch /{sub("refs/heads/","",$2); print $2}' | sort -u > "$TMP/held.txt"

TOTAL=$(gh api "search/issues?q=repo:${REPO_SLUG}+is:pr+is:merged&per_page=1" -q .total_count)
# `gh -q` exits 0 with EMPTY output when the key is missing, and an empty TOTAL would make
# the guard below compare against 0 and always pass -- reproducing the incident it exists for.
[[ "$TOTAL" =~ ^[0-9]+$ ]] || { echo "ABORT: could not read the merged-PR total." >&2; exit 1; }
gh pr list --state merged --limit $((TOTAL + 50)) --json headRefName -q '.[].headRefName' | sort -u > "$TMP/merged.txt"
GOT=$(wc -l < "$TMP/merged.txt")
if [[ "$GOT" -lt $((TOTAL / 2)) ]]; then
  echo "ABORT: fetched $GOT merged head refs against $TOTAL merged PRs -- looks truncated." >&2
  exit 1
fi
# Truncating this list flips a keep into a disposable-looking label, so it is guarded too.
OPEN_TOTAL=$(gh api "search/issues?q=repo:${REPO_SLUG}+is:pr+is:open&per_page=1" -q .total_count)
[[ "$OPEN_TOTAL" =~ ^[0-9]+$ ]] || OPEN_TOTAL=0
gh pr list --state open --limit $((OPEN_TOTAL + 50)) --json headRefName -q '.[].headRefName' | sort -u > "$TMP/open.txt"


: > "$TMP/disposable.txt"
printf '%-46s %s\n' "BRANCH" "OBSERVED"
while read -r b; do
  if grep -qxF "$b" "$TMP/held.txt";   then printf '%-46s %s\n' "$b" "held by a worktree"; continue; fi
  if grep -qxF "$b" "$TMP/open.txt";   then printf '%-46s %s\n' "$b" "head ref of an OPEN pr"; continue; fi
  if grep -qxF "$b" "$TMP/merged.txt"; then
    printf '%-46s %s\n' "$b" "head ref of a merged pr (NAME match only; shas not compared)"
    echo "$b" >> "$TMP/disposable.txt"; continue
  fi
  set +e; absorbed "$b"; rc=$?; set -e
  case "$rc" in
    0) printf '%-46s %s\n' "$b" "patch reverse-applies to origin/main"; echo "$b" >> "$TMP/disposable.txt"; continue ;;
    2) printf '%-46s %s\n' "$b" "no content difference from its merge-base (empty commits only)"; continue ;;
  esac
  n=$(grep -oE '[0-9]{3,4}' <<<"$b" | head -1 || true)
  st=""
  if [[ -n "$n" ]]; then
    st=$(gh issue view "$n" --json state -q .state 2>/dev/null || true)
    [[ -z "$st" ]] && st=$(gh pr view "$n" --json state -q .state 2>/dev/null || true)
  fi
  case "$st" in
    MERGED) printf '%-46s %s\n' "$b" "name contains $n, which is MERGED (no relation established)"
            echo "$b" >> "$TMP/disposable.txt" ;;
    OPEN)   printf '%-46s %s\n' "$b" "name contains $n, which is OPEN" ;;
    CLOSED) printf '%-46s %s\n' "$b" "name contains $n, CLOSED -- closed does not mean merged" ;;
    *)      printf '%-46s %s\n' "$b" "unmerged, no reference found" ;;
  esac
done < <(git for-each-ref refs/heads/ --format='%(refname:lstrip=2)' | grep -vx main)

echo
echo "$(wc -l < "$TMP/disposable.txt" | tr -d ' ') branch(es) show evidence of having landed."
echo "Read the branch before acting. Two of the three signals above are weak by construction:"
echo "  - the merged-pr signal compares NAMES; names are reused after a merge."
echo "  - a 3-4 digit run in a branch name may be a grid size, not an issue number."
echo "Then: git branch -D <name>   (record the sha first -- -D discards the branch's reflog)."
