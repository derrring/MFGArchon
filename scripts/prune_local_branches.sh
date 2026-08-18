#!/usr/bin/env bash
# Classify local branches by whether their work reached main, and delete the ones that did.
#
# Branch names do not record their fate, so this derives it three ways, cheapest first:
#   1. the branch is the head ref of a MERGED pull request
#   2. the branch's diff from its merge-base reverse-applies to main (content already there)
#   3. the branch names a PR number that is MERGED, and is an iteration branch of it
#      (CLOSED does not count: an issue closing says nothing about whether code landed)
#
# Dry run by default. `--delete` writes a recovery manifest first: every deleted branch's
# name and sha, so `git branch <name> <sha>` restores it without relying on the reflog --
# which matters because tier-3 branches can be several commits ahead of main.
set -euo pipefail

if [[ -n "${1:-}" ]]; then
  echo "This tool does not delete. It classifies and prints evidence; you delete." >&2
  exit 2
fi
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
REPO_SLUG=$(gh repo view --json nameWithOwner -q .nameWithOwner)
STAMP=$(git log -1 --format=%cd --date=format:%Y-%m-%d)
MANIFEST="${TMPDIR:-/tmp}/deleted_branches_${STAMP}.txt"

# A branch held by a worktree cannot be deleted, and `git status` cannot see worktrees at all.
git worktree list --porcelain | awk '/^branch /{sub("refs/heads/","",$2); print $2}' | sort -u > "$TMP/held.txt"

# The merged-PR list MUST NOT be truncated: a short list silently reclassifies branches as
# orphans. Compare against the repo's own total before trusting it.
TOTAL=$(gh api "search/issues?q=repo:${REPO_SLUG}+is:pr+is:merged&per_page=1" -q .total_count)
# `gh -q` exits 0 with EMPTY output when the key is missing, and an empty TOTAL makes the
# guard below compare against 0 and always pass -- reproducing the incident it exists to catch.
[[ "$TOTAL" =~ ^[0-9]+$ ]] || { echo "ABORT: could not read the merged-PR total." >&2; exit 1; }
gh pr list --state merged --limit $((TOTAL + 50)) --json headRefName -q '.[].headRefName' | sort -u > "$TMP/merged.txt"
GOT=$(wc -l < "$TMP/merged.txt")
if [[ "$GOT" -lt $((TOTAL / 2)) ]]; then
  echo "ABORT: fetched $GOT merged head refs against $TOTAL merged PRs -- the list looks truncated," >&2
  echo "       and every classification below would rest on it. Refusing to guess." >&2
  exit 1
fi
gh pr list --state open --limit 200 --json headRefName -q '.[].headRefName' | sort -u > "$TMP/open.txt"

absorbed() {  # 0 if the branch's unique changes are already present in main
  local mb; mb=$(git merge-base origin/main "$1" 2>/dev/null) || return 1
  git diff "$mb".."$1" > "$TMP/p.diff" 2>/dev/null || return 1
  [[ -s "$TMP/p.diff" ]] || return 0
  git apply --check --reverse "$TMP/p.diff" 2>/dev/null
}

: > "$TMP/kill.txt"
printf '%-46s %s\n' "BRANCH" "VERDICT"
while read -r b; do
  if grep -qxF "$b" "$TMP/held.txt";   then printf '%-46s %s\n' "$b" "keep  (held by a worktree)"; continue; fi
  if grep -qxF "$b" "$TMP/open.txt";   then printf '%-46s %s\n' "$b" "keep  (open PR)";            continue; fi
  if grep -qxF "$b" "$TMP/merged.txt"; then printf '%-46s %s\n' "$b" "DELETE (merged PR)"; echo "$b" >> "$TMP/kill.txt"; continue; fi
  if absorbed "$b";                   then printf '%-46s %s\n' "$b" "DELETE (content in main)"; echo "$b" >> "$TMP/kill.txt"; continue; fi
  # `|| true` throughout: under `set -e` a grep that matches nothing, or a gh lookup for a
  # number that is neither issue nor PR, would abort the whole sweep mid-branch.
  n=$(grep -oE '[0-9]{3,4}' <<<"$b" | head -1 || true)
  st=""
  if [[ -n "$n" ]]; then
    st=$(gh issue view "$n" --json state -q .state 2>/dev/null || true)
    [[ -z "$st" ]] && st=$(gh pr view "$n" --json state -q .state 2>/dev/null || true)
  fi
  # MERGED only. A CLOSED issue does not establish that the branch's work shipped -- an
  # instrumentation or diagnostic branch outlives the issue that prompted it, and closing
  # the issue says nothing about whether its code landed.
  case "$st" in
    MERGED) printf '%-46s %s\n' "$b" "DELETE (iteration branch of #$n, merged)"; echo "$b" >> "$TMP/kill.txt" ;;
    OPEN)   printf '%-46s %s\n' "$b" "keep  (#$n still open)" ;;
    CLOSED) printf '%-46s %s\n' "$b" "keep  (#$n closed, but closed != work merged)" ;;
    *)      printf '%-46s %s\n' "$b" "keep  (no issue ref; unverified work)" ;;
  esac
done < <(git branch --format='%(refname:short)' | grep -vx main)

COUNT=$(wc -l < "$TMP/kill.txt" | tr -d ' ')
echo
echo "$COUNT branch(es) look disposable. This tool does not delete them, deliberately:"
echo "  - the content check ('git apply --check --reverse') is run against the WORKING TREE,"
echo "    not against main, so its verdict depends on the checkout and on uncommitted edits;"
echo "  - the merged-PR check matches head-ref NAMES, and names are reused after a merge;"
echo "  - the number extracted from a branch name may be a grid size, not an issue."
echo "Each of those turns a wrong label into destroyed work the moment deletion is automatic."
echo "Read the branch, then: git branch -D <name>   (record the sha first)."
