# changelog.d — changelog fragments (Issue #1521)

Every PR adds **its own** fragment file here instead of editing `CHANGELOG.md`, so PRs never
conflict on the shared `### …` sections (batched or not).

**Fragment file**: `changelog.d/<slug>.<category>.md`
- `<slug>` — the issue/PR number (e.g. `1521`) or a short kebab name.
- `<category>` — one of `added` / `changed` / `deprecated` / `removed` / `fixed` (Keep a Changelog).
- **Content** — the markdown bullet(s) for the change, e.g.
  ```
  - **Short title** (Issue #123). One-sentence what + why.
  ```

**At release** (part of the version-bump checklist): promote the fragments into `CHANGELOG.md`:
```bash
python scripts/collate_changelog.py --version X.Y.Z --date $(date +%F)   # preview the section
```
then paste under a new `## [X.Y.Z]`, and `git rm changelog.d/*.md` (keep this README).

**CI** may run `python scripts/collate_changelog.py --check` to reject a malformed fragment name.

> The existing `## [Unreleased]` block in `CHANGELOG.md` (pre-#1521 entries) is promoted **once** by hand
> at the next version bump; from then on fragments own the changelog.
