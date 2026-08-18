# MFGArchon project instructions

`AGENTS.md` is this repository's authoritative project guidance. `CLAUDE.md` is only a
compatibility symlink to the same bytes. Cross-project guidance may be supplied by the host, but
project facts, rulings, and operating procedures are owned here and must not depend on a central
project manifest. Keep incident evidence in this repository's issues, docs, code, tests, or git
history. Record a separately generalized cross-project lesson outside the repository only when it
remains true after the MFGArchon-specific names and facts are removed.

---

## 🎯 Repository Mission & Scope ⚠️ CRITICAL

### MFGArchon: Public Infrastructure Package
Production-ready infrastructure for Mean Field Games research and applications.

**Scope**: ✅ core infrastructure (solvers, backends, config, geometry, workflow, visualization); ✅ classical numerical algorithms (FDM, FEM, GFDM); ⛔ the neural and RL families (DGM, PINN, Actor-Critic, PPO) are **in scope as a direction but FROZEN as code** — see the next section; ✅ standard examples (LQ, crowd motion, traffic flow, tutorials).

### `alg/neural/` and `alg/reinforcement/` were deleted, not frozen

They were frozen design prototypes for months and are now gone (23,118 lines, 60 files). The
deciding measurement: `boundary_conditions` appeared **zero times** in `alg/neural/`. They were
designed before the geometry and boundary-condition layer this library is built on, so they were
not BC-aware and had no seam to make them so — the gap was architectural, not a set of defects to
patch. Severing them touched exactly one import (`alg/__init__.py`).

Git history is the archive, the same reasoning that removed `archive/` in #1710. If either
paradigm returns, it returns designed around the geometry/BC layer rather than beside it.

### MFG-Research: Private Research Repository
Novel/experimental algorithms, unpublished methods. **Key principle**: MFG-Research **imports** MFGArchon but **never modifies** it.

### Decision criteria

| Criterion | MFGArchon (Public) | MFG-Research (Private) |
|:----------|:-----------------|:-----------------------|
| Maturity | Production-ready, tested | Experimental |
| Publication | Published methods | Unpublished |
| Stability | Stable API, versioned | Breaking changes OK |
| Documentation | Comprehensive | Minimal |
| Testing | Full coverage | Exploratory |

**Migration (research → infra)**: when research matures — add tests, write docs, ensure API consistency, open a PR in MFGArchon.

### Bug fixes from research ⚠️ CRITICAL
Before modifying MFGArchon for a bug found in mfg-research:
1. GitHub issue with quantified validation evidence.
2. Standalone validation experiment in mfg-research demonstrating the fix.
3. Discussion + approval of approach.
4. Reference the validation experiment in code comments:
```python
# Issue #542 fix. Validated in:
# mfg-research/experiments/crowd_evacuation_2d/runners/exp14b_fdm_bc_fix_validation.py
# Achieves 23x error reduction (47.98% -> 2.06%) for 1D corridor evacuation.
```
Keep changes minimal + focused; no legacy fallbacks (use mature utilities directly); run tests before and after; verify by re-running the research experiment.

---

## 🏗️ Repository Structure

**Top-level**: `mfgarchon/` (package), `tests/` (unit + integration only), `benchmarks/` (perf scripts), `examples/` (`basic/ advanced/ notebooks/ tutorials/`), `docs/`. There is no `archive/`: it was 16,634 lines excluded from pytest, ruff and the fail-fast ratchet, so it cost nothing to keep and nothing to delete — git history is the archive (#1710).

**Package** (`mfgarchon/`): `alg/ backends/ config/ core/ factory/ geometry/ hooks/ solvers/ types/ utils/ visualization/ workflow/ compat/ meta/`.

---

## 🎨 MFGArchon-specific conventions

### Primary API pattern
The domain model **is** the API (kernel scope discipline: no premature convenience/factory explosion — wrappers wait until post-1.0):
```python
problem = MFGProblem(...)
result = problem.solve()                                     # Auto Mode
result = problem.solve(scheme=NumericalScheme.FDM_UPWIND)    # Safe Mode: pick the scheme
result = problem.solve(hjb_solver=hjb, fp_solver=fp)         # Expert Mode: bring your own pair
result = problem.solve(max_iterations=200, tolerance=1e-8)   # explicit params, not magic "mode=" strings
```
There is no factory to reach for. `create_standard_solver` does not exist and
`create_fast_solver` / `create_research_solver` / `create_basic_solver` /
`create_accurate_solver` raise on call; `create_solver` is deprecated in favour of the
same three modes (#580). This file taught two of them as current until #1709.

### Import style
```python
from mfgarchon import MFGProblem, BoundaryConditions
from mfgarchon.types import NumericalScheme
from mfgarchon.utils.mfg_logging import get_logger, configure_research_logging
```

### Mathematical typesetting & emoji
There is no local override to the host's cross-project typography rules. Project notation is `u(t,x)`, `m(t,x)`.

### Physics conventions (single-source; #811/#1412/#1512)
- `problem.sigma` = SDE volatility $\sigma$; `problem.diffusion` = PDE coefficient $D = \sigma^2/2$. Never conflate.
- Resolve $\sigma \to D$ through the one converter `diffusion_from_volatility(sigma)`; never inline `0.5*sigma**2` in a solver.
- FP drift scale comes from `fp_drift_coefficient(problem)` (= 1/control_cost), not a private per-solver copy.
- The weak-form FP family receives the value function through `potential_field`; each backend differentiates it on its own FEM or MLS basis and obtains the velocity from `H.optimal_control`. Do not replace that path in isolation with a coupling-layer velocity computed on another basis.
- Preserve the paired HJB--FP operator relation when changing weak-form advection. Test conservation and discrete adjointness separately; one does not imply the other.

### File-path anchoring ⚠️ CRITICAL
Anchor output paths to **project root**, never CWD: ✅ `Path(__file__).resolve().parent.parent / "results"` or `${hydra:runtime.cwd}/results`; ❌ `Path("results")` / `os.getcwd()` (recursive nesting under `cd`).

### Boundary-condition coupling — adjoint-consistent BC (Issue #574, #625) ⚠️
Reflecting-boundary HJB couples to the FP density gradient for equilibrium consistency, via the **BCValueProvider** pattern: `AdjointConsistentProvider` stored in `BCSegment.value`, resolved at iteration time.
```python
from mfgarchon.geometry.boundary import AdjointConsistentProvider, BCSegment, BCType, BoundaryConditions
bc = BoundaryConditions(segments=[
    BCSegment(name="left_ac", bc_type=BCType.ROBIN, alpha=0.0, beta=1.0,
              value=AdjointConsistentProvider(side="left", sigma=sigma), boundary="x_min"),  # sigma is volatility (#1512); diffusion= is the deprecated alias
    BCSegment(name="right_ac", bc_type=BCType.ROBIN, alpha=0.0, beta=1.0,
              value=AdjointConsistentProvider(side="right", sigma=sigma), boundary="x_max"),
], dimension=1)
```
Internally: the iterator calls `problem.using_resolved_bc(state)` each Picard step; the provider computes $g = -\sigma^2/2 \cdot \partial\ln(m)/\partial n$; the solver receives a resolved BC (no coupling knowledge). **Use for**: boundary-stall reflecting configs (>1000× improvement in some cases). **Not for**: interior stall or periodic BC. Implementation: `geometry/boundary/providers.py`, `geometry/boundary/bc_coupling.py`, `alg/numerical/coupling/fixed_point_iterator.py`. Ref: `mfg-research/docs/archon-notes/development/TOWEL_ON_BEACH_1D_PROTOCOL.md`.

---

## 🧪 Testing — repo strategy

Cross-project testing discipline governs **what** a test must cover (edge/stress/failure cases, "coverage = paths whose failure you'd catch"). This section governs **where** a test lives — a hybrid approach for research code that evolves fast:

- **Unit tests (`tests/unit`, `tests/integration`)** — stable public APIs (`solve_mfg()`, factories), core infra (config/problem/result/backend), numerical correctness that must not regress. Run in CI on every commit.
- **Inline smoke tests (`if __name__ == "__main__"`)** — rapidly-changing algorithm implementations; visual verification; low-maintenance; delete naturally on refactor. `python mfgarchon/alg/numerical/hjb_solvers/my_solver.py`.
- **Examples (`examples/`)** — complete user workflows, not quick algorithm testing.

| Code type | Changes often? | Public API? | Test type |
|:----------|:--------------|:------------|:----------|
| `solve_mfg()`, config system | No | Yes | Unit |
| New HJB/FP solver | Yes | Maybe/No | Smoke |
| Visualization | Sometimes | Yes | Smoke |
| Utility function | No | Internal | Unit or smoke |

### Closing out a fix ⚠️ — name the oracle, or say there isn't one

"Add a test" is **not** the default close-out for a fix here. Measured on this repo: the six load-bearing
conventions the discrimination ratchet tracks are noticed by **212 distinct tests** out of 5,872 —
**3.6%** react when the physics the library exists to get right is broken, so **96.4% notice
nothing**. (The baseline's kill counts sum to 220; 8 tests are killed by more than one mutation, so
the sum over-counts and the honest figure is the lower one. Current numbers from
`scripts/discrimination_baseline.json` + `discrimination_killmatrix.json`, measured at `db3496f9`;
`./scripts/local_ci.sh` prints them beside the suite result and flags the denominator when it moves.)
Of the tests whose *names* claim `single_source` / `cross_path` / `_agree`, **60% are inert** —
[#1715's comment of 2026-07-27](https://github.com/derrring/MFGArchon/issues/1715#issuecomment-5090690985),
not its body, which says the prevalence "is not established". **Inert is not the same as worthless**,
and that distinction has cost real time: all five tests #1715 names are genuine cross-path pins —
delegation shims, builder-vs-operator GFDM weights, Newton-vs-Picard agreement — inert on six
conventions *because those conventions are not what they pin*. The deletable set is the
**structurally tautological** one, found by reading, not the inert one, found by counting (#1901). And
the yield runs the other way too: #1660's 17 nightly "failures" resolved as 8 fixture rot from the
#1442 drift migration, 2 tests measuring the wrong quantity, 7 timeouts — **zero** product
regressions caught by a test.

So state which of these the fix ends with, in the PR body, in this order of preference:

1. **An external oracle** — a law the scheme must reproduce, computed independently of the scheme.
   Regime masses against `M(0) @ expm(Qt)`; an LQG analytic solution; a closed form. Cannot go
   tautological when the two paths are later consolidated, which is what kills agreement tests.
2. **A mutation-verified convention pin** — the mutation and its kill count stated. Unverified, a
   pin is a claim about discrimination with no measurement behind it.
3. **A capability cell** — when the question is "can this configuration solve at all". Deliberately
   fixed-size: the matrix must not grow with the library.
4. **A happy-path assertion** — admit it as such, and say what it would not catch.

"There is no oracle for this yet" is an acceptable close-out and a fileable issue. A green suite is
not evidence: on every defect independent review found in #1802, the full local gate was green
(5776 and 5781 passed) with the defect present.

Adding tests must be justified by a failure they discriminate, not by "should test more". Assert on
the relevant disagreement, not merely on validity; every defect above sat on a covered line.

---

## 🔧 Development workflow

### Deprecation policy ⚠️ CRITICAL
Deprecated code MUST immediately redirect to the new standard: (1) old API calls new internally (zero behavior difference); (2) **mandatory equivalence test** (old == new); (3) update ALL call sites (direct, factory, defaults, examples, tests); (4) timeline 3 minor versions OR 6 months before removal. Ref: `DEPRECATION_LIFECYCLE_POLICY.md`. Lesson (Issue #616, `conservative=`): deprecated-with-wrong-default + factory-not-updated + no-equivalence-test → 1 month of 99.4%-mass-error bugs.

This repository is pre-1.0: fix a cross-cutting defect at the highest owning abstraction and remove
the redundant path instead of preserving a local fork. Breaking changes accumulate in
`CHANGELOG.md`'s `[Unreleased]` section and ship in the current `v0.MINOR.x` patch line unless the
maintainer explicitly chooses a minor release; do not add compatibility shims for unreleased code.

### Version-bump checklist ⚠️ MANDATORY
In a single commit:
1. `pyproject.toml:11` — `version = "X.Y.Z"` + inline `# vX.Y.Z: <one-line scope>`.
2. `CHANGELOG.md` — collate the `changelog.d/` fragments into a new `## [X.Y.Z] - YYYY-MM-DD` section: `python scripts/collate_changelog.py --version X.Y.Z --date $(date +%F)`, paste it under the new heading, then `git rm changelog.d/*.md` (keep the README). Keep-a-Changelog categories. *One-time (#1521):* the pre-#1521 `## [Unreleased]` block is promoted by hand at the first release after #1521; from then on fragments own the changelog.

Do **not** edit: `mfgarchon/__init__.py` (reads `importlib.metadata`), `workflow/__init__.py` (independent subpackage version), backend version reporting (external libs), historical version notes in docstrings. Sanity check: `grep -rn "^version =\|^__version__ =" pyproject.toml mfgarchon/` — only `pyproject.toml:11` should change.

### Branches & PRs
- **Branch naming (MANDATORY)**: `<type>/<short-description>` — `feature/ fix/ chore/ docs/ refactor/ test/`.
- **Never commit directly to `main`.** ⚠️ **Nothing enforces this — not the server, and not the local hook either.** Re-check with `gh api repos/derrring/MFGArchon/rules/branches/main` (the effective-rules endpoint, which covers org rules and migrated required workflows); an empty array means no branch rule applies. As of 2026-07-25 it is empty, classic protection returns 404, and the sole ruleset is `enforcement=disabled`. The pre-push hook does not close the gap: it runs `scripts/local_ci.sh`, which contains no branch logic at all (`grep -i branch scripts/local_ci.sh` → nothing) and no `no-commit-to-branch` hook is configured — it gates *test quality*, not *which ref you are on*, so a green suite on `main` pushes cleanly. Treat this as a discipline you keep, not a wall that stops you. Create the PR when you push; delete merged branches.
- **Prune local branches periodically**: `./scripts/prune_local_branches.sh` (dry run; `--delete` to act). Branch names do not record their fate, so it derives it — merged PR head ref, then a reverse-apply of the branch's diff against `main`, then a MERGED number in the name. Left alone they accumulate: 43 locally by 2026-08-18, of which 30 were dead. The concrete costs are that a branch held by a worktree blocks its own deletion while `git status` cannot see worktrees at all, and that classifying by hand needs a full merged-PR fetch every time. The script aborts rather than guessing if that fetch looks truncated — at 1121 merged PRs a `--limit 300` silently reports live branches as orphans.
- **PR granularity is a preference, not a mandate.** Granular (one fix / PR) is fine; batch *related, low-risk* fixes into one PR (one commit each, `Closes #A #B #C`) when convenient to save CI runs. Split out anything *risky / independent / large (>~1d)* regardless. The two pains that made granularity costly — CHANGELOG conflicts and red-main — are removed by *mechanism*: the fragment changelog, and the full-suite gate that now runs **locally** (`./scripts/local_ci.sh`, wired as a pre-push hook) rather than on GitHub. So this stays a convenience call, not a rule to remember.
- **Changelog per PR (#1521)**: add a `changelog.d/<slug>.<category>.md` fragment (category ∈ `added/changed/deprecated/removed/fixed`) — do **not** edit `CHANGELOG.md`. Fragments are separate files, so PRs never conflict on the changelog (batched or not). See `changelog.d/README.md`.
- **Before merge**: the **local** full suite is authoritative — `./scripts/local_ci.sh` (see *Pre-commit / pre-merge checks*). GitHub's PR checks are a fast tier only and green there is **not** sufficient.
- **Review before merge (MANDATORY)**: run an **independent adversarial review** of the PR before merging — a fresh reviewer (subagent / cross-model / worktree-isolated), *not* just author self-review. Merge only when it returns MERGE-OK, or after fixing every blocker it raises; re-review after applying fixes. Local-green ≠ correct: this has caught real bugs a passing suite hid (e.g. a level-set boundary regression invisible to symmetric test configs, #1602/#1605).

### GitHub issue/PR management ⚠️ MANDATORY

**Every issue carries all 4 label dimensions**: `priority:` (high/medium/low), `area:` (algorithms/config/core/documentation/geometry/performance/testing/visualization), `size:` (small=hrs–1d / medium=1–3d / large=1+wk), `type:` (bug/enhancement/chore/refactor/infrastructure/research/type-checking/question). Multiple `area:` allowed; one `priority:`/`size:` each; no bare labels (all prefixed). Workflow-state prefixes: `status:` (blocked/in-review/needs-testing), `resolution:` (merged/superseded/wontfix/duplicate/invalid). Non-taxonomic (GitHub conventions): `good first issue`, `help wanted`, `automated`.

```bash
gh issue edit N --add-label "priority: medium,area: algorithms,size: small,type: enhancement"
git checkout -b feature/descriptive-name
gh pr create --title "…" --body "Fixes #N" --label "priority: medium,area: algorithms,size: small,type: enhancement,status: in-review"
```
Feature process: issue (labelled) → branch → core code in `mfgarchon/<sub>` → examples → tests → docs → benchmarks → label PR to match.

### Ruff pinning
Pin the ruff version (reproducible formatting, no surprise CI failures). Monthly automated update via GitHub Action, or `python scripts/update_ruff_version.py`.

### `.gitignore` — targeted patterns (preserve valuable code) ⚠️
Root-level only: `/*.png`, `/*_analysis.py` (not global `*.png`). Always `!examples/**/*.py`, `!tests/**/*.py`, `!docs/**/*.md`.

### Pre-commit / pre-merge checks
```bash
pytest tests/unit/test_affected_module.py            # iterate on the affected module
./scripts/local_ci.sh --fast                         # lint/format/ratchet only — this is the one you run
./scripts/local_ci.sh                                # THE GATE — the pre-push hook runs this; see the note below
ruff check mfgarchon/affected_module.py && ruff format --check mfgarchon/affected_module.py
mypy mfgarchon/affected_module.py
```
⚠️ **Do not run the full gate by hand before pushing — the `pre-push` hook runs it.** Running it
manually and then pushing pays for it twice, ~2.5 min each, and the second run is the one that
decides. Measured 2026-08-18: a session that did this on every push spent about half its push time
re-running a gate it had just watched go green. The ladder for this repo is:

| step | command | cost |
|---|---|---|
| after every edit | `ruff check --fix … && ruff format …`, or `./scripts/local_ci.sh --fast` | seconds |
| after every edit | the test files you touched, by path | seconds |
| when behaviour could have changed | the blast radius **by path**, not by `-k` name match — a `-k "hjb or newton"` sweep measured 600 s+ and did not finish, while the whole gate is ~150 s | ~30 s |
| at the boundary | **`git push`** — the hook runs the gate, once | ~150 s |

The one case for running it by hand: you need to *read* its diagnostics (discrimination fraction,
capability baseline, fail-fast counts) rather than just pass. Then run it, and push with
`--no-verify` only if the working tree has not moved since.

⚠️ `local_ci.sh` runs `-n auto` (xdist parallel) + skip `slow` for you. If you invoke pytest by hand, match that: a bare `pytest tests/` is *serial* and includes `@slow`, which takes **hours** (not a hang — Issue #1522). A 900s per-test `timeout` (pytest-timeout) is the safety net for a genuine infinite loop. Set `MFG_PYTHON` if `python` is not the env you want.

**CI shape — the full suite runs LOCALLY, not on GitHub (2026-07-19):**

| Tier | What | Where | Cost |
|:-----|:-----|:------|:-----|
| Gate (authoritative) | fail-fast ratchet, capability matrix, full suite (CI marker set, `-n auto`, no coverage) | **local**, `./scripts/local_ci.sh` | ~4 min |
| PR checks | syntax, ruff format+lint, fail-fast ratchet, imports, **smoke subset** (`test_core` + `test_config`) | GitHub `ci.yml` | ~3 min |
| Backstop | full suite **incl. `@slow`**, excluding `@manual`; plus the capability matrix + its self-test | GitHub `nightly.yml`, 03:00 | `timeout-minutes: 300` |
| Release | full suite incl. `@slow`, excluding `@manual`, with coverage | GitHub `ci.yml` on `release` | 35 min budget |

Why: the full suite is ~141 s locally and **exceeded a 25-minute step budget** on a GitHub runner. Measured — coverage accounts for 1.5x, the runner itself for the rest (~7x slower than an M-series Mac). Online execution of the full suite was buying latency, not signal.

**The consequence you must not forget:** a GitHub-green PR has NOT had its tests run. `./scripts/local_ci.sh` before every push is the gate; state its result in the PR. A regression outside `test_core`/`test_config` will otherwise reach `main` and only surface in nightly.

---

## 📚 Documentation

### Three-tier policy ⚠️ CRITICAL

| Content | Location |
|---------|----------|
| User docs (tutorials, guides, API) | `mfgarchon/docs/user/` (public, future book) |
| Theory & design, architecture, roadmaps | **Joplin MFG notebook** (private) |
| Development guides (coding style, CI/CD, tooling) | `mfg-research/docs/archon-notes/development/` |
| Research notes (experiments, analysis) | `mfg-research/docs/`, `experiments/*/docs/` |
| Completed/historical | `mfg-research/docs/archon-notes/archive/` |

**Joplin ↔ archon-notes split**: see the Joplin Dev `[Principle]` note "Joplin MFG vs archon-notes — doc division of labor" (Joplin = evergreen knowledge-graph; archon-notes = git-versioned chronicle + dev handbook). Rules: never create `docs/theory|development|architecture/` in mfgarchon; never put internal planning/theory in the public repo; never create markdown design docs in repos — use Joplin. Cross-repo: design in Joplin → GitHub issue → implement with issue ref → update user docs if user-facing → bidirectional-link Joplin + issue.

### Development Plan Management
Plans, their naming, status prefixes and lifecycle: Joplin `MFG/Dev/Dev Principles` →
`[Principle] Joplin MFG vs archon-notes — doc division of labor` § Plan management.
Agent-facing rule not already stated above: do not start a second cross-Plan roadmap.

### Progressive logging ⚠️
Log incrementally, summarize at milestones only. During work: technical notes + TodoWrite, no frequent summaries. Bugs → GitHub issues (`gh issue create`), not `docs/bugs/*.md`. Create a summary only at phase completion / milestone / investigation conclusion — and **ask first**.

---

## 🎨 Visualization & Output

- **Plotting**: matplotlib with immediate `plt.show()` (primary); Plotly/Bokeh on demand. Publication-ready; notation `u(t,x)`, `m(t,x)`.
- **Notebook-based reporting** ⚠️ is the primary research-output form (algorithm comparisons, convergence, validation). Track `.ipynb` with cleared outputs + exported HTML.
- **Output dirs**: `examples/outputs/` (gitignored), `examples/outputs/reference/` (tracked), `benchmarks/results/` (gitignored), `benchmarks/reports/*.html` (tracked). Python scripts → `examples/outputs/[category]/`, never root.
- **Incremental data saving** ⚠️: long computations (GFDM, Picard sweeps) save after each iteration to HDF5 (append + metadata) — a crash must not lose progress.
- **Heavy tasks** ⚠️: never cap timeouts on long solvers; run in background (`run_in_background` / `&`); monitor via logs; trust incremental saves.

---

## 📊 Package management
Core: numpy, scipy, matplotlib. Interactive: plotly, jupyter, nbformat (with fallbacks). Progress: rich. Optional: psutil. Support dev (`pip install -e .`) and user installs.

---

## 🪵 Logging & progress bars
```python
from mfgarchon.utils.mfg_logging import get_logger, configure_research_logging
from mfgarchon.utils.progress import create_progress_bar, solver_progress
configure_research_logging("session_name", level="INFO"); logger = get_logger(__name__)
progress = create_progress_bar(range(max_iterations), verbose=True, desc="Picard")
for i in progress:
    progress.update_metrics(error=error)        # type-safe Protocol (Issue #587) — no hasattr
    if converged: progress.log("Converged!"); break
```
Rich-only backend (v0.16.15+; external tqdm removed — legacy alias kept). Use `create_progress_bar()`; no hasattr checks on progress bars.

---

## 📜 Solo Maintainer's Protocol
1. Propose in issue → 2. implement in a feature-branch PR → 3. **independent adversarial review** (fresh reviewer, not just self-review — mandatory, see *Branches & PRs*) → 4. **verify issue completion** → 5. merge only after review is MERGE-OK **and** all checks pass. **Self-enforced** — see *Branches & PRs*: the `main` ruleset exists but is `enforcement=disabled`, so no step of this protocol is blocked server-side.

**Issue-completion verification** ⚠️ (before closing an issue / opening a PR): read the *original* issue (not commit messages); check every acceptance criterion; answer every discussion point; confirm all subtasks; document deviations (update the issue before closing). Anti-pattern: closing on commit messages without re-reading the requirements.

---

## 🤖 AI interaction design
Pointer: `mfg-research/docs/archon-notes/development/AI_INTERACTION_DESIGN.md` (graduate-level rigor, complexity analysis, journal-quality exposition).

---

**Last restructured**: 2026-08-15 (project-owned `AGENTS.md`; central project scope retired). Pre-1.0.0.
