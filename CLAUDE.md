# MFGArchon — Claude Code project instructions

@~/code/dotfiles/agent_axiom/domains/cs/_core.md
@~/code/dotfiles/agent_axiom/domains/cs/python.md
@~/code/dotfiles/agent_axiom/domains/math/_core.md
@~/code/dotfiles/agent_axiom/domains/math/mfg.md

> **Composition.** The global `~/.claude/CLAUDE.md` already loads the axiom **kernel + tools + audit mode**; the four imports above add the **CS + MFG domains**. **Universal behavior lives in the axiom, not here** — do not restate it in this file. Already owned by the axiom (`core/00_kernel.md`, `modes/audit.md`, `domains/cs/python.md`):
> - cold, honest, no-flattery stance;
> - fail-fast / no-silent-fallback / no-over-defensive-guards;
> - no `hasattr()` duck typing → `getattr(o,"x",None)` + `callable`/Protocol/ABC;
> - testing discipline ("coverage = paths whose failure you'd catch", edge/stress/failure cases);
> - single-source-of-truth (one owner + pinning test for any quantity computed on ≥2 paths);
> - scope discipline (no premature abstraction/convenience), doc status tags (`[SUPERSEDED]` etc.).
>
> This file holds **only what is true for THIS repo**. A new universal pattern starts here as a project override and graduates into the axiom after it recurs across N≥3 repos (`agent_axiom/README.md` § Editing protocol).

---

## 🎯 Repository Mission & Scope ⚠️ CRITICAL

### MFGArchon: Public Infrastructure Package
Production-ready infrastructure for Mean Field Games research and applications.

**Scope**: ✅ core infrastructure (solvers, backends, config, geometry, workflow, visualization); ✅ classical algorithms (FDM, FEM, GFDM, DGM, PINN, Actor-Critic, PPO); ✅ standard examples (LQ, crowd motion, traffic flow, tutorials).

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

**Top-level**: `mfgarchon/` (package), `tests/` (unit + integration only), `benchmarks/` (perf scripts), `examples/` (`basic/ advanced/ notebooks/ tutorials/`), `docs/`, `archive/` (historical — do not modify).

**Package** (`mfgarchon/`): `alg/ backends/ config/ core/ factory/ geometry/ hooks/ solvers/ types/ utils/ visualization/ workflow/ compat/ meta/`.

---

## 🎨 MFGArchon-specific conventions

### Primary API pattern
The domain model **is** the API (kernel scope discipline: no premature convenience/factory explosion — wrappers wait until post-1.0):
```python
problem = MFGProblem(...)
result = problem.solve()
result = problem.solve(max_iterations=200, tolerance=1e-8)   # explicit params, not magic "mode=" strings
solver = create_standard_solver(problem, custom_config=config)   # factory only when truly needed
```

### Import style
```python
from mfgarchon import MFGProblem, BoundaryConditions
from mfgarchon.factory import create_fast_solver
from mfgarchon.utils.mfg_logging import get_logger, configure_research_logging
```

### Mathematical typesetting & emoji
Graduated to axiom (2026-05-01): markdown LaTeX-only (no Unicode 𝒯/ℝ) → `core/00_kernel.md`; Python docstring UTF-8 math + ASCII logs + no code emojis → `domains/cs/python.md`. No local override; notation is `u(t,x)`, `m(t,x)`.

### Physics conventions (single-source; #811/#1412/#1512)
- `problem.sigma` = SDE volatility **σ**; `problem.diffusion` = PDE coefficient **D = σ²/2**. Never conflate.
- Resolve σ→D through the one converter `diffusion_from_volatility(σ)`; never inline `0.5*sigma**2` in a solver.
- FP drift scale comes from `fp_drift_coefficient(problem)` (= 1/control_cost), not a private per-solver copy.

### File-path anchoring ⚠️ CRITICAL
Anchor output paths to **project root**, never CWD: ✅ `Path(__file__).resolve().parent.parent / "results"` or `${hydra:runtime.cwd}/results`; ❌ `Path("results")` / `os.getcwd()` (recursive nesting under `cd`).

### Boundary-condition coupling — adjoint-consistent BC (Issue #574, #625) ⚠️
Reflecting-boundary HJB couples to the FP density gradient for equilibrium consistency, via the **BCValueProvider** pattern: `AdjointConsistentProvider` stored in `BCSegment.value`, resolved at iteration time.
```python
from mfgarchon.geometry.boundary import AdjointConsistentProvider, BCSegment, BCType, BoundaryConditions
bc = BoundaryConditions(segments=[
    BCSegment(name="left_ac", bc_type=BCType.ROBIN, alpha=0.0, beta=1.0,
              value=AdjointConsistentProvider(side="left", sigma=sigma), boundary="x_min"),  # sigma=σ (#1512); diffusion= is the deprecated alias
    BCSegment(name="right_ac", bc_type=BCType.ROBIN, alpha=0.0, beta=1.0,
              value=AdjointConsistentProvider(side="right", sigma=sigma), boundary="x_max"),
], dimension=1)
```
Internally: the iterator calls `problem.using_resolved_bc(state)` each Picard step; the provider computes `g = -σ²/2 · ∂ln(m)/∂n`; the solver receives a resolved BC (no coupling knowledge). **Use for**: boundary-stall reflecting configs (>1000× improvement in some cases). **Not for**: interior stall or periodic BC. Implementation: `geometry/boundary/providers.py`, `geometry/boundary/bc_coupling.py`, `alg/numerical/coupling/fixed_point_iterator.py`. Ref: `mfg-research/docs/archon-notes/development/TOWEL_ON_BEACH_1D_PROTOCOL.md`.

---

## 🧪 Testing — repo strategy (*what* counts as tested is axiom)

The axiom testing discipline governs **what** a test must cover (edge/stress/failure cases, "coverage = paths whose failure you'd catch"). This section governs **where** a test lives — a hybrid approach for research code that evolves fast:

- **Unit tests (`tests/unit`, `tests/integration`)** — stable public APIs (`solve_mfg()`, factories), core infra (config/problem/result/backend), numerical correctness that must not regress. Run in CI on every commit.
- **Inline smoke tests (`if __name__ == "__main__"`)** — rapidly-changing algorithm implementations; visual verification; low-maintenance; delete naturally on refactor. `python mfgarchon/alg/numerical/hjb_solvers/my_solver.py`.
- **Examples (`examples/`)** — complete user workflows, not quick algorithm testing.

| Code type | Changes often? | Public API? | Test type |
|:----------|:--------------|:------------|:----------|
| `solve_mfg()`, config system | No | Yes | Unit |
| New HJB/FP solver, experimental RL | Yes | Maybe/No | Smoke |
| Visualization | Sometimes | Yes | Smoke |
| Utility function | No | Internal | Unit or smoke |

---

## 🔧 Development workflow

### Deprecation policy ⚠️ CRITICAL
Deprecated code MUST immediately redirect to the new standard: (1) old API calls new internally (zero behavior difference); (2) **mandatory equivalence test** (old == new); (3) update ALL call sites (direct, factory, defaults, examples, tests); (4) timeline 3 minor versions OR 6 months before removal. Ref: `DEPRECATION_LIFECYCLE_POLICY.md`. Lesson (Issue #616, `conservative=`): deprecated-with-wrong-default + factory-not-updated + no-equivalence-test → 1 month of 99.4%-mass-error bugs.

### Version-bump checklist ⚠️ MANDATORY
In a single commit:
1. `pyproject.toml:11` — `version = "X.Y.Z"` + inline `# vX.Y.Z: <one-line scope>`.
2. `CHANGELOG.md` — collate the `changelog.d/` fragments into a new `## [X.Y.Z] - YYYY-MM-DD` section: `python scripts/collate_changelog.py --version X.Y.Z --date $(date +%F)`, paste it under the new heading, then `git rm changelog.d/*.md` (keep the README). Keep-a-Changelog categories. *One-time (#1521):* the pre-#1521 `## [Unreleased]` block is promoted by hand at the first release after #1521; from then on fragments own the changelog.

Do **not** edit: `mfgarchon/__init__.py` (reads `importlib.metadata`), `workflow/__init__.py` (independent subpackage version), backend version reporting (external libs), historical version notes in docstrings. Sanity check: `grep -rn "^version =\|^__version__ =" pyproject.toml mfgarchon/` — only `pyproject.toml:11` should change.

### Branches & PRs
- **Branch naming (MANDATORY)**: `<type>/<short-description>` — `feature/ fix/ chore/ docs/ refactor/ test/`.
- **Never commit directly to `main`** (branch protection enforces this). Create the PR when you push; delete merged branches.
- **PR granularity is a preference, not a mandate.** Granular (one fix / PR) is fine; batch *related, low-risk* fixes into one PR (one commit each, `Closes #A #B #C`) when convenient to save CI runs. Split out anything *risky / independent / large (>~1d)* regardless. The two pains that made granularity costly — CHANGELOG conflicts and red-main — are being removed by *mechanism* (fragment changelog + a full-suite PR gate; see the enforcement issues), so this stays a convenience call, not a rule to remember.
- **Changelog per PR (#1521)**: add a `changelog.d/<slug>.<category>.md` fragment (category ∈ `added/changed/deprecated/removed/fixed`) — do **not** edit `CHANGELOG.md`. Fragments are separate files, so PRs never conflict on the changelog (batched or not). See `changelog.d/README.md`.
- **Before merge**: the full **Test Suite** is authoritative (see *Pre-commit / pre-merge checks*); a green fast tier alone is not enough.
- **Review before merge (MANDATORY)**: run an **independent adversarial review** of the PR before merging — a fresh reviewer (subagent / cross-model / worktree-isolated), *not* just author self-review. Merge only when it returns MERGE-OK, or after fixing every blocker it raises; re-review after applying fixes. Local-green ≠ correct: this has caught real bugs a passing suite hid (e.g. a level-set boundary regression invisible to symmetric test configs, #1602/#1605). Cf. axiom `feedback_pre_pr_adversarial_review`.

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
pytest tests/ -n auto -m "not slow and not benchmark and not experimental and not optional_torch and not environment"   # full local sign-off — MATCH CI (parallel + skip slow)
ruff check mfgarchon/affected_module.py && ruff format --check mfgarchon/affected_module.py
mypy mfgarchon/affected_module.py
```
⚠️ Run the full suite **the way CI does** — `-n auto` (xdist parallel) + skip `slow`. A bare `pytest tests/` is *serial* and includes `@slow`, which takes **hours** (not a hang — Issue #1522). A 900s per-test `timeout` (pytest-timeout) is the safety net for a genuine infinite loop.

**CI shape (light gate):** the fast **`ci.yml` Test Suite** (parallel, skip-slow) gates PRs; **`nightly.yml`** runs the full suite incl. `@slow` at 03:00. So *fast-tier-green ≠ full-green* — a slow-only or `@slow`-only regression surfaces in nightly, not on the PR. Before a release, run the full suite locally or trigger nightly on demand.

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

### Development Plan Management ⚠️ CRITICAL
Plans live in Joplin **Dev** folder. Naming `{焦点} Plan — {日期范围}`. Scope 2–4 months (longer → split). **Only 1 active Plan** at a time. On completion → `[COMPLETED]` + move to Archive; on replacement → `[SUPERSEDED by …]` + Archive. Plan explains *why this order*; issues define *what to do*. `[Principle]` prefix = permanent design-philosophy doc (never archived).

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
1. Propose in issue → 2. implement in a feature-branch PR → 3. **independent adversarial review** (fresh reviewer, not just self-review — mandatory, see *Branches & PRs*) → 4. **verify issue completion** → 5. merge only after review is MERGE-OK **and** all checks pass. Enforced by branch protection on `main`.

**Issue-completion verification** ⚠️ (before closing an issue / opening a PR): read the *original* issue (not commit messages); check every acceptance criterion; answer every discussion point; confirm all subtasks; document deviations (update the issue before closing). Anti-pattern: closing on commit messages without re-reading the requirements.

---

## 🤖 AI interaction design
Pointer: `mfg-research/docs/archon-notes/development/AI_INTERACTION_DESIGN.md` (graduate-level rigor, complexity analysis, journal-quality exposition).

---

**Last restructured**: 2026-07-04 (composed from `agent_axiom` domains + pruned axiom-duplication). Pre-1.0.0.
