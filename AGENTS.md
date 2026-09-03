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

**Scope**: ✅ core infrastructure (solvers, backends, config, geometry, workflow, visualization); ✅ classical numerical algorithms (FDM, FEM, GFDM); ⛔ the neural and RL families (DGM, PINN, Actor-Critic, PPO) are **out of scope as code (deleted); the direction stays open** — see the next section; ✅ standard examples (LQ, crowd motion, traffic flow, tutorials).

### `alg/neural/` and `alg/reinforcement/` were deleted, not frozen

They are gone: designed before the geometry and boundary-condition layer this library is built
on, they had no seam to make them BC-aware. The gap was architectural, not a set of defects to patch.

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
```
Keep changes minimal + focused; no legacy fallbacks (use mature utilities directly); run tests before and after; verify by re-running the research experiment.

---

## 🏗️ Repository Structure

**Top-level**: `mfgarchon/` (package), `tests/` (unit + integration only), `benchmarks/` (perf scripts), `examples/` (`basic/ advanced/ notebooks/ tutorials/`), `docs/`. There is no `archive/`: git history is the archive (#1710).

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
same three modes (#580).

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
Internally: the iterator calls `problem.using_resolved_bc(state)` each Picard step; the provider computes $g = -\sigma^2/2 \cdot \partial\ln(m)/\partial n$; the solver receives a resolved BC (no coupling knowledge). **Use for**: boundary-stall reflecting configs. **Not for**: interior stall or periodic BC. Implementation: `geometry/boundary/providers.py`, `geometry/boundary/bc_coupling.py`, `alg/numerical/coupling/fixed_point_iterator.py`. Ref: `mfg-research/docs/archon-notes/development/TOWEL_ON_BEACH_1D_PROTOCOL.md`.

---

## 🧪 Testing — repo strategy

### Admission ⚠️ — a test earns its place before it is written (#2227)

**Do not write a test because a change "should have one".** Most of this suite was deleted for being
unauditable; adding to it casually is how it got there.

**A new test must be one of these, and the PR says which:**

1. **It kills a mutation.** `scripts/discrimination_killmatrix.json` maps node ID → mutations killed:
   read it. Re-running the sweep is not the price of admission — it takes ~26 min and leaves
   mutations in the tree when killed (#1849, #2229) — so re-measure only when adding a mutation.
2. **It is an external oracle** — a law the scheme must reproduce, computed independently of it.
   `M(0) @ expm(Qt)`, `rho_i * exp(v_n*dx/D)`, an LQG closed form, an Itô isometry. These do not rot
   when the API moves, because they pin mathematics rather than signatures.
3. **It is a labelled defect pin** carrying its own retirement condition, so fixing the defect trips
   it and the failure message is the instruction.
4. **It guards an instrument** in `scripts/` — the capability matrix, the ratchets, their baselines.

`hasattr`, a signature, `raises(TypeError)`: these rot the first time the API moves, and they rot
**silently**, still passing while measuring nothing.

**Three questions, three sections. None overrules another.**

| question | answered by |
|---|---|
| what does a **fix** end with? | § *Closing out a fix* — and *"there is no oracle for this yet"* plus a filed issue is an accepted close-out that writes **no test** |
| does a proposed **test** earn its place? | here |
| **where** does an admitted test live? | the table below, under this same `##`. It says which *kind* a thing gets **if** a test is written; it does not oblige one |

~~This section is the same ladder as § *Closing out a fix*, used for admission rather than as a
report.~~ **[RETRACTED 2026-09-03]** The ladders differ in order and in membership. The residue is
real and is not dissolved: that section's class 4 is a happy-path assertion this one does not admit.
Read it as — a happy-path assertion can be an honest way to **close** a fix, admitted as such and
saying what it would not catch; it is not a reason to **add** a test. Writing one anyway is the
moment to file the "no oracle yet" issue instead.

**"Which convention does it defend" is a question, not a lookup.** The Joplin `[Principle]
Conventions Index` names 35 conventions and `scripts/discrimination_baseline.json` carries 24
mutations, and **the two sets are nearly disjoint** (measured in #2227). Neither is the list. A test
defending something on neither is admissible
under class 1 or 2 on its own merits. The index is a private Joplin note that a reader outside this
machine cannot open, so state the convention in the test rather than cite the index.

**Nothing here licenses deleting a test on a count.** The deletable set is the structurally
tautological one, found by **reading**; inert-by-counting is a different set, and #1715 records five
genuine cross-path pins inside it. Classes 2–4 are judgements: apply them by reading the candidate,
never by matching a keyword.

Cross-project testing discipline governs **what** a test must cover (edge/stress/failure cases, "coverage = paths whose failure you'd catch"). This section governs **where** a test lives — a hybrid approach for research code that evolves fast:

- **Unit tests (`tests/unit`, `tests/integration`)** — stable public APIs (`problem.solve()`, factories), core infra (config/problem/result/backend), numerical correctness that must not regress. Run in CI on every commit.
- **Inline smoke tests (`if __name__ == "__main__"`)** — rapidly-changing algorithm implementations; visual verification; low-maintenance; delete naturally on refactor. `python mfgarchon/alg/numerical/hjb_solvers/my_solver.py`.
- **Examples (`examples/`)** — complete user workflows, not quick algorithm testing.

| Code type | Changes often? | Public API? | Test type |
|:----------|:--------------|:------------|:----------|
| `problem.solve()`, config system | No | Yes | Unit |
| New HJB/FP solver | Yes | Maybe/No | Smoke |
| Visualization | Sometimes | Yes | Smoke |
| Utility function | No | Internal | Unit or smoke |

### The dimension must be able to express the property under test ⚠️

Not "prefer 2D" and not "1D is cheaper". The rule is that **a test in a dimension that cannot
express the property is not a weak test — it cannot fail at all**, and it reads exactly like a
passing one.

- **Needs d ≥ 2**: anything involving a normal/tangential decomposition, an axis pairing, a corner,
  a transpose, or anisotropy. A 1D wall's normal is always the coordinate axis and there is no
  tangential component, so a scheme that mishandles the tangential part passes 1D by construction.
  Note the scope: d ≥ 2 is **necessary** for these, not sufficient.
- **Fully expressed in d = 1**: sign conventions, time-stepping order, convergence criteria, scalar
  contracts, source-term plumbing. Here 2D buys runtime and nothing else.

**The discriminator is measurable, so do not argue it.** Write a mutation that breaks the property,
run it in 1D, and read the difference.

**`max|diff| = 0` in 1D means the test must go up a dimension. It does not mean going up is
sufficient.** A 2D fixture that is symmetric in the relevant variable cannot express a defect in
that variable either, and it looks exactly as healthy as one that can.

Two things make a fixture able to separate a defect from its absence: a half-period wavenumber, so the mirror
ghost and the periodic ghost stop coinciding; and an asymmetry between the axes, so transposing the
field is detectable. Neither is about dimension. **The question is never "is this 2D" — it is "can this fixture express the defect",
and dimension is only the first thing that can make the answer no.**

The burden runs both ways: a 2D test whose 1D reduction separates the same mutants owes its runtime
an explanation, and a test of a directional property — in any dimension — owes a mutant showing it
can fail.

This is the cross-project "non-discriminating test data" rule applied to dimension — the same shape
as a uniform density that cannot separate a gradient-form bug from a correct scheme.

### Capturing a log record in a test ⚠️

Use the **`mfg_caplog`** fixture (`tests/conftest.py`), never plain `caplog`. `MFGLogger` sets
`propagate = False` (`logger.py:211`), so whether `caplog` sees an mfgarchon record differs by
pytest version, and the two in use here — the gate interpreter's and the one `uv run --group dev`
resolves — disagree. `mfg_caplog` removes the dependence.

```python
def test_the_drift_is_reported(mfg_caplog):
    with mfg_caplog.at_level(logging.WARNING, logger="mfgarchon.alg....fp_gfdm"):
        solver.solve_fp_system(m0, drift)
    assert mfg_caplog.messages  # or .records for the LogRecord itself
```

`logger=` is required — there is no root to fall back to, and the no-argument form would capture
nothing silently.

**A *wrong* name is the same failure wearing a different face, and the fixture does not catch it.**
`assert not mfg_caplog.records` is satisfied by a typo exactly as it is by a solve that did not warn.
The discipline that catches it is at the call site: **pair every absence assertion with a presence
assertion on the same logger name**, so a typo fails the presence half loudly.

### Closing out a fix ⚠️ — name the oracle, or say there isn't one

"Add a test" is **not** the default close-out for a fix here. Most of this suite does not react when
the physics the library exists to get right is broken, and the conventions that are defended are
defended very unevenly — some by a single-digit number of tests, some by over a hundred.

**The current numbers are not written here, on purpose.** `./scripts/local_ci.sh` prints them beside
the suite result, with its own staleness flag when the suite has moved since the baseline was
recorded. A fraction copied into this file goes stale the day the mutation list or the suite moves,
and both move — the baseline was re-recorded six times in the month to 2026-08-22.

**Read the vector, not the fraction** — an aggregate over the whole suite cannot show a convention
held by two tests, which is the thing you would act on (#2148). Two cautions when you do:
`scripts/discrimination_baseline.json` holds the per-convention kill counts — the vector. The
distinct-test figure the gate prints is in `scripts/discrimination_killmatrix.json` and nowhere
else; summing the baseline's counts does not give it, because a test that kills several mutations is
counted in each.
**Inert is not the same as worthless.** The deletable set is the **structurally tautological**
one, found by reading, not the inert one, found by counting (#1901, #1715).

So state which of these the fix ends with, in the PR body, in this order of preference:

1. **An external oracle** — a law the scheme must reproduce, computed independently of the scheme.
   Regime masses against `M(0) @ expm(Qt)`; an LQG analytic solution; a closed form. Cannot go
   tautological when the two paths are later consolidated, which is what kills agreement tests.
2. **A mutation-verified convention pin** — the mutation and its kill count stated. Unverified, a
   pin is a claim about discrimination with no measurement behind it.
3. **A capability cell** — when the question is "can this configuration solve at all". Deliberately
   fixed-size: the matrix must not grow with the library.
4. **A happy-path assertion** — admit it as such, and say what it would not catch.

"There is no oracle for this yet" is an acceptable close-out and a fileable issue. A green suite is not evidence: independent review has found defects with the full local gate green (#1802).

Adding tests must be justified by a failure they discriminate, not by "should test more". Assert on
the relevant disagreement, not merely on validity; every defect above sat on a covered line.

---

## 🔧 Development workflow

### Deprecation policy ⚠️ CRITICAL
Deprecated code MUST immediately redirect to the new standard: (1) old API calls new internally (zero behavior difference); (2) **mandatory equivalence test** (old == new); (3) update ALL call sites (direct, factory, defaults, examples, tests); (4) timeline 3 minor versions OR 6 months before removal. Ref: `DEPRECATION_LIFECYCLE_POLICY.md`.

This repository is pre-1.0: fix a cross-cutting defect at the highest owning abstraction and remove
the redundant path instead of preserving a local fork. Breaking changes accumulate as `changelog.d/` fragments (#1521) and ship in the current `v0.MINOR.x` patch line unless the
maintainer explicitly chooses a minor release; do not add compatibility shims for unreleased code.

### Version-bump checklist ⚠️ MANDATORY
In a single commit:
1. `pyproject.toml:11` — `version = "X.Y.Z"` + inline `# vX.Y.Z: <one-line scope>`.
2. `CHANGELOG.md` — collate the `changelog.d/` fragments into a new `## [X.Y.Z] - YYYY-MM-DD` section: `python scripts/collate_changelog.py --version X.Y.Z --date $(date +%F)`, paste it under the new heading, then `git rm changelog.d/*.md` (keep the README). Keep-a-Changelog categories. *One-time (#1521):* the pre-#1521 `## [Unreleased]` block is promoted by hand at the first release after #1521; from then on fragments own the changelog.
3. `scripts/citation_baseline.json` — re-record it: `python scripts/check_citations.py --write-baseline`. Step 2 moves `changelog.d/` prose into `CHANGELOG.md`, which the citation ratchet exempts, so **both its pins fire on a correct release**. This is mechanical and expected; the gate says so in its own message.

Do **not** edit: `mfgarchon/__init__.py` (reads `importlib.metadata`), `workflow/__init__.py` (independent subpackage version), backend version reporting (external libs), historical version notes in docstrings. Sanity check: `grep -rn "^version =\|^__version__ =" pyproject.toml mfgarchon/` — only `pyproject.toml:11` should change.

### Branches & PRs
- **Branch naming (MANDATORY)**: `<type>/<short-description>` — `feature/ fix/ chore/ docs/ refactor/ test/`.
- **Never commit directly to `main`.** ⚠️ **Nothing enforces this — not the server, and not the local hook either.** Re-check with `gh api repos/derrring/MFGArchon/rules/branches/main` (the effective-rules endpoint, which covers org rules and migrated required workflows); an empty array means no branch rule applies. The pre-push hook does not close the gap: it runs `scripts/local_ci.sh`, which contains no branch logic at all (`grep -i branch scripts/local_ci.sh` → nothing) and no `no-commit-to-branch` hook is configured — it gates *test quality*, not *which ref you are on*, so a green suite on `main` pushes cleanly. Treat this as a discipline you keep, not a wall that stops you. Create the PR when you push; delete merged branches.
- **Prune local branches periodically**: `./scripts/prune_local_branches.sh` **classifies and prints evidence; it never deletes.** Deletion stays manual because each of its three signals can be wrong in a way that costs unmerged work — the content check reverse-applies against the *working tree* rather than `main`, so its verdict moves with the checkout and with uncommitted edits; the merged-PR check matches head-ref *names*, and a head-ref name can belong to more than one merged PR; and a 3–4 digit run in a branch name may be a grid size rather than an issue. Read the branch, record the sha, then `git branch -D`. The script aborts rather than guessing if the merged-PR fetch looks truncated. Note `git status` never reports what a worktree costs: one `??` entry at best, and nothing at all under an ignored path such as `.claude/worktrees/` (`.gitignore:77`), so use `git worktree list`.
- **PR granularity is a preference, not a mandate.** Granular (one fix / PR) is fine; batch *related, low-risk* fixes into one PR (one commit each, `Closes #A #B #C`) when convenient to save CI runs. Split out anything *risky / independent / large (>~1d)* regardless.
- **Changelog per PR (#1521)**: add a `changelog.d/<slug>.<category>.md` fragment (category ∈ `added/changed/deprecated/removed/fixed`) — do **not** edit `CHANGELOG.md`. Fragments are separate files, so PRs never conflict on the changelog (batched or not). See `changelog.d/README.md`.
- **Before merge**: the **local** full suite is authoritative — `./scripts/local_ci.sh` (see *Pre-commit / pre-merge checks*). GitHub's PR checks are a fast tier only and green there is **not** sufficient.
- **Review before merge (MANDATORY)**: run an **independent adversarial review** of the PR before merging — a fresh reviewer (subagent / cross-model / worktree-isolated), *not* just author self-review. Merge only when it returns MERGE-OK, or after fixing every blocker it raises; re-review after applying fixes. Local-green ≠ correct.

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
decides. The ladder for this repo is:

| step | command | cost |
|---|---|---|
| after every edit | `ruff check --fix … && ruff format …`, or `./scripts/local_ci.sh --fast` | seconds |
| after every edit | the test files you touched, by path | seconds |
| when behaviour could have changed | the blast radius **by path**, not by `-k` name match | ~30 s |
| at the boundary | **`git push`** — the hook runs the gate, once | ~150 s |

The one case for running it by hand: you need to *read* its diagnostics (discrimination fraction,
capability baseline, fail-fast counts) rather than just pass. Then run it, and push with
`--no-verify` only if the working tree has not moved since.

⚠️ **From a `git worktree` the gate names the tree it measured, and refuses if it is the wrong
one.** `gate package : <path>` sits beside `gate interpreter`, in the head and in the pasted tail,
and `scripts/local_ci.sh` exits before any check when that path is not under the tree being gated
(#2154).

**No static count tells you which steps import.** `capability_census.py:78` is
`importlib.import_module(package)` — a real in-process import that no import-shaped text pattern and
no AST import-scan can see. Run the blocking finder over the actual invocation instead.

**Set `MFG_PYTHON`; `PYTHONPATH` is not yours to remember** — the gate binds it for itself
(#2154) — bound onto its own `scripts/*.py` invocations, deliberately **not** an `export`, because
`-P` removes CWD from `sys.path` but not `PYTHONPATH`, and an exported root re-arms the `-m`
shadowing the `-P` exists to prevent. That is what makes the pre-push hook usable from a worktree. `MFG_PYTHON` stays yours:
interpreter selection is a judgement the gate cannot make for you, and it is **not optional when a
virtualenv is active**, because the gate's `CANDIDATES=(python python3 <mfg_env>)` tries PATH first.

⚠️ **Do not give a worktree its own virtualenv.** The hazard is the activated venv itself, whatever
it holds: it satisfies the gate's probe in full, so nothing about the selection looks wrong while it
runs the wrong toolchain and reports warning identities the pinned one cannot emit. `uv sync`
reproduces the project's resolved set; ruff is not in it and has one owner:
`uv pip install "ruff==$(python scripts/update_ruff_version.py --print-current)"`.

The gate names the mismatch while it happens, in a line that reads as a nag: `WARN ruff 0.13.1 ran,
but .pre-commit-config.yaml pins 0.16.0`. Treat that WARN as a refusal.

⚠️ `local_ci.sh` runs `-n auto` (xdist parallel) + skip `slow` for you. If you invoke pytest by hand, match that: a bare `pytest tests/` is *serial* and includes `@slow`, which takes **hours** (not a hang — Issue #1522). A 900s per-test `timeout` (pytest-timeout) is the safety net for a genuine infinite loop. Set `MFG_PYTHON` if `python` is not the env you want.

**CI shape — the full suite runs LOCALLY, not on GitHub (2026-07-19):**

| Tier | What | Where | Cost |
|:-----|:-----|:------|:-----|
| Gate (authoritative) | fail-fast ratchet, capability matrix, full suite (CI marker set, `-n auto`, no coverage) | **local**, `./scripts/local_ci.sh` | ~4 min |
| PR checks | syntax, ruff format+lint, fail-fast ratchet, imports, **smoke subset** (`test_core` + `test_config`) | GitHub `ci.yml` | ~3 min |
| Backstop | full suite **incl. `@slow`**, excluding `@manual`; plus the capability matrix + its self-test | GitHub `nightly.yml`, 03:00 | `timeout-minutes: 300` |
| Release | full suite incl. `@slow`, excluding `@manual`, with coverage | GitHub `ci.yml` on `release` | 35 min budget |

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
