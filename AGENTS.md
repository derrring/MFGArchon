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
  Note the scope: d ≥ 2 is **necessary** for these, not sufficient — see the symmetry measurement
  below.
- **Fully expressed in d = 1**: sign conventions, time-stepping order, convergence criteria, scalar
  contracts, source-term plumbing. Here 2D buys runtime and nothing else — measured, the same FP
  MMS study runs 0.1 s at d = 1 and 1.5 s at d = 2.

**The discriminator is measurable, so do not argue it.** Write a mutation that breaks the property,
run it in 1D, and read the difference. Measured against `test_fp_mms_wall_order_1728.py` (#1728/#2006):

```
drop the tangential advection at wall rows   1D: max|diff| = 0.000e+00
flip its sign                                1D: max|diff| = 0.000e+00
read the potential along the wrong axis      1D: max|diff| = 0.000e+00
```

Three mutants, three exact zeros — **and only two of the three separate in 2D.** That last figure is
the more useful one, and it is the second half of the rule.

**`max|diff| = 0` in 1D means the test must go up a dimension. It does not mean going up is
sufficient.** A 2D fixture that is symmetric in the relevant variable cannot express a defect in
that variable either, and it looks exactly as healthy as one that can. Measured on
`test_coupled_mms_2d_no_flux.py` (#2016), where the paper's verbatim manufactured pair is
transpose-symmetric and periodic on the box:

| mutant | verbatim 2D pair | after breaking the symmetry |
|---|---|---|
| swap the whole BC family, `no_flux` → `periodic` | byte-identical | `eu` 3.0125e-01 → 1.2939e+01 (42×), EOC → −0.118 |
| read the potential along the wrong axis (`U[0].T`) | bit-identical | 45.7× / 81.8× / 119.5× |

Two changes bought those: a half-period wavenumber, so the mirror ghost and the periodic ghost stop
coinciding; and an asymmetry between the axes, so transposing the field is detectable. Neither is
about dimension. **The question is never "is this 2D" — it is "can this fixture express the defect",
and dimension is only the first thing that can make the answer no.**

The burden runs both ways: a 2D test whose 1D reduction separates the same mutants owes its runtime
an explanation, and a test of a directional property — in any dimension — owes a mutant showing it
can fail.

This is the cross-project "non-discriminating test data" rule applied to dimension — the same shape
as a uniform density that cannot separate a gradient-form bug from a correct scheme.

### Capturing a log record in a test ⚠️

Use the **`mfg_caplog`** fixture (`tests/conftest.py`), never plain `caplog`. `MFGLogger` sets
`propagate = False` (`logger.py:211`), so whether `caplog` sees an mfgarchon record differs by
pytest version, and the two versions in use here disagree:

| pytest | what its capture handler attaches to | consequence |
|---|---|---|
| 8.4.1 (`uv run --extra dev`) | the root logger only | **no mfgarchon record is ever seen**, whatever the logger's creation site. Here `propagate = False` is the whole story |
| 9.1.1 (the gate interpreter) | root, plus every non-propagating logger that **already exists** when `catching_logs.__enter__` runs | a logger that existed before this phase's sweep is visible; one born after it is not |

That sweep runs **once per test phase** (setup / call / teardown), so the discriminator is not
"module level vs inside a function": a logger created in a *fixture* is visible in the test body.
It is whether the logger existed before this phase's sweep — and a logger born mid-solve did not.
pytest's own comment names the gap: the sweep "will miss loggers that *become* non-propagating
after the `__enter__`", which is exactly when `MFGLogger` sets it.

34 of the package's 104 `get_logger` calls are inside a function. 19 of those 34 are
`mfg_logging`'s own plumbing; the other 15 are consumer-side, and 12 of the 15 have `fp_gfdm`'s
shape — a module logger obtained mid-call. So on 9.1.1 the result depends on test order: measured at #2083, the gfdm drift test **fails run alone** and **passes** when a
sibling test ran a solve first. Six test modules had each rediscovered some part of this and
written their own collecting handler.

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

Two guards were built and both removed, because each re-created the order-dependence this fixture
exists to remove — the record is in `at_level`'s docstring and in the issue tracking a sound design.
The short version: a runtime criterion ("has the package handed this name out") fires on a correct
absence assertion over a logger created inside a function, and a static one (`find_spec`) does not
help because **10 of the 11 logger names this package actually uses are not module paths** —
`MFGSolver`, `mfgarchon.performance`, `mfgarchon.solvers`, `mfgarchon.solvers.<class>`,
`__name__ + ".PluginManager"` — so they fall through to the runtime arm anyway, and `find_spec`
imports every parent package to answer. (The one that *is* a module path comes from a demo
function.)

### Closing out a fix ⚠️ — name the oracle, or say there isn't one

"Add a test" is **not** the default close-out for a fix here. Most of this suite does not react when
the physics the library exists to get right is broken, and the conventions that are defended are
defended very unevenly — some by a single-digit number of tests, some by over a hundred.

**The current numbers are not written here, on purpose.** `./scripts/local_ci.sh` prints them beside
the suite result, with its own staleness flag when the suite has moved since the baseline was
recorded. A fraction copied into this file goes stale the day the mutation list or the suite moves,
and both move — the baseline was re-recorded six times in the month to 2026-08-22.

**And the fraction mostly measures the mutation list.** When that list went from six conventions to
twenty-four the fraction rose about two and a half times; holding the list at the original six and
recomputing against the current tree barely moves it at all. Almost the whole rise is the
list growing, so the aggregate is not a reading about the suite's health at all.

**Read the vector, not the fraction** — an aggregate over the whole suite cannot show a convention
held by two tests, which is the thing you would act on (#2148). Two cautions when you do:
`scripts/discrimination_baseline.json` holds the per-convention kill counts — the vector. The
distinct-test figure the gate prints is in `scripts/discrimination_killmatrix.json` and nowhere
else; summing the baseline's counts does not give it, because a test that kills several mutations is
counted in each.
Of the 65 tests whose *names* claim `single_source` / `cross_path` / `_agree`, **39 are inert** —
[#1715's comment of 2026-07-27](https://github.com/derrring/MFGArchon/issues/1715#issuecomment-5090690985),
not its body, which says the prevalence "is not established". That figure stays written here because
nothing recomputes it: the sweep's `AGREEMENT_SHAPED` regex selects a different, wider population
(308 tests, of which 193 were inert), so 39-of-65 is a dated hand measurement over a named set and
not a number that moves with the tree. **Inert is not the same as worthless**,
and that distinction has cost real time: all five tests #1715 names are genuine cross-path pins —
delegation shims, builder-vs-operator GFDM weights, Newton-vs-Picard agreement — inert on the
conventions the ratchet tracked *because those conventions are not what they pin*. The deletable set is the
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
3. `scripts/citation_baseline.json` — re-record it: `python scripts/check_citations.py --write-baseline`. Step 2 moves `changelog.d/` prose into `CHANGELOG.md`, which the citation ratchet exempts, so **both its pins fire on a correct release** — measured at the time of writing, 11 of 32 adjudicable citations and 5 of 13 drifted live in `changelog.d/`. This is mechanical and expected; the gate says so in its own message. It is listed here because a red gate on a mandated step, with no documented remedy, is how a team learns `--no-verify`.

Do **not** edit: `mfgarchon/__init__.py` (reads `importlib.metadata`), `workflow/__init__.py` (independent subpackage version), backend version reporting (external libs), historical version notes in docstrings. Sanity check: `grep -rn "^version =\|^__version__ =" pyproject.toml mfgarchon/` — only `pyproject.toml:11` should change.

### Branches & PRs
- **Branch naming (MANDATORY)**: `<type>/<short-description>` — `feature/ fix/ chore/ docs/ refactor/ test/`.
- **Never commit directly to `main`.** ⚠️ **Nothing enforces this — not the server, and not the local hook either.** Re-check with `gh api repos/derrring/MFGArchon/rules/branches/main` (the effective-rules endpoint, which covers org rules and migrated required workflows); an empty array means no branch rule applies. As of 2026-07-25 it is empty, classic protection returns 404, and the sole ruleset is `enforcement=disabled`. The pre-push hook does not close the gap: it runs `scripts/local_ci.sh`, which contains no branch logic at all (`grep -i branch scripts/local_ci.sh` → nothing) and no `no-commit-to-branch` hook is configured — it gates *test quality*, not *which ref you are on*, so a green suite on `main` pushes cleanly. Treat this as a discipline you keep, not a wall that stops you. Create the PR when you push; delete merged branches.
- **Prune local branches periodically**: `./scripts/prune_local_branches.sh` **classifies and prints evidence; it never deletes.** Deletion stays manual because each of its three signals can be wrong in a way that costs unmerged work — the content check reverse-applies against the *working tree* rather than `main`, so its verdict moves with the checkout and with uncommitted edits; the merged-PR check matches head-ref *names*, and 8 names in this repo have been the head of more than one merged PR; and a 3–4 digit run in a branch name may be a grid size rather than an issue. Read the branch, record the sha, then `git branch -D`. Left alone they accumulate: 43 locally by 2026-08-18, of which 30 were dead. The script aborts rather than guessing if the merged-PR fetch looks truncated — at 1121 merged PRs a `--limit 300` silently misclassifies. Note `git status` never reports what a worktree costs: one `??` entry at best, and nothing at all under an ignored path such as `.claude/worktrees/` (`.gitignore:77`), so use `git worktree list`.
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

### Tool pinning — exact only where a tracked baseline is keyed on the tool

**The test is decidable:** *does a tracked baseline change when this tool's version changes, and is
that dependence deliberate?* Both halves matter — the first alone would pin far too much.

| tool | pinned | where | why |
|---|---|---|---|
| ruff | `v0.16.0` | `.pre-commit-config.yaml` — **one owner**, read by `ci.yml`, bumped monthly by `check-ruff-updates.yml` | the formatted state of 941 files *is* the baseline |
| pytest | `==9.1.1` | `pyproject.toml` `[dev]` + `environment.yml` | `warning_baseline.json` keys each identity on `(origin file, class, message)` **by design** — one deprecated API called from 153 test files is 153 identities, and that count going down is the migration it tracks. pytest computes the origin attribution, so a major upgrade moves some of them: measured, 8 → 9 moved **6 of 224** |
| mypy | floor `>=1.5` | — | the criterion admits it, the second half does not: the baseline is one subpackage type-checking clean, and pinning a type checker suppresses the new checks that find real defects |
| numpy, scipy, matplotlib, jax, … | floors | `pyproject.toml`, `environment.yml` | nothing stores their output |

**A version-dependent red is usually a defect in the check, not a reason to pin.** That is the
default and it was tested here: the census was measured before pinning, and the alternatives were
rejected on numbers, not taste. Scoping it to `mfgarchon`-origin identities would delete 194 that
come from our own test files. Excluding `site-packages` would delete three real findings and one of
our own warnings whose frame is inside pytest. Re-keying without the origin file collapses 224 to 44
and destroys the call-site count that is the point. What is left is a 2.7% coupling in a check whose
value is elsewhere, and a pin makes the moment of re-recording chosen rather than an ambush on an
unrelated PR.

ruff is **not** pinned a second time in `pyproject.toml`. It already has an owner and an automated
bump; a `ruff==` in the dev extra would be a second site restating one value, which is what
`scripts/check_single_source.py` exists for and what #2135 removed. Forty-nine ruff releases in
twelve months is the reason not to pin it twice, not a reason to leave it unpinned.

`scripts/local_ci.sh` compares every `==` pin in the dev extra against what the gate interpreter has
and prints a `WARN` per mismatch, in the head and in the pasted tail. Treat those as refusals. The
population is the dev extra itself, so a second exact pin is covered without editing the gate.

**Trial policy, #2147.** The known cost: pytest moved seven times last year and
`.github/dependabot.yml` already watches pip weekly, so bumps arrive as PRs where re-recording the
affected identities is a deliberate act. The known limit: the exact pin lives in published metadata,
where the conventional home would be a lock file. `uv.lock` is that home and is currently five
months stale with `uv lock --check` exiting 1 — unresolved, and the other half of #2147.

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

⚠️ **From a `git worktree` the gate measures more than one tree and names none of them.** Measured
2026-08-27 with a blocking meta-path finder over every step and a path audit over their reads, both
controlled in each direction:

| step | reads |
|---|---|
| the suite, `PYTHONSAFEPATH=1 "$PY" -P -m pytest tests/ -n auto` | the **worktree** — pytest puts the tree root at `sys.path[0]` because `tests/__init__.py` exists, and setuptools' finder only *appends* to `sys.meta_path`, so `PathFinder` answers first. Remove that `__init__.py` and it flips |
| 6 of the 12 `scripts/*.py` invocations that never import | the **worktree**, by flag (`:476`, `:484`), by `Path(__file__).resolve().parent.parent` (`:570`), or by the gate's own `cd` |
| 4 more — the `--self-test` runs at `:471` | **neither tree**: they build a synthetic corpus in a `TemporaryDirectory`. (`:558` and `:569` not measured; both read a baseline beside the script) |
| 3 invocations across 2 scripts — `check_internal_deprecation.py --self-test`, the only importer under `--fast`, and `capability_matrix.py` twice in the full gate | the **main checkout** — `sys.path[0]` is the worktree's `scripts/`, which holds no `mfgarchon`, so the editable finder answers and its MAPPING is hard-wired to the original tree |

**No static count tells you which steps import.** `capability_census.py:78` is
`importlib.import_module(package)` — a real in-process import that no import-shaped text pattern and
no AST import-scan can see. Run the blocking finder over the actual invocation instead.

**Set both flags, for different reasons.** `PYTHONPATH=<worktree>` fixes the tree: it lands on
`sys.path`, which `PathFinder` reads before the appended editable finder. `MFG_PYTHON=<mfg_env>`
fixes the interpreter, and is not optional when a virtualenv is active — the gate's own
`CANDIDATES=(python python3 <mfg_env>)` tries PATH first, so an activated `.venv` is selected and
satisfies the probe in full, at pytest 8.4.1 against the gate's 9.1.1 and ruff 0.13.1 against the
pinned 0.16.0. That combination reports six warning identities GONE and one NEW over a two-file
documentation diff and goes `GATE RED`, one of the six being `PytestRemovedIn10Warning`, which
pytest 8 cannot emit. Do not build the worktree a fresh `uv venv`: `uv.lock` is tracked, last touched
2026-03-26, and pins exactly that toolchain — a fresh venv reproduces the wrong versions rather than
risking them.

The gate names the mismatch while it happens, in a line that reads as a nag: `WARN ruff 0.13.1 ran,
but .pre-commit-config.yaml pins 0.16.0`. Treat that WARN as a refusal.

`scripts/test_discrimination.py:452-468` already solves this for one script (#1677, "prove the
process under measurement imports what we mutate"). Port the refusal, not the code: that function is
17 lines because it checks a **subprocess** it is about to spawn, while `check_internal_deprecation.py`
and `capability_matrix.py` import in-process and need three lines — resolve `mfgarchon.__file__`,
compare against `Path(__file__).resolve().parent.parent`, refuse. `test_discrimination.py` is not
itself a gate step (`local_ci.sh` names it only at `:235` and `:567`, both comments), so the prior
art currently guards nothing the gate runs.

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
