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
| Stability | Pre-1.0; breaking changes ship as changelog fragments | Breaking changes OK |
| Documentation | Comprehensive | Minimal |
| Testing | Admitted tests only (§ *Admission*) | Exploratory |

**Migration (research → infra)**: when research matures — add tests, write docs, ensure API consistency, open a PR in MFGArchon.

### Bug fixes from research ⚠️ CRITICAL
Before modifying MFGArchon for a bug found in mfg-research:
1. GitHub issue with quantified validation evidence — anchored so it does not expire (§ *Closing out a fix*).
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

**Top-level**: `mfgarchon/` (package), `tests/`, `benchmarks/`, `examples/` (`basic/ advanced/ applications/ tutorials/ validation/ outputs/`), `scripts/`, `changelog.d/`, `docs/`. There is no `archive/`: git history is the archive (#1710).

**Package** (`mfgarchon/`): `alg/ backends/ config/ core/ extensions/ factory/ geometry/ hooks/ meta/ operators/ types/ utils/ visualization/ workflow/`.

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
There is no factory to reach for. `create_fast_solver` / `create_research_solver` /
`create_basic_solver` / `create_accurate_solver` raise on call, and so does `create_standard_solver`
in `factory/solver_factory.py`, which `mfgarchon.factory` does not export; `create_solver` is deprecated in favour of the same three modes (#580).

No interface is fixed for v1.0 yet, so user-facing text does not call any constructor form "the v1.0 API" (user ruling, 2026-10-12). `MFGProblem(model=Model(...), domain=..., conditions=Conditions(...))` is "the model/domain/conditions form", and `MFGProblem(geometry=, components=, ...)` is "the components= form".

### Import style
```python
from mfgarchon import MFGProblem, BoundaryConditions
from mfgarchon.types import NumericalScheme
from mfgarchon.utils.mfg_logging import get_logger, configure_research_logging
```

### Mathematical typesetting & emoji
There is no local override to the host's cross-project typography rules. Project notation is `u(t,x)`, `m(t,x)`.

### Physics conventions → `docs/user/CONVENTIONS.md`
The library's conventions are stated in `docs/user/CONVENTIONS.md` and only there; this file does not
restate them. Cite a rule by its section name, never by line number. The sections a code change most
often needs: *The HJB equation, and the sign of every term in it*; *Hamiltonian and Lagrangian* (one
drift owner, and what the FP solver receives); *Volatility and diffusion* (the kinds, the per-solver
table, the one converter); *Callable signatures*; *Arrays, grids and time*. Where the code does not yet
meet a convention, the file says so and points at the tracking issue; #2429 collects them.

### The library does not repair data ⚠️

A value the computation produced, or the caller supplied, is not moved to where the code would like it to be, whatever the quantity: no clip, floor, renormalization, filled NaN or fallback of the library's own, warned or not (maintainer ruling, #1884; user ruling, 2026-11-04). A step is the method's only if its equations or a cited source define it; describing it does not qualify. Refuse it, report it with its size, or expose the bound as a parameter the caller sets. Any other change is the caller's, because it changes the equation solved. `docs/user/CONVENTIONS.md` § *The measure* states the input-side rule for rescaling a caller's density. Not repairs in this sense include:

- **The operator itself**: a projection onto an admissible set (a bounded cost's proximal map or
  optimal control, an obstacle projection, a closest-point query the caller asked for), a blend
  weight, a kernel's compact support. A position clamped onto a wall while a boundary condition is
  applied is not this case: that clamp chooses the condition (see #2485, class 10).
- **A clamp bounded by a refusal**: `clip_nonnegative_or_raise` zeroes a negative density only while
  that adds at most its `threshold` of the mass, and raises beyond it. The default threshold,
  `MAX_CLIP_MASS_FABRICATION`, is admitted (ruled 2026-10-04). A threshold set to a scheme's own tolerance is
  not admitted until ruled, and the open case is on #2485.
- **What is not data**: an array index, an optimiser's start, library-generated points, a diagnostic's denominator.

#2485 classifies the `np.clip` calls against this and lists the other shapes it keeps; the
`np.maximum(·, 0)` clamps are not yet classified.

### File-path anchoring ⚠️ CRITICAL
Anchor output paths to **project root**, never CWD: ✅ `Path(__file__).resolve().parent.parent / "results"` or `${hydra:runtime.cwd}/results`; ❌ `Path("results")` / `os.getcwd()` (recursive nesting under `cd`).

---

## 🧪 Testing — repo strategy

### Admission ⚠️ — a test earns its place before it is written (#2227)

Before a stable version, write few tests; a necessary oracle is exempt (user ruling, 2026-09-10).

**Do not write a test because a change "should have one".** Most of this suite was deleted for being
unauditable; adding to it casually is how it got there.

**A new test must be one of these, and the PR says which:**

1. **It kills a mutation.** `scripts/discrimination_killmatrix.json` maps node ID → mutations killed:
   read it. **The matrix is evidence FOR keeping a test; its complement is not a delete-list.** A
   sweep validates this ground and only this one, while a test earns its place on any of the four
   here, so the zero-killer set is full of classes 2–4 that are invisible to it. `test_discrimination_ratchet.py`
   must be subtracted from any such count: `--ignore={SELF_TESTS}` removes it from every run, because it
   fails under *every* mutation and would add +1 to every kill count. Deleting on a zero count is the
   error #1715 and #1901 already record.

   **Kill count is a property column, not a criterion**, neither sufficient nor necessary. Not
   sufficient: a test can kill a mutation incidentally while pinning nothing it claims. Not necessary,
   and this is the direction that is invisible: the mutations are a **fixed, small alphabet**
   (`MUTATIONS` declares them and `test_the_mutation_list_matches_the_parametrisation` owns the count),
   so a test pinning a convention outside it reads zero however sharp it is.

   **And every count is over `tests` under `scripts/ci_markers.txt`, a selection whose complement is
   not merely unmeasured but unmeasurable by this instrument.** The excluded set is disproportionately
   the oracles, because value correlates with runtime, runtime with `slow`, and `slow` with exclusion.

   A cost quoted over the excluded tests takes numerator and denominator from the same set. A run over
   part of the set gives no per-test figure, and does not license widening the marker filter.

   So `kill_count: 2` means two defenders *in that selection*, and the marker filter is not a knob
   that could be widened. Re-running the sweep is not the price of
   admission, so re-measure only when adding a mutation. It leaves mutations in the tree when killed
   (#1849, #2229), and it is **expensive**. Derive its current cost, which is a property of the file
   beside it:

   ```bash
   "$MFG_PYTHON" -c "import json; d = json.load(open('scripts/discrimination_killmatrix.json')); print((d['baseline_seconds'] + sum(m['seconds'] for m in d['mutations'].values())) / 60)"
   ```

   ⚠️ **Run it in a detached worktree or the kills are fake.** The script mutates the checkout in
   place (`REPO = Path(__file__).resolve().parent.parent`) and its own docstring records #1677 — a
   run that leaves the *original* module imported gives all-zero kills that read as "nothing
   discriminates". Both `.venv` and `mfg_env` editable-install `mfgarchon` to the main checkout, so
   that is the tree an import can fall back to. Setuptools *appends* `_EditableFinder` to
   `sys.meta_path`, after `PathFinder`, so under `-c` and `-m`, which both put cwd on `sys.path`, a
   `cd <worktree>` alone resolves `mfgarchon` to the worktree with no `PYTHONPATH`. **Not under script
   form**: `python scripts/foo.py` puts the *script's* directory on `sys.path[0]` and never cwd, and the
   import then falls through to the editable install. Bind `PYTHONPATH` because of that asymmetry, and
   because `_assert_import_is_the_mutated_tree` in `scripts/test_discrimination.py` then *confirms* the
   tree rather than you assuming it. The recipe:

   ```bash
   git worktree add --detach /tmp/wt-discrim <sha>
   cd /tmp/wt-discrim && PYTHONPATH=/tmp/wt-discrim "$MFG_PYTHON" scripts/test_discrimination.py \
       --json scripts/discrimination_killmatrix.json \
       --write-baseline scripts/discrimination_baseline.json
   ```

   Before recording, compare the sweep's kill lists with the latest `kill-matrix-shard-*` artifacts of
   `discrimination.yml` (kept 90 days), and remove by script any pair the runner cannot fire, since
   `$MFG_PYTHON` holds backends the runner does not install.

   Write **both** files from the one run: `--write-baseline` alone leaves the matrix at the old run,
   which the script says in its own error message.

   **The two JSONs land inside the worktree, so copy them back and tear it down** — neither is
   automatic, and a worktree `git status` cannot see:

   ```bash
   cp /tmp/wt-discrim/scripts/discrimination_killmatrix.json scripts/ \
     && cp /tmp/wt-discrim/scripts/discrimination_baseline.json scripts/ \
     && git worktree remove --force /tmp/wt-discrim && git worktree prune
   ```

   `--force` is **required**, not a convenience: the sweep leaves mutations in the tree when killed,
   so the worktree is dirty by construction and plain `git worktree remove` refuses it every time. The
   `&&` chain is what makes that safe: `--force` deletes the only copy of a long measurement, so a
   failed `cp` must not reach it. The `cp` clobbers silently, so check `git status` afterwards
   rather than assuming you were the only writer.

   And **do not let two sessions write these files**: they are one artifact pair recorded by one run,
   `git status` shows no sign of another checkout mid-sweep, and the second writer silently wins.

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

This section and § *Closing out a fix* differ in order and in membership. A happy-path assertion can
be an honest way to **close** a fix, admitted as such and saying what it would not catch; it is not a
reason to **add** a test. Writing one anyway is the moment to file the "no oracle yet" issue instead.

**"Which convention does it defend" is a question, not a lookup.** The Joplin `[Principle]
Conventions Index` and the mutations in `scripts/discrimination_baseline.json` are nearly disjoint (#2227),
and neither is the list. A test defending something on neither is admissible under class 1 or 2 on its
own merits. The index is a private Joplin note that a reader outside this machine cannot open, so state
the convention in the test rather than cite the index.

**Nothing here licenses deleting a test on a count.** The deletable set is the structurally
tautological one, found by **reading**; inert-by-counting is a different set, and #1715 records five
genuine cross-path pins inside it. Classes 2–4 are judgements: apply them by reading the candidate,
never by matching a keyword.

Cross-project testing discipline governs **what** a test must cover (edge/stress/failure cases, "coverage = paths whose failure you'd catch"). This section governs **where** a test lives — a hybrid approach for research code that evolves fast:

- **Unit tests (`tests/unit`, `tests/integration`)** — public APIs (`problem.solve()`), core infra (config/problem/result/backend), numerical correctness that must not regress. They run in the local gate on every push, and on GitHub in the smoke subset, `python-compat.yml`, `discrimination.yml`, nightly and release.
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
defended very unevenly.

**The current numbers are not written here, on purpose.** `./scripts/local_ci.sh` prints them beside
the suite result, with its own staleness flag when the suite has moved since the baseline was
recorded. A fraction copied into this file goes stale the day the mutation list or the suite moves.

**The same expiry governs a quantified claim in an issue; where the number is mandatory
(§ *Bug fixes from research*), so is the anchor.** Write it over something that cannot move — a
commit sha, a count over past commits, a published constant — or name the command that owns it.
A count of solvers that drop a source term expires on the next fix; the same sentence at a named sha
does not, and neither does a count of the tests one campaign added, which is quantified over history
rather than over the tree. That is the whole test: **a claim quantified over the current
tree drifts the moment someone fixes it; one quantified over immutable history cannot.** It also
runs backwards — read what an existing claim quantifies over and you know whether it has already
expired, without executing it.

The **title** carries this hardest, because it is what a reader takes without opening anything. When
a re-derivation moves the number, move the title with it in the same pass.

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

A rule here states what to do and why. An incident, a measurement or a figure that supports it stays in the issue or pull request that holds it; a passage here that holds one is rewritten to its rule.

### Deprecation policy ⚠️ CRITICAL
Deprecated code MUST immediately redirect to the new standard: (1) old API calls new internally (zero behavior difference); (2) **mandatory equivalence test** (old == new); (3) update ALL call sites (direct, factory, defaults, examples, tests); (4) timeline 3 minor versions OR 6 months before removal, **whichever comes FIRST** (user ruling, #2366). Ref: the last in-repo version of the policy, `cc21745c^:docs/development/DEPRECATION_LIFECYCLE_POLICY.md` — historical, and a git ref rather than a path because the document left this repository in `cc21745c`. **These four clauses are not the only statement of the removal rule in-tree, and they are not its owner.** The owner is `_removable_by_policy` in `mfgarchon/utils/deprecation.py` — it implements the timeline, and `check_removal_readiness` gates on further conditions clause (4) does not name. Every other statement restates it and must agree with it: `docs/user/LEGACY_PARAMETERS.md` for users, and `scripts/audit_deprecated_symbols.py`'s module docstring for the audit. Do not take that list as complete — it is a statement about the tree, which moves; `git grep "3 minor versions"` enumerates the live population, and #2366 reconciled the ones standing when it landed. A deprecation with no `deprecated_on` is judged by versions alone.

The input contract is `docs/user/CONVENTIONS.md`. Code that reads a user's input accepts no form the contract does not name. A defect in reading a form is fixed by refusing it there, and widening the contract is the user's ruling, written there before the code.

This repository is pre-1.0: fix a cross-cutting defect at the highest owning abstraction and remove
the redundant path instead of preserving a local fork. Breaking changes accumulate as `changelog.d/` fragments (#1521) and ship in the current `v0.MINOR.x` patch line unless the
maintainer explicitly chooses a minor release; do not add compatibility shims for unreleased code.

### Local patches are the anti-pattern ⚠️ (user ruling, 2026-10-12; #2626)

The paragraph above states the target. A **local patch** fixes a defect where it was found while other members of its class remain, which that paragraph rules out. The level is chosen when the work is planned, so the plan and the PR body of every defect fix carry a **class line**, whose class is the population of the fact its item 1 names:

1. the fact the defect is about — the quantity, meaning, layout or decision several components must agree on — named, with its tracking issue;
2. its population: every place that decides the fact and every place that reads it, found by the census in item 3, each with the source it asks and whether it agrees with the others on identical inputs. A member is a place that decides on its own, other than the owner the plan names, or a reader that disagrees with the owner; a place that reads the owner and agrees with it is not a member, and a place that decides on its own is a member even when a pin shows it agrees today. A control the census must find at both shas, outside the count; the count at the base sha and at the head sha. The table is posted on the fact's tracking issue, which the class line links, and the PR body carries its counts; the plan's author opens that issue if none exists, one issue per fact;
3. a census selected by what is called, handed or held, or by input and route, not by the found defect's own predicate. The plan's approver names it, and runs it if the author has not;
4. what a case the census did not select does at the head, per arm: reads right, raises, reds a test enumerating the class, or returns silently; one case on an axis the census held constant, with its output. If silent, the open issue;
5. class fix (head count 0: the owner is the one deciding place, and no reader disagrees with it; the others deleted or made readers of it) or partial fix; for a partial fix, which admission below holds, and an open issue this PR does not close that lists the remaining members.

A partial fix is admitted only when:
- it is one stage of a class fix whose remaining stages are named, in order, in an issue comment that records the plan's approval by someone other than its author and that the class line links;
- the class fix waits on a ruling that is not the author's to make, or on code outside this repository, and the issue that records it as open is named; or
- it is a one-line truth repair (a false message or comment), and the class's other members are listed in an open issue this PR does not close.

Urgency is not an admission: it changes the order of work, not its level. Nor is "keep changes minimal + focused" (§ *Bug fixes from research*): that bounds a change to its defect's class.

The class line is checked when the plan is approved, and again at review (§ *Review before merge*).

How soon a class is searched follows its worst known member. Wrong with no error on `problem.solve()`'s default scheme, warned or not: class fix and census now. Off it: an issue and its own plan, unless *Before filing an issue* ledgers it as low priority. A raise: *Before filing an issue*.

A census is written once per fact, at plan, by script, and re-run at the head; it holds for its sha only, and the next plan in that fact re-runs it. A review tries one case on an axis it held constant. A missed member goes to the approver. A second miss, or a second blocker in code the previous round added, returns to the plan.

### A PR stays at its plan ⚠️ (user ruling, 2026-10-25)

The plan approved for a PR fixes its scope, as its class line (§ *Local patches are the anti-pattern*) fixes its population. A finding made during the work that is neither in the plan nor a member of the class its class line names goes through § *GitHub issue/PR management*'s *Before filing an issue*, and the work continues on the plan. Its author does not widen the PR on their own judgement. A member of that class found during the work is not such a finding: § *Local patches are the anti-pattern* decides it. Nor is anything § *Review before merge* makes a blocker in the diff, including a defect the diff newly exposes: the PR fixes it.

A finding is **structural** when any of these holds:
- it falsifies the plan's premise or its acceptance check;
- it concerns a convention or an owner that several implementations share, so fixing it at one site would fork the convention;
- it is a default-path wrong result, present at the base sha, in code the PR already changes;
- it changes a denominator the gate prints for #2512 (`scripts/check_oracle_pins.py` over `scripts/oracle_pins.txt`).

The author sends a structural finding to the plan's approver in one line and keeps working the plan. The approver rules whether the PR takes it or it goes through *Before filing an issue*; anything the first paragraph has the PR fix, the PR takes. A PR that takes it records this where the plan's approval is recorded (for a defect fix, the issue comment that holds it), and the PR body and its class lines follow. From review 1 on, the finding's fix is a class fix with its own class line; for a defect present at the base sha, § *Review before merge*'s bullet on a defect that predates the PR already requires this.

### Version-bump checklist ⚠️ MANDATORY
In a single commit:
1. `pyproject.toml:11` — `version = "X.Y.Z"` + inline `# vX.Y.Z: <one-line scope>`.
2. `CHANGELOG.md` — collate the `changelog.d/` fragments into a new `## [X.Y.Z] - YYYY-MM-DD` section: `python scripts/collate_changelog.py --version X.Y.Z --date $(date +%F)`, paste it under the new heading, then `git rm changelog.d/*.md` (keep the README). Keep-a-Changelog categories. *One-time (#1521):* the pre-#1521 `## [Unreleased]` block is promoted by hand at the first release after #1521; from then on fragments own the changelog.
3. `scripts/citation_baseline.json` — re-record it: `python scripts/check_citations.py --write-baseline`. Step 2 moves `changelog.d/` prose into `CHANGELOG.md`, which the citation ratchet exempts, so **both its pins fire on a correct release**. This is mechanical and expected; the gate says so in its own message.

Do **not** edit: `mfgarchon/__init__.py` (reads `importlib.metadata`), `workflow/__init__.py` (independent subpackage version), backend version reporting (external libs), historical version notes in docstrings. Sanity check: `grep -rn "^version =\|^__version__ =" pyproject.toml mfgarchon/` — only `pyproject.toml:11` should change.

### Branches & PRs
- **Branch naming (MANDATORY)**: `<type>/<short-description>` — `feature/ fix/ chore/ docs/ refactor/ test/`.
- **Never commit directly to `main`.** The server enforces it: a ruleset on `main` requires a pull request, signed commits and the fast-tier status checks, and refuses a force push or deletion; re-check with `gh api repos/derrring/MFGArchon/rules/branches/main`. The local hook does not check the ref. Create the PR when you push; delete merged branches.
- **Prune local branches periodically**: `./scripts/prune_local_branches.sh` **classifies and prints evidence; it never deletes.** Deletion stays manual because each of its three signals can be wrong in a way that costs unmerged work — the content check reverse-applies against the *working tree* rather than `main`, so its verdict moves with the checkout and with uncommitted edits; the merged-PR check matches head-ref *names*, and a head-ref name can belong to more than one merged PR; and a 3–4 digit run in a branch name may be a grid size rather than an issue. Read the branch, record the sha, then `git branch -D`. The script aborts rather than guessing if the merged-PR fetch looks truncated. Note `git status` never reports what a worktree costs: one `??` entry at best, and nothing at all under an ignored path such as `.claude/worktrees/` (`.gitignore:77`), so use `git worktree list`.
- **PR granularity is a preference, not a mandate.** Granular (one fix / PR) is fine; batch *related, low-risk* fixes into one PR (one commit each, `Closes #A #B #C`) when the plan is made, to save CI runs. Split out anything *risky / independent / large (>~1d)* regardless.
- **Changelog per PR (#1521)**: add a `changelog.d/<slug>.<category>.md` fragment (category ∈ `added/changed/deprecated/removed/fixed`) — do **not** edit `CHANGELOG.md`. Fragments are separate files, so PRs never conflict on the changelog (batched or not). See `changelog.d/README.md`.
- **Before merge**: the **local** full suite is authoritative — `./scripts/local_ci.sh` (see *Pre-commit / pre-merge checks*). GitHub's PR checks are a fast tier only and green there is **not** sufficient.
- **Review before merge (MANDATORY)**: run an **independent adversarial review** of the PR before merging — a fresh reviewer (subagent / cross-model / worktree-isolated), *not* just author self-review. Merge only when it returns MERGE-OK, or after fixing every blocker it raises and re-reviewing. Local-green ≠ correct. A review blocks only on substance (user ruling, 2026-10-08):
  - **Blocker**: a behaviour defect in the diff; a user-facing false statement (docs, error message, changelog) that would lead a user to solve a different problem; or a claimed fix with no pin that goes red without it; or a partial fix that no admission in § *Local patches are the anti-pattern* covers, whatever its class line says. Fix it, then re-review. For a user-facing false statement, the re-review reads only the corrected text, since a corrected statement needs a second reader too. It counts as the next of the PR's three reviews.
  - **A refusal, a warning and the advice in either say what was run**: what has cleared the case, and on what. They state no universal ("any", "always", a mechanism given as the reason for every case), because one input nobody ran refutes it, and the review then rebuilds the sentence rather than the fix.
  - **Must-fix, no re-review**: a false number or claim in a test docstring, a comment or the PR body, or in user-facing text where it would not lead a user to solve a different problem; or a missing or incomplete class line, or a false number in one. Correct it, re-run the check, state the command and its output in the PR, and merge without another review round.
  - **Should-fix**: wording, clarity, style. Fix it or file it; it never blocks.
  - A review whose findings are all must-fix or should-fix returns MERGE-OK, and the must-fix corrections are a condition of merging.
  - After a MERGE-OK, its must-fix and should-fix corrections merge without a further review, whatever they touch; a test change among them needs revert-red evidence. A Blocker's correction is re-reviewed whatever it touches.
  - **A re-review is narrow** (user rulings, 2026-10-13). Review 1 reads the whole PR. Every later review reads only each finding of the review before it with the change that answers it, every file the fix round touched, the PR body's class line and gating claims at the new head, and a regression pass over the PR's own tests and the gate's result; it does not re-read what did not change. When the only blockers were user-facing false statements, it reads only the corrected text, as the Blocker bullet says. Whatever it finds is classified as above.
  - **A code change ordered between reviews owes its reach first** (user ruling, 2026-10-20). The order — the author's, a reviewer's prescription, or a ruling — states the reach: the inputs each condition the change adds, removes or alters admits, and every reader of each value it removes, moves or adds, found by following the code path. The session that pushes it does not push an order without one. Before the next review, the change is measured at the base sha and the head on one case of each input class — one combination of the branches those readers take, and of where each value a reader compares with another comes from — not only on the probe that raised the finding, and the PR body states each case beside its command and both shas. Evidence counts only for the classes it holds a case of. The statement owes no test row; a missing or incomplete one is a must-fix, as a class line is.
  - **A boundary or geometric-kernel PR is fuzzed before review 1** (user rulings, 2026-10-26 and 2026-10-27). Its author runs a seeded fuzz with an oracle written from the rules, not from the code, and at least one rival that must fire on it. Its inputs are aimed at the edge of each branch condition the diff adds, removes or alters. For a boundary, that means starts and ends, each exactly on the face, inside the tolerance, just past it or clear, with every BC kind and priority the code reads. For a kernel, it means n, d, the sample's shape, scale per axis and placement, and the bandwidth, with samples at the edge of each refusal (a point, a line, a plane, a sample thin in its own units) and against walls and corners. The PR's plan names the population against the code's branch table, and the rival it must catch. When a correction after a review adds kernel or predicate logic, that logic gets a fuzz of its own before the next review, whose population reaches the branches it adds. The fuzz stays in scratch; only the few pins it motivates are committed.
  - **A defect that predates the PR stays out of its fix rounds** (user ruling, 2026-10-20). A defect present at the base sha, found once review 1 has begun, goes through § *GitHub issue/PR management*'s *Before filing an issue*, unless its fix is a class fix with its own class line (§ *Local patches are the anti-pattern*). A defect the diff newly exposes is a behaviour defect in the diff.
  - **The PR body carries claims, not tallies** (user ruling, 2026-10-13). It states what changed, the class line where the PR fixes a defect, and the evidence for each claim that gates the merge. Every number in it sits beside the command that printed it and the sha it ran at; a number with no command is not written, except the one-line record of each review (its number, verdict and tokens), which stays in the body. Measurements no gating claim rests on go in a comment on an issue the PR addresses, which the body links. A review rebuilds the claims that gate the merge. A number no gating claim rests on is not rebuilt: it moves to the issue comment or is deleted, as a must-fix.

### GitHub issue/PR management ⚠️ MANDATORY

**What needs no per-item approval.** Filing an issue under the filing rule, a comment, a label, a fold line, a push and opening a PR need no per-item approval. A merge needs none when its last review is MERGE-OK, its gate is green and it is not a partial fix. Everything else stays with the user: a merge with a Blocker outstanding or under an admission, a fourth review, a change to `AGENTS.md`, anything sent to an external contributor, and any ruling other than a plan approver's on its plan's scope and structure.

**A quantified claim in an issue — above all in its title — must be anchored or it expires**; see § *Closing out a fix*, which owns the rule.

**Every issue carries all 4 label dimensions**: `priority:` (high/medium/low), `area:` (algorithms/config/core/documentation/geometry/performance/testing/visualization), `size:` (small=hrs–1d / medium=1–3d / large=1+wk), `type:` (bug/enhancement/chore/refactor/infrastructure/research/type-checking/question). Multiple `area:` allowed; one `priority:`/`size:` each. Family prefixes: `family:` (a defect family), `blocks:` (what an issue stands in front of). Workflow-state prefixes: `status:` (blocked/in-review/needs-testing), `resolution:` (merged/superseded/wontfix/duplicate/invalid). Bare, and no others (GitHub conventions): `good first issue`, `help wanted`, `automated`, `chore`, `documentation`.

```bash
gh issue edit N --add-label "priority: medium,area: algorithms,size: small,type: enhancement"
git checkout -b feature/descriptive-name
gh pr create --title "…" --body "Fixes #N" --label "priority: medium,area: algorithms,size: small,type: enhancement,status: in-review"
```
Feature process: issue (labelled) → branch → core code in `mfgarchon/<sub>` → examples → tests → docs → benchmarks → label PR to match.

**Before filing an issue** (user rulings, 2026-10-09, 2026-10-18 and 2026-10-19). Search issues by family label, `gh issue list --label "family: <name>" --state all --limit 1000`, and by code site or owner, `gh issue list --search "<name>" --state all --limit 1000`. Leave out any `in:` qualifier, so the search reads comments too: folds and ledger entries sit in comments. `--state all` brings in the closed issues the reopen case needs. 1000 is the search API's cap. Piped or with `--json`, a smaller limit truncates without any sign, and so does the cap, so a result as long as its limit may have been cut off. Then take the first of these that applies:
1. **Open a new issue** when the finding needs a separate decision, or when folding would lose its priority or size, even if an open issue's fix or approved staged plan covers it. A default-path wrong result "pre-empts as before" (#2573), so folding it into a lower-priority issue loses its priority.
2. **Fold it** when an open issue's fix would also fix the finding, or when an open issue carries an approved staged plan for the finding's family whose change would remove it. Being in the same family is not enough: the fix or the plan must cover it. Comment there with the reproducer, and add a checklist line (`- [ ]`) under `## Folded findings` in that issue's body, creating the heading if it is absent and leaving the rest of the body unchanged.
3. **Reopen** a closed issue with the same cause, rather than filing a new one.
4. **Ledger it**, during the #2512 phase, when it is a low-priority behaviour defect off the default path, a duplicate or a dead surface: one line in the phase ledger, #2573, with the command that shows it. A pin gap, where the code is right and only the pin could be stronger, goes under "Known pin gaps" in the body of the PR it concerns. Anything where `problem.solve()` with the default scheme can return a wrong result without an error, and any medium-or-higher behaviour defect, is never a ledger line.
5. Otherwise **open a new issue**: the finding needs a fix that no open issue makes (different code or owner).

**Before merging** (user ruling, 2026-10-18). Search open issues for the functions and files the diff changes (`gh issue list --search "<name>" --limit 1000`, again with no `in:` qualifier), and judge each hit as still live or fixed. An issue the diff fixes is closed by reference (`Fixes #N`) or, found at merge, closed with the commit named. Report each issue the merge closed beyond the PR's `Fixes` list, or that there was none.

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
manually and then pushing pays for it twice, and the second run is the one that decides. The ladder
for this repo is:

| step | command |
|---|---|
| after every edit | `ruff check --fix … && ruff format …`, or `./scripts/local_ci.sh --fast` |
| after every edit | the test files you touched, by path |
| when behaviour could have changed | the blast radius **by path**, not by `-k` name match |
| at the boundary | **`git push`** — the hook runs the gate, once |

The one case for running it by hand: you need to *read* its diagnostics (discrimination fraction,
capability baseline, fail-fast counts) rather than just pass. Then run it, and push with
`--no-verify` only if the working tree has not moved since.

Between fix rounds, push, so the hook runs the gate, and run what the fix's reach touches and the mutants for the pins the round adds. The second stack (main's `.venv`, beside `mfg_env`) and a repeated fuzz run once, when a review returns MERGE-OK; a failure there that does not reproduce at the base sha is a Blocker, and one that does goes through *Before filing an issue*. Logic a correction adds still gets its own fuzz before the next review.

⚠️ **From a `git worktree` the gate names the tree it measured, and refuses if it is the wrong
one.** `gate package : <path>` sits beside `gate interpreter`, in the head and in the pasted tail,
and `scripts/local_ci.sh` exits before any check when that path is not under the tree being gated
(#2154).

**No static count tells you which steps import.** `capability_census.py` calls
`importlib.import_module(package)`, a real in-process import that no import-shaped text pattern and
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

The gate names the mismatch while it happens, in a line that reads as a nag: `WARN ruff <version> ran,
but .pre-commit-config.yaml pins <version>`. Treat that WARN as a refusal.

⚠️ `local_ci.sh` runs `-n auto` (xdist parallel) + skip `slow` for you. If you invoke pytest by hand, match that: a bare `pytest tests/` is *serial* and includes `@slow`, which takes **hours** (not a hang — Issue #1522). A 900s per-test `timeout` (pytest-timeout) is the safety net for a genuine infinite loop. Set `MFG_PYTHON` if `python` is not the env you want.

**CI shape — the full suite runs LOCALLY, not on GitHub:**

| Tier | What | Where |
|:-----|:-----|:------|
| Gate (authoritative) | fail-fast ratchet, capability matrix, full suite (CI marker set, `-n auto`, no coverage) | **local**, `./scripts/local_ci.sh` |
| PR checks | syntax, ruff format+lint, fail-fast ratchet, imports, **smoke subset** (`test_core` + `test_backends`) | GitHub `ci.yml` |
| Backstop | full suite **incl. `@slow`**, excluding `@manual`; plus the capability matrix + its self-test | GitHub `nightly.yml`, 03:00 |
| Release | full suite incl. `@slow`, excluding `@manual`, with coverage | GitHub `ci.yml` on `release` |

**The consequence you must not forget:** a GitHub-green PR has NOT had its tests run. The gate is `scripts/local_ci.sh`, run by the pre-push hook; state its result in the PR. A regression outside `test_core`/`test_backends` will otherwise reach `main` and only surface in nightly.

---

## 📚 Documentation

### Three-tier policy ⚠️ CRITICAL

| Content | Location |
|---------|----------|
| User docs (tutorials, guides, API) | `docs/user/` (public, future book) |
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
- **Output dirs**: `examples/outputs/` and `benchmarks/results/`, both gitignored. Python scripts → `examples/outputs/[category]/`, never root.
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
1. Propose in issue → 2. implement in a feature-branch PR → 3. **independent adversarial review** (fresh reviewer, not just self-review — mandatory, see *Branches & PRs*) → 4. **verify issue completion** → 5. merge under § *Review before merge* with the gate green and all checks passing.

**Issue-completion verification** ⚠️ (before closing an issue / opening a PR): read the *original* issue (not commit messages); check every acceptance criterion; answer every discussion point; confirm all subtasks; document deviations (update the issue before closing). Anti-pattern: closing on commit messages without re-reading the requirements.
