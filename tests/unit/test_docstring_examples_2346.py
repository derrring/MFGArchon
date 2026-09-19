"""The docstring examples that can run, run — and the list says how little of the package that is (#2346).

#674 removed a positional constructor and left 36 of 37 docstring examples dead, 7 raising `TypeError` on the removed
form, invisible because nothing executed them. #2341 repeated it at one line: `MFGProblem`'s Hamiltonian example taught
`QuadraticControlCost(1.0)`, which #2345 made raise, and nothing had ever run it.

Not the first executable coverage in the repo: `tests/unit/test_geometry/test_tensor_grid_docstring_examples.py`
(#1638) already runs `TensorProductGrid`'s examples, and the review of #2351 measured 48 of its collected tests going
red under a simulated `Nx` removal. That file is one class; this one is a list.

**Why this is an allowlist and not `--doctest-modules`.** Measured at 408415d4: 2439 examples across 182 modules,
**1348 of them fail**, and **1153 of those failures are `NameError`** — examples written as narrative fragments that
use names defined in a neighbouring docstring. `doctest`'s unit of isolation is the docstring, so those can never pass
under any runner, and turning execution on package-wide is a documentation rewrite rather than a check. The 24 modules
below are the ones whose examples already pass, all 178 of them.

**So this covers 24 of 182 modules with examples, and nothing else.** The number is in `test_the_allowlist_states_its_own_coverage`
so that it cannot quietly be read as "the package's examples are checked". The complement is covered differently and
statically, by `scripts/check_docstring_kwargs.py`, which needs no executability and analyses 624 of the 643
docstring blocks — the other 19 do not parse as Python.

What reddens this file: any change that breaks an example in a listed module — a renamed parameter, a moved import,
a changed repr.

What it does NOT redden on, measured rather than assumed: the removal batches. Across the at-risk families of #2343
and #2331, the listed modules' examples touch two (`Nx` in `types.pde_coefficients`, `num_points` in
`geometry.graph.maze_hybrid`), and under a simulated `Nx` removal this file stayed GREEN while
`scripts/check_docstring_kwargs.py` produced 30 findings (review of #2351). The static checker is what protects the
removal programme; this file's value is that 178 examples in 24 modules now break loudly on a repr, import or
signature change, where before nothing ran them at all.

Growing the list is deliberate: make a module's examples self-contained, confirm they pass, add it here.
"""

from __future__ import annotations

import doctest
import importlib
import warnings

import pytest

# Measured clean at 408415d4 — every example in each of these runs and passes. The trailing count is that module's
# example count at the time of listing; it is a description, not an assertion, since examples get added.
#: THE POPULATION WAS CHOSEN ON A MACHINE WITH THE OPTIONAL EXTRAS, AND CI HAS FEWER. A module
#: enters this list because its examples "already all pass" -- measured here, where jax, torch and
#: numba are installed. An example needing one passes locally and fails in CI, silently to the
#: author. That happened: `mfgarchon.backends`'s `create_backend("jax")` was green on every local
#: gate and red on the first nightly to run this list, 1 failure in 3935 (#2367).
#:
#: WHICH extras CI lacks is per-workflow and is NOT "none" -- `nightly.yml` runs
#: `pip install -e ".[numerical]" --group dev numba`, so numba IS present and jax and torch are not.
#: `test_optional_backends_are_not_imported_eagerly.py` holds the authoritative per-workflow table
#: and already struck the phrase "CI installs no extras" as false; this comment said it anyway in
#: its first draft. Testing with numba absent instead would make 20 modules unimportable and produce
#: breakage unrelated to the candidate.
#:
#: SO BEFORE ADDING A MODULE, run its examples with jax and torch absent -- faithfully. Three
#: simulations do not work, and each looks like it does:
#:   - `sys.modules["jax"] = None` breaks scipy's array-API layer. It also passes all three of the
#:     obvious acceptance checks below, which is why the fourth one is here.
#:   - a meta-path finder that RETURNS None does not block at all: returning None means "I cannot
#:     handle this, try the next finder", and the next one resolves jax.
#:   - a finder that RAISES propagates through `variational_mfg_solver.py`'s
#:     `find_spec("jax") is not None`, which expects None and gets an exception.
#: Accept a simulation only when all four hold, and the last is the one that rejects the traps:
#:   1. `importlib.util.find_spec("jax") is None`
#:   2. `import jax` raises ModuleNotFoundError
#:   3. numpy and scipy still work
#:   4. `"jax" not in sys.modules` -- true absence leaves it out entirely; the None-injection trap
#:      leaves it present-and-None, and passes checks 1-3.
#:
#: The cheapest thing that satisfies all four is a venv running the nightly's own install line:
#:   python -m venv /tmp/nojax && /tmp/nojax/bin/pip install -e ".[numerical]" --group dev numba
#: Measured that way, every module then on this list passes with jax and torch absent.
EXECUTABLE = [
    "mfgarchon.alg.numerical.coupling.graph_coupling",  # 2
    "mfgarchon.alg.numerical.gfdm_components.grid_collocation_mapper",  # 7
    "mfgarchon.backends",  # 2
    "mfgarchon.core.regime_switching",  # 4
    "mfgarchon.core.stochastic.noise_processes",  # 8
    "mfgarchon.geometry.boundary.bc_utils",  # 6
    "mfgarchon.geometry.boundary.corner.position",  # 3
    "mfgarchon.geometry.boundary.corner.velocity",  # 10
    "mfgarchon.geometry.boundary.periodic",  # 13
    "mfgarchon.geometry.graph.maze_cellular_automata",  # 3
    "mfgarchon.geometry.graph.maze_config",  # 5
    "mfgarchon.geometry.graph.maze_hybrid",  # 4
    "mfgarchon.geometry.level_set.eikonal.godunov_update",  # 4
    "mfgarchon.operators.integro_diff.graphon_coupling",  # 5
    "mfgarchon.operators.integro_diff.levy_integro_diff",  # 5
    "mfgarchon.operators.interaction.convolution",  # 6
    "mfgarchon.types.callable_protocols",  # 16
    "mfgarchon.types.pde_coefficients",  # 10
    "mfgarchon.utils.adjoint_validation",  # 15
    "mfgarchon.utils.callable_adapter",  # 3
    "mfgarchon.utils.numerical._compat.gfdm_operators",  # 11
    "mfgarchon.utils.numerical.autodiff",  # 3
    "mfgarchon.utils.numerical.monotonicity_stats",  # 14
    "mfgarchon.utils.numerical.particle.sampling",  # 19
]

_OPTIONS = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE


@pytest.mark.parametrize("module_name", EXECUTABLE)
def test_every_example_in_this_module_still_runs(module_name):
    """The pin: a listed module's examples execute and produce what they claim.

    Deprecation warnings are silenced rather than asserted -- several listed modules document a deprecated alias on
    purpose, and the warning census (`scripts/check_warnings.py`) owns whether that set changes.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        module = importlib.import_module(module_name)
        tests = [t for t in doctest.DocTestFinder().find(module) if t.examples]
        assert tests, (
            f"{module_name} carries no docstring examples; drop it from the list rather than passing vacuously"
        )
        failures = []
        for test in tests:
            runner = doctest.DocTestRunner(optionflags=_OPTIONS)
            report: list[str] = []
            runner.run(test, out=report.append, clear_globs=True)
            result = runner.summarize(verbose=False)
            if result.failed:
                failures.append(f"{test.name}: {result.failed} of {result.attempted}\n{''.join(report)}")
    assert not failures, "\n".join(failures)[:4000]


def test_the_allowlist_states_its_own_coverage():
    """A list of 24 modules must not read as a claim about the package's 182.

    This is the positive control on the file's honesty rather than on the code: it fails if the package grows modules
    with examples and the list stays put without the docstring above being updated to say so. The ratio is the thing a
    reader needs; `scripts/check_docstring_kwargs.py` is what covers the rest, statically.
    """
    import io
    import pkgutil
    from contextlib import redirect_stderr, redirect_stdout

    import mfgarchon

    with_examples = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for info in pkgutil.walk_packages(mfgarchon.__path__, prefix="mfgarchon."):
            try:
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    module = importlib.import_module(info.name)
            except BaseException:  # an unimportable module is not this test's subject
                continue
            if any(t.examples for t in doctest.DocTestFinder().find(module)):
                with_examples += 1

    # `len(EXECUTABLE) <= with_examples` was the first version of this and is vacuous -- 24 <= 182 holds
    # however few of the listed modules carry examples, and the property it names is actually tested above by
    # `assert tests` per module (review of #2351). The ratio is what this test is for, and it is asserted
    # EXACTLY: a docstring that says "24 of 182" stops being true at 183, so 183 is where this fires.
    assert with_examples == 182, (
        f"modules carrying examples moved {182} -> {with_examples}. This file's docstring states the ratio "
        f"24 of 182 and `scripts/check_docstring_kwargs.py`'s docstring states 624 analysed blocks; both are "
        f"now wrong. Update them and this number together, or the coverage claim overstates itself silently."
    )
