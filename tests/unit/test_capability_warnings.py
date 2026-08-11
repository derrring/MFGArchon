r"""The capability harness must not be deaf to the library it measures (#1879).

`scripts/capability_matrix.py` carried a bare `warnings.filterwarnings("ignore")` at import.
While it measured, the library could tell it nothing -- including the warning `base_hjb`
raises 39 times in the `fdm_upwind` cell, saying in as many words that the value function
returned "is not a root of the discrete HJB, and the outer iteration will consume it as if
it were" (#1878). That cell has been PASS on every run, because the mass oracle measures a
property of the FP time-stepping that holds on whatever drift field it is handed.

The suppression was also process-wide: anything importing this module inherited it, which
is a trap for the instrumentation someone would write to investigate a cell.

Recorded, not gated. Nothing here decides a status; `--check-baseline` still compares
status only. What the field buys is that the next regeneration shows it in a diff.
"""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "capability_matrix.py"


def _load():
    spec = importlib.util.spec_from_file_location("capability_matrix_warn", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_importing_the_harness_does_not_silence_the_process():
    """The suppression was global and survived the import, so this is the load-bearing check.

    Deliberately NO `simplefilter` inside the context. The first version of this test called
    `simplefilter("always")`, which resets the filters and so overrode the very suppression it
    was meant to detect -- restoring the `filterwarnings("ignore")` line left all five tests
    green. The setup masked the defect, which is the failure mode this whole file is about.
    Here the ambient filter state, as the import left it, is what governs.
    """
    _load()

    with warnings.catch_warnings(record=True) as caught:
        warnings.warn("a library would like to say something", RuntimeWarning, stacklevel=1)

    assert len(caught) == 1, "importing the capability harness suppressed a later warning"

    # And structurally, since the behavioural check above depends on this process not having
    # been silenced by something else first: no blanket entry may sit in the filter list.
    #
    # The test is whether a filter would swallow a category this harness has to hear, not what
    # shape it has. Two earlier forms were wrong in opposite directions. `f[2] is Warning and
    # f[3] is None` missed `simplefilter("ignore")` (whose `module` is `""`, not `None`) and
    # `filterwarnings("ignore", category=UserWarning)`, which reopens this file's trap for one
    # category and left every test green. Widening it to any `Warning` subclass then flagged
    # CPython's own defaults -- Deprecation, PendingDeprecation, Import, Resource -- which are
    # always installed and are exactly the categories `_warning_summary` discards anyway.
    silences = [
        f
        for f in warnings.filters
        if f[0] == "ignore"
        and f[1] is None
        and f[3] in (None, "")
        and (issubclass(RuntimeWarning, f[2]) or issubclass(UserWarning, f[2]))
    ]
    assert not silences, f"a filter is installed that would swallow what the harness records: {silences}"


def test_a_cell_records_what_the_library_said():
    """A cell that provokes warnings must carry them into its artifact.

    Uses a stub cell rather than a real solve: this is about the harness's plumbing, and
    the real solves take minutes. The kill for the removed suppression is the test above.
    """
    cm = _load()

    def noisy():
        warnings.warn("HJB inner Newton did not converge at t_idx=3: residual 6.6e+05", RuntimeWarning, stacklevel=1)
        warnings.warn("HJB inner Newton did not converge at t_idx=4: residual 1.2e+05", RuntimeWarning, stacklevel=1)
        return "PASS", {"all_finite": True}

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cm, "CELLS", {"stub/noisy": noisy})
        out = cm.evaluate()

    said = out["stub/noisy"]["artifact"]["library_said"]
    assert sum(said.values()) == 2, said
    key = next(iter(said))
    assert key.startswith("RuntimeWarning: "), key
    assert "t_idx=N" in key, f"digits should be collapsed so two near-identical warnings fold: {key}"
    assert said[key] == 2


def test_environment_noise_is_not_recorded():
    """Import and deprecation warnings differ between machines and would fake a baseline diff.

    They are also not statements about whether the configuration solves, which is the only
    question this file asks.
    """
    cm = _load()

    def noisy():
        warnings.warn("Optimal transport solvers require POT", ImportWarning, stacklevel=1)
        warnings.warn("Legacy MFGProblem(...) is deprecated", DeprecationWarning, stacklevel=1)
        warnings.warn("something the solver said", RuntimeWarning, stacklevel=1)
        return "PASS", {"all_finite": True}

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cm, "CELLS", {"stub/mixed": noisy})
        out = cm.evaluate()

    said = out["stub/mixed"]["artifact"]["library_said"]
    assert sum(said.values()) == 1, f"only the solve-time warning should be recorded: {said}"
    assert next(iter(said)).startswith("RuntimeWarning: ")


def test_each_cell_carries_only_its_own_warnings():
    """Attribution is the whole claim, and a single-cell fixture cannot test it.

    "This cell said 39 things" is what the field asserts. Every other test here monkeypatches
    `CELLS` to ONE stub, so hoisting `catch_warnings` out of the per-cell loop -- which makes
    `caught` accumulate and credits every cell with its predecessors' output -- is invisible to
    all of them by construction. Measured: that mutation leaves all five green, and on a real
    two-cell run it credited `gfdm_rbf/construction` with two non-convergence warnings emitted
    by `fdm_centered/mass_conservation`.

    Two cells, each with a distinguishable message, and each must carry its own and nothing else.
    """
    cm = _load()

    def first():
        warnings.warn("alpha said something at t_idx=1", RuntimeWarning, stacklevel=1)
        return "PASS", {"all_finite": True}

    def second():
        warnings.warn("beta said something at t_idx=2", RuntimeWarning, stacklevel=1)
        warnings.warn("beta said something at t_idx=3", RuntimeWarning, stacklevel=1)
        return "PASS", {"all_finite": True}

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cm, "CELLS", {"stub/alpha": first, "stub/beta": second})
        out = cm.evaluate()

    alpha = out["stub/alpha"]["artifact"]["library_said"]
    beta = out["stub/beta"]["artifact"]["library_said"]

    assert sum(alpha.values()) == 1, f"alpha must carry exactly its own one warning: {alpha}"
    assert sum(beta.values()) == 2, f"beta must carry exactly its own two warnings: {beta}"
    assert all("alpha" in k for k in alpha), alpha
    assert all("beta" in k for k in beta), beta
    # The direction the hoist breaks: beta running second must not inherit alpha's.
    assert not any("alpha" in k for k in beta), f"beta was credited with alpha's warnings: {beta}"


def test_an_environment_report_is_not_recorded_even_as_a_runtime_warning():
    """The category filter is not enough: the JAX-autodiff fallback is a `RuntimeWarning`.

    It fires only where JAX is importable, and forcing `_JAX_AVAILABLE = False` leaves every
    number in the affected cells byte-identical -- so it describes the machine, not whether the
    configuration solves. Recorded, it makes the committed baseline machine-dependent and, via
    `_note_still_applies`, drops the `intended` note of every cell whose artifact moved.
    """
    cm = _load()

    def noisy():
        warnings.warn(
            "JAX autodiff failed: TracerArrayConversionError(...). Falling back to "
            "finite-difference Jacobian (O(N) complexity).",
            RuntimeWarning,
            stacklevel=1,
        )
        warnings.warn("HJB inner Newton did not converge at t_idx=3", RuntimeWarning, stacklevel=1)
        return "PASS", {"all_finite": True}

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cm, "CELLS", {"stub/jax": noisy})
        out = cm.evaluate()

    said = out["stub/jax"]["artifact"]["library_said"]
    assert sum(said.values()) == 1, f"only the solve-time warning should survive: {said}"
    assert "did not converge" in next(iter(said)), said


def test_number_collapsing_does_not_eat_hyphens_in_words():
    """`[-+0-9][0-9.eE+-]*` matched the hyphen inside ordinary words.

    "non-negativity" folded to "nonNnegativity" and "stable-baselines3" to "stableNbaselinesN",
    which silently merges warnings that differ in their text, not in their numbers.
    """
    cm = _load()

    def noisy():
        warnings.warn("mass non-negativity violated at t_idx=7", RuntimeWarning, stacklevel=1)
        return "PASS", {"all_finite": True}

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cm, "CELLS", {"stub/hyphen": noisy})
        out = cm.evaluate()

    key = next(iter(out["stub/hyphen"]["artifact"]["library_said"]))
    assert "non-negativity" in key, f"the hyphen was consumed as a sign: {key}"
    assert "t_idx=N" in key, f"the digit should still collapse: {key}"


def test_a_quiet_cell_carries_no_field_at_all():
    """Absence is currency: a cell with nothing to report must not carry an empty dict."""
    cm = _load()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cm, "CELLS", {"stub/quiet": lambda: ("PASS", {"all_finite": True})})
        out = cm.evaluate()

    assert "library_said" not in out["stub/quiet"]["artifact"]


def test_the_shipped_baseline_records_the_non_convergence_it_was_hiding():
    """The point of #1879, pinned against the artifact it produced.

    `fdm_upwind/mass_conservation` emits 39 non-convergence warnings, and the harness silenced every
    one of them until #1879. The count is what this file exists to keep visible.

    The cell is now **FAIL**, and not because the solve improved: `picard_converged` entered the
    verdict on 2026-08-11 (#1891), so a cell that does not reach a fixed point stops being PASS. It
    was PASS on the mass oracle alone, which holds on whatever drift field the FP step is handed --
    the same fact these 39 warnings state in words. An earlier version of this test asserted
    `status == "PASS"` and said "if that changed, #1878 moved"; #1878 has not moved, the verdict did,
    and the two are worth keeping apart. The pin below is written so the interesting event -- the
    warnings going away, which IS #1878 moving -- still fails it.
    """
    import json

    cells = json.loads((_SCRIPT.parent / "capability_baseline.json").read_text())["cells"]
    cell = cells["fdm_upwind/mass_conservation"]
    said = cell["artifact"]["library_said"]

    newton = {k: v for k, v in said.items() if "inner Newton did not converge" in k}
    assert newton, f"the recorded warnings no longer mention the inner Newton: {said}"
    assert sum(newton.values()) == 39, f"expected 39 non-convergence warnings (#1878), got {newton}"

    assert cell["artifact"]["picard_converged"] is False, (
        "the coupled solve now converges; that is #1878/#1873 moving and this file must be updated "
        "deliberately rather than adjusted to match"
    )
    assert cell["status"] != "PASS", (
        "the cell is PASS while recording 39 inner-Newton failures and picard_converged=False -- "
        "the verdict has stopped requiring convergence (#1891)"
    )


def test_the_baseline_comment_is_what_the_generator_writes():
    """The committed baseline must be a fixed point of its own generator.

    `_comment` is emitted from a literal in `capability_matrix.py`, so a paragraph appended to the
    JSON by hand survives exactly until the next `--write-baseline` and then vanishes. That happened:
    the paragraph explaining why the JAX-autodiff entries are excluded was added to the JSON only, and
    the next in-place regeneration deleted it -- taking with it the only place a future regenerator
    learns why `_ENVIRONMENT_MARKERS` exists, which makes that list read as arbitrary and the obvious
    thing to remove.

    Nothing else catches it. `--check-baseline` compares status only, no other test reads `_comment`,
    and the gate is green either way. Compared by AST rather than by running the generator, so this
    costs nothing and cannot be defeated by the solve.
    """
    import ast
    import json

    source = _SCRIPT.read_text()
    baseline = json.loads((_SCRIPT.parent / "capability_baseline.json").read_text())

    literals = [
        ast.literal_eval(value)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values, strict=True)
        if isinstance(key, ast.Constant) and key.value == "_comment"
    ]
    assert len(literals) == 1, f"expected exactly one `_comment` literal in the generator, found {len(literals)}"

    assert literals[0] == baseline["_comment"], (
        "the committed baseline's `_comment` is not what the generator writes, so the next "
        "`--write-baseline` will silently replace it. Put the text in the generator's literal, "
        "not in the JSON.\n"
        f"  generator: {len(literals[0])} chars\n"
        f"  committed: {len(baseline['_comment'])} chars"
    )
