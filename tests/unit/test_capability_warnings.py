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
    blanket = [f for f in warnings.filters if f[0] == "ignore" and f[1] is None and f[2] is Warning and f[3] is None]
    assert not blanket, f"a blanket ignore is installed: {blanket}"


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


def test_a_quiet_cell_carries_no_field_at_all():
    """Absence is currency: a cell with nothing to report must not carry an empty dict."""
    cm = _load()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cm, "CELLS", {"stub/quiet": lambda: ("PASS", {"all_finite": True})})
        out = cm.evaluate()

    assert "library_said" not in out["stub/quiet"]["artifact"]


def test_the_shipped_baseline_records_the_non_convergence_it_was_hiding():
    """The point of the change, pinned against the artifact it produced.

    `fdm_upwind/mass_conservation` is PASS and emits 39 non-convergence warnings. If a
    future change makes the solve produce roots the count drops and this test must be
    updated -- deliberately, since that is exactly the event #1878 tracks and it should not
    pass silently.
    """
    import json

    cells = json.loads((_SCRIPT.parent / "capability_baseline.json").read_text())["cells"]
    said = cells["fdm_upwind/mass_conservation"]["artifact"]["library_said"]

    newton = {k: v for k, v in said.items() if "inner Newton did not converge" in k}
    assert newton, f"the recorded warnings no longer mention the inner Newton: {said}"
    assert sum(newton.values()) == 39, f"expected 39 non-convergence warnings (#1878), got {newton}"
    assert cells["fdm_upwind/mass_conservation"]["status"] == "PASS", (
        "the cell is PASS while its inner solves fail; if that changed, #1878 moved"
    )
