"""Issue #2090: the test suite must never open a GUI window, because a GUI window blocks it.

`ConvergenceInfo.plot_convergence()` ends in a bare `plt.show()` and a unit test calls it. On an
interactive backend that call waits for the window to be dismissed, so the suite does not fail --
it hangs. A full `pytest tests/unit` stopped at 82% inside `tests/unit/test_types/test_state.py`
and never returned; the same test exits 124 under a 60s cap without a headless backend and passes
in 0.03s with one.

Both interpreters resolve to `macosx` by default, so this is not a property either environment
supplies -- it has to be configured, and then asserted rather than assumed.

These tests assert the PROPERTY (the resolved backend, and that `show()` returns), not the
mechanism that produces it. That distinction has a visible consequence: with only `MPLBACKEND` set
and `use(force=True)` removed, running this file alone passes -- correctly, because in isolation
nothing has imported matplotlib yet, so the variable arrives in time and the backend really is
Agg. The same configuration fails in a full-suite run, where something imports matplotlib first
and the variable arrives too late. Observed exactly that way: the gate run that carried only the
env var failed the two assertions below.

So the sensitivity of this file tracks whether the property is actually broken in the context it
runs in, which is the behaviour to want. It does not certify the configuration in the abstract.
"""

from __future__ import annotations

import os

import pytest


def test_the_configured_backend_is_headless() -> None:
    """`tests/conftest.py` sets `MPLBACKEND` before anything imports pyplot."""
    assert os.environ.get("MPLBACKEND") == "Agg", (
        f"MPLBACKEND is {os.environ.get('MPLBACKEND')!r}; tests/conftest.py must set it to 'Agg' "
        f"before pyplot is imported, or a blocking plt.show() hangs the suite (#2090)"
    )


def test_matplotlib_actually_resolves_to_a_non_interactive_backend() -> None:
    """The env var is the mechanism; this asserts the property it exists to produce.

    Checking `MPLBACKEND` alone would pass if matplotlib ignored it -- a different failure with
    the same symptom, and the one the environment variable cannot rule out by itself.
    """
    matplotlib = pytest.importorskip("matplotlib")
    backend = matplotlib.get_backend()
    # Asserted against the resolved backend name rather than a list of interactive ones:
    # `matplotlib.rcsetup.interactive_bk` no longer exists (removed by 3.11), and
    # `backends.backend_registry` is its second spelling in as many releases. The name we
    # configure is the thing we can check without tracking that churn.
    assert backend.lower() == "agg", (
        f"matplotlib resolved to {backend!r}, not Agg. `MPLBACKEND` is only read when matplotlib "
        f"is FIRST imported, so if something imports it before tests/conftest.py the variable "
        f"arrives too late -- which is why conftest also calls use(..., force=True) (#2090)"
    )


def test_a_show_call_returns_instead_of_blocking() -> None:
    """The end-to-end property: a plot can be shown and the process continues.

    This is what actually failed -- neither the variable nor the backend name, but `show()` not
    returning. Asserting the mechanism without the outcome would not have caught it.
    """
    matplotlib = pytest.importorskip("matplotlib")
    backend = matplotlib.get_backend()
    if backend.lower() != "agg":
        # Bail out rather than call show(): on an interactive backend this test would HANG, and a
        # hanging test is worse than a failing one -- it is the exact symptom under diagnosis. The
        # two tests above already fail loudly in that case, and under random ordering this one
        # cannot rely on running after them.
        pytest.fail(f"backend is {backend!r}, not Agg; refusing to call show() (#2090)")

    plt = pytest.importorskip("matplotlib.pyplot")
    fig = plt.figure()
    try:
        plt.semilogy([1e-1, 1e-3, 1e-5])
        plt.show()  # must return immediately under a headless backend
    finally:
        plt.close(fig)
