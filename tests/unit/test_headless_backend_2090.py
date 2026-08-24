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


def _interactive_backends() -> frozenset[str]:
    """Ask matplotlib which backends can open a window, rather than hardcoding a list.

    A hardcoded set was wrong in both directions: it admitted `cairo`, which resolves fine but
    raises `ImportError` at `plt.figure()` because pycairo is not installed, and it rejected
    `pgf`, which works and whose `show()` returns. `backend_registry` is present in every
    matplotlib this repository supports; if a future version moves it, this import fails loudly
    instead of silently certifying a stale list -- which is what `rcsetup.interactive_bk`
    (removed in 3.11) did to an earlier revision of this file.
    """
    from matplotlib.backends import BackendFilter, backend_registry

    return frozenset(b.lower() for b in backend_registry.list_builtin(BackendFilter.INTERACTIVE))


def test_the_configured_backend_is_headless() -> None:
    """`tests/conftest.py` sets `MPLBACKEND` before anything imports pyplot."""
    configured = os.environ.get("MPLBACKEND")
    assert configured is not None, "MPLBACKEND is unset; tests/conftest.py must set it (#2090)"
    assert configured.lower() not in _interactive_backends(), (
        f"MPLBACKEND is {configured!r}, which can open a window. conftest must select a "
        f"non-interactive backend before pyplot is imported, or a blocking plt.show() hangs "
        f"the suite (#2090)"
    )


def test_matplotlib_actually_resolves_to_a_non_interactive_backend() -> None:
    """The env var is the mechanism; this asserts the property it exists to produce.

    Checking `MPLBACKEND` alone would pass if matplotlib ignored it -- a different failure with
    the same symptom, and the one the environment variable cannot rule out by itself.
    """
    import matplotlib

    backend = matplotlib.get_backend()
    # Asserted against the resolved backend name rather than a list of interactive ones:
    # `matplotlib.rcsetup.interactive_bk` no longer exists (removed by 3.11), and
    # `backends.backend_registry` is its second spelling in as many releases. The name we
    # configure is the thing we can check without tracking that churn.
    assert backend.lower() not in _interactive_backends(), (
        f"matplotlib resolved to {backend!r}, which can open a window. `MPLBACKEND` is only read "
        f"when matplotlib is FIRST imported, so if something imports it before tests/conftest.py "
        f"the variable arrives too late -- which is why conftest also calls use(force=True) (#2090)"
    )


def test_a_show_call_returns_instead_of_blocking() -> None:
    """The end-to-end property: a plot can be shown and the process continues.

    This is what actually failed -- neither the variable nor the backend name, but `show()` not
    returning. Asserting the mechanism without the outcome would not have caught it.
    """
    import matplotlib

    backend = matplotlib.get_backend()
    if backend.lower() in _interactive_backends():
        # Bail out rather than call show(): on an interactive backend this test would HANG, and a
        # hanging test is worse than a failing one -- it is the exact symptom under diagnosis. The
        # two tests above already fail loudly in that case, and under random ordering this one
        # cannot rely on running after them.
        pytest.fail(f"backend {backend!r} can open a window; refusing to call show() (#2090)")

    import matplotlib.pyplot as plt

    # A backend can be non-interactive (will not block) and still unusable (cannot draw):
    # `cairo` resolves and passes the two checks above, then raises ImportError here because
    # pycairo is not installed. Distinguish it, or the failure reads as a blocking backend.
    try:
        fig = plt.figure()
    except ImportError as exc:
        pytest.fail(
            f"backend {backend!r} will not block but cannot draw: {exc}. A non-interactive "
            f"backend is not enough -- the suite has to be able to create figures (#2090)"
        )

    try:
        plt.semilogy([1e-1, 1e-3, 1e-5])
        plt.show()  # must return immediately under a headless backend
    finally:
        plt.close(fig)
