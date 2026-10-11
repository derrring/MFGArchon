"""`deprecated_value` takes a type key for a deprecated argument form, and calls its function once (#2597).

#2597 deprecates HJB-GFDM's dict form of ``boundary_conditions``. A dict is unhashable, so the value lookup raised
inside the wrapper's ``try`` and the decorator passed it through without a warning. A type key now matches by
``isinstance``, warns, and passes the value through for the function body to translate; a hashable key still
remaps. The ``try`` also wrapped the call itself, so a TypeError raised by the decorated function ran it a second
time with the unremapped arguments; only the bind and the lookup are guarded now.
"""

from __future__ import annotations

import warnings

import pytest

from mfgarchon.utils.deprecation import deprecated_value


def test_a_type_key_warns_and_passes_through_a_value_key_remaps_and_the_function_runs_once():
    received, calls = [], []

    @deprecated_value(
        "spec", {dict: "Spec", "old": "new", "old_boom": "boom"}, since="v0.0.0", deprecated_on="2026-10-11"
    )
    def build(spec=None):
        calls.append(spec)
        received.append(spec)
        if spec == "boom":
            raise TypeError("raised by the function body")
        return spec

    form = {"type": "dirichlet"}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert build(form) is form  # passed through, for the body to translate
        assert build("old") == "new"  # remapped, as before
        assert build("current") == "current"
    messages = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
    assert messages == [
        "A dict for parameter 'spec' in "
        "'test_a_type_key_warns_and_passes_through_a_value_key_remaps_and_the_function_runs_once.<locals>.build' "
        "is deprecated since v0.0.0. Use a Spec instead. Will be removed in v0.25.0.",
        "Value 'old' for parameter 'spec' in "
        "'test_a_type_key_warns_and_passes_through_a_value_key_remaps_and_the_function_runs_once.<locals>.build' "
        "is deprecated since v0.0.0. Use 'new' instead. Will be removed in v0.25.0.",
    ]
    assert build._deprecated_values[0]["deprecated_on"] == "2026-10-11"

    calls.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(TypeError, match="raised by the function body"):
            build("old_boom")  # remapped to "boom", which raises in the body
    assert calls == ["boom"]  # once: the TypeError is not swallowed into a second call with "old_boom"


def test_a_warning_raised_while_hjb_gfdm_is_constructed_names_the_constructing_line():
    """`deprecated_value` wraps `HJBGFDMSolver.__init__`, a frame of its own, so a warning in `__init__` or in what it
    calls directly, at a stacklevel that named the caller, named `deprecation.py` instead. Those skip `WRAPPER_FRAMES`
    (#2579). Recorded by ``warning.filename``, which `pytest.warns` does not check: the dict's own deprecation, the
    no-`monotonicity_scheme` warning in `__init__`, and the neighbourhood builder's (review 1 of #2603)."""
    import numpy as np

    from mfgarchon.alg.numerical.hjb_solvers import HJBGFDMSolver
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
    from mfgarchon.core.mfg_components import MFGComponents
    from mfgarchon.core.mfg_problem import MFGProblem
    from mfgarchon.geometry import TensorProductGrid
    from mfgarchon.geometry.boundary import dirichlet_bc

    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=dirichlet_bc(dimension=1))
    comps = MFGComponents(
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
        m_initial=lambda x: np.ones_like(np.asarray(x, dtype=float)),
        u_terminal=lambda x: np.zeros_like(np.asarray(x, dtype=float)),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        problem = MFGProblem(geometry=grid, components=comps, T=1.0, Nt=4, volatility=0.5)
    x = np.linspace(0.0, 1.0, 11).reshape(-1, 1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        HJBGFDMSolver(
            problem,
            collocation_points=x,
            adaptive_neighborhoods=True,
            k_min=9,
            boundary_conditions={"type": "dirichlet"},
        )
    kinds = {"A dict for parameter": [], "no `monotonicity_scheme`": [], "Could not find 9 neighbors": []}
    for w in caught:
        for marker, found in kinds.items():
            if marker in str(w.message):
                found.append(w.filename)
    assert all(kinds.values()), {marker: len(found) for marker, found in kinds.items()}
    assert {marker: set(found) for marker, found in kinds.items()} == {marker: {__file__} for marker in kinds}
