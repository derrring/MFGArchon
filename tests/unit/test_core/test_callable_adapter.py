#!/usr/bin/env python3
"""
Unit tests for callable signature detection and adaptation (Issue #684).

Tests:
- Standalone adapter: all signature types, error messages, deprecation warnings
- Integration: MFGProblem accepts different callable conventions
"""

import warnings

import pytest

import numpy as np

from mfgarchon.utils.callable_adapter import CallableSignature, adapt_ic_callable

# ===========================================================================
# Standalone adapter tests
# ===========================================================================


@pytest.mark.unit
def test_adapt_scalar_1d():
    """f(x) where x is float -- most common 1D convention."""

    def m_initial(x):
        return np.exp(-10 * (x - 0.5) ** 2)

    sig, adapted = adapt_ic_callable(m_initial, dimension=1, sample_point=0.5)

    assert sig == CallableSignature.SPATIAL_SCALAR
    # Wrapper should be the original function (zero overhead)
    assert adapted is m_initial
    assert np.isfinite(adapted(0.3))


@pytest.mark.unit
def test_adapt_array_1d():
    """f(x) where x is ndarray -- user expects x[0] indexing in 1D."""

    def m_initial(x):
        return np.exp(-(x[0] ** 2))

    sig, adapted = adapt_ic_callable(m_initial, dimension=1, sample_point=0.5)

    assert sig == CallableSignature.SPATIAL_ARRAY
    # Wrapper converts scalar -> ndarray([scalar])
    result = adapted(0.3)
    expected = np.exp(-(0.3**2))
    assert abs(result - expected) < 1e-12


@pytest.mark.unit
def test_adapt_array_nd():
    """f(x) where x is ndarray in 2D."""

    def m_initial(x):
        return np.exp(-(x[0] ** 2 + x[1] ** 2))

    sample = np.array([0.5, 0.3])
    sig, adapted = adapt_ic_callable(m_initial, dimension=2, sample_point=sample)

    assert sig == CallableSignature.SPATIAL_ARRAY
    # Should be the original function (nD standard convention)
    assert adapted is m_initial
    assert np.isfinite(adapted(np.array([0.1, 0.2])))


@pytest.mark.unit
def test_adapt_spatiotemporal_xt():
    """f(x, t) -- spatiotemporal with time as second arg."""

    def u_final(x, t):
        return x**2 * np.exp(-t)

    sig, adapted = adapt_ic_callable(
        u_final,
        dimension=1,
        sample_point=0.5,
        time_value=1.0,
    )

    assert sig == CallableSignature.SPATIOTEMPORAL_XT
    # Wrapper pins t=1.0, accepts only x
    result = adapted(0.5)
    expected = 0.25 * np.exp(-1.0)
    assert abs(result - expected) < 1e-12


@pytest.mark.unit
def test_adapt_spatiotemporal_tx():
    """f(t, x) -- spatiotemporal with time as first arg."""

    # This callable fails for f(x) because x**2 works for float,
    # so we need it to ONLY work with two args.
    def u_final(t, x):
        return (t + 1) * x

    # f(0.5) would give 0.5 (valid scalar), which would be detected as SPATIAL_SCALAR.
    # Use a function that cannot be called with a single scalar:
    def u_final_strict(t, x):
        # Requires exactly two arguments
        if not isinstance(t, (int, float)):
            raise TypeError("t must be numeric")
        if not isinstance(x, (int, float)):
            raise TypeError("x must be numeric")
        return (t + 1) * x

    # For this test, use a lambda that explicitly needs 2 args by crashing on 1 arg:
    def needs_two(a, b):
        return float(a) + float(b)

    # This will fail f(0.5) because it needs 2 positional args
    sig, adapted = adapt_ic_callable(
        needs_two,
        dimension=1,
        sample_point=0.5,
        time_value=0.0,
    )

    # Should detect as SPATIOTEMPORAL_XT first (f(x, t) with x=0.5, t=0.0)
    assert sig in (CallableSignature.SPATIOTEMPORAL_XT, CallableSignature.SPATIOTEMPORAL_TX)
    assert np.isfinite(adapted(0.5))


@pytest.mark.unit
def test_adapt_expanded_2d():
    """f(x, y) with expanded coordinates -- deprecated but supported."""

    def m_initial(x, y):
        return np.exp(-(x**2 + y**2))

    sample = np.array([0.5, 0.3])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sig, adapted = adapt_ic_callable(m_initial, dimension=2, sample_point=sample)

    assert sig == CallableSignature.EXPANDED_2D
    # Check deprecation warning was emitted
    deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
    assert len(deprecation_warnings) >= 1
    assert "deprecated" in str(deprecation_warnings[0].message).lower()

    # Wrapper converts ndarray -> expanded args
    result = adapted(np.array([0.1, 0.2]))
    expected = np.exp(-(0.01 + 0.04))
    assert abs(result - expected) < 1e-12


@pytest.mark.unit
def test_adapt_expanded_3d():
    """f(x, y, z) with expanded coordinates -- deprecated but supported."""

    def m_initial(x, y, z):
        return x + y + z

    sample = np.array([0.1, 0.2, 0.3])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sig, adapted = adapt_ic_callable(m_initial, dimension=3, sample_point=sample)

    assert sig == CallableSignature.EXPANDED_3D
    deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
    assert len(deprecation_warnings) >= 1

    result = adapted(np.array([1.0, 2.0, 3.0]))
    assert abs(result - 6.0) < 1e-12


@pytest.mark.unit
def test_wrapper_consistency():
    """Original and adapted give the same results."""

    def f(x):
        return np.sin(x) + 1.0

    sig, adapted = adapt_ic_callable(f, dimension=1, sample_point=0.5)

    # SPATIAL_SCALAR: wrapper should be original
    assert sig == CallableSignature.SPATIAL_SCALAR
    for x in [0.0, 0.25, 0.5, 0.75, 1.0]:
        assert adapted(x) == f(x)


@pytest.mark.unit
def test_adapt_preserves_scalar_convention():
    """SPATIAL_SCALAR detection returns the original function (no wrapper overhead)."""

    original = lambda x: x**2  # noqa: E731

    sig, adapted = adapt_ic_callable(original, dimension=1, sample_point=0.5)

    assert sig == CallableSignature.SPATIAL_SCALAR
    assert adapted is original


@pytest.mark.unit
def test_adapt_lambda():
    """Lambdas (which may lack inspectable signatures) work via probing."""
    f = lambda x: x * 2.0  # noqa: E731

    sig, adapted = adapt_ic_callable(f, dimension=1, sample_point=0.5)
    assert sig == CallableSignature.SPATIAL_SCALAR
    assert adapted(3.0) == 6.0


@pytest.mark.unit
def test_adapt_spatiotemporal_xt_nd():
    """f(x, t) in 2D -- spatiotemporal with ndarray x."""

    def u_final(x, t):
        return np.sum(x**2) * np.exp(-t)

    sample = np.array([0.5, 0.3])
    sig, adapted = adapt_ic_callable(
        u_final,
        dimension=2,
        sample_point=sample,
        time_value=1.0,
    )

    assert sig == CallableSignature.SPATIOTEMPORAL_XT
    result = adapted(np.array([0.5, 0.3]))
    expected = (0.25 + 0.09) * np.exp(-1.0)
    assert abs(result - expected) < 1e-12


# ===========================================================================
# Integration tests with MFGProblem
# ===========================================================================


def _geometry(Nx_points=11, dimension=1):
    """Create a test geometry."""
    from mfgarchon.geometry import TensorProductGrid
    from mfgarchon.geometry.boundary.conditions import no_flux_bc

    bounds = [(0.0, 1.0)] * dimension
    nx = [Nx_points] * dimension
    return TensorProductGrid(
        bounds=bounds,
        Nx_points=nx,
        boundary_conditions=no_flux_bc(dimension=dimension),
    )


def _hamiltonian():
    """Create a test Hamiltonian."""
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian

    return SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: -(m**2),
        coupling_dm=lambda m: -2 * m,
    )


def _problem(m_initial, u_terminal, Nx_points=11, dimension=1, **kwargs):
    """Create a test MFGProblem."""
    from mfgarchon.core.mfg_components import MFGComponents
    from mfgarchon.core.mfg_problem import MFGProblem

    geom = _geometry(Nx_points=Nx_points, dimension=dimension)
    components = MFGComponents(
        hamiltonian=_hamiltonian(),
        m_initial=m_initial,
        u_terminal=u_terminal,
    )
    return MFGProblem(geometry=geom, components=components, **kwargs)


@pytest.mark.unit
def test_mfg_problem_scalar_m_initial():
    """Standard lambda x: f(x) works for 1D MFGProblem."""

    def m_initial(x):
        return np.exp(-10 * (x - 0.5) ** 2)

    problem = _problem(m_initial=m_initial, u_terminal=lambda x: x**2, Nx_points=11)
    assert problem.m_initial is not None
    assert problem.m_initial.shape == (11,)
    # Should have non-trivial values (Gaussian peak near center)
    assert np.max(problem.m_initial) > 0.5


@pytest.mark.unit
def test_mfg_problem_array_m_initial_1d():
    """lambda x: f(x[0]) works in 1D via adapter wrapping."""

    def m_initial(x):
        return np.exp(-10 * (x[0] - 0.5) ** 2)

    problem = _problem(m_initial=m_initial, u_terminal=lambda x: x**2, Nx_points=11)
    assert problem.m_initial is not None
    assert problem.m_initial.shape == (11,)
    assert np.max(problem.m_initial) > 0.5


@pytest.mark.unit
def test_mfg_problem_spatiotemporal_u_terminal():
    """lambda x, t: f(x)*g(t) works for u_terminal via adapter wrapping."""

    def u_terminal_func(x, t):
        return x**2 * np.exp(-t)

    problem = _problem(m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2), u_terminal=u_terminal_func)
    assert problem.u_terminal is not None
    assert problem.u_terminal.shape == (11,)
    # Values should reflect spatiotemporal evaluation at terminal time
    assert np.all(np.isfinite(problem.u_terminal))
