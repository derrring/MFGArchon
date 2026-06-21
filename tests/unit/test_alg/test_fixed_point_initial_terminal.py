#!/usr/bin/env python3
"""Issue #1425: initial density and terminal value resolve via INDEPENDENT priority cascades.

Before the fix, ``_get_initial_and_terminal_conditions`` resolved both fields together per
priority, so a problem supplying its initial density via Priority 1 (``get_m_init()``) but its
terminal via a lower priority (``u_terminal`` attribute) silently received ``u(T,·)=0`` — the
Priority-1 terminal accessor raised and the method returned zeros before the attribute was tried.

These tests bind the resolver methods to a minimal object (no full iterator/solver construction)
and pin: the mixed-API case now resolves the terminal correctly, the standard single-API cases
are unchanged, and the genuinely-absent-terminal case warns (not silent) before defaulting to 0.
"""

import pytest

import numpy as np

from mfgarchon.alg.numerical.coupling.fixed_point_iterator import FixedPointIterator


class _Resolver:
    """Minimal carrier of the FixedPointIterator resolution methods for isolated unit testing."""

    _reshape_to = staticmethod(FixedPointIterator._reshape_to)
    _resolve_initial_density = FixedPointIterator._resolve_initial_density
    _resolve_terminal_value = FixedPointIterator._resolve_terminal_value
    _get_initial_and_terminal_conditions = FixedPointIterator._get_initial_and_terminal_conditions

    def __init__(self, problem):
        self.problem = problem


SHAPE = (5,)
M = np.ones(SHAPE)
U = np.arange(5.0)


class TestInitialTerminalResolution:
    def test_mixed_api_terminal_from_attribute_not_zeros(self):
        """get_m_init() (P1 initial) + u_terminal attribute (P2 terminal), no get_u_terminal method:
        the terminal must come from the attribute, NOT silently default to zeros (the #1425 bug)."""

        class MixedProblem:
            def get_m_init(self):
                return M.copy()

            u_terminal = U.copy()

        m_init, u_term = _Resolver(MixedProblem())._get_initial_and_terminal_conditions(SHAPE)
        np.testing.assert_array_equal(m_init, M)
        np.testing.assert_array_equal(u_term, U)  # resolved from the attribute
        assert not np.array_equal(u_term, np.zeros(SHAPE)), "regressed to the pre-#1425 silent zeros"

    def test_modern_method_api_unchanged(self):
        class ModernProblem:
            def get_m_init(self):
                return M.copy()

            def get_u_terminal(self):
                return U.copy()

        m_init, u_term = _Resolver(ModernProblem())._get_initial_and_terminal_conditions(SHAPE)
        np.testing.assert_array_equal(m_init, M)
        np.testing.assert_array_equal(u_term, U)

    def test_attribute_api_unchanged(self):
        class AttrProblem:
            m_initial = M.copy()
            u_terminal = U.copy()

        m_init, u_term = _Resolver(AttrProblem())._get_initial_and_terminal_conditions(SHAPE)
        np.testing.assert_array_equal(m_init, M)
        np.testing.assert_array_equal(u_term, U)

    def test_no_terminal_warns_then_defaults_zero(self):
        class NoTerminalProblem:
            def get_m_init(self):
                return M.copy()

        with pytest.warns(UserWarning, match="No terminal condition"):
            u_term = _Resolver(NoTerminalProblem())._resolve_terminal_value(SHAPE)
        np.testing.assert_array_equal(u_term, np.zeros(SHAPE))

    def test_missing_initial_fails_loud(self):
        class NoInitProblem:
            u_terminal = U.copy()

        with pytest.raises(ValueError, match="initial density"):
            _Resolver(NoInitProblem())._resolve_initial_density(SHAPE)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
