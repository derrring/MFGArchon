"""Equivalence tests for B1.5b.3: NetworkMFGSolver + MultiPopulationIterator rename.

Legacy `damping_factor` / `damping` kwargs redirect to canonical `relaxation`
with `@deprecated_parameter` decorators. MultiPopulationIterator also gets a
silent `@property` alias for `iter.damping_factor` attribute access.

Removal of legacy kwargs/attrs scheduled for v0.25.0.
"""

from __future__ import annotations

import warnings

from mfgarchon.alg.numerical.coupling.multi_population_iterator import MultiPopulationIterator

# ------------------------------------------------------------------
# MultiPopulationIterator
# ------------------------------------------------------------------


class _StubMultiProblem:
    """Minimal stub for MultiPopulationProblem (ctor dimension checks only)."""

    def __init__(self, K=2):
        self.K = K
        self.population_names = [f"pop_{i}" for i in range(K)]


class _StubSolver:
    pass


class TestMultiPopulationIteratorAlias:
    def _make(self, **kwargs):
        problem = _StubMultiProblem(K=2)
        hjbs = [_StubSolver(), _StubSolver()]
        fps = [_StubSolver(), _StubSolver()]
        return MultiPopulationIterator(problem, hjbs, fps, **kwargs)

    def test_canonical_kwarg_works(self):
        inst = self._make(relaxation=0.7)
        assert inst.relaxation == 0.7

    def test_legacy_kwarg_redirects_with_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            inst = self._make(damping_factor=0.7)
            dep = [x for x in w if issubclass(x.category, DeprecationWarning) and "damping_factor" in str(x.message)]
        assert len(dep) == 1
        assert inst.relaxation == 0.7

    def test_legacy_and_canonical_produce_equivalent_state(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            legacy = self._make(damping_factor=0.6)
        canonical = self._make(relaxation=0.6)
        assert legacy.relaxation == canonical.relaxation == 0.6

    def test_damping_factor_property_silent_alias(self):
        inst = self._make(relaxation=0.3)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assert inst.damping_factor == 0.3
            dep = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep) == 0


# ------------------------------------------------------------------
# create_network_mfg_solver / create_simple_network_solver
#
# These are factory functions. We can't instantiate the full solver
# without a real NetworkMFGProblem, so we verify decorator + signature
# behavior directly — the @deprecated_parameter warns on legacy kwargs.
# ------------------------------------------------------------------


class TestNetworkFactorySignatures:
    def test_deprecation_metadata_present_on_factory_functions(self):
        from mfgarchon.alg.numerical.coupling.network_mfg_solver import (
            create_network_mfg_solver,
            create_simple_network_solver,
        )

        # @deprecated_parameter stores metadata on the decorated function
        meta1 = getattr(create_network_mfg_solver, "_deprecated_parameters", None)
        meta2 = getattr(create_simple_network_solver, "_deprecated_parameters", None)
        assert meta1 is not None
        assert any(p["param"] == "damping_factor" for p in meta1)
        assert meta2 is not None
        assert any(p["param"] == "damping" for p in meta2)
