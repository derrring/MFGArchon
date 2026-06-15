"""Removed-deprecation pins for the Tier-3b / Tier-3c legacy 1D-geometry families.

Issue #1363 finishes the geometry-first migration that #1360 (Tier-3a, removed the
``xmin/xmax/Nx/Lx/dx/xSpace/_grid`` read-properties) began. This file pins the
removal of the remaining write/construct surfaces on ``MFGProblem``:

- **Tier-3b** -- constructor kwargs ``xmin`` / ``xmax`` / ``Nx`` / ``Lx``
  (deprecated since v0.17.1) and the ``_init_1d_legacy`` / ``_init_nd`` grid
  construction methods (deprecated since v0.17.0).
- **Tier-3c** -- the ``get_u_final`` / ``get_u_fin`` aliases (deprecated since
  v0.17.6).

Note on the kwarg pins: unlike Tier-1 (which removed *named* parameters and so
fails with ``TypeError: unexpected keyword argument``), ``MFGProblem.__init__``
accepts ``**kwargs``. The removed names would otherwise be silently swallowed, so
they are registered in ``_DEPRECATED_KWARGS`` and surface as a loud
``ValueError`` pointing at the geometry-first API.
"""

from __future__ import annotations

import pytest

from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def _components() -> MFGComponents:
    H = SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0))
    return MFGComponents(hamiltonian=H, u_terminal=lambda x: 0.0, m_initial=lambda x: 1.0)


def _geometry() -> TensorProductGrid:
    return TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=no_flux_bc(dimension=1))


class TestLegacy1DConstructorKwargsRemoved:
    """``xmin`` / ``xmax`` / ``Nx`` / ``Lx`` are gone; use ``geometry=`` instead."""

    @pytest.mark.parametrize("kwarg", ["xmin", "xmax", "Nx", "Lx"])
    def test_kwarg_raises_value_error(self, kwarg):
        with pytest.raises(ValueError) as excinfo:
            MFGProblem(T=1.0, Nt=10, components=_components(), **{kwarg: 1.0 if kwarg != "Nx" else 10})
        msg = str(excinfo.value)
        assert kwarg in msg
        # The error must point at the geometry-first replacement, not the
        # Hamiltonian-kwargs guide.
        assert "GEOMETRY_FIRST_API_GUIDE" in msg
        assert "TensorProductGrid" in msg

    def test_full_legacy_signature_raises(self):
        with pytest.raises(ValueError, match="GEOMETRY_FIRST_API_GUIDE"):
            MFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=10, sigma=0.1, components=_components())

    def test_geometry_first_replacement_still_works(self):
        problem = MFGProblem(geometry=_geometry(), T=1.0, Nt=10, sigma=0.1, components=_components())
        assert problem.dimension == 1
        # Nx=10 intervals -> 11 grid points
        assert problem.geometry.num_spatial_points == 11


class TestLegacyGridConstructionMethodsRemoved:
    """``_init_1d_legacy`` / ``_init_nd`` are gone (manual grid construction)."""

    @pytest.mark.parametrize("method", ["_init_1d_legacy", "_init_nd", "_normalize_to_array"])
    def test_method_removed(self, method):
        assert not hasattr(MFGProblem, method)

    def test_replacement_helper_present(self):
        # The non-deprecated replacement that the default / spatial_bounds paths use.
        assert hasattr(MFGProblem, "_init_grid")

    def test_spatial_bounds_path_still_constructs(self):
        problem = MFGProblem(
            spatial_bounds=[(0.0, 1.0), (0.0, 1.0)],
            spatial_discretization=[20, 20],
            T=1.0,
            Nt=10,
            components=_components(),
        )
        assert problem.dimension == 2
        # spatial_discretization preserves interval semantics (Nx, not Nx + 1)
        assert problem.spatial_discretization == [20, 20]


class TestUFinalAliasesRemoved:
    """``get_u_final`` / ``get_u_fin`` are gone; only ``get_u_terminal`` remains."""

    @pytest.fixture
    def problem(self):
        return MFGProblem(geometry=_geometry(), T=1.0, Nt=10, sigma=0.1, components=_components())

    @pytest.mark.parametrize("alias", ["get_u_final", "get_u_fin"])
    def test_alias_removed(self, problem, alias):
        assert not hasattr(problem, alias)
        with pytest.raises(AttributeError):
            getattr(problem, alias)()

    def test_get_u_terminal_still_works(self, problem):
        assert problem.get_u_terminal() is not None
