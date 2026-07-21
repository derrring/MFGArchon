"""Campaign-3 geometry hygiene (findings #18, #32) — two TensorProductGrid infrastructure fixes.

#18 (silent wrong-BC): `get_boundary_handler`'s docstring shows `grid.boundary_conditions = bc`, but
the class read only `_boundary_conditions`, so a bare assignment created a DEAD instance attribute the
handlers ignored -- the grid silently kept its old BC. `boundary_conditions` is now a property whose
setter routes through `set_boundary_conditions` (which binds the dimension), so the documented usage
actually takes effect.

#32 (spurious deprecation): `refine`/`coarsen` passed the deprecated `dimension=` kwarg to their own
constructor, so every legitimate call emitted a `DeprecationWarning` blaming the caller. Dimension is
inferred from `len(bounds)` (#676); the kwarg is gone.
"""

from __future__ import annotations

import warnings

from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import dirichlet_bc, no_flux_bc


def _grid(dim=1):
    bounds = [(0.0, 1.0)] * dim
    return TensorProductGrid(bounds=bounds, Nx_points=[11] * dim, boundary_conditions=no_flux_bc(dimension=dim))


def test_setting_boundary_conditions_takes_effect():
    """#18: `grid.boundary_conditions = bc` must actually update the read path. Discriminating --
    without the property setter the assignment is silently ignored and the grid keeps NO_FLUX."""
    grid = _grid()
    assert str(grid.get_boundary_conditions().segments[0].bc_type).endswith("NO_FLUX")
    grid.boundary_conditions = dirichlet_bc(dimension=1)
    assert str(grid.get_boundary_conditions().segments[0].bc_type).endswith("DIRICHLET")
    # and the boundary handler (the solver-facing read path) reflects it too
    assert grid.get_boundary_handler() is grid._boundary_conditions


def test_boundary_conditions_getter_reads_private():
    """The property getter exposes the same object the handlers read (`_boundary_conditions`)."""
    grid = _grid()
    assert grid.boundary_conditions is grid._boundary_conditions


def test_refine_and_coarsen_emit_no_dimension_deprecation():
    """#32: refine/coarsen must not emit the deprecated-`dimension=` warning (they infer it from
    bounds). Discriminating -- restoring `dimension=self._dimension` in either ctor call re-warns."""
    grid = _grid(dim=2)
    for op in ("refine", "coarsen"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = getattr(grid, op)(2)
        dim_warns = [w for w in caught if issubclass(w.category, DeprecationWarning) and "dimension" in str(w.message)]
        assert not dim_warns, f"{op} emitted a spurious dimension deprecation warning"
        assert result.dimension == 2  # dimension still correctly inferred from bounds
