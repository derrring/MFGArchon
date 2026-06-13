"""
Regression tests for Issue #1100: remove the silent PERIODIC default BC.

``BoundaryConditions.default_bc`` no longer defaults to ``BCType.PERIODIC``.
It is ``None`` ("unspecified") unless set explicitly. The contract:

(a) Resolving a point/segment that matches no explicit BC segment, when
    ``default_bc`` is unset, fails loud with a clear ValueError (no silent
    PERIODIC wrapping).
(b) Setting ``default_bc=BCType.PERIODIC`` explicitly still resolves
    fall-through points to PERIODIC, exactly as the old implicit default did
    (equivalence — the explicit path is unchanged).
(c) Global-property reads (``__str__``/``__repr__``, ``validate()``,
    ``validate_boundary_conditions`` has_periodic) on a ``default_bc=None``
    object do not crash; None is treated as "not specified / no periodic
    default".
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.geometry import BCSegment, BCType, BoundaryConditions


def _bounds_2d() -> np.ndarray:
    return np.array([[0.0, 1.0], [0.0, 1.0]])


# =============================================================================
# (a) Unspecified default + unmatched point -> fail loud
# =============================================================================


class TestUnspecifiedDefaultFailsLoud:
    def test_default_bc_is_none_when_unset(self):
        """The dataclass field defaults to None, not PERIODIC (Issue #1100)."""
        bc = BoundaryConditions(segments=[], dimension=2, domain_bounds=_bounds_2d())
        assert bc.default_bc is None

    def test_get_bc_type_at_boundary_unmatched_raises(self):
        """A coverage gap (segment only on x_min) must raise on the unmatched face."""
        bc = BoundaryConditions(
            segments=[BCSegment(name="wall", bc_type=BCType.NO_FLUX, boundary="x_min")],
            dimension=2,
            domain_bounds=_bounds_2d(),
        )
        with pytest.raises(ValueError, match=r"default_bc was not specified.*Issue #1100"):
            bc.get_bc_type_at_boundary("x_max")

    def test_get_bc_at_point_unmatched_raises(self):
        """A point matching no segment falls through to default -> fail loud."""
        bc = BoundaryConditions(
            segments=[BCSegment(name="wall", bc_type=BCType.NO_FLUX, boundary="x_min")],
            dimension=2,
            domain_bounds=_bounds_2d(),
        )
        # Point on x_max boundary: matches no segment.
        with pytest.raises(ValueError, match=r"matched no BC and default_bc was not specified"):
            bc.get_bc_at_point(np.array([1.0, 0.5]), boundary_id="x_max")

    def test_error_message_lists_explicit_choices(self):
        """The diagnostic must name the explicit fix (greppable)."""
        bc = BoundaryConditions(segments=[], dimension=1, domain_bounds=np.array([[0.0, 1.0]]))
        with pytest.raises(ValueError) as exc_info:
            bc.get_bc_type_at_boundary("x_min")
        msg = str(exc_info.value)
        assert "BCType.NO_FLUX" in msg
        assert "PERIODIC" in msg
        assert "DIRICHLET" in msg
        assert "Issue #1100" in msg


# =============================================================================
# (b) Explicit PERIODIC unchanged (equivalence with old implicit behavior)
# =============================================================================


class TestExplicitPeriodicUnchanged:
    def test_explicit_periodic_default_resolves_to_periodic(self):
        """default_bc=PERIODIC resolves unmatched faces to PERIODIC (no raise)."""
        bc = BoundaryConditions(
            segments=[BCSegment(name="wall", bc_type=BCType.NO_FLUX, boundary="x_min")],
            dimension=2,
            domain_bounds=_bounds_2d(),
            default_bc=BCType.PERIODIC,
        )
        # Matched face keeps its segment type...
        assert bc.get_bc_type_at_boundary("x_min") == BCType.NO_FLUX
        # ...unmatched face falls through to the explicit PERIODIC default.
        assert bc.get_bc_type_at_boundary("x_max") == BCType.PERIODIC

    def test_explicit_periodic_default_bc_at_point(self):
        """get_bc_at_point returns a PERIODIC default segment for unmatched points."""
        bc = BoundaryConditions(
            segments=[BCSegment(name="wall", bc_type=BCType.NO_FLUX, boundary="x_min")],
            dimension=2,
            domain_bounds=_bounds_2d(),
            default_bc=BCType.PERIODIC,
        )
        seg = bc.get_bc_at_point(np.array([1.0, 0.5]), boundary_id="x_max")
        assert seg.bc_type == BCType.PERIODIC

    def test_periodic_bc_factory_sets_explicit_default(self):
        """periodic_bc() is the canonical explicit-periodic path and is unaffected."""
        from mfgarchon.geometry.boundary import periodic_bc

        bc = periodic_bc(dimension=1)
        assert bc.default_bc == BCType.PERIODIC
        # Uniform periodic: resolution never raises.
        assert bc.get_bc_type_at_boundary("x_min") == BCType.PERIODIC


# =============================================================================
# (c) Global-property reads on default_bc=None must not crash
# =============================================================================


class TestGlobalPropertyReadsNoneSafe:
    def test_str_does_not_crash_on_none_default(self):
        """__str__ guards default_bc.value (mixed BC path)."""
        bc = BoundaryConditions(
            segments=[
                BCSegment(name="a", bc_type=BCType.NO_FLUX, boundary="x_min"),
                BCSegment(name="b", bc_type=BCType.DIRICHLET, boundary="x_max"),
            ],
            dimension=2,
            domain_bounds=_bounds_2d(),
        )
        text = str(bc)
        assert "unspecified" in text  # not a crash, default reported as unspecified

    def test_validate_does_not_crash_on_none_default(self):
        """validate() guards default_bc.value in coverage warnings."""
        bc = BoundaryConditions(
            segments=[BCSegment(name="wall", bc_type=BCType.NO_FLUX, boundary="x_min")],
            dimension=2,
            domain_bounds=_bounds_2d(),
        )
        is_valid, warnings = bc.validate()
        # x_max/y_min/y_max uncovered -> warnings emitted, no exception.
        assert isinstance(is_valid, bool)
        assert any("will raise" in w for w in warnings)

    def test_has_periodic_check_none_safe(self):
        """validate_boundary_conditions has_periodic read treats None as not-periodic."""
        from mfgarchon.geometry import TensorProductGrid
        from mfgarchon.geometry.boundary import no_flux_bc
        from mfgarchon.utils.validation.components import validate_boundary_conditions

        # Grid needs its own explicit BC (Issue #674); the object under test is `bc`.
        geometry = TensorProductGrid(
            bounds=[(0, 1), (0, 1)], Nx_points=[10, 10], boundary_conditions=no_flux_bc(dimension=2)
        )
        bc = BoundaryConditions(
            segments=[BCSegment(name="wall", bc_type=BCType.NO_FLUX, boundary="x_min")],
            dimension=2,
            domain_bounds=_bounds_2d(),
        )
        # Does not raise; None default_bc is not flagged as periodic.
        result = validate_boundary_conditions(bc, geometry)
        assert result is not None
