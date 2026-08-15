"""Tests for ImplicitApplicator — Issue #712 consolidation.

Verifies:
- Inheritance from MeshfreeApplicator (not BaseBCApplicator)
- Protocol-based boundary detection (no hasattr fallbacks)
- Dirichlet and Neumann BC application
- Dispatch integration
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, dirichlet_bc
from mfgarchon.geometry.boundary.applicator_base import DiscretizationType
from mfgarchon.geometry.boundary.applicator_implicit import ImplicitApplicator
from mfgarchon.geometry.boundary.applicator_meshfree import MeshfreeApplicator
from mfgarchon.geometry.protocol import GeometryType

# ---------------------------------------------------------------------------
# Test fixture: lightweight mock geometry implementing GeometryProtocol
# ---------------------------------------------------------------------------

_CENTER = np.array([0.5, 0.5])
_RADIUS = 0.4


class _CircleGeometry:
    """Minimal GeometryProtocol-compliant circle domain for testing."""

    dimension = 2
    geometry_type = GeometryType.IMPLICIT
    num_spatial_points = 441

    def sdf(self, points: np.ndarray) -> np.ndarray:
        return np.linalg.norm(points - _CENTER, axis=-1) - _RADIUS

    def is_on_boundary(self, points: np.ndarray, tolerance: float = 1e-10) -> np.ndarray:
        return np.abs(self.sdf(points)) < tolerance

    def get_boundary_normal(self, points: np.ndarray) -> np.ndarray:
        diff = points - _CENTER
        norms = np.linalg.norm(diff, axis=-1, keepdims=True)
        return diff / np.maximum(norms, 1e-10)

    def get_bounds(self):
        return np.array([0.0, 0.0]), np.array([1.0, 1.0])

    def get_collocation_points(self):
        x = np.linspace(0, 1, 21)
        y = np.linspace(0, 1, 21)
        xx, yy = np.meshgrid(x, y)
        return np.column_stack([xx.ravel(), yy.ravel()])

    def get_spatial_grid(self):
        return self.get_collocation_points()

    def get_grid_shape(self):
        return (21, 21)

    def get_problem_config(self):
        return {"num_spatial_points": 441, "spatial_shape": (21, 21)}

    def get_boundary_conditions(self):
        return None

    def get_boundary_regions(self):
        return {"all": {}}

    def get_boundary_indices(self, points, tolerance=1e-10):
        return np.where(self.is_on_boundary(points, tolerance))[0]

    def get_boundary_info(self, points, tolerance=1e-10):
        indices = self.get_boundary_indices(points, tolerance)
        if len(indices) == 0:
            return indices, np.array([], dtype=np.float64).reshape(0, 2)
        normals = self.get_boundary_normal(points[indices])
        return indices, normals

    def project_to_boundary(self, points):
        diff = points - _CENTER
        norms = np.linalg.norm(diff, axis=-1, keepdims=True)
        return _CENTER + diff / np.maximum(norms, 1e-10) * _RADIUS

    def project_to_interior(self, points):
        sdf_vals = self.sdf(points)
        outside = sdf_vals > 0
        result = points.copy()
        if np.any(outside):
            result[outside] = self.project_to_boundary(points[outside])
        return result


@pytest.fixture
def geometry():
    return _CircleGeometry()


@pytest.fixture
def applicator(geometry):
    return ImplicitApplicator(geometry=geometry, boundary_tolerance=0.03)


@pytest.fixture
def grid_points():
    x = np.linspace(0, 1, 21)
    y = np.linspace(0, 1, 21)
    xx, yy = np.meshgrid(x, y)
    return np.column_stack([xx.ravel(), yy.ravel()])


# ---------------------------------------------------------------------------
# Inheritance and type identity tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInheritance:
    """ImplicitApplicator inherits from MeshfreeApplicator (Issue #712)."""

    def test_inherits_from_meshfree(self, applicator):
        """ImplicitApplicator IS-A MeshfreeApplicator."""
        assert isinstance(applicator, MeshfreeApplicator)

    def test_discretization_type_meshfree(self, applicator):
        """Returns MESHFREE discretization type (inherited from base)."""
        assert applicator.discretization_type == DiscretizationType.MESHFREE


# ---------------------------------------------------------------------------
# Boundary condition application tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBCApplication:
    """ImplicitApplicator applies BCs using geometry protocol methods."""

    def test_apply_dirichlet(self, applicator, geometry, grid_points):
        """Dirichlet BC sets boundary values to prescribed value."""
        from mfgarchon.geometry.boundary import dirichlet_bc

        field = np.linalg.norm(grid_points - _CENTER, axis=-1)
        bc = dirichlet_bc(dimension=2, value=0.0)
        result = applicator.apply(field, bc, grid_points)

        boundary_mask = geometry.is_on_boundary(grid_points, tolerance=0.03)
        if np.any(boundary_mask):
            assert np.allclose(result[boundary_mask], 0.0)

    def test_apply_neumann_no_flux(self, applicator, geometry, grid_points):
        """Neumann zero-flux BC uses interpolation along normals."""
        from mfgarchon.geometry.boundary import neumann_bc

        field = np.linalg.norm(grid_points - _CENTER, axis=-1)
        bc = neumann_bc(dimension=2)
        result = applicator.apply(field, bc, grid_points)

        # `result = field` unchanged satisfies a finiteness check, which is what the comment above
        # this line used to say was wrong while asserting nothing that could see it.
        mask = geometry.is_on_boundary(grid_points, tolerance=0.03)
        assert mask.sum() == 56, "no boundary points detected; the assertions below are then vacuous"
        assert np.array_equal(result[~mask], field[~mask]), "interior must be untouched"
        # The radial field |x - c| has unit normal derivative on this circle, so copying the inward
        # value must pull every boundary value down. Measured drops: 0.0426 to 0.0707, all 56.
        assert np.all(result[mask] < field[mask]), "no-flux must pull boundary values toward the interior"

        # A field with zero normal derivative is a fixed point of the no-flux operator. Measured
        # exact (max deviation 0.0), so array_equal rather than a tolerance.
        constant = np.full(len(grid_points), 3.0)
        np.testing.assert_array_equal(applicator.apply(constant, neumann_bc(dimension=2), grid_points), constant)

    def test_apply_time_positional(self, applicator, grid_points):
        """time parameter can be passed positionally (LSP compliance)."""
        from mfgarchon.geometry.boundary import dirichlet_bc

        field = np.ones(len(grid_points))
        bc = dirichlet_bc(dimension=2, value=0.0)
        # Pass time as positional argument (4th arg)
        result = applicator.apply(field, bc, grid_points, 0.5)
        assert result.shape == field.shape


# ---------------------------------------------------------------------------
# Protocol usage tests (no hasattr fallbacks)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProtocolUsage:
    """Boundary detection uses protocol methods directly, not hasattr."""

    def test_boundary_detection_uses_protocol(self, applicator, geometry, grid_points):
        """_detect_boundary_points calls geometry.is_on_boundary directly."""
        mask = applicator._detect_boundary_points(grid_points)
        assert mask.dtype == np.bool_
        assert mask.shape == (len(grid_points),)

        # dtype and length are satisfied by any boolean mask, including an all-False one. This is
        # the oracle for the claim the test's name makes: detection went through the protocol at
        # the applicator's own tolerance. Measured: a bounding-box detector on this circle selects
        # 80 points with ZERO overlap with the true 56, so the two are fully separated.
        assert np.array_equal(mask, geometry.is_on_boundary(grid_points, tolerance=0.03)), (
            "detection did not go through the geometry protocol at the applicator's tolerance"
        )
        assert mask.sum() == 56, "no boundary points detected; every BC assertion in this file is then vacuous"

    def test_normals_use_protocol(self, applicator, geometry, grid_points):
        """_compute_boundary_normals calls geometry.get_boundary_normal directly."""
        boundary_mask = geometry.is_on_boundary(grid_points, tolerance=0.03)
        if np.any(boundary_mask):
            boundary_pts = grid_points[boundary_mask]
            normals = applicator._compute_boundary_normals(boundary_pts)
            assert normals.shape == boundary_pts.shape
            # Normals should be approximately unit vectors
            norms = np.linalg.norm(normals, axis=-1)
            assert np.allclose(norms, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Dispatch integration test
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDispatch:
    """dispatch.py selects ImplicitApplicator for IMPLICIT geometry type."""

    def test_dispatch_implicit_geometry(self, geometry):
        """get_applicator_for_geometry returns ImplicitApplicator for MESHFREE + IMPLICIT."""
        from mfgarchon.geometry.boundary.dispatch import get_applicator_for_geometry

        applicator = get_applicator_for_geometry(geometry, discretization="MESHFREE")
        assert isinstance(applicator, ImplicitApplicator)

    def test_dispatch_gfdm_uses_meshfree(self, geometry):
        """get_applicator_for_geometry returns MeshfreeApplicator for GFDM."""
        from mfgarchon.geometry.boundary.dispatch import get_applicator_for_geometry

        applicator = get_applicator_for_geometry(geometry, discretization="GFDM")
        # isinstance alone cannot see this branch: ImplicitApplicator IS-A MeshfreeApplicator, so
        # it passes whichever of the two dispatch returns. This fixture's geometry is IMPLICIT, and
        # the MESHFREE branch tested above specialises on exactly that -- so the exact type is the
        # only form that separates the GFDM branch from it.
        assert type(applicator) is MeshfreeApplicator, (
            "GFDM must not pick up the MESHFREE branch's implicit-geometry specialisation"
        )


class TestAgainstAPackageGeometry:
    """The fixtures above supply `_CircleGeometry`, defined in this file.

    That is what hid #1938. `_CircleGeometry.is_on_boundary` takes `tolerance=`, matching the
    applicator's call; every `ImplicitDomain` the package ships -- Hyperrectangle, Hypersphere, the
    three CSG domains, DifferenceDomain, and the base class -- takes `tol=`. So the suite exercised
    a geometry written against the caller rather than against `GeometryProtocol`, and
    `ImplicitApplicator.apply` raised `TypeError` for every real geometry, before any BC code ran.

    These tests use `Hypersphere`, reached the way a user reaches it: through
    `get_applicator_for_geometry(geometry, MESHFREE)`.
    """

    @staticmethod
    def _on_sphere(app_geometry, n=12):
        theta = np.linspace(0.0, 2.0 * np.pi, n + 1)[:-1]
        return np.column_stack(
            [
                app_geometry.center[0] + app_geometry.radius * np.cos(theta),
                app_geometry.center[1] + app_geometry.radius * np.sin(theta),
            ]
        )

    @pytest.fixture
    def package_applicator(self):
        from mfgarchon.geometry.boundary.dispatch import DiscretizationType, get_applicator_for_geometry
        from mfgarchon.geometry.implicit.hypersphere import Hypersphere

        geometry = Hypersphere(center=np.array([0.5, 0.5]), radius=0.4)
        applicator = get_applicator_for_geometry(geometry, DiscretizationType.MESHFREE)
        assert type(applicator).__name__ == "ImplicitApplicator", (
            f"dispatch returned {type(applicator).__name__}; this test is about ImplicitApplicator "
            f"and would otherwise pass by testing something else"
        )
        return applicator, geometry

    @pytest.mark.parametrize("value", [3.0, -7.5, 0.0])
    def test_dirichlet_applies_the_value_it_was_given(self, package_applicator, value):
        """`getattr(bc, "value", 0.0)` read an attribute `BoundaryConditions` does not have, so the
        default fired every time and every Dirichlet BC was applied as 0.0.

        `value=0.0` is included deliberately: it is the case the pre-#1938 fixtures used, and it
        passes either way. It is here as the positive control -- the parametrisation only
        discriminates because the other two rows exist.
        """
        applicator, geometry = package_applicator
        points = self._on_sphere(geometry)
        on_boundary = applicator._detect_boundary_points(points)
        assert on_boundary.any(), "no boundary points detected, so the assertion below is vacuous"

        result = applicator.apply(np.ones(len(points)), dirichlet_bc(dimension=2, value=value), points)

        np.testing.assert_allclose(result[on_boundary], value, atol=1e-12)

    def test_robin_coefficients_reach_the_scheme(self, package_applicator):
        """alpha/beta live on `BCSegment`; the pre-#1938 code read them off `BoundaryConditions`,
        which carries neither, so every Robin BC collapsed onto one answer.

        Asserting *disagreement*, not a value: any two parameter sets that produce the same field
        would satisfy a value assertion computed from the same broken path.
        """
        applicator, geometry = package_applicator
        points = self._on_sphere(geometry)
        on_boundary = applicator._detect_boundary_points(points)

        def robin(alpha, beta, g):
            return BoundaryConditions(
                segments=[BCSegment(name="r", bc_type=BCType.ROBIN, alpha=alpha, beta=beta, value=g)],
                dimension=2,
                default_bc=BCType.ROBIN,
            )

        # `g` is HELD FIXED across the three. Varying it too is what a natural fixture does, and it
        # destroys the discrimination: `g` reaches the scheme through `bc_value`, which was never
        # broken, so three rows differing in `g` differ even with alpha/beta pinned at the getattr
        # defaults (1.0, 1.0). Measured: with `g` varied this test passes over the reverted code.
        # Only alpha and beta move here, so nothing but the alpha/beta channel can explain a change.
        field = np.ones(len(points))
        g = 3.0
        a = applicator.apply(field.copy(), robin(1.0, 1.0, g), points)[on_boundary]
        b = applicator.apply(field.copy(), robin(5.0, 0.2, g), points)[on_boundary]
        c = applicator.apply(field.copy(), robin(0.1, 9.0, g), points)[on_boundary]

        assert not np.allclose(a, b), "changing (alpha, beta) at fixed g left the field unchanged"
        assert not np.allclose(b, c), "changing (alpha, beta) at fixed g left the field unchanged"

        same = applicator.apply(field.copy(), robin(1.0, 1.0, g), points)[on_boundary]
        np.testing.assert_allclose(a, same, atol=1e-12)  # control: identical input, identical output

    def test_a_callable_value_is_still_evaluated(self, package_applicator):
        """Routing the value through `bc.default_value` -- the owner the parent class uses -- would
        have fixed the scalar case and silently broken this one: that field is a plain float and the
        factory writes 0.0 into it whenever the value is callable (`conditions.py:929`). The value is
        read from the segment instead, which keeps the callable.
        """
        applicator, geometry = package_applicator
        points = self._on_sphere(geometry)
        on_boundary = applicator._detect_boundary_points(points)

        assert on_boundary.any(), "no boundary points detected, so the assertion below is vacuous"

        bc = BoundaryConditions(
            segments=[BCSegment(name="d", bc_type=BCType.DIRICHLET, value=lambda p, t: 42.0)],
            dimension=2,
            default_bc=BCType.DIRICHLET,
        )
        result = applicator.apply(np.ones(len(points)), bc, points)

        np.testing.assert_allclose(result[on_boundary], 42.0, atol=1e-12)

    def test_the_value_follows_the_resolved_type_and_not_the_segment_order(self, package_applicator):
        """A segment's value is read only for a *uniform* BC whose type is the one resolved.

        `apply()` resolves ONE `bc_type` for the whole boundary and this applicator has no spatial
        dispatch -- review measured that moving a segment's `normal_direction` to a different arc,
        or deleting its region spec, gives byte-identical output, on base as well. So for a mixed BC
        no segment's value is the right one: whichever is chosen gets imposed everywhere, including
        where that segment was never meant to act. `default_value` is the field that means "where no
        segment governs".

        Three versions of this selector have been wrong, and each row below killed one of them:

        * D killed the `NEUMANN`/`NO_FLUX` family clause, which let a NO_FLUX resolution impose
          `du/dn = 2.5`. NO_FLUX means zero flux by definition.
        * A and C killed the sort-order form (`segments[0]` in disguise).
        * F and G kill any within-type selection among several matches, which the by-type form still
          had: review showed that taking the *last* match instead of the first passed all 6027 tests
          while changing behaviour on a real configuration.

        E carries a non-zero `default_value` deliberately. With `default_value=0.0` it returned 0.0
        in every version of this code and discriminated nothing -- review measured that, and that a
        mutation replacing the fallback with a literal `0.0` passed the entire suite.
        """
        applicator, geometry = package_applicator
        points = self._on_sphere(geometry)
        on_boundary = applicator._detect_boundary_points(points)
        assert on_boundary.any(), "no boundary points detected, so the assertions below are vacuous"
        boundary_points = points[on_boundary]

        def bc_of(segments, default_bc, default_value=0.0):
            return BoundaryConditions(
                segments=segments,
                dimension=2,
                default_bc=default_bc,
                default_value=default_value,
            )

        exit_hi = BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=7.0, priority=5)
        walls = BCSegment(name="walls", bc_type=BCType.NO_FLUX, value=0.0)
        walls_hi = BCSegment(name="walls", bc_type=BCType.NO_FLUX, value=0.0, priority=5)
        exit_lo = BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=7.0)
        robin_wall = BCSegment(name="wall", bc_type=BCType.ROBIN, alpha=2.0, beta=3.0, value=1.0)
        neumann_only = BCSegment(name="n", bc_type=BCType.NEUMANN, value=2.5)
        dir_a = BCSegment(name="a", bc_type=BCType.DIRICHLET, value=7.0, priority=5)
        dir_b = BCSegment(name="b", bc_type=BCType.DIRICHLET, value=0.0)

        cases = [
            ("A mixed, exit sorts first, resolved NO_FLUX", bc_of([exit_hi, walls], BCType.NO_FLUX), 0.0),
            ("B mixed, walls sort first (control)", bc_of([walls_hi, exit_lo], BCType.NO_FLUX), 0.0),
            ("C mixed, Robin resolved, Dirichlet sorts first", bc_of([exit_hi, robin_wall], BCType.ROBIN, 4.0), 4.0),
            ("D NEUMANN segment under default NO_FLUX", bc_of([neumann_only], BCType.NO_FLUX), 0.0),
            ("E no segments, non-zero default_value", bc_of([], BCType.DIRICHLET, 9.0), 9.0),
            ("F two DIRICHLET, high priority first", bc_of([dir_a, dir_b], BCType.DIRICHLET, 9.0), 9.0),
            ("G two DIRICHLET, order swapped", bc_of([dir_b, dir_a], BCType.DIRICHLET, 9.0), 9.0),
        ]

        for label, bc, expected in cases:
            resolved = bc._resolve_default_bc("test")
            value = applicator._resolve_bc_value(bc, boundary_points, 0.0, resolved)
            np.testing.assert_allclose(
                value,
                expected,
                atol=1e-12,
                err_msg=f"{label}: resolved {resolved.name} but took the value from somewhere else",
            )

    def test_a_uniform_bc_still_reads_its_own_segment(self, package_applicator):
        """The gate above must not throw away the case this PR exists to fix.

        Without this, `_resolve_bc_value` returning `bc.default_value` unconditionally would satisfy
        every row of the previous test -- review measured exactly that mutation passing the whole
        suite when the fallback had nothing non-zero to distinguish it.
        """
        applicator, geometry = package_applicator
        points = self._on_sphere(geometry)
        on_boundary = applicator._detect_boundary_points(points)
        assert on_boundary.any(), "no boundary points detected, so the assertions below are vacuous"
        boundary_points = points[on_boundary]

        for value in (3.0, -7.5):
            bc = dirichlet_bc(dimension=2, value=value)
            assert bc.is_uniform, "this test is about the uniform path and the fixture is not uniform"
            resolved = bc._resolve_default_bc("test")
            got = applicator._resolve_bc_value(bc, boundary_points, 0.0, resolved)
            np.testing.assert_allclose(got, value, atol=1e-12)
