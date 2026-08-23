"""
Unit tests for BC applicator utilities.

Tests the application of uniform and mixed boundary conditions to grid fields.
"""

import pytest

import numpy as np

from mfgarchon.geometry.boundary import (
    GridType,
    dirichlet_bc,
    neumann_bc,
    no_flux_bc,
    # Factory functions for creating BCs
    periodic_bc,
)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_unsupported_bc_type_error(self):
        """Test that invalid BC type raises error during creation."""
        # With the new unified class, invalid types raise error during creation
        from mfgarchon.geometry.boundary import uniform_bc

        with pytest.raises(ValueError):
            uniform_bc(bc_type="unknown_type")

        # The ghost values at top should vary with x (stored in _padded_t1 for future assertions)


class TestExtrapolationBC:
    """Tests for extrapolation boundary conditions (for unbounded domains)."""

    def test_linear_extrapolation_function(self):
        """Test ghost_cell_linear_extrapolation function directly."""
        from mfgarchon.geometry.boundary import ghost_cell_linear_extrapolation

        # Linear function: f(x) = 2x + 1
        # At x=0: f(0)=1, at x=1: f(1)=3
        # Extrapolated to x=-1: f(-1) = -1
        u_0, u_1 = 1.0, 3.0  # Note: u_0 is at boundary, u_1 is one step inside
        # For left boundary extrapolation: ghost = 2*u_0 - u_1 = 2*1 - 3 = -1
        ghost = ghost_cell_linear_extrapolation((u_0, u_1))
        assert np.isclose(ghost, -1.0)

    def test_quadratic_extrapolation_function(self):
        """Test ghost_cell_quadratic_extrapolation function directly."""
        from mfgarchon.geometry.boundary import ghost_cell_quadratic_extrapolation

        # Quadratic function: f(x) = x^2
        # At x=0: f(0)=0, x=1: f(1)=1, x=2: f(2)=4
        # Extrapolated to x=-1: f(-1) = 1
        u_0, u_1, u_2 = 0.0, 1.0, 4.0
        # For left boundary: ghost = 3*u_0 - 3*u_1 + u_2 = 0 - 3 + 4 = 1
        ghost = ghost_cell_quadratic_extrapolation((u_0, u_1, u_2))
        assert np.isclose(ghost, 1.0)


class TestLazyDimensionBinding:
    """Tests for lazy dimension binding (Issue #495)."""

    def test_dirichlet_bc_no_dimension(self):
        """Test creating Dirichlet BC without dimension."""
        bc = dirichlet_bc(value=0.0)  # No dimension specified
        assert bc.dimension is None
        assert not bc.is_bound
        assert str(bc) == "BoundaryConditions(unbound, dirichlet, value=0.0)"

    def test_neumann_bc_no_dimension(self):
        """Test creating Neumann BC without dimension."""
        bc = neumann_bc(value=0.0)  # No dimension specified
        assert bc.dimension is None
        assert not bc.is_bound

    def test_periodic_bc_no_dimension(self):
        """Test creating Periodic BC without dimension."""
        bc = periodic_bc()  # No dimension specified
        assert bc.dimension is None
        assert not bc.is_bound

    def test_bind_dimension_explicit(self):
        """Test explicit dimension binding via bind_dimension()."""
        bc = dirichlet_bc(value=0.0)
        assert bc.dimension is None

        bc_2d = bc.bind_dimension(2)
        assert bc_2d.dimension == 2
        assert bc_2d.is_bound
        assert str(bc_2d) == "BoundaryConditions(2D, dirichlet, value=0.0)"

        # Original BC should be unchanged (immutable via replace)
        assert bc.dimension is None

    def test_bind_dimension_idempotent(self):
        """Test that binding same dimension twice returns same BC."""
        bc = dirichlet_bc(value=0.0, dimension=2)
        bc_bound = bc.bind_dimension(2)

        # Should return same object since dimension already matches
        assert bc_bound is bc

    def test_bind_dimension_mismatch_error(self):
        """Test that binding different dimension raises error."""
        bc = dirichlet_bc(value=0.0, dimension=2)

        with pytest.raises(ValueError, match="BC dimension mismatch"):
            bc.bind_dimension(3)

    def test_grid_binds_dimension_automatically(self):
        """Test that TensorProductGrid automatically binds dimension."""
        from mfgarchon.geometry import TensorProductGrid

        # Create BC without dimension
        bc = dirichlet_bc(value=0.0)
        assert bc.dimension is None

        # Create grid with BC - dimension should be bound automatically
        grid = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx=[10, 10],
            boundary_conditions=bc,
        )

        # Grid's stored BC should have dimension bound
        stored_bc = grid.get_boundary_conditions()
        assert stored_bc.dimension == 2
        assert stored_bc.is_bound

    def test_unbound_bc_cannot_apply(self):
        """Test that unbound BC cannot be applied directly."""
        bc = dirichlet_bc(value=0.0)  # No dimension

        # Applicator requires dimension to compute ghost cells
        with pytest.raises(ValueError, match="BC dimension not set"):
            # This should fail because dimension is needed for ghost cell computation
            bc._require_dimension("apply BC")

    def test_validate_unbound_bc_warns(self):
        """Test that validate() warns about unbound dimension."""
        bc = dirichlet_bc(value=0.0)
        _is_valid, warnings = bc.validate()

        # Should have warning about dimension not set
        assert any("Dimension not set" in w for w in warnings)


class TestSDFParticleBCHandler:
    """Tests for SDF-based particle BC handler (Issue #497)."""

    def test_sdf_handler_creation(self):
        """Test creating an SDF particle BC handler."""
        from mfgarchon.geometry.boundary import SDFParticleBCHandler
        from mfgarchon.utils.numerical import sdf_sphere

        def circle_sdf(points):
            return sdf_sphere(points, center=[0, 0], radius=1.0)

        handler = SDFParticleBCHandler(circle_sdf, dimension=2)
        assert handler.dimension == 2
        assert handler.sdf is not None

    def test_particles_inside_unchanged(self):
        """Test that particles inside domain are unchanged."""
        from mfgarchon.geometry.boundary import SDFParticleBCHandler
        from mfgarchon.utils.numerical import sdf_sphere

        def circle_sdf(points):
            return sdf_sphere(points, center=[0, 0], radius=1.0)

        handler = SDFParticleBCHandler(circle_sdf, dimension=2)

        # Particles inside the unit circle
        X_old = np.array([[0.0, 0.0], [0.3, 0.0], [0.0, 0.5]])
        X_new = np.array([[0.1, 0.0], [0.4, 0.0], [0.0, 0.6]])

        X_result, _ = handler.apply_bc(X_old, X_new)

        # Should be unchanged (all inside)
        np.testing.assert_array_almost_equal(X_result, X_new)

    def test_particle_crossing_reflected(self):
        """Test that particles crossing boundary are reflected."""
        from mfgarchon.geometry.boundary import SDFParticleBCHandler
        from mfgarchon.utils.numerical import sdf_sphere

        def circle_sdf(points):
            return sdf_sphere(points, center=[0, 0], radius=1.0)

        handler = SDFParticleBCHandler(circle_sdf, dimension=2)

        # One particle crosses the boundary
        X_old = np.array([[0.9, 0.0]])  # Inside
        X_new = np.array([[1.2, 0.0]])  # Outside

        X_result, _ = handler.apply_bc(X_old, X_new)

        # Should be reflected back inside
        assert handler.sdf(X_result)[0] <= 0, "Reflected particle should be inside"

    def test_velocity_reflection(self):
        """Test that velocity is reflected at boundary."""
        from mfgarchon.geometry.boundary import SDFParticleBCHandler
        from mfgarchon.utils.numerical import sdf_sphere

        def circle_sdf(points):
            return sdf_sphere(points, center=[0, 0], radius=1.0)

        handler = SDFParticleBCHandler(circle_sdf, dimension=2)

        # Particle moving outward along x-axis
        X_old = np.array([[0.9, 0.0]])  # Inside
        X_new = np.array([[1.2, 0.0]])  # Outside
        V = np.array([[1.0, 0.0]])  # Velocity toward boundary

        _X_result, V_result = handler.apply_bc(X_old, X_new, velocities=V)

        # Normal at x=1, y=0 is (1, 0), so velocity should reverse
        assert V_result[0, 0] < 0, "x-velocity should reverse (reflect)"
        np.testing.assert_almost_equal(V_result[0, 1], 0.0)

    def test_contains_method(self):
        """Test the contains() method."""
        from mfgarchon.geometry.boundary import SDFParticleBCHandler
        from mfgarchon.utils.numerical import sdf_sphere

        def circle_sdf(points):
            return sdf_sphere(points, center=[0, 0], radius=1.0)

        handler = SDFParticleBCHandler(circle_sdf, dimension=2)

        points = np.array([[0.0, 0.0], [0.5, 0.0], [2.0, 0.0]])
        inside = handler.contains(points)

        assert inside[0] is True or inside[0] == True  # noqa: E712 - Center inside
        assert inside[1] is True or inside[1] == True  # noqa: E712 - Mid inside
        assert inside[2] is False or inside[2] == False  # noqa: E712 - Outside

    def test_multiple_particles_mixed(self):
        """Test handling multiple particles with mixed inside/crossing."""
        from mfgarchon.geometry.boundary import SDFParticleBCHandler
        from mfgarchon.utils.numerical import sdf_sphere

        def circle_sdf(points):
            return sdf_sphere(points, center=[0, 0], radius=1.0)

        handler = SDFParticleBCHandler(circle_sdf, dimension=2)

        # Mix of inside moves and boundary crossings
        X_old = np.array(
            [
                [0.0, 0.0],  # Stays inside
                [0.9, 0.0],  # Will cross
                [0.0, 0.5],  # Stays inside
            ]
        )
        X_new = np.array(
            [
                [0.1, 0.0],  # Still inside
                [1.2, 0.0],  # Crossed
                [0.0, 0.6],  # Still inside
            ]
        )

        X_result, _ = handler.apply_bc(X_old, X_new)

        # All should end up inside
        sdf_result = handler.sdf(X_result)
        assert np.all(sdf_result <= 0), "All particles should be inside after BC"

        # Non-crossing particles should be unchanged
        np.testing.assert_array_almost_equal(X_result[0], X_new[0])
        np.testing.assert_array_almost_equal(X_result[2], X_new[2])

    def test_box_sdf(self):
        """Test with box/rectangular SDF."""
        from mfgarchon.geometry.boundary import SDFParticleBCHandler
        from mfgarchon.utils.numerical import sdf_box

        def box_sdf(points):
            return sdf_box(points, bounds=[[0, 1], [0, 1]])

        handler = SDFParticleBCHandler(box_sdf, dimension=2)

        # Particle crossing right boundary
        X_old = np.array([[0.9, 0.5]])
        X_new = np.array([[1.2, 0.5]])

        X_result, _ = handler.apply_bc(X_old, X_new)

        # Should be reflected back into box
        assert 0 <= X_result[0, 0] <= 1, "x should be in [0, 1]"
        assert 0 <= X_result[0, 1] <= 1, "y should be in [0, 1]"


# =============================================================================
# Topology/Calculator Composition Tests (Issue #516)
# =============================================================================


class TestTopologyClasses:
    """Test Topology implementations (PeriodicTopology, BoundedTopology)."""

    def test_periodic_topology_creation(self):
        """Test PeriodicTopology initialization."""
        from mfgarchon.geometry.boundary import PeriodicTopology

        topo = PeriodicTopology(dimension=2, shape=(10, 15))
        assert topo.is_periodic is True
        assert topo.dimension == 2
        assert topo.shape == (10, 15)

    def test_bounded_topology_creation(self):
        """Test BoundedTopology initialization."""
        from mfgarchon.geometry.boundary import BoundedTopology

        topo = BoundedTopology(dimension=3, shape=(5, 6, 7))
        assert topo.is_periodic is False
        assert topo.dimension == 3
        assert topo.shape == (5, 6, 7)

    def test_topology_dimension_shape_mismatch_error(self):
        """Test that mismatched dimension and shape raises error."""
        from mfgarchon.geometry.boundary import PeriodicTopology

        with pytest.raises(ValueError, match="Shape length"):
            PeriodicTopology(dimension=2, shape=(10, 15, 20))

    def test_topology_repr(self):
        """Test topology string representation."""
        from mfgarchon.geometry.boundary import BoundedTopology, PeriodicTopology

        periodic = PeriodicTopology(dimension=2, shape=(10, 10))
        assert "PeriodicTopology" in repr(periodic)
        assert "dimension=2" in repr(periodic)

        bounded = BoundedTopology(dimension=2, shape=(10, 10))
        assert "BoundedTopology" in repr(bounded)


class TestCalculatorClasses:
    """Test BoundaryCalculator implementations."""

    def test_dirichlet_calculator(self):
        """Test DirichletCalculator computes correct ghost values."""
        from mfgarchon.geometry.boundary import DirichletCalculator

        calc = DirichletCalculator(boundary_value=5.0)
        # Cell-centered: u_ghost = 2*g - u_interior = 2*5 - 3 = 7
        ghost = calc.compute(interior_value=3.0, dx=0.1, side="min")
        assert np.isclose(ghost, 7.0)

    def test_neumann_calculator_zero_flux(self):
        """Test NeumannCalculator with zero flux (edge extension)."""
        from mfgarchon.geometry.boundary import NeumannCalculator

        calc = NeumannCalculator(flux_value=0.0)
        ghost = calc.compute(interior_value=3.0, dx=0.1, side="min")
        # Zero flux: ghost = interior
        assert np.isclose(ghost, 3.0)

    def test_neumann_calculator_reproduces_a_linear_field(self):
        """An external oracle, not the formula restated. #1972

        The version here asserted `expected_min = 5.0 - 2*dx*g` with the comment
        `# For min side (outward_sign = -1): ghost = interior - 2*dx*g` -- `expected` copied from
        the implementation, so it could only check that the code equals itself. It protected two
        defects at once: a factor of 2 (the ghost-to-interior separation is `dx`, not `2*dx`, on
        both centrings) and a sign (`flux_value` is du/dn, which already carries the direction, so
        the wrapper's `outward_sign` inverted the min wall). Measured before the fix, u0=1, dx=0.25,
        value=2: this calculator gave 0.0/2.0 where the live applicator path gave 1.5/1.5.

        `u = a*x` is reproduced exactly by any consistent first-order ghost rule, so it decides the
        question without reference to any implementation.
        """
        from mfgarchon.geometry.boundary import NeumannCalculator

        dx, slope = 0.1, 3.0

        for side, x_interior, x_ghost in (("min", dx / 2, -dx / 2), ("max", 1.0 - dx / 2, 1.0 + dx / 2)):
            g = -slope if side == "min" else +slope  # du/dn: the outward normal flips at the min wall
            got = NeumannCalculator(flux_value=g).compute(interior_value=slope * x_interior, dx=dx, side=side)
            assert np.isclose(got, slope * x_ghost), f"{side}: {got} != {slope * x_ghost} for u = {slope}x"

    @pytest.mark.parametrize("grid_type", [GridType.CELL_CENTERED, GridType.VERTEX_CENTERED])
    @pytest.mark.parametrize("side", ["min", "max"])
    def test_robin_with_alpha_zero_reproduces_a_linear_field(self, grid_type, side):
        """#2063: the vertex-centred min wall was inverted, and only this path reached it.

        `alpha = 0, beta = 1` makes Robin the same condition as Neumann, so `u = a*x` must come
        back exactly -- an external oracle, not either formula restated. Measured before the fix,
        VERTEX_CENTERED/min: +0.3000 where -0.3000 is exact. The other three cells passed, which is
        why nothing caught it; the cell-centred branch never used the sign and `grid_type` defaults
        to cell-centred, so the only path that could reach the bug had no test.

        The cause was `RobinCalculator` deriving an outward sign from `side` and passing it on --
        the only caller in the package that did. Passing the physically correct -1.0 was what broke
        it, so the seven callers that omitted the argument were right by accident.
        """
        from mfgarchon.geometry.boundary import RobinCalculator

        slope, dx = 3.0, 0.1
        x_interior, x_ghost = (0.0, -dx) if side == "min" else (1.0, 1.0 + dx)
        outward_normal = -1.0 if side == "min" else +1.0

        calc = RobinCalculator(alpha=0.0, beta=1.0, rhs_value=slope * outward_normal, grid_type=grid_type)
        got = calc.compute(interior_value=slope * x_interior, dx=dx, side=side)

        assert np.isclose(got, slope * x_ghost), f"{grid_type.name}/{side}: {got} != {slope * x_ghost}"

    def test_the_fp_no_flux_ghost_zeroes_the_TOTAL_flux_given_an_axis_velocity(self):
        """#2063: `drift_velocity` is v_x, not v*n, and the docstring said the opposite.

        The contract is J.n = 0 with J = v*rho - D*grad(rho), so the check is the residual itself,
        not a formula restated. Fed v*n as the Args line instructed, the min wall leaves a residual
        of 1.25 instead of 0; fed v_x it is machine zero. The max wall cannot discriminate, since
        v*n = v_x there.

        `outward_normal_sign` has no default, because the two callers hold different quantities --
        `ghost_cell_advection_diffusion_no_flux` already has v*n and passes 1.0, `ZeroFluxCalculator`
        has v_x and passes the wall's sign -- and +1.0 silently means "max wall".
        """
        from mfgarchon.geometry.boundary.ghost_cells import ghost_cell_fp_no_flux

        D, dx, rho_interior = 0.125, 0.1, 1.0
        for outward_normal in (+1.0, -1.0):
            for v_x in (+0.5, -0.5):
                ghost = ghost_cell_fp_no_flux(rho_interior, v_x, D, dx, outward_normal)
                v_n = v_x * outward_normal
                rho_face = (ghost + rho_interior) / 2.0
                drho_dn = (ghost - rho_interior) / dx
                assert np.isclose(v_n * rho_face - D * drho_dn, 0.0, atol=1e-12)

        with pytest.raises(TypeError, match="outward_normal_sign"):
            ghost_cell_fp_no_flux(rho_interior, 0.5, D, dx)

    def test_neumann_calculator_agrees_with_the_live_applicator_path(self):
        """The two implementations of this ghost disagreed by a factor of 2 and a sign until #1972.

        Not a tautology: they are still separate call paths -- `pad_array_with_ghosts` reaches
        `ghost_cell_neumann` through the applicator, this reaches it through the calculator -- and
        nothing but this test compares them.
        """
        from mfgarchon.geometry.boundary import NeumannCalculator, neumann_bc
        from mfgarchon.geometry.boundary.applicator_fdm import pad_array_with_ghosts

        u = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        for g in (0.0, 2.0, -7.0):
            for dx in (0.05, 0.25):
                padded = pad_array_with_ghosts(u, neumann_bc(g, dimension=1), ghost_depth=1, spacing=dx)
                calc = NeumannCalculator(flux_value=g)
                assert np.isclose(padded[0], calc.compute(interior_value=u[0], dx=dx, side="min"))
                assert np.isclose(padded[-1], calc.compute(interior_value=u[-1], dx=dx, side="max"))

    @pytest.mark.parametrize(
        ("name", "args", "replacement"),
        [
            ("high_order_ghost_neumann", ([1.0, 2.0, 3.0, 4.0], 1.0, 1.0), "ghost_cell_neumann"),
            ("high_order_ghost_dirichlet", ([1.0, 1.0, 1.0, 1.0], 1.0), "ghost_cell_dirichlet"),
        ],
    )
    def test_the_retired_high_order_ghosts_refuse_and_name_their_replacement(self, name, args, replacement):
        """#1936. Both shipped in v0.21.0, so the import must survive the retirement.

        The refusal has to NAME the replacement, not merely refuse: neither function had a caller
        or a test, so the next reader has nothing but this message to go on and would otherwise
        reimplement the same formula.

        Neither is uniformly wrong, which is why reading them did not settle it. `u = x`, max wall:
        +1.5 / -0.5909 / -0.4600 against +0.5; the SAME order=4 and order=5 branches are exact at
        the min wall. `u = 1`, dirichlet: [1.6, 4.0] and [1.5, 3.3333] against [1.0, 1.0] on the
        default cell-centred path, correct on the vertex-centred one. What condemns them is the
        rate, not any single row: the neumann ghost is O(h) where `ghost_cell_neumann` is O(h^3).
        """
        import mfgarchon.geometry.boundary as boundary

        func = getattr(boundary, name)
        with pytest.raises(NotImplementedError, match="RETIRED") as excinfo:
            func(*args)
        assert replacement in str(excinfo.value), f"{name}'s refusal does not name {replacement}"

    def test_robin_calculator(self):
        """Test RobinCalculator for mixed boundary conditions."""
        from mfgarchon.geometry.boundary import RobinCalculator

        # Robin: alpha*u + beta*du/dn = g
        # With alpha=1, beta=0, it reduces to Dirichlet: u = g
        calc_dirichlet = RobinCalculator(alpha=1.0, beta=0.0, rhs_value=2.0)
        ghost = calc_dirichlet.compute(interior_value=1.0, dx=0.1, side="min")
        # Should behave like Dirichlet: ghost = 2*2 - 1 = 3
        assert np.isclose(ghost, 3.0)

    def test_no_flux_calculator(self):
        """Test NoFluxCalculator (edge extension)."""
        from mfgarchon.geometry.boundary import NoFluxCalculator

        calc = NoFluxCalculator()
        ghost = calc.compute(interior_value=7.5, dx=0.1, side="max")
        assert np.isclose(ghost, 7.5)

    def test_linear_extrapolation_calculator(self):
        """Test LinearExtrapolationCalculator (zero second derivative)."""
        from mfgarchon.geometry.boundary import LinearExtrapolationCalculator

        calc = LinearExtrapolationCalculator()
        # ghost = 2*u_0 - u_1 = 2*5 - 3 = 7
        ghost = calc.compute(interior_value=5.0, dx=0.1, side="min", second_interior_value=3.0)
        assert np.isclose(ghost, 7.0)

    def test_quadratic_extrapolation_calculator(self):
        """Test QuadraticExtrapolationCalculator (zero third derivative)."""
        from mfgarchon.geometry.boundary import QuadraticExtrapolationCalculator

        calc = QuadraticExtrapolationCalculator()
        # ghost = 3*u_0 - 3*u_1 + u_2 = 3*5 - 3*3 + 1 = 7
        ghost = calc.compute(
            interior_value=5.0,
            dx=0.1,
            side="min",
            second_interior_value=3.0,
            third_interior_value=1.0,
        )
        assert np.isclose(ghost, 7.0)

    def test_fp_no_flux_calculator(self):
        """Test FPNoFluxCalculator (physics-aware zero total flux)."""
        from mfgarchon.geometry.boundary import FPNoFluxCalculator

        # Zero drift: reduces to Neumann (ghost = interior)
        calc = FPNoFluxCalculator(drift_velocity=0.0, diffusion_coeff=1.0)
        ghost = calc.compute(interior_value=3.0, dx=0.1, side="min")
        assert np.isclose(ghost, 3.0)


class TestGhostBuffer:
    """Test GhostBuffer with Topology/Calculator composition."""

    def test_ghost_buffer_periodic_2d(self):
        """Test GhostBuffer with periodic topology in 2D."""
        from mfgarchon.geometry.boundary import GhostBuffer, PeriodicTopology

        topo = PeriodicTopology(dimension=2, shape=(5, 5))
        buffer = GhostBuffer(topo)

        # Set interior to a gradient
        buffer.interior[:] = np.arange(25).reshape(5, 5)
        buffer.update()

        # Check periodic wrap-around
        # Low ghost should equal high interior
        np.testing.assert_array_equal(buffer.padded[0, 1:-1], buffer.padded[-2, 1:-1])
        # High ghost should equal low interior
        np.testing.assert_array_equal(buffer.padded[-1, 1:-1], buffer.padded[1, 1:-1])

    def test_ghost_buffer_bounded_dirichlet_2d(self):
        """Test GhostBuffer with bounded topology and Dirichlet BC."""
        from mfgarchon.geometry.boundary import (
            BoundedTopology,
            DirichletCalculator,
            GhostBuffer,
        )

        topo = BoundedTopology(dimension=2, shape=(5, 5))
        calc = DirichletCalculator(boundary_value=0.0)
        buffer = GhostBuffer(topo, calc, dx=0.1)

        buffer.interior[:] = 1.0
        buffer.update()

        # Dirichlet g=0, interior=1: ghost = 2*0 - 1 = -1
        assert np.allclose(buffer.padded[0, 1:-1], -1.0)
        assert np.allclose(buffer.padded[-1, 1:-1], -1.0)
        assert np.allclose(buffer.padded[1:-1, 0], -1.0)
        assert np.allclose(buffer.padded[1:-1, -1], -1.0)

    def test_ghost_buffer_bounded_neumann_2d(self):
        """Test GhostBuffer with bounded topology and Neumann BC (zero flux)."""
        from mfgarchon.geometry.boundary import (
            BoundedTopology,
            GhostBuffer,
            NeumannCalculator,
        )

        topo = BoundedTopology(dimension=2, shape=(5, 5))
        calc = NeumannCalculator(flux_value=0.0)
        buffer = GhostBuffer(topo, calc, dx=0.1)

        buffer.interior[:] = np.arange(25).reshape(5, 5).astype(float)
        buffer.update()

        # Zero Neumann: ghost = interior (edge extension)
        np.testing.assert_array_almost_equal(buffer.padded[0, 1:-1], buffer.interior[0, :])

    def test_ghost_buffer_bounded_requires_calculator(self):
        """Test that bounded topology requires calculator."""
        from mfgarchon.geometry.boundary import BoundedTopology, GhostBuffer

        topo = BoundedTopology(dimension=2, shape=(5, 5))
        with pytest.raises(ValueError, match="requires a BoundaryCalculator"):
            GhostBuffer(topo)  # No calculator provided

    def test_ghost_buffer_periodic_ignores_calculator(self):
        """Test that periodic topology ignores calculator (uses wrap-around)."""
        from mfgarchon.geometry.boundary import (
            DirichletCalculator,
            GhostBuffer,
            PeriodicTopology,
        )

        topo = PeriodicTopology(dimension=2, shape=(5, 5))
        calc = DirichletCalculator(boundary_value=999.0)  # Should be ignored
        buffer = GhostBuffer(topo, calc)

        buffer.interior[:] = np.arange(25).reshape(5, 5)
        buffer.update()

        # Should use wrap-around, not Dirichlet
        # If Dirichlet were used, ghost would be 2*999 - interior
        # With wrap-around, ghost[0] = interior[-1]
        np.testing.assert_array_equal(buffer.padded[0, 1:-1], buffer.padded[-2, 1:-1])

    def test_ghost_buffer_properties(self):
        """Test GhostBuffer property accessors."""
        from mfgarchon.geometry.boundary import (
            BoundedTopology,
            DirichletCalculator,
            GhostBuffer,
        )

        topo = BoundedTopology(dimension=2, shape=(10, 15))
        calc = DirichletCalculator(boundary_value=0.0)
        buffer = GhostBuffer(topo, calc, dx=(0.1, 0.2), ghost_depth=2)

        assert buffer.shape == (10, 15)
        assert buffer.padded_shape == (14, 19)
        assert buffer.ghost_depth == 2
        assert buffer.dx == (0.1, 0.2)
        assert buffer.topology is topo
        assert buffer.calculator is calc

    def test_ghost_buffer_reset(self):
        """Test GhostBuffer reset method."""
        from mfgarchon.geometry.boundary import GhostBuffer, PeriodicTopology

        topo = PeriodicTopology(dimension=1, shape=(10,))
        buffer = GhostBuffer(topo)

        buffer.interior[:] = 5.0
        buffer.reset(fill_value=0.0)

        assert np.allclose(buffer.padded, 0.0)

    def test_ghost_buffer_copy_to_interior(self):
        """Test GhostBuffer copy_to_interior method."""
        from mfgarchon.geometry.boundary import GhostBuffer, PeriodicTopology

        topo = PeriodicTopology(dimension=2, shape=(3, 3))
        buffer = GhostBuffer(topo)

        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        buffer.copy_to_interior(data)

        np.testing.assert_array_equal(buffer.interior, data)

    def test_ghost_buffer_3d_periodic(self):
        """Test GhostBuffer with 3D periodic topology."""
        from mfgarchon.geometry.boundary import GhostBuffer, PeriodicTopology

        topo = PeriodicTopology(dimension=3, shape=(4, 4, 4))
        buffer = GhostBuffer(topo)

        # Set a 3D gradient
        for i in range(4):
            buffer.interior[i, :, :] = float(i)

        buffer.update()

        # Check wrap-around on first axis
        np.testing.assert_array_equal(buffer.padded[0, 1:-1, 1:-1], buffer.padded[-2, 1:-1, 1:-1])
        np.testing.assert_array_equal(buffer.padded[-1, 1:-1, 1:-1], buffer.padded[1, 1:-1, 1:-1])


class TestLinearReflectionNeumannFlux:
    """Nonzero Neumann flux in the order<=2 linear-reflection ghost path (Issue #1186 sibling).

    The FDM/SL ghost path (``_apply_linear_reflection``) used a pure mirror for
    Neumann/no-flux, silently dropping a nonzero ``neumann_bc(value=g)`` (it always encoded
    du/dn=0). The fix adds the linear flux offset (Robin-branch sign convention). g=0 stays a
    pure mirror -> byte-identical for the no-flux/zero-Neumann case the paper uses (EOC-safe).
    """

    @staticmethod
    def _ghosts(bc, u, dx=0.1, g=1):
        from mfgarchon.geometry.boundary.applicator_fdm import PreallocatedGhostBuffer

        buf = PreallocatedGhostBuffer(
            interior_shape=(len(u),),
            boundary_conditions=bc,
            domain_bounds=np.array([[0.0, (len(u) - 1) * dx]]),
            order=2,  # order<=2 -> linear reflection path
            ghost_depth=g,
        )
        buf.interior[:] = u
        buf.update_ghosts()
        return buf.padded.copy()

    def test_zero_neumann_byte_identical_to_no_flux(self):
        """g=0 must reproduce the pure mirror (no-flux), bit-for-bit -- the EOC-safe property."""
        u = np.sin(np.linspace(0.0, 1.0, 11))
        g_noflux = self._ghosts(no_flux_bc(dimension=1), u)
        g_neumann0 = self._ghosts(neumann_bc(dimension=1, value=0.0), u)
        np.testing.assert_array_equal(g_neumann0, g_noflux)

    def test_nonzero_neumann_flux_recovered(self):
        """A nonzero neumann_bc(value=v) is applied (not dropped): the cell-centered ghost
        encodes du/dn = v at both walls (was silently 0)."""
        dx = 0.1
        u = np.sin(np.linspace(0.0, 1.0, 11))
        for v in (0.5, -1.3):
            gh = self._ghosts(neumann_bc(dimension=1, value=v), u, dx=dx)
            # du/dn is the OUTWARD-normal derivative: low wall outward = -x, so
            # du/dn = (u_ghost - u_interior)/dx (Issue #1262, 2026-06-10 audit). The previous
            # (u_interior - u_ghost)/dx computed -du/dn; that sign error cancelled the old
            # ghost-sign bug (u_g = u_i - dx*v), so this test passed against incorrect ghosts.
            # With the corrected ghost (u_g = u_i + dx*v) the low-wall du/dn is +v.
            dudn_low = (gh[0] - gh[1]) / dx  # (u_ghost - u_interior)/dx, low outward = -x
            dudn_high = (gh[-1] - gh[-2]) / dx  # (u_ghost - u_interior)/dx, high outward = +x
            assert abs(dudn_low - v) < 1e-12, f"low du/dn={dudn_low} != {v}"
            assert abs(dudn_high - v) < 1e-12, f"high du/dn={dudn_high} != {v}"

    def test_no_flux_ignores_stray_value(self):
        """NO_FLUX is definitionally zero-flux: the mirror is unchanged regardless of input."""
        u = np.cos(np.linspace(0.0, 1.0, 11))
        # no_flux must equal a true zero-Neumann mirror (no offset applied to NO_FLUX).
        np.testing.assert_array_equal(
            self._ghosts(no_flux_bc(dimension=1), u), self._ghosts(neumann_bc(dimension=1, value=0.0), u)
        )
