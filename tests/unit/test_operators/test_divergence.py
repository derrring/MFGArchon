"""
Unit tests for DivergenceOperator and AdvectionOperator.

Tests the higher-level operators that combine gradient stencils with
velocity fields for transport equations.

Created: 2026-02-10 (Issue #768 - Test coverage for operators/)
"""

import pytest

import numpy as np
from scipy.sparse.linalg import LinearOperator

from mfgarchon.geometry.boundary import neumann_bc
from mfgarchon.operators.differential.advection import AdvectionOperator
from mfgarchon.operators.differential.divergence import DivergenceOperator

# =============================================================================
# Fixtures
# =============================================================================


def _2d_grid(nx=50, ny=50):
    """Create 2D uniform grid on [0, 1]^2."""
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing="ij")
    return X, Y, dx, dy


# =============================================================================
# DivergenceOperator Tests
# =============================================================================


class TestDivergenceOperator:
    """Tests for DivergenceOperator."""

    @pytest.mark.unit
    def test_2d_identity_field(self):
        """div(x, y) = d(x)/dx + d(y)/dy = 1 + 1 = 2."""
        X, Y, dx, dy = _2d_grid(50, 50)
        F = np.stack([X, Y], axis=0)

        bc = neumann_bc(dimension=2)
        div_op = DivergenceOperator(spacings=[dx, dy], field_shape=(50, 50), bc=bc)
        div_F = div_op(F)

        assert div_F.shape == (50, 50)
        # Interior should be close to 2
        mean_div = np.mean(div_F[5:-5, 5:-5])
        assert abs(mean_div - 2.0) < 0.1

    @pytest.mark.unit
    def test_2d_constant_field(self):
        """div of constant field should be 0."""
        _X, _Y, dx, dy = _2d_grid(40, 40)
        F = np.ones((2, 40, 40)) * 5.0

        bc = neumann_bc(dimension=2)
        div_op = DivergenceOperator(spacings=[dx, dy], field_shape=(40, 40), bc=bc)
        div_F = div_op(F)

        np.testing.assert_allclose(div_F[5:-5, 5:-5], 0.0, atol=1e-10)

    @pytest.mark.unit
    def test_1d_quadratic(self):
        """div(x^2) = d(x^2)/dx = 2x."""
        n = 100
        x = np.linspace(0, 1, n)
        dx = x[1] - x[0]
        F = x[np.newaxis, :] ** 2  # Shape (1, 100)

        bc = neumann_bc(dimension=1)
        div_op = DivergenceOperator(spacings=[dx], field_shape=(n,), bc=bc)
        div_F = div_op(F)

        expected = 2.0 * x
        error = np.max(np.abs(div_F[5:-5] - expected[5:-5]))
        assert error < 1e-10  # Central diff exact for quadratic

    @pytest.mark.unit
    def test_matvec_callable_consistency(self):
        """div_op(F) and div_op @ F.ravel() should match."""
        X, Y, dx, dy = _2d_grid(30, 30)
        F = np.stack([X, Y], axis=0)

        bc = neumann_bc(dimension=2)
        div_op = DivergenceOperator(spacings=[dx, dy], field_shape=(30, 30), bc=bc)

        div_callable = div_op(F)
        div_matvec = div_op @ F.ravel()

        np.testing.assert_allclose(div_callable.ravel(), div_matvec, atol=1e-14)

    @pytest.mark.unit
    def test_isinstance_linear_operator(self):
        """Should be a scipy LinearOperator."""
        div_op = DivergenceOperator(spacings=[0.1, 0.1], field_shape=(20, 20))
        assert isinstance(div_op, LinearOperator)

    @pytest.mark.unit
    def test_input_shape_validation(self):
        """Should raise ValueError for wrong input shape."""
        div_op = DivergenceOperator(spacings=[0.1, 0.1], field_shape=(20, 20))
        with pytest.raises(ValueError, match="doesn't match"):
            div_op(np.zeros((3, 20, 20)))

    @pytest.mark.unit
    def test_spacings_validation(self):
        """Should raise ValueError for mismatched spacings."""
        with pytest.raises(ValueError, match="spacings length"):
            DivergenceOperator(spacings=[0.1], field_shape=(20, 20))

    @pytest.mark.unit
    def test_repr(self):
        """repr should include key attributes."""
        div_op = DivergenceOperator(spacings=[0.1, 0.1], field_shape=(20, 20))
        r = repr(div_op)
        assert "DivergenceOperator" in r
        assert "dimension=2" in r


# =============================================================================
# AdvectionOperator Tests
# =============================================================================


class TestAdvectionGradientForm:
    """Tests for AdvectionOperator in gradient form: v . nabla m."""

    @pytest.mark.unit
    def test_constant_velocity_linear_density(self):
        """v=(1,0) . nabla(x) = 1."""
        X, _Y, dx, dy = _2d_grid(50, 50)
        m = X  # m = x
        v = np.zeros((2, 50, 50))
        v[0] = 1.0

        bc = neumann_bc(dimension=2)
        adv = AdvectionOperator(
            velocity_field=v,
            spacings=[dx, dy],
            field_shape=(50, 50),
            scheme="centered",
            form="gradient",
            bc=bc,
        )
        result = adv(m)

        assert result.shape == (50, 50)
        mean = np.mean(result[5:-5, 5:-5])
        assert abs(mean - 1.0) < 0.1

    @pytest.mark.unit
    def test_zero_velocity(self):
        """v=0 should give zero advection."""
        X, _Y, dx, dy = _2d_grid(30, 30)
        m = X**2
        v = np.zeros((2, 30, 30))

        adv = AdvectionOperator(
            velocity_field=v,
            spacings=[dx, dy],
            field_shape=(30, 30),
            form="gradient",
        )
        result = adv(m)

        np.testing.assert_allclose(result, 0.0, atol=1e-14)

    @pytest.mark.unit
    def test_1d_upwind(self):
        """1D upwind: v=1, m=x^2 -> dm/dx = 2x."""
        n = 100
        x = np.linspace(0, 1, n)
        dx = x[1] - x[0]
        m = x**2
        v = np.ones((1, n))

        adv = AdvectionOperator(
            velocity_field=v,
            spacings=[dx],
            field_shape=(n,),
            scheme="upwind",
            form="gradient",
        )
        result = adv(m)

        expected = 2.0 * x
        error = np.max(np.abs(result[5:-5] - expected[5:-5]))
        # Upwind is O(h) so larger error
        assert error < 0.1


class TestAdvectionDivergenceForm:
    """Tests for AdvectionOperator in divergence form: div(v * m)."""

    @pytest.mark.unit
    def test_constant_v_linear_m(self):
        """div(v*m) with v=(1,0), m=x gives div(x, 0) = 1."""
        X, _Y, dx, dy = _2d_grid(50, 50)
        m = X
        v = np.zeros((2, 50, 50))
        v[0] = 1.0

        bc = neumann_bc(dimension=2)
        adv = AdvectionOperator(
            velocity_field=v,
            spacings=[dx, dy],
            field_shape=(50, 50),
            scheme="centered",
            form="divergence",
            bc=bc,
        )
        result = adv(m)

        mean = np.mean(result[5:-5, 5:-5])
        assert abs(mean - 1.0) < 0.1

    @pytest.mark.unit
    def test_divergence_gradient_equivalence_for_incompressible(self):
        """For constant v (div(v)=0), gradient and divergence forms should match."""
        X, Y, dx, dy = _2d_grid(40, 40)
        m = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
        v = np.zeros((2, 40, 40))
        v[0] = 1.0  # Constant -> div(v) = 0

        bc = neumann_bc(dimension=2)

        adv_grad = AdvectionOperator(
            velocity_field=v,
            spacings=[dx, dy],
            field_shape=(40, 40),
            scheme="centered",
            form="gradient",
            bc=bc,
        )
        adv_div = AdvectionOperator(
            velocity_field=v,
            spacings=[dx, dy],
            field_shape=(40, 40),
            scheme="centered",
            form="divergence",
            bc=bc,
        )

        result_grad = adv_grad(m)
        result_div = adv_div(m)

        # For incompressible v, div(vm) = v.grad(m)
        np.testing.assert_allclose(result_div[5:-5, 5:-5], result_grad[5:-5, 5:-5], atol=1e-10)


class TestAdvectionInterface:
    """Tests for AdvectionOperator interfaces and validation."""

    @pytest.mark.unit
    def test_isinstance_linear_operator(self):
        """Should be a scipy LinearOperator."""
        v = np.ones((2, 20, 20))
        adv = AdvectionOperator(velocity_field=v, spacings=[0.1, 0.1], field_shape=(20, 20))
        assert isinstance(adv, LinearOperator)

    @pytest.mark.unit
    def test_matvec_callable_consistency(self):
        """adv(m) and adv @ m.ravel() should match."""
        X, Y, dx, dy = _2d_grid(30, 30)
        m = X + Y
        v = np.ones((2, 30, 30))

        adv = AdvectionOperator(
            velocity_field=v,
            spacings=[dx, dy],
            field_shape=(30, 30),
            form="gradient",
        )

        result_callable = adv(m)
        result_matvec = adv @ m.ravel()

        np.testing.assert_allclose(result_callable.ravel(), result_matvec, atol=1e-14)

    @pytest.mark.unit
    def test_velocity_shape_validation(self):
        """Should raise ValueError for wrong velocity shape."""
        with pytest.raises(ValueError, match="velocity_field shape"):
            AdvectionOperator(
                velocity_field=np.ones((3, 20, 20)),  # Wrong: 3 components for 2D
                spacings=[0.1, 0.1],
                field_shape=(20, 20),
            )

    @pytest.mark.unit
    def test_invalid_scheme(self):
        """Should raise ValueError for unknown scheme."""
        with pytest.raises(ValueError, match="Unknown scheme"):
            AdvectionOperator(
                velocity_field=np.ones((1, 20)),
                spacings=[0.1],
                field_shape=(20,),
                scheme="weno5",
            )

    @pytest.mark.unit
    def test_invalid_form(self):
        """Should raise ValueError for unknown form."""
        with pytest.raises(ValueError, match="Unknown form"):
            AdvectionOperator(
                velocity_field=np.ones((1, 20)),
                spacings=[0.1],
                field_shape=(20,),
                form="flux",
            )

    @pytest.mark.unit
    def test_repr(self):
        """repr should include key attributes."""
        v = np.ones((2, 20, 20))
        adv = AdvectionOperator(
            velocity_field=v,
            spacings=[0.1, 0.1],
            field_shape=(20, 20),
            scheme="upwind",
            form="divergence",
        )
        r = repr(adv)
        assert "AdvectionOperator" in r
        assert "upwind" in r
        assert "divergence" in r

    @pytest.mark.unit
    def test_field_shape_validation(self):
        """Should raise ValueError for wrong field shape in callable."""
        v = np.ones((2, 20, 20))
        adv = AdvectionOperator(velocity_field=v, spacings=[0.1, 0.1], field_shape=(20, 20))
        with pytest.raises(ValueError, match="doesn't match"):
            adv(np.zeros((30, 30)))


class TestAdvectionPeriodicGridConvention:
    """The conservative wrap face must sit on the torus the grid describes. Issue #1822.

    ``bc=None`` selects the wrap closure but states no convention, so it means the layout this
    package always used: all N cells distinct. A periodic ``BoundaryConditions`` carrying
    ``ENDPOINT_INCLUSIVE`` says the last cell IS the first, and then the wrap face is one cell
    earlier and the repeated cell is not its own control volume.

    Pinned here rather than through a solver because nothing downstream can see it: reverting the
    operator's convention handling left 3096 tests green across `test_operators`, `test_alg`,
    `test_geometry`, `test_fp_fvm` and `test_fdm_centered_conservation`. The route that reaches it
    -- FP-FDM with a callable ``drift_field`` -- had no periodic test at all.
    """

    @staticmethod
    def _stamped_periodic(dimension):
        from mfgarchon.geometry.boundary import periodic_bc
        from mfgarchon.geometry.boundary.types import PeriodicGridConvention

        return periodic_bc(dimension=dimension, convention=PeriodicGridConvention.ENDPOINT_INCLUSIVE)

    @pytest.mark.unit
    def test_inclusive_periodic_conserves_mass_over_the_distinct_cells(self):
        """Telescoping must close over the N-1 real cells, not over N with one counted twice.

        Summing the divergence over the distinct cells is the discrete statement that no mass
        enters or leaves a domain with no boundary. Wrapping all N cells telescopes over a torus
        one cell too long and leaves a residue: measured -1.1e-01 in 1D and -4.4e-01 in 2D.
        """
        rng = np.random.default_rng(1822)
        for shape, spacings in (((13,), [1 / 12]), ((13, 9), [1 / 12, 2 / 8])):
            m = 1.0 + 0.3 * rng.random(shape)
            m[-1] = m[0]  # the repeated cell IS cell 0, on every axis
            if len(shape) == 2:
                m[:, -1] = m[:, 0]
            v = rng.standard_normal((len(shape), *shape))
            adv = AdvectionOperator(
                velocity_field=v,
                spacings=spacings,
                field_shape=shape,
                scheme="upwind",
                form="divergence",
                bc=self._stamped_periodic(len(shape)),
                mass_conservative=True,
            )
            div = adv(m)
            distinct = div[tuple(slice(0, n - 1) for n in shape)]
            cell = float(np.prod(spacings))
            assert float(np.sum(distinct)) * cell == pytest.approx(0.0, abs=1e-12), (
                f"advective flux did not telescope over the {shape} inclusive torus"
            )

    @pytest.mark.unit
    def test_the_repeated_cell_carries_the_first_cells_divergence(self):
        """It is the same control volume, so it must not be given a divergence of its own."""
        m = np.array([1.0, 2.0, 3.0, 4.0, 1.0])
        v = np.array([[0.5, 0.7, 0.2, 0.9, 0.5]])
        adv = AdvectionOperator(
            velocity_field=v,
            spacings=[0.25],
            field_shape=(5,),
            scheme="upwind",
            form="divergence",
            bc=self._stamped_periodic(1),
            mass_conservative=True,
        )
        div = adv(m)
        assert div[-1] == pytest.approx(div[0])

    @pytest.mark.unit
    def test_the_wrap_face_velocity_comes_from_the_last_real_cell_not_the_repeat(self):
        """Pinned on the value, because conservation and the seam are both blind to it.

        The wrap face joins cell ``span-1`` to cell 0, so its velocity is
        ``0.5*(v[span-1] + v[0])``. Reading ``v[N-1]`` there instead -- the repeated node -- is
        still a *consistent* face velocity: each face is shared by the two cells touching it, so
        the flux still telescopes to 0 and ``div[-1] == div[0]`` still holds. Every other test in
        this class passes under that substitution while the divergence moves by 3.2.

        Same shape as the FVM case in
        ``tests/unit/test_alg/test_periodic_torus_oracle_1822.py``: when two candidate expressions
        are both conservative, only the arithmetic separates them.
        """
        span, h = 4, 0.25
        m = np.array([1.0, 2.0, 3.0, 4.0, 1.0])  # cell 4 repeats cell 0
        v = np.array([[0.5, 0.7, 0.2, 0.9, 0.5]])
        adv = AdvectionOperator(
            velocity_field=v,
            spacings=[h],
            field_shape=(5,),
            scheme="upwind",
            form="divergence",
            bc=self._stamped_periodic(1),
            mass_conservative=True,
        )
        div = adv(m)

        # Every velocity is positive, so upwind takes the left cell at each face.
        v_wrap = 0.5 * (v[0, span - 1] + v[0, 0])
        flux_wrap = v_wrap * m[span - 1]
        flux_first = 0.5 * (v[0, 0] + v[0, 1]) * m[0]
        assert div[0] == pytest.approx((flux_first - flux_wrap) / h), (
            "cell 0's inflow must cross a face whose velocity averages the LAST REAL cell with "
            "cell 0; taking it from the repeated node is conservative and still wrong"
        )
        assert div[span - 1] == pytest.approx((flux_wrap - 0.5 * (v[0, span - 2] + v[0, span - 1]) * m[span - 2]) / h)

    @pytest.mark.unit
    def test_bc_none_still_means_the_exclusive_layout(self):
        """The historical caller stated nothing and must keep its numbers (#1822).

        `bc=None` and an unstated periodic BC are the same request, and neither may pick up the
        inclusive layout by accident -- that is what makes this change safe for callers that never
        met a grid.
        """
        from mfgarchon.geometry.boundary import periodic_bc

        m = np.array([1.0, 2.0, 3.0, 4.0, 1.0])
        v = np.array([[0.5, 0.7, 0.2, 0.9, 0.5]])
        kw = {
            "velocity_field": v,
            "spacings": [0.25],
            "field_shape": (5,),
            "scheme": "upwind",
            "form": "divergence",
            "mass_conservative": True,
        }
        unstated = AdvectionOperator(bc=periodic_bc(dimension=1), **kw)(m)
        no_bc = AdvectionOperator(bc=None, **kw)(m)
        inclusive = AdvectionOperator(bc=self._stamped_periodic(1), **kw)(m)

        np.testing.assert_allclose(unstated, no_bc, atol=0, rtol=0)
        assert not np.allclose(unstated, inclusive), (
            "the two conventions must give different operators, or this pins nothing"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
