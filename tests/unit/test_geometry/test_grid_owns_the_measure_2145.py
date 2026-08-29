"""The grid owns the measure: `quadrature_weights` and `integrate` (#2145).

Before this, every caller that wanted a total mass chose a quadrature itself -- a census found 169
rectangle-form sites across 55 files -- and the choice was invisible at each one. These tests pin
the weights to the grid's own geometry rather than to a formula a caller remembers, so a future
cell-centred grid changes the weights and no caller changes at all.

Each test states the mutation that reddens it, because a weight test that passes under the wrong
weights pins nothing.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.geometry.boundary.invariants import mass_drift


def _grid(n=21, dim=1, hi=1.0):
    return TensorProductGrid(
        bounds=[(0.0, hi)] * dim,
        Nx_points=[n] * dim,
        boundary_conditions=no_flux_bc(dimension=dim),
    )


class TestWeightsAreTheGeometry:
    def test_end_nodes_own_half_a_cell(self):
        """The whole of #2145 in one assertion. Mutation: `w[0] = x[1]-x[0]` reddens it."""
        g = _grid(n=5)
        w = g.quadrature_weights()
        h = 0.25
        assert w[0] == pytest.approx(h / 2)
        assert w[-1] == pytest.approx(h / 2)
        assert np.allclose(w[1:-1], h)

    def test_the_weights_sum_to_the_interval(self):
        """An external oracle, not a restatement: the measure of the domain is its length.

        `sum(m)*dx` fails this by exactly one cell -- 5 nodes at h=0.25 give 1.25, not 1.0 -- which
        is the 3.5%-on-a-fixture error in its clearest form.
        """
        g = _grid(n=5)
        assert g.quadrature_weights().sum() == pytest.approx(1.0)
        assert np.full(5, 0.25).sum() == pytest.approx(1.25)  # the rectangle rule, for contrast

    def test_a_non_uniform_grid_gets_its_own_weights(self):
        """Written on coordinates, not on a single dx, so a graded grid is not a special case.

        Mutation: any formula using `spacing[0]` reddens this, because a graded grid has no single
        spacing -- `spacing` is None and the widest cell here is 169x the narrowest (measured).
        """
        xs = np.linspace(0.0, 1.0, 9) ** 3.0
        g = TensorProductGrid(
            bounds=[(0.0, 1.0)],
            Nx_points=[9],
            spacing_type="custom",
            custom_coordinates=[xs],
            boundary_conditions=no_flux_bc(dimension=1),
        )
        assert not g.is_uniform
        w = g.quadrature_weights()
        assert w.sum() == pytest.approx(1.0)
        assert np.diff(xs).max() / np.diff(xs).min() > 20  # the control: it really is graded (169.0)

    def test_a_single_node_axis_is_refused(self):
        """A one-node axis has no measure. Returning 0.0 or dx would both be inventions.

        The refusal stands; what independent review caught is that it reached callers as a bare
        `ValueError` naming neither the geometry nor `m_initial`, and that `MFGProblem` used to
        CONSTRUCT on such a grid. Both consequences are pinned below.
        """
        g = _grid(n=2)
        assert g.quadrature_weights().sum() == pytest.approx(1.0)  # two nodes is the minimum
        with pytest.raises(ValueError, match="at least two"):
            TensorProductGrid(
                bounds=[(0.0, 1.0)],
                Nx_points=[1],
                boundary_conditions=no_flux_bc(dimension=1),
            ).quadrature_weights()


class TestIntegrate:
    def test_it_integrates_a_linear_field_exactly(self):
        """The trapezoid is exact on linears, so this is an external oracle rather than a
        comparison against another implementation of the same rule."""
        g = _grid(n=7)
        x = np.asarray(g.coordinates[0])
        assert g.integrate(3.0 * x + 1.0) == pytest.approx(3.0 / 2 + 1.0)

    def test_a_corner_owns_the_product_of_half_cells(self):
        """nD is the tensor product, and the corner is where a per-axis rule would go wrong."""
        g = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 2.0)],
            Nx_points=[5, 5],
            boundary_conditions=no_flux_bc(dimension=2),
        )
        wx, wy = g.quadrature_weights(0), g.quadrature_weights(1)
        assert wx[0] * wy[0] == pytest.approx((0.25 / 2) * (0.5 / 2))
        assert g.integrate(np.ones((5, 5))) == pytest.approx(2.0)  # the area, exactly

    def test_a_time_history_reduces_only_the_trailing_axes(self):
        """What every mass check wants: one value per time row."""
        g = _grid(n=5)
        m = 1.0 + 0.4 * np.cos(2 * np.pi * np.asarray(g.coordinates[0]))
        out = g.integrate(np.vstack([m, 2 * m, 3 * m]))
        assert out.shape == (3,)
        assert out == pytest.approx([1.0, 2.0, 3.0])

    def test_a_shape_that_is_not_this_grid_is_refused(self):
        """Silently broadcasting a wrong-shaped field is how a mass number stops describing the
        solve it is reported for."""
        g = TensorProductGrid(
            bounds=[(0.0, 1.0)] * 2,
            Nx_points=[5, 5],
            boundary_conditions=no_flux_bc(dimension=2),
        )
        with pytest.raises(ValueError, match="do not match the grid"):
            g.integrate(np.ones(5))


class TestMassDriftUsesTheOwner:
    def test_the_grid_path_and_the_axis_path_agree_in_1d(self):
        """Five call sites pass an axis; they must not change meaning. Mutation: give the grid
        rectangle weights and this splits."""
        g = _grid(n=21)
        x = np.asarray(g.coordinates[0])
        m = 1.0 + 0.4 * np.cos(2 * np.pi * x)
        field = np.vstack([m, 1.01 * m])
        assert mass_drift(field, g) == pytest.approx(mass_drift(field, x))
        assert mass_drift(field, g) == pytest.approx(0.01)

    def test_it_works_in_2d_through_the_grid(self):
        """#2144: this raised `TypeError` on any n-D field, so `bc_residual` could not evaluate a
        no-flux FP residual in 2-D at all."""
        g = TensorProductGrid(
            bounds=[(0.0, 1.0)] * 2,
            Nx_points=[11, 11],
            boundary_conditions=no_flux_bc(dimension=2),
        )
        m = np.ones((11, 11))
        assert mass_drift(np.stack([m, 1.05 * m]), g) == pytest.approx(0.05)

    def test_an_nd_field_with_axis_coordinates_is_refused_by_name(self):
        """The old failure was `TypeError: only 0-dimensional arrays can be converted`, which names
        neither the cause nor the remedy."""
        g = TensorProductGrid(
            bounds=[(0.0, 1.0)] * 2,
            Nx_points=[11, 11],
            boundary_conditions=no_flux_bc(dimension=2),
        )
        m = np.ones((11, 11))
        with pytest.raises(ValueError, match="pass the grid instead"):
            mass_drift(np.stack([m, m]), np.asarray(g.coordinates[0]))

    def test_a_zero_initial_mass_is_refused_and_so_is_a_non_finite_one(self):
        """`initial == 0.0` was an exact float comparison, so a denormal passed it and the function
        returned 1e+300 instead of refusing (#2144)."""
        g = _grid(n=5)
        z = np.zeros(5)
        with pytest.raises(ValueError, match="undefined or meaningless"):
            mass_drift(np.vstack([z, np.ones(5)]), g)
        with pytest.raises(ValueError, match="undefined or meaningless"):
            mass_drift(np.vstack([np.full(5, np.nan), np.ones(5)]), g)


class TestWhatTheRefusalDoesToCallers:
    """#2145 made `integrate` refuse a measureless geometry. Two callers had to learn to say so."""

    def test_mfg_problem_says_what_to_do_about_a_one_node_grid(self):
        """The diagnostic must name the geometry and the remedy, not just the arithmetic.

        Mutation: drop the re-raise in `_measure_initial_density` and the message becomes
        "axis has 1 node(s); a measure needs at least two" -- true, and useless to someone who
        wrote `Nx_points=[1]` three files away.
        """
        from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
        from mfgarchon.core.mfg_components import MFGComponents
        from mfgarchon.core.mfg_problem import MFGProblem

        with pytest.raises(ValueError, match=r"cannot measure m_initial.*at least two points"):
            MFGProblem(
                geometry=TensorProductGrid(
                    bounds=[(0.0, 1.0)], Nx_points=[1], boundary_conditions=no_flux_bc(dimension=1)
                ),
                Nt=2,
                T=0.1,
                sigma=0.1,
                components=MFGComponents(
                    m_initial=lambda x: np.ones_like(np.asarray(x, dtype=float)),
                    u_terminal=lambda x: 0.0,
                    hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
                ),
            )

    def test_the_gate_is_integrate_and_the_non_positive_raise_still_escapes(self):
        """Two outcomes through one `try`, separated by structure rather than by an exception class.

        The gate on this block used to be `geometry.volume_element()` -- a different method from the
        one that supplies the number, and present on exactly the class that has `integrate`, so it
        was a spelling of "is this a TensorProductGrid" (#2157). It is now `integrate` itself, which
        is what `MFGProblem` has always gated on.

        The `except` used to be wide enough that the deliberate non-positive-mass `raise ValueError`
        sat inside it, and a private `_MassNotMeasurableError` existed only to climb out. The try is
        now narrowed to the `integrate` call and the raise lives outside it, so the sentinel is gone.
        Both outcomes are exercised here against a real completed solve:

        Mutation -- widen the `except` to `except Exception`, or move the raise back inside the try,
        and the second half stops raising and returns `None`.
        """
        from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
        from mfgarchon.core.mfg_components import MFGComponents
        from mfgarchon.core.mfg_problem import MFGProblem
        from mfgarchon.types import NumericalScheme

        def _fresh():
            return MFGProblem(
                geometry=TensorProductGrid(
                    bounds=[(0.0, 1.0)], Nx_points=[9], boundary_conditions=no_flux_bc(dimension=1)
                ),
                Nt=3,
                T=0.05,
                sigma=0.3,
                components=MFGComponents(
                    m_initial=lambda x: np.exp(-30.0 * (np.asarray(x, dtype=float) - 0.5) ** 2),
                    u_terminal=lambda x: 0.0,
                    hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
                ),
            )

        # Control: the unpatched fixture reports a NUMBER. Without this the two halves below would
        # both pass on a solve that never reached the block at all.
        control = _fresh().solve(scheme=NumericalScheme.FDM_UPWIND, max_iterations=2, verbose=False)
        assert control.mass_conservation_error is not None, (
            "the gate refused a plain TensorProductGrid -- the two halves below prove nothing"
        )

        def _refuses(_field):
            raise ValueError("this geometry has no measure")

        refusing = _fresh()
        refusing.geometry.integrate = _refuses
        result = refusing.solve(scheme=NumericalScheme.FDM_UPWIND, max_iterations=2, verbose=False)
        assert result.mass_conservation_error is None, (
            "a geometry that cannot integrate must report 'not measured', not a fabricated number"
        )
        assert result.M is not None, "and the solve itself must still return its result"

        def _reports_zero_mass(field):
            return np.zeros(np.asarray(field, dtype=float).shape[0])

        wrong = _fresh()
        wrong.geometry.integrate = _reports_zero_mass
        with pytest.raises(ValueError, match="non-positive or non-finite total mass"):
            wrong.solve(scheme=NumericalScheme.FDM_UPWIND, max_iterations=2, verbose=False)
