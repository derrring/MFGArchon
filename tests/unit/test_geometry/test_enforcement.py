"""BC value enforcement, in a tier that runs (Issue #1820).

`mfgarchon/geometry/boundary/enforcement.py` had no live test of any kind. Its only assertions
lived in an `if __name__ == "__main__":` block, the same dead-block shape #1736 found in
`bc_utils.py`, and they **pinned the wrong periodic convention**: `field[0] = field[-2]` and
`field[-1] = field[1]`.

That is the halo form -- correct for an array carrying one ghost cell per side, where index 0 and
-1 duplicate the opposite interior. `TensorProductGrid` builds `np.linspace(x_min, x_max, N)`, so
index 0 and -1 are real nodes and the SAME physical point: period `L`, not `L + dx`. Applied there,
the halo form moves both endpoints one cell in opposite directions. Measured on `sin(2 pi x)` with
N=21: seam `2.4e-16` became `6.2e-01`, and `InterpolationApplicator.enforce_values` runs it after
every semi-Lagrangian timestep.

The convention is what these tests pin, not the formula: a rewrite that keeps the formula and
changes the grid is equally wrong.
"""

import pytest

import numpy as np

from mfgarchon.geometry.boundary.enforcement import (
    enforce_dirichlet_value_nd,
    enforce_neumann_value_nd,
    enforce_periodic_value_nd,
)


class TestPeriodicIsAnIdentification:
    """x_min and x_max are one degree of freedom the scheme computed twice."""

    def test_the_two_endpoints_end_up_equal(self):
        field = np.array([0.0, 1.0, 2.0, 3.0, 0.5])
        enforce_periodic_value_nd(field, axis=0)
        assert field[0] == field[-1], "the endpoints are the same physical point and must agree"

    def test_it_is_the_mean_and_privileges_neither_end(self):
        """Any rule taking one side would silently discard the other's discretisation error."""
        field = np.array([0.0, 1.0, 2.0, 3.0, 0.5])
        enforce_periodic_value_nd(field, axis=0)
        assert field[0] == pytest.approx(0.25)

    def test_it_does_not_reach_for_the_interior(self):
        """The halo form -- field[0] = field[-2], field[-1] = field[1] -- is the #1820 defect.

        Constructed so the two rules disagree: under the halo form the endpoints would become
        3.0 and 1.0, which are both interior values and are not even equal to each other.
        """
        field = np.array([0.0, 1.0, 2.0, 3.0, 0.5])
        enforce_periodic_value_nd(field, axis=0)
        assert field[0] not in (3.0, 1.0), "endpoint was taken from the interior (halo convention)"
        assert field[-1] not in (3.0, 1.0), "endpoint was taken from the interior (halo convention)"

    def test_an_already_periodic_field_is_left_alone(self):
        """The property that actually matters, and the one the halo form destroyed.

        Enforcement must be a projection: applying it to a field already in the periodic
        subspace changes nothing.
        """
        x = np.linspace(0.0, 1.0, 21)
        field = np.sin(2 * np.pi * x)
        before = field.copy()
        enforce_periodic_value_nd(field, axis=0)
        np.testing.assert_allclose(field, before, atol=1e-15)

    def test_it_is_idempotent(self):
        """`enforce_values` calls it twice per axis, once for each side."""
        field = np.array([0.0, 1.0, 2.0, 3.0, 0.5])
        enforce_periodic_value_nd(field, axis=0)
        once = field.copy()
        enforce_periodic_value_nd(field, axis=0)
        np.testing.assert_allclose(field, once, atol=1e-15)

    @pytest.mark.parametrize("axis", [0, 1])
    def test_it_identifies_only_the_named_axis_in_2d(self, axis):
        field = np.arange(20.0).reshape(4, 5)
        enforce_periodic_value_nd(field, axis=axis)
        if axis == 0:
            np.testing.assert_allclose(field[0, :], field[-1, :])
            assert not np.allclose(field[:, 0], field[:, -1]), "the other axis must be untouched"
        else:
            np.testing.assert_allclose(field[:, 0], field[:, -1])
            assert not np.allclose(field[0, :], field[-1, :]), "the other axis must be untouched"


class TestNeumannAndDirichlet:
    """The assertions the dead block carried, in a tier that runs."""

    def test_neumann_second_order_extrapolates(self):
        field = np.array([0.9, 1.0, 1.1, 1.2, 1.3])
        enforce_neumann_value_nd(field, axis=0, side="min", order=2)
        assert field[0] == pytest.approx((4.0 * 1.0 - 1.1) / 3.0)

    def test_neumann_first_order_copies_the_neighbour(self):
        field = np.array([0.9, 1.0, 1.1, 1.2, 1.3])
        enforce_neumann_value_nd(field, axis=0, side="min", order=1)
        assert field[0] == 1.0

    def test_neumann_honours_a_non_zero_gradient(self):
        """The `else` branch: g != 0 is a different formula, not a scaled version of g = 0."""
        field = np.array([0.0, 1.0, 1.1, 1.2, 0.0])
        enforce_neumann_value_nd(field, axis=0, side="min", grad_value=2.0, spacing=0.1)
        assert field[0] == pytest.approx(1.0 - 2.0 * 0.1)

    def test_dirichlet_assigns_each_side_independently(self):
        field = np.array([0.9, 1.0, 1.1, 1.2, 1.3])
        enforce_dirichlet_value_nd(field, axis=0, side="min", value=0.0)
        enforce_dirichlet_value_nd(field, axis=0, side="max", value=5.0)
        assert field[0] == 0.0
        assert field[-1] == 5.0
