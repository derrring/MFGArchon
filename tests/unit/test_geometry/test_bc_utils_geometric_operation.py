"""The BC-type to geometric-operation mapping, in a tier that runs (Issue #1736).

These assertions existed since at least 2026-03-26, in `bc_utils.py`'s
`if __name__ == "__main__":` block. They ran nowhere and could not have:

    $ python mfgarchon/geometry/boundary/bc_utils.py
    ImportError: attempted relative import with no known parent package
    $ python -m mfgarchon.geometry.boundary.bc_utils
    ModuleNotFoundError: No module named 'mfgarchon.geometry.boundary.factories'

`factories` has never existed at that path -- the constructors are in
`conditions.py`. So the block was dead under both invocations, not merely
uncollected, and the mutation sweep (#1701) measured the consequence: flipping the
`None` default from `"clamp"` to `"reflect"` killed **0 of 5,665** tests, while the
five sibling conventions killed 129, 34, 19, 5 and 5.

Why this mapping is worth watching rather than merely covering: it decides between
**absorbing and reflecting**, which are opposite mass behaviours. Getting it wrong on
an unlabelled boundary does not crash -- the solve completes and conserves the wrong
thing.

The block these came from is deleted rather than left beside them. Two copies of the
same assertions, one of which cannot run, is the state that produced this issue.
"""

import pytest

from mfgarchon.geometry.boundary.bc_utils import (
    bc_type_to_geometric_operation,
    get_bc_type_string,
)
from mfgarchon.geometry.boundary.conditions import dirichlet_bc, no_flux_bc, periodic_bc


class TestGeometricOperationMapping:
    """Every branch of `bc_type_to_geometric_operation`, including the default."""

    @pytest.mark.parametrize(
        ("bc_type", "expected"),
        [
            ("no_flux", "reflect"),
            ("neumann", "reflect"),
            ("robin", "reflect"),
            ("periodic", "periodic"),
            ("dirichlet", "clamp"),
            ("absorbing", "clamp"),
        ],
    )
    def test_each_declared_type_maps_to_its_operation(self, bc_type, expected):
        assert bc_type_to_geometric_operation(bc_type) == expected

    def test_absent_bc_defaults_to_clamp_not_reflect(self):
        """The branch the sweep found unwatched.

        Absorbing is the conservative fallback for a boundary whose type could not be
        determined: it removes mass that leaves the domain. Reflecting would silently
        retain it, and the solve would conserve a quantity the problem never declared.
        """
        assert bc_type_to_geometric_operation(None) == "clamp"

    def test_an_unrecognised_type_also_clamps(self):
        """Same conservative fallback, reached by a different route."""
        assert bc_type_to_geometric_operation("something_nobody_declared") == "clamp"

    def test_the_mapping_is_case_insensitive(self):
        assert bc_type_to_geometric_operation("NO_FLUX") == "reflect"
        assert bc_type_to_geometric_operation("Periodic") == "periodic"

    def test_only_three_operations_are_ever_returned(self):
        """Semi-Lagrangian foot placement dispatches on exactly these three."""
        produced = {
            bc_type_to_geometric_operation(t)
            for t in ("no_flux", "neumann", "robin", "periodic", "dirichlet", "absorbing", None, "junk")
        }
        assert produced == {"reflect", "periodic", "clamp"}


class TestBCTypeStringFromConditions:
    """The other half: a `BoundaryConditions` object to the string above."""

    @pytest.mark.parametrize(
        ("factory", "expected"),
        [
            (lambda: no_flux_bc(dimension=1), "no_flux"),
            (lambda: periodic_bc(dimension=1), "periodic"),
            (lambda: dirichlet_bc(dimension=1, value=0.0), "dirichlet"),
        ],
    )
    def test_each_factory_round_trips_to_its_type_string(self, factory, expected):
        assert get_bc_type_string(factory()) == expected

    def test_none_conditions_give_no_type_string(self):
        assert get_bc_type_string(None) is None

    @pytest.mark.parametrize(
        ("factory", "expected"),
        [
            (lambda: no_flux_bc(dimension=1), "reflect"),
            (lambda: periodic_bc(dimension=1), "periodic"),
            (lambda: dirichlet_bc(dimension=1, value=0.0), "clamp"),
        ],
    )
    def test_the_two_halves_compose(self, factory, expected):
        """What a solver actually does: conditions -> string -> operation."""
        assert bc_type_to_geometric_operation(get_bc_type_string(factory())) == expected

    def test_conditions_that_yield_no_string_still_clamp(self):
        """The composed path for an undeclared boundary -- the #1736 case end to end."""
        assert bc_type_to_geometric_operation(get_bc_type_string(None)) == "clamp"
