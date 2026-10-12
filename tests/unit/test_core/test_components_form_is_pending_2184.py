"""The components= form warns that it is planned to be replaced, not that it is deprecated (#2184).

User ruling, 2026-10-12, option A: the model/domain/conditions form is not fixed yet, so a caller of the
components= form is told that nothing needs to change now -- a ``PendingDeprecationWarning``, with no version
and no "use X". The message names the legacy form generically, "the components= form", and lists none of its
routes: #2184's 10-05 finding was the old message naming ``geometry=`` to a caller who had passed
``spatial_bounds=``, and a list of two routes still left out ``network=`` (review 1 of #2631). Each route is a
row here.

Red with the old category: putting ``DeprecationWarning`` back, and changing nothing else, fails every row on
the category assertion (measured on all three rows).
"""

from __future__ import annotations

import warnings

import pytest

from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

MESSAGE = (
    "The components= form of MFGProblem is planned to be replaced by the model/domain/conditions form, "
    "MFGProblem(model=, domain=, conditions=). Nothing needs to change now."
)


def _components() -> MFGComponents:
    return MFGComponents(
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
        m_initial=lambda x: 1.0,
        u_terminal=lambda x: 0.0,
    )


def _construct(route: str) -> None:
    if route == "geometry":
        grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[9], boundary_conditions=no_flux_bc(dimension=1))
        MFGProblem(components=_components(), T=0.5, Nt=4, volatility=0.3, geometry=grid)
    elif route == "spatial_bounds":
        MFGProblem(
            components=_components(),
            T=0.5,
            Nt=4,
            volatility=0.3,
            spatial_bounds=[(0.0, 1.0)],
            spatial_discretization=[8],
        )
    else:
        nx = pytest.importorskip("networkx")
        MFGProblem(network=nx.path_graph(5), components=_components(), T=1.0, Nt=10)


@pytest.mark.parametrize("route", ["geometry", "spatial_bounds", "network"])
def test_the_components_form_warns_pending_not_deprecated(route: str):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _construct(route)
    mine = [w for w in caught if "components= form" in str(w.message)]
    assert len(mine) == 1, [f"{w.category.__name__}: {str(w.message)[:80]}" for w in caught]
    assert mine[0].category is PendingDeprecationWarning, mine[0].category
    assert str(mine[0].message) == MESSAGE
    # No DeprecationWarning about the form itself (PendingDeprecationWarning is not a subclass of it).
    old = [w for w in caught if issubclass(w.category, DeprecationWarning) and "MFGProblem(geometry=" in str(w.message)]
    assert old == [], [str(w.message)[:80] for w in old]
