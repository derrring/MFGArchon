"""`problem.spatial_discretization` is refused on both construction paths, naming the grid's counts (#1889).

It held the interval count when the problem was built from `spatial_bounds=` and the point count when
it was built from `geometry=`, for the identical grid. #1888's initial-density normaliser read it under
the interval reading and started every `geometry=` problem `(n/(n-1))^d` heavy. Space is the domain's
(CONVENTIONS.md, Counts: intervals versus points): `problem.geometry.Nx` counts intervals and
`problem.geometry.Nx_points` counts points. That the two constructors describe one grid is pinned by
`test_initial_density_mass_1888.py::test_both_construction_paths_agree`.
"""

from __future__ import annotations

import warnings

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

HAMILTONIAN = SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0))


def M0(x):
    return np.exp(-30 * np.sum((np.asarray(x) - 0.5) ** 2, axis=-1))


def _via_bounds():
    # `spatial_discretization=` exists only on the legacy constructor, so this path warns that it is deprecated.
    components = MFGComponents(m_initial=M0, u_terminal=lambda x: 0.0, hamiltonian=HAMILTONIAN)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="initial density mass")
        return MFGProblem(
            spatial_bounds=[(0.0, 1.0), (0.0, 1.0)],
            spatial_discretization=[10, 10],
            Nt=4,
            T=0.2,
            volatility=0.4,
            components=components,
        )


def _via_geometry():
    grid = TensorProductGrid(bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[11, 11], boundary_conditions=no_flux_bc(2))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="initial density mass")
        return MFGProblem(
            model=Model(hamiltonian=HAMILTONIAN, volatility=0.4),
            domain=grid,
            conditions=Conditions(m_initial=M0, u_terminal=lambda x: 0.0, T=0.2),
            Nt=4,
        )


@pytest.mark.parametrize("build", [_via_bounds, _via_geometry], ids=["spatial_bounds", "geometry"])
def test_the_attribute_is_refused_naming_both_counts(build):
    problem = build()
    assert problem.geometry.Nx == [10, 10]
    assert problem.geometry.Nx_points == [11, 11]
    with pytest.raises(AttributeError, match=r"retired \(#1889\).*geometry\.Nx counts intervals.*geometry\.Nx_points"):
        _ = problem.spatial_discretization
