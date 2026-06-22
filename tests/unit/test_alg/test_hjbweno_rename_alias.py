#!/usr/bin/env python3
"""Issue #1426: HJBWenoSolver renamed to HJBWENOSolver (WENO is an acronym; matches the all-caps
HJBGFDMSolver / HJBFDMSolver siblings and PEP 8). The old name remains as a deprecated alias.

Pins: the new name is importable; the old name still works but warns; both construct the same class.
"""

import warnings

import pytest

import numpy as np

from mfgarchon.alg.numerical.hjb_solvers import HJBWENOSolver, HJBWenoSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def _problem():
    comp = MFGComponents(
        m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2),
        u_terminal=lambda x: 0.0,
        hamiltonian=SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=1.0),
            coupling=lambda m: m,
            coupling_dm=lambda m: 1.0,
        ),
    )
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31], boundary_conditions=no_flux_bc(dimension=1))
    return MFGProblem(geometry=grid, T=1.0, Nt=20, sigma=0.1, components=comp)


def test_new_name_constructs():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver = HJBWENOSolver(_problem(), weno_variant="weno5")
    assert isinstance(solver, HJBWENOSolver)


def test_deprecated_alias_warns_and_builds_new_class():
    """Old name still works (backward compat) but emits a DeprecationWarning and returns an
    instance of the new class."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.warns(DeprecationWarning, match="HJBWenoSolver"):
            solver = HJBWenoSolver(_problem(), weno_variant="weno5")
    assert isinstance(solver, HJBWENOSolver)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
