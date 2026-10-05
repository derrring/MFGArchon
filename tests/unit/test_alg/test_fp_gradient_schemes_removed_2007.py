"""The FP gradient advection schemes are removed, and asking for one names its replacement (#2007).

`gradient_upwind` and `gradient_centered` discretized v.grad(m). That drops m*div(v) from div(v*m), so
with a non-constant drift they solved a different equation, and their wall imposed dm/dn = 0 instead
of J.n = 0: EOC 0.00 at a drifting wall, and EOC -0.007 with no wall at all, under a periodic
boundary (test_fp_mms_wall_order_1728.py). The maintainer ruled on 2026-10-04 to remove them rather than repair the wall. Their legacy aliases
"centered" and "upwind" went with them; "flux" (-> divergence_upwind) is the control that must still
resolve.
"""

from __future__ import annotations

import warnings

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.alg.numerical.fp_solvers.fp_fdm_time_stepping import solve_fp_nd_full_system
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

NX, NT = 11, 4
_REMOVED = {
    "gradient_upwind": "divergence_upwind",
    "gradient_centered": "divergence_centered",
    "upwind": "divergence_upwind",
    "centered": "divergence_centered",
}


def _problem():
    return MFGProblem(
        model=Model(hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0)), volatility=0.2),
        domain=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[NX], boundary_conditions=no_flux_bc(dimension=1)),
        conditions=Conditions(m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0, T=0.2),
        Nt=NT,
    )


@pytest.mark.parametrize(("name", "replacement"), sorted(_REMOVED.items()))
def test_a_removed_scheme_is_refused_with_its_replacement(name, replacement):
    with pytest.raises(ValueError, match=rf"advection_scheme='{name}' was removed .*'{replacement}'"):
        FPFDMSolver(_problem(), advection_scheme=name)
    # The standalone time-stepping entry validates its own input too.
    with pytest.raises(ValueError, match=rf"advection_scheme='{name}' was removed"):
        solve_fp_nd_full_system(np.ones(NX), np.zeros((NT + 1, NX)), _problem(), advection_scheme=name)


def test_the_surviving_alias_still_resolves():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solver = FPFDMSolver(_problem(), advection_scheme="flux")
    assert solver.advection_scheme == "divergence_upwind"
    assert any(issubclass(w.category, DeprecationWarning) and "'flux' is deprecated" in str(w.message) for w in caught)
