"""The FP gradient advection schemes are removed, and asking for one is refused (#2007).

`gradient_upwind` and `gradient_centered` discretized v.grad(m). That drops m*div(v) from div(v*m), so
with a non-constant drift they solved a different equation, and their wall imposed dm/dn = 0 instead
of J.n = 0: EOC 0.00 at a drifting wall, and EOC -0.007 with no wall at all, under a periodic
boundary (test_fp_mms_wall_order_1728.py). The maintainer ruled on 2026-10-04 to remove them rather than repair the wall. Their legacy aliases
"centered" and "upwind" went with them, and so did "flux" (the removal before 1.0): each is an unknown scheme now,
refused with the valid options listed. The two divergence schemes are the control that must still construct.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.alg.numerical.fp_solvers.fp_fdm_time_stepping import solve_fp_nd_full_system
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

NX, NT = 11, 4
_REMOVED = ("gradient_upwind", "gradient_centered", "upwind", "centered", "flux")
_VALID = r"Valid options: \['divergence_centered', 'divergence_upwind'\]"


def _problem():
    return MFGProblem(
        model=Model(hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0)), volatility=0.2),
        domain=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[NX], boundary_conditions=no_flux_bc(dimension=1)),
        conditions=Conditions(m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0, T=0.2),
        Nt=NT,
    )


@pytest.mark.parametrize("name", _REMOVED)
def test_a_removed_scheme_is_refused_with_the_valid_options(name):
    with pytest.raises(ValueError, match=rf"Unknown advection_scheme '{name}'\. {_VALID}"):
        FPFDMSolver(_problem(), advection_scheme=name)
    # The standalone time-stepping entry validates its own input too.
    with pytest.raises(ValueError, match=rf"Unknown advection_scheme '{name}'"):
        solve_fp_nd_full_system(np.ones(NX), np.zeros((NT + 1, NX)), _problem(), advection_scheme=name)


@pytest.mark.parametrize("name", ["divergence_upwind", "divergence_centered"])
def test_the_divergence_schemes_construct(name):
    assert FPFDMSolver(_problem(), advection_scheme=name).advection_scheme == name
