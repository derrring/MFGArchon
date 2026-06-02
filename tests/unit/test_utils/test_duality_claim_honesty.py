"""check_solver_duality reports the FDM dual pairing honestly.

The default FDM factory pair (gradient_upwind HJB + divergence_upwind FP) is
structure-preserving (both mass-conservative, consistent with the continuous
adjoint) but the FP operator is assembled independently and is NOT a bit-exact
discrete transpose of the HJB operator -- there isn't even a single ``L_HJB``
(the velocity-advection and the linearized-Jacobian operators differ). The exact
``L_FP = L_HJB^T`` is opt-in via the iterator's ``adjoint_mode="jacobian_transpose"``.
SL (splatting = transpose of interpolation, #708) and MESHLESS_GALERKIN (Galerkin
MLS, #1131) are exact transposes by construction, so their message is unchanged.
"""

from __future__ import annotations

import numpy as np

from mfgarchon import MFGProblem
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_problem import MFGComponents
from mfgarchon.factory.scheme_factory import create_paired_solvers
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.types.schemes import NumericalScheme
from mfgarchon.utils import check_solver_duality


def _problem(n=21):
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[n], boundary_conditions=no_flux_bc(dimension=1))
    H = SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0))
    comp = MFGComponents(hamiltonian=H, m_initial=lambda x: np.ones_like(x), u_terminal=lambda x: 0.5 * (x - 0.5) ** 2)
    return MFGProblem(geometry=grid, components=comp, T=0.5, Nt=8, sigma=0.3, coupling_coefficient=0.5)


def test_fdm_duality_message_does_not_claim_bare_exact_transpose():
    """FDM: still a valid discrete dual pair, but the message must flag that exactness is
    opt-in, not assert an unconditional exact transpose."""
    prob = _problem()
    hjb, fp = create_paired_solvers(prob, NumericalScheme.FDM_UPWIND)
    result = check_solver_duality(hjb, fp, warn_on_mismatch=False)
    msg = result.message
    assert "adjoint_mode" in msg, f"FDM message should flag the opt-in exact mode: {msg!r}"
    assert "jacobian_transpose" in msg, f"FDM message should name the exact mode: {msg!r}"
    # It must not present a bare, unconditional 'L_FP = L_HJB^T' with no caveat.
    assert "structure-preserving" in msg, f"FDM message should say structure-preserving: {msg!r}"


def test_sl_duality_message_keeps_exact_transpose_claim():
    """SL splatting is the exact transpose of interpolation (#708); its message is unchanged."""
    prob = _problem()
    hjb, fp = create_paired_solvers(prob, NumericalScheme.SL_LINEAR)
    result = check_solver_duality(hjb, fp, warn_on_mismatch=False)
    assert "L_FP = L_HJB^T" in result.message, f"SL message should keep the exact claim: {result.message!r}"


def test_factory_fdm_docstring_is_honest():
    """The _create_fdm_pair docstring no longer claims an unconditional exact transpose."""
    from mfgarchon.factory import scheme_factory

    doc = scheme_factory._create_fdm_pair.__doc__
    assert "L_FP = L_HJB^T exactly" not in doc
    assert "adjoint_mode" in doc
    assert "structure-preserving" in doc
