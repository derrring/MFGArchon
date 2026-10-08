"""Two cost inputs that no solver read are refused at construction, naming the channel that reaches H (#2554).

``MFGComponents(potential_func=V)`` was stored as ``problem.f_potential`` and ``MFGProblem(gamma=)`` as
``problem.gamma``, and nothing added either to H. Measured at 37ca5651 on the default HJB-FDM solve (41 points on
[0, 1], no-flux, T = 0.5, Nt = 20): V = 2 through ``potential_func`` moved u(0) by 0.0, where the same V through
``SeparableHamiltonian(potential=)`` moved it by 1.0 = V T; ``gamma`` 5 against 0, at m = 2, moved it by 0.0.
"""

from __future__ import annotations

import pytest

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def _hamiltonian() -> SeparableHamiltonian:
    return SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0))


def _grid() -> TensorProductGrid:
    return TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=no_flux_bc(dimension=1))


def test_gamma_is_refused_naming_the_coupling_channel():
    with pytest.raises(TypeError, match=r"MFGProblem\(gamma=\.\.\.\) is retired \(#2554\).*coupling="):
        MFGProblem(
            model=Model(hamiltonian=_hamiltonian(), volatility=0.5),
            domain=_grid(),
            conditions=Conditions(m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0, T=0.5),
            Nt=5,
            gamma=5.0,
        )


def test_potential_func_is_refused_naming_the_potential_channel():
    with pytest.raises(TypeError, match=r"MFGComponents\(potential_func=\.\.\.\) is retired \(#2554\).*potential="):
        MFGComponents(hamiltonian=_hamiltonian(), potential_func=lambda t, x: 2.0)


def test_the_legacy_potential_keyword_points_at_the_hamiltonian():
    """``MFGProblem(potential=)`` was redirected to ``MFGComponents.potential_func``, the input refused above."""
    components = MFGComponents(hamiltonian=_hamiltonian(), m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0)
    with pytest.raises(ValueError, match=r"'potential' -> SeparableHamiltonian\(potential=\.\.\.\)"):
        MFGProblem(geometry=_grid(), components=components, T=0.5, Nt=5, potential=lambda t, x: 2.0)
