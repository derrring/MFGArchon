"""Two cost inputs that no solver read are refused at construction (#2554).

``MFGComponents(potential_func=V)`` was stored as ``problem.f_potential`` and ``MFGProblem(gamma=)`` as
``problem.gamma``, and nothing added either to H. Measured at 37ca5651 on the default HJB-FDM solve (41 points on
[0, 1], no-flux, T = 0.5, Nt = 20): V = 2 through ``potential_func`` moved u(0) by 0.0, where the same V through
``SeparableHamiltonian(potential=)`` moved it by 1.0 = V T; ``gamma`` 5 against 0, at m = 2, moved it by 0.0.

Both are deleted now (the removal before 1.0): passing either is Python's own ``TypeError``.
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


def _problem(**extra) -> MFGProblem:
    return MFGProblem(
        model=Model(hamiltonian=_hamiltonian(), volatility=0.5),
        domain=_grid(),
        conditions=Conditions(m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0, T=0.5),
        Nt=5,
        **extra,
    )


def test_gamma_is_refused():
    """``MFGProblem`` takes no ``**kwargs`` now (the removal before 1.0), so ``gamma=`` is Python's own TypeError."""
    with pytest.raises(TypeError, match=r"unexpected keyword argument 'gamma'"):
        _problem(gamma=5.0)


def test_potential_func_is_refused():
    with pytest.raises(TypeError, match=r"unexpected keyword argument 'potential_func'"):
        MFGComponents(hamiltonian=_hamiltonian(), potential_func=lambda t, x: 2.0)


def test_a_fifth_positional_argument_is_refused():
    """``u_final`` was the fifth field and ``potential_func`` the sixth; every field after ``u_terminal`` is keyword-only,
    so a positional argument there cannot slide into the next field."""
    with pytest.raises(TypeError, match=r"positional argument"):
        MFGComponents(_hamiltonian(), None, lambda x: 1.0, lambda x: 0.0, None)


@pytest.mark.parametrize(
    ("keyword", "row"),
    [
        ("hamiltonian", r"'hamiltonian' -> Model\(hamiltonian=\.\.\.\), or MFGComponents\(hamiltonian=\.\.\.\)"),
        ("dH_dm", r"'dH_dm' -> the Hamiltonian's coupling_dm=, which takes f'\(m\); H carries -f"),
        ("dH_dp", r"'dH_dp' -> the Hamiltonian's control_cost=, whose dp\(\) is dH/dp"),
        ("running_cost", r"'running_cost' -> the Hamiltonian's potential= \(a cost of x\) or coupling="),
        ("potential", r"'potential' -> SeparableHamiltonian\(potential=\.\.\.\)"),
        ("xmin", r"'xmin' -> geometry=TensorProductGrid"),
    ],
)
def test_a_retired_name_in_components_parameters_names_the_api_that_replaces_it(keyword, row):
    """``MFGComponents.parameters`` is a free-form dict, so a name it once carried into the constructor is refused by
    name rather than accepted in silence (``MFGProblem._RETIRED_PARAMETER_KEYS``)."""
    components = MFGComponents(
        hamiltonian=_hamiltonian(),
        m_initial=lambda x: 1.0,
        u_terminal=lambda x: 0.0,
        parameters={keyword: lambda *args: 0.0},
    )
    with pytest.raises(ValueError, match=row) as refusal:
        MFGProblem(geometry=_grid(), T=0.5, Nt=5, volatility=0.5, components=components)
    assert "hamiltonian_func" not in str(refusal.value)
    assert "potential_func" not in str(refusal.value)
