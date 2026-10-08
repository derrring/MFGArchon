"""Two cost inputs that no solver read are refused at construction, naming the channel that reaches H (#2554).

``MFGComponents(potential_func=V)`` was stored as ``problem.f_potential`` and ``MFGProblem(gamma=)`` as
``problem.gamma``, and nothing added either to H. Measured at 37ca5651 on the default HJB-FDM solve (41 points on
[0, 1], no-flux, T = 0.5, Nt = 20): V = 2 through ``potential_func`` moved u(0) by 0.0, where the same V through
``SeparableHamiltonian(potential=)`` moved it by 1.0 = V T; ``gamma`` 5 against 0, at m = 2, moved it by 0.0.

The refusal for the legacy keywords prints a migration guide, and that guide is advice a user will run: it is run here,
as printed, and must build a problem.
"""

from __future__ import annotations

import textwrap

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


def test_gamma_is_refused_naming_the_coupling_channel():
    with pytest.raises(TypeError, match=r"MFGProblem\(gamma=\.\.\.\) is retired \(#2554\).*coupling="):
        _problem(gamma=5.0)


def test_potential_func_is_refused_naming_the_potential_channel():
    with pytest.raises(TypeError, match=r"MFGComponents\(potential_func=\.\.\.\) is retired \(#2554\).*potential="):
        MFGComponents(hamiltonian=_hamiltonian(), potential_func=lambda t, x: 2.0)


def test_a_sixth_positional_argument_is_refused():
    """``potential_func`` was the sixth field; a sixth positional argument must not slide into the next one."""
    with pytest.raises(TypeError, match=r"at most 5 positional arguments.*potential_func"):
        MFGComponents(_hamiltonian(), None, lambda x: 1.0, lambda x: 0.0, None, lambda t, x: 2.0)


@pytest.mark.parametrize(
    ("keyword", "row"),
    [
        ("hamiltonian", r"'hamiltonian' -> Model\(hamiltonian=\.\.\.\)"),
        ("dH_dm", r"'dH_dm' -> the Hamiltonian's coupling_dm=, which takes f'\(m\); H carries -f"),
        ("dH_dp", r"'dH_dp' -> the Hamiltonian's control_cost=, whose dp\(\) is dH/dp"),
        ("running_cost", r"'running_cost' -> the Hamiltonian's potential= \(a cost of x\) or coupling="),
        ("potential", r"'potential' -> SeparableHamiltonian\(potential=\.\.\.\)"),
    ],
)
def test_each_retired_keyword_names_the_api_that_replaces_it(keyword, row):
    """Each row used to name an ``MFGComponents.hamiltonian_*`` field that does not exist, or the refused
    ``potential_func``. ``hamiltonian`` is a named parameter of ``MFGProblem``, so its row is reached through
    ``components.parameters``; the validator the constructor calls is called directly for every row."""
    with pytest.raises(ValueError, match=row) as refusal:
        _problem()._validate_kwargs({keyword: lambda *args: 0.0})
    assert "hamiltonian_func" not in str(refusal.value)
    assert "potential_func" not in str(refusal.value)


def _guide(message: str) -> str:
    """The code block a refusal message prints, dedented: from its first import to the line before ``See:``."""
    lines = message.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("  from mfgarchon"))
    end = next(i for i, line in enumerate(lines) if line.startswith("See:"))
    return textwrap.dedent("\n".join(lines[start:end]))


@pytest.mark.parametrize(
    "refused",
    [{"dH_dm": lambda m: m}, {"xmin": 0.0}],
    ids=["hamiltonian_guide", "geometry_guide"],
)
def test_the_advice_a_refused_keyword_prints_constructs(refused):
    """Run the code block the refusal prints, with its placeholders bound: it must build a problem. Until #2554
    the Hamiltonian guide built MFGComponents(hamiltonian_func=...), which raises TypeError, and the geometry
    guide built MFGProblem(geometry=..., T=..., Nt=..., volatility=...), which raises for a missing u_terminal."""
    with pytest.raises(ValueError, match=r"Deprecated kwargs detected") as refusal:
        _problem(**refused)
    namespace = {
        "my_potential": lambda t, x: 0.0,
        "my_coupling": lambda m: m**2,
        "my_coupling_dm": lambda m: 2 * m,
        "my_hamiltonian": _hamiltonian(),
        "my_m0": lambda x: 1.0 + 0.0 * x,
        "my_uT": lambda x: 0.0 * x,
        "my_geometry": _grid(),
        "xmin": 0.0,
        "xmax": 1.0,
        "Nx": 10,
        "T": 0.5,
        "Nt": 5,
        "sigma": 0.5,
    }
    exec(_guide(str(refusal.value)), namespace)
    assert isinstance(namespace["problem"], MFGProblem)
