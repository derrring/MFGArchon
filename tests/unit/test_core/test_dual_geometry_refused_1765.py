r"""Dual geometry is refused rather than silently ignored (#1765).

`MFGProblem(hjb_geometry=..., fp_geometry=...)` accepted two geometries, built
`self.geometry_projector`, and then discarded the distinction: `_init_geometry` sets
`self.geometry` from the HJB one, and that is the single attribute every solver reads.

    grep -rn "geometry_projector" mfgarchon/     -> only inside core/mfg_problem.py
    grep -rln "hjb_geometry\|fp_geometry" mfgarchon/alg/   -> nothing

Measured before the change: a 41-point HJB grid with an 11-point FP grid returned
`M.shape == (11, 41)` and the FP CFL log reported the HJB grid's `dx=0.025`. No error, no
warning, and every downstream number computed on a grid the caller did not choose.

Refusing is the honest state until the projector is wired. Had the parameter never existed, the
call would have raised `TypeError`; a feature that accepts input and ignores it is worse.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_problem import MFGComponents, MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def _grid(n: int) -> TensorProductGrid:
    return TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[n], boundary_conditions=no_flux_bc(dimension=1))


def _components() -> MFGComponents:
    return MFGComponents(
        m_initial=lambda x: np.ones_like(np.asarray(x)),
        u_terminal=lambda x: 0.0,
        hamiltonian=SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=1.0),
            coupling=lambda m: m,
            coupling_dm=lambda m: 1.0,
        ),
    )


def _problem(hjb, fp) -> MFGProblem:
    return MFGProblem(hjb_geometry=hjb, fp_geometry=fp, Nt=5, T=0.2, sigma=0.3, components=_components())


def test_differing_geometries_are_refused():
    with pytest.raises(NotImplementedError, match="1765"):
        _problem(_grid(41), _grid(11))


def test_the_message_names_the_workaround():
    """A refusal that does not say what to do instead reads as a wall."""
    with pytest.raises(NotImplementedError) as exc:
        _problem(_grid(41), _grid(11))
    message = str(exc.value)
    assert "no solver reads it" in message
    assert "geometry=" in message
    assert "GeometryProjector" in message, "the projector still works standalone; say so"


def test_identical_geometries_are_still_accepted():
    """Two separately-constructed identical grids are equivalent to the unified path.

    The first version of the comparison required `Nx_points`, `bounds` AND `num_nodes` to be
    present on both, and a `TensorProductGrid` has no `num_nodes` -- so it rejected a pair it
    should have accepted. Compared on the attributes both objects actually have.
    """
    problem = _problem(_grid(41), _grid(41))
    assert problem.geometry.Nx_points == [41]

    same = _grid(21)
    assert _problem(same, same).geometry.Nx_points == [21]


def test_the_projector_itself_is_untouched():
    """`GeometryProjector` works standalone; it is the MFGProblem plumbing that is missing.

    Refusing at the problem constructor must not take the projector with it -- a user doing the
    projection by hand is doing the thing this refusal says to do.
    """
    from mfgarchon.geometry import GeometryProjector

    projector = GeometryProjector(hjb_geometry=_grid(41), fp_geometry=_grid(11), projection_method="auto")
    assert projector is not None
