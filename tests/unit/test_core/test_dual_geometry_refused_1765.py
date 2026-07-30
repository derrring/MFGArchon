r"""Dual geometry is refused rather than silently ignored (#1765).

`MFGProblem(hjb_geometry=..., fp_geometry=...)` accepted two geometries, built
`self.geometry_projector`, and then discarded the distinction: `_init_geometry` sets
`self.geometry` from the HJB one, and that is the single attribute every solver reads.

    grep -rn "geometry_projector" mfgarchon/     -> only inside core/mfg_problem.py
    grep -rln "hjb_geometry\|fp_geometry" mfgarchon/alg/   -> nothing

Measured before the change: a 41-point HJB grid with an 11-point FP grid returned
a density whose spatial axis was 41, the HJB grid's rather than the FP grid's 11, and the FP
CFL log reported the HJB grid's `dx=0.025`. No error, no
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


def test_the_workaround_the_error_message_names_actually_works():
    """The refusal points users at `GeometryProjector`; that path must round-trip, not just import.

    An error message that names an action is only as good as the action. Asserting the projector
    is merely constructible does not test the advice -- this drives both directions and checks the
    shapes land on the target grids, which is the whole content of "do the projection by hand".
    """
    import numpy as np

    from mfgarchon.geometry import GeometryProjector

    projector = GeometryProjector(hjb_geometry=_grid(41), fp_geometry=_grid(11), projection_method="auto")

    on_hjb = np.linspace(0.0, 1.0, 41)
    on_fp = projector.project_hjb_to_fp(on_hjb)
    assert on_fp.shape == (11,), f"HJB->FP landed on {on_fp.shape}, not the 11-point FP grid"

    back = projector.project_fp_to_hjb(on_fp)
    assert back.shape == (41,), f"FP->HJB landed on {back.shape}, not the 41-point HJB grid"
    assert np.all(np.isfinite(back))


def test_the_same_object_passed_twice_is_accepted():
    """One object bound to both names is the unified path, and must still work."""
    g = _grid(21)
    assert _problem(g, g).geometry.Nx_points == [21]


def test_two_separately_built_geometries_are_refused_even_when_identical():
    """The comparison is identity, not equality -- deliberately, and this pins that.

    Three attempts at deciding whether two separately-constructed geometries describe the same
    discretisation each shipped a wrong answer: an attribute-name list accepted a 10x mesh-size
    difference, a node-set comparison crashed on `Hyperrectangle` and refused two identical
    `Hypersphere`, and `vars()` refused a grid that had merely been used. Identity cannot be
    fooled. Over-refusing costs the caller one edit; the silent wrong answer this prevents costs
    them a paper.
    """
    a, b = _grid(21), _grid(21)
    assert a is not b
    with pytest.raises(NotImplementedError, match="SAME geometry object"):
        _problem(a, b)


def test_the_refusal_names_two_routes_that_both_work():
    """An error message is only as good as the actions it names; both are executed here."""
    import numpy as np

    from mfgarchon.geometry import GeometryProjector

    with pytest.raises(NotImplementedError) as exc:
        _problem(_grid(41), _grid(11))
    message = str(exc.value)

    assert "geometry=" in message
    g = _grid(21)
    assert _problem(g, g).geometry.Nx_points == [21], "route 1 from the message must work"

    assert "GeometryProjector" in message
    projector = GeometryProjector(hjb_geometry=_grid(41), fp_geometry=_grid(11))
    assert projector.project_hjb_to_fp(np.zeros(41)).shape == (11,), "route 2 must work"
