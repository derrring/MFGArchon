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


def test_grids_with_equal_bounds_and_count_but_different_nodes_are_refused():
    """Same bounds, same Nx_points, different node coordinates is a different discretisation.

    The first version of ``_same_geometry`` compared the attribute names
    ``Nx_points`` / ``bounds`` / ``num_nodes``. Both are equal here, so the pair was accepted and
    the FP geometry silently discarded -- the exact failure this refusal exists to prevent,
    passing through the guard written to prevent it. A Chebyshev grid on the same interval with
    the same point count has dx wrong by several times near the endpoints.
    """
    import numpy as np

    from mfgarchon.core.mfg_problem import _same_geometry
    from mfgarchon.geometry import TensorProductGrid
    from mfgarchon.geometry.boundary import no_flux_bc

    bc = no_flux_bc(dimension=1)
    uniform = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=bc)
    chebyshev = TensorProductGrid(
        bounds=[(0.0, 1.0)],
        Nx_points=[11],
        boundary_conditions=bc,
        spacing_type="custom",
        custom_coordinates=[np.sort(0.5 - 0.5 * np.cos(np.linspace(0, np.pi, 11)))],
    )

    assert np.array_equal(uniform.bounds, chebyshev.bounds)
    assert list(uniform.Nx_points) == list(chebyshev.Nx_points)
    assert not np.allclose(uniform.get_collocation_points(), chebyshev.get_collocation_points())

    assert not _same_geometry(uniform, chebyshev)


def test_meshes_that_expose_only_bounds_are_refused_rather_than_assumed_equal():
    """A geometry that cannot produce its node set is refused, not accepted on one attribute.

    Two ``Mesh2D`` instances over the same rectangle with a tenfold difference in ``mesh_size``
    have neither ``Nx_points`` nor ``num_nodes``, so attribute-name comparison found only
    ``bounds``, matched it, and returned True. ``examples/advanced/geometry_advanced/
    dual_geometry_fem_mesh.py`` is exactly a ``Mesh2D``-as-``fp_geometry`` script.

    Fail-closed also covers the reason a naive node-set comparison is not enough: an ungenerated
    mesh RAISES from ``get_collocation_points()`` rather than returning points, so the comparison
    must treat "cannot obtain" as "refuse" rather than letting the exception escape.
    """
    import pytest

    from mfgarchon.core.mfg_problem import _node_set, _same_geometry
    from mfgarchon.geometry.meshes import Mesh2D

    fine = Mesh2D(domain_type="rectangle", bounds=(0.0, 1.0, 0.0, 1.0), mesh_size=0.05)
    coarse = Mesh2D(domain_type="rectangle", bounds=(0.0, 1.0, 0.0, 1.0), mesh_size=0.5)

    assert not hasattr(fine, "Nx_points")
    assert not hasattr(fine, "num_nodes")
    with pytest.raises(ValueError, match="not yet generated"):
        fine.get_collocation_points()

    assert _node_set(fine) is None
    assert not _same_geometry(fine, coarse)
    assert _same_geometry(fine, fine), "identity must still short-circuit"
