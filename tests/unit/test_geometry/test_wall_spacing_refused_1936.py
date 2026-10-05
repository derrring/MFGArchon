"""A wall formula that needs the grid spacing refuses when it has none, instead of using 1.0 (#1936).

`PreallocatedGhostBuffer` built without `spacing=` or `domain_bounds=` used dx = 1.0 at every site that
needs a spacing, so a Neumann value g was applied as g/h (#1904 measured 40 / 80 / 160 / 320 for g = 2).
`enforce_values` did the same for an axis its `spacing` had no entry for, and `GhostBuffer` defaulted
`dx` to 1.0. Each wall formula now refuses, and only where it reads the spacing: a pure mirror (zero flux,
no Robin coefficient), a Robin wall with beta = 0, and the polynomial extrapolation of Dirichlet data are the
same at every scale, so they still run without one. `GhostBuffer` cannot see inside its calculator, so a bounded
one refuses at construction without `dx`, whatever the calculator.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, dirichlet_bc, neumann_bc, no_flux_bc
from mfgarchon.geometry.boundary.applicator_fdm import (
    FDMApplicator,
    GhostBuffer,
    PreallocatedGhostBuffer,
    _write_wall_ghosts,
)
from mfgarchon.geometry.boundary.calculators import BoundedTopology, NeumannCalculator, PeriodicTopology

REFUSAL = r"needs the grid spacing.*#1936\)"
INTERIOR = np.array([0.0, 1.0, 4.0, 9.0, 16.0, 25.0])


def _ghosts(bc, **kwargs):
    buffer = PreallocatedGhostBuffer(interior_shape=INTERIOR.shape, boundary_conditions=bc, **kwargs)
    buffer.interior[:] = INTERIOR
    buffer.update_ghosts(time=0.0)
    return buffer.padded.copy()


def _per_face(bc_type, **values):
    segments = [BCSegment(name=side, bc_type=bc_type, boundary=f"x_{side}", **values) for side in ("min", "max")]
    return BoundaryConditions(segments=segments, dimension=1)


@pytest.mark.parametrize(
    "bc",
    [
        neumann_bc(value=2.0, dimension=1),
        _per_face(BCType.NEUMANN, value=2.0),
        BoundaryConditions(
            segments=[BCSegment(name="w", bc_type=BCType.ROBIN, alpha=1.0, beta=1.0, value=0.0)], dimension=1
        ),
        _per_face(BCType.ROBIN, alpha=1.0, beta=1.0, value=0.0),
    ],
    ids=["neumann-uniform", "neumann-per-face", "robin-uniform", "robin-per-face"],
)
def test_a_spacing_dependent_wall_refuses_without_a_spacing(bc):
    with pytest.raises(ValueError, match=REFUSAL):
        _ghosts(bc)
    # Control: the same condition with the spacing runs.
    assert np.all(np.isfinite(_ghosts(bc, spacing=0.2)))


@pytest.mark.parametrize(
    "bc",
    [
        no_flux_bc(dimension=1),
        neumann_bc(value=0.0, dimension=1),
        _per_face(BCType.NEUMANN, value=0.0),
        BoundaryConditions(
            segments=[BCSegment(name="w", bc_type=BCType.ROBIN, alpha=1.0, beta=0.0, value=3.0)], dimension=1
        ),
        _per_face(BCType.ROBIN, alpha=1.0, beta=0.0, value=3.0),
    ],
    ids=["no-flux", "neumann-zero", "neumann-zero-per-face", "robin-beta0-uniform", "robin-beta0-per-face"],
)
def test_a_spacing_free_wall_runs_without_a_spacing(bc):
    # The ghost does not read h, so it equals the one computed at any spacing.
    np.testing.assert_array_equal(_ghosts(bc), _ghosts(bc, spacing=0.37))


def test_an_array_condition_refuses_unless_every_point_is_spacing_free():
    # A 2-D wall of two points, the first beta = 0 and the second a Robin point that reads h.
    buf = np.zeros((6, 2))
    buf[1:-1] = np.outer(INTERIOR[:4], [1.0, 2.0])
    with pytest.raises(ValueError, match=REFUSAL):
        _write_wall_ghosts(buf, 0, "min", 1, None, 0.0, 1.0, np.array([0.0, 1.0]))
    # Control: with both points at beta = 0 nothing reads h.
    _write_wall_ghosts(buf, 0, "min", 1, None, 0.0, 1.0, np.array([0.0, 0.0]))


@pytest.mark.parametrize("flux", [2.0, -2.0])
def test_extrapolation_refuses_a_flux_and_keeps_dirichlet_scale_free(flux):
    with pytest.raises(ValueError, match=REFUSAL):
        _ghosts(neumann_bc(value=flux, dimension=1), order=3)
    # Dirichlet data fixes the fitted polynomial in x/h, so the ghosts cannot depend on h.
    without = _ghosts(dirichlet_bc(value=1.0, dimension=1), order=3)
    np.testing.assert_allclose(without, _ghosts(dirichlet_bc(value=1.0, dimension=1), order=3, spacing=0.37))


@pytest.mark.parametrize("spacing", [(0.2,), (0.2, 0.2, 0.2)], ids=["short", "long"])
def test_enforce_values_wants_one_spacing_per_axis(spacing):
    field = np.outer(INTERIOR, np.ones(4))
    with pytest.raises(ValueError, match=rf"spacing has {len(spacing)} entries for a 2-D field.*\(#1936\)"):
        FDMApplicator(dimension=2).enforce_values(field, neumann_bc(value=2.0, dimension=2), spacing=spacing)


def test_a_bounded_ghost_buffer_wants_its_spacing():
    with pytest.raises(ValueError, match=r"bounded GhostBuffer passes the grid spacing.*\(#1936\)"):
        GhostBuffer(BoundedTopology(1, (5,)), NeumannCalculator(2.0))
    with pytest.raises(ValueError, match=r"Grid spacing dimension 0 must match topology dimension 1"):
        GhostBuffer(BoundedTopology(1, (5,)), NeumannCalculator(2.0), dx=())
    # Control: a periodic buffer does not read the spacing.
    GhostBuffer(PeriodicTopology(1, (5,))).update()
