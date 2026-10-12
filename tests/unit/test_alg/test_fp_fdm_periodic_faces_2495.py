"""FP-FDM reads periodicity from the faces, not from how the BC is spelt (#2495).

ADMISSION (#2257). Class 1, two pins on one convention: a boundary condition means what its faces
say. A uniform `periodic_bc`, one PERIODIC segment per face, and an empty segment list over a
periodic `default_bc` are one boundary condition. Until #2495 FP-FDM wrapped only the first: its
periodic test was `bc.is_uniform and bc.type == "periodic"`, and every wall node of a non-uniform BC
went to the no-flux handler, so the other two solved as no-flux, bit for bit, with no error. HJB-FDM
reads its ghosts per face and already agreed.

FP-FDM wraps every axis or none -- a wall node is assembled by a handler that cannot wrap -- so a BC
periodic on some faces only is refused rather than solved as no-flux on every face. The faces are read
from the face owner (#2512 row B1), which places a periodic segment the same way for HJB-FDM's ghosts: one
placed by `normal_direction` is on the faces its normal selects, so x-periodic segments over a NO_FLUX
default are a channel and refused as one, and with no `default_bc` the y faces have no condition, which the
owner refuses (#1100). A BC that declares periodicity but shows it on no face is no-flux, as on main.

Mutations, each run through this file on a scratch copy of the tree:

- the wall routing back to `is_uniform` (`and not wraps_everywhere` removed): the three periodic cases
  fail, at 2.2e+01 against `periodic_bc`;
- `periodic_axis_span` back to the uniform test: the three periodic cases fail, at 4.8e+00;
- the refusal in `FPFDMSolver.__init__` removed: both construction cases fail;
- the refusal in `solve_timestep_full_nd`, where the wall handlers are dispatched, removed: both
  assembly cases fail;
- the refusal keyed on "not every face periodic" instead of "some face periodic, not all": the
  overridden-default case fails;
- the solver's dimension ignored in favour of the BC's own: the unbound case fails;
- (`refuse_unplaceable_segments`, which two mutations above exercised, was folded into the face owner by
  #2512 row B1; the `normal_direction` cases now reach the channel refusal and the unset-default refusal.)
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.alg.numerical.fp_solvers.fp_fdm_time_stepping import solve_fp_nd_full_system
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, no_flux_bc, periodic_bc

N, NT = 17, 15


def _faces(bc_type, axes="xy"):
    return [BCSegment(name=f"{a}_{s}", bc_type=bc_type, boundary=f"{a}_{s}") for a in axes for s in ("min", "max")]


def _problem(bc):
    grid = TensorProductGrid(bounds=[(0.0, 1.0)] * 2, Nx_points=[N, N], boundary_conditions=bc)
    problem = MFGProblem(
        model=Model(hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0)), volatility=0.3),
        domain=grid,
        conditions=Conditions(m_initial=lambda z: 1.0, u_terminal=lambda z: 0.0, T=0.3),
        Nt=NT,
    )
    X, Y = np.meshgrid(*grid.coordinates, indexing="ij")
    # Periodic data, off-centre so that a reflection at a wall changes the answer.
    m0 = 1 + 0.5 * np.cos(2 * np.pi * (X - 0.2)) * np.sin(2 * np.pi * Y)
    U = np.broadcast_to(0.5 * np.sin(2 * np.pi * X), (NT + 1, N, N)).copy()
    return problem, m0, U


def _solve(bc, handed=False):
    """Solve with ``bc`` on the grid, or -- ``handed`` -- handed to the solver over a periodic grid."""
    problem, m0, U = _problem(periodic_bc(dimension=2) if handed else bc)
    solver = FPFDMSolver(problem, boundary_conditions=bc) if handed else FPFDMSolver(problem)
    return solver.solve_fp_system(M_initial=m0, potential_field=U, show_progress=False)


@pytest.mark.parametrize(
    ("bc_factory", "handed", "means"),
    [
        pytest.param(
            lambda: BoundaryConditions(dimension=2, segments=_faces(BCType.PERIODIC)),
            False,
            "periodic",
            id="one-segment-per-face",
        ),
        pytest.param(
            lambda: BoundaryConditions(dimension=2, segments=[], default_bc=BCType.PERIODIC),
            False,
            "periodic",
            id="periodic-default",
        ),
        # Never bound to a grid, so it has no dimension of its own: the faces are read with the solver's.
        pytest.param(
            lambda: BoundaryConditions(segments=_faces(BCType.PERIODIC)),
            True,
            "periodic",
            id="one-segment-per-face-unbound",
        ),
        # Declares periodicity, shows it on no face: no-flux, as on main and as HJB-FDM reads it.
        pytest.param(
            lambda: BoundaryConditions(dimension=2, segments=_faces(BCType.NO_FLUX), default_bc=BCType.PERIODIC),
            False,
            "no_flux",
            id="periodic-default-overridden-everywhere",
        ),
    ],
)
def test_a_bc_is_read_from_its_faces_2495(bc_factory, handed, means):
    reference = _solve(periodic_bc(dimension=2) if means == "periodic" else no_flux_bc(dimension=2))
    other = _solve(no_flux_bc(dimension=2) if means == "periodic" else periodic_bc(dimension=2))
    solved = _solve(bc_factory(), handed=handed)
    assert np.array_equal(solved, reference), (
        f"FP-FDM solved a BC whose faces are all {means} {np.abs(solved - reference).max():.3e} away from "
        f"the uniform {means} BC, and {np.abs(solved - other).max():.3e} from the other one. 0.0 from the "
        "other one is a reader deciding by the BC's spelling rather than its faces (#2495)."
    )


def _channel():
    return BoundaryConditions(dimension=2, segments=_faces(BCType.NO_FLUX, "x") + _faces(BCType.PERIODIC, "y"))


def _periodic_by_normal(default_bc=BCType.NO_FLUX):
    """Periodic in x by `normal_direction`, which carries no `boundary`: the face owner places each segment on
    the face its normal selects, so this is a channel (x periodic, y over the default)."""
    return BoundaryConditions(
        dimension=2,
        default_bc=default_bc,
        segments=[
            BCSegment(name=f"x_{s}", bc_type=BCType.PERIODIC, normal_direction=np.array([sign, 0.0]))
            for s, sign in (("min", -1.0), ("max", 1.0))
        ],
    )


@pytest.mark.parametrize("entry", ["construction", "assembly"])
@pytest.mark.parametrize(
    ("bc_factory", "error", "refusal"),
    [
        pytest.param(_channel, NotImplementedError, "periodic on some faces", id="channel"),
        pytest.param(_periodic_by_normal, NotImplementedError, "periodic on some faces", id="periodic-by-normal-direction"),
        pytest.param(
            lambda: _periodic_by_normal(default_bc=None),
            ValueError,
            "default_bc was not\\s+specified",
            id="periodic-by-normal-no-default",
        ),
    ],
)
def test_a_bc_periodic_on_some_faces_only_is_refused_2495(bc_factory, error, refusal, entry):
    """FP-FDM cannot wrap one axis and wall another. A channel was solved as no-flux on every face, and so
    is periodic placed by `normal_direction` on the x faces: both are refused as channels. With no
    `default_bc` the y faces have no condition at all, which the face owner refuses (#1100)."""
    if entry == "construction":
        problem, _, _ = _problem(bc_factory())
        with pytest.raises(error, match=refusal):
            FPFDMSolver(problem)
        return
    problem, m0, U = _problem(no_flux_bc(dimension=2))
    with pytest.raises(error, match=refusal):
        solve_fp_nd_full_system(m0, U, problem, boundary_conditions=bc_factory(), show_progress=False)
