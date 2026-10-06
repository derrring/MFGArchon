"""FP-FDM reads periodicity from the faces, not from how the BC is spelt (#2495).

ADMISSION (#2257). Class 1, two pins on one convention: a boundary condition means what its faces
say. A uniform `periodic_bc`, one PERIODIC segment per face, and an empty segment list over a
periodic `default_bc` are one boundary condition. Until #2495 FP-FDM wrapped only the first: its
periodic test was `bc.is_uniform and bc.type == "periodic"`, and every wall node of a non-uniform BC
went to the no-flux handler, so the other two solved as no-flux, bit for bit, with no error. HJB-FDM
reads its ghosts per face and already agreed.

FP-FDM wraps every axis or none -- a wall node is assembled by a handler that cannot wrap -- so a BC
periodic on some faces only is refused rather than solved as no-flux on every face.

Mutations, each run through this file on a scratch copy of the tree:

- the wall routing back to `is_uniform` (`and not wraps_everywhere` removed): both spellings fail,
  at 2.2e+01 against `periodic_bc`;
- `periodic_on_every_face` in `periodic_axis_span` back to the uniform test: both spellings fail,
  at 4.8e+00;
- the refusal in `FPFDMSolver.__init__` removed: the construction case fails;
- the refusal in `solve_timestep_full_nd`, where the wall handlers are dispatched, removed: the
  assembly case fails.
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


def _solve(bc):
    problem, m0, U = _problem(bc)
    return FPFDMSolver(problem).solve_fp_system(M_initial=m0, potential_field=U, show_progress=False)


@pytest.mark.parametrize(
    "bc_factory",
    [
        pytest.param(
            lambda: BoundaryConditions(dimension=2, segments=_faces(BCType.PERIODIC)), id="one-segment-per-face"
        ),
        pytest.param(
            lambda: BoundaryConditions(dimension=2, segments=[], default_bc=BCType.PERIODIC), id="periodic-default"
        ),
    ],
)
def test_a_periodic_bc_is_read_from_its_faces_2495(bc_factory):
    reference = _solve(periodic_bc(dimension=2))
    walled = _solve(no_flux_bc(dimension=2))
    solved = _solve(bc_factory())
    assert np.array_equal(solved, reference), (
        f"FP-FDM solved a periodic BC spelt per face {np.abs(solved - reference).max():.3e} away from "
        f"periodic_bc, and {np.abs(solved - walled).max():.3e} from no_flux_bc -- 0.0 from the second "
        "is the #2495 defect: a reader recognising periodicity only through a uniform BC's type."
    )


def _channel():
    return BoundaryConditions(dimension=2, segments=_faces(BCType.NO_FLUX, "x") + _faces(BCType.PERIODIC, "y"))


@pytest.mark.parametrize("entry", ["construction", "assembly"])
def test_a_bc_periodic_on_some_faces_only_is_refused_2495(entry):
    """A channel -- no-flux walls in x, periodic in y. FP-FDM cannot wrap one axis and wall the other,
    and solved this as no-flux on every face; HJB-FDM solves it as the channel it is."""
    if entry == "construction":
        problem, _, _ = _problem(_channel())
        with pytest.raises(NotImplementedError, match="periodic on some faces"):
            FPFDMSolver(problem)
        return
    problem, m0, U = _problem(no_flux_bc(dimension=2))
    with pytest.raises(NotImplementedError, match="periodic on some faces"):
        solve_fp_nd_full_system(m0, U, problem, boundary_conditions=_channel(), show_progress=False)
