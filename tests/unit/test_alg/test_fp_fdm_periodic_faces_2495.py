"""FP-FDM reads periodicity from the faces, not from how the BC is spelt (#2495).

ADMISSION (#2257). Class 1, two pins on one convention: a boundary condition means what its faces
say. A uniform `periodic_bc`, one PERIODIC segment per face, and an empty segment list over a
periodic `default_bc` are one boundary condition. Until #2495 FP-FDM wrapped only the first: its
periodic test was `bc.is_uniform and bc.type == "periodic"`, and every wall node of a non-uniform BC
went to the no-flux handler, so the other two solved as no-flux, bit for bit, with no error. HJB-FDM
reads its ghosts per face and already agreed.

FP-FDM wraps every axis or none -- a wall node is assembled by a handler that cannot wrap -- so a BC
periodic on some faces only is refused rather than solved as no-flux on every face. So is a periodic
segment the face reader cannot place, in a mix of operations or with no `default_bc`: one with no
`boundary`, which that reader puts on every face, would otherwise be wrapped on faces it does not reach,
where HJB-FDM walls them over a NO_FLUX default, or finds no BC and raises without one. A BC that
declares periodicity but shows it on no face is no-flux, as on main.

Mutations, each run through this file on a scratch copy of the tree:

- the wall routing back to `is_uniform` (`and not wraps_everywhere` removed): the three periodic cases
  fail, at 2.2e+01 against `periodic_bc`;
- `periodic_axis_span` back to the uniform test: the three periodic cases fail, at 4.8e+00;
- the refusal in `FPFDMSolver.__init__` removed: both construction cases fail;
- the refusal in `solve_timestep_full_nd`, where the wall handlers are dispatched, removed: both
  assembly cases fail;
- `refuse_unplaceable_segments` left out of that refusal: both `normal_direction` cases fail;
- the refusal keyed on "not every face periodic" instead of "some face periodic, not all": the
  overridden-default case fails;
- the solver's dimension ignored in favour of the BC's own: the unbound case fails;
- `refuse_unplaceable_segments` exempting every single-operation BC, as its first version did, rather
  than only one with a `default_bc`: both no-default cases fail.
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
    """Periodic in x by `normal_direction`, which carries no `boundary`: the face reader puts it on every
    face, so read naively this BC is periodic everywhere -- while HJB-FDM walls y over a NO_FLUX default,
    and finds no BC for y and raises without one."""
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
    ("bc_factory", "refusal"),
    [
        pytest.param(_channel, "periodic on some faces", id="channel"),
        pytest.param(_periodic_by_normal, "have no `boundary`", id="periodic-by-normal-direction"),
        pytest.param(
            lambda: _periodic_by_normal(default_bc=None), "have no `boundary`", id="periodic-by-normal-no-default"
        ),
    ],
)
def test_a_bc_periodic_on_some_faces_only_is_refused_2495(bc_factory, refusal, entry):
    """FP-FDM cannot wrap one axis and wall another. A channel was solved as no-flux on every face; a
    periodic segment placed by `normal_direction` would be wrapped on every face, where HJB-FDM walls it.
    Both are refused, the second because the face reader cannot place its segments."""
    if entry == "construction":
        problem, _, _ = _problem(bc_factory())
        with pytest.raises(NotImplementedError, match=refusal):
            FPFDMSolver(problem)
        return
    problem, m0, U = _problem(no_flux_bc(dimension=2))
    with pytest.raises(NotImplementedError, match=refusal):
        solve_fp_nd_full_system(m0, U, problem, boundary_conditions=bc_factory(), show_progress=False)
