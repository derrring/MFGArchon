"""A particle at or past a corner of the box takes the owner's corner condition (#2512 row B1, stage B1d).

`ParticleApplicator.apply` and ``apply_with_flux_limits`` named the first face they found a particle on, x before y and
min before max, and asked the owner on that face alone, so the owner's corner rule (`BoundaryConditions._corner_choice`:
priority, then DIRICHLET before a natural condition) never saw a particle at a corner. With x periodic and an exit on y,
a particle past a corner the two share was wrapped, where the corner order absorbs it; with y periodic, x's exit was
already read first. Both methods now read each boundary particle at its nearest point of the step's box through the
contact loop's reader, `segment_at_box_contact` (#2733), at the owner's default tolerance (``SDF_BOUNDARY_TOL``), as
the first-face read asked the owner.

The particle solver reaches ``apply`` only on its periodic path: a segment-aware BC that declares a periodic face
(`FPParticleSolver._apply_boundary_conditions_segment_aware`). ``apply_with_flux_limits`` is reached only from
`FPParticleSolver._apply_boundary_conditions_with_flux_limits`, which has no library caller; C6 (#2550) decides both.

Held fixed:
- every segment at priority 0, placed by a face label, except the tolerance pin's ``region`` exit; no ``sdf_region``,
  ``region_name`` or ``normal_direction``;
- under ``corner_strategy="priority"``, every corner pairs a PERIODIC face with a DIRICHLET exit: no periodic/wall or
  exit/wall corner. Under a blending strategy, the owner refuses a corner whose faces differ, and the control is a
  wall/wall corner;
- constant values, and a BC that carries the step's box or no ``domain_bounds``.
"""

from __future__ import annotations

import itertools
import warnings

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.fp_solvers.fp_particle import FPParticleSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions
from mfgarchon.geometry.boundary.applicator_particle import ParticleApplicator

BOX = [(0.0, 1.0), (0.0, 1.0)]
AXES = "xy"
CORNERS = list(itertools.product((0.0, 1.0), repeat=2))
#: Where a particle is put near a corner. The applicator finds the boundary within 1e-10, so 5e-11 is inside it.
POSITIONS = {
    "beyond both faces": (0.05, 0.03),
    "beyond the x face, within the tolerance of the y face": (0.05, -5e-11),
    "within the tolerance inside both faces": (-5e-11, -5e-11),
}


def _bc(periodic_axis: int, exit_side: str) -> BoundaryConditions:
    """One axis periodic; on the other, a DIRICHLET exit on ``exit_side`` and NO_FLUX on the opposite face."""
    p, other = AXES[periodic_axis], AXES[1 - periodic_axis]
    wall_side = "min" if exit_side == "max" else "max"
    return BoundaryConditions(
        dimension=2,
        segments=[
            BCSegment(name=f"periodic_{p}_min", bc_type=BCType.PERIODIC, boundary=f"{p}_min"),
            BCSegment(name=f"periodic_{p}_max", bc_type=BCType.PERIODIC, boundary=f"{p}_max"),
            BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=0.0, boundary=f"{other}_{exit_side}"),
            BCSegment(name="wall", bc_type=BCType.NO_FLUX, boundary=f"{other}_{wall_side}"),
        ],
        domain_bounds=np.array(BOX),
    )


def _step(method: str, bc: BoundaryConditions, points, bounds=BOX):
    applicator = ParticleApplicator()
    points = np.array(points, dtype=float)
    if method == "apply":
        remaining, absorbed, _ = applicator.apply(points, bc, bounds)
    else:
        remaining, absorbed, _, _ = applicator.apply_with_flux_limits(points, bc, bounds, {})
    return remaining, absorbed


@pytest.mark.parametrize("method", ["apply", "apply_with_flux_limits"])
@pytest.mark.parametrize("periodic_axis", [0, 1], ids=["periodic_x", "periodic_y"])
@pytest.mark.parametrize("corner", CORNERS, ids=[f"corner_{int(x)}{int(y)}" for x, y in CORNERS])
@pytest.mark.parametrize("position", list(POSITIONS))
def test_a_particle_at_or_past_a_corner_takes_the_exit_the_corner_order_gives(method, periodic_axis, corner, position):
    """At every corner one face is periodic and the other the exit; the corner order puts DIRICHLET first, so the
    particle is absorbed. Under the first-face read the periodic_x rows wrapped it (x's face is read first)."""
    exit_side = "max" if corner[1 - periodic_axis] == 1.0 else "min"
    offset = np.array(POSITIONS[position])
    outward = np.where(np.array(corner) == 1.0, 1.0, -1.0)
    point = np.array(corner) + outward * offset
    _, absorbed = _step(method, _bc(periodic_axis, exit_side), [point])
    assert absorbed.tolist() == [True], (corner, point.tolist())


@pytest.mark.parametrize("method", ["apply", "apply_with_flux_limits"])
@pytest.mark.parametrize("strategy", ["average", "mollify"])
def test_a_blending_corner_strategy_refuses_a_corner_whose_faces_differ(method, strategy):
    """The other side of ``corner_strategy="priority"``: a blending strategy says nothing about which condition a corner
    takes, so where its faces impose different conditions the owner refuses it, as it does for the face families. The
    first-face read wrapped the particle in silence. Where the faces impose one condition there is nothing to choose."""
    bc = _bc(0, "max")
    bc.corner_strategy = strategy
    with pytest.raises(ValueError, match=r"corner shared by x_max, y_max.*Use corner_strategy='priority'"):
        _step(method, bc, [(1.05, 1.03)])
    walls = BoundaryConditions(
        dimension=2, segments=[], default_bc=BCType.NO_FLUX, corner_strategy=strategy, domain_bounds=np.array(BOX)
    )
    remaining, absorbed = _step(method, walls, [(1.05, 1.03)])
    assert absorbed.tolist() == [False]
    np.testing.assert_allclose(remaining, [[0.95, 0.97]])


@pytest.mark.parametrize("method", ["apply", "apply_with_flux_limits"])
@pytest.mark.parametrize("periodic_axis", [0, 1], ids=["periodic_x", "periodic_y"])
def test_a_particle_past_the_middle_of_a_face_takes_that_faces_condition(method, periodic_axis):
    """The control: away from a corner a particle takes its one face's condition, as before -- the exit absorbs, the
    wall reflects, and the periodic axis wraps."""
    p, other = periodic_axis, 1 - periodic_axis
    past = {}
    for axis, side in itertools.product((0, 1), ("min", "max")):
        point = [0.5, 0.5]
        point[axis] = 1.05 if side == "max" else -0.05
        past[(axis, side)] = point
    points = list(past.values())
    remaining, absorbed = _step(method, _bc(periodic_axis, "max"), points)
    assert absorbed.tolist() == [key == (other, "max") for key in past]
    kept = dict(zip([key for key in past if key != (other, "max")], remaining, strict=True))
    np.testing.assert_allclose(kept[(other, "min")][other], 0.05)  # reflected
    np.testing.assert_allclose(kept[(p, "min")][p], 0.95)  # wrapped
    np.testing.assert_allclose(kept[(p, "max")][p], 0.05)  # wrapped


@pytest.mark.parametrize("method", ["apply", "apply_with_flux_limits"])
def test_a_placement_holds_a_particle_within_the_owners_default_tolerance(method):
    """A ``region`` exit on part of x_max holds a particle past x_max within the owner's default tolerance of its ends,
    1e-8, as the first-face read's query did, and not 2e-8 past them. Read at the applicator's own 1e-10, the particle
    5e-9 past an end was reflected."""
    bc = BoundaryConditions(
        dimension=2,
        segments=[
            BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=0.0, boundary="x_max", region={"y": (0.4, 0.6)})
        ],
        default_bc=BCType.NO_FLUX,
        domain_bounds=np.array(BOX),
    )
    points = [(1.05, 0.5), (1.05, 0.6 + 5e-9), (1.05, 0.4 - 5e-9), (1.05, 0.6 + 2e-8), (1.05, 0.4 - 2e-8)]
    _, absorbed = _step(method, bc, points)
    assert absorbed.tolist() == [True, True, True, False, False]


@pytest.mark.parametrize("method", ["apply", "apply_with_flux_limits"])
def test_a_one_dimensional_face_named_left_and_right_reads_as_x_min_and_x_max(method):
    """The first-face read named 1-D faces "left" and "right"; the contact reader names them by their axis. A BC written
    either way steps the same particles bit for bit: the exit absorbs, the wall reflects."""

    def bc(low: str, high: str) -> BoundaryConditions:
        return BoundaryConditions(
            dimension=1,
            segments=[
                BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=0.0, boundary=low),
                BCSegment(name="wall", bc_type=BCType.NO_FLUX, boundary=high),
            ],
            domain_bounds=np.array([[0.0, 1.0]]),
        )

    points = [[-0.03], [1.02], [0.5], [1.0 - 5e-11]]
    named = _step(method, bc("left", "right"), points, [(0.0, 1.0)])
    by_axis = _step(method, bc("x_min", "x_max"), points, [(0.0, 1.0)])
    assert named[1].tolist() == by_axis[1].tolist() == [True, False, False, False]
    np.testing.assert_array_equal(named[0], by_axis[0])
    np.testing.assert_allclose(named[0][:, 0], [0.98, 0.5, 1.0 - 5e-11])


def test_the_particle_solvers_periodic_path_reads_a_corner_by_the_owner():
    """One indirection up: the solver's periodic path hands the applicator its own BC and the grid's box. A particle
    past the corner the periodic x faces share with the y_max exit is absorbed; past the middle of y_max it is absorbed
    and past the middle of x_max it is wrapped, as before."""
    bc = BoundaryConditions(
        dimension=2,
        segments=[
            BCSegment(name="periodic_x_min", bc_type=BCType.PERIODIC, boundary="x_min"),
            BCSegment(name="periodic_x_max", bc_type=BCType.PERIODIC, boundary="x_max"),
            BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=0.0, boundary="y_max"),
            BCSegment(name="wall", bc_type=BCType.NO_FLUX, boundary="y_min"),
        ],
    )
    grid = TensorProductGrid(bounds=BOX, Nx_points=[11, 11], boundary_conditions=bc)
    model = Model(hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0)), volatility=0.2)
    conditions = Conditions(m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0, T=0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver = FPParticleSolver(MFGProblem(model=model, domain=grid, conditions=conditions, Nt=4), num_particles=50)
    points = np.array([[1.05, 1.03], [0.5, 1.05], [1.05, 0.5]])
    remaining, n_absorbed, _, absorbed = solver._apply_boundary_conditions_segment_aware(points, BOX)
    assert absorbed.tolist() == [True, True, False]
    assert n_absorbed == 2
    np.testing.assert_allclose(remaining, [[0.05, 0.5]])
