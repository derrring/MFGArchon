"""A marked region on part of a face is read as a ``region=`` placement on it (#2795).

``BoundaryConditions._face_cover`` read a ``region_name`` through ``faces_covered_by_region``, which raised #2472's
refusal on a partial cover of any face of the region, whichever face was asked, before a reader built to tolerate a
partial cover could see one. So every box corner was refused, HJB-GFDM refused a cloud holding one, and the particle
FP's array drift refused the BC, where the same window placed by ``boundary=`` + ``region=`` is read. A marked region's
cover is now read on its grid mask, as a ``region``'s is.

**The oracle is the ``region=`` placement**, measured in the boundary-conditions guide's § Priority Resolution
(#2796): a partial marked window must answer as ``boundary`` + ``region`` placing the same nodes does, at every entry
point that reaches the face owner's cover -- the same segment, or the same refusal, whose message differs only in the
word naming the placement. Under the old raise every partial-window case fails this. Held fixed: 2-D, a 21 x 21
``TensorProductGrid`` of [0, 1]^2 the BC is bound to, one window at priority 1 beside ``default_bc=NO_FLUX``, constant
values, ``corner_strategy="priority"``.
"""

from __future__ import annotations

import re
import warnings

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.fp_solvers.fp_particle import FPParticleSolver
from mfgarchon.alg.numerical.hjb_solvers.hjb_gfdm import HJBGFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions
from mfgarchon.geometry.boundary.applicator_particle import ParticleApplicator

N = 21
H = 1.0 / (N - 1)
AXES = "xy"
FACES = [(0, "min"), (0, "max"), (1, "min"), (1, "max")]


def _window(form: str, face: tuple[int, str], cover: str, cond: str) -> tuple[BoundaryConditions, TensorProductGrid]:
    """A window on ``face``, y (or x) in [0.3, 0.7] for a partial cover, the whole face for a full one, placed by
    ``region_name`` or by ``boundary`` + ``region``; DIRICHLET(0) ("differ") or NO_FLUX ("same") beside a NO_FLUX
    default. The BC returned is the grid's bound copy, as a solver reads it."""
    axis, side = face
    other, at = 1 - axis, 0.0 if side == "min" else 1.0
    lo, hi = (0.3, 0.7) if cover == "partial" else (0.0, 1.0)
    place = (
        {"region_name": "window"}
        if form == "region_name"
        else {"boundary": f"{AXES[axis]}_{side}", "region": {AXES[other]: (lo, hi)}}
    )
    kind = {"bc_type": BCType.DIRICHLET, "value": 0.0} if cond == "differ" else {"bc_type": BCType.NO_FLUX}
    bc = BoundaryConditions(
        dimension=2,
        segments=[BCSegment(name="window", priority=1, **kind, **place)],
        default_bc=BCType.NO_FLUX,
    )
    grid = TensorProductGrid(bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[N, N], boundary_conditions=bc)
    if form == "region_name":
        grid.mark_region(
            "window",
            predicate=lambda x: (
                (np.abs(np.asarray(x)[..., axis] - at) <= 1e-9)
                & (np.asarray(x)[..., other] >= lo - 1e-9)
                & (np.asarray(x)[..., other] <= hi + 1e-9)
            ),
        )
    return grid.get_boundary_conditions(), grid


def _answers(bc: BoundaryConditions, grid: TensorProductGrid, face: tuple[int, str]) -> dict[str, str]:
    """Each entry point that reaches the face owner's cover, read on the window's face: the segment(s) it answers, or
    the refusal it raises with the word naming the placement taken out."""
    axis, side = face
    name = f"{AXES[axis]}_{side}"
    opposite = f"{AXES[axis]}_{'max' if side == 'min' else 'min'}"
    corner = np.zeros(2)
    corner[axis] = 0.0 if side == "min" else 1.0
    other_corner, middle = corner.copy(), corner.copy()
    other_corner[1 - axis], middle[1 - axis] = 1.0, 0.5

    def same_periodicity(part: BCSegment, governing: BCSegment) -> bool:  # `value_gradient`'s equivalent
        return (part.bc_type == BCType.PERIODIC) == (governing.bc_type == BCType.PERIODIC)

    entries = {
        "segment_for_face, the window's face": lambda: bc.segment_for_face(name, grid).name,
        "segment_for_face, the opposite face": lambda: bc.segment_for_face(opposite, grid).name,
        "segment_for_face, value_gradient's equivalent": lambda: (
            bc.segment_for_face(name, grid, equivalent=same_periodicity).name
        ),
        "segment_for_point, a corner": lambda: bc.segment_for_point(corner, geometry=grid).name,
        "segment_for_point, mid-face": lambda: bc.segment_for_point(middle, geometry=grid).name,
        "face_segments_at_point, the other corner": lambda: ",".join(
            seg.name for _, seg, _ in bc.face_segments_at_point(other_corner, geometry=grid)
        ),
        "corner_nodes": lambda: ",".join(sorted({seg.name for _, seg in bc.corner_nodes((N, N), grid).values()})),
        "the particle reader's whole-BC check": lambda: str(
            ParticleApplicator().refuse_segments_it_would_drop(bc, [(0.0, 1.0), (0.0, 1.0)])
        ),
    }
    out = {}
    for entry, ask in entries.items():
        try:
            out[entry] = ask()
        except ValueError as error:
            out[entry] = "refused: " + re.sub(r"placed by `\w+`", "placed by `<placement>`", str(error))
    return out


@pytest.mark.parametrize("cond", ["differ", "same"])
@pytest.mark.parametrize("cover", ["partial", "full"])
@pytest.mark.parametrize("face", FACES, ids=[f"{AXES[a]}_{s}" for a, s in FACES])
def test_a_marked_window_answers_as_a_region_window_at_every_entry_point(face, cover, cond):
    marked = _answers(*_window("region_name", face, cover, cond), face)
    placed = _answers(*_window("boundary+region", face, cover, cond), face)
    mismatches = {entry: (marked[entry], placed[entry]) for entry in marked if marked[entry] != placed[entry]}
    assert not mismatches, mismatches


def _exit_problem(form: str) -> tuple[TensorProductGrid, MFGProblem]:
    _, grid = _window(form, (0, "min"), "partial", "differ")
    model = Model(hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0)), volatility=0.2)
    conditions = Conditions(m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0, T=0.2)
    return grid, MFGProblem(model=model, domain=grid, conditions=conditions, Nt=4)


def test_a_marked_exit_window_absorbs_only_in_its_cells_on_both_drift_routes_and_in_a_coupled_solve():
    """The particle FP's array drift -- what every coupled iterator passes -- reads a marked exit window as its callable
    drift does: it absorbs only on x_min, within the marked nodes' cells, y in [0.3 - h/2, 0.7 + h/2]. Then the coupled
    HJB-GFDM + particle solve runs, so HJB-GFDM's classification of the cloud's corners is reached too."""
    reach = (0.3 - H / 2 - 1e-9, 0.7 + H / 2 + 1e-9)
    for drift in (np.zeros((5, N, N)), lambda t, x, m: np.zeros_like(np.asarray(x, dtype=float))):
        _, problem = _exit_problem("region_name")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            solver = FPParticleSolver(problem, num_particles=4000, seed=7)
            solver.solve_fp_system(M_initial=np.ones((N, N)), drift_field=drift, show_progress=False)
        exits = np.concatenate(solver.exit_positions_history)
        assert solver.total_absorbed > 0
        assert np.all(exits[:, 0] < 1e-9), "an exit off x_min"
        assert reach[0] <= exits[:, 1].min(), exits[:, 1].min()
        assert exits[:, 1].max() <= reach[1], exits[:, 1].max()

    grid, problem = _exit_problem("region_name")
    hjb = HJBGFDMSolver(
        problem,
        collocation_points=grid.get_spatial_grid(),
        delta=0.12,
        k_neighbors=12,
        derivative_method="taylor",
        taylor_order=2,
        weight_function="wendland",
        monotonicity_scheme="none",
    )
    fp = FPParticleSolver(problem, num_particles=2000, seed=7)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = problem.solve(hjb_solver=hjb, fp_solver=fp, max_iterations=3, tolerance=1e-12)
    m = np.asarray(result.M)
    assert np.all(np.isfinite(np.asarray(result.U)))
    assert fp.total_absorbed > 0
    assert grid.integrate(m[-1]) < grid.integrate(m[0])
