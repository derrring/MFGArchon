"""n-D HJB-FDM: a Dirichlet node's row is ``u - g``, and nothing is written after the solve (#2537, #2474).

`_solve_single_timestep` used to solve every wall node's PDE row and then call
`FDMApplicator.enforce_values`, which overwrote Dirichlet faces with ``g`` and Neumann faces with the
first-order ``u[0] = u[1] + g h``. So a step reporting ``converged=True`` returned an array that was not
the root Newton certified, and ``neumann_bc(0)`` and ``no_flux_bc`` (one condition on the HJB side, #1685)
returned different arrays. 1-D had the same shape and lost it in #1900 and #2515.

Now every map the step solves (Newton's residual and both fixed-point maps) gives a Dirichlet node the
equation ``u - g = 0`` (`applicator_fdm.dirichlet_wall_rows`), and a Neumann node keeps its PDE row, whose
node-centred ghost already imposes ``du/dn = g`` to second order. Faces are read through `face_segment`,
the resolver the ghosts use, so a face is Dirichlet in the rows exactly when it is in the residual.
"""

from __future__ import annotations

import warnings

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import (
    BCSegment,
    BCType,
    BoundaryConditions,
    dirichlet_bc,
    neumann_bc,
    no_flux_bc,
    periodic_bc,
)

N, NT, T, SIGMA = 11, 2, 0.3, 0.5
BOUNDS = np.array([[0.0, 1.0], [0.0, 1.0]])


def _faces(spec: dict[str, tuple[BCType, float]], **priority: int) -> BoundaryConditions:
    return BoundaryConditions(
        dimension=2,
        segments=[
            BCSegment(name=f, bc_type=t, boundary=f, value=v, priority=priority.get(f, 0)) for f, (t, v) in spec.items()
        ],
    )


ARMS = {
    # The uniform BCs take the ghost buffer's uniform path, the faced ones its per-face path.
    "dirichlet_uniform": lambda: dirichlet_bc(dimension=2, value=0.4),
    "dirichlet_faced": lambda: _faces(
        {
            "x_min": (BCType.DIRICHLET, 0.3),
            "x_max": (BCType.DIRICHLET, 0.8),
            "y_min": (BCType.NO_FLUX, 0.0),
            "y_max": (BCType.NO_FLUX, 0.0),
        }
    ),
    "neumann_uniform": lambda: neumann_bc(dimension=2, value=-0.5),
    "neumann_faced": lambda: _faces(
        {
            "x_min": (BCType.NEUMANN, -0.4),
            "x_max": (BCType.NEUMANN, -0.9),
            "y_min": (BCType.NEUMANN, -0.6),
            "y_max": (BCType.NEUMANN, -0.2),
        }
    ),
    "neumann_zero": lambda: neumann_bc(dimension=2, value=0.0),
    "no_flux": lambda: no_flux_bc(dimension=2),
    "periodic": lambda: periodic_bc(dimension=2),
    # BoundaryConditions.get_bc_type_at_boundary's own example -- an exit, and a wall with no `boundary` --
    # as its `mixed_bc` builds it, with the NEUMANN default.
    "exit_and_wall": lambda: BoundaryConditions(
        dimension=2,
        segments=[
            BCSegment(name="exit", bc_type=BCType.DIRICHLET, boundary="x_max", value=0.7),
            BCSegment(name="wall", bc_type=BCType.NEUMANN),
        ],
        domain_bounds=BOUNDS,
        default_bc=BCType.NEUMANN,
    ),
    # The same, spelled with the NO_FLUX default #1100's message suggests: on the HJB side one zero flux.
    "exit_and_wall_no_flux_default": lambda: BoundaryConditions(
        dimension=2,
        segments=[
            BCSegment(name="exit", bc_type=BCType.DIRICHLET, boundary="x_max", value=0.7),
            BCSegment(name="wall", bc_type=BCType.NEUMANN),
        ],
        domain_bounds=BOUNDS,
        default_bc=BCType.NO_FLUX,
    ),
    # A face-label region name places its segment for both readers (#2472).
    "region_name_face_label": lambda: BoundaryConditions(
        dimension=2,
        segments=[BCSegment(name="exit", bc_type=BCType.DIRICHLET, region_name="x_max", value=0.6)],
        domain_bounds=BOUNDS,
        default_bc=BCType.NO_FLUX,
    ),
}


def _terminal(X, Y):
    """Not symmetric about either midline, so mirrored walls do not coincide."""
    return np.cos(np.pi * X) + 0.5 * np.sin(np.pi * Y) + 0.2 * X * Y


def _solver(bc, n: int = N, solver_type: str = "newton") -> tuple[HJBFDMSolver, np.ndarray, np.ndarray]:
    grid = TensorProductGrid(bounds=[(0.0, 1.0)] * 2, Nx_points=[n, n], boundary_conditions=bc)
    problem = MFGProblem(
        model=Model(
            hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)), volatility=SIGMA
        ),
        domain=grid,
        conditions=Conditions(m_initial=lambda x: 1.0 + 0.0 * float(np.sum(x)), u_terminal=lambda x: 0.0, T=T),
        Nt=NT,
    )
    X, Y = np.meshgrid(*grid.coordinates, indexing="ij")
    return HJBFDMSolver(problem, solver_type=solver_type), X, Y


def _solve(
    bc, n: int = N, solver_type: str = "newton", source_term=None, terminal=_terminal
) -> tuple[np.ndarray, list]:
    """Solve, recording each step's (map, root, info) as the inner solver returned them."""
    solver, X, Y = _solver(bc, n, solver_type)
    steps = []
    inner = solver.nonlinear_solver.solve

    def spy(F, U0):
        with warnings.catch_warnings():
            # NewtonSolver warns at stacklevel 2, which this wrapper would become. The warning is hjb_fdm's
            # and is recorded under that file; this test is not about it.
            warnings.filterwarnings("ignore", message="JAX autodiff failed", category=RuntimeWarning)
            root, info = inner(F, U0)
        steps.append((F, np.array(root, copy=True), info))
        return root, info

    solver.nonlinear_solver.solve = spy
    u_T = terminal(X, Y)
    shape = (NT + 1, n, n)
    U = solver.solve_hjb_system(
        M_density=np.ones(shape),
        U_terminal=u_T,
        U_coupling_prev=np.broadcast_to(u_T, shape).copy(),
        source_term=source_term,
    )
    return np.asarray(U), steps


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_a_converged_step_returns_the_root_it_certified(arm: str):
    """The #1900 law, n-D: ``converged`` is a claim about the array handed back.

    The steps run backward from t = T, so step k returned ``U[NT - 1 - k]``. Before #2537 the
    Dirichlet and Neumann arms failed here; no-flux and periodic, which had no overwrite, are the
    controls that the comparison itself holds.
    """
    U, steps = _solve(ARMS[arm]())
    assert len(steps) == NT
    for k, (F, root, info) in enumerate(steps):
        assert info.converged, f"{arm}: step {k} did not converge; the law has nothing to bind"
        returned = U[NT - 1 - k]
        assert np.array_equal(returned, root), (
            f"{arm}: step {k} returned an array that differs from Newton's root by "
            f"{np.abs(returned - root).max():.3e}; something wrote to it after the solve"
        )
        assert np.abs(F(returned)).max() < 1e-6


def _apply_once(outputs: list):
    """A stand-in for the inner fixed-point solve: one application of the map, reported converged, and
    recorded, so the test can tell what the map returned from what the step handed back."""
    from mfgarchon.utils.numerical import SolverInfo

    def solve(self, G, U0):
        outputs.append(np.array(G(U0), copy=True))
        return outputs[-1].copy(), SolverInfo(converged=True, iterations=1, residual=0.0, residual_history=[])

    return solve


def test_the_fixed_point_map_returns_g_at_a_dirichlet_node(monkeypatch):
    """Value iteration's map gives a Dirichlet node ``g``, so its fixed point holds the condition with
    nothing written afterwards. One application is enough to read the map's wall rows: the 2-D value
    iteration does not converge on this fixture (dt D / h^2 = 1.875, its 30 iterations), nor with
    dt D / h^2 = 0.117 and 600, which is neither here nor there for what the map returns."""
    from mfgarchon.utils.numerical.nonlinear_solvers import FixedPointSolver

    outputs: list = []
    monkeypatch.setattr(FixedPointSolver, "solve", _apply_once(outputs))
    solver, X, Y = _solver(dirichlet_bc(dimension=2, value=0.4), solver_type="fixed_point")
    u_T = _terminal(X, Y)
    shape = (NT + 1, N, N)
    U = solver.solve_hjb_system(
        M_density=np.ones(shape), U_terminal=u_T, U_coupling_prev=np.broadcast_to(u_T, shape).copy()
    )
    assert len(outputs) == NT
    for k, output in enumerate(outputs):
        returned = U[NT - 1 - k]
        assert np.array_equal(returned, output), "the step wrote to the map's output after the solve"
        assert np.all(returned[[0, -1], :] == 0.4)
        assert np.all(returned[:, [0, -1]] == 0.4)


def test_the_value_iteration_fallback_also_returns_g_at_a_dirichlet_node(monkeypatch):
    """The map Newton falls back to under ``on_newton_failure="warn_and_fallback"`` is a second copy of
    the fixed-point map, and it carries the same wall rows."""
    from mfgarchon.utils.numerical import SolverInfo
    from mfgarchon.utils.numerical.nonlinear_solvers import FixedPointSolver

    outputs: list = []
    monkeypatch.setattr(FixedPointSolver, "solve", _apply_once(outputs))
    solver, X, Y = _solver(dirichlet_bc(dimension=2, value=0.4))
    solver.on_newton_failure = "warn_and_fallback"
    failed = SolverInfo(converged=False, iterations=1, residual=1.0, residual_history=[1.0])
    solver.nonlinear_solver.solve = lambda F, U0: (U0, failed)
    u_T = _terminal(X, Y)
    shape = (NT + 1, N, N)
    with pytest.warns(UserWarning, match="Falling back to Value Iteration"):
        U = solver.solve_hjb_system(
            M_density=np.ones(shape), U_terminal=u_T, U_coupling_prev=np.broadcast_to(u_T, shape).copy()
        )
    assert len(outputs) == NT
    for k, output in enumerate(outputs):
        returned = U[NT - 1 - k]
        assert np.array_equal(returned, output), "the step wrote to the fallback map's output after the solve"
        assert np.all(returned[[0, -1], :] == 0.4)
        assert np.all(returned[:, [0, -1]] == 0.4)


def test_neumann_zero_and_no_flux_are_one_condition_in_2d():
    """#1685: on the HJB side both are du/dn = 0. The overwrite applied to NEUMANN only, so in 2-D
    they returned arrays differing by 4.3e-02 at 21^2 (1-D agreed exactly, #1900)."""
    U_neumann, _ = _solve(neumann_bc(dimension=2, value=0.0), n=21)
    U_no_flux, _ = _solve(no_flux_bc(dimension=2), n=21)
    np.testing.assert_array_equal(U_neumann, U_no_flux)


def test_a_dirichlet_face_holds_g_where_it_meets_a_neumann_face():
    """A prescribed node has no PDE row, corner included. Before, the Neumann overwrite of y_min ran
    after the Dirichlet one and moved the shared corner to 0.25 against g = 0.2."""
    bc = _faces(
        {
            "x_min": (BCType.DIRICHLET, 0.2),
            "y_min": (BCType.NEUMANN, 0.5),
            "x_max": (BCType.NO_FLUX, 0.0),
            "y_max": (BCType.NO_FLUX, 0.0),
        }
    )
    U, _ = _solve(bc)
    assert np.all(U[:-1, 0, :] == 0.2)


@pytest.mark.parametrize(
    ("priority", "winner"),
    [({}, 0.2), ({"y_min": 1}, 0.9)],
    ids=["equal_priority_declaration_order", "y_min_outranks"],
)
def test_where_dirichlet_faces_meet_the_first_segment_in_the_bcs_order_holds_the_corner(priority: dict, winner: float):
    """A node on two Dirichlet faces takes the value of the segment first in the BC's own order --
    priority, then declaration, the order `get_bc_at_point` reads segments in. The overwrite applied
    segments in that order, so the LAST one -- the lowest priority -- won: 0.9 and 0.2 here."""
    spec = {
        "x_min": (BCType.DIRICHLET, 0.2),
        "y_min": (BCType.DIRICHLET, 0.9),
        "x_max": (BCType.NO_FLUX, 0.0),
        "y_max": (BCType.NO_FLUX, 0.0),
    }
    bc = _faces(spec, **priority)
    U, _ = _solve(bc)
    assert np.all(U[:-1, 0, 0] == winner)
    assert np.all(U[:-1, 0, 1:] == 0.2)
    assert np.all(U[:-1, 1:, 0] == 0.9)


def test_the_documented_exit_and_wall_idiom_holds_its_exit():
    """`get_bc_type_at_boundary`'s example: DIRICHLET on x_max, and a NEUMANN wall with no `boundary`.
    The overwrite applied the wall to every face after the exit, so x_max came back at 0.045..0.174
    against g = 0.7."""
    U, _ = _solve(ARMS["exit_and_wall"]())
    assert np.all(U[:-1, -1, :] == 0.7)


# A manufactured solution for the Neumann wall: u = phi + (T - t) psi with H = |p|^2/2, so the source is
# S = psi + |grad u|^2/2 - D lap u. Outward slopes x_min -0.4, x_max -0.9, y_min -0.6, y_max -0.2, so the
# drift -grad u points into every wall, and no two walls carry the same datum.
_D = SIGMA**2 / 2
_G = {"x_min": -0.4, "x_max": -0.9, "y_min": -0.6, "y_max": -0.2}


def _exact(t, x, y):
    tau = T - t
    p = np.pi
    u = 0.4 * x - 0.65 * x**2 + 0.6 * y - 0.4 * y**2 + 0.4 * np.cos(p * x) * np.cos(p * y) + tau * 0.5 * np.cos(p * x)
    ux = 0.4 - 1.3 * x - 0.4 * p * np.sin(p * x) * np.cos(p * y) - tau * 0.5 * p * np.sin(p * x)
    uy = 0.6 - 0.8 * y - 0.4 * p * np.cos(p * x) * np.sin(p * y)
    lap = -2.1 - 0.8 * p**2 * np.cos(p * x) * np.cos(p * y) - tau * 0.5 * p**2 * np.cos(p * x)
    return u, ux, uy, lap


def _source(t, x):
    x = np.asarray(x, dtype=float)
    _, ux, uy, lap = _exact(t, x[:, 0], x[:, 1])
    return 0.5 * np.cos(np.pi * x[:, 0]) + 0.5 * (ux**2 + uy**2) - _D * lap


def _wall_slope_miss(n: int) -> float:
    """Largest gap between g and the returned field's second-order one-sided outward slope, t = 0."""
    bc = _faces({face: (BCType.NEUMANN, g) for face, g in _G.items()})
    U, _ = _solve(bc, n=n, source_term=_source, terminal=lambda X, Y: _exact(T, X, Y)[0])
    u, h = U[0], 1.0 / (n - 1)
    slopes = {
        "x_min": (3 * u[0, :] - 4 * u[1, :] + u[2, :]) / (2 * h),
        "x_max": (3 * u[-1, :] - 4 * u[-2, :] + u[-3, :]) / (2 * h),
        "y_min": (3 * u[:, 0] - 4 * u[:, 1] + u[:, 2]) / (2 * h),
        "y_max": (3 * u[:, -1] - 4 * u[:, -2] + u[:, -3]) / (2 * h),
    }
    return max(float(np.abs(slope - _G[face]).max()) for face, slope in slopes.items())


def test_a_neumann_wall_holds_its_slope_to_second_order():
    """The residual's node-centred ghost imposes du/dn = g to second order; the overwrite replaced it
    with a first-order one-sided value. Measured on this fixture at 11/21/41: the overwrite missed by
    0.422 / 0.238 / 0.123 (ratios 1.77, 1.94), the residual's own wall by 0.0328 / 0.00683 / 0.000984."""
    coarse, fine = _wall_slope_miss(11), _wall_slope_miss(21)
    assert fine < 0.02, f"wall slope misses g by {fine:.3e} at 21^2"
    assert coarse / fine > 3.0, f"wall slope error ratio {coarse / fine:.2f} under refinement is not second order"


_UNPLACED = {
    # A Dirichlet with no `boundary` under a NEUMANN x_min: the accessors give it every face, the ghosts none.
    "dirichlet_everywhere_else": lambda: BoundaryConditions(
        dimension=2,
        segments=[
            BCSegment(name="rest", bc_type=BCType.DIRICHLET, value=0.5),
            BCSegment(name="left", bc_type=BCType.NEUMANN, boundary="x_min", value=0.0),
        ],
        domain_bounds=BOUNDS,
        default_bc=BCType.NEUMANN,
    ),
    # The exit-and-wall example with a wall whose datum the NEUMANN(0) default does not carry.
    "wall_with_a_datum": lambda: BoundaryConditions(
        dimension=2,
        segments=[
            BCSegment(name="exit", bc_type=BCType.DIRICHLET, boundary="x_max", value=0.7),
            BCSegment(name="wall", bc_type=BCType.NEUMANN, value=-0.8),
        ],
        domain_bounds=BOUNDS,
        default_bc=BCType.NEUMANN,
    ),
    # One operation everywhere, so a reader of operations alone sees nothing; the datum still differs.
    "one_operation_other_datum": lambda: BoundaryConditions(
        dimension=2,
        segments=[
            BCSegment(name="wall", bc_type=BCType.NEUMANN, value=-0.8),
            BCSegment(name="left", bc_type=BCType.NEUMANN, boundary="x_min", value=0.0),
        ],
        domain_bounds=BOUNDS,
        default_bc=BCType.NEUMANN,
    ),
    # Both readers apply 'all' to no face (#1953).
    "boundary_all": lambda: BoundaryConditions(
        dimension=2,
        segments=[BCSegment(name="all", bc_type=BCType.DIRICHLET, boundary="all", value=0.5)],
        domain_bounds=BOUNDS,
        default_bc=BCType.NO_FLUX,
    ),
    # A face of no 2-D domain: the overwrite raised IndexError on it, the ghosts drop it.
    "z_max_in_2d": lambda: BoundaryConditions(
        dimension=2,
        segments=[BCSegment(name="top", bc_type=BCType.DIRICHLET, boundary="z_max", value=0.5)],
        domain_bounds=BOUNDS,
        default_bc=BCType.NO_FLUX,
    ),
    # Part of a face: both readers stretch it over the whole face (#2490).
    "partial_face": lambda: BoundaryConditions(
        dimension=2,
        segments=[
            BCSegment(name="half", bc_type=BCType.DIRICHLET, boundary="x_max", region={1: (0.0, 0.5)}, value=0.5)
        ],
        domain_bounds=BOUNDS,
        default_bc=BCType.NO_FLUX,
    ),
}


@pytest.mark.parametrize("case", sorted(_UNPLACED))
def test_a_segment_the_ghosts_cannot_place_is_refused(case: str):
    """`bc_utils.refuse_unplaceable_segments`, with ``ghosts=True``: a segment the FDM ghosts cannot place
    is refused instead of silently solving `default_bc` on its faces. The post-solve overwrite used to write
    the accessor's datum, so these were imposed (or, for `z_max_in_2d`, raised IndexError) before #2537. A
    segment with no `boundary` is refused only where it changes a face's effect, so `exit_and_wall`, its
    NO_FLUX-default spelling and a face-label `region_name` above all solve."""
    with pytest.raises(NotImplementedError, match="HJBFDMSolver"):
        _solve(_UNPLACED[case]())
