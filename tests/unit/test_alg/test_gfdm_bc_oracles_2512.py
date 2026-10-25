"""HJB-GFDM against exact solutions on the BC cells #2512 (a) still owed it: NEUMANN with a datum in 2-D, ROBIN in 2-D
and positive_qp_fan's NEUMANN, both by routing, and PERIODIC in 1-D and 2-D (stage A2a).

Every solve is on the default configuration, ``monotonicity_scheme="none"`` with Newton (user ruling 2026-10-22, #2512
comment 6333790265), except positive_qp_fan's routing pin, which is that configuration's own cell. Each fixture is
exp17's construction, as `test_positive_qp_fan_reflecting_1689` builds it: ``u* = phi + C (T - t)`` with ``m*`` frozen
and ``V`` absorbing the residual, so the equation holds exactly, and implicit Euler is exact in time, so the error is
the spatial one.

**Fixtures.**
- **NEUMANN, 2-D.** ``phi = a (q(x) + q(y)) + c1 x + c2 x^2 + c3 y + c4 y^2`` with ``q(s) = s^3 - 1.5 s^2``, so the
  outward derivative is ``-c1`` on x_min, ``c1 + 2 c2`` on x_max, ``-c3`` on y_min and ``c3 + 2 c4`` on y_max, whatever
  ``q`` adds (``q'(0) = q'(1) = 0``). Two BCs:
  - per face, four NEUMANN segments with the outward data 0.3, 0.7, -0.2 and 0.5. They differ, so a face or an axis
    read for another is visible, and the mixed path (`_preclassify_boundary_points`) builds each row;
  - uniform, ``neumann_bc(value=0.4)``, the same datum on every face, which the uniform path builds.
  The cloud is exp17's Sobol cloud (scramble 42, ``2 sqrt(N_int)`` boundary nodes per face) with the four box corners
  added, refined N_int 900 -> 1600 -> 2500 with ``delta = 4.5 h``, ``h = 1/sqrt(N_int)``, as #2792's reflecting cell. A
  corner takes one face's segment and normal, and ``u*`` meets each face's datum there, so a correct corner row
  converges whichever face it takes. Scrambles 3 and 7, without the corners, run in the slow tier.
- **PERIODIC, 1-D and 2-D.** ``phi = 0.3 sin(2 pi x + 0.7) [+ 0.2 sin(2 pi y + 0.3)]``: not even about the seam. The
  problem's own periodic `TensorProductGrid` is the torus, collocated at its nodes, both endpoints included: boundary
  detection skips its periodic axes (no boundary point, #1841) and the operator's ghost points make the wrap. Levels
  21/41/81 (1-D) and 11/21/41 (2-D) with ``delta = 2.5 h``, ``h = 1/(n - 1)``; in 1-D the neighbourhood falls back to
  k nearest neighbours at most nodes, so ``delta`` sets the stencil near the seam. A fixed ``delta`` is not a
  refinement: at the default 0.1 the 1-D error stops falling (EOC 1.07, then 0.00, measured at 63ea63dd).

**Assertions.** Per cell, the order between levels in [1.7, 2.3], each level's error within a factor of 1.25 either way
of the value recorded here, and at each level the error on each face's nodes and on the corners (NEUMANN), or on the
seam nodes (PERIODIC), within the same factor. The Robin pin asserts the t = 0 error arrays equal on the mixed path;
the positive_qp_fan pin, the whole solutions on 1689's ``box-900`` input.

**Rivals**, each a copy of the package at 63ea63dd with one replacement (the corner rival, one per path), run with this
file (the slow tier left out):
- NEUMANN, per face / uniform, the orders over the two refinements:
  - the row read along the face's tangent: 0.02, -0.09 / -0.18, 1.30;
  - the normal's sign flipped: -0.00, -0.00 / -0.01, -0.01;
  - the datum dropped (#2513's defect): -0.00, -0.00 / -0.01, -0.01;
  - every face read with its opposite face's segment (the mixed path): -0.01, -0.01 per face; uniform passes, as it must
    with one datum. Review 1 of #2806 read the lower half of x_min with x_max's segment: 0.02, -0.04;
  - the corner's normal taken as the diagonal average of its faces': 2.06, 1.31 / 1.71, 2.34, and the corner error
    x5.9 to x12.7 / x2.7 to x5.2 of the value recorded here. On the uniform fixture the rate leaves its band by only
    0.04, and the face and corner bounds leave theirs by x2.5 to x5.2.
- The routing pins: ROBIN(0, 1)'s datum dropped in its arm alone fails the Robin pin; NEUMANN taken out of
  positive_qp_fan's ``reflecting_types`` makes it refuse the zero datum.
- PERIODIC, 1-D / 2-D: no ghost points -2.66, 2.86 / 0.08, -9.83; mirror ghosts (a reflecting closure) -0.03, -0.01 /
  -0.08, 0.02; the period 2% long 0.56, 0.20 / 1.37, 0.65. Dropping `_wrap_displacement` passes, to the digit: on these
  inputs every offset is below half a period and the ghost points are already shifted, so it is an equivalent mutant
  here, not a rival.
- Passing, by what is held fixed: the datum read at the next time level (constant data).

**Branch conditions.** Varied: the mixed path (per-point segment and normal) against the uniform path
(``uniform_segment``); a corner node on both; the NEUMANN arm of `_bc_row_for_point`, and its ROBIN(0, 1) arm through
the routing pin; positive_qp_fan's ``reflecting_types``, which take NEUMANN(0) as a reflecting face; the periodic route,
where `_detect_boundary_indices` skips every axis of the torus. Held fixed, as instrumented by review 1 of #2806:
- every corner takes its x face's row on both paths (the mixed path's first matching reading);
- the uniform path's normal comes from its legacy normals;
- detection's per-axis ``np.abs(.) < tol`` selection is not evaluated: the NEUMANN cells pass ``boundary_indices`` and
  the torus skips every axis, so a box periodic on one axis and walled on the other is not reached;
- no local coordinate rotation, no ghost nodes (#2516); the degenerate-stencil arm of `_build_neumann_bc_row` (#2752)
  is not reached; constant data; Newton only, since Howard's value-form rows come from the same `_bc_row_for_point`
  (6333790265).
"""

from __future__ import annotations

import warnings

import pytest
from test_positive_qp_fan_reflecting_1689 import (
    A_PHI,
    C_TIME,
    NT,
    NU,
    SIGMA,
    T_END,
    Z_M,
    ZETA,
    _m_star,
    _problem,
    _sobol,
    _u_star,
)

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.hjb_solvers import HJBGFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, neumann_bc, no_flux_bc, periodic_bc

LEVEL_FACTOR = 1.25
ORDER = (1.7, 2.3)
LADDER = (900, 1600, 2500)
FACES = ("x_min", "x_max", "y_min", "y_max")
#: Outward Neumann data, in the order of `FACES`.
PER_FACE = (0.3, 0.7, -0.2, 0.5)
UNIFORM = 0.4
CORNERS = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])


def _hamiltonian(phi):
    """exp17's Hamiltonian, whose ``V`` absorbs the residual of ``u* = phi + C (T - t)`` against
    ``m* = exp(-phi/nu)``."""

    def m_star(x):
        return np.exp(-phi(x)[0] / NU) / Z_M

    def potential(t, x):
        _, lap, grad2 = phi(x)
        v = C_TIME - NU * lap + 0.5 * grad2 - ZETA * m_star(x)
        return float(v[0]) if np.ndim(x) == 1 else v

    hamiltonian = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0), potential=potential, coupling=lambda m: ZETA * m
    )
    return hamiltonian, m_star


def _neumann_phi(g):
    """``phi`` with outward derivative ``g[k]`` on face ``FACES[k]``: (value, Laplacian, |grad|^2)."""
    c1, c3 = -g[0], -g[2]
    c2, c4 = (g[1] - c1) / 2, (g[3] - c3) / 2

    def phi(x):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        X, Y = x[:, 0], x[:, 1]
        value = A_PHI * (X**3 - 1.5 * X**2 + Y**3 - 1.5 * Y**2) + c1 * X + c2 * X**2 + c3 * Y + c4 * Y**2
        gx = A_PHI * (3 * X**2 - 3 * X) + c1 + 2 * c2 * X
        gy = A_PHI * (3 * Y**2 - 3 * Y) + c3 + 2 * c4 * Y
        return value, A_PHI * (6 * X - 3 + 6 * Y - 3) + 2 * c2 + 2 * c4, gx**2 + gy**2

    return phi


def _neumann_bc(kind: str, robin: bool = False) -> BoundaryConditions:
    if kind == "uniform":
        return neumann_bc(value=UNIFORM, dimension=2)
    if robin:
        segments = [
            BCSegment(name=f, bc_type=BCType.ROBIN, alpha=0.0, beta=1.0, value=g, boundary=f)
            for f, g in zip(FACES, PER_FACE, strict=True)
        ]
    else:
        segments = [
            BCSegment(name=f, bc_type=BCType.NEUMANN, value=g, boundary=f) for f, g in zip(FACES, PER_FACE, strict=True)
        ]
    return BoundaryConditions(dimension=2, segments=segments)


def _neumann_solve(kind: str, n_int: int, scramble: int, corners: bool, robin: bool = False):
    """``(U at t = 0 - u* at t = 0, points, boundary indices)`` on exp17's cloud, the corners added if ``corners``."""
    g = (UNIFORM,) * 4 if kind == "uniform" else PER_FACE
    phi = _neumann_phi(g)
    hamiltonian, m_star = _hamiltonian(phi)
    bc = _neumann_bc(kind, robin=robin)
    grid = TensorProductGrid(bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[5, 5], boundary_conditions=bc)
    problem = MFGProblem(
        model=Model(hamiltonian=hamiltonian, volatility=SIGMA),
        domain=grid,
        conditions=Conditions(m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0, T=T_END),
        Nt=NT,
    )
    points, boundary = _sobol(n_int, seed=scramble)
    if corners:
        boundary = np.concatenate([boundary, len(points) + np.arange(len(CORNERS))])
        points = np.vstack([points, CORNERS])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver = HJBGFDMSolver(
            problem, collocation_points=points, boundary_indices=boundary, delta=4.5 / np.sqrt(n_int)
        )
        U = solver.solve_hjb_system(M_density=np.tile(m_star(points), (NT + 1, 1)), U_terminal=phi(points)[0])
    return np.asarray(U)[0] - (phi(points)[0] + C_TIME * T_END), points, boundary


def _neumann_errors(error, points, boundary):
    """The L2 error over the interior nodes, and the max error on each face's nodes and on the corners."""
    interior = np.setdiff1d(np.arange(len(points)), boundary)
    on = points[boundary]
    corner = (np.isclose(on[:, 0], 0.0) | np.isclose(on[:, 0], 1.0)) & (
        np.isclose(on[:, 1], 0.0) | np.isclose(on[:, 1], 1.0)
    )
    faces = {}
    for name, (axis, value) in zip(FACES, ((0, 0.0), (0, 1.0), (1, 0.0), (1, 1.0)), strict=True):
        faces[name] = float(np.abs(error[boundary[np.isclose(on[:, axis], value) & ~corner]]).max())
    if corner.any():
        faces["corners"] = float(np.abs(error[boundary[corner]]).max())
    return float(np.sqrt(np.mean(error[interior] ** 2))), faces


def _orders(errors, sizes):
    return [float(np.log(errors[i] / errors[i + 1]) / np.log(sizes[i + 1] / sizes[i])) for i in range(len(errors) - 1)]


def _assert_recorded(name, errors, faces, recorded_errors, recorded_faces):
    for got, recorded in zip(errors, recorded_errors, strict=True):
        assert recorded / LEVEL_FACTOR < got < recorded * LEVEL_FACTOR, (name, errors)
    for level, (got, recorded) in enumerate(zip(faces, recorded_faces, strict=True)):
        off = {k: (got[k], v) for k, v in recorded.items() if not v / LEVEL_FACTOR < got[k] < v * LEVEL_FACTOR}
        assert not off, (name, level, off)


#: Scramble 42 with the corners, at 63ea63dd: the L2 error per level, and each face's and the corners' max error.
NEUMANN_RECORDED = {
    "per_face": (
        (3.4157e-04, 1.9218e-04, 1.2244e-04),
        (
            {"x_min": 5.8348e-04, "x_max": 8.1944e-04, "y_min": 5.9488e-04, "y_max": 8.2434e-04, "corners": 8.1489e-04},
            {"x_min": 3.4094e-04, "x_max": 4.6112e-04, "y_min": 3.3059e-04, "y_max": 4.6626e-04, "corners": 4.6075e-04},
            {"x_min": 2.4062e-04, "x_max": 3.0314e-04, "y_min": 2.3800e-04, "y_max": 3.0250e-04, "corners": 3.0319e-04},
        ),
    ),
    "uniform": (
        (6.8147e-04, 3.9024e-04, 2.5136e-04),
        (
            {"x_min": 1.3122e-03, "x_max": 4.8904e-04, "y_min": 1.3195e-03, "y_max": 5.1068e-04, "corners": 1.2637e-03},
            {"x_min": 7.3771e-04, "x_max": 2.9692e-04, "y_min": 7.3933e-04, "y_max": 2.8392e-04, "corners": 7.2501e-04},
            {"x_min": 4.9442e-04, "x_max": 1.8745e-04, "y_min": 4.9562e-04, "y_max": 1.8352e-04, "corners": 4.8645e-04},
        ),
    ),
}


@pytest.mark.parametrize("kind", sorted(NEUMANN_RECORDED))
def test_a_neumann_datum_is_reproduced_at_second_order_on_every_face_and_corner(kind):
    errors, faces = [], []
    for n_int in LADDER:
        e, f = _neumann_errors(*_neumann_solve(kind, n_int, 42, corners=True))
        errors.append(e)
        faces.append(f)
    orders = _orders(errors, np.sqrt(LADDER))
    assert all(ORDER[0] <= o <= ORDER[1] for o in orders), (kind, errors, orders)
    _assert_recorded(kind, errors, faces, *NEUMANN_RECORDED[kind])


@pytest.mark.slow
@pytest.mark.parametrize("scramble", [3, 7])
def test_the_per_face_neumann_cell_converges_on_other_clouds(scramble):
    """Robustness across exp17's clouds, without the corners; every rival above was measured on scramble 42."""
    errors = [_neumann_errors(*_neumann_solve("per_face", n_int, scramble, corners=False))[0] for n_int in LADDER]
    orders = _orders(errors, np.sqrt(LADDER))
    assert all(ORDER[0] <= o <= ORDER[1] for o in orders), (scramble, errors, orders)


def test_a_robin_with_alpha_0_and_beta_1_solves_as_the_neumann_row_in_2d():
    """``hjb_gfdm/robin_2d``'s routing pin: ROBIN(0, 1, g) is the condition n.grad u = g, which `_bc_row_for_point`
    builds with the Neumann row, so each face's ROBIN(0, 1, g) solves as its NEUMANN(g), array for array."""
    neumann, _, _ = _neumann_solve("per_face", 900, 42, corners=True)
    robin, _, _ = _neumann_solve("per_face", 900, 42, corners=True, robin=True)
    np.testing.assert_array_equal(robin, neumann)


def test_positive_qp_fan_reads_a_zero_neumann_datum_as_its_reflecting_face():
    """``hjb_gfdm_positive_qp_fan/neumann_2d``'s routing pin (owed by #2512 comment 6333097202): positive_qp_fan admits
    NEUMANN with a zero datum as a reflecting face (``reflecting_types``), so on 1689's box it solves as NO_FLUX, array
    for array, whether written uniform or per face. A nonzero datum is refused (1689's reflecting test)."""
    points, boundary = _sobol(900, seed=42)

    def solve(bc):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            solver = HJBGFDMSolver(
                _problem(bc),
                collocation_points=points,
                boundary_indices=boundary,
                delta=0.15,
                monotonicity_scheme="positive_qp_fan",
                inner_solver="howard",
            )
            return np.asarray(
                solver.solve_hjb_system(
                    M_density=np.tile(_m_star(points), (NT + 1, 1)), U_terminal=_u_star(T_END, points)
                )
            )

    no_flux = solve(no_flux_bc(dimension=2))
    per_face = BoundaryConditions(
        dimension=2, segments=[BCSegment(name=f, bc_type=BCType.NEUMANN, value=0.0, boundary=f) for f in FACES]
    )
    np.testing.assert_array_equal(solve(neumann_bc(value=0.0, dimension=2)), no_flux)
    np.testing.assert_array_equal(solve(per_face), no_flux)


AMPLITUDES, PHASES = (0.3, 0.2), (0.7, 0.3)


def _periodic_phi(x):
    x = np.atleast_2d(np.asarray(x, dtype=float))
    d = x.shape[1]
    angles = [2 * np.pi * x[:, k] + PHASES[k] for k in range(d)]
    value = sum(AMPLITUDES[k] * np.sin(angles[k]) for k in range(d))
    lap = sum(-AMPLITUDES[k] * (2 * np.pi) ** 2 * np.sin(angles[k]) for k in range(d))
    grad2 = sum((AMPLITUDES[k] * 2 * np.pi * np.cos(angles[k])) ** 2 for k in range(d))
    return value, lap, grad2


def _periodic_errors(d: int, n: int):
    """The L2 error over every node, and the max error on the seam nodes (a coordinate at 0 or 1)."""
    hamiltonian, m_star = _hamiltonian(_periodic_phi)
    grid = TensorProductGrid(bounds=[(0.0, 1.0)] * d, Nx_points=[n] * d, boundary_conditions=periodic_bc(dimension=d))
    problem = MFGProblem(
        model=Model(hamiltonian=hamiltonian, volatility=SIGMA),
        domain=grid,
        conditions=Conditions(m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0, T=T_END),
        Nt=NT,
    )
    axes = np.meshgrid(*[np.linspace(0.0, 1.0, n)] * d, indexing="ij")
    points = np.stack([a.ravel() for a in axes], axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver = HJBGFDMSolver(problem, collocation_points=points, delta=2.5 / (n - 1))
        assert len(solver.boundary_indices) == 0, "a periodic axis has no boundary point (#1841)"
        U = solver.solve_hjb_system(M_density=np.tile(m_star(points), (NT + 1, 1)), U_terminal=_periodic_phi(points)[0])
    error = np.asarray(U)[0] - (_periodic_phi(points)[0] + C_TIME * T_END)
    seam = np.any(np.isclose(points, 0.0) | np.isclose(points, 1.0), axis=1)
    return float(np.sqrt(np.mean(error**2))), {"seam": float(np.abs(error[seam]).max())}


#: At 63ea63dd: the L2 error per level, and the max error on the seam nodes.
PERIODIC_RECORDED = {
    1: (
        (21, 41, 81),
        (1.0432e-02, 2.7422e-03, 6.9985e-04),
        ({"seam": 1.2221e-02}, {"seam": 3.3228e-03}, {"seam": 8.6723e-04}),
    ),
    2: (
        (11, 21, 41),
        (7.5413e-02, 1.8400e-02, 4.5686e-03),
        ({"seam": 9.0674e-02}, {"seam": 2.2331e-02}, {"seam": 5.5675e-03}),
    ),
}


@pytest.mark.parametrize("d", sorted(PERIODIC_RECORDED))
def test_a_periodic_solution_that_is_not_even_about_the_seam_is_reproduced_at_second_order(d):
    levels, recorded_errors, recorded_faces = PERIODIC_RECORDED[d]
    errors, faces = [], []
    for n in levels:
        e, f = _periodic_errors(d, n)
        errors.append(e)
        faces.append(f)
    orders = _orders(errors, np.array(levels) - 1)
    assert all(ORDER[0] <= o <= ORDER[1] for o in orders), (d, errors, orders)
    _assert_recorded(f"periodic_{d}d", errors, faces, recorded_errors, recorded_faces)
