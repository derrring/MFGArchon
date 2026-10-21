"""#1689 PR 4: ``monotonicity_scheme="positive_qp_fan"`` on reflecting faces -- the mirror row under the paper's
obliqueness certificate (eq:bc_gamma_main) and the value-copy row -- against an exact reflected reference, and what it
refuses (R-D3).

The oracle is exp17's construction with an asymmetric profile: ``u* = phi(x) + C (T - t)``, ``phi = a (q(x) + q(y))``,
``q(s) = s^3 - 1.5 s^2``, ``a = 0.5``, so ``d_n phi = 0`` on every edge (``q'(0) = q'(1) = 0``), and ``m*`` frozen. ``V``
absorbs the residual, so ``-u_t - nu Lap u + |grad u|^2 / 2 - V - zeta m* = 0`` holds exactly; ``psi`` is linear, so
implicit Euler is exact in time and the error is the spatial one. On exp17's Sobol clouds the scheme's error (B) is
compared with the no-mirror operator's (A), the plain rows with value-copy boundary rows, which is the mutant the pin
kills. Measured on #1689 (comment 6296766771, "Ps'", and the PR's ladder): at N_int 900 and 1600, on the box with every
face no-flux, B 5.0524e-3 and 2.7573e-3 against A 1.0458e-2 and 8.2644e-3; with x_max Dirichlet (u* its data), B
2.7067e-3 and 1.6068e-3 against A 6.5181e-3 and 8.0987e-3. Each threshold is the geometric mean of B and A, rounded to
two digits: B sits 1.4 to 2.2 times below it and A 1.4 to 2.2 times above. Those numbers are exp17's clouds' (Sobol
scramble seed 42): on other scrambles A is below a threshold, or below B, in some cases, so the pin also counts the
mirror rows, which depend on the cloud's geometry alone, and checks every row against the paper's QP solved here.
"""

from __future__ import annotations

import warnings

import pytest

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import i0
from scipy.stats import qmc

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.gfdm_components import GuardedFanParameters
from mfgarchon.alg.numerical.gfdm_components.positive_qp_fan import MIRROR_GAMMA0, PositiveQPFan
from mfgarchon.alg.numerical.hjb_solvers import HJBGFDMSolver, HJBHowardSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, dirichlet_bc, no_flux_bc

SIGMA, ZETA, C_TIME, T_END, NT = 0.5, 0.5, 1.0, 1.0, 10
NU = SIGMA**2 / 2
A_PHI = 0.5
#: exp17's normalisation of m*, kept so the numbers above reproduce; V absorbs any m*.
Z_M = float(i0(np.log(5.0) * NU / 4 / NU) ** 2)


def _phi(x):
    q, dq, d2q = (lambda s: s**3 - 1.5 * s**2), (lambda s: 3 * s**2 - 3 * s), (lambda s: 6 * s - 3)
    x = np.atleast_2d(np.asarray(x, dtype=float))
    return (
        A_PHI * (q(x[:, 0]) + q(x[:, 1])),
        A_PHI * (d2q(x[:, 0]) + d2q(x[:, 1])),
        A_PHI**2 * (dq(x[:, 0]) ** 2 + dq(x[:, 1]) ** 2),
    )


def _m_star(x):
    return np.exp(-_phi(x)[0] / NU) / Z_M


def _u_star(t, x):
    return _phi(x)[0] + C_TIME * (T_END - t)


def _potential(t, x):
    """V = C - nu Lap phi + |grad phi|^2 / 2 - zeta m*: a point gives a number, a batch an array."""
    _, lap, grad2 = _phi(x)
    v = C_TIME - NU * lap + 0.5 * grad2 - ZETA * _m_star(x)
    return float(v[0]) if np.ndim(x) == 1 else v


def _sobol(n_int, seed=42):
    """exp17's cloud: a scrambled Sobol interior and n_b = 2 sqrt(N_int) boundary nodes per face, corners excluded."""
    p = qmc.Sobol(d=2, scramble=True, seed=seed).random_base2(int(np.ceil(np.log2(n_int))))[:n_int]
    margin = 0.15 / np.sqrt(n_int)
    interior = margin + p * (1 - 2 * margin)
    nb = max(4, round(2 * np.sqrt(n_int)))
    s = (np.arange(nb) + 0.5) / nb
    z, o = np.zeros(nb), np.ones(nb)
    frame = np.vstack([np.c_[s, z], np.c_[s, o], np.c_[z, s], np.c_[o, s]])
    return np.vstack([interior, frame]), np.arange(n_int, n_int + len(frame))


def _bc(fixture):
    if fixture == "box":
        return no_flux_bc(dimension=2)
    return BoundaryConditions(
        segments=[
            BCSegment(
                name="exit", bc_type=BCType.DIRICHLET, boundary="x_max", value=lambda t, x: float(_u_star(t, x)[0])
            ),
            BCSegment(name="walls", bc_type=BCType.NO_FLUX),
        ],
        dimension=2,
    )


def _problem(bc, nt=NT):
    hamiltonian = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0), potential=_potential, coupling=lambda m: ZETA * m
    )
    grid = TensorProductGrid(bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[5, 5], boundary_conditions=bc)
    return MFGProblem(
        model=Model(hamiltonian=hamiltonian, volatility=SIGMA),
        domain=grid,
        conditions=Conditions(m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0, T=T_END),
        Nt=nt,
    )


def _solver(problem, points, boundary, **extra):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return HJBGFDMSolver(
            problem,
            collocation_points=points,
            boundary_indices=boundary,
            delta=0.15,
            monotonicity_scheme="positive_qp_fan",
            inner_solver="howard",
            **extra,
        )


@pytest.mark.parametrize(
    ("fixture", "n_int", "threshold", "mirrored"),
    [("box", 900, 7.3e-3, 130), ("box", 1600, 4.8e-3, 190), ("mixed", 900, 4.2e-3, 99), ("mixed", 1600, 3.6e-3, 143)],
)
def test_the_mirror_rows_close_a_reflecting_face_below_the_no_mirror_operator(
    fixture, n_int, threshold, mirrored, monkeypatch
):
    """B's L2 error at t = 0 over the interior nodes is below the threshold the module docstring derives, where the
    no-mirror operator's is above it. Its operator is certified: every interior row Metzler with zero row sum, every
    mirror row's gamma_hat at least MIRROR_GAMMA0, and every sampled node of a reflecting face a value-copy row. The
    mirror rows are as many as the paper's reach places, every interior node whose nearest reflecting face lies within
    3/5 of its pool's radius, x_max not mirrored where it is Dirichlet (R-D4). And every matrix Howard solves has the
    M-matrix sign pattern, boundary rows included: no off-diagonal entry is positive (review 1 of #2783, B3)."""
    from mfgarchon.alg.numerical.hjb_solvers import hjb_howard

    solved = []
    real_spsolve = hjb_howard.spsolve

    def spsolve(matrix, rhs):
        off = matrix.tocoo()
        solved.append(float(np.max(off.data[off.row != off.col], initial=0.0)))
        return real_spsolve(matrix, rhs)

    monkeypatch.setattr(hjb_howard, "spsolve", spsolve)
    points, boundary = _sobol(n_int)
    solver = _solver(_problem(_bc(fixture)), points, boundary)
    U = solver.solve_hjb_system(M_density=np.tile(_m_star(points), (NT + 1, 1)), U_terminal=_u_star(T_END, points))
    assert solved, "Howard solved no system"
    assert max(solved) <= 0.0, f"a solved matrix has a positive off-diagonal entry, {max(solved):.3e}"
    interior = np.setdiff1d(np.arange(len(points)), boundary)
    error = float(np.sqrt(np.mean((U[0] - _u_star(0.0, points))[interior] ** 2)))
    assert error < threshold, f"B's L2 error {error:.4e} is not below {threshold:.1e}"

    fan = solver._positive_qp_fan
    rows = fan.laplacian.tocsr()[interior].toarray()
    rows[np.arange(len(interior)), interior] = 0.0
    assert rows.min() >= 0.0
    assert np.abs(fan.laplacian.tocsr()[interior].sum(axis=1)).max() < 1e-10
    assert fan.report["n_mirror_rows"] == mirrored
    assert fan.report["mirror_gamma_hat_min"] >= MIRROR_GAMMA0 - 1e-8
    on_dirichlet = np.isclose(points[boundary, 0], 1.0) if fixture == "mixed" else np.zeros(len(boundary), bool)
    assert set(fan.copy_rows) == set(boundary[~on_dirichlet].tolist())


def test_each_mirror_row_is_the_papers_qp():
    """Every mirror row on exp17's N_int 900 cloud, every face no-flux, against the paper's QP solved here
    independently: over the node's 21 nearest nodes (the centre included) and their images across its nearest face,
    ``min ||L||^2`` subject to the P2 moments, ``L >= 0`` off the centre and ``2 delta_i sum_j L_ghost_ij s_ij >= 0.5``,
    with ``s_ij`` the source's inward distance to the face and ``delta_i`` the pool's radius (chap_01
    eq:bc_gamma_main), the ghosts then folded onto their sources. Solved in units of ``delta_i``. A different
    ``s_ij`` or ``delta_i`` moves the rows where the certificate binds."""
    import cvxpy as cp

    points, boundary = _sobol(900)
    interior = np.setdiff1d(np.arange(len(points)), boundary)
    fan = PositiveQPFan(
        points,
        interior,
        GuardedFanParameters(),
        reflecting_faces={(-1.0, 0.0): 0.0, (1.0, 0.0): 1.0, (0.0, -1.0): 0.0, (0.0, 1.0): 1.0},
        reflecting_nodes=boundary,
    )
    library = fan.laplacian.tocsr()
    tree = cKDTree(points)
    normals = [np.array(n) for n in ((-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0))]
    offsets = [0.0, 1.0, 0.0, 1.0]
    worst, checked = 0.0, 0
    for i in interior:
        _, pool = tree.query(points[i], k=21)
        sources = pool[pool != i]
        offset = points[sources] - points[i]
        delta = float(np.linalg.norm(offset, axis=1).max())
        inward = [c - n @ points[i] for n, c in zip(normals, offsets, strict=True)]
        k = int(np.argmin(inward))
        if not inward[k] < 0.6 * delta:
            continue
        n, c = normals[k], offsets[k]
        ghosts = points[sources] - 2.0 * ((points[sources] @ n) - c)[:, None] * n
        z = np.vstack([np.zeros((1, 2)), offset, ghosts - points[i]]) / delta
        taylor = np.column_stack(
            [np.ones(len(z)), z[:, 0], z[:, 1], z[:, 0] ** 2 / 2, z[:, 0] * z[:, 1], z[:, 1] ** 2 / 2]
        )
        m = len(sources)
        weights = cp.Variable(1 + 2 * m)
        s = (c - points[sources] @ n) / delta
        cp.Problem(
            cp.Minimize(cp.sum_squares(weights)),
            [taylor.T @ weights == np.array([0, 0, 0, 1, 0, 1.0]), weights[1:] >= 0, 2.0 * s @ weights[1 + m :] >= 0.5],
        ).solve(solver="CLARABEL")
        folded = np.r_[weights.value[0], weights.value[1 : 1 + m] + weights.value[1 + m :]] / delta**2
        row = library[i].toarray().ravel()[np.r_[i, sources]]
        worst = max(worst, float(np.abs(row - folded).max() / np.abs(folded).max()))
        checked += 1
    assert checked == fan.report["n_mirror_rows"] == 130
    assert worst < 1e-3, f"a mirror row differs from the paper's QP by {worst:.2e} of its largest weight"


def _frame_cloud():
    """PR 3's jittered cloud on [0, 20]^2 with a frame of boundary nodes, 225 interior nodes."""
    n_side, length = 15, 20.0
    s = np.linspace(0.0, 1.0, n_side + 1)[:-1]
    frame = length * np.column_stack(
        [
            np.concatenate([s, np.ones_like(s), 1.0 - s, np.zeros_like(s)]),
            np.concatenate([np.zeros_like(s), s, np.ones_like(s), 1.0 - s]),
        ]
    )
    margin = 0.5 * length / (n_side + 1)
    g = np.linspace(margin, length - margin, n_side)
    interior = np.column_stack([a.ravel() for a in np.meshgrid(g, g)])
    h = (length - 2 * margin) / (n_side - 1)
    interior += np.random.default_rng(42).uniform(-0.3 * h, 0.3 * h, interior.shape)
    points = np.vstack([interior, frame])
    return points, np.arange(len(interior), len(points))


def _frame_problem(bc, nt=4):
    grid = TensorProductGrid(bounds=[(0.0, 20.0), (0.0, 20.0)], Nx_points=[5, 5], boundary_conditions=bc)
    return MFGProblem(
        model=Model(hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost()), volatility=0.3),
        domain=grid,
        conditions=Conditions(m_initial=lambda x: 1.0 / 400.0, u_terminal=lambda x: 0.0, T=1.0),
        Nt=nt,
    )


@pytest.mark.parametrize(
    "row",
    [
        "an obstacle beside a reflecting face",
        "a boundary read from domain_sdf",
        "a nonzero Neumann datum",
        "a no-flux datum other than 0",
        "a Neumann datum given as a callable",
        "a Robin face",
        "a periodic face",
        "a door",
        "a layer node no mirror row places",
        "a layer node with no complete physical fan",
        "a reflecting face adopted after construction",
        "a reflecting face adopted after construction, solved by HJBHowardSolver directly",
        "a corner given a value-copy row after construction",
    ],
)
def test_the_reflecting_closure_refuses_what_it_cannot_close(row):
    """Each refusal names what is not ported (R-D3 of #1689 comment 6296766771): an obstacle or a door needs exp09's
    reflected pool, a curved boundary the mirror across the SDF's tangent plane, a nonzero Neumann or a Robin face the
    affine correction b_N(g). A layer node no mirror row places, and one with no complete physical fan (the fan over
    reflected rays is not ported), are refused by name; so is a BC adopted after construction whose reflecting faces,
    or whose value-copy points, are not the ones the operator was built with. A no-flux datum is read as a Neumann one
    is, since HJB-GFDM writes them as one row (review 1 of #2783, B1)."""
    from mfgarchon.geometry.boundary import neumann_bc, periodic_bc, robin_bc

    points, boundary = _frame_cloud()
    if row == "a layer node no mirror row places":
        # A node 0.003 off y = 0 whose 20 nearest nodes all lie on that face: they and their images all lie on y = 0,
        # so no weights meet the P2 moments in y, with or without the obliqueness certificate.
        wall = np.c_[np.linspace(9.99, 10.01, 24), np.zeros(24)]
        dense = np.vstack([points, wall, [[10.0, 0.003]]])
        faces = {(0.0, -1.0): 0.0, (0.0, 1.0): 20.0, (-1.0, 0.0): 0.0, (1.0, 0.0): 20.0}
        interior = np.r_[np.arange(boundary[0]), len(dense) - 1]
        with pytest.raises(ValueError, match=r"no mirror row across it .* the P2 moments cannot be met there"):
            PositiveQPFan(dense, interior, GuardedFanParameters(), reflecting_faces=faces)
        return
    if row == "an obstacle beside a reflecting face":
        from mfgarchon.geometry.implicit import Hypersphere

        with pytest.raises(ValueError, match=r"reflecting face beside an obstacle .*not ported \(#1689 R-D3\)"):
            _solver(
                _frame_problem(no_flux_bc(dimension=2)),
                points,
                boundary,
                obstacle_sdf=Hypersphere(center=[10.0, 10.0], radius=1.0).signed_distance,
            )
        return
    if row == "a layer node with no complete physical fan":
        with pytest.raises(ValueError, match=r"no fan with every gap at most 120.0 degrees; a mirror row's advection"):
            _solver(
                _frame_problem(no_flux_bc(dimension=2)),
                points,
                boundary,
                positive_qp_fan=GuardedFanParameters(widened_caps_deg=()),
            )
        return
    if row == "a corner given a value-copy row after construction":
        # The exit's corner (20, 0) is Dirichlet at equal priority; a floor on y_min outranking it takes the corner,
        # which then reflects, while every face keeps its condition.
        exit_ = BCSegment(name="exit", bc_type=BCType.DIRICHLET, boundary="x_max")
        walls = BCSegment(name="walls", bc_type=BCType.NO_FLUX)
        problem = _frame_problem(BoundaryConditions(segments=[exit_, walls], dimension=2))
        provider = _solver(problem, points, boundary)
        floor = BCSegment(name="floor", bc_type=BCType.NO_FLUX, boundary="y_min", priority=2)
        problem.geometry.set_boundary_conditions(BoundaryConditions(segments=[exit_, walls, floor], dimension=2))
        with pytest.raises(ValueError, match=r"now give 1 boundary point\(s\) another row than the operator was built"):
            provider.solve_hjb_system(M_density=np.zeros((5, len(points))), U_terminal=np.zeros(len(points)))
        return
    if row.startswith("a reflecting face adopted after construction"):
        problem = _frame_problem(dirichlet_bc(dimension=2))
        provider = _solver(problem, points, boundary)
        problem.geometry.set_boundary_conditions(no_flux_bc(dimension=2))
        refusal = r"now reflect on the faces .* the operator's mirror rows were built for \[\]"
        if row.endswith("directly"):
            provider._refresh_boundary_conditions_if_changed()  # the adoption the provider's own solve begins with
            howard = HJBHowardSolver(
                problem,
                stencil_provider=provider,
                discretisation="guarded_fan",
                control_cost=QuadraticControlCost(),
                use_provider_bc_rows=True,
            )
            with pytest.raises(ValueError, match=refusal):
                howard.solve_hjb_system(np.zeros((5, len(points))), np.zeros(len(points)))
        else:
            with pytest.raises(ValueError, match=refusal):
                provider.solve_hjb_system(M_density=np.zeros((5, len(points))), U_terminal=np.zeros(len(points)))
        return
    bc, refusal = {
        "a boundary read from domain_sdf": (
            BoundaryConditions(
                segments=[BCSegment(name="walls", bc_type=BCType.NO_FLUX)],
                dimension=2,
                domain_bounds=np.array([[0.0, 20.0], [0.0, 20.0]]),
                domain_sdf=lambda x: float(np.max(np.abs(np.asarray(x) - 10.0)) - 10.0),
            ),
            r"a reflecting point on a boundary read from domain_sdf",
        ),
        "a nonzero Neumann datum": (
            neumann_bc(value=0.3, dimension=2),
            r"a neumann datum other than the constant 0 \(0.3\): only g = 0 reflects",
        ),
        "a no-flux datum other than 0": (
            BoundaryConditions(segments=[BCSegment(name="walls", bc_type=BCType.NO_FLUX, value=0.3)], dimension=2),
            r"a no_flux datum other than the constant 0 \(0.3\)",
        ),
        "a Neumann datum given as a callable": (
            neumann_bc(value=lambda t, x: 0.0, dimension=2),
            r"a neumann datum other than the constant 0",
        ),
        "a Robin face": (robin_bc(value=0.0, dimension=2), r"a robin condition, which the scheme has no row for"),
        "a periodic face": (periodic_bc(dimension=2), r"periodic"),
        "a door": (
            BoundaryConditions(
                segments=[
                    BCSegment(name="door", bc_type=BCType.DIRICHLET, boundary="x_max", region={"y": (6.0, 14.0)}),
                    BCSegment(name="walls", bc_type=BCType.NO_FLUX),
                ],
                dimension=2,
            ),
            r"carry both a Dirichlet and a reflecting condition: \[\(0, 1\)\]",
        ),
    }[row]
    with pytest.raises(ValueError, match=refusal):
        _solver(_frame_problem(bc), points, boundary)
