"""#1689 PR 3: ``monotonicity_scheme="positive_qp_fan"`` -- the positive-QP Laplacian, the guarded fan and Howard's
sectorwise policy -- against the decoupled LQ HJB in closed form, and the configurations it refuses.

The oracle is exp14's problem, on which the research measured this operator: ``-u_t + |grad u|^2 / (2c) -
(sigma^2/2) Lap u = 0`` on [0, 20]^2 with ``u(T, x) = G |x - (20, 20)|^2``, solved by
``u* = P(t) |x - (20, 20)|^2 + Q(t)`` with ``P = 1 / (1/G + 2 (T - t) / c)`` and ``Q = sigma^2 c ln(1 + 2 G (T - t) / c)``
(substitution gives ``P' = 2 P^2 / c`` and ``Q' = -2 sigma^2 P``). The frame is Dirichlet, valued ``u*``. A
first-order monotone scheme: the research measured EOC 0.867 (Sobol) and 0.899 (jittered) in L2 over five levels
(exp15 ``results/lq_posqp_fan_published-levels_converted_20261014_170130.h5``); this pin asks EOC >= 0.6 between
N_int = 225 and 400, pre-registered on #1689 (comment 6246915117, ruled).
"""

from __future__ import annotations

import warnings

import pytest

import numpy as np
from scipy.sparse import eye
from scipy.stats import qmc

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.gfdm_components import GuardedFanParameters
from mfgarchon.alg.numerical.gfdm_components.positive_qp_fan import PositiveQPFan, cone_policy
from mfgarchon.alg.numerical.hjb_solvers import HJBGFDMSolver, HJBHowardSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, no_flux_bc

L_BOX, T_END, SIGMA, COST, G_TERM = 20.0, 4.0, 0.3, 1.0, 0.5


def _u_star(t, x):
    x = np.atleast_2d(np.asarray(x, dtype=float))
    p = 1.0 / (1.0 / G_TERM + 2.0 * (T_END - t) / COST)
    q = SIGMA**2 * COST * np.log(1.0 + 2.0 * G_TERM * (T_END - t) / COST)
    return p * ((x[:, 0] - L_BOX) ** 2 + (x[:, 1] - L_BOX) ** 2) + q


def _cloud(n_interior, kind, seed=42):
    """exp14's cloud: a Dirichlet frame of n_side nodes per edge around a Sobol or jittered interior."""
    n_side = round(float(np.sqrt(n_interior)))
    s = np.linspace(0.0, 1.0, n_side + 1)[:-1]
    frame = L_BOX * np.column_stack(
        [
            np.concatenate([s, np.ones_like(s), 1.0 - s, np.zeros_like(s)]),
            np.concatenate([np.zeros_like(s), s, np.ones_like(s), 1.0 - s]),
        ]
    )
    margin = 0.5 * L_BOX / (n_side + 1)
    if kind == "sobol":
        with warnings.catch_warnings():
            # scipy's note that a Sobol sample of 225 points is not balanced: exp14 drew these points so.
            warnings.filterwarnings("ignore", message="The balance properties of Sobol")
            interior = margin + qmc.Sobol(d=2, seed=seed).random(n_interior) * (L_BOX - 2 * margin)
    else:
        g = np.linspace(margin, L_BOX - margin, n_side)
        interior = np.column_stack([a.ravel() for a in np.meshgrid(g, g)])
        h = (L_BOX - 2 * margin) / (n_side - 1)
        interior += np.random.default_rng(seed).uniform(-0.3 * h, 0.3 * h, interior.shape)
    points = np.vstack([interior, frame])
    return points, np.arange(n_interior, len(points))


def _lq_problem(nt, control_cost=None, hamiltonian=None):
    bc = BoundaryConditions(
        segments=[BCSegment(name="frame", bc_type=BCType.DIRICHLET, value=lambda t, x: float(_u_star(t, x)[0]))],
        dimension=2,
        domain_bounds=np.array([[0.0, L_BOX], [0.0, L_BOX]]),
    )
    grid = TensorProductGrid(bounds=[(0.0, L_BOX), (0.0, L_BOX)], Nx_points=[5, 5], boundary_conditions=bc)
    hamiltonian = hamiltonian or SeparableHamiltonian(control_cost=control_cost or QuadraticControlCost(lambda_=COST))
    return MFGProblem(
        model=Model(hamiltonian=hamiltonian, volatility=SIGMA),
        domain=grid,
        conditions=Conditions(m_initial=lambda x: 1.0 / L_BOX**2, u_terminal=lambda x: 0.0, T=T_END),
        Nt=nt,
    )


def _solve(problem, points, boundary, nt, terminal, **extra):
    solver = HJBGFDMSolver(
        problem,
        collocation_points=points,
        boundary_indices=boundary,
        delta=4.0,
        monotonicity_scheme="positive_qp_fan",
        inner_solver="howard",
        **extra,
    )
    return solver, solver.solve_hjb_system(M_density=np.zeros((nt + 1, len(points))), U_terminal=terminal)


def _sector_hamiltonian(u, sectors, lam):
    """Per node, the minimum over its sectors and over ``l >= 0`` of ``l . (u_j0 - u_i, u_j1 - u_i) + lam/2 |Z l|^2``,
    ``Z = [z0 z1]``: the discrete ``min_alpha alpha . grad u + L(alpha)`` over the fan. Found here by enumerating the
    KKT points in the rays' coordinates (both rays, either ray alone, the apex), not by ``cone_policy``'s projection
    of ``optimal_control``, so a wrong sector or a wrong projection in the scheme's policy shows in the residual."""
    node = sectors["node"]
    du = np.stack([u[sectors["j0"]] - u[node], u[sectors["j1"]] - u[node]], axis=1)
    z = np.stack([sectors["z0"], sectors["z1"]], axis=2)
    gram = np.einsum("mki,mkj->mij", z, z)

    def value(coords):
        return np.einsum("mi,mi->m", coords, du) + 0.5 * lam * np.einsum("mi,mij,mj->m", coords, gram, coords)

    both = np.linalg.solve(lam * gram, -du[:, :, None])[:, :, 0]
    on0 = np.column_stack([np.maximum(0.0, -du[:, 0] / (lam * gram[:, 0, 0])), np.zeros(len(du))])
    on1 = np.column_stack([np.zeros(len(du)), np.maximum(0.0, -du[:, 1] / (lam * gram[:, 1, 1]))])
    best = np.minimum(np.minimum(value(on0), value(on1)), 0.0)
    best = np.where(np.all(both >= 0.0, axis=1), np.minimum(best, value(both)), best)
    hamiltonian = np.full(len(u), np.inf)
    np.minimum.at(hamiltonian, node, best)
    return hamiltonian


def _lq_run(n_interior, kind):
    points, boundary = _cloud(n_interior, kind)
    nt = round(float(120 * np.sqrt(n_interior / 225)))
    solver, U = _solve(_lq_problem(nt), points, boundary, nt, _u_star(T_END, points))
    interior = np.setdiff1d(np.arange(len(points)), boundary)
    err = np.abs(U[nt // 2] - _u_star((nt // 2) * T_END / nt, points))[interior]
    return float(np.sqrt(np.mean(err**2)) * L_BOX), solver, U, nt, points, interior


@pytest.mark.parametrize("kind", ["sobol", "jittered"])
def test_positive_qp_fan_converges_at_first_order_to_the_closed_form_lq_value(kind):
    """Between N_int = 225 and 400 the mid-time L2 error falls at an order of at least 0.6. And on the operator that
    reached it, at N_int = 225:

    - the Laplacian is exact on P2, the diffusion half the error rate alone cannot see (#2750 review 1 measured the
      rate green with the diffusion flipped, dropped or scaled ten-fold);
    - the policy-evaluation matrix's parts at the mid-time policy, ``I/dt - A_adv - (sigma^2/2) L`` built here from
      the operator, have non-positive interior off-diagonals. This reads the parts, not the matrix Howard assembles;
      the residual below ties the solve to them;
    - every step solves the discrete Bellman equation, with the minimum over each node's fan sectors taken here
      (`_sector_hamiltonian`) and not from the scheme's policy: the largest interior residual over the steps is at
      most 1e-7 of the largest |u_next / dt|."""
    (e_coarse, solver, U, nt, points, interior), (e_fine, *_) = _lq_run(225, kind), _lq_run(400, kind)
    order = np.log(e_coarse / e_fine) / np.log(np.sqrt(L_BOX**2 / 225) / np.sqrt(L_BOX**2 / 400))
    assert order >= 0.6, f"{kind}: EOC {order:.3f} from L2 errors {e_coarse:.4e} -> {e_fine:.4e}"

    fan, n = solver._positive_qp_fan, len(points)
    x, y = points[:, 0], points[:, 1]
    for f, lap in ((np.ones(n), 0.0), (x, 0.0), (y, 0.0), (x**2, 2.0), (x * y, 0.0), (y**2, 2.0)):
        assert np.max(np.abs((fan.laplacian @ f)[interior] - lap)) <= 1e-6

    cost, dt, diffusion = QuadraticControlCost(lambda_=COST), T_END / nt, 0.5 * SIGMA**2
    policy = cone_policy(U[nt // 2], fan.sectors, cost)
    rows = (eye(n, format="csr") / dt - policy.advection(n) - diffusion * fan.laplacian)[interior].tocoo()
    assert rows.data[interior[rows.row] != rows.col].max() <= 0.0

    worst, scale = 0.0, 0.0
    for k in range(nt):
        residual = (
            (U[k] - U[k + 1]) / dt - _sector_hamiltonian(U[k], fan.sectors, COST) - diffusion * (fan.laplacian @ U[k])
        )
        worst = max(worst, float(np.max(np.abs(residual[interior]))))
        scale = max(scale, float(np.max(np.abs(U[k + 1][interior] / dt))))
    assert worst <= 1e-7 * scale, f"{kind}: Bellman residual {worst:.3e} against max|u_next / dt| {scale:.3e}"


@pytest.mark.parametrize("lam", [0.25, 4.0])
def test_positive_qp_fan_is_exact_on_a_linear_value_at_any_control_cost(lam):
    """``u* = a.x + 1 - (|a|^2 / (2 lam) - V)(T - t)`` solves ``-u_t + |grad u|^2 / (2 lam) - V - (sigma^2/2) Lap u = 0``
    with V = 0.5, a potential that reaches the solve through the running cost. The scheme is exact on it: a linear u's
    sector gradient is ``a``, the projection of ``-a / lam`` onto its sector is exact, and L is exact on P2. #2750's
    round 2 measured max|U - u*| 1.6e-12 (lam 0.25) and 7.8e-12 (lam 4) here; with the control cost's lambda set to 1
    in the policy and the Lagrangian, 0.78 and 0.19; in the policy alone, 0.59 and 7.6e+111."""
    a, potential = np.array([0.3, -0.2]), 0.5

    def u_linear(t, x):
        return np.atleast_2d(np.asarray(x, dtype=float)) @ a + 1.0 - (a @ a / (2.0 * lam) - potential) * (T_END - t)

    nt = 20
    bc = BoundaryConditions(
        segments=[BCSegment(name="frame", bc_type=BCType.DIRICHLET, value=lambda t, x: float(u_linear(t, x)[0]))],
        dimension=2,
        domain_bounds=np.array([[0.0, L_BOX], [0.0, L_BOX]]),
    )
    grid = TensorProductGrid(bounds=[(0.0, L_BOX), (0.0, L_BOX)], Nx_points=[5, 5], boundary_conditions=bc)
    hamiltonian = SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=lam), potential=lambda t, x: potential)
    problem = MFGProblem(
        model=Model(hamiltonian=hamiltonian, volatility=SIGMA),
        domain=grid,
        conditions=Conditions(m_initial=lambda x: 1.0 / L_BOX**2, u_terminal=lambda x: 0.0, T=T_END),
        Nt=nt,
    )
    points, boundary = _cloud(225, "jittered")
    _, U = _solve(problem, points, boundary, nt, u_linear(T_END, points))
    for k in range(nt + 1):
        assert np.max(np.abs(U[k] - u_linear(k * T_END / nt, points))) <= 1e-8


@pytest.mark.parametrize(
    "row",
    [
        "one dimension",
        "an L1 control cost",
        "a congestion Hamiltonian",
        "the Newton inner solver",
        "ghost nodes",
        "positive_qp_fan= with another scheme",
        "a no-flux face",
        "a no-flux face with degenerate stencils",
        "a no-flux face set after construction",
        "a no-flux face adopted after construction, solved by HJBHowardSolver directly",
        "a fan that cannot close",
    ],
)
def test_positive_qp_fan_refuses_what_it_cannot_make_monotone(row):
    """Each refusal names what it cannot do (D5 and P3-D1 of #1689 comment 6246915117). A boundary point of any other
    condition than Dirichlet is refused by its type: at construction, also where its stencil is degenerate and its
    row an identity, and where HJBHowardSolver installs the BC rows, for a condition set on the grid after
    construction (#2750 review 1), on HJBGFDMSolver's route and on HJBHowardSolver's own (#2750 review 2)."""
    if row == "one dimension":
        from mfgarchon.geometry.boundary import dirichlet_bc

        grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=dirichlet_bc(dimension=1))
        problem = MFGProblem(
            model=Model(hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost()), volatility=SIGMA),
            domain=grid,
            conditions=Conditions(m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0, T=1.0),
            Nt=4,
        )
        with pytest.raises(ValueError, match=r"two-dimensional"):
            HJBGFDMSolver(
                problem,
                collocation_points=np.linspace(0, 1, 11)[:, None],
                monotonicity_scheme="positive_qp_fan",
                inner_solver="howard",
            )
        return
    points, boundary = _cloud(225, "jittered")
    kwargs = {"collocation_points": points, "boundary_indices": boundary, "monotonicity_scheme": "positive_qp_fan"}
    with_howard = {**kwargs, "delta": 4.0, "inner_solver": "howard"}
    if row == "an L1 control cost":
        from mfgarchon.core.hamiltonian import L1ControlCost

        with pytest.raises(ValueError, match=r"QuadraticControlCost.*joint_socp"):
            HJBGFDMSolver(_lq_problem(4, control_cost=L1ControlCost()), **with_howard)
    elif row == "a congestion Hamiltonian":
        from mfgarchon.core.hamiltonian import CongestionHamiltonian

        congestion = CongestionHamiltonian(control_cost=QuadraticControlCost(), congestion_factor=lambda m: 1.0 + m)
        with pytest.raises(ValueError, match=r"SeparableHamiltonian with a QuadraticControlCost"):
            HJBGFDMSolver(_lq_problem(4, hamiltonian=congestion), **with_howard)
    elif row == "the Newton inner solver":
        with pytest.raises(ValueError, match=r"inner_solver='howard'"):
            HJBGFDMSolver(_lq_problem(4), **{**with_howard, "inner_solver": "newton"})
    elif row == "ghost nodes":
        with pytest.raises(ValueError, match=r"use_ghost_nodes=True\) is refused"):
            HJBGFDMSolver(_lq_problem(4), use_ghost_nodes=True, **with_howard)
    elif row == "positive_qp_fan= with another scheme":
        with pytest.raises(ValueError, match=r"read only by monotonicity_scheme='positive_qp_fan'"):
            HJBGFDMSolver(
                _lq_problem(4), positive_qp_fan=GuardedFanParameters(), **{**with_howard, "monotonicity_scheme": "none"}
            )
    elif row == "a no-flux face":
        with pytest.raises(ValueError, match=r"condition other than Dirichlet"):
            HJBGFDMSolver(_lq_problem(4), boundary_conditions=no_flux_bc(dimension=2), **with_howard)
    elif row == "a no-flux face with degenerate stencils":
        # A radius of 0.5 on a 1.33 spacing leaves the stencils degenerate, and the no-flux rows identities; the
        # operator says so, and the scheme refuses the face.
        with (
            pytest.warns(UserWarning, match=r"degenerate stencils"),
            pytest.raises(ValueError, match=r"condition other than Dirichlet"),
        ):
            HJBGFDMSolver(
                _lq_problem(4),
                boundary_conditions=no_flux_bc(dimension=2),
                **{**with_howard, "delta": 0.5, "neighborhood_mode": "radius"},
            )
    elif row == "a no-flux face set after construction":
        problem = _lq_problem(4)
        solver = HJBGFDMSolver(problem, **with_howard)
        problem.geometry.set_boundary_conditions(no_flux_bc(dimension=2))
        with pytest.raises(ValueError, match=r"condition other than Dirichlet"):
            solver.solve_hjb_system(M_density=np.zeros((5, len(points))), U_terminal=_u_star(T_END, points))
    elif row == "a no-flux face adopted after construction, solved by HJBHowardSolver directly":
        problem = _lq_problem(4)
        provider = HJBGFDMSolver(problem, **with_howard)
        problem.geometry.set_boundary_conditions(no_flux_bc(dimension=2))
        provider._refresh_boundary_conditions_if_changed()  # the adoption the provider's own solve begins with
        howard = HJBHowardSolver(
            problem,
            stencil_provider=provider,
            discretisation="guarded_fan",
            control_cost=QuadraticControlCost(lambda_=COST),
            use_provider_bc_rows=True,
        )
        with pytest.raises(ValueError, match=r"condition other than Dirichlet"):
            howard.solve_hjb_system(np.zeros((5, len(points))), _u_star(T_END, points))
    else:
        # Without the widened caps, the nodes within about 0.3h of the frame have no fan with every gap below 120.
        with pytest.raises(ValueError, match=r"interior row\(s\) cannot be built.*no fan with every gap at most 120"):
            HJBGFDMSolver(_lq_problem(4), positive_qp_fan=GuardedFanParameters(widened_caps_deg=()), **with_howard)


def test_a_dirichlet_frame_valued_by_a_provider_constructs():
    """The scheme's boundary check reads each point's condition type and evaluates no value, so a Dirichlet frame whose
    value a provider resolves per iteration is admitted (#2750 review 1)."""
    from mfgarchon.geometry.boundary import ConstantProvider

    bc = BoundaryConditions(
        segments=[BCSegment(name="frame", bc_type=BCType.DIRICHLET, value=ConstantProvider(0.0))],
        dimension=2,
        domain_bounds=np.array([[0.0, L_BOX], [0.0, L_BOX]]),
    )
    points, boundary = _cloud(225, "jittered")
    HJBGFDMSolver(
        _lq_problem(4),
        collocation_points=points,
        boundary_indices=boundary,
        boundary_conditions=bc,
        delta=4.0,
        monotonicity_scheme="positive_qp_fan",
        inner_solver="howard",
    )


def test_the_positive_laplacian_scales_with_the_cloud():
    """The cloud scaled from a box of side 20 to one of side 1e-3 gives the same fans and the Laplacian scaled by
    (20 / 1e-3)^2: the row's QP is solved in units of the node's h6. Solved in the cloud's own units, CLARABEL called
    every row of the small box infeasible and the operator refused all 225 (#2750 review 2)."""
    points, _ = _cloud(225, "jittered")
    interior = np.arange(225)
    big = PositiveQPFan(points, interior, GuardedFanParameters())
    small = PositiveQPFan(points * (1e-3 / L_BOX), interior, GuardedFanParameters())
    for key in ("node", "j0", "j1"):
        np.testing.assert_array_equal(small.sectors[key], big.sectors[key])
    scaled = small.laplacian * (1e-3 / L_BOX) ** 2
    assert abs(scaled - big.laplacian).max() <= 1e-10 * abs(big.laplacian).max()
