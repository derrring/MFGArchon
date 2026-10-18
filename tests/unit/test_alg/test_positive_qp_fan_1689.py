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
from scipy.stats import qmc

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.hjb_solvers import HJBGFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions

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


def _mid_time_errors(n_interior, kind):
    points, boundary = _cloud(n_interior, kind)
    nt = round(float(120 * np.sqrt(n_interior / 225)))
    solver = HJBGFDMSolver(
        _lq_problem(nt),
        collocation_points=points,
        boundary_indices=boundary,
        delta=4.0,
        monotonicity_scheme="positive_qp_fan",
        inner_solver="howard",
    )
    U = solver.solve_hjb_system(M_density=np.zeros((nt + 1, len(points))), U_terminal=_u_star(T_END, points))
    interior = np.setdiff1d(np.arange(len(points)), boundary)
    err = np.abs(U[nt // 2] - _u_star((nt // 2) * T_END / nt, points))[interior]
    return float(np.sqrt(np.mean(err**2)) * L_BOX), solver


@pytest.mark.parametrize("kind", ["sobol", "jittered"])
def test_positive_qp_fan_converges_at_first_order_to_the_closed_form_lq_value(kind):
    """Between N_int = 225 and 400 the mid-time L2 error falls at an order of at least 0.6, and every interior row of
    the operator has non-negative off-centre Laplacian weights. Discrimination is shown in the PR by mutation."""
    (e_coarse, solver), (e_fine, _) = _mid_time_errors(225, kind), _mid_time_errors(400, kind)
    order = np.log(e_coarse / e_fine) / np.log(np.sqrt(L_BOX**2 / 225) / np.sqrt(L_BOX**2 / 400))
    assert order >= 0.6, f"{kind}: EOC {order:.3f} from L2 errors {e_coarse:.4e} -> {e_fine:.4e}"
    laplacian = solver._positive_qp_fan.laplacian.tocoo()
    off = laplacian.row != laplacian.col
    assert laplacian.data[off].min() >= 0.0


@pytest.mark.parametrize(
    "row",
    ["one dimension", "an L1 control cost", "the Newton inner solver", "a no-flux face", "a fan that cannot close"],
)
def test_positive_qp_fan_refuses_what_it_cannot_make_monotone(row):
    """Each refusal names what it cannot do (D5 and P3-D1 of #1689 comment 6246915117)."""
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
    kwargs = {
        "collocation_points": points,
        "boundary_indices": boundary,
        "delta": 4.0,
        "monotonicity_scheme": "positive_qp_fan",
    }
    if row == "an L1 control cost":
        from mfgarchon.core.hamiltonian import L1ControlCost

        with pytest.raises(ValueError, match=r"QuadraticControlCost.*joint_socp"):
            HJBGFDMSolver(_lq_problem(4, control_cost=L1ControlCost()), inner_solver="howard", **kwargs)
        return
    if row == "the Newton inner solver":
        with pytest.raises(ValueError, match=r"inner_solver='howard'"):
            HJBGFDMSolver(_lq_problem(4), inner_solver="newton", **kwargs)
        return
    if row == "a no-flux face":
        from mfgarchon.geometry.boundary import no_flux_bc

        with pytest.raises(ValueError, match=r"not M-matrix rows"):
            HJBGFDMSolver(_lq_problem(4), inner_solver="howard", boundary_conditions=no_flux_bc(dimension=2), **kwargs)
        return
    # Without the widened caps, the nodes within about 0.3h of the frame have no fan with every gap below 120 degrees.
    from mfgarchon.alg.numerical.gfdm_components.positive_qp_fan import GuardedFanParameters

    with pytest.raises(ValueError, match=r"interior row\(s\) cannot be built.*no fan with every gap at most 120"):
        HJBGFDMSolver(
            _lq_problem(4), inner_solver="howard", positive_qp_fan=GuardedFanParameters(widened_caps_deg=()), **kwargs
        )
