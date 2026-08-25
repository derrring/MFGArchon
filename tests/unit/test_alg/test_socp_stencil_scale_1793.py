"""The joint-SOCP stencil scale `h_i` cannot be checked by the diagnostic it defines.

Issue #1793. `h_i` non-dimensionalises the monotonicity cone -- the constraint is
``||D_j||_2 <= (C / h_i) * L_j`` (`joint_socp.py`, in `solve_joint_socp_at_stencil`) -- and it
also appears in the per-edge ratio reported about that constraint, ``kappa_j = h_i * ||D_j|| / L_j``.
Inflate `h_i` and the constraint tightens, the optimum's `||D_j||` shrinks by the same factor, and
kappa comes back inside the bound looking healthy.

**A quantity that appears in both the constraint and the diagnostic reported about that constraint
cannot be checked by a BOUND on that diagnostic.** It can be checked by a PIN on it, and the two
tests below do different jobs -- neither is sufficient alone:

- `test_a_uniform_cross_returns_the_analytic_weights` is an external oracle. It says the
  construction is right, against weights known independently of the solver. It **cannot** catch a
  change to `h_i`, because on a uniform stencil `median == max` and every candidate scale agrees.
- `test_the_stencil_scale_is_pinned_on_a_non_uniform_stencil` catches the change. It uses a
  geometry where the two differ by 3x and pins the reported kappa.

Measured while writing these, through `PrecomputedJointSocpStencils` (the path that computes `h_i`,
which a direct `solve_joint_socp_at_stencil` call bypasses by taking it as an argument):

    h_i           kappa_max        L weights
    median        0.9999999108     -- current
    max           0.9496925650     max|dL| relative 0.018%

**kappa moves 5.03%; the Laplacian weights move 0.018%.** So kappa is the discriminator and L is
not -- pinning L would be pinning solver noise. That is the non-obvious half of this file.

Two mechanisms the issue proposed that do NOT work here, recorded so they are not retried:

- **Feasibility does not flip.** The issue reasoned that stencils the paper's C admits would come
  back infeasible. Measured over five geometries (cross-plus-outlier, one-sided fan, near-cluster,
  collinear-biased, narrow cone) at C in {0.8, 1.0, 1.5}: each is either feasible under both scales
  or infeasible under both. `via` is `socp_clarabel` in both cases here, so the fall-through to the
  relaxed penalised SOCP -- the step that would make `eps_M > 0` -- never happens either.
- **A uniform-stencil oracle alone.** `median == max` there by construction.
"""

from __future__ import annotations

import pytest

import numpy as np

pytest.importorskip("cvxpy", reason="joint_socp requires cvxpy (the `numerical` extra)")

from mfgarchon.alg.numerical.gfdm_components.joint_socp import (
    PrecomputedJointSocpStencils,
    build_taylor_matrix_2d,
    solve_joint_socp_at_stencil,
    wendland_stencil_weights,
)

H = 0.1


def test_a_uniform_cross_returns_the_analytic_weights():
    """External oracle: on a uniform 2D cross the exactly-monotone Laplacian weights are known.

    (1, 1, 1, 1, -4) / h^2 is the standard five-point stencil, and it is already an M-matrix, so a
    construction that claims exact monotonicity must return it without relaxation. This is computed
    independently of the solver, which is what makes it an oracle rather than a pin.

    It says nothing about `h_i`: on this geometry every neighbour is at distance h, so median, max
    and mean agree and the scale is unambiguous. That is the other test's job.
    """
    offsets = np.array([[0.0, 0.0], [H, 0.0], [-H, 0.0], [0.0, H], [0.0, -H]])
    A, _ = build_taylor_matrix_2d(offsets)
    dists = np.linalg.norm(offsets, axis=1)
    nz = dists[dists > 1e-12]
    assert np.isclose(np.median(nz), np.max(nz)), "this fixture must be uniform, or it pins the scale too"

    result = solve_joint_socp_at_stencil(
        A, 0, float(np.median(nz)), C=1.0, dimension=2, wendland_w=wendland_stencil_weights(offsets, delta=4 * H)
    )
    assert result["status"] == "feasible", result

    analytic = np.array([-4.0, 1.0, 1.0, 1.0, 1.0]) / H**2
    np.testing.assert_allclose(np.asarray(result["L"]), analytic, rtol=1e-9, atol=1e-9)


def _non_uniform_stencil_kappa() -> float:
    """A centre, a four-way cross at h, and one neighbour at 3h -- the k-NN-fallback shape.

    `median(nz) = h` and `max(nz) = 3h`, a 3x spread, which is the inflation #1793's mutation
    produces. Routed through `PrecomputedJointSocpStencils` deliberately: it is the object that
    COMPUTES `h_i`, and a direct `solve_joint_socp_at_stencil` call takes the scale as an argument
    and so cannot see a change to how it is derived.
    """
    points = np.array([[0.0, 0.0], [H, 0.0], [-H, 0.0], [0.0, H], [0.0, -H], [3 * H, 0.0]])
    dists = np.linalg.norm(points - points[0], axis=1)
    nz = dists[dists > 1e-12]
    assert np.max(nz) / np.median(nz) == pytest.approx(3.0), "the fixture must keep its 3x spread"

    pre = PrecomputedJointSocpStencils(
        points=points,
        interior_indices=np.array([0]),
        delta=4 * H,
        neighborhoods={0: {"indices": np.arange(len(points))}},
        cone_constant_C=1.0,
    )
    data = pre.stencils[0]
    assert data.via == "socp_clarabel", f"the fast path would bypass the scale entirely: via={data.via}"
    return float(data.kappa_max)


def test_the_stencil_scale_is_pinned_on_a_non_uniform_stencil():
    """Pins the reported kappa at a geometry where candidate scales disagree by 3x.

    This is a PIN, and it pins a value the oracle above establishes the construction can produce --
    not a value of unknown correctness. What it defends is that nobody changes the scale silently,
    including in a way that leaves every bound check green: under `max` the reported kappa is
    0.9497, comfortably inside C = 1.0, so nothing about the output looks wrong.

    Tolerance: the solve hits the C = 1.0 bound to 1e-7, and the mutation moves kappa by 5%. 1e-4
    is three orders above the solver's noise and three orders below the signal.
    """
    assert _non_uniform_stencil_kappa() == pytest.approx(0.9999999108, rel=1e-4)


def test_the_pin_would_not_have_worked_on_the_laplacian_weights():
    """The negative control for the choice of quantity, and the reason this file pins kappa.

    Under the mutation the Laplacian weights move by 0.018% relative -- close enough to solver noise
    that a pin on them would be a flake generator, and far too small to distinguish a 3x change of
    scale. Asserting the weights are stable to 1% documents that they are NOT the discriminator, so
    a later reader does not "strengthen" this file by pinning them instead.
    """
    points = np.array([[0.0, 0.0], [H, 0.0], [-H, 0.0], [0.0, H], [0.0, -H], [3 * H, 0.0]])
    pre = PrecomputedJointSocpStencils(
        points=points,
        interior_indices=np.array([0]),
        delta=4 * H,
        neighborhoods={0: {"indices": np.arange(len(points))}},
        cone_constant_C=1.0,
    )
    L = np.asarray(pre.stencils[0].L)
    expected = np.array([-2.6671e02, 3.5361e-02, 5.0018e01, 1.0000e02, 1.0000e02, 1.6661e01])
    assert np.abs(L - expected).max() / np.abs(L).max() < 1e-2
