"""The joint-SOCP stencil scale `h_i` is pinned through the constraint, because the diagnostic
that reports on it cannot pin it at any `C`.

Issue #1793. `h_i = median(neighbour distances)` non-dimensionalises the monotonicity cone. It
enters the solve in two distinct places:

    (a) the CONSTRAINT   ||D_j||_2 <= (C / h_i) * L_j        -- shapes the solution
    (b) the DIAGNOSTIC   kappa_j = h_i * ||D_j||_2 / L_j     -- reports on it, exposed as kappa_max

A 25-axis mutation sweep found the suite blind to the scale entirely: 5770 passed with `median`
replaced by `max`. This file pins (a). It does not pin (b), and the reason is a measurement, not
an omission.

WHY `kappa_max` CANNOT PIN THE SCALE AT ANY `C` -- MEASURED, AND IT COST TWO REVIEW ROUNDS
-------------------------------------------------------------------------------------------
`kappa_max` is the only scale-carrying quantity `JointSocpStencilData` exposes, so it is the
obvious probe, and earlier versions of this file pinned it. It is unusable, because the two
conditions that would make it meaningful are mutually exclusive on any fixture:

    C     kappa_max vs C   L_1 (the argmax edge)   kappa_max separation, min/mean/max vs median
    1.0      SLACK             9.88e-08                3.705% /  1.327% / 21.767%
    1.5      SLACK             3.16e-08                0.768% /  1.683% /  5.164%
    2.0      binds             5.48e-02                0.000% /  0.241% /  2.199%
    4.0      binds             9.51e-02                0.000% /  0.000% /  0.303%
    8.0      binds             6.46e-02                0.000% /  0.000% /  0.000%

- **Where the cone binds, `kappa_max` IS `C`** by construction and carries nothing -- at
  production's `C = 8.0` it agrees across all four candidate scales to five decimal places.
- **Where the cone is slack, the argmax edge is the one the optimiser DELETES.** Nothing binds, so
  the objective drives an x-axis weight onto the `eps_pos = 0.0` bound: `L_1 = 9.88e-08` against
  `||D_1|| = 4.76e-07`. The exact optimum is `L_1 = ||D_1|| = 0`, so `kappa_1 = h_i * ||D_1|| / L_1`
  is **0/0** -- whichever point CLARABEL's central path is carrying when it stops. Tightening the
  solver tracks it one-for-one (`L_1` = 9.87e-06, 9.88e-10, 9.89e-12, 1.03e-13 at tol 1e-6, 1e-9,
  1e-12, 1e-14) until it crosses the `L_j <= 1e-12 -> inf` guard at `joint_socp.py:307` and the pin
  fails as `inf`. **Tightening the solver breaks it**, the opposite of the direction anyone hardens
  against.

Slack is exactly the condition under which some edge gets zeroed, so there is no `C` in between.
The separations in the third column are differences between indeterminate forms.

WHAT IS PINNED INSTEAD
-----------------------
At `C = 8.0` the cone binds on edge 1, so ``||D_1|| = (C / h_i) * L_1`` holds with equality and
`L_1` is set by the constraint -- the load-bearing use of `h_i`, the one that shapes the solution
rather than reporting on it. `L_1` reads 3.65e-02 / 6.46e-02 / 7.50e-02 / 9.77e-02 under
min / median / mean / max, and at 6.46e-02 it is seven orders clear of the positivity bound that
made the slack-regime reading meaningless. It is not a trivial readout of `h_i`: `||D_1||` moves
too, so `L_1 / h_i` is 0.365 / 0.323 / 0.302 / 0.195 rather than constant.

This leaves the diagnostic path unpinned. That is a real gap and it is not closable with the
current dataclass -- `kappa_max` is the only channel and it is degenerate in both directions.
Issue #2113 records the production consequence: at slack `C` the reported `kappa_max` is the cone
ratio of a neighbour that has been given zero weight, and exposing `h_i` on the dataclass is the
cheapest fix listed there.
"""

from __future__ import annotations

import pytest

import numpy as np

pytest.importorskip("cvxpy", reason="joint_socp requires cvxpy (the `numerical` extra)")

from mfgarchon.alg.numerical.gfdm_components.joint_socp import (
    JointSocpStencilData,
    PrecomputedJointSocpStencils,
    build_taylor_matrix_2d,
    solve_joint_socp_at_stencil,
    wendland_stencil_weights,
)

H = 0.1

#: Neighbour distances are 0.10, 0.14, 0.20, 0.30, 0.50, so min / median / mean / max are four
#: DISTINCT values -- 0.10, 0.20, 0.248, 0.50. The previous fixture had `median == min` exactly, so
#: `median -> min` was invisible to it and passed every test in this file. Ratio max/min is 5x
#: (median/min and max/median are each 2.5x); realistic kNN stencils run 1.29-1.43, so this fixture
#: is more favourable to a scale pin than production geometry is.
#:
#: The x-axis carries three neighbours for two conditions, hence a one-dimensional nullspace and a
#: genuine optimisation. The y-axis carries two for two and is FORCED -- `L_3 = 20.0` and
#: `||D_3|| = 3.0` exactly, at every scale and at every `C` in {1, 2, 4, 8}. No y-axis quantity can
#: discriminate anything, which is why both tests below read the x-axis.
SCALE_STENCIL = np.array([[0.0, 0.0], [H, 0.0], [-1.4 * H, 0.0], [0.0, 2 * H], [0.0, -3 * H], [5 * H, 0.0]])

#: `hjb_gfdm.py:1073` builds the production cache with `cone_constant_C=8.0`, `eps_pos=0.0`.
#: The pin runs at production's setting, which the previous version of this file claimed was
#: impossible -- that claim was an artifact of pinning `kappa_max`.
PRODUCTION_C = 8.0


def test_a_uniform_cross_returns_the_analytic_weights():
    """Oracle for the consistency solve. See the module docstring for what it does NOT establish.

    On a uniform 5-point cross the equality system has an EMPTY nullspace, so the feasible set is a
    single point and `L` is fixed by ``A.T @ L = e_lap`` alone: this passes unchanged at `C = 1e6`,
    at `C = 0.01`, and with `h_i` scaled 100x either way. It pins `build_taylor_matrix_2d` and "the
    solver returned the consistency solution", and says nothing about the cone or the scale.
    """
    offsets = np.array([[0.0, 0.0], [H, 0.0], [-H, 0.0], [0.0, H], [0.0, -H]])
    A, _ = build_taylor_matrix_2d(offsets)
    nullspace_dim = A.shape[0] - np.linalg.matrix_rank(A.T)
    assert nullspace_dim == 0, (
        f"this fixture is an oracle only because the feasible set is a single point; with "
        f"nullspace dim {nullspace_dim} the SOCP would be choosing among solutions and the "
        f"analytic weights would no longer be forced"
    )
    dists = np.linalg.norm(offsets, axis=1)
    nz = dists[dists > 1e-12]
    assert np.isclose(np.median(nz), np.max(nz)), "uniform by construction, so it cannot see the scale"

    result = solve_joint_socp_at_stencil(
        A, 0, float(np.median(nz)), C=1.0, dimension=2, wendland_w=wendland_stencil_weights(offsets, delta=4 * H)
    )
    assert result["status"] == "feasible", result
    np.testing.assert_allclose(np.asarray(result["L"]), np.array([-4.0, 1.0, 1.0, 1.0, 1.0]) / H**2, rtol=1e-9)


def _production_stencil() -> JointSocpStencilData:
    """Route through `PrecomputedJointSocpStencils`: it is the object that COMPUTES `h_i`.

    A direct `solve_joint_socp_at_stencil` call takes the scale as an argument and therefore cannot
    see any change to how it is derived. `joint_socp.py`'s `_build_stencil_arrays` is the only site
    in the package that derives one.
    """
    nz = np.linalg.norm(SCALE_STENCIL - SCALE_STENCIL[0], axis=1)
    nz = nz[nz > 1e-12]
    assert len({round(float(f(nz)), 9) for f in (np.min, np.median, np.mean, np.max)}) == 4, (
        "the fixture must keep min, median, mean and max distinct, or a scale change is invisible"
    )

    pre = PrecomputedJointSocpStencils(
        points=SCALE_STENCIL,
        interior_indices=np.array([0]),
        delta=8 * H,
        neighborhoods={0: {"indices": np.arange(len(SCALE_STENCIL))}},
        cone_constant_C=PRODUCTION_C,
    )
    data = pre.stencils[0]
    assert data.via == "socp_clarabel", f"via={data.via}"
    assert list(data.neighbor_indices) == [0, 1, 2, 3, 4, 5], (
        f"the edge indices below name specific neighbours; got {data.neighbor_indices} and a "
        f"reordering would silently repoint them at a different point of the fixture"
    )
    assert data.center_in_neighbors == 0, f"the centre must be index 0; got {data.center_in_neighbors}"
    return data


def test_the_scale_is_pinned_through_the_binding_cone_constraint():
    """The discriminator. Verified red against `median -> min`, `-> mean` and `-> max`.

    Edge 1 is the neighbour at `(H, 0)`, and at `C = 8.0` it is the one the cone binds on, so its
    weight is what ``||D_1|| <= (C / h_i) * L_1`` forces. Separations from the pinned value are
    43.4% (min), 16.2% (mean) and 51.3% (max); measured drift across solver tolerances 1e-6 to
    1e-14 is 9.6e-05. `rel=1e-3` therefore sits 10x above the noise and 162x below the smallest
    signal.

    The two assertions before the pin are not decoration. Each states a condition under which the
    pinned number means what the name says, and each has been observed to fail on this same
    fixture at a different `C`.
    """
    data = _production_stencil()
    L = np.asarray(data.L)

    assert data.kappa_max == pytest.approx(PRODUCTION_C, rel=1e-5), (
        f"this pin means 'L_1 is what the BINDING cone forces'. With kappa_max={data.kappa_max!r} "
        f"!= C={PRODUCTION_C} nothing binds, the weight is objective-determined instead, and the "
        f"assertion below would be pinning a different quantity under the same name"
    )
    assert L[1] > 1e-3, (
        f"edge 1 must stay clear of the eps_pos=0.0 bound for this to be an optimum rather than a "
        f"0/0 artifact of where the solver stopped; got L_1={L[1]!r}. Where the cone is slack this "
        f"same weight is 9.88e-08 and every quantity read off it is meaningless -- module docstring"
    )
    assert L[1] == pytest.approx(0.0645589, rel=1e-3)


def test_the_reported_diagnostic_carries_no_signal_about_the_scale():
    """Negative control for the CHOICE OF QUANTITY: why the test above reads a weight, not `kappa`.

    Everything asserted here holds under `min`, `median`, `mean` AND `max` -- measured at
    `C = 8.0` -- which is exactly why none of it can serve as the discriminator:

        scale     kappa_max     sum(L)       centre weight
        min       7.999991     -7.19e-14       -61.9549
        median    7.999995     -8.79e-14       -61.9933
        mean      7.999996     -1.19e-13       -62.0076
        max       8.000000     -4.44e-15       -62.0387

    `kappa_max` moves 0.0001% across a 5x change of scale and the centre weight moves 0.14%; the
    binding edge's weight moves 43-51%. These quantities DO change -- they change by amounts no
    tolerance can separate from solver noise.

    So this asserts the structural properties instead, which a broken construction WOULD break even
    though a rescaled one does not: Laplacian consistency and the M-matrix sign pattern.
    """
    data = _production_stencil()
    L = np.asarray(data.L)

    assert abs(L.sum()) < 1e-9, f"Laplacian weights must sum to zero; got {L.sum():.3e}"
    assert L[0] < 0, f"the centre weight must be negative; got {L[0]}"
    assert (L[1:] >= -1e-9).all(), (
        f"off-centre weights must be non-negative (M-matrix); got {L[1:]}. The tolerance is not "
        f"cosmetic: eps_pos is 0.0, so the bound is respected only to solver tolerance and an "
        f"exactly-zero comparison would flake on any weight the solver drives onto it"
    )
    assert L[0] == pytest.approx(-61.99, rel=1e-2), (
        "the centre weight is pinned at 1% to document that it CANNOT discriminate: it moves 0.14% "
        "across a 5x change of scale, so a scale change passes this assertion by construction"
    )
