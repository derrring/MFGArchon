"""The joint-SOCP stencil scale `h_i` is pinned on both dispatch paths, by different quantities.

Issue #1793. `h_i = median(neighbour distances)` non-dimensionalises the monotonicity cone and
enters in two places:

    (a) the CONSTRAINT   ||D_j||_2 <= (C / h_i) * L_j        -- shapes the solution
    (b) the DIAGNOSTIC   kappa_j = h_i * ||D_j||_2 / L_j     -- reports on it, exposed as kappa_max

A 25-axis mutation sweep found the suite blind to the scale entirely: 5770 passed with `median`
replaced by `max`. This file pins one non-degenerate quantity on each of the two dispatch paths,
both at production's `C = 8.0` (`hjb_gfdm.py:1073`).

DO NOT REACH FOR `kappa_max` ON `SCALE_STENCIL`
-----------------------------------------------
It is the quantity the dataclass names after the diagnostic, so it is the obvious probe, and it is
unusable there in both directions. Where the cone binds it reports `C` back -- at `C = 8.0` all
four candidate scales agree to five significant figures. Where it is slack, the objective drives an
x-axis weight onto the `eps_pos = 0.0` bound and the argmax edge is the one it deleted, so the
ratio is `0/0`: at `C = 1.0`, `L_1 = 9.88e-08` against `||D_1|| = 4.76e-07`. That value tracks
CLARABEL's tolerance one-for-one until it crosses the `L_j <= 1e-12 -> inf` guard at
`joint_socp.py:306`, so tightening the solver breaks any pin on it.

**That is a property of this geometry, not of the dispatch**, and the distinction is what the
narrow claim below rests on. Three exits reach the SOCP and only two force anything:

    fast path rejected, cone fails      -> the optimum sits on the cone -> kappa_max = C
    fast path rejected, M-matrix fails  -> a positivity bound is active -> 0/0 at the argmax
    `np.linalg.solve(ATA, rhs)` RAISES  -> NEITHER check ran -> nothing is forced

The third is `except (np.linalg.LinAlgError, ValueError): pass` at `joint_socp.py:246`, and **both
fixtures here take it** -- `A` has more columns than rows, so `ATA` is singular.
`test_a_uniform_cross_returns_the_analytic_weights` is therefore a standing counterexample to any
wider claim: CLARABEL runs, the cone is slack (0.005 against `C = 1.0`), every off-centre weight is
100.0. So the statement this file makes is exactly: **on `SCALE_STENCIL` at `C = 8.0`, the cone
binds and `L_1` is what it forces.** Widening it needs conditioning on why the fast path was
skipped, which is measured here for two geometries and no more.

ON THE FAST PATH `kappa_max` IS A READOUT OF THE SCALE
-------------------------------------------------------
`wendland_lsq_fast_path` runs no solver: it accepts the Wendland least-squares weights when
`m_matrix_ok and cone_ok` (`joint_socp.py:237`) -- both, and the M-matrix half is what the other
half of this file leans on. `(L, D)` then do not depend on `h_i` at all, so

    kappa_max = h_i * max_j ||D_j|| / L_j = h_i * const

and on `FAST_PATH_STENCIL` that constant is 0.2840929695, holding to 1 ulp across all four scales.
Claims of this kind are stated as BOUNDS throughout this file, deliberately: an exactness is a
property of one `DELTA_MULT` on one LAPACK and silently survives changes to either.

`hjb_gfdm.py:1078` picked `C = 8.0` partly to reach this path. How often it is reached is not
measured here and is not asserted -- it swings on `k_neighbors`, which `hjb_gfdm.py:276` leaves
auto-computed. Reachable and otherwise untested is reason enough.

The measurements behind each number are in the commit history; the production consequence of
`kappa_max`'s degeneracy is #2113.
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
#: `median -> min` was invisible to it and passed every test in this file. The four ratios are
#: median/min = 2.0, mean/median = 1.24, max/median = 2.5, max/min = 5.0.
#:
#: The x-axis carries three neighbours for two conditions, hence a one-dimensional nullspace and a
#: genuine optimisation. The y-axis carries two for two and is FORCED -- `L_3 = 20.0` and
#: `||D_3|| = 3.0` to within 6e-13, at every scale and at every `C` in {1, 2, 4, 8}. A BOUND, not
#: an exactness: these are solver outputs wearing integers' clothing, and `L_3` is bit-equal to
#: 20.0 in 0 of those 16 combinations. No y-axis quantity can
#: discriminate anything, which is why both tests below read the x-axis.
SCALE_STENCIL = np.array([[0.0, 0.0], [H, 0.0], [-1.4 * H, 0.0], [0.0, 2 * H], [0.0, -3 * H], [5 * H, 0.0]])

#: `DELTA_MULT` is NOT a free parameter, and the safe set is NOT AN INTERVAL. Scanned at 0.01
#: resolution, all three assertions of the fast-path test hold on
#:
#:     [0.57, 0.94]  u  [1.11, 1.57]
#:
#: with a band between them where the least-squares weights go near-singular and `minL` falls to
#: ~1e-07. So "lower is worse" is the wrong model -- 0.60 and 0.80 are as usable as 1.30 -- and
#: anything in 0.95-1.10 is not. The upper edge is set by the `minL > 1.0` GUARD, not by the
#: dispatch: `via` stays on the fast path past it, so a bad `DELTA_MULT` fails on the weight
#: assertion rather than on `via`.
#:
#: 1.3 is roughly centred in the upper interval: 14.6% above the band, 20.8% below the edge.
DELTA_MULT = 1.3

#: The OTHER dispatch path. Wendland least-squares weights that already satisfy the cone at
#: `C = 8.0`, so `joint_socp.py:237` accepts them (`m_matrix_ok and cone_ok`) and no solver runs.
#: `(L, D)` are then independent of `h_i`, which makes `kappa_max = h_i * const` -- the readout
#: `SCALE_STENCIL` cannot give. Found by adversarial review as the counterexample to this file's
#: earlier claim that no fixture could pin the diagnostic.
FAST_PATH_STENCIL = np.array(
    [
        [0.0, 0.0],
        [0.21985, 0.261539],
        [0.195229, 0.053855],
        [0.093786, -0.137865],
        [-0.152137, -0.092983],
        [-0.147989, 0.036126],
        [0.102412, -0.341169],
        [0.005843, 0.337088],
    ]
)

#: `hjb_gfdm.py:1073` builds the production cache with `cone_constant_C=8.0`, `eps_pos=0.0`.
#: Both pins run at that setting.
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

    # rel=1e-2, not tighter: this guard only has to tell `binds` (8.0) from `slack` (0.96), which
    # is 88% of C away. At rel=1e-5 the whole test went RED at CLARABEL tolerances 1e-6 and 1e-7 --
    # on THIS assertion, not the pin -- and then blamed it on the cone not binding, which was
    # false; the cone still bound to 5.8e-5 there. The green band was 1e-8..1e-14 with the default
    # sitting on its edge. A guard whose message misdiagnoses is worse than a looser guard.
    assert data.kappa_max == pytest.approx(PRODUCTION_C, rel=1e-2), (
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


def _fast_path_stencil() -> JointSocpStencilData:
    """`FAST_PATH_STENCIL` through the object that derives `h_i`, at production's `C`."""
    nz = np.linalg.norm(FAST_PATH_STENCIL - FAST_PATH_STENCIL[0], axis=1)
    nz = nz[nz > 1e-12]  # the centre's own 0.0 is not a neighbour distance; `joint_socp.py:662`
    assert len({round(float(f(nz)), 9) for f in (np.min, np.median, np.mean, np.max)}) == 4, (
        "the fixture must keep min, median, mean and max distinct, or a scale change is invisible"
    )
    pre = PrecomputedJointSocpStencils(
        points=FAST_PATH_STENCIL,
        interior_indices=np.array([0]),
        delta=DELTA_MULT * float(nz.max()),
        neighborhoods={0: {"indices": np.arange(len(FAST_PATH_STENCIL))}},
        cone_constant_C=PRODUCTION_C,
    )
    return pre.stencils[0]


def test_the_scale_is_pinned_through_the_diagnostic_on_the_fast_path():
    """The discriminator for path (b). Verified red against `median -> min`, `-> mean`, `-> max`.

    `kappa_max` reads 0.043277 / 0.057535 / 0.070411 / 0.101196 under min / median / mean / max --
    separations of 24.78%, 22.38% and 75.89% from the pinned value, against a `kappa_max` that
    moves 0.0001% on `SCALE_STENCIL` at the same `C`. `kappa_max / h_i` is 0.2840929695 for all
    four to within 1 ulp, because the fast path never consults `h_i` when computing `(L, D)` --
    only when deciding whether to accept them, and when forming the ratio.

    `rel=1e-7` rather than something tighter: no solver runs, so there is no solver tolerance to
    protect against. The margin is for a different LAPACK, and review measured that -- reference
    LAPACK 3.9.0 against Apple Accelerate differ by 1.5e-16 relative, 0.7 ulp, so `rel=1e-7` is
    conservative by about seven orders rather than the one I guessed at.
    """
    data = _fast_path_stencil()
    L = np.asarray(data.L)

    assert data.via == "wendland_lsq_fast_path", (
        f"this pin means 'the accepted Wendland weights scaled by h_i'; got via={data.via!r}. On "
        f"the solver path the same quantity is either C or a 0/0 ratio -- see the module docstring"
    )
    assert L[1:].min() > 1.0, (
        f"every edge must carry real weight for kappa_max to be a ratio of two real numbers; got "
        f"min off-centre weight {L[1:].min()!r}"
    )
    assert data.kappa_max == pytest.approx(0.05753476984, rel=1e-7)
