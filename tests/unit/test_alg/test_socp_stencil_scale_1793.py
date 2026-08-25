"""The joint-SOCP stencil scale `h_i` cannot be checked by a BOUND on the diagnostic it defines.

Issue #1793. `h_i` non-dimensionalises the monotonicity cone -- the constraint is
``||D_j||_2 <= (C / h_i) * L_j`` -- and it also appears in the per-edge ratio reported about that
constraint, ``kappa_j = h_i * ||D_j|| / L_j``. Inflate `h_i`, the constraint tightens, the optimum's
``||D_j||`` shrinks by the same factor, and kappa comes back inside the bound looking healthy. A
25-axis mutation sweep found the suite blind: 5770 passed with `median` replaced by `max`.

WHERE KAPPA CARRIES INFORMATION ABOUT `h_i`, AND WHERE IT CARRIES NONE
----------------------------------------------------------------------
Sharper than "kappa is self-consistent", and measured on the fixture below:

    scale     C = 1.0 (cone slack)    C = 8.0 (cone binding)
    min          0.9999997               7.9999915
    median       0.9642736               7.9999954
    mean         0.9770679               7.9999961
    max          0.7543788               8.0000000
    spread       3.70 / 1.33 / 21.77%    0.00 / 0.00 / 0.00%

**Where the cone BINDS, kappa is `C` by construction and carries zero information about the scale.**
Where it does not bind, kappa is objective-determined and carries all of it. The pin therefore runs
at `C = 1.0`, and **that is not production's setting** -- `hjb_gfdm.py` uses `C = C_max = 8.0` with
`use_relaxed_fallback=True`, where no pin on kappa can discriminate anything. That is a real limit
of this approach, stated rather than hidden.

It is also why the previous version of this pin was fragile: its fixture bound at `C = 1.0`, so the
pinned `0.9999999108` was `1.0` minus CLARABEL's constraint residual (8.9e-08), and a solver
tolerance change moved it. The value pinned below sits 3.6% inside the cone and is bit-identical
across repeats.

WHAT THE TWO MECHANISMS EACH ESTABLISH
---------------------------------------
- `test_a_uniform_cross_returns_the_analytic_weights` is an oracle for the CONSISTENCY SOLVE, and
  weaker than it looks. On a uniform 5-point cross the equality system has an EMPTY nullspace, so
  the feasible set is a single point and `L` is fixed by ``A.T @ L = e_lap`` alone: the assertion
  passes unchanged at `C = 1e6`, at `C = 0.01`, and with `h_i` scaled 100x either way. It pins
  `build_taylor_matrix_2d` and "the solver returned the consistency solution". It says nothing
  about the cone, the objective, or `eps_pos` -- and nothing about `h_i`.
- `test_the_stencil_scale_is_pinned_where_the_cone_has_slack` is the discriminator. Its fixture has
  a one-dimensional nullspace, so the SOCP genuinely optimises there.

TWO CLAIMS THE PREVIOUS VERSION OF THIS FILE RECORDED AS FINDINGS, BOTH FALSE
-----------------------------------------------------------------------------
Kept as a warning, because they were written here as reasons for a later reader NOT to look.

- **"Feasibility does not flip."** It does. Over ~1,900 stencils (four cloud families x three seeds
  x k in {9,13} x C in {1.0, 8.0}) review found a graded-cloud stencil at a spread of only **1.19x**
  that is feasible under `median` and INFEASIBLE under `max`, rescued only at `C >= 1.2`. The
  earlier claim rested on five geometries at three C values -- fifteen solves -- against a rate near
  0.05%, which misses it with probability ~99%. **An underpowered negative result written down as a
  settled one.**
- **"`eps_M > 0` never happens."** That measured the harness, not the geometry.
  `PrecomputedJointSocpStencils` defaults `use_relaxed_fallback=False`, which makes the relaxed
  branch unreachable by construction. At production settings review measured `eps_M` reaching
  **5.3e2** on ordinary clouds with no mutation at all, on roughly a third of the interior stencils
  of an irregular cloud. This is the claim closest to the paper's load-bearing `eps_M == 0`, and it
  was the one stated most confidently and supported least.
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

#: min, median, mean and max of the neighbour distances are four DISTINCT values here --
#: 0.10, 0.20, 0.228, 0.50. The previous fixture had `median == min` exactly, so `median -> min`
#: was invisible to it and passed all three tests. The spread is 2.5x; realistic kNN stencils run
#: 1.29-1.43, so this is more favourable to the pin than production geometry.
SLACK_STENCIL = np.array([[0.0, 0.0], [H, 0.0], [-1.4 * H, 0.0], [0.0, 2 * H], [0.0, -3 * H], [5 * H, 0.0]])


def test_a_uniform_cross_returns_the_analytic_weights():
    """Oracle for the consistency solve. See the module docstring for what it does NOT establish."""
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


def _slack_stencil_kappa() -> float:
    """Route through `PrecomputedJointSocpStencils`: it is the object that COMPUTES `h_i`.

    A direct `solve_joint_socp_at_stencil` call takes the scale as an argument and therefore cannot
    see any change to how it is derived. `joint_socp.py`'s `_build_stencil_arrays` is the only site
    in the package that derives one.
    """
    points = SLACK_STENCIL
    nz = np.linalg.norm(points - points[0], axis=1)
    nz = nz[nz > 1e-12]
    assert len({round(float(f(nz)), 9) for f in (np.min, np.median, np.mean, np.max)}) == 4, (
        "the fixture must keep min, median, mean and max distinct, or a scale change is invisible"
    )

    pre = PrecomputedJointSocpStencils(
        points=points,
        interior_indices=np.array([0]),
        delta=8 * H,
        neighborhoods={0: {"indices": np.arange(len(points))}},
        cone_constant_C=1.0,
    )
    data = pre.stencils[0]
    assert data.via == "socp_clarabel", f"via={data.via}"
    return float(data.kappa_max)


def test_the_stencil_scale_is_pinned_where_the_cone_has_slack():
    """The discriminator. Verified red against `median -> min`, `-> mean` and `-> max`.

    Tolerance: the four candidate scales separate by 3.70%, 1.33% and 21.77%, and the value is
    bit-identical across repeated solves. `rel=1e-3` leaves 13x margin on the smallest signal and
    an order of slack against a solver-tolerance change -- the previous `1e-4` around a value that
    WAS the solver residual is what made the earlier pin fragile.
    """
    assert _slack_stencil_kappa() == pytest.approx(0.9642735774, rel=1e-3)


def test_the_weights_carry_no_usable_signal_about_the_scale():
    """Negative control for the CHOICE OF QUANTITY, and the reason this file pins kappa.

    Everything asserted here holds under `min`, `median`, `mean` AND `max` -- measured -- which is
    exactly why none of it can serve as the discriminator:

        scale     sum(L)      centre weight   min off-centre weight
        min       1.07e-14      -61.9799          5.48e-02
        median   -1.06e-13      -61.9048          9.88e-08
        mean     -1.60e-14      -61.9048          1.50e-08
        max      -3.02e-14      -61.9048          1.81e-08

    The centre weight moves 0.12% across a 5x change of scale; kappa moves 21.77%. The earlier
    version of this test asserted `max|L|` stable to 1%, quoting a 0.018% max-norm figure -- which
    hid that one off-centre weight collapses from 5.48e-02 to 9.88e-08, effectively deleting that
    neighbour. The weights DO change; they change in a place a pin cannot read.

    So this asserts the two structural properties instead, which a broken construction WOULD break
    even though a rescaled one does not: Laplacian consistency and the M-matrix sign pattern.
    """
    pre = PrecomputedJointSocpStencils(
        points=SLACK_STENCIL,
        interior_indices=np.array([0]),
        delta=8 * H,
        neighborhoods={0: {"indices": np.arange(len(SLACK_STENCIL))}},
        cone_constant_C=1.0,
    )
    L = np.asarray(pre.stencils[0].L)

    assert abs(L.sum()) < 1e-9, f"Laplacian weights must sum to zero; got {L.sum():.3e}"
    assert L[0] < 0, f"the centre weight must be negative; got {L[0]}"
    assert (L[1:] >= 0).all(), f"off-centre weights must be non-negative (M-matrix); got {L[1:]}"
    assert L[0] == pytest.approx(-61.905, rel=1e-2), (
        "the centre weight is pinned at 1% to document that it CANNOT discriminate: it moves 0.12% "
        "across a 5x change of scale, so a scale change passes this assertion by construction"
    )
