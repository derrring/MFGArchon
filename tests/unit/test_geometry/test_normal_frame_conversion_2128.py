#!/usr/bin/env python3
"""Issue #2128: the FP no-flux ghost is Robin's algebra plus a frame conversion, not a second copy.

WHAT THIS PINS
--------------
`ghost_cell_robin` takes coefficients ALREADY in the normal frame, which is why it correctly has no
sign argument -- `du/dn` carries the wall's direction, and #2063 removed a sign from it after
measuring it unused at cell-centring and applied BACKWARDS at vertex-centring.
`ghost_cell_fp_no_flux` takes axis-frame `v` and `D`, so `J . n = 0` needs the outward normal to
project them. That projection is `normal_frame_coefficients`, and before #2128 it existed only as a
second copy of Robin's closed form.

THE ORACLE IS EXTERNAL, NOT AN AGREEMENT
----------------------------------------
Two implementations agreeing proves nothing when one was derived from the other -- which is exactly
what this change makes true of these two. So the load-bearing test here is NOT `fp_no_flux ==
robin`; it is both against the **exact zero-flux profile**, which neither implements:

    J . n = 0  =>  v_n*rho - D*drho/dn = 0  =>  rho(s) = rho_wall * exp(v_n*s/D)

so the exact ghost one cell out is `rho_interior * exp(z)`, `z = v_n*dx/D`, and each branch is a
rational approximation to `exp(z)`: cell-centred is its Pade(1,1) and converges at order 3, the
vertex form is the truncated series `1 + z` and converges at order 2 (#2068 measured 3.04 and 2.01).
A delegation that got the conversion wrong would not merely disagree with the old code -- it would
lose the order against the analytic profile, which is checkable without either implementation.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.geometry.boundary.ghost_cells import (
    GridType,
    ghost_cell_fp_no_flux,
    ghost_cell_robin,
    normal_frame_coefficients,
)

_CENTRINGS = [GridType.CELL_CENTERED, GridType.VERTEX_CENTERED]
#: Stated orders against exp(z) (#2068): 3.04 cell-centred, 2.01 vertex. Asserted with margin.
_MIN_ORDER = {GridType.CELL_CENTERED: 2.6, GridType.VERTEX_CENTERED: 1.7}


def test_the_conversion_returns_the_normal_frame_pair():
    """`J . n = v_n*rho - D*drho/dn = 0` is Robin with `alpha = v.n` and `beta = -D`.

    Pinned as the NORMAL-FRAME pair, not merely as a pair that happens to give the right ghost. An
    earlier draft returned `(v, -sign*D)` -- `sign` times this one. It yields an identical ghost,
    because scaling both coefficients leaves a homogeneous Robin condition unchanged, and it passed
    every value test in this file. It is still wrong for a function with this name: it diverges the
    moment a caller has a non-zero right-hand side (1.045455 against 0.590909 at `g = 0.5`), and it
    puts the sign on the diffusion coefficient rather than on the velocity being projected.
    """
    assert normal_frame_coefficients(0.4, 0.2, -1.0) == (-0.4, -0.2)
    assert normal_frame_coefficients(0.4, 0.2, +1.0) == (0.4, -0.2)
    assert normal_frame_coefficients(-0.7, 0.35, +1.0) == (-0.7, -0.35)
    assert normal_frame_coefficients(-0.7, 0.35, -1.0) == (0.7, -0.35)


#: [DELETED round 7] `test_the_ghost_is_robin_evaluated_at_the_converted_pair` lived here: 12
#: parametrisations asserting `ghost_cell_fp_no_flux(...) == ghost_cell_robin(u, 0.0, 2*dx*alpha,
#: 2*dx*beta, ...)`. That is the function's body restated, so it is structurally tautological, which
#: AGENTS.md names as the deletable set -- and it could not even see a wrong scale, because scaling a
#: homogeneous Robin pair leaves the ghost unchanged, as its own comment said. Measured in round 7:
#: 2 kills out of 10 mutants, both also caught by arms that remain. The external oracle below and
#: the per-element value pins are what carry the delegation.


@pytest.mark.parametrize("grid_type", _CENTRINGS, ids=lambda g: g.name)
@pytest.mark.parametrize("sign", [-1.0, +1.0], ids=["min_wall", "max_wall"])
def test_the_ghost_converges_to_the_exact_zero_flux_profile(grid_type, sign):
    """THE external oracle: `rho_exact = rho_interior * exp(v_n*dx/D)`, which neither branch implements.

    Refining `dx` alone at fixed `v` and `D` drives `z -> 0`, so the error must fall at the branch's
    order. This is what a wrong conversion loses: flip the sign of `beta` and `z` flips with it, so
    the approximation is to `exp(-z)` and the observed order collapses.
    """
    v, D, u = 0.6, 0.25, 1.0
    v_n = v * sign
    errors, spacings = [], [0.02, 0.01, 0.005]
    for dx in spacings:
        exact = u * np.exp(v_n * dx / D)
        got = ghost_cell_fp_no_flux(u, v, D, dx, sign, grid_type)
        errors.append(abs(got - exact))

    orders = [np.log(errors[i] / errors[i + 1]) / np.log(spacings[i] / spacings[i + 1]) for i in range(len(errors) - 1)]
    observed = float(np.min(orders))
    assert observed >= _MIN_ORDER[grid_type], (
        f"{grid_type.name} at the {'max' if sign > 0 else 'min'} wall converged at order "
        f"{observed:.2f} against the analytic profile rho*exp(v_n*dx/D), below the "
        f"{_MIN_ORDER[grid_type]} this branch is required to hold (#2068 measured "
        f"{'3.04' if grid_type is GridType.CELL_CENTERED else '2.01'}). Errors {errors} at dx "
        f"{spacings}. A conversion with the wrong sign on beta approximates exp(-z) and lands here."
    )


def test_the_cell_centred_degeneracy_refuses_instead_of_reflecting():
    """The behaviour change this file's degeneracy carries, pinned deliberately.

    At `alpha/2 + beta/dx = 0` -- equivalently `2D = v_n*dx` -- the ghost's own coefficient vanishes,
    so the condition does not determine it. The pre-#2128 code returned `interior_value` there, a
    silent fallback it called "the pure advection limit"; `ghost_cell_robin` raises, and the
    delegation makes the refusal win. Measured on `u=1, v=4, D=0.2, dx=0.1`: `1.0` before, ValueError
    after. It is not the branch's only one -- three are enumerated in the changelog fragment, and an
    earlier draft of this docstring said otherwise, which round 3 corrected in the changelog and left
    here.

    The second arm is the over-fire control -- halving the drift leaves the degeneracy and must solve.
    """
    D, dx, u, sign = 0.2, 0.1, 1.0, +1.0
    v_degenerate = 2.0 * D / dx * sign
    with pytest.raises(ValueError, match="singular ghost cell formula"):
        ghost_cell_fp_no_flux(u, v_degenerate, D, dx, sign, GridType.CELL_CENTERED)

    # Over-fire control, asserted against the closed form rather than `isfinite`, which passes on
    # any finite wrong answer -- including the `interior_value` this change stopped returning.
    v_near = 0.5 * v_degenerate
    v_n = v_near * sign
    expected = u * (2.0 * D + v_n * dx) / (2.0 * D - v_n * dx)
    assert ghost_cell_fp_no_flux(u, v_near, D, dx, sign, GridType.CELL_CENTERED) == pytest.approx(expected)

    # And the case the scaling exists for: `v_n = 0` is homogeneous Neumann, so the ghost is exactly
    # `interior_value`. Delegating an UNSCALED pair refused this on a coarse grid (#2217).
    assert ghost_cell_fp_no_flux(u, 0.0, 5e-12, 10.0, sign, GridType.CELL_CENTERED) == pytest.approx(u)


@pytest.mark.parametrize("diffusion", [0.0, 1e-13], ids=["D_zero", "D_below_threshold"])
def test_the_vertex_zero_diffusion_value_is_preserved_not_inherited(diffusion):
    """The OTHER degeneracy, and this one is pinned so the delegation does NOT change it (#2220).

    Vertex-centred with `D ~ 0` is a different condition from the cell-centred one above: there the
    ghost's coefficient vanishes, here `beta` does. `J . n = v_n*rho_wall = 0` with `v_n != 0` forces
    `rho_wall = 0`, and on a vertex grid the wall IS the interior node -- so the condition constrains
    the interior value and says nothing about the ghost. The pre-#2128 code returned
    `interior_value`; `ghost_cell_robin`'s `beta == 0` branch returns the implied Dirichlet `0.0`.
    Both put a number where nothing is determined.

    #2128 is a consolidation and does not get to pick between them by side effect, so the old value
    is preserved behind an explicit guard and the choice is #2220's. This test exists to make the
    preservation deliberate: if the guard is removed, this fails rather than the value quietly
    becoming 0.0.

    [CORRECTED round 7, refined round 8] An earlier version called the preserved value "correct" at
    `v_n > 0`, on the argument that copying the interior into the ghost is the standard outflow
    treatment. That flattered the value being preserved and the measurement does not support it.
    With `J . n = v_n*rho_wall - D*d(rho)/dn` in each centring's own wall geometry, at `D = 0`,
    SWEEPING the ghost rather than evaluating it at one point:

        ghost ->        1.0     0.0    -1.0    17.0
        vertex J.n     +0.4    +0.4    +0.4    +0.4     <- CONSTANT in the ghost
        cell   J.n     +0.4    +0.2     0.0    +3.6     <- -1.0 is the unique zero
        control, D = 0.5, all four cases: |J.n| <= 6.7e-16

    Read the vertex row as a NEGATIVE result: `J . n` does not respond to the ghost at all, because
    the wall IS the node and `D * anything = 0`. So the flux this centring leaks through a wall
    `types.py` defines as impermeable -- NO_FLUX and REFLECTING are one concept there, so
    "non-reflecting treatment" named the condition this function is not -- is a property of the
    CENTRING at `D = 0`, not of the value `1.0`. No ghost can fix it and no ghost can be blamed for
    it, so this function's stated contract is UNSATISFIABLE at `D = 0` on this centring: a sharper
    form of #2220's over-specification, not an exception to it.

    The cell row is the discriminating one and runs the other way -- there the ghost does move
    `rho_wall`, and the `-1.0` that the `ghost_cells.py` docstring frames as a pathology is the
    unique value satisfying the contract.

    Two consequences for how to read this test. It preserves the status quo and does not endorse
    it; #2220 must not cite it as evidence for keeping this value. And no first-order argument
    covers it, because the parametrisation also runs `D = 1e-13`, where the equation is second
    order and the exact profile `rho_i*exp(v_n*dx/D)` overflows to `inf` while the code returns
    `1.0`.
    """
    u, v, dx, sign = 1.0, 0.4, 0.1, +1.0
    assert ghost_cell_fp_no_flux(u, v, diffusion, dx, sign, GridType.VERTEX_CENTERED) == u

    #: The threshold is the pre-#2128 one. Just above it the branch is live and enormous, which is
    #: the discontinuity #2220 records -- pinned here so "preserved" means the value AND its edge.
    live = ghost_cell_fp_no_flux(u, v, 1e-9, dx, sign, GridType.VERTEX_CENTERED)
    assert live > 1e6, "above the guard's threshold the vertex branch must be live, not clamped"


@pytest.mark.parametrize("grid_type", _CENTRINGS)
def test_zero_diffusion_ignores_the_sign_of_the_normal_velocity(grid_type):
    """RECORDED DEFECT, not a contract. Retirement condition is in the assertion message (#2220).

    At `D = 0` the Fokker-Planck equation drops to first order, and a first-order equation admits a
    boundary condition on the INFLOW side only. So `v_n < 0` and `v_n > 0` are two different
    problems: outflow may impose nothing, inflow may impose one condition and `J . n = v_n*rho = 0`
    forces `rho_wall = 0`. Both centrings currently return the same number for both signs -- the
    vertex guard tests `abs(D)`, and the cell-centred closed form cancels `sign` against itself.

    Pinned rather than fixed because #2216 is a consolidation. Pinned rather than left silent
    because the pin one function up is unlabelled and would otherwise read as a specification for a
    value we now know is imposed on the wrong half of the domain.

    Measured before writing, with a control that separates: at `D = 0.5` the same call returns 1.08
    at `sign=+1` and 0.92 at `sign=-1`, so the equality below is a property of `D = 0` and not of
    the probe.
    """
    u, v, dx = 1.0, 0.4, 0.1
    outflow = ghost_cell_fp_no_flux(u, v, 0.0, dx, +1.0, grid_type)
    inflow = ghost_cell_fp_no_flux(u, v, 0.0, dx, -1.0, grid_type)
    assert outflow == inflow, (
        "the two signs now differ, which is what #2220 asks for -- delete this defect pin AND "
        "retarget test_zero_diffusion_has_a_stated_behaviour_on_both_centrings, which pins the "
        "same status quo and fails on the same change, then assert the intended behaviour"
    )

    #: The control, in the same invocation: away from the degeneracy the same call IS sign-sensitive,
    #: so a broken probe cannot pass this test by returning one number for everything.
    assert ghost_cell_fp_no_flux(u, v, 0.5, dx, +1.0, grid_type) != ghost_cell_fp_no_flux(
        u, v, 0.5, dx, -1.0, grid_type
    ), "probe is inert: this call must separate the signs at D = 0.5"


def test_field_valued_inputs_behave_the_same_on_both_centrings():
    """#2128's own acceptance bullet: "works or raises identically on both centrings".

    Two different surfaces, and the delegation moved them in opposite directions:

    - **Multi-element `D`** — `main` raised `ValueError` for a multi-element ndarray and `TypeError`
      for a list or tuple, and — because `bool()` on a one-element array is legal — silently ACCEPTED
      shape-(1,) and shape-(1,1). The delegation made the CELL path
      answer for any array while the vertex guard, a scalar `abs()`, kept raising: an asymmetry
      introduced by side effect. Refused on both for size > 1, size-1 still accepted — `main`'s
      acceptance set exactly. A draft guarded on `ndim != 0`, which also refused size-1: an
      undeclared capability removal inside a consolidation, and the reason this arm pins the size-1
      case rather than only the refusal.
    - **Field-valued drift** — `main` raised on the cell path (`truth value of an array is
      ambiguous`) and the delegation makes it work on both. That is a genuine gain, unclaimed until
      the review found it, and pinned here so it is not lost silently.
    """
    #: NON-UNIT interior, and it is load-bearing. An earlier draft used `np.ones(3)` and a size-1
    #: interior of 1.0, so `interior_value` factored out of both oracles entirely -- a mutant
    #: dropping it (returning the ghost for interior 1 whatever the caller passed) went green on all
    #: 1348 geometry tests. A value oracle whose fixture makes a factor invisible is not a value
    #: oracle for that factor.
    u = np.array([1.0, 2.5, -0.4])

    for grid_type in _CENTRINGS:
        with pytest.raises(NotImplementedError, match="multi-element diffusion_coeff"):
            ghost_cell_fp_no_flux(u, 0.4, np.array([0.2, 1e-13, 0.5]), 0.1, +1.0, grid_type)

        #: size-1 is `main`'s acceptance set and must survive the guard, which a draft's
        #: `ndim != 0` predicate silently removed. Asserted against the closed form, not against
        #: shape and finiteness: the THIRD time on this branch that an arm pinning an array
        #: capability shipped without a value oracle. Measured before this line existed -- a defect
        #: corrupting only the array-`D` path (`D -> 2D` for arrays) passed this arm, this file, and
        #: the whole geometry suite (1348 tests at that measurement).
        #: Both accepted shapes, both wall directions, and a non-unit interior -- `main` accepts
        #: shape-(1,1) too, `ghost_cells.py` names it in the acceptance set it preserves, and a
        #: predicate refusing only it went green on the whole geometry suite (1348 tests then).
        for shape in ((1,), (1, 1)):
            for sign in (-1.0, +1.0):
                interior, v, D, dx = 2.5, 0.4, 0.2, 0.1
                v_n = v * sign
                got = ghost_cell_fp_no_flux(interior, v, np.full(shape, D), dx, sign, grid_type)
                assert np.shape(got) == shape, f"{grid_type.name}: shape {shape} must be preserved"
                expected_1 = {
                    GridType.CELL_CENTERED: interior * (2 * D + v_n * dx) / (2 * D - v_n * dx),
                    GridType.VERTEX_CENTERED: interior * (D + v_n * dx) / D,
                }[grid_type]
                np.testing.assert_allclose(got, np.full(shape, expected_1), rtol=1e-15)

    #: Asserted against the closed form PER ELEMENT, not against shape and finiteness. An earlier
    #: draft of this arm checked `isfinite` plus "three distinct values", and an array-only defect --
    #: each boundary point receiving another point's drift -- passed it AND the entire 1348-test
    #: geometry suite, because no scalar test can express a permutation. That is the same defect
    #: round 1 raised against this file's over-fire control, re-created in the arm added to pin an
    #: array capability. Only a per-element oracle sees it.
    drift = np.array([0.4, -0.2, 1.1])
    D, dx = 0.2, 0.1
    #: BOTH wall directions. An earlier draft passed `sign = +1.0` only, so a defect applying the
    #: sign wrongly on the array path -- and only there -- went green on the whole geometry suite (1348 tests then).
    for grid_type in _CENTRINGS:
        for sign in (-1.0, +1.0):
            v_n = drift * sign
            expected = {
                GridType.CELL_CENTERED: u * (2.0 * D + v_n * dx) / (2.0 * D - v_n * dx),
                GridType.VERTEX_CENTERED: u * (D + v_n * dx) / D,
            }[grid_type]
            got = ghost_cell_fp_no_flux(u, drift, D, dx, sign, grid_type)
            assert np.shape(got) == (3,), f"{grid_type.name}: a field drift must stay elementwise"
            np.testing.assert_allclose(got, expected, rtol=1e-15)


@pytest.mark.parametrize("grid_type", _CENTRINGS, ids=lambda g: g.name)
def test_the_singularity_verdict_is_invariant_under_rescaling_the_pair(grid_type):
    """[REPLACED #2217] This was `test_the_scales_magnitude_is_pinned_from_both_sides`, and it
    pinned a defect.

    That test asserted `ghost_cell_fp_no_flux` must RAISE at `v_n = 0, D = 3.75e-13, dx = 1`. But
    `v_n = 0` is homogeneous Neumann: the condition is `-D * drho/dn = 0`, the ghost is exactly the
    interior value, and the problem is well posed at every `D > 0`. It asserted the raise because
    `ghost_cell_robin`'s threshold was ABSOLUTE while the quantity it tests carries units, so the
    refusal boundary moved with `dx` and with however the caller happened to scale its pair. Round 6
    of #2216 wrote this test to pin the `2*dx` scaling that papered over that -- pinning the
    workaround, and with it the defect underneath.

    #2217 made the threshold relative, so the scaling is inert and gone, and what is worth pinning
    is the invariance itself: scaling `(alpha, beta)` by any non-zero constant leaves the Robin
    condition unchanged, so it must leave BOTH the verdict and the ghost unchanged.

    Measured before this was written: against the absolute threshold, 96 of 144 coefficient
    families built to straddle the cancellation point changed verdict under rescaling. Against the
    relative one, 0 of 144.

    THE VERTEX ARM IS INERT AGAINST THAT MUTANT and is kept anyway -- stated so it does not read as
    coverage it is not. Restoring the absolute threshold kills the cell arm and this one PASSES,
    because the vertex branch never computes `coeff_ghost`; its own predicate is `abs(beta) == 0`,
    which is scale-invariant for a different reason (`s*0 == 0` for every finite `s`). It guards a
    change that would give the vertex branch a magnitude-based threshold, which is exactly the
    change #2217 asks someone to consider next.
    """
    u, dx = 1.0, 0.1
    #: built AT the cancellation point -- generic pairs put |alpha/2 + beta/dx| near 1, where no
    #: scaling in 12 decades crosses 1e-12, so they cannot express this defect at all.
    for alpha in (-2.0, 0.5, 2.0):
        for r in (0.0, 1e-15, 1e-11, 1e-3):
            beta = (r - 1.0) * alpha * dx / 2.0
            verdicts, ghosts = set(), []
            for s in (1e-6, 1e-3, 1.0, 1e3, 1e6):
                try:
                    ghosts.append(ghost_cell_robin(u, 0.0, s * alpha, s * beta, dx, grid_type))
                    verdicts.add("ok")
                except ValueError:
                    verdicts.add("raise")
            assert len(verdicts) == 1, (
                f"alpha={alpha} r={r}: rescaling the pair changed the refuse/answer verdict "
                f"({verdicts}) -- the threshold is scale-blind again (#2217)"
            )
            #: The VALUE is a separate claim and it does NOT hold everywhere, which the first
            #: version of this test asserted and was wrong about. Near cancellation the ghost is
            #: `something / tiny`: `alpha/2` and `beta/dx` nearly annihilate, rescaling perturbs
            #: each rounding, and the quotient moves by ~7e-6 relative at `r = 1e-11`. That is
            #: conditioning, not a defect, and it is why #2217's acceptance test must be read as
            #: "same verdict" -- the "same ghost" half holds only where the condition is well
            #: conditioned. Asserted there, and deliberately not asserted in the cancellation band.
            if ghosts and r >= 1e-3:
                np.testing.assert_allclose(
                    ghosts,
                    ghosts[0],
                    rtol=1e-9,
                    err_msg=f"alpha={alpha} r={r}: ghost moved under rescaling AWAY from cancellation",
                )


def test_homogeneous_neumann_is_answered_not_refused():
    """The well-posed case #2217 reports being rejected, pinned on the public alias.

    `v_n = 0` makes the wall condition `-D * drho/dn = 0` at every `D > 0`, so the ghost is the
    interior value. The absolute threshold refused it whenever `2D < 1e-12` -- measured, that is the
    ENTIRE difference the #2217 fix makes to this function: over 720 inputs exactly 12 verdicts
    moved, all of them this case at `D = 1e-13`, every `dx`, both wall signs, cell-centred, and all
    of them refusals that became answers.
    """
    for D in (1e-13, 3.75e-13, 7.0e-13, 5e-12):
        for dx in (0.001, 0.1, 1.0, 10.0):
            got = ghost_cell_fp_no_flux(1.0, 0.0, D, dx, +1.0, GridType.CELL_CENTERED)
            assert got == pytest.approx(1.0), f"D={D} dx={dx}: homogeneous Neumann must answer"

    #: CONTROL, in the same invocation: the guard must still fire where the ghost really is
    #: undetermined, or this test would pass over a predicate that never refuses anything.
    with pytest.raises(ValueError, match="singular ghost cell formula"):
        ghost_cell_fp_no_flux(1.0, 2.0, 0.1, 0.1, +1.0, GridType.CELL_CENTERED)
    #: and the wholly degenerate pair, where the scale itself is 0 and a strict `<` would answer
    with pytest.raises(ValueError, match="singular ghost cell formula"):
        ghost_cell_robin(1.0, 0.0, 0.0, 0.0, 0.1, GridType.CELL_CENTERED)


def test_zero_diffusion_has_a_stated_behaviour_on_both_centrings():
    """#2128 acceptance: `D -> 0` must have a stated, TESTED behaviour. It now has both, and they differ.

    Cell-centred returns a NEGATIVE ghost density. That is what the condition says -- the wall is the
    face, the face value is `(rho_g + rho_i)/2`, and `v_n * rho_wall = 0` with `v_n != 0` forces
    `rho_g = -rho_i` -- and `main` returns the same value by the same expression, so #2128 inherited
    it rather than causing it. Whether a Fokker-Planck solver should accept a negative ghost is
    #2220's question.

    Vertex-centred returns `interior_value`, because the guard preserved from before #2128 fires
    there. Whether THAT is right is #2220's question.

    Pinned as the status quo, not as an endorsement: the point is that both are now stated and a
    change to either fails a test instead of moving silently.
    """
    interior, v, dx, sign = 1.0, 0.4, 0.1, +1.0
    assert ghost_cell_fp_no_flux(interior, v, 0.0, dx, sign, GridType.CELL_CENTERED) == pytest.approx(-interior)
    assert ghost_cell_fp_no_flux(interior, v, 0.0, dx, sign, GridType.VERTEX_CENTERED) == pytest.approx(interior)


def test_the_conversion_is_reachable_from_the_package_surface():
    """The owner must be importable the way a second caller would import it.

    This function exists so a caller holding axis-frame `v` and `D` does not write the copy again.
    That argument is only true if such a caller can reach it. It was not: `normal_frame_coefficients`
    was absent from `geometry/boundary/__init__.py`'s lazy map while both its siblings resolved, so
    the public path raised `AttributeError` and only the deep module path worked -- which is how
    THIS FILE imports it, so the tests could not see the gap. Two rounds called it non-blocking; it
    stops being non-blocking once the deliverable's stated purpose depends on it.

    Pinned here because removing that line failed nothing: 2064 tests passed with the export gone.
    """
    import mfgarchon.geometry.boundary as boundary
    import mfgarchon.geometry.boundary.ghost_cells as ghost_cells

    #: `hasattr` alone passes through the lazy compat map and cannot see a missing `__all__`
    #: entry -- round 7 found the export half-landed with this test green. The convention is
    #: measured from the siblings, not assumed: they sit in `ghost_cells.__all__` and NOT in
    #: `boundary.__all__`, and each also has a lazy-map line. All three are required. Adding to
    #: `ghost_cells.__all__` also widens `applicator_base.__all__`, built here by star-import --
    #: consistent with the siblings, and not asserted here because it is that module's convention.
    assert "normal_frame_coefficients" in ghost_cells.__all__, (
        "the owner is missing from ghost_cells.__all__, so `import *` does not yield it"
    )
    #: CONTROL: a sibling must satisfy the same assertion, or this pins a convention that does
    #: not exist. And `boundary.__all__` is NOT the surface -- no sibling is in it.
    assert "ghost_cell_fp_no_flux" in ghost_cells.__all__, "control failed: wrong __all__ surface"
    assert "ghost_cell_fp_no_flux" not in boundary.__all__, "control failed: siblings are not in boundary.__all__"

    assert hasattr(boundary, "normal_frame_coefficients"), (
        "normal_frame_coefficients is not on the package surface. Add it to the lazy map in "
        "geometry/boundary/__init__.py beside ghost_cell_fp_no_flux -- the docstring's 'a second "
        "caller would otherwise write the copy again' is false while it is unreachable."
    )
    #: Same object by both routes, so the export cannot drift to a different symbol.
    assert boundary.normal_frame_coefficients is normal_frame_coefficients
    #: Control: a name that is not exported must still fail, or this test passes on any module.
    assert not hasattr(boundary, "zzz_not_an_exported_name")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
