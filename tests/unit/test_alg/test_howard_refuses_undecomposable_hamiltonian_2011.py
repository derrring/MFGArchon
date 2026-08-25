"""Howard must refuse a Hamiltonian it cannot decompose, not solve a different problem (#2011).

`_solve_backward_howard` reconstructs the pieces Howard's policy evaluation needs by reading
attributes that only `SeparableHamiltonian` has:

- `control_cost` -> `control_lagrangian`. Absent, `hjb_howard` substitutes the UNIT quadratic
  `L(alpha) = (1/2)|alpha|^2`. Measured against Newton on `u_T = cos(2 pi x)`: a `lambda = 2`
  subclass is **31.4%** wrong relative, against a **5.5% control** from a unit quadratic on the
  same problem -- the control being the two inner solvers' own discretisation difference, so the
  signal is the gap between them, not the 31.4% alone.
- `_potential` / `_coupling` -> `has_H_extra`. Absent, the alpha-free part is dropped **bitwise**:
  `|u(g=1) - u(g=0)| = 0.000e+00` on Howard for `H = |p|^2/2 + g*x*m`, against `1.469e-01` for the
  same Hamiltonian on Newton.

The whole `control_cost` gate sat behind `if control_cost is not None:`, so the class it was written
to stop was the class that skipped it. Same shape as the `_congestion_factor` gate immediately below
it in that function, whose comment already says "Gate on the factor directly. Fail loud." -- and the
same shape as #1979 one layer over, where a capability gate keyed on BC *type* let a provider-valued
*coefficient* through.

#2011 SPLIT THIS FILE'S SUBJECT IN TWO, and the split is on which piece Howard can recover.

The alpha-free part it CAN. `howard_running_cost` evaluates the Hamiltonian at p = 0 through the
same `eval_H_batch` the Newton residual uses -- no private attribute, no decomposition -- and the
switch that builds it now also fires on the guard's own measurement of |H(x, m, 0, t)|, not only on
`_potential`/`_coupling`. Measured on `H = |p|^2/2 + g*x*m`, where it gave 0.0000e+00 before:
Howard 1.4687e-01 against Newton 1.4687e-01 under this file's `M_MATRIX_QP`. (Under the
`joint_socp`/`precompute` this file used before #2093, Howard gives 1.4586e-01 and Newton is
unchanged -- the scheme moves Howard by 0.7%, not the extraction.)

The control cost it CANNOT, and `_ke` still refuses on it. That is the real limit: Howard
substitutes a quadratic Lagrangian, and nothing recovers one that is not quadratic.

An earlier version of this docstring called the extraction unsound, on the grounds that
`H(x, m, p=0, t)` equals the alpha-free part only when `H_control(0) = 0`, with `H = sqrt(1+|p|^2)`
injecting a spurious 1.0. The objection is right in principle and `_ke` covers it in practice:
measured through the guard, that Hamiltonian is REFUSED with `_ke = 3.121e+03` against a tolerance
of 8.0e-09 -- six orders of magnitude, not a margin. What `_ke` cannot separate is `H_control(p) =
(1/2)|p|^2 + C` from a constant potential `V = C`, and no probe can: they are the same function of
(x, m, p, t). Treating `H(0)` as the alpha-free part is the standard normalisation, not an error.
That argument is now measured as well as reasoned: on `H = |p|^2/2 + C + g*x*m`, extracting `C` into
the running cost shifts `u` by exactly `C*(T - t)` -- `-0.200000` at `t = 0` and `0.000000` at
`t = T` for `C = 1, T = 0.2` -- IDENTICALLY under Howard and Newton, to six decimals at `C` of 0, 1
and 10. So `nabla u` is untouched, the policy and the FP drift are untouched, and the Howard-Newton
gap is bit-for-bit unchanged at `1.6907e-09` across all three. A constant cannot be extracted
wrongly because there is nothing to get wrong.

The Lagrangian-substitution case is unaffected -- its alpha-free part is exactly zero, so no
alpha-free gate was ever going to catch it, which is why `_ke` is the one that does.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.alg.numerical.hjb_solvers.hjb_gfdm import HJBGFDMSolver
from mfgarchon.core.hamiltonian import (
    HamiltonianBase,
    QuadraticControlCost,
    SeparableHamiltonian,
)
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

L, T, NX, NT = 1.0, 0.2, 21, 10
# `qp_m_matrix` rather than `joint_socp`: what this file pins is Howard's DECOMPOSITION gate,
# not SOCP stencils. joint_socp needs cvxpy (the `numerical` extra) and would make all 17 tests
# below skip wherever it is absent -- including the accept controls, which is the half you least
# want silently missing. qp_m_matrix runs on osqp, a base dependency, and exercises the same
# Howard path. SOCP itself is covered by test_socp_m_matrix_property, test_socp_stencil_enlargement
# and test_joint_socp_mirror_symmetry, which are about SOCP.
M_MATRIX_QP = {"monotonicity_scheme": "qp_m_matrix", "monotonicity_application": "always"}


def _first_coord(x):
    arr = np.asarray(x, dtype=float)
    return arr[..., 0] if arr.ndim > 1 else arr


class LocalCoupling(HamiltonianBase):
    """H = |p|^2/2 + g*x*m -- an alpha-free local coupling F(x, m) with a spatial weight.

    Dual-convention `__call__` (batch and single point), which is what a HamiltonianBase subclass
    owes; `dp` and `dm` are supplied in closed form rather than differenced.
    """

    def __init__(self, g: float) -> None:
        super().__init__()
        self.g = g

    def __call__(self, x, m, p, t=0.0):
        # axis=-1, matching the (n_points, dim) batch convention the solver passes. An earlier
        # draft summed over axis=0, which broadcasts to the wrong shape and makes this a DIFFERENT
        # Hamiltonian -- caught by the mutation check below, not by any assertion.
        pa = np.asarray(p, dtype=float)
        kinetic = 0.5 * np.sum(pa**2, axis=-1) if pa.ndim > 1 else 0.5 * pa**2
        return kinetic + self.g * _first_coord(x) * np.asarray(m, dtype=float)

    def dp(self, x, m, p, t=0.0):
        return np.asarray(p, dtype=float)

    def dm(self, x, m, p, t=0.0):
        return self.g * _first_coord(x) * np.ones_like(np.asarray(m, dtype=float))


def _solve(hamiltonian, inner_solver, *, terminal=None, **solver_kwargs):
    """`terminal=None` gives u_T = 0, for which the exact solution is u == 0 -- fine for a
    `raises` test and USELESS as an accept control, since a solver returning zeros passes any
    `isfinite` assertion. Accept controls pass `terminal="cos"`."""
    x = np.linspace(0.0, L, NX)
    grid = TensorProductGrid(bounds=[(0.0, L)], Nx_points=[NX], boundary_conditions=no_flux_bc(dimension=1))

    def _uT(point):
        return float(np.cos(2.0 * np.pi * np.asarray(point).ravel()[0])) if terminal == "cos" else 0.0

    components = MFGComponents(hamiltonian=hamiltonian, m_initial=lambda p: 1.0, u_terminal=_uT)
    problem = MFGProblem(geometry=grid, components=components, T=T, Nt=NT, sigma=0.5)
    solver = HJBGFDMSolver(problem, collocation_points=x.reshape(-1, 1), inner_solver=inner_solver, **solver_kwargs)
    m = np.tile(np.exp(-((x - 0.3) ** 2) / 0.02), (NT + 1, 1))
    m /= m[0].sum() * (L / (NX - 1))
    u_terminal = np.cos(2.0 * np.pi * x) if terminal == "cos" else np.zeros(NX)
    return np.asarray(
        solver.solve_hjb_system(M_density=m, U_terminal=u_terminal, U_coupling_prev=np.tile(u_terminal, (NT + 1, 1)))
    )


def test_howard_extracts_an_alpha_free_part_it_was_never_told_about():
    """#2011 item 1. Howard used to DROP this bitwise; it now agrees with Newton.

    `H = |p|^2/2 + g*x*m` carries an alpha-free local coupling F(x, m) on a class that exposes
    neither `_potential` nor `_coupling`. The switch that built Howard's running-cost closure was
    keyed on exactly those two SeparableHamiltonian internals, so the coupling was discarded and
    the solve returned the g = 0 answer. Measured on the issue: 0.0000e+00 on Howard against
    1.469e-01 on Newton.

    It is now keyed on the guard's OWN measurement of |H(x, m, 0, t)| as well, and the extraction
    was already convention-free -- `howard_running_cost` evaluates the Hamiltonian at p = 0 through
    the same `eval_H_batch` the Newton residual uses, which needs no private attribute.

    Newton is the live control, not a recorded number: it reads the Hamiltonian through H() and
    dp() and needs no decomposition, so it says what the coupling is worth on this problem. The two
    inner solvers discretise differently, hence a tolerance rather than equality.
    """

    def spread(inner, **kw):
        u0 = _solve(LocalCoupling(0.0), inner, **kw)
        u1 = _solve(LocalCoupling(1.0), inner, **kw)
        assert np.all(np.isfinite(u1))
        return float(np.abs(u1 - u0).max())

    newton = spread("newton")
    howard = spread("howard", **M_MATRIX_QP)

    assert newton > 1e-3, "the control itself must see the coupling, or this test proves nothing"
    assert howard > 1e-3, f"Howard still dropped the alpha-free part: {howard:.4e}"
    assert abs(howard - newton) / newton < 0.05, (
        f"Howard {howard:.4e} vs Newton {newton:.4e} -- extracted, but not the same problem"
    )


def test_the_refusal_names_the_consequence_and_the_alternative():
    """The refusal that REMAINS is the kinetic one: Howard substitutes a quadratic Lagrangian.

    `H = |p|^4/2` agrees with the unit quadratic at |p| = 0 and 1 -- the only two points an earlier
    probe sampled -- and no extraction recovers a control cost that is not quadratic. That is why
    `_ke` is still a refusal while the alpha-free gate is not.
    """
    with pytest.raises(NotImplementedError) as exc:
        _solve(_quartic(), "howard", **M_MATRIX_QP)
    text = str(exc.value)
    assert "L(alpha) = (1/2)|alpha|^2" in text, "must say WHAT would be substituted"
    assert "not the problem" in text, (
        "must distinguish the alpha-free part, which is now extracted, from the control cost, "
        "which is what it is refusing -- otherwise a reader retries with a different coupling"
    )
    assert "M_collocation slices" in text, (
        "must disclose WHERE it probed -- an earlier version pinned m to ones and t to 0 and said "
        "neither, so a reader would believe their own density had been checked"
    )
    assert "newton" in text, "must name the working alternative"
    assert "2011" in text


def test_a_separable_hamiltonian_is_not_refused():
    """Control: a blanket ban on the Howard path would satisfy the two tests above.

    `terminal="cos"` is not decoration. With u_T = 0 the exact solution is identically zero, so a
    solver returning `np.zeros(...)` passes any `isfinite` assertion -- an earlier version of this
    control did exactly that and was verified to survive replacing the whole Howard sweep with
    zeros.
    """
    u = _solve(
        SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
        "howard",
        terminal="cos",
        **M_MATRIX_QP,
    )
    assert np.all(np.isfinite(u)), "the Howard path must still run for the class it was built for"
    assert np.ptp(u) > 1e-6, "the solve returned a constant field; the Howard path did not run"


def test_newton_still_accepts_the_same_hamiltonian():
    """Control: the refusal is Howard's, not a ban on non-separable Hamiltonians.

    Newton reads the Hamiltonian through H() and dp() and needs no decomposition, so the same
    subclass must go through -- and must actually be honoured, not merely tolerated.
    """
    u0 = _solve(LocalCoupling(0.0), "newton")
    u1 = _solve(LocalCoupling(1.0), "newton")
    assert np.all(np.isfinite(u1))
    assert np.abs(u1 - u0).max() > 1e-3, (
        "Newton accepted the Hamiltonian but the coupling had no effect -- if this fails, the "
        "Newton path has acquired the same blindness and the refusal above is hiding it"
    )


class BareUnitQuadratic(HamiltonianBase):
    """H = |p|^2/2, written as a bare subclass with no `control_cost` attribute.

    Howard's substituted Lagrangian is EXACT for this, so it must be accepted. The first version of
    this guard keyed on `getattr(H_class, "control_cost", None) is None` and refused it -- breaking
    seven existing tests whose fixture is exactly this shape. That is the same mistake the guard
    exists to fix, one level down: a predicate on an ATTRIBUTE standing in for a question about
    BEHAVIOUR.
    """

    def __call__(self, x, m, p, t=0.0):
        pa = np.asarray(p, dtype=float)
        return 0.5 * np.sum(pa**2, axis=-1) if pa.ndim == 2 else 0.5 * float(np.sum(pa**2))

    def dp(self, x, m, p, t=0.0):
        return np.asarray(p, dtype=float)


class LambdaTwoQuadratic(HamiltonianBase):
    """H = |p|^2/4, i.e. a lambda = 2 control cost, and an alpha-free part of exactly ZERO.

    The case that makes an attribute predicate insufficient in the other direction: nothing about
    this class's shape is unusual, its alpha-free part is zero so a guard on "alpha-free part is
    non-zero" cannot fire, and Howard's unit-quadratic substitution is nonetheless wrong -- measured
    at 31.4% against Newton, against a 5.5% control.
    """

    def __call__(self, x, m, p, t=0.0):
        pa = np.asarray(p, dtype=float)
        return 0.25 * np.sum(pa**2, axis=-1) if pa.ndim == 2 else 0.25 * float(np.sum(pa**2))

    def dp(self, x, m, p, t=0.0):
        return 0.5 * np.asarray(p, dtype=float)


def test_a_bare_unit_quadratic_subclass_is_ACCEPTED():
    """The discriminating control: refusing this is what the attribute-keyed first version did."""
    u = _solve(BareUnitQuadratic(), "howard", terminal="cos", **M_MATRIX_QP)
    assert np.all(np.isfinite(u)), "Howard must accept a Hamiltonian its substitution is exact for"
    assert np.ptp(u) > 1e-6, "the solve returned a constant field; the Howard path did not run"


def test_a_lambda_two_quadratic_is_refused_though_its_alpha_free_part_is_zero():
    """The other direction: zero alpha-free part, wrong Lagrangian, and it must still be refused."""
    with pytest.raises(NotImplementedError, match=r"departs from \(1/2\)\|p\|\^2"):
        _solve(LambdaTwoQuadratic(), "howard", **M_MATRIX_QP)


# ---------------------------------------------------------------------------------------------
# The sampling matrix. An earlier version of the guard probed at m = ones, t = 0 and p = e_0, and
# ACCEPTED every "refuse" row below -- measured. Nothing in the old test file pinned any of that:
# four mutations of the probe's sampling (m = 2*ones, t = 1.0, one collocation point instead of the
# cloud, .mean() instead of .max()) all passed it. These cases are the pin.
# ---------------------------------------------------------------------------------------------


def _dual(fn):
    """Wrap a batch formula fn(x1, m, |p|^2, p) so it also answers a single point.

    A HamiltonianBase subclass owes both conventions; without the single-point branch, MFGProblem
    construction fails in `dm`'s finite-difference default with a message naming a different method.
    """

    def call(self, x, m, p, t=0.0):
        pa = np.asarray(p, dtype=float)
        xa = np.asarray(x, dtype=float)
        ma = np.asarray(m, dtype=float)
        batch = pa.ndim > 1
        x1 = xa[..., 0] if xa.ndim > 1 else xa.ravel()
        psq = np.sum(pa**2, axis=-1) if batch else float(np.sum(pa**2))
        out = fn(x1, ma, psq, pa)
        return out if batch else float(np.asarray(out).ravel()[0])

    return call


def _make(name, fn, *, dp=None, extra=None):
    ns = {"__call__": _dual(fn), "dp": dp or (lambda s, x, m, p, t=0.0: np.asarray(p, dtype=float))}
    if extra:
        ns.update(extra)
    return type(name, (HamiltonianBase,), ns)()


def _congestion():
    """H = |p|^2/(2m): Lasry-Lions multiplicative kinetic congestion with c(1) = 1.

    The sharpest case. The `_congestion_factor` gate twenty lines below the probe refuses the real
    `CongestionHamiltonian` by name; the same physics as a bare subclass has to be refused here or
    the two gates disagree about the same object. c(1) = 1 makes it invisible at m = ones.
    """

    def call(self, x, m, p, t=0.0):
        pa = np.asarray(p, dtype=float)
        ma = np.asarray(m, dtype=float)
        batch = pa.ndim > 1
        q = np.sum(pa**2, axis=-1) if batch else float(np.sum(pa**2))
        v = 0.5 * q / np.maximum(ma, 1e-12)
        return v if batch else float(np.asarray(v).ravel()[0])

    def dp(self, x, m, p, t=0.0):
        pa = np.asarray(p, dtype=float)
        ma = np.asarray(m, dtype=float)
        return pa / np.maximum(ma, 1e-12)[..., None] if pa.ndim > 1 else pa / float(np.asarray(ma).ravel()[0])

    return type("CongestionBare", (HamiltonianBase,), {"__call__": call, "dp": dp})()


def _quartic():
    """H = |p|^4/2: agrees with the unit quadratic at |p| = 0 AND |p| = 1, the only two points the
    earlier probe sampled."""

    def call(self, x, m, p, t=0.0):
        pa = np.asarray(p, dtype=float)
        batch = pa.ndim > 1
        q = np.sum(pa**2, axis=-1) if batch else float(np.sum(pa**2))
        v = 0.5 * q**2
        return v if batch else float(np.asarray(v).ravel()[0])

    def dp(self, x, m, p, t=0.0):
        pa = np.asarray(p, dtype=float)
        q = np.sum(pa**2, axis=-1) if pa.ndim > 1 else float(np.sum(pa**2))
        return 2.0 * (q[..., None] if pa.ndim > 1 else q) * pa

    return type("QuarticBare", (HamiltonianBase,), {"__call__": call, "dp": dp})()


# #2011 item 1 SPLIT THIS LIST, and the split line landed on the physics rather than being drawn.
# Every entry here has a control cost that is NOT the unit quadratic -- m-dependent, quartic,
# anisotropic -- and no probe recovers one, so Howard's substituted Lagrangian would be a different
# problem. These stay refusals.
_REFUSE = [
    ("congestion_c1_is_one", _congestion),
    ("quartic_kinetic", _quartic),
    (
        "anisotropic_kinetic",
        lambda: _make(
            "H",
            lambda x, m, q, p: (
                0.5 * np.asarray(p)[..., 0] ** 2 + 2.0 * np.asarray(p)[..., 1] ** 2
                if np.asarray(p).ndim > 1
                else 0.5 * q
            ),
        ),
    ),
]


# The other half: kinetic part IS (1/2)|p|^2, everything else is an alpha-free F(x, m). These were
# refused until #2011 because the switch was keyed on `_potential`/`_coupling` and none of them
# sets either. They are now extracted as H(x, m, 0, t).
_EXTRACT = [
    ("hidden_coupling_f1_is_zero", lambda: _make("C", lambda x, m, q, p: 0.5 * q + 2.0 * x * (m - 1.0))),
    ("log_coupling", lambda: _make("E", lambda x, m, q, p: 0.5 * q + 2.0 * np.log(np.maximum(m, 1e-12)))),
    ("affine_coupling", lambda: _make("F", lambda x, m, q, p: 0.5 * q + 2.0 * (m - 1.0))),
]


@pytest.mark.parametrize(("name", "factory"), _REFUSE, ids=[n for n, _ in _REFUSE])
def test_the_probe_sees_hamiltonians_a_fixed_sample_point_cannot(name, factory):
    """Each of these was ACCEPTED by a probe pinned to m = ones, t = 0, p = e_0.

    All that survives here is the KINETIC refusal. The alpha-free entries moved to `_EXTRACT`
    below, because #2011 made them solvable rather than merely detectable -- a gate that refuses
    what it could compute is not a safety property.
    """
    with pytest.raises(NotImplementedError):
        _solve(factory(), "howard", **M_MATRIX_QP)


def _pure_bump():
    """H = |p|^2/2 + a C^1 bump supported on |p| in (3, 10), and NO alpha-free part.

    The bump sits where the solve lives -- max|grad u_T| for u_T = cos(2 pi x) is 2*pi = 6.28 --
    and the old probe ladder stepped over it: `{0.5, 1, 2} x _gT` with _gT = 40 gives
    {0.5, 1, 2, 20, 40, 80}, so (2, 20) was never sampled. `_gT` is a spacing bound, not a
    gradient, and overestimates by 6.4x here, which is what pushed the upper rungs past the hole.

    Alpha-free part is exactly zero on purpose: it makes `_af_bad` unable to fire, so a refusal
    can only come from `_ke`. Without that, this fixture would be caught for the wrong reason and
    would keep passing if the ladder regressed.
    """

    def call(self, x, m, p, t=0.0):
        pa = np.asarray(p, dtype=float)
        q = np.asarray(np.sum(pa**2, axis=-1) if pa.ndim > 1 else float(np.sum(pa**2)), dtype=float)
        return 0.5 * q + np.maximum(0.0, q - 9.0) ** 2 * np.maximum(0.0, 100.0 - q) ** 2 / 1.0e5

    def dp(self, x, m, p, t=0.0):
        pa = np.asarray(p, dtype=float)
        q = np.asarray(np.sum(pa**2, axis=-1) if pa.ndim > 1 else float(np.sum(pa**2)), dtype=float)
        db = (
            2 * np.maximum(0.0, q - 9.0) * np.maximum(0.0, 100.0 - q) ** 2
            - 2 * np.maximum(0.0, q - 9.0) ** 2 * np.maximum(0.0, 100.0 - q)
        ) / 1.0e5
        f = 1.0 + db
        return pa * (f[..., None] if pa.ndim > 1 else float(f))

    return type("PureBump", (HamiltonianBase,), {"__call__": call, "dp": dp})()


def _nan_at_one_probe_point():
    """H = |p|^4/2, returning NaN at |p| = 1.

    |p| = 1 is in the `{0.5, 1, 2}` union, so it is a probe magnitude at EVERY value of the
    gradient bound. An earlier version placed the NaN at |p| = 80, which existed only because
    `_gT = spread/hmin` was 40 on this fixture -- so the test was coupled to the very quantity
    #2072 replaces, and would have gone green-by-vacuity the moment the bound was corrected.
    """

    def call(self, x, m, p, t=0.0):
        pa = np.asarray(p, dtype=float)
        q = np.asarray(np.sum(pa**2, axis=-1) if pa.ndim > 1 else float(np.sum(pa**2)), dtype=float)
        return np.where(np.abs(q - 1.0) < 1e-9, np.nan, 0.5 * q**2)

    def dp(self, x, m, p, t=0.0):
        pa = np.asarray(p, dtype=float)
        q = np.asarray(np.sum(pa**2, axis=-1) if pa.ndim > 1 else float(np.sum(pa**2)), dtype=float)
        return 2.0 * (q[..., None] if pa.ndim > 1 else float(q)) * pa

    return type("NanQuartic", (HamiltonianBase,), {"__call__": call, "dp": dp})()


def _pure_unit_quadratic():
    """H = |p|^2/2 exactly. The false-refusal control for a denser probe ladder."""

    def call(self, x, m, p, t=0.0):
        pa = np.asarray(p, dtype=float)
        return 0.5 * np.asarray(np.sum(pa**2, axis=-1) if pa.ndim > 1 else float(np.sum(pa**2)), dtype=float)

    def dp(self, x, m, p, t=0.0):
        return np.asarray(p, dtype=float)

    return type("PureUnitQuadratic", (HamiltonianBase,), {"__call__": call, "dp": dp})()


def _bump_on(lo_q, hi_q, amplitude):
    """H = |p|^2/2 + a C^1 bump on |p|^2 in (lo_q, hi_q). No alpha-free part, so `_af` is 0 and a
    refusal can only come from `_ke` -- otherwise the fixture would be caught for the wrong reason
    and would keep passing if the probe regressed."""

    def call(self, x, m, p, t=0.0):
        pa = np.asarray(p, dtype=float)
        q = np.asarray(np.sum(pa**2, axis=-1) if pa.ndim > 1 else float(np.sum(pa**2)), dtype=float)
        return 0.5 * q + amplitude * np.maximum(0.0, q - lo_q) ** 2 * np.maximum(0.0, hi_q - q) ** 2

    def dp(self, x, m, p, t=0.0):
        pa = np.asarray(p, dtype=float)
        q = np.asarray(np.sum(pa**2, axis=-1) if pa.ndim > 1 else float(np.sum(pa**2)), dtype=float)
        db = amplitude * (
            2 * np.maximum(0.0, q - lo_q) * np.maximum(0.0, hi_q - q) ** 2
            - 2 * np.maximum(0.0, q - lo_q) ** 2 * np.maximum(0.0, hi_q - q)
        )
        f = 1.0 + db
        return pa * (f[..., None] if pa.ndim > 1 else float(f))

    return type("BumpedKinetic", (HamiltonianBase,), {"__call__": call, "dp": dp})()


@pytest.mark.parametrize(
    ("name", "lo_q", "hi_q", "amp"),
    [
        # straddles the magnitude the solve actually visits (|p| ~ 6.18 on this fixture)
        ("operating_range", 26.01, 62.41, 5e-4),
        # the wider bump the first version of this guard fix used
        ("wide", 9.0, 100.0, 1e-5),
    ],
)
def test_the_probe_finds_a_kinetic_defect_at_the_magnitudes_the_solve_visits(name, lo_q, hi_q, amp):
    """#2072. The first version of this fix replaced the ladder with a geometric span and claimed
    "a span cannot have that hole". It can: adjacent rungs of a 12-point geomspace sit at a ratio
    of ~1.59, so a bump narrower than that in relative width fits between two -- and the
    `operating_range` case below is exactly such a bump, accepted at 17.4% error against a 2.23%
    control by that version.

    What closes it is the gradient bound, not the ladder shape. `_gT` was `spread/hmin`, which
    divides a GLOBAL range by a LOCAL spacing and so diverges under refinement (20, 40, 80, 200 at
    nx = 11, 21, 41, 201) while the gradient it stands for converges (3.09 -> 3.14). The discrete
    Lipschitz constant tracks the truth, so the ladder's rungs land where the solve lives.
    """
    with pytest.raises(NotImplementedError, match="cannot decompose"):
        _solve(_bump_on(lo_q, hi_q, amp), "howard", terminal="cos", **M_MATRIX_QP)


def test_a_probe_that_cannot_be_evaluated_is_not_a_probe_that_passed():
    """#2072: NaN made the guard ACCEPT, and the comment claimed the opposite.

    `np.maximum(0.0, nan)` is `nan` and `nan > tol` is False; `max(0.0, nan)` is `0.0` and
    `0.0 > tol` is False. Identical acceptance -- the choice of `max` was never what made this
    safe. Measured before the fix: accepted, all-finite output, 153% wrong against Newton.
    """
    with pytest.raises(NotImplementedError, match="non-finite"):
        _solve(_nan_at_one_probe_point(), "howard", terminal="cos", **M_MATRIX_QP)


def test_the_denser_ladder_does_not_false_refuse_a_genuine_quadratic():
    """The control for the two above. A denser probe can only make a TRUE refusal stricter --
    a genuine quadratic matches the kinetic reference at every |p| -- and this pins that."""
    u = _solve(_pure_unit_quadratic(), "howard", terminal="cos", **M_MATRIX_QP)
    assert np.all(np.isfinite(u))
    assert np.abs(u).max() > 1e-6


@pytest.mark.parametrize(("name", "factory"), _EXTRACT, ids=[n for n, _ in _EXTRACT])
def test_an_unwired_alpha_free_part_is_extracted_not_refused(name, factory):
    """#2011: accepted AND acted on. Asserting only that it does not raise would be vacuous.

    The terminal condition is `cos` deliberately: with u_T = 0 the exact solution is u == 0, so a
    solver that silently returned zeros would satisfy any `isfinite` check and this test would pass
    while proving nothing -- the trap `_solve`'s own docstring names.
    """
    u = _solve(factory(), "howard", terminal="cos", **M_MATRIX_QP)
    assert np.all(np.isfinite(u))
    assert np.abs(u).max() > 1e-6, "accepted, but the solve produced nothing to check"

    # the alpha-free part must MOVE the answer: compare against the same Hamiltonian with it gone
    baseline = _solve(BareUnitQuadratic(), "howard", terminal="cos", **M_MATRIX_QP)
    assert np.abs(u - baseline).max() > 1e-3, (
        f"{name} was accepted but its alpha-free part changed nothing -- extraction is a no-op, "
        "which is the pre-#2011 behaviour wearing a green test"
    )


def test_a_wired_alpha_free_part_is_NOT_refused():
    """The false-refusal control, and it is the one a naive fix gets wrong.

    `has_H_extra` is duck-typed on `_potential` / `_coupling`, so ANY class setting them gets its
    alpha-free part wired through the same `H(x, m, p=0, t)` the probe measures. A guard that
    refuses on a non-zero alpha-free part without asking whether it is wired refuses a Hamiltonian
    that works: measured on main, this one agrees with Newton to 2.30% -- pure discretisation error.
    """
    u = _solve(
        _make(
            "Wired",
            lambda x, m, q, p: 0.5 * q + 1.0 * x,
            extra={"_potential": (lambda xx, t=0.0: 1.0 * np.asarray(xx).ravel()[0])},
        ),
        "howard",
        **M_MATRIX_QP,
    )
    assert np.all(np.isfinite(u))
