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

This pins the refusal, not a repair. Reading the alpha-free part as `H(x, m, p=0, t)` is unsound --
it equals that part only when `H_control(0) = 0`, and `H = sqrt(1 + |p|^2)` injects a spurious 1.0 --
and a guard on "alpha-free part is non-zero" cannot catch the Lagrangian substitution, whose
alpha-free part is exactly zero. Admitting such a Hamiltonian correctly is #2011's remaining work.
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
SOCP = {"monotonicity_scheme": "joint_socp", "monotonicity_application": "precompute"}


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


def test_howard_refuses_a_hamiltonian_without_a_control_cost():
    with pytest.raises(NotImplementedError, match=r"exposes no\s+`?control_cost`?"):
        _solve(LocalCoupling(1.0), "howard", **SOCP)


def test_the_refusal_names_the_consequence_and_the_alternative():
    with pytest.raises(NotImplementedError) as exc:
        _solve(LocalCoupling(1.0), "howard", **SOCP)
    text = str(exc.value)
    assert "L(alpha) = (1/2)|alpha|^2" in text, "must say WHAT would be substituted"
    assert "bitwise" in text, "must say the alpha-free part is dropped, not merely approximated"
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
        **SOCP,
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
    u = _solve(BareUnitQuadratic(), "howard", terminal="cos", **SOCP)
    assert np.all(np.isfinite(u)), "Howard must accept a Hamiltonian its substitution is exact for"
    assert np.ptp(u) > 1e-6, "the solve returned a constant field; the Howard path did not run"


def test_a_lambda_two_quadratic_is_refused_though_its_alpha_free_part_is_zero():
    """The other direction: zero alpha-free part, wrong Lagrangian, and it must still be refused."""
    with pytest.raises(NotImplementedError, match=r"departs from \(1/2\)\|p\|\^2"):
        _solve(LambdaTwoQuadratic(), "howard", **SOCP)


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


_REFUSE = [
    ("hidden_coupling_f1_is_zero", lambda: _make("C", lambda x, m, q, p: 0.5 * q + 2.0 * x * (m - 1.0))),
    ("congestion_c1_is_one", _congestion),
    ("log_coupling", lambda: _make("E", lambda x, m, q, p: 0.5 * q + 2.0 * np.log(np.maximum(m, 1e-12)))),
    ("affine_coupling", lambda: _make("F", lambda x, m, q, p: 0.5 * q + 2.0 * (m - 1.0))),
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


@pytest.mark.parametrize(("name", "factory"), _REFUSE, ids=[n for n, _ in _REFUSE])
def test_the_probe_sees_hamiltonians_a_fixed_sample_point_cannot(name, factory):
    """Each of these was ACCEPTED by a probe pinned to m = ones, t = 0, p = e_0."""
    with pytest.raises(NotImplementedError):
        _solve(factory(), "howard", **SOCP)


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
        **SOCP,
    )
    assert np.all(np.isfinite(u))
