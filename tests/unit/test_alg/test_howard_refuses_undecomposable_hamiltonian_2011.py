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


def _solve(hamiltonian, inner_solver, **solver_kwargs):
    x = np.linspace(0.0, L, NX)
    grid = TensorProductGrid(bounds=[(0.0, L)], Nx_points=[NX], boundary_conditions=no_flux_bc(dimension=1))
    components = MFGComponents(hamiltonian=hamiltonian, m_initial=lambda p: 1.0, u_terminal=lambda p: 0.0)
    problem = MFGProblem(geometry=grid, components=components, T=T, Nt=NT, sigma=0.5)
    solver = HJBGFDMSolver(problem, collocation_points=x.reshape(-1, 1), inner_solver=inner_solver, **solver_kwargs)
    m = np.tile(np.exp(-((x - 0.3) ** 2) / 0.02), (NT + 1, 1))
    m /= m[0].sum() * (L / (NX - 1))
    return np.asarray(
        solver.solve_hjb_system(M_density=m, U_terminal=np.zeros(NX), U_coupling_prev=np.zeros((NT + 1, NX)))
    )


def test_howard_refuses_a_hamiltonian_without_a_control_cost():
    with pytest.raises(NotImplementedError, match=r"exposes no\s+`?control_cost`?"):
        _solve(LocalCoupling(1.0), "howard", **SOCP)


def test_the_refusal_names_the_consequence_and_the_alternative():
    with pytest.raises(NotImplementedError) as exc:
        _solve(LocalCoupling(1.0), "howard", **SOCP)
    text = str(exc.value)
    assert "unit quadratic" in text, "must say WHAT would be substituted"
    assert "bitwise" in text, "must say the alpha-free part is dropped, not merely approximated"
    assert "newton" in text, "must name the working alternative"
    assert "2011" in text


def test_a_separable_hamiltonian_is_not_refused():
    """Control: a blanket ban on the Howard path would satisfy the two tests above."""
    u = _solve(SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)), "howard", **SOCP)
    assert np.all(np.isfinite(u)), "the Howard path must still run for the class it was built for"


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
    u = _solve(BareUnitQuadratic(), "howard", **SOCP)
    assert np.all(np.isfinite(u)), "Howard must accept a Hamiltonian its substitution is exact for"


def test_a_lambda_two_quadratic_is_refused_though_its_alpha_free_part_is_zero():
    """The other direction: zero alpha-free part, wrong Lagrangian, and it must still be refused."""
    with pytest.raises(NotImplementedError, match=r"unit quadratic"):
        _solve(LambdaTwoQuadratic(), "howard", **SOCP)
