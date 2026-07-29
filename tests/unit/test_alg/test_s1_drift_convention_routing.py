"""Pinning tests for Issue #1489 (S1) — route FP drift by ``_drift_convention``.

``resolve_fp_drift_kwargs`` decides how the value function ``U`` enters the FP solver.
Before this fix the ``use_velocity`` gate keyed on ``"drift_field" in params``, but
parameter presence cannot disambiguate the drift convention: some solvers expose
``drift_field`` as a real VELOCITY channel (fp_fvm / fp_gfdm / FPFDM), while the weak-form
family exposes ``drift_field`` as a DEPRECATED ALIAS for ``potential_field=U``
(``DriftConvention.VALUE_FUNCTION``). For a non-smooth ``H`` + a VALUE_FUNCTION solver the
old gate fired and set ``drift_field=alpha*`` (a velocity), which such a solver treats as
``U`` and DIFFERENTIATES — a silently wrong drift.

The fix threads the solver-declared ``_drift_convention`` into ``resolve_fp_drift_kwargs``:

(a) VALUE_FUNCTION + non-smooth H     -> raise (U cannot represent the Clarke velocity).
(b) VELOCITY      + non-smooth H     -> route ``drift_field=alpha*`` (computed velocity).
(c) any convention + smooth quadratic H -> ``potential_field=U`` (UNCHANGED no-regression).
(d) ``drift_convention=None``         -> pre-#1489 param-presence behavior, byte-identical.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.alg.numerical.coupling.fixed_point_utils import resolve_fp_drift_kwargs
from mfgarchon.alg.numerical.fp_solvers.base_fp import DriftConvention
from mfgarchon.core.hamiltonian import (
    L1ControlCost,
    QuadraticControlCost,
    SeparableHamiltonian,
)
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

# Solver signatures, reduced to the two params resolve_fp_drift_kwargs actually inspects.
# weak-form / network family: both present (drift_field is a deprecated alias for U).
_VALUE_FUNCTION_SIG = {"m_initial_condition", "potential_field", "drift_field"}
# meshfree velocity-only family (e.g. FPGFDMSolver): drift_field is a true velocity alpha*.
_VELOCITY_ONLY_SIG = {"m_initial_condition", "drift_field"}
# FDM reference solver: exposes both, drift_field is a true velocity alpha*.
_VELOCITY_BOTH_SIG = {"m_initial_condition", "potential_field", "drift_field"}


def _problem(control_cost) -> MFGProblem:
    """1D LQ-style MFG problem with the given control cost (smooth or non-smooth)."""
    geometry = TensorProductGrid(
        bounds=[(0.0, 1.0)],
        Nx_points=[21],
        boundary_conditions=no_flux_bc(dimension=1),
    )
    components = MFGComponents(
        m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2),
        u_terminal=lambda x: (x - 0.8) ** 2,
        hamiltonian=SeparableHamiltonian(
            control_cost=control_cost,
            coupling=lambda m: m,
            coupling_dm=lambda m: 1.0,
        ),
    )
    return MFGProblem(geometry=geometry, T=0.3, Nt=6, sigma=0.2, components=components)


def _state(problem: MFGProblem) -> tuple[np.ndarray, np.ndarray]:
    """Return (U, M) with real spatial gradients so alpha* is non-trivial."""
    nt = problem.Nt + 1
    x = np.linspace(0.0, 1.0, 21)
    U = np.tile((x - 0.8) ** 2, (nt, 1))
    M = np.tile(np.exp(-10 * (x - 0.5) ** 2), (nt, 1))
    return U, M


def test_l1_control_cost_is_non_smooth():
    """Guard the disambiguator's premise: L1 (bang-bang) cost is non-smooth."""
    assert L1ControlCost(lambda_=1.0).is_smooth() is False
    assert QuadraticControlCost(control_cost=1.0).is_smooth() is True


# (a) VALUE_FUNCTION + non-smooth H -> raise (the S1 fix).
def test_value_function_solver_nonsmooth_h_raises():
    problem = _problem(L1ControlCost(lambda_=1.0))
    U, M = _state(problem)
    with pytest.raises(ValueError, match="1489"):
        resolve_fp_drift_kwargs(
            problem,
            _VALUE_FUNCTION_SIG,
            None,
            U,
            M,
            drift_convention=DriftConvention.VALUE_FUNCTION,
        )


# (b) VELOCITY + non-smooth H -> route drift_field = alpha* (computed velocity, not U).
def test_velocity_solver_nonsmooth_h_routes_drift_field():
    problem = _problem(L1ControlCost(lambda_=1.0))
    U, M = _state(problem)
    drift_kwargs, use_positional = resolve_fp_drift_kwargs(
        problem,
        _VELOCITY_ONLY_SIG,
        None,
        U,
        M,
        drift_convention=DriftConvention.VELOCITY,
    )
    assert "drift_field" in drift_kwargs
    assert "potential_field" not in drift_kwargs
    assert not use_positional
    alpha_star = drift_kwargs["drift_field"]
    assert alpha_star is not U, "must be a computed alpha*, not U passed through"
    assert np.all(np.isfinite(alpha_star))


# (c) smooth quadratic H -> potential_field = U (UNCHANGED), for either declared convention.
@pytest.mark.parametrize(
    "convention",
    [DriftConvention.VALUE_FUNCTION, DriftConvention.VELOCITY],
)
def test_smooth_h_routes_potential_field_unchanged(convention):
    problem = _problem(QuadraticControlCost(control_cost=1.0))
    U, M = _state(problem)
    sig = _VALUE_FUNCTION_SIG if convention is DriftConvention.VALUE_FUNCTION else _VELOCITY_BOTH_SIG
    drift_kwargs, use_positional = resolve_fp_drift_kwargs(
        problem,
        sig,
        None,
        U,
        M,
        drift_convention=convention,
    )
    assert "potential_field" in drift_kwargs
    assert "drift_field" not in drift_kwargs
    assert drift_kwargs["potential_field"] is U
    assert not use_positional


# (d) drift_convention=None -> pre-#1489 behavior preserved (byte-identical fallback).
def test_none_convention_preserves_legacy_behavior():
    """Same inputs as case (a): with None the OLD param-presence gate fires (drift_field=alpha*,
    the pre-#1489 buggy-but-byte-identical path); declaring VALUE_FUNCTION instead raises."""
    problem = _problem(L1ControlCost(lambda_=1.0))
    U, M = _state(problem)

    # None -> old behavior: non-smooth H + drift_field in params -> use_velocity -> drift_field.
    drift_kwargs, use_positional = resolve_fp_drift_kwargs(
        problem, _VALUE_FUNCTION_SIG, None, U, M, drift_convention=None
    )
    assert "drift_field" in drift_kwargs
    assert "potential_field" not in drift_kwargs
    assert not use_positional

    # Smooth H + None -> potential_field=U (unchanged legacy path).
    smooth_problem = _problem(QuadraticControlCost(control_cost=1.0))
    Us, Ms = _state(smooth_problem)
    dk_smooth, _ = resolve_fp_drift_kwargs(smooth_problem, _VALUE_FUNCTION_SIG, None, Us, Ms, drift_convention=None)
    assert dk_smooth["potential_field"] is Us


def test_declared_conventions_match_solver_traits():
    """The disambiguator's source of truth: solver-declared _drift_convention. If any of these
    flip, resolve_fp_drift_kwargs silently reroutes drift, so pin them."""
    from mfgarchon.alg.numerical.fp_solvers.base_fp import BaseFPSolver
    from mfgarchon.alg.numerical.fp_solvers.fp_fvm import FPFVMSolver
    from mfgarchon.alg.numerical.fp_solvers.fp_gfdm import FPGFDMSolver
    from mfgarchon.alg.numerical.fp_solvers.fp_particle import FPParticleSolver
    from mfgarchon.alg.numerical.fp_solvers.fp_semi_lagrangian import FPSLJacobianSolver
    from mfgarchon.alg.numerical.weak_form_fp_solver import WeakFormFPSolver

    assert BaseFPSolver._drift_convention is DriftConvention.VELOCITY
    assert FPFVMSolver._drift_convention is DriftConvention.VELOCITY
    assert FPGFDMSolver._drift_convention is DriftConvention.VELOCITY
    assert WeakFormFPSolver._drift_convention is DriftConvention.VALUE_FUNCTION
    assert FPParticleSolver._drift_convention is DriftConvention.VALUE_FUNCTION
    assert FPSLJacobianSolver._drift_convention is DriftConvention.VALUE_FUNCTION


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_resolver_never_emits_a_deprecated_kwarg():
    """The kwarg ``resolve_fp_drift_kwargs`` emits must be live on the receiving solver.

    Issue #919 attached a deprecation to ``potential_field`` on ``FPFDMSolver`` — the
    *destination* of the v0.18.6 rename — pointing users back at ``drift_field``, whose
    meaning had just been changed to the velocity. Two failures followed, and this test
    pins both: the advice was wrong (obeying it advects ``U`` as if it were alpha*), and
    the deprecation was un-completable, because the resolver emits ``potential_field=U``
    on the default smooth-separable path that both Picard and Newton take. Removing it
    at the scheduled v0.25.0 would have broken the library's own primary path.
    """
    import inspect

    from mfgarchon.alg.numerical.coupling.fixed_point_utils import resolve_fp_drift_kwargs
    from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
    from mfgarchon.alg.numerical.fp_solvers.fp_fvm import FPFVMSolver
    from mfgarchon.alg.numerical.fp_solvers.fp_particle import FPParticleSolver
    from mfgarchon.alg.numerical.fp_solvers.fp_semi_lagrangian import FPSLJacobianSolver
    from mfgarchon.utils.deprecation import get_deprecated_parameters

    problem = _problem(QuadraticControlCost(control_cost=1.0))
    U, M = _state(problem)

    for solver_cls in (FPFDMSolver, FPFVMSolver, FPParticleSolver, FPSLJacobianSolver):
        method = solver_cls.solve_fp_system
        sig_params = set(inspect.signature(method).parameters)
        emitted, _ = resolve_fp_drift_kwargs(
            problem, sig_params, None, U, M, drift_convention=solver_cls._drift_convention
        )
        deprecated = {d["param"] for d in get_deprecated_parameters(method)}
        collision = set(emitted) & deprecated
        assert not collision, (
            f"{solver_cls.__name__}: resolve_fp_drift_kwargs emits {sorted(emitted)}, but "
            f"{sorted(collision)} is deprecated on this solver. Either the routing or the "
            f"deprecation is wrong — the library cannot schedule its own primary path for removal."
        )


def test_potential_field_is_never_deprecated_on_any_fp_solver():
    """``potential_field`` is the value-function channel on every FP solver that has one.

    On VALUE_FUNCTION solvers it is the canonical name; on VELOCITY solvers it is a live
    second channel (the solver forms alpha = -c*grad(U) internally). It is never an alias
    for ``drift_field``, so it must never carry a deprecation — in either direction.
    """
    import inspect

    from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
    from mfgarchon.alg.numerical.fp_solvers.fp_fvm import FPFVMSolver
    from mfgarchon.alg.numerical.fp_solvers.fp_particle import FPParticleSolver
    from mfgarchon.alg.numerical.fp_solvers.fp_semi_lagrangian import FPSLJacobianSolver
    from mfgarchon.alg.numerical.fp_solvers.fp_semi_lagrangian_adjoint import FPSLAdjointSolver
    from mfgarchon.alg.numerical.network_solvers.fp_network import FPNetworkSolver
    from mfgarchon.utils.deprecation import get_deprecated_parameters

    for solver_cls in (
        FPFDMSolver,
        FPFVMSolver,
        FPParticleSolver,
        FPSLJacobianSolver,
        FPSLAdjointSolver,
        FPNetworkSolver,
    ):
        method = solver_cls.solve_fp_system
        if "potential_field" not in inspect.signature(method).parameters:
            continue
        deprecated = {d["param"] for d in get_deprecated_parameters(method)}
        assert "potential_field" not in deprecated, (
            f"{solver_cls.__name__}: potential_field is marked deprecated. It is the value-"
            f"function channel, not an alias for drift_field, which carries the velocity."
        )
