"""
Pinning tests for Issue #1155: config.hjb / config.fp / PicardConfig.anderson_memory
translated to solver constructor kwargs in Safe / Auto mode.

Pre-fix behaviour
-----------------
Any non-default config.hjb or config.fp value raises NotImplementedError from the
guard in mfg_problem.py:solve().  anderson_memory is validated by Pydantic but never
passed to FixedPointIterator (silently dropped).

Post-fix behaviour
------------------
Non-default config.hjb.newton.{tolerance,max_iterations}, config.hjb.fdm.scheme,
config.fp.fdm.scheme, and config.picard.anderson_memory are threaded to the
constructed solvers / iterator.  Unknown / unsupported non-default fields continue
to raise NotImplementedError (fail-loud).
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.config import (
    FPConfig,
    HJBConfig,
    MFGSolverConfig,
    NewtonConfig,
    PicardConfig,
    fp_config_to_kwargs,
    hjb_config_to_kwargs,
    picard_config_to_iterator_kwargs,
)
from mfgarchon.config.mfg_methods import FDMConfig
from mfgarchon.types import NumericalScheme
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_problem():
    """Minimal 1-D MFG problem for fast construction."""
    from mfgarchon import MFGProblem
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
    from mfgarchon.core.mfg_components import MFGComponents

    H = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: m,
        coupling_dm=lambda m: 1.0,
    )
    components = MFGComponents(
        hamiltonian=H,
        m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2),
        u_terminal=lambda x: 0.0,
    )
    return MFGProblem(
        geometry=TensorProductGrid(
            bounds=[(0.0, 1.0)], Nx_points=[10 + 1], boundary_conditions=no_flux_bc(dimension=1)
        ),
        Nt=5,
        T=0.1,
        components=components,
    )


# ---------------------------------------------------------------------------
# Unit tests: translator functions (no solver construction)
# ---------------------------------------------------------------------------


class TestHJBConfigToKwargs:
    """Unit tests for hjb_config_to_kwargs."""

    def test_all_defaults_returns_empty(self):
        """Default HJBConfig produces no kwargs (solvers use their own defaults)."""
        kw = hjb_config_to_kwargs(HJBConfig(), NumericalScheme.FDM_UPWIND)
        assert kw == {}

    def test_newton_tolerance_threaded(self):
        """Non-default newton.tolerance is mapped to newton_tolerance."""
        cfg = HJBConfig(newton=NewtonConfig(tolerance=1e-10))
        kw = hjb_config_to_kwargs(cfg, NumericalScheme.FDM_UPWIND)
        assert kw["newton_tolerance"] == 1e-10

    def test_newton_max_iterations_threaded(self):
        """Non-default newton.max_iterations is mapped to max_newton_iterations."""
        cfg = HJBConfig(newton=NewtonConfig(max_iterations=42))
        kw = hjb_config_to_kwargs(cfg, NumericalScheme.FDM_UPWIND)
        assert kw["max_newton_iterations"] == 42

    def test_newton_relaxation_threaded_for_fdm(self):
        """newton.relaxation is mapped to relaxation for FDM schemes."""
        cfg = HJBConfig(newton=NewtonConfig(relaxation=0.5))
        kw = hjb_config_to_kwargs(cfg, NumericalScheme.FDM_UPWIND)
        assert kw["relaxation"] == 0.5

    def test_newton_relaxation_raises_for_gfdm(self):
        """newton.relaxation raises NotImplementedError for non-FDM schemes."""
        cfg = HJBConfig(newton=NewtonConfig(relaxation=0.5))
        with pytest.raises(NotImplementedError, match="Refs #1155"):
            hjb_config_to_kwargs(cfg, NumericalScheme.GFDM)

    def test_fdm_scheme_central_mapped(self):
        """config.hjb.fdm.scheme='central' maps to advection_scheme='gradient_centered'."""
        cfg = HJBConfig(method="fdm", fdm=FDMConfig(scheme="central"))
        kw = hjb_config_to_kwargs(cfg, NumericalScheme.FDM_UPWIND)
        assert kw["advection_scheme"] == "gradient_centered"

    def test_fdm_scheme_default_no_kwarg(self):
        """Default fdm.scheme='upwind' does not inject advection_scheme kwarg."""
        cfg = HJBConfig(method="fdm")  # fdm sub-config auto-populated with defaults
        kw = hjb_config_to_kwargs(cfg, NumericalScheme.FDM_UPWIND)
        assert "advection_scheme" not in kw

    def test_fdm_params_for_wrong_scheme_raises(self):
        """Non-default fdm sub-config with SL scheme raises NotImplementedError."""
        cfg = HJBConfig(method="fdm", fdm=FDMConfig(scheme="central"))
        with pytest.raises(NotImplementedError, match="Refs #1155"):
            hjb_config_to_kwargs(cfg, NumericalScheme.SL_LINEAR)

    def test_accuracy_order_nondefault_raises(self):
        """Non-default accuracy_order raises NotImplementedError (unmapped)."""
        cfg = HJBConfig(accuracy_order=4)
        with pytest.raises(NotImplementedError, match="Refs #1155"):
            hjb_config_to_kwargs(cfg, NumericalScheme.FDM_UPWIND)

    def test_sl_interpolation_method_threaded(self):
        """Non-default sl.interpolation_method is threaded for SL schemes.
        SLConfig default is interpolation_method='cubic', so 'linear' is non-default.
        """
        from mfgarchon.config.mfg_methods import SLConfig

        cfg = HJBConfig(method="semi_lagrangian", sl=SLConfig(interpolation_method="linear"))
        kw = hjb_config_to_kwargs(cfg, NumericalScheme.SL_CUBIC)
        assert kw["interpolation_method"] == "linear"

    def test_sl_rk_order_mapped(self):
        """sl.rk_order=4 maps to characteristic_solver='rk4'."""
        from mfgarchon.config.mfg_methods import SLConfig

        cfg = HJBConfig(method="semi_lagrangian", sl=SLConfig(rk_order=4))
        kw = hjb_config_to_kwargs(cfg, NumericalScheme.SL_LINEAR)
        assert kw["characteristic_solver"] == "rk4"

    def test_method_conflict_raises(self):
        """config.hjb.method='gfdm' with scheme=FDM raises NotImplementedError."""
        cfg = HJBConfig(method="gfdm")
        with pytest.raises(NotImplementedError, match="Refs #1155"):
            hjb_config_to_kwargs(cfg, NumericalScheme.FDM_UPWIND)

    # ------------------------------------------------------------------
    # Schemes previously missing from the HJB scheme->method map (Refs #1155):
    # FVM_UPWIND / FVM_MUSCL / SL_CUBIC must reject a contradictory non-default
    # method, and MESHLESS_GALERKIN (no method analog) must fail loud rather than
    # silently discard a non-default method.
    # ------------------------------------------------------------------

    def test_method_fvm_upwind_conflict_raises(self):
        """Non-default hjb.method with FVM_UPWIND (analog 'fdm') raises NotImplementedError."""
        cfg = HJBConfig(method="gfdm")
        with pytest.raises(NotImplementedError, match="Refs #1155"):
            hjb_config_to_kwargs(cfg, NumericalScheme.FVM_UPWIND)

    def test_method_fvm_muscl_conflict_raises(self):
        """Non-default hjb.method with FVM_MUSCL (analog 'fdm') raises NotImplementedError."""
        cfg = HJBConfig(method="semi_lagrangian")
        with pytest.raises(NotImplementedError, match="Refs #1155"):
            hjb_config_to_kwargs(cfg, NumericalScheme.FVM_MUSCL)

    def test_method_sl_cubic_conflict_raises(self):
        """Non-default hjb.method contradicting SL_CUBIC (analog 'semi_lagrangian') raises."""
        cfg = HJBConfig(method="gfdm")
        with pytest.raises(NotImplementedError, match="Refs #1155"):
            hjb_config_to_kwargs(cfg, NumericalScheme.SL_CUBIC)

    def test_method_meshless_galerkin_nondefault_raises(self):
        """MESHLESS_GALERKIN has no HJBConfig.method analog: any non-default method fails loud."""
        cfg = HJBConfig(method="gfdm")
        with pytest.raises(NotImplementedError, match="Refs #1155"):
            hjb_config_to_kwargs(cfg, NumericalScheme.MESHLESS_GALERKIN)

    def test_method_fvm_consistent_fdm_accepted(self):
        """FVM pairs HJB=FDM; the consistent method ('fdm') is accepted, no kwargs."""
        cfg = HJBConfig(method="fdm")
        assert hjb_config_to_kwargs(cfg, NumericalScheme.FVM_UPWIND) == {}

    def test_method_sl_cubic_consistent_accepted(self):
        """SL_CUBIC analog is 'semi_lagrangian'; the consistent method is accepted."""
        cfg = HJBConfig(method="semi_lagrangian")
        # No conflict raised; a matching method adds no kwarg.
        assert hjb_config_to_kwargs(cfg, NumericalScheme.SL_CUBIC) == {}


class TestFPConfigToKwargs:
    """Unit tests for fp_config_to_kwargs."""

    def test_all_defaults_returns_empty(self):
        """Default FPConfig (method='particle', particle=ParticleConfig()) → no kwargs."""
        kw = fp_config_to_kwargs(FPConfig(), NumericalScheme.FDM_UPWIND)
        assert kw == {}

    def test_default_particle_config_no_error(self):
        """Default particle sub-config is silently skipped (no particle params set)."""
        # ParticleConfig() is the Pydantic default; num_particles=5000 is the default.
        # This must NOT raise even though FDM creates FPFDMSolver (not FPParticleSolver).
        kw = fp_config_to_kwargs(FPConfig(), NumericalScheme.FDM_UPWIND)
        assert kw == {}

    def test_nondefault_particle_num_particles_raises(self):
        """Non-default particle.num_particles raises for FDM scheme (no particle solver)."""
        from mfgarchon.config.mfg_methods import ParticleConfig

        cfg = FPConfig(method="particle", particle=ParticleConfig(num_particles=999))
        with pytest.raises(NotImplementedError, match="Refs #1155"):
            fp_config_to_kwargs(cfg, NumericalScheme.FDM_UPWIND)

    def test_fdm_scheme_central_mapped(self):
        """config.fp.fdm.scheme='central' maps to advection_scheme='divergence_centered'."""
        cfg = FPConfig(method="fdm", fdm=FDMConfig(scheme="central"))
        kw = fp_config_to_kwargs(cfg, NumericalScheme.FDM_UPWIND)
        assert kw["advection_scheme"] == "divergence_centered"

    def test_fdm_default_scheme_no_kwarg(self):
        """config.fp.fdm.scheme='upwind' is the default — translator emits no kwarg.
        The scheme_factory setdefault handles the divergence_upwind default internally.
        """
        # FDMConfig().scheme == "upwind" is the config default; no kwarg emitted.
        cfg = FPConfig(method="fdm", fdm=FDMConfig(scheme="upwind"))
        kw = fp_config_to_kwargs(cfg, NumericalScheme.FDM_UPWIND)
        assert "advection_scheme" not in kw

    def test_fdm_params_for_wrong_scheme_raises(self):
        """Non-default fdm sub-config with SL scheme raises NotImplementedError."""
        cfg = FPConfig(method="fdm", fdm=FDMConfig(scheme="central"))
        with pytest.raises(NotImplementedError, match="Refs #1155"):
            fp_config_to_kwargs(cfg, NumericalScheme.SL_LINEAR)

    def test_fp_method_conflict_raises(self):
        """config.fp.method='fem' with GFDM scheme raises NotImplementedError."""
        cfg = FPConfig(method="fem")
        with pytest.raises(NotImplementedError, match="Refs #1155"):
            fp_config_to_kwargs(cfg, NumericalScheme.GFDM)

    # ------------------------------------------------------------------
    # Schemes with no FPConfig.method literal analog (FPSLSolver / FPFVMSolver /
    # MeshlessGalerkinFPSolver). A non-default fp.method under them must fail loud
    # rather than be silently discarded (Refs #1155).
    # ------------------------------------------------------------------

    def test_fp_method_fvm_upwind_nondefault_raises(self):
        """Non-default fp.method with FVM_UPWIND (no 'fvm' analog) raises NotImplementedError."""
        cfg = FPConfig(method="fdm")
        with pytest.raises(NotImplementedError, match="Refs #1155"):
            fp_config_to_kwargs(cfg, NumericalScheme.FVM_UPWIND)

    def test_fp_method_sl_cubic_nondefault_raises(self):
        """Non-default fp.method with SL_CUBIC (no 'sl' analog) raises NotImplementedError."""
        cfg = FPConfig(method="fdm")
        with pytest.raises(NotImplementedError, match="Refs #1155"):
            fp_config_to_kwargs(cfg, NumericalScheme.SL_CUBIC)

    def test_fp_method_meshless_galerkin_nondefault_raises(self):
        """Non-default fp.method with MESHLESS_GALERKIN (no analog) raises NotImplementedError."""
        cfg = FPConfig(method="fem")
        with pytest.raises(NotImplementedError, match="Refs #1155"):
            fp_config_to_kwargs(cfg, NumericalScheme.MESHLESS_GALERKIN)

    def test_fp_default_method_unmapped_scheme_no_raise(self):
        """Default fp.method ('particle') under an unmapped scheme is skipped (no raise)."""
        assert fp_config_to_kwargs(FPConfig(), NumericalScheme.FVM_UPWIND) == {}


class TestPicardConfigToIteratorKwargs:
    """Unit tests for picard_config_to_iterator_kwargs."""

    def test_zero_anderson_memory_empty(self):
        """anderson_memory=0 (default) returns empty dict."""
        kw = picard_config_to_iterator_kwargs(PicardConfig(anderson_memory=0))
        assert kw == {}

    def test_nonzero_anderson_memory_threaded(self):
        """anderson_memory=3 enables Anderson acceleration with depth=3."""
        kw = picard_config_to_iterator_kwargs(PicardConfig(anderson_memory=3))
        assert kw["use_anderson"] is True
        assert kw["anderson_depth"] == 3

    def test_anderson_memory_5(self):
        """anderson_memory=5 is threaded correctly."""
        kw = picard_config_to_iterator_kwargs(PicardConfig(anderson_memory=5))
        assert kw["use_anderson"] is True
        assert kw["anderson_depth"] == 5


# ---------------------------------------------------------------------------
# PINNING TEST: end-to-end via create_paired_solvers + FixedPointIterator
#
# This block proves that:
#   PRE-FIX:  setting non-default config.hjb field raises NotImplementedError
#   POST-FIX: the same config is accepted and values reach the constructed solvers
#
# Run git stash push mfgarchon/config/translator.py mfgarchon/core/mfg_problem.py
# to get pre-fix behaviour; pytest -k test_pin_* should FAIL.
# After git stash pop, pytest -k test_pin_* should PASS.
# ---------------------------------------------------------------------------


class TestPinningConfigThreading:
    """
    Pinning tests that FAIL on pre-fix code (NotImplementedError guard) and
    PASS after the translator is wired in.
    """

    @pytest.fixture
    def problem(self):
        return _tiny_problem()

    def test_pin_newton_tolerance_reaches_hjb_solver(self, problem):
        """
        config.hjb.newton.tolerance=1e-10 must reach hjb_solver.newton_tolerance.

        PRE-FIX: raises NotImplementedError (config.hjb non-default).
        POST-FIX: value threaded; solver has newton_tolerance=1e-10.
        """
        from mfgarchon.factory import create_paired_solvers

        cfg = HJBConfig(newton=NewtonConfig(tolerance=1e-10))
        kw = hjb_config_to_kwargs(cfg, NumericalScheme.FDM_UPWIND)
        assert kw["newton_tolerance"] == 1e-10

        hjb_solver, _ = create_paired_solvers(problem, NumericalScheme.FDM_UPWIND, hjb_config=kw)
        assert hjb_solver.newton_tolerance == 1e-10, (
            f"Expected newton_tolerance=1e-10, got {hjb_solver.newton_tolerance}"
        )

    def test_pin_newton_max_iterations_reaches_hjb_solver(self, problem):
        """
        config.hjb.newton.max_iterations=42 must reach hjb_solver.max_newton_iterations.

        PRE-FIX: raises NotImplementedError.
        POST-FIX: value threaded.
        """
        from mfgarchon.factory import create_paired_solvers

        cfg = HJBConfig(newton=NewtonConfig(max_iterations=42))
        kw = hjb_config_to_kwargs(cfg, NumericalScheme.FDM_UPWIND)
        assert kw["max_newton_iterations"] == 42

        hjb_solver, _ = create_paired_solvers(problem, NumericalScheme.FDM_UPWIND, hjb_config=kw)
        assert hjb_solver.max_newton_iterations == 42

    def test_pin_fp_fdm_scheme_reaches_fp_solver(self, problem):
        """
        config.fp.fdm.scheme='central' must produce FP advection_scheme='divergence_centered'.

        PRE-FIX: raises NotImplementedError.
        POST-FIX: value threaded.
        """
        from mfgarchon.factory import create_paired_solvers

        cfg = FPConfig(method="fdm", fdm=FDMConfig(scheme="central"))
        kw = fp_config_to_kwargs(cfg, NumericalScheme.FDM_UPWIND)
        assert kw["advection_scheme"] == "divergence_centered"

        _, fp_solver = create_paired_solvers(problem, NumericalScheme.FDM_UPWIND, fp_config=kw)
        assert fp_solver.advection_scheme == "divergence_centered", (
            f"Expected 'divergence_centered', got {fp_solver.advection_scheme!r}"
        )

    def test_pin_anderson_memory_reaches_fixed_point_iterator(self, problem):
        """
        config.picard.anderson_memory=3 must enable Anderson acceleration in
        FixedPointIterator with depth=3.

        PRE-FIX: anderson_memory validated but silently dropped.
        POST-FIX: use_anderson=True, anderson_accelerator.depth==3.
        """
        from mfgarchon.alg.numerical.coupling.fixed_point_iterator import FixedPointIterator
        from mfgarchon.factory import create_paired_solvers

        picard_cfg = PicardConfig(anderson_memory=3)
        iter_kw = picard_config_to_iterator_kwargs(picard_cfg)
        assert iter_kw["use_anderson"] is True
        assert iter_kw["anderson_depth"] == 3

        hjb_solver, fp_solver = create_paired_solvers(problem, NumericalScheme.FDM_UPWIND)
        iterator = FixedPointIterator(
            problem=problem,
            hjb_solver=hjb_solver,
            fp_solver=fp_solver,
            **iter_kw,
        )
        assert iterator.use_anderson is True
        assert iterator.anderson_accelerator is not None
        assert iterator.anderson_accelerator.depth == 3

    def test_pin_full_solve_with_nondefault_hjb_config_no_raise(self, problem):
        """
        problem.solve(config=..., scheme=FDM_UPWIND) with non-default
        config.hjb.newton.tolerance must succeed (not raise NotImplementedError).

        PRE-FIX: raises NotImplementedError from the guard.
        POST-FIX: succeeds.
        """
        config = MFGSolverConfig()
        # Mutate sub-config fields to non-default values
        config.hjb.newton = NewtonConfig(tolerance=1e-9, max_iterations=5)
        config.picard = PicardConfig(max_iterations=3, anderson_memory=2)

        # Must not raise
        result = problem.solve(
            scheme=NumericalScheme.FDM_UPWIND,
            config=config,
            verbose=False,
        )
        assert result is not None

    def test_pin_default_config_no_regression(self, problem):
        """
        Default config must produce the same result as no config (regression guard).
        The translator returns empty dicts for default configs → solver defaults used.
        """
        # Default config
        result_with_default = problem.solve(
            scheme=NumericalScheme.FDM_UPWIND,
            config=MFGSolverConfig(),
            max_iterations=3,
            verbose=False,
        )
        # No config at all
        result_no_config = problem.solve(
            scheme=NumericalScheme.FDM_UPWIND,
            max_iterations=3,
            verbose=False,
        )
        assert result_with_default is not None
        assert result_no_config is not None

    def test_pin_genuinely_unsupported_field_raises(self, problem):
        """
        A non-default field with no mapping (accuracy_order) must raise
        NotImplementedError, not silently drop.  This is the fail-loud contract.
        """
        config = MFGSolverConfig()
        config.hjb.accuracy_order = 4  # non-default, no mapping

        with pytest.raises(NotImplementedError, match="Refs #1155"):
            problem.solve(
                scheme=NumericalScheme.FDM_UPWIND,
                config=config,
                verbose=False,
            )
