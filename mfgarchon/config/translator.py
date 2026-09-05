"""
Translator: MFGSolverConfig fields -> solver constructor kwargs.

Maps validated config.hjb / config.fp / config.picard fields to the
corresponding HJB/FP/FixedPointIterator constructor kwargs (Issue #1155).

Rule
----
Only pass fields whose value differs from the Pydantic config default.
If equal to default: skip (let solver use its own default).
If non-default with a clear mapping: thread it.
If non-default with no mapping (unsupported or wrong scheme): raise
  NotImplementedError with "Refs #1155".

This design preserves backward compatibility: existing code that relies on
solver-level defaults is unaffected unless the user explicitly sets a
non-default config value.

BackendConfig.type threading
-----------------------------
config.backend.type is threaded to solvers that accept a ``backend`` kwarg
(HJBFDMSolver, FixedPointIterator).  BackendConfig.device and .precision are
not yet mapped; non-default values raise NotImplementedError (Refs #1155).

LoggingConfig
-------------
Non-default values raise NotImplementedError; they are validated by Pydantic
but have no mapping in the current solver infrastructure (Refs #1155).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mfgarchon.config.core import BackendConfig, LoggingConfig, PicardConfig
    from mfgarchon.config.mfg_methods import FPConfig, HJBConfig
    from mfgarchon.types import NumericalScheme


# ---------------------------------------------------------------------------
# HJB translator
# ---------------------------------------------------------------------------


def hjb_config_to_kwargs(
    hjb_cfg: HJBConfig,
    scheme: NumericalScheme,
) -> dict[str, Any]:
    """Map HJBConfig non-default fields to HJB solver constructor kwargs.

    Parameters
    ----------
    hjb_cfg : HJBConfig
        Validated HJB sub-configuration.
    scheme : NumericalScheme
        Active numerical scheme (determines which solver class is created).

    Returns
    -------
    dict
        Kwargs to pass to the HJB solver constructor (or hjb_config dict in
        create_paired_solvers).

    Raises
    ------
    NotImplementedError
        For non-default fields that cannot be applied to the selected scheme,
        or that have no current mapping.
    """
    from mfgarchon.types import NumericalScheme

    kwargs: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Newton sub-config: tolerance, max_iterations, relaxation
    # Applies to FDM, GFDM, SL (all Newton-based inner solvers).
    # ------------------------------------------------------------------
    newton = hjb_cfg.newton

    if "tolerance" in newton.model_fields_set:
        kwargs["newton_tolerance"] = newton.tolerance

    if "max_iterations" in newton.model_fields_set:
        kwargs["max_newton_iterations"] = newton.max_iterations

    if "relaxation" in newton.model_fields_set:
        # `relaxation` is a constructor kwarg for HJBFDMSolver; GFDM / SL
        # solvers do not expose it at construction time.
        if scheme in (NumericalScheme.FDM_UPWIND, NumericalScheme.FDM_CENTERED):
            kwargs["relaxation"] = newton.relaxation
        else:
            raise NotImplementedError(
                f"config.hjb.newton.relaxation is not mapped for "
                f"scheme={scheme.value} (only FDM HJB solvers accept it). "
                "Refs #1155."
            )

    # ------------------------------------------------------------------
    # Top-level fields: method, accuracy_order, boundary_conditions
    # ------------------------------------------------------------------
    if "method" in hjb_cfg.model_fields_set:
        # Each scheme maps to the HJBConfig.method literal whose solver it builds.
        # FVM_{UPWIND,MUSCL} pair the upwind HJB-FDM solver (method "fdm").
        # MESHLESS_GALERKIN has no HJBConfig.method literal analog and is absent here
        # on purpose -> handled by the unmapped-scheme fallback below.
        _scheme_method = {
            NumericalScheme.FDM_UPWIND: "fdm",
            NumericalScheme.FDM_CENTERED: "fdm",
            NumericalScheme.SL_LINEAR: "semi_lagrangian",
            NumericalScheme.SL_CUBIC: "semi_lagrangian",
            NumericalScheme.GFDM: "gfdm",
            NumericalScheme.FEM_P1: "fem",
            NumericalScheme.FEM_P2: "fem",
            NumericalScheme.FVM_UPWIND: "fdm",
            NumericalScheme.FVM_MUSCL: "fdm",
        }
        expected = _scheme_method.get(scheme)
        if expected is None:
            # No HJBConfig.method literal corresponds to this scheme, so a non-default
            # method cannot be reconciled with it. Fail loud rather than silently
            # discarding the user's intent (the scheme selects the solver class). Refs #1155.
            raise NotImplementedError(
                f"config.hjb.method={hjb_cfg.method!r} is non-default but "
                f"scheme={scheme.value} has no HJBConfig.method analog to validate "
                "against (the scheme selects the HJB solver class directly). Refs #1155."
            )
        if hjb_cfg.method != expected:
            raise NotImplementedError(
                f"config.hjb.method={hjb_cfg.method!r} conflicts with "
                f"scheme={scheme.value} (expected method {expected!r}). "
                "In Safe/Auto mode the scheme selects the solver class. Refs #1155."
            )

    if "accuracy_order" in hjb_cfg.model_fields_set:
        raise NotImplementedError(
            f"config.hjb.accuracy_order={hjb_cfg.accuracy_order} is not yet mapped to a solver kwarg. Refs #1155."
        )

    if "boundary_conditions" in hjb_cfg.model_fields_set:
        raise NotImplementedError(
            f"config.hjb.boundary_conditions={hjb_cfg.boundary_conditions!r} is "
            "not yet mapped to HJB solver kwargs. Refs #1155."
        )

    # ------------------------------------------------------------------
    # FDM sub-config
    # ------------------------------------------------------------------
    is_fdm_scheme = scheme in (NumericalScheme.FDM_UPWIND, NumericalScheme.FDM_CENTERED)
    if hjb_cfg.fdm is not None:
        if hjb_cfg.fdm.model_fields_set:
            if not is_fdm_scheme:
                raise NotImplementedError(
                    f"config.hjb.fdm has non-default values but "
                    f"scheme={scheme.value} does not use a FDM HJB solver. "
                    "Refs #1155."
                )
            _fdm_hjb_map = {"upwind": "gradient_upwind", "central": "gradient_centered"}
            if "scheme" in hjb_cfg.fdm.model_fields_set:
                if hjb_cfg.fdm.scheme in _fdm_hjb_map:
                    kwargs["advection_scheme"] = _fdm_hjb_map[hjb_cfg.fdm.scheme]
                else:
                    raise NotImplementedError(
                        f"config.hjb.fdm.scheme={hjb_cfg.fdm.scheme!r} has no "
                        "mapping to HJBFDMSolver advection_scheme. Refs #1155."
                    )
            if "time_stepping" in hjb_cfg.fdm.model_fields_set:
                raise NotImplementedError(
                    f"config.hjb.fdm.time_stepping={hjb_cfg.fdm.time_stepping!r} is not yet mapped. Refs #1155."
                )

    # ------------------------------------------------------------------
    # GFDM sub-config
    # ------------------------------------------------------------------
    if hjb_cfg.gfdm is not None:
        if hjb_cfg.gfdm.model_fields_set:
            if scheme != NumericalScheme.GFDM:
                raise NotImplementedError(
                    f"config.hjb.gfdm has non-default values but "
                    f"scheme={scheme.value} does not use a GFDM HJB solver. "
                    "Refs #1155."
                )
            _map_gfdm_to_hjb_kwargs(hjb_cfg.gfdm, kwargs)

    # ------------------------------------------------------------------
    # SL sub-config
    # ------------------------------------------------------------------
    is_sl_scheme = scheme in (NumericalScheme.SL_LINEAR, NumericalScheme.SL_CUBIC)
    if hjb_cfg.sl is not None:
        if hjb_cfg.sl.model_fields_set:
            if not is_sl_scheme:
                raise NotImplementedError(
                    f"config.hjb.sl has non-default values but "
                    f"scheme={scheme.value} does not use a SL HJB solver. "
                    "Refs #1155."
                )
            _map_sl_to_hjb_kwargs(hjb_cfg.sl, kwargs)

    # ------------------------------------------------------------------
    # WENO sub-config — not yet mapped
    # ------------------------------------------------------------------
    if hjb_cfg.weno is not None:
        if hjb_cfg.weno.model_fields_set:
            raise NotImplementedError("config.hjb.weno is not yet mapped to HJBWENOSolver kwargs. Refs #1155.")

    # ------------------------------------------------------------------
    # FEM sub-config — element order comes from scheme, not config
    # ------------------------------------------------------------------
    if hjb_cfg.fem is not None:
        if hjb_cfg.fem.model_fields_set:
            raise NotImplementedError(
                "config.hjb.fem sub-config is not yet mapped to HJBFEMSolver "
                "kwargs (element order is taken from the NumericalScheme). "
                "Refs #1155."
            )

    return kwargs


def _map_gfdm_to_hjb_kwargs(gfdm_cfg: Any, kwargs: dict[str, Any]) -> None:
    """Map GFDMConfig non-default fields to HJBGFDMSolver kwargs (in-place)."""

    if "delta" in gfdm_cfg.model_fields_set:
        kwargs["delta"] = gfdm_cfg.delta
    if "taylor_order" in gfdm_cfg.model_fields_set:
        kwargs["taylor_order"] = gfdm_cfg.taylor_order
    if "weight_function" in gfdm_cfg.model_fields_set:
        kwargs["weight_function"] = gfdm_cfg.weight_function
    if "weight_scale" in gfdm_cfg.model_fields_set:
        kwargs["weight_scale"] = gfdm_cfg.weight_scale
    if "congestion_mode" in gfdm_cfg.model_fields_set:
        kwargs["congestion_mode"] = gfdm_cfg.congestion_mode

    # QP sub-config
    _qp_map = {"none": "none", "auto": "qp_m_matrix", "always": "joint_socp"}
    if "optimization_level" in gfdm_cfg.qp.model_fields_set:
        kwargs["monotonicity_scheme"] = _qp_map[gfdm_cfg.qp.optimization_level]
    if "solver" in gfdm_cfg.qp.model_fields_set:
        kwargs["qp_solver"] = gfdm_cfg.qp.solver
    if "warm_start" in gfdm_cfg.qp.model_fields_set:
        kwargs["qp_warm_start"] = gfdm_cfg.qp.warm_start
    if "constraint_mode" in gfdm_cfg.qp.model_fields_set:
        kwargs["qp_constraint_mode"] = gfdm_cfg.qp.constraint_mode

    # Neighborhood sub-config
    if "mode" in gfdm_cfg.neighborhood.model_fields_set:
        kwargs["neighborhood_mode"] = gfdm_cfg.neighborhood.mode
    if "k_neighbors" in gfdm_cfg.neighborhood.model_fields_set:
        kwargs["k_neighbors"] = gfdm_cfg.neighborhood.k_neighbors
    if "adaptive" in gfdm_cfg.neighborhood.model_fields_set:
        kwargs["adaptive_neighborhoods"] = gfdm_cfg.neighborhood.adaptive
    if "k_min" in gfdm_cfg.neighborhood.model_fields_set:
        kwargs["k_min"] = gfdm_cfg.neighborhood.k_min
    if "max_delta_multiplier" in gfdm_cfg.neighborhood.model_fields_set:
        kwargs["max_delta_multiplier"] = gfdm_cfg.neighborhood.max_delta_multiplier

    # Derivative sub-config
    if "method" in gfdm_cfg.derivative.model_fields_set:
        kwargs["derivative_method"] = gfdm_cfg.derivative.method
    if "rbf_kernel" in gfdm_cfg.derivative.model_fields_set:
        kwargs["rbf_kernel"] = gfdm_cfg.derivative.rbf_kernel
    if "rbf_poly_degree" in gfdm_cfg.derivative.model_fields_set:
        kwargs["rbf_poly_degree"] = gfdm_cfg.derivative.rbf_poly_degree

    # Boundary accuracy sub-config
    if "local_coordinate_rotation" in gfdm_cfg.boundary_accuracy.model_fields_set:
        kwargs["use_local_coordinate_rotation"] = gfdm_cfg.boundary_accuracy.local_coordinate_rotation
    if "ghost_nodes" in gfdm_cfg.boundary_accuracy.model_fields_set:
        kwargs["use_ghost_nodes"] = gfdm_cfg.boundary_accuracy.ghost_nodes
    if "wind_dependent_bc" in gfdm_cfg.boundary_accuracy.model_fields_set:
        kwargs["use_wind_dependent_bc"] = gfdm_cfg.boundary_accuracy.wind_dependent_bc


def _map_sl_to_hjb_kwargs(sl_cfg: Any, kwargs: dict[str, Any]) -> None:
    """Map SLConfig non-default fields to HJBSemiLagrangianSolver kwargs (in-place)."""

    _rk_map: dict[int, str] = {1: "explicit_euler", 2: "rk2", 4: "rk4"}

    if "interpolation_method" in sl_cfg.model_fields_set:
        kwargs["interpolation_method"] = sl_cfg.interpolation_method

    if "rk_order" in sl_cfg.model_fields_set:
        if sl_cfg.rk_order in _rk_map:
            kwargs["characteristic_solver"] = _rk_map[sl_cfg.rk_order]
        else:
            raise NotImplementedError(
                f"config.hjb.sl.rk_order={sl_cfg.rk_order} is not in supported "
                f"values {list(_rk_map.keys())}. Refs #1155."
            )

    if "cfl_number" in sl_cfg.model_fields_set:
        raise NotImplementedError(
            f"config.hjb.sl.cfl_number={sl_cfg.cfl_number} is not yet mapped to "
            "HJBSemiLagrangianSolver params. Refs #1155."
        )


# ---------------------------------------------------------------------------
# FP translator
# ---------------------------------------------------------------------------


def fp_config_to_kwargs(
    fp_cfg: FPConfig,
    scheme: NumericalScheme,
) -> dict[str, Any]:
    """Map FPConfig non-default fields to FP solver constructor kwargs.

    Parameters
    ----------
    fp_cfg : FPConfig
        Validated FP sub-configuration.
    scheme : NumericalScheme
        Active numerical scheme.

    Returns
    -------
    dict
        Kwargs to pass to the FP solver constructor (or fp_config dict in
        create_paired_solvers).

    Raises
    ------
    NotImplementedError
        For non-default fields that cannot be applied to the selected scheme.
    """
    from mfgarchon.types import NumericalScheme

    kwargs: dict[str, Any] = {}
    is_fdm_scheme = scheme in (NumericalScheme.FDM_UPWIND, NumericalScheme.FDM_CENTERED)

    # ------------------------------------------------------------------
    # method field: validate scheme consistency
    # ------------------------------------------------------------------
    # Each scheme maps to the FPConfig.method literal whose FP solver it builds.
    # SL_{LINEAR,CUBIC} (FPSLSolver), FVM_{UPWIND,MUSCL} (FPFVMSolver) and
    # MESHLESS_GALERKIN (MeshlessGalerkinFPSolver) have NO FPConfig.method literal
    # analog (no "sl"/"fvm"/"meshless" member), so they are absent here on purpose ->
    # a non-default fp.method under them fails loud via the unmapped-scheme fallback.
    _scheme_fp_method: dict[NumericalScheme, str] = {
        NumericalScheme.FDM_UPWIND: "fdm",
        NumericalScheme.FDM_CENTERED: "fdm",
        NumericalScheme.GFDM: "gfdm",
        NumericalScheme.FEM_P1: "fem",
        NumericalScheme.FEM_P2: "fem",
    }
    if "method" in fp_cfg.model_fields_set:
        expected = _scheme_fp_method.get(scheme)
        if expected is None:
            # No FPConfig.method literal corresponds to this scheme, so a non-default
            # method cannot be reconciled with it. Fail loud rather than silently
            # discarding the user's intent (the scheme selects the FP solver class). Refs #1155.
            raise NotImplementedError(
                f"config.fp.method={fp_cfg.method!r} is non-default but "
                f"scheme={scheme.value} has no FPConfig.method analog to validate "
                "against (the scheme selects the FP solver class directly). Refs #1155."
            )
        if fp_cfg.method != expected:
            raise NotImplementedError(
                f"config.fp.method={fp_cfg.method!r} conflicts with "
                f"scheme={scheme.value} (expected method {expected!r}). "
                "In Safe/Auto mode the scheme selects the FP solver class. Refs #1155."
            )

    # ------------------------------------------------------------------
    # FDM sub-config
    # ------------------------------------------------------------------
    if fp_cfg.fdm is not None:
        if fp_cfg.fdm.model_fields_set:
            if not is_fdm_scheme:
                raise NotImplementedError(
                    f"config.fp.fdm has non-default values but "
                    f"scheme={scheme.value} does not use a FDM FP solver. "
                    "Refs #1155."
                )
            # FP needs divergence (conservative) form; HJB uses gradient form.
            _fdm_fp_map = {
                "upwind": "divergence_upwind",
                "central": "divergence_centered",
            }
            if "scheme" in fp_cfg.fdm.model_fields_set:
                if fp_cfg.fdm.scheme in _fdm_fp_map:
                    kwargs["advection_scheme"] = _fdm_fp_map[fp_cfg.fdm.scheme]
                else:
                    raise NotImplementedError(
                        f"config.fp.fdm.scheme={fp_cfg.fdm.scheme!r} has no "
                        "mapping to FPFDMSolver advection_scheme. Refs #1155."
                    )
            if "time_stepping" in fp_cfg.fdm.model_fields_set:
                raise NotImplementedError(
                    f"config.fp.fdm.time_stepping={fp_cfg.fdm.time_stepping!r} is not yet mapped. Refs #1155."
                )

    # ------------------------------------------------------------------
    # Particle sub-config: no current Safe/Auto scheme creates FPParticleSolver
    # ------------------------------------------------------------------
    if fp_cfg.particle is not None:
        if fp_cfg.particle.model_fields_set:
            raise NotImplementedError(
                "config.fp.particle has non-default values but no Safe/Auto mode "
                "scheme creates FPParticleSolver. Use Expert Mode "
                "(solve(hjb_solver=..., fp_solver=FPParticleSolver(...))) to "
                "configure particle-based FP solving. Refs #1155."
            )

    # ------------------------------------------------------------------
    # GFDM sub-config (FPConfig has no gfdm field; skip if absent)
    # ------------------------------------------------------------------
    # Note: FPConfig does not include a gfdm sub-config — GFDM FP params
    # are threaded via the pair factory's delta/collocation_points syncing.
    # This gap is tracked in Refs #1155.

    # ------------------------------------------------------------------
    # Network sub-config — not yet mapped
    # ------------------------------------------------------------------
    if fp_cfg.network is not None:
        if fp_cfg.network.model_fields_set:
            raise NotImplementedError("config.fp.network is not yet mapped. Refs #1155.")

    # ------------------------------------------------------------------
    # FEM sub-config — not yet mapped
    # ------------------------------------------------------------------
    if fp_cfg.fem is not None:
        if fp_cfg.fem.model_fields_set:
            raise NotImplementedError("config.fp.fem sub-config is not yet mapped to FPFEMSolver kwargs. Refs #1155.")

    return kwargs


# ---------------------------------------------------------------------------
# Picard (FixedPointIterator) extras
# ---------------------------------------------------------------------------


def picard_config_to_iterator_kwargs(picard_cfg: PicardConfig) -> dict[str, Any]:
    """Map PicardConfig.anderson_memory to FixedPointIterator constructor kwargs.

    Returns an empty dict when anderson_memory==0 (disabled), so callers can
    safely unpack the result without conditionals.

    Parameters
    ----------
    picard_cfg : PicardConfig
        Validated Picard iteration configuration.

    Returns
    -------
    dict
        ``use_anderson`` and ``anderson_depth`` entries (if anderson_memory > 0).
    """
    if picard_cfg.anderson_memory > 0:
        return {
            "use_anderson": True,
            "anderson_depth": picard_cfg.anderson_memory,
        }
    return {}


# ---------------------------------------------------------------------------
# Backend / Logging extras
# ---------------------------------------------------------------------------


def backend_config_to_kwargs(backend_cfg: BackendConfig) -> dict[str, Any]:
    """Map BackendConfig to solver constructor kwargs.

    BackendConfig.type is threaded to solvers that accept a ``backend`` kwarg.
    BackendConfig.device and .precision are not yet mapped; non-default values
    raise NotImplementedError (Refs #1155).

    Parameters
    ----------
    backend_cfg : BackendConfig
        Validated backend configuration.

    Returns
    -------
    dict
        ``backend`` entry if type is non-default; otherwise empty.
    """

    kwargs: dict[str, Any] = {}

    if "type" in backend_cfg.model_fields_set:
        kwargs["backend"] = backend_cfg.type

    if "device" in backend_cfg.model_fields_set:
        raise NotImplementedError(
            f"config.backend.device={backend_cfg.device!r} is not yet threaded to solvers. Refs #1155."
        )

    if "precision" in backend_cfg.model_fields_set:
        raise NotImplementedError(
            f"config.backend.precision={backend_cfg.precision!r} is not yet threaded to solvers. Refs #1155."
        )

    return kwargs


def check_logging_config(logging_cfg: LoggingConfig) -> None:
    """Raise NotImplementedError for non-default LoggingConfig fields.

    LoggingConfig is validated by Pydantic but its fields have no mapping in
    the current solver infrastructure (Refs #1155).  Calling this function
    ensures users get a clear error rather than silent discard.
    """

    non_default = [
        f"{name}={getattr(logging_cfg, name)!r}"
        for name in ("level", "progress_bar", "save_intermediate", "output_dir")
        if name in logging_cfg.model_fields_set
    ]
    if non_default:
        raise NotImplementedError(
            f"config.logging has non-default values ({', '.join(non_default)}) "
            "that are not yet threaded to solvers. Refs #1155."
        )


# ---------------------------------------------------------------------------
# Convenience: translate a full MFGSolverConfig
# ---------------------------------------------------------------------------


def translate_solver_config(
    config: Any,
    scheme: NumericalScheme,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Translate a full MFGSolverConfig into solver constructor kwarg dicts.

    Parameters
    ----------
    config : MFGSolverConfig
        Validated solver configuration.
    scheme : NumericalScheme
        Active numerical scheme (determines HJB/FP solver classes).

    Returns
    -------
    hjb_kwargs : dict
        Kwargs for HJB solver constructor.
    fp_kwargs : dict
        Kwargs for FP solver constructor.
    iterator_kwargs : dict
        Extra kwargs for FixedPointIterator (anderson acceleration).
    backend_kwargs : dict
        Backend kwarg for both HJB solver and FixedPointIterator.
    """
    hjb_kwargs = hjb_config_to_kwargs(config.hjb, scheme)
    fp_kwargs = fp_config_to_kwargs(config.fp, scheme)
    iterator_kwargs = picard_config_to_iterator_kwargs(config.picard)
    backend_kwargs = backend_config_to_kwargs(config.backend)
    check_logging_config(config.logging)
    return hjb_kwargs, fp_kwargs, iterator_kwargs, backend_kwargs
