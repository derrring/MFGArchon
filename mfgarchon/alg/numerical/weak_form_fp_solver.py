"""
Backend-agnostic weak-form Fokker-Planck solver.

Solves the FP equation forward in time on the assembled weak-form operators of a
``WeakFormDiscretization`` (stiffness ``K``, mass ``M``), so one solver serves
finite elements and meshless Galerkin. A concrete subclass supplies the
discretization, the boundary-condition strategy, and how the advection matrix is
built from the drift (the one piece that is genuinely backend-specific: FEM uses
the exact quadrature-point gradient of ``U``; meshless uses the protocol's
``advection`` with a recovered nodal gradient).

Time discretization: implicit Euler (forward), its time derivative on the discretization's
``time_mass``: row-sum lumped for P1 FEM, the consistent ``M`` otherwise (#2845). Mass
conservation follows from the Galerkin weak form (the test space contains constants); the
lumped mass keeps ``M``'s row sums, so it holds either way.

Issue #1131 Phase 2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.sparse.linalg import spsolve

from mfgarchon.alg.numerical.fp_solvers.base_fp import BaseFPSolver, DriftConvention
from mfgarchon.alg.numerical.weak_form_discretization import time_mass
from mfgarchon.types.callable_protocols import evaluate_solver_source
from mfgarchon.utils.deprecation import deprecated_parameter
from mfgarchon.utils.mfg_logging import get_logger
from mfgarchon.utils.numerical.mass_fabrication_gate import (
    MAX_CLIP_MASS_FABRICATION,
    negative_density_report,
    refuse_what_a_flux_takes_negative,
)
from mfgarchon.utils.pde_coefficients import (
    diffusion_from_volatility,
    resolve_volatility_override,
    retired_sigma_keyword,
    retired_volatility_keywords,
    scalar_volatility,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray
    from scipy import sparse

    from mfgarchon.alg.numerical.weak_form_discretization import WeakFormDiscretization
    from mfgarchon.core.mfg_problem import MFGProblem
    from mfgarchon.geometry.boundary import BoundaryConditions

logger = get_logger(__name__)


class WeakFormFPSolver(BaseFPSolver):
    """FP solver on assembled weak-form operators; BC + advection are subclass hooks."""

    # Issue #1043: this family takes the value function U and recovers α = -coupling·∇U on its
    # own quadrature/MLS basis (a genuine feature: avoids differentiating U on a coarse FP grid).
    _drift_convention = DriftConvention.VALUE_FUNCTION
    reports_negative_density = True

    #: What the warning says about a negative on this solver: what has cleared one, and on what. Each subclass states
    #: its own runs (#2845).
    _negative_density_note: str

    def __init__(
        self,
        problem: MFGProblem,
        discretization: WeakFormDiscretization,
        boundary_conditions: BoundaryConditions | None = None,
    ) -> None:
        super().__init__(problem)
        # An explicit BC is the FP's own and is used as given, a NEUMANN refused (#2532); without one the
        # solver reads the problem's shared BC, translated (`BaseFPSolver._lookup_boundary_conditions`).
        self._boundary_conditions = boundary_conditions
        self._disc = discretization
        self._n_dof = discretization.n_dof
        # A discretization may keep the caller's array (#2779), so the points it was built on are copied here.
        self._dof_points = np.array(discretization.dof_coordinates, dtype=float)
        self._dof_points.flags.writeable = False
        self._K = discretization.stiffness()
        self._M = discretization.mass()
        # The time derivative's mass, lumped for P1 FEM (#2845); the source, H and the mass a solve measures keep M.
        self._M_time = time_mass(discretization, self._M)
        self._mass_weights = np.asarray(self._M.sum(axis=1)).ravel()
        # Single source of truth for BCs (matches WeakFormHJBSolver); plain
        # getattr(geometry, "boundary_conditions") misses grids that expose BCs via
        # the accessor method (e.g. TensorProductGrid), silently dropping Dirichlet.
        # The shared BC is read through BaseFPSolver: a DIRICHLET(g) is an exit, absorbing here, a NEUMANN(0)
        # is zero flux, and a nonzero NEUMANN or any ROBIN is refused (#2512, row B3). Condensed literally, a
        # shared Dirichlet pinned the FEM exit wall at m = g (#2525); an explicit DIRICHLET(g) is that density.
        self._bc = self.get_boundary_conditions()
        # A FLUX needs the load `_flux_load` builds, which only FP-FEM's override does. This base declares no
        # ``_SUPPORTED_BC_TYPES``, so its gate passes a FLUX; a class that inherits the base's load refuses it here,
        # rather than solve it as a no-flux wall (#2581).
        if self._declares_flux() and type(self)._flux_load is WeakFormFPSolver._flux_load:
            raise NotImplementedError(
                f"{type(self).__name__} builds no prescribed-flux load, so a FLUX boundary would be solved as a no-flux "
                "wall (#2581). FPFEMSolver honours FLUX."
            )
        # The step keeps a negative value, and this says how large it is (#2845): "warn" reports each solve, and a coupler
        # sets "record" while it runs, then reports the density it returns.
        self.on_negative_density = "warn"
        self.last_negative_density: dict[str, Any] | None = None

    @property
    def n_dof(self) -> int:
        return self._n_dof

    @property
    def dof_points(self) -> NDArray:
        """The points this solver's DOFs sat at when it was built, in their order: an (N, d) read-only copy taken
        then. A coupling loop compares them with the points of the problem it solves (`assert_paired_solver_geometry`,
        #2775)."""
        return self._dof_points

    # --- Boundary-condition strategy (subclass-supplied) ----------------------
    def _is_pure_neumann(self) -> bool:
        raise NotImplementedError

    def _imposes_dirichlet(self) -> bool:
        """Does the BC impose a Dirichlet wall anywhere (by condensation, or by Nitsche terms)?"""
        raise NotImplementedError

    def _dirichlet_dofs_and_values(self, t: float | None = None) -> tuple[NDArray, NDArray]:
        """The Dirichlet dofs and their values, read at ``t``, the time of the level solved for (#2512 row B2)."""
        raise NotImplementedError

    def _apply_bc_to_system(
        self, matrix: sparse.csr_matrix, rhs: NDArray, t: float | None = None
    ) -> tuple[sparse.csr_matrix, NDArray]:
        """Condense the linear system onto interior dofs for Dirichlet BC, its values read at ``t``."""
        raise NotImplementedError

    def _weak_bc_terms(self, D: float):
        """Optional weak (Nitsche) boundary terms for non-interpolatory bases.

        Returns ``(A_extra, rhs_extra)`` to ADD to the full-size operator and RHS,
        solved on ALL dofs, or ``(None, None)`` for no weak BC (default no-op; FEM
        uses condensation). Meshless Galerkin overrides this with the symmetric
        Nitsche block for absorbing (``m = 0``) boundaries -- the SAME block as the
        HJB solver, so the Type-A transpose identity ``A_FP = A_HJB^T`` is preserved.
        ``rhs_extra`` is ``None`` for the homogeneous absorbing case."""
        return None, None

    def _robin_operator_terms(self, D: float):
        """Optional Robin boundary operator augmentation, the FP's total-flux Robin ``J.n = (D/beta)(alpha*m - g)``.

        Returns ``(A_robin, rhs_robin)`` to ADD to the spatial operator ``M_t/dt + D*K`` (M_t the time mass, #2845) and each
        timestep RHS, or ``(None, None)`` for no Robin BC. The term is the boundary mass
        ``D*(alpha/beta)*int_dOmega phi_i phi_j``, lumped where the discretization lumps its mass (P1 FEM, #2845), and the
        load ``D*(g/beta)*int_dOmega phi_i``. The
        weak form integrates the whole flux by parts and leaves ``J.n`` on the boundary, so this
        imposes ``alpha*m - (beta/D)*J.n = g``, which is not ``alpha*m + beta*dm/dn = g`` where the
        drift crosses the wall (#2512, user ruling 2026-10-10). The boundary mass is symmetric, the
        same block as the HJB's Robin term, so ``A_FP = A_HJB^T`` is preserved for the
        diffusion+Robin block. Only an explicit ROBIN reaches it; a shared one is refused. Default
        no-op; only the FEM solver overrides this."""
        return None, None

    # --- Advection from drift (subclass-supplied) -----------------------------
    def _flux_load(self, t: float | None) -> NDArray | None:
        """The load of a prescribed flux (FLUX, #2581) at the level ``t``, or None. Only FP-FEM's override builds one;
        a class that inherits this one refuses a FLUX at construction."""
        return None

    def _declares_flux(self) -> bool:
        """Whether the BC holds a FLUX segment or ``default_bc`` (#2581), read from the types alone, with no value read
        and so no time."""
        from mfgarchon.geometry.boundary.types import BCType

        bc = self._bc
        if bc is None:
            return False
        return any(segment.bc_type == BCType.FLUX for segment in getattr(bc, "segments", None) or []) or (
            getattr(bc, "default_bc", None) == BCType.FLUX
        )

    def _build_advection(self, U_n: NDArray, D: float) -> sparse.csr_matrix:
        r"""Advection matrix for drift $v = -\text{coupling}\cdot\nabla U_n$ in divergence form.

        ``D`` is the diffusion coefficient of the CURRENT solve (``_diffusion_coefficient``,
        volatility-aware), passed so a subclass that adds a diffusion-scaled stabilization
        term (e.g. meshless streamline diffusion) uses the same ``D`` as the stiffness block."""
        raise NotImplementedError

    def _diffusion_coefficient(self, volatility) -> float:
        # D = sigma^2 / 2 via the single-source converter (Issue #811). This solver assembles one
        # scalar D, so an array or callable volatility is refused, not averaged (#2376).
        volatility = volatility if volatility is not None else self.problem.volatility
        return diffusion_from_volatility(scalar_volatility(volatility, consumer=type(self).__name__))

    @retired_volatility_keywords
    @deprecated_parameter(param_name="drift_field", since="v0.20.0", replacement="potential_field")
    @deprecated_parameter(param_name="m_initial", since="v0.22.0", replacement="M_initial")
    def solve_fp_system(
        self,
        M_initial: NDArray | None = None,
        potential_field: NDArray | None = None,
        volatility: float | NDArray | None = None,
        drift_field: NDArray | None = None,  # DEPRECATED alias for potential_field (Issue #1043)
        source_term: Callable | None = None,
        m_initial: NDArray | None = None,  # DEPRECATED alias for M_initial, the base's name (#2377)
        volatility_kind: str | None = None,
        show_progress: bool | None = None,
    ) -> NDArray:
        """Solve the FP equation forward in time on the weak-form operators.

        ``potential_field`` is the value function ``U(t,x)`` (Issue #1043); this solver recovers
        the advective velocity ``α = -coupling·∇U`` on its own quadrature/MLS basis. It was
        historically -- and misleadingly -- named ``drift_field``, but on this solver that input
        always meant ``U``, never the velocity; ``drift_field`` is kept as a deprecated alias.

        ``show_progress`` is the base interface's display switch; this solver shows no progress. It does
        not take the base's ``progress_callback``, which it would never call. A keyword this signature does
        not name raises (#2419).
        """
        # Issue #1043: drift_field is the deprecated name for the U (potential) input here.
        if drift_field is not None:
            if potential_field is not None:
                raise ValueError("Pass only potential_field; drift_field is its deprecated alias (Issue #1043).")
            potential_field = drift_field
        if m_initial is not None:
            if M_initial is not None:
                raise ValueError("Pass only M_initial; m_initial is its deprecated alias (#2377).")
            M_initial = m_initial
        if M_initial is None:
            raise ValueError("M_initial is required")

        Nt = self.problem.Nt
        dt = self.problem.dt
        N = self._n_dof

        volatility, _ = resolve_volatility_override(
            volatility, volatility_kind, problem=self.problem, consumer=f"{type(self).__name__}.solve_fp_system"
        )
        D = self._diffusion_coefficient(volatility)

        M = np.zeros((Nt + 1, N))
        # Issue #1489 (S4): the initial density must live on ALL N solver DOFs. The former length-only
        # reconciliation (truncate if longer, zero-pad if shorter) silently produced a WRONG IC: for P2
        # the caller resolves M_initial on num_vertices (< n_dof = vertices + edges), so np.pad zero-
        # filled every edge-midpoint DOF. Fail loud instead of silently mis-placing the density. (A
        # same-length-but-reordered IC still cannot be detected here — evaluate M_initial on the
        # solver's dof_coordinates, self._disc.dof_coordinates, upstream.)
        if len(M_initial) != N:
            raise ValueError(
                f"Initial density has {len(M_initial)} entries but the discretization has {N} DOFs. The "
                f"weak-form FP solver needs M_initial on all {N} DOFs (basis.doflocs / "
                f"self._disc.dof_coordinates), not a subset — this is the P2-vs-P1 DOF-count mismatch "
                f"(P2 adds edge/face DOFs beyond the vertices). Silently padding/truncating would zero-fill "
                f"the missing DOFs and mis-place the density (Issue #1489). Evaluate M_initial on the "
                f"solver's dof_coordinates."
            )
        M[0] = M_initial

        A_base = self._M_time / dt + D * self._K
        A_extra, rhs_extra = self._weak_bc_terms(D)
        weak_bc = A_extra is not None
        if weak_bc:
            A_base = A_base + A_extra
        # Robin operator augmentation (Issue #1237): adjoint of the HJB Robin term (symmetric
        # boundary mass, so A_FP = A_HJB^T is preserved). No-op for non-Robin problems.
        A_robin, rhs_robin = self._robin_operator_terms(D)
        if A_robin is not None:
            A_base = A_base + A_robin
        pure_neumann = self._is_pure_neumann()

        for n in range(Nt):
            rhs = (self._M_time / dt) @ M[n]
            if source_term is not None:
                # S of `d_t m + div(alpha m) - D Lap(m) = S`, evaluated implicitly at t_{n+1} to
                # match FPFDMSolver, on the solver's own (N, d) dof coordinates (#2020).
                s_vals = np.asarray(
                    evaluate_solver_source(source_term, t=(n + 1) * dt, x=self._disc.dof_coordinates), dtype=float
                ).ravel()
                if s_vals.shape != (N,):
                    raise ValueError(
                        f"source_term returned shape {s_vals.shape}; this solver needs ({N},), one "
                        f"value per dof, evaluated on self._disc.dof_coordinates (shape "
                        f"{self._disc.dof_coordinates.shape}). A grid-shaped return is the "
                        f"FPFVMSolver convention, not this one (#2019)."
                    )
                rhs = rhs + self._M @ s_vals
            if rhs_robin is not None:
                rhs = rhs + rhs_robin

            if potential_field is not None:
                U_n = potential_field[n] if potential_field.ndim > 1 else potential_field
                A_system = A_base + self._build_advection(U_n, D)
            else:
                A_system = A_base

            # A prescribed flux (#2581) enters by linearity, m = m0 + m_q: m0 is the step as it was, and m_q the
            # flux's load alone on the same matrix, with the Dirichlet values homogeneous; none when every q is zero.
            flux_load = self._flux_load((n + 1) * dt)
            if flux_load is not None and not np.any(flux_load):
                flux_load = None
            m_flux = np.zeros(N) if flux_load is not None else None
            if weak_bc:
                if rhs_extra is not None:
                    rhs = rhs + rhs_extra
                M[n + 1] = spsolve(A_system, rhs)
                if flux_load is not None:
                    m_flux = spsolve(A_system, flux_load)
            elif pure_neumann:
                M[n + 1] = spsolve(A_system, rhs)
                if flux_load is not None:
                    m_flux = spsolve(A_system, flux_load)
            else:
                A_bc, rhs_bc = self._apply_bc_to_system(A_system, rhs, t=(n + 1) * dt)
                d_dofs, d_vals = self._dirichlet_dofs_and_values((n + 1) * dt)
                interior = np.setdiff1d(np.arange(N), d_dofs)
                M[n + 1, interior] = spsolve(A_bc, rhs_bc)
                M[n + 1, d_dofs] = d_vals
                if flux_load is not None:
                    m_flux[interior] = spsolve(A_bc, flux_load[interior])
            if m_flux is not None:
                # B-3's owner; otherwise a step whose own negatives are past the threshold is kept, with the flux's
                # share, as any step, and reported (#2581, #2845).
                M[n + 1] = refuse_what_a_flux_takes_negative(
                    M[n + 1],
                    m_flux,
                    context=type(self).__name__,
                    time=(n + 1) * dt,
                    unit="DOF",
                    locate=lambda i: f"DOF {i}",
                    weights=self._mass_weights,
                    remedy=(
                        "The Galerkin step is not monotone, so the flux's part of it can be negative away from the "
                        "wall, an inflow's included."
                    ),
                    remedy_past_gate=(
                        "A q on this step with which the flux takes no DOF negative past the threshold leaves the step "
                        "as FP-FEM's own, whose negative values are kept and reported as any step's."
                    ),
                )

        self._report_negative_density(M)
        return M

    def negative_density_of(self, M: NDArray) -> dict[str, Any]:
        """The negative part of ``M``, one row per time level, on this solver's mass (#2845)."""
        return negative_density_report(np.asarray(M, dtype=float).reshape(-1, self._n_dof), weights=self._mass_weights)

    def warn_negative_density(self, report: dict[str, Any]) -> None:
        """One warning for ``report``, past the gate's threshold of a level's mass (#2845)."""
        if report["largest"] > MAX_CLIP_MASS_FABRICATION:
            logger.warning(
                "%s: %d of %d time levels hold a negative value, down to %.3e at level %d, at most %.3e of a level's "
                "mass (sum over levels %.3e). The values are kept: nothing is added to the mass. %s",
                type(self).__name__,
                report["levels_negative"],
                report["levels"],
                report["lowest"],
                report["lowest_level"],
                report["largest"],
                report["cumulative"],
                self._negative_density_note,
            )

    def _report_negative_density(self, M: NDArray) -> None:
        self.last_negative_density = self.negative_density_of(M)
        if self.on_negative_density == "warn":
            self.warn_negative_density(self.last_negative_density)

    @retired_sigma_keyword
    def solve_fp_step_adjoint_mode(
        self,
        M_current: NDArray,
        A_advection_T: sparse.csr_matrix,
        volatility: float | NDArray | None = None,
        time: float = 0.0,
        volatility_kind: str | None = None,
    ) -> NDArray:
        """Single FP timestep with an externally provided (transposed) advection matrix.

        Used by BlockIterator's adjoint_mode="jacobian_transpose": the FP operator is supplied directly,
        as the transpose of the HJB's linearised operator. ``volatility`` is read by its declared
        ``volatility_kind`` (#2378); this solver assembles one scalar D, so an array is refused.
        """
        volatility, _ = resolve_volatility_override(
            volatility,
            volatility_kind,
            problem=self.problem,
            consumer=f"{type(self).__name__}.solve_fp_step_adjoint_mode",
        )
        dt = self.problem.dt
        D = self._diffusion_coefficient(volatility)
        # The forward step imposes a Dirichlet wall by Nitsche terms (meshless Galerkin) or by condensing the DOFs the
        # owner gives a DIRICHLET segment (FEM, `bc_adapter.has_dirichlet_dofs`). This step does neither, so it would
        # drop the wall in silence; refused, as FP-FDM's is (#2531, #2608). The test reads no boundary value; `time`
        # is passed to nothing.
        if self._declares_flux():
            raise NotImplementedError(
                f"{type(self).__name__}.solve_fp_step_adjoint_mode builds no prescribed-flux load, so a FLUX facet would "
                "be solved as a no-flux wall (#2581). With BlockIterator, adjoint_mode='off' solves the FP side through "
                "solve_fp_system, which does."
            )
        if self._imposes_dirichlet():
            raise NotImplementedError(
                f"{type(self).__name__}.solve_fp_step_adjoint_mode imposes no Dirichlet wall, so an absorbing exit "
                "or a prescribed density would not hold its value (#2608, as #2531 for FP-FDM). With BlockIterator, "
                "adjoint_mode='off' solves the FP side through solve_fp_system, which does."
            )

        A_system = self._M_time / dt + A_advection_T + D * self._K
        rhs = (self._M_time / dt) @ M_current.ravel()
        # Robin operator augmentation (Issue #1237): same symmetric boundary mass + load as the
        # forward FP path, so the adjoint timestep carries the Robin BC consistently. No-op otherwise.
        A_robin, rhs_robin = self._robin_operator_terms(D)
        if A_robin is not None:
            A_system = A_system + A_robin
        if rhs_robin is not None:
            rhs = rhs + rhs_robin
        M_next = spsolve(A_system, rhs)
        self._report_negative_density(M_next)
        return M_next.reshape(M_current.shape)
