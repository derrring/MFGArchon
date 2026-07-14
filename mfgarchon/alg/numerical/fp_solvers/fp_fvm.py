"""Conservative Finite Volume Method (FVM) solver for the Fokker-Planck equation (Issue #422).

The FVM solver evolves *cell averages* ``m_bar_i`` of the density on a structured
(tensor-product) grid, whose nodes are interpreted as cell centers with uniform spacing
``dx`` (so ``m_bar_i`` approximates the point value ``m(x_i)`` to ``O(dx^2)``). The
semi-discrete update is the flux-difference form

.. math::

    \\frac{d \\bar m_i}{dt} = -\\frac{F_{i+1/2} - F_{i-1/2}}{\\Delta x},
    \\qquad
    F_{i+1/2} = \\alpha_{i+1/2}\\, m_{i+1/2} - D\\,\\frac{m_{i+1} - m_i}{\\Delta x}.

The interface velocity ``alpha_{i+1/2}`` is *shared* by the two cells that touch the face, so
the divergence telescopes and the total mass ``sum_i m_bar_i dx`` is conserved to machine
precision for no-flux / periodic boundaries. This is the higher-order extension of the
conservative divergence-upwind FDM stencil
(:mod:`fp_fdm_alg_divergence_upwind`); see Issue #422.

Reconstruction (``reconstruction`` ctor arg):

- ``"upwind"`` -- 1st-order upwind face value ``m_{i+1/2}`` (robust, ``O(dx)``).
- ``"muscl"`` [default] -- 2nd-order MUSCL with a ``minmod`` slope limiter
  (TVD -> positivity, ``O(dx^2)`` in smooth regions).

Interface velocity source (one of, mirroring the divergence-upwind FDM options):

- ``potential_field`` ``U`` -> ``alpha_{i+1/2} = -coupling*(U_{i+1} - U_i)/dx`` (the MFG-coupling
  entry point; matches the FDM divergence-upwind stencil exactly).
- ``drift_field`` ``alpha`` (the SDE/optimal-control velocity) -> averaged to the face,
  ``alpha_{i+1/2} = 1/2 (alpha_i + alpha_{i+1})``.

Time stepping: IMEX by Strang operator splitting -- explicit (CFL-bounded, sub-cycled)
MUSCL/upwind advection on each half step, implicit (backward-Euler) central diffusion in the
middle. Both sub-operators are individually mass-conserving (advection telescopes; the implicit
diffusion uses the conservative finite-volume Laplacian with ``1^T L = 0``), so the composite
step conserves mass exactly. The diffusion solve is an M-matrix, so positivity is preserved.

Diffusion ``D`` comes from the single-source converter ``diffusion_from_volatility`` (``D =
sigma^2/2``), matching the other FP solvers.

Boundary conditions: no-flux (zero wall flux -> exact conservation) and periodic (wrap face)
are fully supported (advection + diffusion). Dirichlet is supported for the diffusion operator
but advective Dirichlet inflow is deferred (Issue #422 scope note). Robin is not supported.

Scope (Issue #422 v1, standalone conservative FVM): 1D and 2D on ``TensorProductGrid``,
upwind + MUSCL, scalar diffusion. Deferred and out of scope: corner handling (#663), 3D,
WENO/PPM (3rd-order+), unstructured meshes, spatially-varying / tensor / callable volatility,
and the HASL/FVCN adjoint-SL research framework.
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from mfgarchon.alg.base_solver import SchemeFamily
from mfgarchon.alg.numerical.fp_solvers.base_fp import BaseFPSolver, DriftConvention
from mfgarchon.alg.numerical.fp_solvers.fp_fvm_flux import advective_divergence
from mfgarchon.geometry.boundary.types import BCType
from mfgarchon.operators.differential.laplacian import LaplacianOperator
from mfgarchon.utils.mfg_logging import get_logger
from mfgarchon.utils.pde_coefficients import assert_quadratic_minimize_drift, diffusion_from_volatility

if TYPE_CHECKING:
    from collections.abc import Callable

    from mfgarchon.geometry.boundary import BoundaryConditions

logger = get_logger(__name__)

Reconstruction = Literal["upwind", "muscl"]

# CFL targets for the explicit advection sub-steps (forward Euler). MUSCL needs the tighter
# bound for TVD/positivity; pure upwind tolerates a looser bound.
_CFL_TARGET = {"upwind": 0.8, "muscl": 0.4}


class FPFVMSolver(BaseFPSolver):
    """Conservative finite-volume Fokker-Planck solver (1D/2D, upwind or MUSCL).

    Parameters
    ----------
    problem : MFGProblem
        Problem definition (provides geometry, ``dt``, ``Nt``, ``sigma``, ``coupling_coefficient``).
    boundary_conditions : BoundaryConditions | None
        Explicit BC override. If ``None``, resolved from the problem/geometry (default no-flux).
    reconstruction : {"upwind", "muscl"}
        Face reconstruction. ``"muscl"`` (default) is 2nd order with a minmod limiter;
        ``"upwind"`` is 1st order.
    """

    _scheme_family = SchemeFamily.FVM

    # BoundaryCapable protocol (Issue #1456): the conservative FV scheme handles no-flux / Neumann
    # and periodic (wrap) walls; Dirichlet fails loud via the specific Issue #422 guard below, and
    # Robin / Reflecting / Extrapolation are not represented.
    _SUPPORTED_BC_TYPES: frozenset = frozenset({BCType.NO_FLUX, BCType.NEUMANN, BCType.PERIODIC})

    @property
    def supported_bc_types(self) -> frozenset:
        """BC types this solver supports (BoundaryCapable protocol)."""
        return self._SUPPORTED_BC_TYPES

    # The FP equation consumes the advective velocity alpha directly via ``drift_field``; a
    # value function ``U`` may instead be passed via ``potential_field`` (the solver then forms
    # alpha = -coupling*grad(U) at the faces). Default convention is VELOCITY.
    _drift_convention: DriftConvention = DriftConvention.VELOCITY

    def __init__(
        self,
        problem: Any,
        boundary_conditions: BoundaryConditions | None = None,
        reconstruction: Reconstruction = "muscl",
    ) -> None:
        super().__init__(problem)
        self.fp_method_name = "FVM"

        if reconstruction not in ("upwind", "muscl"):
            raise ValueError(f"Invalid reconstruction: {reconstruction!r}. Use 'upwind' or 'muscl'.")
        self.reconstruction: Reconstruction = reconstruction

        self.dimension = self._detect_dimension()

        from mfgarchon.geometry.protocols import SupportsLaplacian

        if not isinstance(problem.geometry, SupportsLaplacian):
            raise TypeError(
                f"FP FVM solver requires geometry with SupportsLaplacian trait for the diffusion "
                f"term. {type(problem.geometry).__name__} does not implement it. "
                f"Compatible geometries: TensorProductGrid, ImplicitDomain."
            )

        self.boundary_conditions = self._resolve_boundary_conditions(boundary_conditions)
        self._bc_types = self._resolve_bc_types(self.boundary_conditions, self.dimension)

        # Fail loud at construction (not at solve-time): the advective flux closure has no
        # Dirichlet inflow handling (deferred, Issue #422 scope; the diffusion operator alone
        # supports Dirichlet). Without this guard the solver would only raise from
        # ``fp_fvm_flux.axis_flux_divergence`` once an advected solve is attempted.
        if any(seg.bc_type == BCType.DIRICHLET for seg in self.boundary_conditions.segments):
            raise NotImplementedError(
                "FP FVM (v1) does not support Dirichlet BC (Issue #422 scope); use no_flux/neumann/periodic."
            )
        # Issue #1456: fail loud on a type FVM cannot honor at all (Robin / Reflecting /
        # Extrapolation). Dirichlet is handled by the specific guard above (its own message).
        self._validate_bc_support(self.boundary_conditions)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _resolve_boundary_conditions(self, boundary_conditions: BoundaryConditions | None) -> BoundaryConditions:
        """Resolve BC using the same hierarchy as FPFDMSolver (explicit > components > geometry)."""
        if boundary_conditions is not None:
            return boundary_conditions

        try:
            if self.problem.components is not None and self.problem.components.boundary_conditions is not None:
                return self.problem.components.boundary_conditions
        except AttributeError:
            pass

        try:
            bc = self.problem.geometry.boundary_conditions
            if bc is not None:
                return bc
        except AttributeError:
            pass

        try:
            bc = self.problem.geometry.get_boundary_conditions()
            if bc is not None:
                return bc
        except AttributeError:
            pass

        from mfgarchon.geometry.boundary import no_flux_bc

        return no_flux_bc(dimension=self.dimension)

    @staticmethod
    def _resolve_bc_types(bc: BoundaryConditions, ndim: int) -> list[str]:
        """Per-axis uniform BC type strings. Only uniform BCs are supported in v1."""
        try:
            bc_type = bc.type
        except ValueError as exc:
            raise NotImplementedError(
                "FP FVM solver supports only uniform boundary conditions in v1 (Issue #422). "
                "Mixed/per-segment BCs are deferred."
            ) from exc
        return [bc_type] * ndim

    def _scalar_diffusion(self, volatility_field: float | np.ndarray | Callable | None) -> float:
        """Resolve the scalar diffusion coefficient D = sigma^2/2 (single source)."""
        if volatility_field is None:
            sigma = self.problem.sigma
        elif isinstance(volatility_field, (int, float)):
            sigma = float(volatility_field)
        elif isinstance(volatility_field, np.ndarray):
            arr = np.asarray(volatility_field, dtype=float)
            if float(np.ptp(arr)) > 1e-12:
                raise NotImplementedError(
                    "FP FVM solver supports only scalar (constant) volatility in v1 (Issue #422). "
                    "Spatially-varying / tensor volatility is deferred."
                )
            sigma = float(arr.reshape(-1)[0])
        else:
            raise NotImplementedError(
                "FP FVM solver supports only scalar/None volatility in v1 (Issue #422). "
                f"Callable volatility is deferred (got {type(volatility_field).__name__})."
            )
        return float(diffusion_from_volatility(sigma))

    def _build_diffusion_lu(self, shape: tuple[int, ...], dt: float, diffusion: float):
        """Prefactor the backward-Euler diffusion matrix B = I - dt*D*L (conservative L).

        Returns ``None`` when D == 0 (no diffusion solve needed).
        """
        if diffusion == 0.0:
            return None
        spacing = list(self.problem.geometry.get_grid_spacing())
        # Periodic Laplacian closure is requested via bc=None (wrap); otherwise pass the BC so
        # the conservative finite-volume no-flux/Neumann closure (1^T L = 0) is used.
        lap_bc = None if all(t == "periodic" for t in self._bc_types) else self.boundary_conditions
        laplacian = LaplacianOperator(
            spacings=spacing, field_shape=shape, bc=lap_bc, mass_conservative=True
        ).as_scipy_sparse()
        n_total = int(np.prod(shape))
        system = sparse.eye(n_total, format="csc") - dt * diffusion * laplacian
        return splu(system.tocsc())

    # ------------------------------------------------------------------
    # Interface velocity
    # ------------------------------------------------------------------
    def _face_velocity_from_potential(self, u_slice: np.ndarray, m_slice: np.ndarray, t: float):
        """alpha_{i+1/2} = H.optimal_control(x_{i+1/2}, m_{i+1/2}, (U_{i+1}-U_i)/dx, t) per axis.

        Routes the interface velocity through the single owner of the control law -- the problem
        Hamiltonian's ``optimal_control`` (Issue #1528 / G-017) -- rather than the hand-coded
        ``-fp_drift_coefficient(problem)*grad(U)`` second-velocity fork. The forward-difference
        face gradient ``p_{i+1/2} = (U_{i+1}-U_i)/dx`` is the exact stencil
        ``compute_fp_velocity_field`` uses, so both velocity paths stay a single source.

        For a quadratic-MINIMIZE control cost ``optimal_control`` returns ``-p/lambda``,
        byte-identical to the old ``-c*grad(U)`` (``c = 1/lambda``) for dyadic lambda (incl. the
        paper's ``control_cost=1.0``) and within 2 ULP otherwise -- the face path composes TWO
        divisions ``(U_{i+1}-U_i)/dx`` then ``/lambda`` (vs. the old single multiply-by-``c``), so
        the non-dyadic rounding envelope is 2 ULP, not the single-op 1 ULP of the FEM/meshless/
        particle families. For a MAXIMIZE or regularized cost it is the correct ``+p/lambda`` /
        soft-thresholded drift instead of the wrong-sign (MAXIMIZE) / wrong-form (regularized)
        scalar. ``x``/``m`` are ignored by a separable control cost but passed face-consistently
        for a general (x/m-dependent) Hamiltonian.
        """
        H = getattr(self.problem, "hamiltonian_class", None)
        if H is None:
            raise ValueError(
                "FP FVM potential_field path needs problem.hamiltonian_class to source the "
                "advective drift alpha* = H.optimal_control(...); the problem has none. Pass a "
                "precomputed drift_field, or set a Hamiltonian on the problem (Issue #1528)."
            )
        # Issue #1528 review-nit: this potential-field path routes through H.optimal_control(x, m, p, t),
        # which is single-valued in p ONLY for a SeparableHamiltonian. A non-separable Hamiltonian (e.g.
        # CongestionHamiltonian) has a density/state-dependent optimal control, so calling optimal_control
        # here raised a cryptic TypeError; fail loud with a clear message instead. The gate lives at this
        # velocity-channel call site, NOT in the shared assert_quadratic_minimize_drift guard, which must
        # keep no-op'ing for non-separable H so fp_drift_coefficient's coupling_coefficient fallback stays
        # intact. Ordered before the #1542 assert so a MAXIMIZE Separable still hits the #1542 guard below.
        from mfgarchon.core.hamiltonian import SeparableHamiltonian

        if not isinstance(H, SeparableHamiltonian):
            raise NotImplementedError(
                f"FP FVM potential_field drift routes through H.optimal_control(x, m, p, t), which is "
                f"single-valued in p only for a SeparableHamiltonian; got {type(H).__name__} (non-separable), "
                f"whose optimal control is density/state-dependent. Provide a SeparableHamiltonian, or pass the "
                f"precomputed optimal-control velocity alpha* through the drift_field channel instead "
                f"(Issue #1528 / RFC #1574 Phase 1)."
            )
        # Issue #1528 PR-1 (behavior-neutral): preserve the #1542 fail-loud the removed
        # `fp_drift_coefficient` read carried. A MAXIMIZE / non-quadratic SeparableHamiltonian has no
        # scalar `-c*grad(U)` form, so raise here rather than silently advect H.optimal_control's
        # wrong-sign / wrong-form drift (that capability is Phase 1, not this byte-safe PR).
        assert_quadratic_minimize_drift(self.problem, context="FP FVM potential_field")
        spacing = list(self.problem.geometry.get_grid_spacing())
        bounds = self.problem.geometry.get_bounds()
        ndim = u_slice.ndim
        shape = u_slice.shape
        alpha_faces: list[np.ndarray] = []
        alpha_wrap: list[np.ndarray | None] = []
        for d in range(ndim):
            dx = spacing[d]
            # Same forward-difference face gradient the old -c*grad(U) fork used; the coefficient
            # owner changes (multiply-by-c -> H.optimal_control). The drift is byte-identical for
            # dyadic control_cost (incl. control_cost=1.0, where alpha* = -grad(U)) and within 2 ULP
            # otherwise -- materializing p_d = (U_{i+1}-U_i)/dx then dividing by lambda is two
            # divisions vs. the old multiply-then-divide (Issue #1528).
            p_d = np.diff(u_slice, axis=d) / dx
            nodes_d = np.linspace(bounds[0][d], bounds[1][d], shape[d])
            face_coord = 0.5 * (nodes_d[:-1] + nodes_d[1:])
            coord_shape = [1] * ndim
            coord_shape[d] = shape[d] - 1
            x_faces = np.broadcast_to(face_coord.reshape(coord_shape), p_d.shape)
            lo = [slice(None)] * ndim
            hi = [slice(None)] * ndim
            lo[d] = slice(0, shape[d] - 1)
            hi[d] = slice(1, shape[d])
            m_faces = 0.5 * (m_slice[tuple(lo)] + m_slice[tuple(hi)])
            alpha_d = np.asarray(H.optimal_control(x_faces, m_faces, p_d, t)).reshape(p_d.shape)
            alpha_faces.append(alpha_d)
            if self._bc_types[d] == "periodic":
                wrap = np.take(u_slice, 0, axis=d) - np.take(u_slice, -1, axis=d)
                p_wrap = wrap / dx
                # Seam face between x[-1] and x[0]+L; position is unused by a separable H but kept
                # face-consistent for a general (x-dependent) control cost.
                x_wrap = np.broadcast_to(np.asarray(nodes_d[-1] + 0.5 * dx), p_wrap.shape)
                m_wrap = 0.5 * (np.take(m_slice, 0, axis=d) + np.take(m_slice, -1, axis=d))
                alpha_w = np.asarray(H.optimal_control(x_wrap, m_wrap, p_wrap, t)).reshape(p_wrap.shape)
                alpha_wrap.append(alpha_w)
            else:
                alpha_wrap.append(None)
        return alpha_faces, alpha_wrap

    def _face_velocity_from_drift(self, drift_slice: np.ndarray, shape: tuple[int, ...]):
        """Average node-centered velocity to faces (+ periodic wrap face).

        Accepts a node-centered velocity (shape ``shape`` in 1D, or ``(*shape, ndim)`` in nD) or,
        in 1D, a face-centered velocity of length ``Nx-1`` (used directly).
        """
        ndim = len(shape)
        alpha_faces: list[np.ndarray] = []
        alpha_wrap: list[np.ndarray | None] = []

        if ndim == 1:
            n = shape[0]
            if drift_slice.shape == (n - 1,):
                # Already face-centered.
                alpha_faces.append(np.asarray(drift_slice, dtype=float))
                if self._bc_types[0] == "periodic":
                    raise NotImplementedError(
                        "Periodic FVM requires node-centered drift_field (shape (Nt, Nx)); a "
                        "face-centered drift has no defined wrap-face velocity."
                    )
                alpha_wrap.append(None)
                return alpha_faces, alpha_wrap
            if drift_slice.shape != (n,):
                raise ValueError(
                    f"1D drift_field slice has shape {drift_slice.shape}; expected node-centered "
                    f"({n},) or face-centered ({n - 1},)."
                )
            vd = np.asarray(drift_slice, dtype=float)
            alpha_faces.append(0.5 * (vd[:-1] + vd[1:]))
            if self._bc_types[0] == "periodic":
                alpha_wrap.append(np.asarray(0.5 * (vd[-1] + vd[0])))
            else:
                alpha_wrap.append(None)
            return alpha_faces, alpha_wrap

        # nD: vector velocity, shape (*shape, ndim).
        if drift_slice.shape != (*shape, ndim):
            raise ValueError(
                f"nD drift_field slice has shape {drift_slice.shape}; expected node-centered "
                f"vector field {(*shape, ndim)}."
            )
        for d in range(ndim):
            comp = np.asarray(drift_slice[..., d], dtype=float)
            lo = [slice(None)] * ndim
            hi = [slice(None)] * ndim
            lo[d] = slice(0, shape[d] - 1)
            hi[d] = slice(1, shape[d])
            alpha_faces.append(0.5 * (comp[tuple(lo)] + comp[tuple(hi)]))
            if self._bc_types[d] == "periodic":
                alpha_wrap.append(0.5 * (np.take(comp, -1, axis=d) + np.take(comp, 0, axis=d)))
            else:
                alpha_wrap.append(None)
        return alpha_faces, alpha_wrap

    # ------------------------------------------------------------------
    # Sub-operators
    # ------------------------------------------------------------------
    def _advect(self, m, alpha_faces, alpha_wrap, dt_adv, spacing):
        """Explicit (CFL-bounded, sub-cycled forward Euler) advection over ``dt_adv``."""
        amax = max((float(np.max(np.abs(a))) for a in alpha_faces), default=0.0)
        for aw in alpha_wrap:
            if aw is not None and aw.size:
                amax = max(amax, float(np.max(np.abs(aw))))
        if amax == 0.0:
            return m
        cfl = _CFL_TARGET[self.reconstruction]
        dx_min = min(spacing)
        n_sub = max(1, math.ceil(dt_adv * amax / (cfl * dx_min)))
        dt_sub = dt_adv / n_sub
        for _ in range(n_sub):
            div = advective_divergence(m, alpha_faces, alpha_wrap, spacing, self.reconstruction, self._bc_types)
            m = m - dt_sub * div
        return m

    # ------------------------------------------------------------------
    # Main solve
    # ------------------------------------------------------------------
    def solve_fp_system(
        self,
        M_initial: np.ndarray | None = None,
        drift_field: np.ndarray | Callable | None = None,
        volatility_field: float | np.ndarray | Callable | None = None,
        show_progress: bool | None = None,
        progress_callback: Callable[[int], None] | None = None,
        potential_field: np.ndarray | None = None,
        source_term: Callable[[float, Any], np.ndarray] | None = None,
        diffusion_field: float | np.ndarray | Callable | None = None,
    ) -> np.ndarray:
        """Evolve the FP density forward in time with the conservative FVM scheme.

        Parameters
        ----------
        M_initial : np.ndarray
            Initial cell averages ``m(0, x)``, shape ``(*spatial_shape)``.
        drift_field : np.ndarray | None
            Advective velocity ``alpha(t, x)`` (node-centered). 1D shape ``(Nt+1, Nx)``;
            nD shape ``(Nt+1, *spatial, ndim)``. Averaged to faces. Mutually exclusive with
            ``potential_field``.
        volatility_field : float | None
            SDE volatility ``sigma`` (``D = sigma^2/2``). ``None`` uses ``problem.sigma``.
            Only scalar/constant volatility is supported in v1.
        potential_field : np.ndarray | None
            Value function ``U(t, x)``, shape ``(Nt+1, *spatial)``. The face velocity is
            ``alpha = -coupling*grad(U)`` (MFG coupling entry point). Mutually exclusive with
            ``drift_field``.
        source_term : Callable | None
            Optional MMS source ``S(t, x_grid)``, applied explicitly (breaks exact conservation
            by design, since a source adds mass).
        diffusion_field : float | np.ndarray | None
            Deprecated alias for ``volatility_field`` (accepted for API parity).

        Returns
        -------
        np.ndarray
            Density evolution, shape ``(Nt+1, *spatial_shape)``.
        """
        if M_initial is None:
            M_initial = self.problem.get_initial_density()
        m0 = np.asarray(M_initial, dtype=float)
        shape = tuple(self.problem.geometry.get_grid_shape())
        if m0.shape != shape:
            raise ValueError(f"M_initial shape {m0.shape} does not match grid shape {shape}.")

        if diffusion_field is not None and volatility_field is None:
            volatility_field = diffusion_field

        if drift_field is not None and potential_field is not None:
            raise ValueError(
                "Specify at most one of drift_field (velocity) or potential_field (value function U), not both."
            )
        if callable(drift_field):
            raise NotImplementedError(
                "FP FVM solver supports only array/None drift_field in v1 (Issue #422). Callable drift is deferred."
            )

        spacing = list(self.problem.geometry.get_grid_spacing())
        diffusion = self._scalar_diffusion(volatility_field)

        # Number of time points: from the provided field, else from the problem.
        if potential_field is not None:
            field = np.asarray(potential_field, dtype=float)
            n_time = field.shape[0]
            velocity_mode = "potential"
        elif drift_field is not None:
            field = np.asarray(drift_field, dtype=float)
            n_time = field.shape[0]
            velocity_mode = "drift"
        else:
            field = None
            n_time = self.problem.Nt + 1
            velocity_mode = "zero"

        dt = self.problem.dt
        n_steps = max(0, n_time - 1)

        # CFL diagnostic for the implicit-diffusion accuracy (stability is unconditional).
        if diffusion > 0.0 and dt > 0.0:
            cfl_diff = 2.0 * diffusion * dt / (min(spacing) ** 2)
            if cfl_diff > 1.0:
                logger.debug(
                    "FVM diffusive CFL=%.2f (implicit, stable; accuracy may degrade for CFL>>1).",
                    cfl_diff,
                )

        lu = self._build_diffusion_lu(shape, dt, diffusion)

        m_solution = np.empty((n_time, *shape), dtype=float)
        m_solution[0] = m0
        m = m0.copy()

        source_grid = self.problem.geometry.meshgrid() if source_term is not None else None

        for k in range(n_steps):
            idx = min(k, field.shape[0] - 1) if field is not None else 0
            if velocity_mode == "potential":
                # t = n*dt at timestep n (matches compute_fp_velocity_field); m is the current
                # density (own-population face density for a general x/m-dependent control cost).
                alpha_faces, alpha_wrap = self._face_velocity_from_potential(field[idx], m, k * dt)
            elif velocity_mode == "drift":
                alpha_faces, alpha_wrap = self._face_velocity_from_drift(field[idx], shape)
            else:
                alpha_faces = [np.zeros(_face_shape(shape, d), dtype=float) for d in range(len(shape))]
                alpha_wrap = [None] * len(shape)

            m = self._strang_step(m, alpha_faces, alpha_wrap, dt, spacing, lu)

            if source_term is not None:
                m = m + dt * np.asarray(source_term(k * dt, source_grid), dtype=float)

            if not np.all(np.isfinite(m)):
                n_bad = int(np.sum(~np.isfinite(m)))
                raise ValueError(
                    f"FP FVM solver produced {n_bad} non-finite values at timestep {k + 1}/{n_steps}. "
                    "Check the CFL/velocity magnitude."
                )

            m_solution[k + 1] = m
            if progress_callback is not None:
                progress_callback(1)

        min_density = float(np.min(m_solution))
        if min_density < -1e-12:
            warnings.warn(
                f"FP FVM solver: min density {min_density:.2e} < 0. The MUSCL limiter should "
                "prevent this; check the CFL/limiter for the advection regime.",
                UserWarning,
                stacklevel=2,
            )

        return m_solution

    def _strang_step(self, m, alpha_faces, alpha_wrap, dt, spacing, lu):
        """One Strang split step: 1/2 advection, full implicit diffusion, 1/2 advection."""
        amax = max((float(np.max(np.abs(a))) for a in alpha_faces), default=0.0)
        has_advection = amax > 0.0

        if lu is not None and has_advection:
            m = self._advect(m, alpha_faces, alpha_wrap, 0.5 * dt, spacing)
            m = lu.solve(m.ravel()).reshape(m.shape)
            m = self._advect(m, alpha_faces, alpha_wrap, 0.5 * dt, spacing)
        elif lu is not None:
            m = lu.solve(m.ravel()).reshape(m.shape)
        elif has_advection:
            m = self._advect(m, alpha_faces, alpha_wrap, dt, spacing)
        return m


def _face_shape(shape: tuple[int, ...], axis: int) -> tuple[int, ...]:
    """Shape of the interior-face velocity array along ``axis`` (``shape`` with axis-1 reduced)."""
    return tuple(n - 1 if d == axis else n for d, n in enumerate(shape))


if __name__ == "__main__":
    """Smoke test: free diffusion of a Gaussian conserves mass to machine precision."""
    import numpy as _np

    from mfgarchon import MFGProblem
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
    from mfgarchon.core.mfg_problem import MFGComponents
    from mfgarchon.geometry import TensorProductGrid
    from mfgarchon.geometry.boundary import no_flux_bc

    H = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: 0.0,
        coupling_dm=lambda m: 0.0,
    )
    comps = MFGComponents(hamiltonian=H, u_terminal=lambda x: 0.0, m_initial=lambda x: 1.0)
    geom = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[101], boundary_conditions=no_flux_bc(dimension=1))
    prob = MFGProblem(geometry=geom, T=0.1, Nt=50, sigma=0.3, components=comps)

    x = _np.linspace(0.0, 1.0, 101)
    dx = x[1] - x[0]
    m_init = _np.exp(-((x - 0.5) ** 2) / (2 * 0.1**2))
    m_init /= m_init.sum() * dx

    solver = FPFVMSolver(prob, reconstruction="muscl")
    M = solver.solve_fp_system(m_init)
    mass = M.sum(axis=1) * dx
    drift = float(_np.max(_np.abs(mass - mass[0])))
    print(f"  shape={M.shape}, mass drift={drift:.2e}, min={M.min():.2e}")
    assert drift < 1e-12, f"mass drift too large: {drift:.2e}"
    assert M.min() >= -1e-14, f"negative density: {M.min():.2e}"
    print("FVM smoke test passed.")
