from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

import numpy as np

# Import MFGComponents and mixins from the dedicated module
from mfgarchon.core.hamiltonian import HamiltonianBase
from mfgarchon.core.mfg_components import (
    ConditionsMixin,
    HamiltonianMixin,
    MFGComponents,
)

# Issue #543: Runtime import for isinstance() checks
from mfgarchon.geometry.protocol import GeometryProtocol  # noqa: TC001

# Deprecation utilities (Issue #616, #666)
from mfgarchon.types.callable_protocols import (
    HAMILTONIAN_SLOTS,
    POTENTIAL_SLOTS,
    SOURCE_TERM_SLOTS,
    Slots,
    bind_user_callable,
    bound_attribute,
    refuse_methods_out_of_order,
)
from mfgarchon.utils.deprecation import validate_kwargs

# Use unified nD-capable BoundaryConditions from conditions.py

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from mfgarchon.types.pde_coefficients import DriftField


_RETIRED_VOLATILITY_ATTRIBUTE = (
    "problem.{name} is retired (#2375 ruling 6). The SDE volatility Sigma is problem.volatility, held "
    "as supplied (scalar, array or callable), and the PDE diffusion A = 1/2 Sigma Sigma^T is "
    "problem.diffusion. A consumer that needs one scalar calls scalar_volatility(problem.volatility, consumer=...) "
    "from mfgarchon.utils.pde_coefficients, which refuses an array or a callable rather than "
    "averaging it (#2376)."
)


# ============================================================================
# Unified MFG Problem Class
# ============================================================================


class SpatialMeasure(NamedTuple):
    """How a problem integrates a field over space, and the name of that measure (#2555).

    ``integrate`` reduces the TRAILING spatial axes, so a ``(time, *spatial)`` history integrates to one
    value per time row. Returned by :meth:`MFGProblem.spatial_measure`.
    """

    integrate: Callable[[NDArray], Any]
    name: str


def _same_geometry(a: object, b: object) -> bool:
    """Whether the two geometry arguments are the same geometry (Issue #1765).

    Identity, not equality. Three attempts at proving two separately-constructed geometries
    describe the same discretisation each shipped a defect that a review had to find:

    - comparing a fixed list of attribute names accepted a pair as soon as ANY ONE matched, so
      two ``Mesh2D`` differing tenfold in ``mesh_size`` compared equal;
    - comparing ``get_collocation_points()`` assumed a uniform accessor, but that method is a
      deterministic accessor on grids, raises on an ungenerated ``Mesh2D``, is a *sampler* with a
      required argument on ``Hyperrectangle``, and a fresh random draw on ``ImplicitDomain``;
    - comparing instance state via ``vars()`` conflated the discretisation with the object's
      lifecycle, so a grid that had merely been USED (a populated ``_flattened_cache``) stopped
      equalling an identical fresh one, and two identically-built ``Mesh1D`` were refused because
      ``MeshData.__eq__`` returns an array.

    Each attempt was a cleverer structural comparison and each had a case it did not anticipate.
    Identity has none: it cannot be fooled, it costs nothing, and its only failure direction is
    refusing something that would have been fine -- which is loud, and which the error message
    tells the caller how to resolve in one edit. A silent wrong answer is the thing being
    prevented; over-refusing is not that.
    """
    return a is b


def _evaluation_shape(x: Any, m: Any) -> tuple[int, ...] | None:
    """The shape of the points a callable volatility was evaluated at, or None if it cannot be read.

    The density's shape when one is given (every solver passes it). Otherwise x's: an array of points
    shaped (..., d) gives its leading axes, a 1-D array its own shape, and a meshgrid list -- d arrays
    of one shape, each with d axes -- that shape. A list of per-axis coordinate vectors is not a set
    of points and gives None.
    """
    if np.ndim(m) > 0:
        return tuple(np.shape(m))
    if isinstance(x, (list, tuple)):
        shapes = {np.shape(axis) for axis in x}
        if len(shapes) == 1 and all(np.ndim(axis) == len(x) for axis in x):
            return next(iter(shapes))
        return None
    return tuple(np.shape(x)[:-1]) if np.ndim(x) >= 2 else tuple(np.shape(x))


def _callable_output_kind(value: Any, declared: str | None, dimension: Any, points: tuple[int, ...] | None) -> str:
    """How a callable volatility's (or diffusion's) output is read: as declared, else per point.

    Per point is how every solver that evaluates a callable volatility reads its array output. The
    exception is an output whose trailing axes are (d, d) and which is not simply one value per
    evaluation point: a constant (d, d) tensor, a per-point tensor, or -- on a d x d grid, where the
    two readings coincide -- either. There the library does not guess (#2375 ruling 6). A 3-D grid
    shaped (Nx, 3, 3) returns one value per point with trailing (3, 3) axes, and is read per point.
    """
    if declared is not None:
        return declared
    shape = np.shape(value)
    if not (isinstance(dimension, int) and dimension >= 2 and shape[-2:] == (dimension, dimension)):
        return "field"
    if shape == points and shape != (dimension, dimension):
        return "field"
    raise ValueError(
        f"A callable volatility with no volatility_kind returned an array of shape {shape} at points "
        f"of shape {points if points is not None else 'unknown (pass the density m)'}; its trailing "
        f"({dimension}, {dimension}) axes make it a tensor, or on a "
        f"{dimension} x {dimension} grid either a tensor or a per-point field. Declare "
        "volatility_kind='tensor' (a noise matrix) or 'field' (isotropic per point) on the problem "
        "(#2375 ruling 6)."
    )


class MFGProblem(HamiltonianMixin, ConditionsMixin):
    """
    Unified MFG problem class that can handle both predefined and custom formulations.

    This class serves as the single constructor for all MFG problems:
    - Default usage: Uses built-in Hamiltonian (standard MFG formulation)
    - Custom usage: Accepts MFGComponents for full mathematical control

    Inherits from two mixins:

    HamiltonianMixin (mathematical Hamiltonian):
    - H(): Hamiltonian function
    - dH_dm(): Hamiltonian derivative w.r.t. density
    - get_hjb_hamiltonian_jacobian_contrib(): Jacobian for Newton methods
    - get_hjb_residual_m_coupling_term(): Coupling terms

    ConditionsMixin (problem setup):
    - get_boundary_conditions(): Boundary condition accessor
    - _setup_custom_initial_density(): Initial density setup
    - _setup_custom_final_value(): Final value setup

    Sign conventions (#2375 ruling 3)
    ---------------------------------
    Every scalar field that enters the HJB equation is **cost-signed**: a positive value raises
    the value function ``u`` and repels agents. Written with all of them on the right-hand side,

        -u_t + H_control(Du) - (sigma^2/2) Lap u = V + f(m) + S

    - ``SeparableHamiltonian(potential=V, coupling=f)``: ``V`` and ``f`` live inside ``H``, which
      carries them as ``H = H_control - V - f``. An attractive well at ``x_c`` is a bowl,
      ``V = +0.5 * C * (x - x_c)**2``.
    - ``MFGProblem(source_term_hjb=S)``: ``S`` stays on the right-hand side, because it may depend
      on ``u``. The residual subtracts it (``Phi_U -= source_term`` in base_hjb.py).

    So the same effect through either channel takes an input of the SAME sign.
    """

    # Methods a subclass may define that take #2375 ruling 8's order, checked at class creation.
    # A subclass that adds such methods declares them here too; the tables are merged along the MRO.
    _ruling8_methods: ClassVar[dict[str, Slots | None]] = {
        "hamiltonian": HAMILTONIAN_SLOTS,
        "running_cost": Slots(("t", "x", "m")),
    }

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        table: dict[str, Slots | None] = {}
        for base in reversed(cls.__mro__):
            table.update(vars(base).get("_ruling8_methods", {}))
        refuse_methods_out_of_order(cls, table)

    # Type annotations for geometry attributes (Phase 6 of Issue #435)
    # These are always non-None after __init__ completes
    geometry: GeometryProtocol
    hjb_geometry: GeometryProtocol | None
    fp_geometry: GeometryProtocol | None

    # drift_field: Optional drift (float, array, or callable). The volatility is the read-only
    # property `volatility`, and the diffusion `A = 1/2 Sigma Sigma^T` is derived from it (#2375 ruling 6).
    drift_field: DriftField

    def __init__(
        self,
        # === API v1.0 parameters (Issue #875) ===
        model: Any | None = None,  # Model instance (game rules)
        domain: GeometryProtocol | None = None,  # Spatial geometry
        conditions: Any | None = None,  # Conditions instance (time + IC/TC)
        constraints: list | None = None,  # Optional constraints (future)
        # === Legacy parameters (all below deprecated in favor of model/domain/conditions) ===
        # N-D grid parameters
        spatial_bounds: list[tuple[float, float]] | None = None,
        spatial_discretization: list[int] | None = None,
        # Complex geometry parameters (NEW)
        geometry: GeometryProtocol | None = None,
        obstacles: list | None = None,
        # Dual geometry parameters (Issue #257)
        hjb_geometry: GeometryProtocol | None = None,
        fp_geometry: GeometryProtocol | None = None,
        # Network parameters (NEW)
        network: Any | None = None,  # NetworkGraph
        # Time domain parameters
        T: float | None = None,
        Nt: int | None = None,
        time_domain: tuple[float, int] | None = None,  # Alternative to T/Nt
        # Physical parameters (#2375 ruling 6): volatility= is Sigma, diffusion= is A = 1/2 Sigma Sigma^T
        diffusion: float | NDArray[np.floating] | Callable | None = None,  # PDE diffusion A
        volatility: float | NDArray[np.floating] | Callable | None = None,  # SDE volatility Sigma
        volatility_kind: str | None = None,  # "field" | "tensor"; required for an array
        drift: float | NDArray[np.floating] | Callable | None = None,  # Optional drift field
        coupling_coefficient: float = 0.5,
        # MFG coupling parameters
        lambda_: float | None = None,  # Control cost (H uses |p|²/(2λ))
        # Class-based Hamiltonian (Issue #673 - recommended)
        hamiltonian: Any | None = None,  # HamiltonianBase instance
        # Advanced
        components: MFGComponents | None = None,
        suppress_warnings: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize MFG problem with support for all spatial dimensions and domain types.

        Supports four initialization modes:
        1. N-D grid mode: Specify spatial_bounds, spatial_discretization
        2. Geometry mode: Specify geometry object (with optional obstacles)
        3. Network mode: Specify network graph
        4. Custom components: Full mathematical control via MFGComponents

        For a 1D grid, use the geometry-first API:
            geometry=TensorProductGrid(bounds=[(xmin, xmax)], Nx_points=[Nx + 1])

        Args:
            spatial_bounds: List of (min, max) tuples for each dimension
                           Example: [(0, 1), (0, 1)] for 2D unit square
            spatial_discretization: Interval count per dimension (Nx); the grid has Nx + 1
                                   points per axis. Example: [50, 50] for a 51×51 grid
            geometry: BaseGeometry object for complex domains (unified mode)
            hjb_geometry / fp_geometry: Must be specified together AND must be the SAME OBJECT.
                Two separately-constructed geometries raise NotImplementedError even when they are
                built identically (Issue #1765) -- the check is identity, not equality, because
                three attempts at deciding whether two objects describe the same discretisation
                each shipped a wrong answer. Nothing downstream resamples between two geometries,
                so bind one object to both names, or drive GeometryProjector yourself.
            obstacles: List of obstacle geometries
            hjb_geometry: Geometry for HJB solver (dual geometry mode, Issue #257)
            fp_geometry: Geometry for FP solver (dual geometry mode, Issue #257)
                        Note: Both hjb_geometry and fp_geometry must be specified together
            network: NetworkGraph for network MFG problems
            T, Nt, time_domain: Time domain parameters (T, Nt) or tuple (T, Nt)
            volatility: The SDE volatility Sigma in dX = alpha dt + Sigma dW (#2375 ruling 6).
                Held as supplied on ``problem.volatility``; the PDE diffusion
                A = 1/2 Sigma Sigma^T is derived from it on ``problem.diffusion``.
                - None (and no diffusion=): deterministic, Sigma = 0
                - float: constant isotropic sigma
                - ndarray: needs ``volatility_kind`` -- there is no default
                - Callable: state-dependent Sigma(t, x, m)
                Mutually exclusive with diffusion=. ``sigma=`` is retired and raises.
            volatility_kind: How an array volatility (or an array diffusion=) is read:
                ``"field"`` -- isotropic per-point sigma, shape ``spatial_shape``;
                ``"tensor"`` -- trailing ``(d, k)`` axes are the noise matrix, constant
                ``(d, k)`` or ``(*spatial_shape, d, k)``. Required for an array, refused for
                a scalar. A ``(d, d)`` tensor and a d x d field have the same shape, which is
                why the library does not guess.
            diffusion: The PDE diffusion A (D = sigma^2/2 in the scalar case). Converted to the
                volatility it implies -- sqrt(2D), or for a tensor the symmetric square root of
                2A -- so ``problem.volatility`` is always Sigma. Same types as volatility=, with
                the same ``volatility_kind`` rule for an array. Mutually exclusive with volatility=.
            drift: Drift field α(t, x, m) for FP equation. None → 0 (no drift).
                Supports:
                - None: No drift (no advection)
                - float: Constant drift (same in all directions)
                - ndarray: Precomputed drift array
                - Callable: State-dependent α(t, x, m) -> float | ndarray
            coupling_coefficient: Control cost coefficient
            components: Optional MFGComponents for custom problem definition
            suppress_warnings: Suppress computational feasibility warnings
            **kwargs: Additional parameters

        Examples:
            # Mode 1: 1D grid via geometry-first API
            geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[101])
            problem = MFGProblem(geometry=geometry, T=1.0, Nt=100)

            # Mode 2: N-D grid
            problem = MFGProblem(
                spatial_bounds=[(0, 1), (0, 1)],
                spatial_discretization=[50, 50],
                Nt=50
            )

            # Mode 3: Complex geometry with obstacles
            from mfgarchon.geometry import Hyperrectangle, Hypersphere
            domain = Hyperrectangle(bounds=[[0, 1], [0, 1]])
            obstacle = Hypersphere(center=[0.5, 0.5], radius=0.1)
            problem = MFGProblem(
                geometry=domain,
                obstacles=[obstacle],
                time_domain=(1.0, 50),
                diffusion=0.1
            )

            # Mode 4: Network MFG
            import networkx as nx
            graph = nx.grid_2d_graph(10, 10)
            problem = MFGProblem(network=graph, time_domain=(1.0, 100))

            # Mode 5: Custom components
            components = MFGComponents(hamiltonian=SeparableHamiltonian(...), m_initial=..., u_terminal=...)
            problem = MFGProblem(
                spatial_bounds=[(0, 1)],
                spatial_discretization=[100],
                Nt=50,
                components=components
            )

            # Mode 6: Dual geometry (Issue #257) - REFUSED when the two geometries differ.
            # Nothing downstream honours the difference: the FP geometry was accepted and then
            # discarded, so the FP equation silently solved on the HJB grid (Issue #1765).
            # Differing geometries now raise NotImplementedError. To resample between two
            # resolutions, drive GeometryProjector explicitly:
            from mfgarchon.geometry import GeometryProjector, TensorProductGrid
            from mfgarchon.geometry.boundary import no_flux_bc
            hjb_grid = TensorProductGrid(bounds=[(0, 1)], Nx_points=[41], boundary_conditions=no_flux_bc(1))
            fp_grid = TensorProductGrid(bounds=[(0, 1)], Nx_points=[11], boundary_conditions=no_flux_bc(1))
            projector = GeometryProjector(hjb_geometry=hjb_grid, fp_geometry=fp_grid)
            m_on_fp = projector.project_hjb_to_fp(np.zeros(41))

            # Advanced: State-dependent volatility (callable)
            def density_dependent_volatility(t, x, m):
                return 0.1 * (1 + m)  # More noise in dense regions
            problem = MFGProblem(
                geometry=domain,
                volatility=density_dependent_volatility,
                time_domain=(1.0, 50)
            )

            # Advanced: Spatially varying volatility (array, isotropic per point)
            sigma_array = np.ones((51, 51)) * 0.1
            sigma_array[20:30, 20:30] = 0.5  # More noise in the center region
            problem = MFGProblem(
                geometry=domain,
                volatility=sigma_array,
                volatility_kind="field",
                time_domain=(1.0, 50)
            )

            # Advanced: Custom drift field
            def crowd_avoidance_drift(t, x, m):
                grad_m = np.gradient(m)  # Density gradient
                return -np.stack(grad_m, axis=-1)  # Move down gradient
            problem = MFGProblem(
                geometry=domain,
                drift=crowd_avoidance_drift,
                time_domain=(1.0, 50)
            )
        """
        import warnings

        if "sigma" in kwargs:
            raise TypeError(
                "MFGProblem(sigma=...) is retired (#2375 ruling 6). Pass volatility= -- the SDE "
                "volatility Sigma, with volatility_kind='field' or 'tensor' for an array -- or "
                "diffusion= -- the PDE coefficient A = 1/2 Sigma Sigma^T. On the v1.0 API it is "
                "Model(volatility=...)."
            )
        if "gamma" in kwargs:
            raise TypeError(
                "MFGProblem(gamma=...) is retired (#2554): it was stored and no solver read it, so it "
                "changed nothing. A density term in H goes through the Hamiltonian's coupling channel, "
                "e.g. SeparableHamiltonian(control_cost=..., coupling=lambda m: gamma * m**2, "
                "coupling_dm=lambda m: 2 * gamma * m); H carries it as -f (cost-signed)."
            )

        # =====================================================================
        # API v1.0 path (Issue #875): Model + Domain + Conditions
        # =====================================================================
        from mfgarchon.core.model import Conditions as _Conditions
        from mfgarchon.core.model import Model as _Model

        _new_api = model is not None or domain is not None or conditions is not None
        if _new_api:
            # Validate all three are provided
            if model is None or domain is None or conditions is None:
                _missing = [
                    n for n, v in [("model", model), ("domain", domain), ("conditions", conditions)] if v is None
                ]
                raise ValueError(
                    f"API v1.0 requires all three: model, domain, conditions. Missing: {', '.join(_missing)}"
                )
            if not isinstance(model, _Model):
                raise TypeError(f"model must be a Model instance, got {type(model).__name__}")
            if not isinstance(conditions, _Conditions):
                raise TypeError(f"conditions must be a Conditions instance, got {type(conditions).__name__}")

            # Check for conflicting legacy parameters
            # Note: Nt is allowed with v1.0 API — sets construction default
            # (can be overridden at solve() time)
            _legacy_params = {
                n: v
                for n, v in [
                    ("geometry", geometry),
                    ("spatial_bounds", spatial_bounds),
                    ("spatial_discretization", spatial_discretization),
                    ("diffusion", diffusion),
                    ("volatility", volatility),
                    ("volatility_kind", volatility_kind),
                    ("T", T),
                    ("time_domain", time_domain),
                    ("components", components),
                    ("hamiltonian", hamiltonian),
                ]
                if v is not None
            }
            if _legacy_params:
                raise ValueError(
                    f"Cannot mix API v1.0 (model/domain/conditions) with legacy parameters. "
                    f"Got legacy parameters: {', '.join(_legacy_params.keys())}"
                )

            # Translate to legacy parameters
            geometry = domain
            volatility = model.volatility
            volatility_kind = model.volatility_kind
            T = conditions.T
            # Nt must be provided — fail fast (no silent defaults)
            if Nt is None:
                raise ValueError(
                    "Nt (time discretization) is required. "
                    "Pass Nt= to MFGProblem() or use problem.solve(Nt=...).\n"
                    "  MFGProblem(model=..., domain=..., conditions=..., Nt=50)"
                )
            # Build MFGComponents from Model + Conditions
            _h = (
                model.effective_hamiltonian if (model.hamiltonian is not None or model.lagrangian is not None) else None
            )
            components = MFGComponents(
                hamiltonian=_h,
                u_terminal=conditions.u_terminal,
                m_initial=conditions.m_initial,
            )
            # Drift-only models: pass drift through
            if model.drift_field is not None:
                drift = model.drift_field
            # Store v1.0 objects for solve() and with_*() methods
            self._v1_model = model
            self._v1_conditions = conditions
            self._v1_constraints = constraints
        else:
            self._v1_model = None
            self._v1_conditions = None
            self._v1_constraints = None

            # Issue #875: Deprecation warning for legacy API path
            warnings.warn(
                "Legacy MFGProblem(geometry=, components=, volatility=, T=, Nt=) is deprecated. "
                "Use the v1.0 API: MFGProblem(model=Model(...), domain=grid, "
                "conditions=Conditions(...), Nt=50). "
                "Legacy support will be removed in v1.0.0.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Normalize parameter aliases
        if time_domain is not None:
            if T is not None or Nt is not None:
                raise ValueError("Specify EITHER (T, Nt) OR time_domain, not both")
            T, Nt = time_domain

        # --- volatility= / diffusion=: mutually exclusive (#811, #2375 ruling 6) ---
        if diffusion is not None and volatility is not None:
            raise ValueError("Specify at most one of: diffusion=, volatility=. Got both.")
        # An array-like volatility becomes a float array, and a 0-d one a float, before any check reads
        # it: a list would otherwise skip the array validation, and a 0-d array the sign check. A
        # diffusion= array-like needs no such step -- volatility_from_diffusion converts it below.
        if volatility is not None and not callable(volatility):
            volatility = np.asarray(volatility, dtype=float) if np.ndim(volatility) > 0 else float(volatility)
        supplied = volatility if volatility is not None else diffusion
        if volatility_kind is not None:
            if volatility_kind not in ("field", "tensor"):
                raise ValueError(f"volatility_kind must be 'field' or 'tensor', got {volatility_kind!r}.")
            if supplied is None or (not callable(supplied) and np.ndim(supplied) == 0):
                raise ValueError(
                    "volatility_kind says how an ARRAY (or a callable's array output) is read; it was "
                    f"given with {'no' if supplied is None else 'a scalar'} volatility/diffusion, where "
                    "it would be read by nothing."
                )
        if supplied is not None and not callable(supplied) and volatility_kind is None:
            if np.ndim(supplied) > 0:
                raise ValueError(
                    f"An array {'volatility' if volatility is not None else 'diffusion'} of shape "
                    f"{np.shape(supplied)} needs volatility_kind='field' (isotropic per-point) or 'tensor' "
                    "(trailing (d, k) noise matrix). A (d, d) tensor and a d x d spatial field have the "
                    "same shape, and the library does not guess (#2375 ruling 6)."
                )

        # After this block `vola_value` is the SDE volatility Sigma, as supplied or as implied by D.
        if volatility is not None:
            # Issue #1077 (case 3): reject a NEGATIVE scalar volatility (fail-fast). A scalar 0 is
            # the legitimate deterministic value, identical to passing nothing, so
            # MFGProblem(volatility=other.volatility) from a deterministic `other` does not raise.
            # Callable / array volatility (Issue #1248) is validated where it is evaluated.
            if isinstance(volatility, float) and volatility < 0:
                raise ValueError(
                    f"volatility (the SDE volatility) must be >= 0, got {volatility}. "
                    "Use volatility=0 or leave it unset for deterministic dynamics."
                )
            vola_value = volatility
        elif diffusion is not None:
            # D -> Sigma by the one reverse converter (a negative D is refused there, #811).
            from mfgarchon.utils.pde_coefficients import volatility_from_diffusion

            if callable(diffusion):
                _D_callable = diffusion

                def vola_value(t, x, m, *, _D=_D_callable, _kind=volatility_kind):
                    D = _D(t, x, m)
                    kind = _callable_output_kind(D, _kind, self.dimension, _evaluation_shape(x, m))
                    return volatility_from_diffusion(D, kind=kind)
            else:
                vola_value = volatility_from_diffusion(diffusion, kind=volatility_kind)
        else:
            # No physical parameter specified -- deterministic (Sigma = 0)
            vola_value = 0.0

        # Issue #1077: T <= 0 is a degenerate (zero/inverted) time horizon — dt = T/Nt = 0
        # silently produces all-zero integration. Fail fast, consistent with MFGGridConfig's
        # T-positive validation. (Nt=0 is intentionally graceful — see test_mfg_problem_zero_nt.)
        if T is not None and T <= 0:
            raise ValueError(f"T (time horizon) must be > 0, got {T}.")

        # Set defaults for T, Nt if not provided
        if T is None:
            T = 1.0
        if Nt is None:
            Nt = 51

        # drift default
        if drift is None:
            drift = 0.0

        # The volatility is held as supplied -- scalar, array or callable -- and never collapsed to
        # a representative scalar (#2376). A consumer that can only use a scalar asks
        # `scalar_volatility(problem.volatility, consumer=...)`, which refuses anything else.
        # Note (Issue #1085): mfgarchon SDE convention is **Itô**, not Stratonovich.
        # For constant sigma, the two coincide. For callable sigma(t, x, m) with
        # spatial dependence, users with Stratonovich-derived drift must apply
        # the correction `alpha_Ito = alpha_Strat - (1/2) sigma * d_x sigma`
        # before passing the drift. mfgarchon does NOT add this correction.
        self._volatility = vola_value
        self._volatility_kind = volatility_kind
        self.drift_field = drift

        # Extended PDE form fields (Issue #921).
        # These enable generalized MFG equations beyond the classical form:
        #   HJB: -du/dt + H(x,m,Du) - S_hjb = 0
        #   FP:  dm/dt - (sigma^2/2)Dm - div(m*alpha*) - S_fp = 0
        # source_term_hjb/fp: Callable(t, x, v, m) -> array (problem-level signature, #2375 ruling 8)
        # nonlocal_operator: LinearOperator for integro-differential terms J[v]
        # state_penalty: Callable(x) -> array, a COST-signed level-set penalty (soft wall).
        #   Composed into the Hamiltonian's potential, NOT into source_term -- see below.
        # obstacle: RETIRED (#2002). It named a variational inequality and delivered a
        #   position penalty; setting it now raises and names both successors.
        #
        # Sign convention (#2375 ruling 3)
        # --------------------------------
        # source_term_hjb enters the canonical HJB on the RIGHT-hand side:
        #     -u_t + H(x, m, Du) - (sigma^2/2) Lap u = S_hjb
        # The residual assembly SUBTRACTS it from H -- `base_hjb.py`'s
        # `Phi_U -= source_term`, on both the batch and per-point paths:
        #     F(u) = (u - u_next)/dt - (sigma^2/2) Lap u + H - S_hjb = 0
        # Consequence for the sign a user must pass:
        #   - A POSITIVE source_term_hjb raises the value / cost-to-go u, so it
        #     acts as a COST: agents are repelled from regions of large S_hjb.
        #   - A NEGATIVE source_term_hjb acts as a reward (attractive).
        # This is the SAME sign as `potential` and `coupling` in SeparableHamiltonian, which H
        # carries as -V - f (#2375 ruling 3).
        # For a repulsive congestion coupling F[m] (penalize crowding), write
        #     source_term_hjb = +gamma * dF/dm   (positive where crowded).
        # The callback receives the density SLICE m(t, .) at time t (Issue
        # #1285; see coupling/source_composition.compose_hjb_source), not the full
        # (Nt+1, Nx) trajectory. See the MFGProblem class docstring
        # "Sign conventions" note for the unified picture.
        self.source_term_hjb: Callable | None = kwargs.pop("source_term_hjb", None)
        self.source_term_fp: Callable | None = kwargs.pop("source_term_fp", None)
        # Bound at acceptance, so a signature that cannot be matched is refused here (#2375 ruling 8)
        for name in ("source_term_hjb", "source_term_fp"):
            if getattr(self, name) is not None:
                bound_attribute(self, name, SOURCE_TERM_SLOTS, role=name)
        self.nonlocal_operator: Any | None = kwargs.pop("nonlocal_operator", None)
        # A soft wall: `state_penalty(x)` is a COST, positive where the region is expensive.
        # It is `alpha`-free and `u`-free, which is what makes it a potential rather than a
        # constraint, and it is composed into H's potential V below. `state_penalty_scale`
        # MULTIPLIES it, where the retired `_penalty_eps` divided: a penalty amplitude is not a
        # regularisation parameter and `eps -> 0` is not its limit.
        #
        # Both are read ONCE, at construction, by the composition below. Assigning either
        # afterwards does nothing -- the closure has already captured them.
        self.state_penalty: Callable | None = kwargs.pop("state_penalty", None)
        self.state_penalty_scale: float = float(kwargs.pop("state_penalty_scale", 1.0))

        if kwargs.pop("obstacle", None) is not None:
            raise NotImplementedError(
                "`obstacle` is RETIRED (#2002). It was documented as the variational inequality "
                "`v >= Psi(x)` and implemented as `max(0, Psi(x))` -- a term with no `v` in it, "
                "byte-identical at a node satisfying the constraint and one violating it. It "
                "penalised POSITION, not VIOLATION, and no scaling changed that.\n\n"
                "It has two successors, because it was conflating two different things:\n"
                "  * A SOFT WALL -- a cost for being somewhere, `alpha`-free and `u`-free. That is "
                "a potential, not a constraint. Pass `state_penalty=Psi_cost` (positive where "
                "expensive); it is composed into the Hamiltonian's potential V, which is cost-signed "
                "too (#2375 ruling 3).\n"
                "  * A REAL CONSTRAINT -- the variational inequality. Pass "
                "`constraint=ObstacleConstraint(psi, 'lower')` to `HJBFDMSolver` (#591). That slot "
                "is reserved and deliberately unfinished: it projects rather than solving the VI, "
                "and only that one solver carries it (#2046).\n\n"
                "Note also that `obstacles` (plural) is a different field entirely -- geometric "
                "regions excluded from the domain -- and was never related to this one."
            )

        # Initialize geometry-related attributes explicitly (Issue #543 - fail-fast principle)
        # These may be set by init methods, but should have explicit defaults
        self.geometry = None  # type: GeometryProtocol | None
        self.hjb_geometry = None  # type: GeometryProtocol | None
        self.fp_geometry = None  # type: GeometryProtocol | None
        self.spatial_shape = None  # type: tuple[int, ...] | None
        self.has_obstacles = False
        self.obstacles = []
        self.geometry_projector = None  # Will be set if dual geometries provided
        self.solver_compatible = {}  # type: dict[str, bool]
        self.solver_recommendations = {}  # type: dict[str, str]

        # Issue #1068: explicit None-init for BC fallback slot (ConditionsMixin.using_resolved_bc).
        # Avoids hasattr() in the fallback path when geometry has no set_boundary_conditions().
        self._temp_resolved_bc = None  # type: BoundaryConditions | None

        if hjb_geometry is not None and fp_geometry is not None:
            # Dual geometry mode: separate geometries for HJB and FP
            if geometry is not None:
                raise ValueError(
                    "Specify EITHER 'geometry' (unified) OR ('hjb_geometry', 'fp_geometry') (dual), not both"
                )
            # Issue #1765: dual geometry is accepted here and then ignored. This branch builds
            # `self.geometry_projector`, but NOTHING on the solver side reads it --
            # `grep -rn "geometry_projector" mfgarchon/` returns hits only inside this file, and
            # nothing under `mfgarchon/alg/` mentions `hjb_geometry` or `fp_geometry`. Below,
            # `_init_geometry(final_hjb_geometry, ...)` sets `self.geometry`, which is the single
            # attribute every solver reads, so the FP solver silently runs on the HJB grid.
            #
            # Measured: a 41-point HJB grid with an 11-point FP grid returned a density whose SPATIAL
            # axis is 41 -- the HJB grid's -- and the FP CFL log reported the HJB grid's dx=0.025.
            # No error, no warning, and
            # every downstream number -- mass, convergence rate, timings -- computed on a grid the
            # caller did not choose.
            #
            # Refusing is the honest state until the projector is wired: had the parameter simply
            # not existed, this call would have raised `TypeError`. `GeometryProjector` itself
            # works and stays usable directly; it is the MFGProblem plumbing that is missing.
            if not _same_geometry(hjb_geometry, fp_geometry):
                raise NotImplementedError(
                    "MFGProblem(hjb_geometry=..., fp_geometry=...) requires the SAME geometry "
                    "object for both (Issue #1765). Two geometries are accepted and a "
                    "GeometryProjector is built, but no solver reads it -- the FP solve would run "
                    "on the HJB grid and return a result that looks like what you asked for.\n"
                    "  - Same discretisation for both? Pass `geometry=` once, or bind one object "
                    "to both names.\n"
                    "  - Genuinely want two resolutions? Drive it yourself:\n"
                    "      from mfgarchon.geometry import GeometryProjector\n"
                    "      projector = GeometryProjector(hjb_geometry=a, fp_geometry=b)\n"
                    "      m_on_fp = projector.project_hjb_to_fp(m_on_hjb)\n"
                    "This compares identity, not equality: three attempts at deciding whether two "
                    "separately-built geometries describe the same discretisation each shipped a "
                    "wrong answer, and over-refusing is recoverable in one edit where a silent "
                    "wrong answer is not."
                )

            # Identical geometries are equivalent to the unified path and are allowed through.
            final_hjb_geometry = hjb_geometry
            final_fp_geometry = fp_geometry
            from mfgarchon.geometry import GeometryProjector

            self.geometry_projector = GeometryProjector(
                hjb_geometry=hjb_geometry, fp_geometry=fp_geometry, projection_method="auto"
            )
        elif hjb_geometry is not None or fp_geometry is not None:
            # Partial dual geometry specification
            raise ValueError("If using dual geometries, both 'hjb_geometry' AND 'fp_geometry' must be specified")
        elif geometry is not None:
            # Unified geometry mode (backward compatible)
            final_hjb_geometry = geometry
            final_fp_geometry = geometry
        else:
            # No explicit geometry provided - will be handled by mode detection
            final_hjb_geometry = None
            final_fp_geometry = None

        # Detect initialization mode
        # For dual geometry, pass the hjb_geometry to mode detection
        geometry_for_detection = final_hjb_geometry if final_hjb_geometry is not None else geometry
        mode = self._detect_init_mode(spatial_bounds=spatial_bounds, geometry=geometry_for_detection, network=network)

        # Dispatch to appropriate initializer
        if mode == "nd_grid":
            # Mode 1: N-dimensional grid
            self._init_grid(spatial_bounds, spatial_discretization, T, Nt, coupling_coefficient, suppress_warnings)

        elif mode == "geometry":
            # Mode 2: Complex geometry
            self._init_geometry(
                final_hjb_geometry,
                obstacles,
                T,
                Nt,
                coupling_coefficient,
                lambda_,
                suppress_warnings,
            )
            # For dual geometry mode, store both geometries explicitly
            if self.geometry_projector is not None:
                self.hjb_geometry = final_hjb_geometry
                self.fp_geometry = final_fp_geometry

        elif mode == "network":
            # Mode 3: Network MFG
            self._init_network(network, T, Nt, coupling_coefficient, lambda_)

        elif mode == "default":
            # Default: 1D unit interval with 51 grid points
            warnings.warn(
                "No spatial domain specified. Using default 1D domain: [0, 1] with 51 points.",
                UserWarning,
                stacklevel=2,
            )
            self._init_grid([(0.0, 1.0)], [51], T, Nt, coupling_coefficient, suppress_warnings)

        else:
            raise ValueError(f"Unknown initialization mode: {mode}")

        # Store dual geometries (Issue #257)
        # For unified mode, both point to self.geometry (set by init methods)
        # For dual mode, these were already set above (lines 406-407)
        # Issue #543: Explicit None check instead of hasattr
        if self.hjb_geometry is None:
            self.hjb_geometry = getattr(self, "geometry", None)
            self.fp_geometry = getattr(self, "geometry", None)

        # Note: has_obstacles and obstacles already initialized explicitly (lines 334-335)
        # Specialized init methods may override these defaults

        # Issue #673: Handle class-based Hamiltonian parameter
        # If hamiltonian= provided without components, create MFGComponents
        if hamiltonian is not None and components is None:
            if isinstance(hamiltonian, HamiltonianBase):
                components = MFGComponents(hamiltonian=hamiltonian)
            else:
                raise TypeError(
                    f"hamiltonian must be a HamiltonianBase instance, got {type(hamiltonian).__name__}.\n"
                    "Use class-based Hamiltonian:\n"
                    "  from mfgarchon.core.hamiltonian import SeparableHamiltonian, QuadraticControlCost\n"
                    "  H = SeparableHamiltonian(control_cost=QuadraticControlCost(...))\n"
                    "  problem = MFGProblem(hamiltonian=H, ...)"
                )

        # Store custom components if provided
        self.components = components
        self.is_custom = components is not None

        # Merge parameters
        if self.is_custom and self.components is not None:
            all_params = {**self.components.parameters, **kwargs}
        else:
            all_params = kwargs

        # Validate kwargs - fail fast on deprecated/unrecognized parameters (Issue #666)
        self._validate_kwargs(all_params)

        # Initialize arrays (Issue #670: unified naming)
        self.u_terminal: NDArray  # Terminal condition u(T, x)
        self.m_initialial: NDArray  # Initial density m(0, x)

        # Initialize functions
        self._initialize_functions(**all_params)

        # A soft wall is a POTENTIAL, so it is composed into H here rather than handed to a
        # solver as a source (#2002). Done after _initialize_functions because that is where the
        # Hamiltonian becomes available.
        if self.state_penalty is not None:
            self._compose_state_penalty_into_potential()

        # Validate custom components if provided
        if self.is_custom:
            self._validate_hamiltonian_components()

        # Detect solver compatibility
        self._detect_solver_compatibility()

    def _init_grid(
        self,
        spatial_bounds: list[tuple[float, float]],
        spatial_discretization: list[int] | None,
        T: float,
        Nt: int,
        coupling_coefficient: float,
        suppress_warnings: bool,
    ) -> None:
        """
        Initialize problem on a tensor-product Cartesian grid.

        Builds a ``TensorProductGrid`` (with default no-flux boundary conditions)
        from ``spatial_bounds`` and the per-dimension interval counts in
        ``spatial_discretization`` and stores the derived grid attributes. Used
        by the ``spatial_bounds=`` (n-D grid) and no-argument (default 1D) paths.

        Args:
            spatial_bounds: List of (min, max) tuples, one per dimension.
            spatial_discretization: Interval count per dimension (Nx); the grid
                allocates Nx + 1 points per axis. Defaults to 51 intervals per
                dimension when None.
            T: Terminal time.
            Nt: Number of time intervals.
            coupling_coefficient: Control cost coefficient.
            suppress_warnings: Skip the computational-feasibility check.
        """
        # Validate inputs
        if not spatial_bounds:
            raise ValueError("spatial_bounds must be a non-empty list of (min, max) tuples")

        dimension = len(spatial_bounds)

        if spatial_discretization is None:
            # Default: 51 intervals per dimension
            spatial_discretization = [51] * dimension
        elif len(spatial_discretization) != dimension:
            raise ValueError(
                f"spatial_discretization must have {dimension} elements (one per dimension), "
                f"got {len(spatial_discretization)}"
            )

        # Create TensorProductGrid for all dimensions (unified approach)
        from mfgarchon.geometry import TensorProductGrid
        from mfgarchon.geometry.boundary import no_flux_bc

        # Convert discretization to Nx_points (add 1 for point count vs intervals)
        Nx_points = [n + 1 for n in spatial_discretization]
        geometry = TensorProductGrid(
            bounds=spatial_bounds, Nx_points=Nx_points, boundary_conditions=no_flux_bc(dimension=dimension)
        )

        # Store geometry for unified interface
        self.geometry = geometry

        # Set dimension from geometry
        self.dimension = geometry.dimension

        # Store n-D parameters
        self.spatial_bounds = spatial_bounds

        # The geometry owns the grid shape, as this comment always said. The three branches that
        # stood here recomputed it: the first two from `spatial_discretization` (agreeing with the
        # geometry by arithmetic, and identical to each other), the third from
        # `num_spatial_points`, which is a COUNT and not a shape -- so at d >= 4 this constructor
        # produced (prod(Nx),) where the `geometry=` constructor produced (Nx, Nx, Nx, Nx). #1888's
        # fork, one dimension further out, and undetectable until a caller checked the shape.
        self.spatial_shape = tuple(geometry.get_grid_shape())

        # Time domain
        self.T: float = T
        self.Nt: int = Nt
        self.dt: float = T / Nt if Nt > 0 else 0.0
        self.tSpace: np.ndarray = np.linspace(0, T, Nt + 1, endpoint=True)

        # Coefficients
        self.coupling_coefficient: float = coupling_coefficient

        # Check computational feasibility and warn if needed
        if not suppress_warnings:
            self._check_computational_feasibility()

    def _check_computational_feasibility(self) -> None:
        """Warn about computational limits for high-dimensional problems."""
        import warnings

        MAX_PRACTICAL_DIMENSION = 4
        MAX_TOTAL_GRID_POINTS = 10_000_000  # 10 million

        # Calculate total grid points
        total_spatial_points = int(np.prod(self.spatial_shape))
        total_points = total_spatial_points * (self.Nt + 1)
        memory_mb = total_points * 8 / (1024**2)  # Assuming float64

        if self.dimension > MAX_PRACTICAL_DIMENSION:
            warnings.warn(
                f"\n{'=' * 80}\n"
                f"HIGH DIMENSION WARNING\n"
                f"{'=' * 80}\n"
                f"Problem dimension: {self.dimension}D\n"
                f"Practical limit for grid-based FDM: {MAX_PRACTICAL_DIMENSION}D\n"
                f"\n"
                f"Grid-based methods scale as O(N^d), becoming impractical for high dimensions.\n"
                f"Your problem will require:\n"
                f"  - Spatial points: {total_spatial_points:,}\n"
                f"  - Total points (space × time): {total_points:,}\n"
                f"  - Estimated memory: {memory_mb:,.1f} MB per array\n"
                f"\n"
                f"RECOMMENDATION:\n"
                f"For dimension > {MAX_PRACTICAL_DIMENSION}, consider alternative methods:\n"
                f"  - Particle-based collocation methods (algorithms/particle_collocation)\n"
                f"  - Network MFG formulations (for very high dimensions)\n"
                f"  - Dimension reduction techniques\n"
                f"\n"
                f"To suppress this warning: MFGProblem(..., suppress_warnings=True)\n"
                f"{'=' * 80}",
                UserWarning,
                stacklevel=3,
            )
        elif total_points > MAX_TOTAL_GRID_POINTS:
            warnings.warn(
                f"\n{'=' * 80}\n"
                f"MEMORY WARNING\n"
                f"{'=' * 80}\n"
                f"Problem requires {total_points:,} grid points ({memory_mb:,.1f} MB per array).\n"
                f"This may cause memory issues on typical machines.\n"
                f"\n"
                f"Consider:\n"
                f"  - Reducing spatial discretization\n"
                f"  - Reducing time steps\n"
                f"  - Using sparse storage methods\n"
                f"\n"
                f"To suppress this warning: MFGProblem(..., suppress_warnings=True)\n"
                f"{'=' * 80}",
                UserWarning,
                stacklevel=3,
            )

    def _detect_init_mode(
        self,
        spatial_bounds: list[tuple[float, float]] | None,
        geometry: GeometryProtocol | None,
        network: Any | None,
    ) -> str:
        """
        Detect which initialization mode to use based on provided parameters.

        Args:
            spatial_bounds: Spatial bounds (or None)
            geometry: Geometry object (or None)
            network: Network object (or None)

        Returns:
            mode: One of "nd_grid", "geometry", "network", "default"

        Raises:
            ValueError: If parameters are ambiguous or conflicting
        """
        # Count how many modes are specified
        mode_indicators = {
            "nd_grid": spatial_bounds is not None,
            "geometry": geometry is not None,
            "network": network is not None,
        }

        num_modes = sum(mode_indicators.values())

        if num_modes == 0:
            return "default"
        elif num_modes > 1:
            specified = [k for k, v in mode_indicators.items() if v]
            raise ValueError(
                f"Ambiguous initialization: Multiple modes specified: {specified}\n"
                f"Provide ONLY ONE of:\n"
                f"  - spatial_bounds (for n-D grid mode)\n"
                f"  - geometry (for complex geometry mode)\n"
                f"  - network (for network MFG mode)"
            )
        else:
            # Exactly one mode specified
            for mode, is_set in mode_indicators.items():
                if is_set:
                    return mode

        # Should never reach here
        return "default"

    def _init_geometry(
        self,
        geometry: GeometryProtocol,
        obstacles: list | None,
        T: float,
        Nt: int,
        coupling_coefficient: float,
        lambda_: float | None,
        suppress_warnings: bool,
    ) -> None:
        """
        Initialize problem with geometry object implementing GeometryProtocol.

        Accepts any geometry type: TensorProductGrid, BaseGeometry,
        ImplicitDomain, NetworkGeometry, etc.

        Args:
            geometry: Any object implementing GeometryProtocol
            obstacles: List of obstacle geometries (for domain geometries)
            T, Nt: Time domain parameters
            coupling_coefficient: Physical parameters
            lambda_: control cost parameter
            suppress_warnings: Suppress warnings
        """
        # Import geometry protocol
        try:
            from mfgarchon.geometry import GeometryProtocol, validate_geometry
        except ImportError as err:
            raise ImportError(
                "Geometry mode requires geometry module. Install with: pip install mfgarchon[geometry]"
            ) from err

        # Validate geometry object implements GeometryProtocol
        if not isinstance(geometry, GeometryProtocol):
            raise TypeError(
                f"geometry must implement GeometryProtocol, got {type(geometry)}. "
                f"Use TensorProductGrid, BaseGeometry, ImplicitDomain, or NetworkGeometry."
            )

        # Validate geometry is properly implemented
        validate_geometry(geometry)

        # Store geometry
        self.geometry = geometry
        self.dimension = geometry.dimension
        self.obstacles = obstacles or []
        self.has_obstacles = len(self.obstacles) > 0

        # Time domain
        self.T = T
        self.Nt = Nt
        self.dt = T / Nt if Nt > 0 else 0.0  # Lowercase (official naming convention)
        self.tSpace = np.linspace(0, T, Nt + 1, endpoint=True)

        # Physical parameters
        self.coupling_coefficient = coupling_coefficient

        # MFG coupling parameters (for custom Hamiltonians)
        self.lambda_ = lambda_

        # Initialize spatial discretization based on geometry type
        from mfgarchon.geometry import GeometryType

        if geometry.geometry_type == GeometryType.CARTESIAN_GRID:
            # CARTESIAN_GRID: Can be TensorProductGrid or AMR mesh
            # Use polymorphic method to get configuration
            config = geometry.get_problem_config()

            # Apply configuration from geometry
            self.num_spatial_points = config["num_spatial_points"]
            self.spatial_shape = config["spatial_shape"]
            self.spatial_bounds = config["spatial_bounds"]

        elif geometry.geometry_type == GeometryType.UNSTRUCTURED_MESH:
            # BaseGeometry - unstructured mesh via Gmsh
            self.mesh_data = geometry.generate_mesh()
            self.collocation_points = self.mesh_data.vertices
            self.num_spatial_points = len(self.collocation_points)

            # Set spatial shape and bounds
            self.spatial_shape = (self.num_spatial_points,)  # Unstructured
            self.spatial_bounds = None  # Not a regular grid

        elif geometry.geometry_type == GeometryType.IMPLICIT:
            # ImplicitDomain - point cloud from SDF
            self.num_spatial_points = geometry.num_spatial_points
            self.collocation_points = geometry.get_spatial_grid()
            self.spatial_shape = (self.num_spatial_points,)
            self.spatial_bounds = geometry.get_bounding_box()

        elif geometry.geometry_type in (GeometryType.MAZE, GeometryType.NETWORK):
            # Graph-based geometries (mazes, networks)
            config = geometry.get_problem_config()
            self.num_spatial_points = config["num_spatial_points"]
            self.collocation_points = geometry.get_spatial_grid()
            self.spatial_shape = config["spatial_shape"]
            self.spatial_bounds = config.get("spatial_bounds")

            # Store graph-specific data if available
            if "graph_data" in config:
                self.graph_data = config["graph_data"]

        else:
            # Generic GeometryProtocol object - extract config
            # Issue #557 fix: Extract spatial_bounds from get_problem_config()
            # to support geometries like PointCloudGeometry that provide bounds
            config = geometry.get_problem_config()
            self.num_spatial_points = config["num_spatial_points"]
            self.collocation_points = geometry.get_spatial_grid()
            self.spatial_shape = config["spatial_shape"]
            self.spatial_bounds = config.get("spatial_bounds")

    def _init_network(
        self,
        network: Any,
        T: float,
        Nt: int,
        coupling_coefficient: float,
        lambda_: float | None,
    ) -> None:
        """
        Initialize problem on network/graph.

        Args:
            network: NetworkGraph or networkx.Graph
            T, Nt: Time domain parameters
            coupling_coefficient: Physical parameters
        """
        # Import CustomNetwork for geometry-first API
        from mfgarchon.geometry.graph import CustomNetwork

        # Store network
        self.network = network
        self.dimension = "network"  # Special dimension indicator

        # Create CustomNetwork geometry from the network
        try:
            import networkx as nx

            if isinstance(network, nx.Graph):
                # Create geometry from networkx graph
                geometry = CustomNetwork.from_networkx(network)
                self.num_nodes = network.number_of_nodes()
                self.adjacency_matrix = nx.adjacency_matrix(network).toarray()
            else:
                # Assume custom NetworkGraph type with adjacency_matrix attribute
                self.num_nodes = len(network.nodes)
                self.adjacency_matrix = network.adjacency_matrix
                # Create geometry from adjacency matrix
                geometry = CustomNetwork(network.adjacency_matrix)
        except ImportError:
            # NetworkX not available - assume custom type
            self.num_nodes = len(network.nodes)
            self.adjacency_matrix = network.adjacency_matrix
            # Create geometry from adjacency matrix
            geometry = CustomNetwork(network.adjacency_matrix)

        # Store geometry (geometry-first API: never None)
        self.geometry = geometry

        # Time domain
        self.T = T
        self.Nt = Nt
        self.dt = T / Nt if Nt > 0 else 0.0  # Lowercase (official naming convention)
        self.tSpace = np.linspace(0, T, Nt + 1, endpoint=True)

        # Physical parameters
        self.coupling_coefficient = coupling_coefficient

        # MFG coupling parameters (for custom Hamiltonians)
        self.lambda_ = lambda_

        # Spatial discretization (nodes)
        self.spatial_shape = (self.num_nodes,)
        self.num_spatial_points = self.num_nodes  # For networks, spatial points = nodes
        self.spatial_bounds = None
        self.obstacles = None
        self.has_obstacles = False

    # =========================================================================
    # Geometry Type Helper Properties (Phase 2 of Issue #435)
    # =========================================================================

    @property
    def domain_type(self) -> str:
        """Domain type derived from geometry (Issue #794).

        Returns a string classification of the domain:
        - ``"grid"`` for Cartesian tensor-product grids
        - ``"mesh"`` for unstructured 2D/3D meshes
        - ``"implicit"`` for implicit/SDF-based domains
        - ``"network"``, ``"maze"``, ``"custom"`` for other geometry types
        """
        from mfgarchon.geometry import GeometryType

        gt = self.geometry.geometry_type
        if gt == GeometryType.CARTESIAN_GRID:
            return "grid"
        elif gt == GeometryType.UNSTRUCTURED_MESH:
            return "mesh"
        elif gt == GeometryType.IMPLICIT:
            return "implicit"
        else:
            return str(gt.value)  # "network", "maze", "custom"

    @property
    def is_network(self) -> bool:
        """
        Check if this problem is defined on a network/graph domain.

        Returns:
            True if domain_type is "network", False otherwise.

        Example:
            >>> import networkx as nx
            >>> G = nx.grid_2d_graph(5, 5)
            >>> problem = MFGProblem(network=G, T=1.0, Nt=10)
            >>> problem.is_network
            True
        """
        return self.domain_type == "network"

    @property
    def is_cartesian(self) -> bool:
        """
        Check if this problem is defined on a Cartesian grid domain.

        Returns:
            True if domain_type is "grid", False otherwise.

        Example:
            >>> grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51])
            >>> problem = MFGProblem(geometry=grid, T=1.0, Nt=10)
            >>> problem.is_cartesian
            True
        """
        return self.domain_type == "grid"

    @property
    def is_implicit(self) -> bool:
        """
        Check if this problem uses an implicit/complex geometry.

        Returns:
            True if domain_type is "implicit", False otherwise.

        Example:
            >>> from mfgarchon.geometry import ImplicitDomain
            >>> domain = ImplicitDomain(...)  # Complex geometry
            >>> problem = MFGProblem(geometry=domain, T=1.0, Nt=10)
            >>> problem.is_implicit
            True
        """
        return self.domain_type == "implicit"

    # =========================================================================
    # Physical Parameter Properties (#811, #2375 ruling 6)
    # =========================================================================

    @property
    def volatility(self) -> float | np.ndarray | Callable:
        """The SDE volatility Sigma, as supplied: a float, an array, or a callable Sigma(t, x, m).

        Never collapsed to a scalar (#2376). A consumer that can only use a scalar calls
        ``scalar_volatility(problem.volatility, consumer=...)``, which refuses an array or a callable.
        """
        return self._volatility

    @property
    def volatility_kind(self) -> str | None:
        """How an array volatility is read: ``"field"``, ``"tensor"``, or ``None`` for a scalar."""
        return self._volatility_kind

    @property
    def diffusion(self) -> float | np.ndarray | Callable:
        """The PDE diffusion A = 1/2 Sigma Sigma^T, derived by the one converter.

        Same form as ``volatility``: a float (D = sigma^2/2), an array read under
        ``volatility_kind``, or a callable A(t, x, m). For one scalar D, a consumer calls
        ``diffusion_from_volatility(scalar_volatility(problem.volatility, consumer=...))``.
        """
        from mfgarchon.utils.pde_coefficients import diffusion_from_volatility

        volatility, kind = self._volatility, self._volatility_kind
        if callable(volatility):

            def diffusion(t, x, m):
                value = volatility(t, x, m)
                output_kind = _callable_output_kind(value, kind, self.dimension, _evaluation_shape(x, m))
                return diffusion_from_volatility(value, kind=output_kind)

            return diffusion
        return diffusion_from_volatility(volatility, kind=kind)

    @property
    def sigma(self):
        raise AttributeError(_RETIRED_VOLATILITY_ATTRIBUTE.format(name="sigma"))

    @property
    def volatility_field(self):
        raise AttributeError(_RETIRED_VOLATILITY_ATTRIBUTE.format(name="volatility_field"))

    @property
    def diffusion_field(self):
        raise AttributeError(_RETIRED_VOLATILITY_ATTRIBUTE.format(name="diffusion_field"))

    @property
    def spatial_discretization(self):
        raise AttributeError(
            "problem.spatial_discretization is retired (#1889): it held the interval count when the "
            "problem was built from spatial_bounds= and the point count when it was built from a "
            "TensorProductGrid passed as geometry=. "
            "Space is the domain's: problem.geometry.Nx counts intervals and problem.geometry.Nx_points "
            "counts points, Nx_points = Nx + 1 per axis."
        )

    # =========================================================================
    # Hamiltonian Properties (Issue #673)
    # =========================================================================

    @property
    def hamiltonian_class(self) -> Any | None:
        """
        Get the class-based Hamiltonian object if available.

        Returns the HamiltonianBase instance for direct access to:
        - H(t, x, p, m): Hamiltonian value
        - dp(t, x, p, m): ∂H/∂p (optimal control)
        - dm(t, x, p, m): ∂H/∂m (density coupling)
        - optimal_control(t, x, p, m): α* = ±∂H/∂p

        Returns:
            HamiltonianBase instance, or None if using function-based API

        Example:
            >>> from mfgarchon.core.hamiltonian import SeparableHamiltonian
            >>> H = SeparableHamiltonian(...)
            >>> problem = MFGProblem(hamiltonian=H, ...)
            >>> problem.hamiltonian_class.dp(t, x, p, m)  # Direct access
        """
        if self.components is not None:
            return getattr(self.components, "_hamiltonian_class", None)
        return None

    @property
    def lagrangian_class(self) -> Any | None:
        """
        Get the class-based Lagrangian object if available.

        Returns the LagrangianBase instance for direct access to:
        - L(t, x, alpha, m): Running cost value
        - optimal_control(t, x, p, m): alpha* (same as HamiltonianBase)
        - evaluate_hamiltonian(t, x, p, m): H value on-the-fly
        - proximal(tau, z): For ADMM/variational solvers

        Issue #899: LagrangianBase as first-class specification.

        Returns:
            LagrangianBase instance, or None
        """
        if self.components is not None:
            return getattr(self.components, "_lagrangian_class", None)
        return None

    # =========================================================================
    # Time Grid Properties
    # =========================================================================

    @property
    def Nt_points(self) -> int:
        """
        Number of time grid points (Nt + 1).

        Nt is the number of time intervals, while Nt_points is the number
        of time grid points including both endpoints.

        Returns:
            Nt + 1 (number of time points)

        Example:
            >>> problem = MFGProblem(geometry=domain, T=1.0, Nt=10)
            >>> problem.Nt         # 10 intervals
            10
            >>> problem.Nt_points  # 11 points
            11
        """
        return self.Nt + 1

    # =========================================================================
    # Internal Geometry Helpers (no deprecation warnings)
    # These are for internal use only - external code should use geometry directly
    # =========================================================================

    def _get_domain_length(self) -> float | None:
        """Get domain length for 1D problems (internal use, no warning)."""
        if self.geometry is not None and self.dimension == 1:
            bounds = self.geometry.get_bounds()
            if bounds is not None:
                return float(bounds[1][0] - bounds[0][0])
        return None

    def _get_spacing(self) -> float | None:
        """Get grid spacing for 1D problems (internal use, no warning)."""
        if self.geometry is not None and self.dimension == 1:
            from mfgarchon.geometry.base import CartesianGrid

            if isinstance(self.geometry, CartesianGrid):
                return float(self.geometry.get_grid_spacing()[0])
            else:
                bounds = self.geometry.get_bounds()
                if bounds is not None:
                    n_points = self.geometry.num_spatial_points
                    if n_points > 1:
                        return float((bounds[1][0] - bounds[0][0]) / (n_points - 1))
        return None

    def _get_num_intervals(self) -> int | None:
        """Get number of intervals for 1D problems (internal use, no warning)."""
        if self.geometry is not None and self.dimension == 1:
            return self.geometry.num_spatial_points - 1
        return None

    def _get_spatial_grid_internal(self) -> np.ndarray | None:
        """Get spatial grid array (internal use, no warning)."""
        if self.geometry is not None:
            return self.geometry.get_spatial_grid()
        return None

    # =========================================================================
    # PDE Coefficient Field Helpers
    # =========================================================================

    def get_diffusion_coefficient_field(
        self,
        override: float | np.ndarray | Callable | None = None,
        *,
        field_name: str = "diffusion",
        dimension: int | None = None,
    ) -> Any:
        """
        Get a CoefficientField wrapper for the diffusion coefficient.

        Returns a CoefficientField that handles scalar, array, and callable
        volatilities uniformly (Issue #1412). What it wraps is the VOLATILITY Sigma, not the
        diffusion; the evaluated value goes through ``diffusion_from_volatility``.

        Precedence -- the single source for the solver-side volatility lookup: a per-solve
        ``override`` (the volatility a solver receives) wins; otherwise ``self.volatility``, as
        supplied. There is no scalar fallback (#2376).

        Args:
            override: Per-solve volatility override (a solver's ``volatility`` argument, already
                resolved to a scalar or a ``field`` kind). ``None`` uses ``self.volatility``.
            field_name: Name used in CoefficientField diagnostics (default ``"diffusion"``;
                solvers pass ``"volatility"`` to preserve their error wording).
            dimension: Spatial dimension for array/spatiotemporal extraction; defaults to
                ``self.dimension``.

        Returns:
            CoefficientField wrapping the resolved volatility

        Example:
            >>> diffusion = problem.get_diffusion_coefficient_field()
            >>> sigma_at_t = diffusion.evaluate_at(
            ...     timestep_idx=5,
            ...     grid=x_coords,
            ...     density=m,
            ...     dt=problem.dt
            ... )
        """
        from mfgarchon.utils.pde_coefficients import CoefficientField

        return CoefficientField(
            field=override if override is not None else self._volatility,
            default_value=0.0,  # read only when `field` is None, which it never is here
            field_name=field_name,
            dimension=self.dimension if dimension is None else dimension,
        )

    def get_drift_coefficient_field(self) -> Any:
        """
        Get a CoefficientField wrapper for the drift field.

        Returns a CoefficientField that handles array and callable drift
        coefficients uniformly. Use this in solvers instead of directly
        accessing self.drift_field.

        Returns:
            CoefficientField wrapping self.drift_field (default is zero drift)

        Example:
            >>> drift = problem.get_drift_coefficient_field()
            >>> alpha_at_t = drift.evaluate_at(
            ...     timestep_idx=5,
            ...     grid=x_coords,
            ...     density=m,
            ...     dt=problem.dt
            ... )
        """
        from mfgarchon.utils.pde_coefficients import CoefficientField

        # Default drift is zero
        default_drift = 0.0

        return CoefficientField(
            field=self.drift_field,
            default_value=default_drift,
            field_name="drift",
            dimension=self.dimension,
        )

    def has_state_dependent_coefficients(self) -> bool:
        """
        Check if problem has state-dependent (callable) PDE coefficients.

        Solvers may need to handle callable coefficients differently from
        constant/precomputed ones (e.g., re-evaluate at each timestep).

        Returns:
            True if the volatility or drift_field is callable
        """
        return callable(self._volatility) or callable(self.drift_field)

    def __repr__(self) -> str:
        """
        Return string representation using geometry-first API.

        Avoids accessing deprecated attributes to prevent DeprecationWarning
        spam in Jupyter notebooks and debuggers.
        """
        # Use geometry for spatial info, not deprecated attrs
        geom_type = type(self.geometry).__name__ if self.geometry else "None"
        dim = self.dimension

        vol = self._volatility
        if callable(vol):
            vol_repr = "<callable>"
        elif np.ndim(vol) != 0:
            vol_repr = f"<array {np.shape(vol)}, {self._volatility_kind}>"
        else:
            vol_repr = repr(vol)
        return f"MFGProblem(geometry={geom_type}, dim={dim}, T={self.T}, Nt={self.Nt}, volatility={vol_repr})"

    def __getstate__(self) -> dict[str, Any]:
        """
        Get state for pickling.

        Returns the instance __dict__ for standard pickle behavior.
        """
        return self.__dict__.copy()

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Restore state from pickle with legacy migration support.

        Handles legacy pickle files where geometry=None but legacy
        attributes (xmin, xmax, Nx) are present. Reconstructs geometry
        from these attributes for backward compatibility.
        """
        # Pickled before #2375 ruling 6: the volatility sat on `volatility_field`, beside a collapsed
        # scalar `sigma` that is not carried over (#2376).
        if "_volatility" not in state and "volatility_field" in state:
            volatility = state.pop("volatility_field")
            state["_volatility"] = volatility
            # The old constructor admitted only grid-shaped fields, so a legacy array is one.
            state["_volatility_kind"] = "field" if isinstance(volatility, np.ndarray) and volatility.ndim > 0 else None
            state.pop("sigma", None)

        # Detect legacy format: geometry=None but has legacy 1D attrs
        if state.get("geometry") is None and state.get("xmin") is not None:
            try:
                from mfgarchon.geometry import TensorProductGrid
                from mfgarchon.geometry.boundary import no_flux_bc

                # Reconstruct geometry from legacy attributes
                xmin = state.get("xmin")
                xmax = state.get("xmax")
                Nx = state.get("Nx")

                if xmin is None or xmax is None or Nx is None:
                    raise KeyError("Missing required legacy fields (xmin, xmax, Nx)")

                # Handle both scalar and list forms
                if isinstance(xmin, (int, float)):
                    bounds = [(float(xmin), float(xmax))]
                    Nx_points = [int(Nx) + 1]
                else:
                    bounds = list(zip(xmin, xmax, strict=True))
                    Nx_points = [n + 1 for n in Nx]

                # Use default no_flux_bc for legacy pickle migration (Issue #674)
                dimension = len(bounds)
                state["geometry"] = TensorProductGrid(
                    bounds=bounds,
                    Nx_points=Nx_points,
                    boundary_conditions=no_flux_bc(dimension=dimension),
                )
            except (KeyError, ImportError) as e:
                import warnings

                warnings.warn(
                    f"Unable to migrate legacy pickle format: {e}. "
                    "This pickle file may be from an incompatible version. "
                    "Consider recreating the MFGProblem.",
                    UserWarning,
                    stacklevel=2,
                )

        self.__dict__.update(state)

    def _detect_solver_compatibility(self) -> None:
        """
        Detect which solver types are compatible with this problem.

        Sets:
            self.solver_compatible: List of compatible solver type strings
            self.solver_recommendations: Dict mapping use cases to solvers

        Called automatically after initialization.
        """
        compatible = []
        recommendations = {}

        # Get problem characteristics
        is_grid = self.domain_type == "grid"
        is_implicit = self.domain_type == "implicit"
        is_network = self.domain_type == "network"
        dim = self.dimension if isinstance(self.dimension, int) else None

        # FDM: Requires regular grid, no complex geometry, works best for dim <= 3
        if is_grid and not self.has_obstacles:
            compatible.append("fdm")
            if dim and dim <= 2:
                recommendations["fast"] = "fdm"
                recommendations["accurate"] = "fdm"

        # Semi-Lagrangian: Works with grids, especially good for higher dimensions
        if is_grid:
            compatible.append("semi_lagrangian")
            if dim and dim >= 3:
                recommendations["fast"] = "semi_lagrangian"

        # GFDM: Works with grids and complex geometry (particle collocation)
        if is_grid or is_implicit:
            compatible.append("gfdm")
            if is_implicit or self.has_obstacles:
                recommendations["obstacles"] = "gfdm"
                recommendations["complex_geometry"] = "gfdm"

        # Particle methods: Work with everything except pure networks
        if not is_network:
            compatible.append("particle")
            if dim and dim >= 4:
                recommendations["high_dimensional"] = "particle"
                recommendations["fast"] = "particle"

        # Network solver: Only for network problems
        if is_network:
            compatible.append("network_solver")
            recommendations["default"] = "network_solver"

        # DGM: Works with grids (experimental)
        if is_grid:
            compatible.append("dgm")

        # PINN: Works with everything (deep learning approach)
        compatible.append("pinn")
        if dim and dim >= 5:
            recommendations["very_high_dimensional"] = "pinn"

        # Set attributes
        self.solver_compatible = compatible
        self.solver_recommendations = recommendations

        # Set default recommendation
        if "default" not in recommendations:
            if is_grid and dim and dim <= 2:
                recommendations["default"] = "fdm"
            elif is_grid and dim and dim == 3:
                recommendations["default"] = "semi_lagrangian"
            elif is_implicit:
                recommendations["default"] = "gfdm"
            elif compatible:
                recommendations["default"] = compatible[0]

    def validate_solver_type(self, solver_type: str) -> None:
        """
        Validate that solver type is compatible with this problem.

        Args:
            solver_type: Solver type identifier (e.g., "fdm", "gfdm", "particle")

        Raises:
            ValueError: If solver type is incompatible with problem configuration

        Note:
            This method is called by solver constructors to provide early
            error detection with helpful messages.
        """
        # Compatibility should already be detected in __init__
        # If empty, initialization failed - raise explicit error
        if not self.solver_compatible:
            raise RuntimeError("Solver compatibility not detected. This indicates __init__ didn't complete properly.")

        if solver_type not in self.solver_compatible:
            # Build helpful error message
            reason = self._get_incompatibility_reason(solver_type)
            suggestion = self._get_solver_suggestion()

            raise ValueError(
                f"Solver type '{solver_type}' is incompatible with this problem.\n\n"
                f"Problem Configuration:\n"
                f"  Domain type: {self.domain_type}\n"
                f"  Dimension: {self.dimension}\n"
                f"  Has obstacles: {self.has_obstacles}\n\n"
                f"Reason: {reason}\n\n"
                f"Compatible solvers: {self.solver_compatible}\n\n"
                f"Suggestion: {suggestion}"
            )

    def _get_incompatibility_reason(self, solver_type: str) -> str:
        """Get human-readable reason why solver is incompatible."""
        reasons = {
            "fdm": {
                "implicit": "FDM requires regular grid, not implicit geometry",
                "network": "FDM requires spatial grid, not network structure",
                "obstacles": "FDM doesn't support obstacles (use GFDM instead)",
            },
            "semi_lagrangian": {
                "implicit": "Semi-Lagrangian requires regular grid",
                "network": "Semi-Lagrangian requires spatial grid",
            },
            "gfdm": {
                "network": "GFDM requires spatial coordinates, not network structure",
            },
            "particle": {
                "network": "Particle methods require spatial domain",
            },
            "network_solver": {
                "grid": "Network solver requires network structure, not spatial grid",
                "implicit": "Network solver requires network structure",
            },
        }

        domain_reasons = reasons.get(solver_type, {})
        return domain_reasons.get(self.domain_type, "Solver not compatible with problem configuration")

    def _get_solver_suggestion(self) -> str:
        """Get helpful suggestion for which solver to use."""
        if not self.solver_recommendations:
            if self.solver_compatible:
                return f"Try using: {self.solver_compatible[0]}"
            return "No compatible solvers found for this configuration"

        # Get default recommendation
        default_solver = self.solver_recommendations.get(
            "default", self.solver_compatible[0] if self.solver_compatible else None
        )

        if not default_solver:
            return "No solver recommendations available"

        # Build recommendation text
        suggestion = f"Use solver '{default_solver}' (recommended for this problem)"

        # Add context-specific recommendations
        additional_recs = []
        if "obstacles" in self.solver_recommendations:
            additional_recs.append(f"obstacles: {self.solver_recommendations['obstacles']}")
        if "fast" in self.solver_recommendations and self.solver_recommendations["fast"] != default_solver:
            additional_recs.append(f"fastest: {self.solver_recommendations['fast']}")
        if "accurate" in self.solver_recommendations and self.solver_recommendations["accurate"] != default_solver:
            additional_recs.append(f"most accurate: {self.solver_recommendations['accurate']}")

        if additional_recs:
            suggestion += f"\n  Alternative recommendations: {', '.join(additional_recs)}"

        suggestion += "\n  Or use create_fast_solver() for automatic selection"

        return suggestion

    def get_solver_info(self) -> dict[str, Any]:
        """
        Get comprehensive solver compatibility information.

        Returns:
            Dictionary with solver compatibility details:
            - compatible: List of compatible solver types
            - recommendations: Dict of use-case specific recommendations
            - dimension: Problem dimension
            - domain_type: Type of spatial domain
            - complexity: Estimated computational complexity
        """
        # Compatibility should already be detected in __init__
        if not self.solver_compatible:
            raise RuntimeError("Solver compatibility not detected. This indicates __init__ didn't complete properly.")

        return {
            "compatible": self.solver_compatible,
            "recommendations": self.solver_recommendations,
            "dimension": self.dimension,
            "domain_type": self.domain_type,
            "has_obstacles": self.has_obstacles,
            "complexity": self._estimate_complexity(),
            "default_solver": self.solver_recommendations.get("default", None),
        }

    def _estimate_complexity(self) -> str:
        """Estimate computational complexity category."""
        if self.domain_type == "network":
            return "O(N_nodes × N_time)"

        if isinstance(self.dimension, int):
            if self.dimension == 1:
                return "O(Nx × Nt)"
            elif self.dimension == 2:
                return "O(Nx × Ny × Nt)"
            elif self.dimension == 3:
                return "O(Nx × Ny × Nz × Nt)"
            else:
                return f"O(N^{self.dimension} × Nt) - curse of dimensionality"

        return "Problem-dependent"

    def get_computational_cost_estimate(self) -> dict:
        """
        Get estimated computational cost for the problem.

        Returns:
            Dictionary with cost estimates:
            - total_spatial_points: Total spatial grid points
            - total_points: Total grid points (space × time)
            - memory_per_array_mb: Memory per solution array (MB)
            - estimated_memory_mb: Total estimated memory (MB)
            - is_feasible: Whether problem is computationally feasible
            - warnings: List of warnings about computational costs
        """
        total_spatial_points = int(np.prod(self.spatial_shape))
        total_points = total_spatial_points * (self.Nt + 1)
        memory_per_array_mb = total_points * 8 / (1024**2)
        estimated_total_mb = memory_per_array_mb * 10  # Rough estimate: ~10 arrays

        warnings_list = []
        is_feasible = True

        if self.dimension > 4:
            warnings_list.append(f"Dimension {self.dimension}D exceeds practical limit (4D)")
            is_feasible = False

        if total_points > 10_000_000:
            warnings_list.append(f"Total points ({total_points:,}) exceeds recommended limit (10M)")
            is_feasible = False

        if estimated_total_mb > 1000:
            warnings_list.append(f"Estimated memory ({estimated_total_mb:.1f} MB) may be excessive")

        return {
            "dimension": self.dimension,
            "spatial_shape": self.spatial_shape,
            "total_spatial_points": total_spatial_points,
            "total_points": total_points,
            "memory_per_array_mb": memory_per_array_mb,
            "estimated_memory_mb": estimated_total_mb,
            "is_feasible": is_feasible,
            "warnings": warnings_list,
        }

    # Issue #670/#671: Legacy default functions removed (Fail Fast principle)
    # - _potential(): Removed - zero potential is now the explicit default
    # - _u_final(): Removed - must be provided via MFGComponents.u_terminal
    # - _m_initial(): Removed - must be provided via MFGComponents.m_initial

    def _compose_state_penalty_into_potential(self) -> None:
        """Fold ``state_penalty`` into the Hamiltonian's potential, with the layer's sign.

        ``state_penalty`` is COST-signed: positive where the region is expensive. So is the
        Hamiltonian's ``potential`` (#2375 ruling 3), so the composition **adds**::

            V_new(x, t) = V_old(x, t) + scale * state_penalty(x)

        Measured on a Gaussian wall at x = 0.5 with no coupling: a potential amplitude of ``+5``
        raises ``u(0, mid)`` to ``+0.524`` while ``-5`` lowers it to ``-1.319``. The composition
        exists in one place so the sign is written once.

        COPIES rather than rebuilds. An earlier version constructed a fresh
        ``SeparableHamiltonian`` from five of its six constructor parameters, which silently
        dropped ``population_index`` (live: a multi-population problem keyed to population k
        became population 0) and downgraded a ``QuadraticMFGHamiltonian`` to its base class.
        ``copy.copy`` preserves the type and every attribute, and leaves the caller's own object
        unmutated.

        SHAPE. The composed closure returns exactly what the base potential returns, so a
        vectorised base stays vectorised -- ``SeparableHamiltonian`` probes the potential on a
        batch to decide whether it can call it that way, and a composition that changed the
        returned shape defeated that probe. With no base there is nothing to match, so the wall
        alone is squeezed: the per-point call sites wrap it in ``float()``, and in 1-D ``x``
        arrives as ``array([0.3])``.

        The LAGRANGIAN is kept in step. ``MFGComponents`` snapshots the potential into
        ``_lagrangian_class`` at construction, before this runs, and
        ``HJBSemiLagrangianSolver`` reads that copy whenever the control cost is non-smooth --
        so composing into the Hamiltonian alone left a solver silently unwalled, which is the
        defect #2002 is about, in a second channel.

        Raises rather than silently skipping when the Hamiltonian cannot carry a potential.
        """
        import copy as _copy

        from mfgarchon.core.hamiltonian import SeparableHamiltonian, SeparableLagrangian

        hamiltonian = getattr(self.components, "_hamiltonian_class", None)
        if not isinstance(hamiltonian, SeparableHamiltonian):
            raise NotImplementedError(
                f"state_penalty needs a Hamiltonian that carries a potential; this problem has "
                f"{type(hamiltonian).__name__}. A soft wall is a potential term V(x), so it can "
                f"only be composed where V lives. Either build the problem with a "
                f"SeparableHamiltonian, or fold the penalty into your own Hamiltonian's potential "
                f"directly: `potential` is cost-signed, so the penalty is added to it (#2002, "
                f"#2375 ruling 3)."
            )

        previous = getattr(hamiltonian, "_potential", None)
        if previous is not None:
            previous = bind_user_callable(previous, POTENTIAL_SLOTS, role="potential")
        penalty = self.state_penalty
        scale = self.state_penalty_scale

        def composed_potential(t: float, x: Any) -> Any:
            import numpy as _np

            wall = scale * _np.asarray(penalty(x), dtype=float)
            if previous is None:
                squeezed = wall.squeeze()
                return float(squeezed) if squeezed.ndim == 0 else squeezed
            base = previous(t=t, x=x)
            # Match the base's own contract exactly -- it decides the shape, not this wrapper.
            if _np.size(wall) == _np.size(base):
                wall = _np.reshape(wall, _np.shape(base))
            return base + wall

        composed = _copy.copy(hamiltonian)
        composed._potential = composed_potential
        self.components._hamiltonian_class = composed

        # Keep every other holder of the potential in step, or a solver reads the unwalled copy.
        # `isinstance`, not `hasattr`: a Lagrangian that carries no potential is a case this does
        # not know how to compose into, and skipping it silently is the very defect being fixed --
        # `HJBSemiLagrangianSolver` would then read an unwalled copy and report a clean answer to
        # a different problem. `SeparableLagrangian` always sets `_potential`; anything else says
        # so out loud.
        lagrangian = getattr(self.components, "_lagrangian_class", None)
        if isinstance(lagrangian, SeparableLagrangian):
            walled_lagrangian = _copy.copy(lagrangian)
            walled_lagrangian._potential = composed_potential
            self.components._lagrangian_class = walled_lagrangian
        elif lagrangian is not None:
            raise NotImplementedError(
                f"state_penalty composed into the Hamiltonian's potential, but this problem's "
                f"Lagrangian is a {type(lagrangian).__name__}, which this composition does not "
                f"know how to fold a potential into. `HJBSemiLagrangianSolver` reads the "
                f"Lagrangian whenever the control cost is non-smooth, so leaving it unwalled "
                f"would silently solve a different problem (#2002). Fold the penalty into that "
                f"Lagrangian's own running cost instead -- there it is COST-signed and enters "
                f"with a plus, the opposite of the Hamiltonian's V."
            )
        if getattr(self.components, "hamiltonian", None) is hamiltonian:
            self.components.hamiltonian = composed

    def _initialize_functions(self, **kwargs: Any) -> None:
        """Initialize the initial density and the terminal value function.

        Issue #670: u_terminal/m_initial must be provided via MFGComponents.
        No silent defaults - Fail Fast principle.
        """
        # Initialize arrays with correct shape for both 1D and n-D
        self.u_terminal = np.zeros(self.spatial_shape)
        self.m_initial = np.zeros(self.spatial_shape)

        # Issue #670: u_terminal and m_initial MUST come from MFGComponents
        has_components = self.components is not None
        has_u_terminal = has_components and self.components.u_terminal is not None
        has_m_initial = has_components and self.components.m_initial is not None

        # Issue #681: Validate IC/BC compatibility before setup
        if has_components and self.geometry is not None:
            from mfgarchon.utils.validation import ValidationError, validate_components

            result = validate_components(
                self.components,
                self.geometry,
                require_m_initial=True,
                require_u_terminal=True,
                terminal_time=float(self.tSpace[-1]),
            )
            if not result.is_valid:
                raise ValidationError(result)

        # Issue #686: Validate custom functions (Hamiltonian, derivatives)
        # Issue #1642 (C1): the derivative-consistency check runs here (it is on by
        # default) and gates only on an exhibited witness, so a wrong dH_dm/dH_dp is
        # refused at construction instead of silently steering every Picard step.
        if has_components and self.geometry is not None:
            from mfgarchon.utils.validation import validate_custom_functions

            h_class = self.components._hamiltonian_class
            if h_class is not None:
                func_result = validate_custom_functions(
                    hamiltonian=h_class,
                    dH_dm=h_class.dm,
                    dH_dp=h_class.dp,
                    geometry=self.geometry,
                )
                if not func_result.is_valid:
                    raise ValidationError(func_result)

            # Validate drift if callable
            if callable(self.drift_field):
                from mfgarchon.utils.validation import validate_drift

                drift_result = validate_drift(self.drift_field, self.geometry)
                if not drift_result.is_valid:
                    raise ValidationError(drift_result)

        # Issue #687: Validate array-type diffusion/drift fields
        if self.geometry is not None and self.spatial_shape is not None:
            from mfgarchon.utils.validation import (
                ValidationResult,
                validate_array_dtype,
                validate_field_shape,
                validate_finite,
            )

            # Validate an array volatility against the shape its kind promises (#2375 ruling 6)
            vol = self._volatility
            if isinstance(vol, np.ndarray) and vol.ndim > 0:
                if self._volatility_kind == "tensor":
                    # Sigma is (d, k), constant or per grid point: grid axes lead, matrix axes trail.
                    d, grid = self.dimension, tuple(self.spatial_shape)
                    if d == 1:
                        raise ValueError(
                            "volatility_kind='tensor' on a 1-D problem: no 1-D solver reads a tensor "
                            "volatility, and a (1, k) noise matrix is the scalar sqrt(sum_k Sigma_1k^2) -- "
                            "pass that as volatility=."
                        )
                    if vol.ndim < 2 or vol.shape[-2] != d or vol.shape[:-2] not in ((), grid):
                        raise ValueError(
                            f"volatility_kind='tensor' needs shape (d, k) or (*spatial_shape, d, k) with "
                            f"d = {d} and spatial_shape = {grid}; got {vol.shape}."
                        )
                    if vol.shape[-1] != d:
                        raise ValueError(
                            f"volatility_kind='tensor' with {vol.shape[-1]} noise sources on a {d}-D problem: "
                            "no solver reads a non-square noise matrix yet. The symmetric (d, d) square root "
                            "of Sigma Sigma^T has the same diffusion A = 1/2 Sigma Sigma^T -- pass that "
                            "(volatility_from_diffusion(A, kind='tensor') builds it)."
                        )
                    shape_check = []
                else:
                    shape_check = [validate_field_shape(vol, self.spatial_shape, "volatility")]
                arr_result = ValidationResult()
                for check in [
                    validate_array_dtype(vol, "volatility"),
                    *shape_check,
                    validate_finite(vol, "volatility"),
                ]:
                    arr_result.issues.extend(check.issues)
                    if not check.is_valid:
                        arr_result.is_valid = False
                if not arr_result.is_valid:
                    raise ValidationError(arr_result)

            # Validate drift_field if ndarray
            if isinstance(self.drift_field, np.ndarray):
                arr_result = ValidationResult()
                for check in [
                    validate_array_dtype(self.drift_field, "drift_field"),
                    validate_field_shape(self.drift_field, self.spatial_shape, "drift_field"),
                    validate_finite(self.drift_field, "drift_field"),
                ]:
                    arr_result.issues.extend(check.issues)
                    if not check.is_valid:
                        arr_result.is_valid = False
                if not arr_result.is_valid:
                    raise ValidationError(arr_result)

        # === u_terminal: MUST be in MFGComponents (Issue #670: no silent default) ===
        if has_u_terminal:
            self._setup_custom_final_value()
        else:
            raise ValueError(
                "u_terminal (terminal condition) must be provided in MFGComponents. "
                "Example: MFGComponents(u_terminal=lambda x: ..., m_initial=lambda x: ...). "
                "See examples/tutorials/01_hello_mfg.py for the classic LQ-MFG setup."
            )

        # === m_initial: MUST be in MFGComponents (Issue #670: no silent default) ===
        if has_m_initial:
            self._setup_custom_initial_density()
        else:
            raise ValueError(
                "m_initial (initial density) must be provided in MFGComponents. "
                "Example: MFGComponents(u_terminal=lambda x: ..., m_initial=lambda x: ...). "
                "See examples/tutorials/01_hello_mfg.py for the classic LQ-MFG setup."
            )

        # Issue #687: Validate computed arrays for NaN/Inf (after setup methods)
        if self.geometry is not None:
            from mfgarchon.utils.validation import ValidationError, validate_finite

            u_result = validate_finite(self.u_terminal, "u_terminal")
            if not u_result.is_valid:
                raise ValidationError(u_result)

            m_result = validate_finite(self.m_initial, "m_initial")
            if not m_result.is_valid:
                raise ValidationError(m_result)

        # === Issue #672: Validate m_initial before normalization (Fail Fast) ===
        # Check 1: Non-negativity (density must be >= 0)
        if np.any(self.m_initial < 0):
            min_val = np.min(self.m_initial)
            raise ValueError(
                f"m_initial contains negative values (min={min_val:.6e}). "
                "Initial density must be non-negative. "
                "Check your m_initial function in MFGComponents."
            )

        # Check 2: Non-zero mass (must have some mass to normalize)
        # #1887: VALIDATE, DO NOT NORMALISE. This block used to rescale whatever `m_initial`
        # returned so that a discrete integral came out 1, silently. A user who writes
        # `m_initial=lambda x: np.exp(-10*(x-0.5)**2)` hands in a function whose integral is about
        # 0.56 and got back something else with nothing said -- the same shape as silent clipping,
        # which this repository has already decided against. Normalising is the caller's job; the
        # library's job is to say what it received.
        #
        # #2145: on the GEOMETRY'S OWN MEASURE. `TensorProductGrid` is endpoint-inclusive, so the
        # two wall nodes own half a cell each and the integral is the trapezoid. Reporting
        # `sum(m)*dx` here while the FP wall conserves the trapezoid would put the two ends of every
        # conservation check on different measures.
        #
        # Three tiers, and only the third presumes a target of 1 -- which is why the third is the
        # one that can be silenced and the one a split-population design may move. `mass = 1` is not
        # a law of the FP equation; it conserves whatever it starts with, and split populations each
        # carry a share by design.
        mass, measure = self._measure_initial_density()

        # TIER 1 -- refuse. One branch, because the other two already have owners upstream and
        # restating them here made a dead branch look load-bearing: a negative density is refused
        # by Check 1 above ("m_initial contains negative values"), and a non-finite one by
        # `validate_finite`, which raises ValidationError -- not even the same exception type this
        # tier would have used. Measured: of the three cases a duplicated tier 1 claimed to own,
        # only a non-positive mass ever reached it. Found by independent review of #2174.
        if not np.isfinite(mass) or mass <= 0.0:
            raise ValueError(
                f"m_initial has total mass {mass!r} on the {measure} measure. Initial density must "
                "integrate to a positive value. Check your m_initial function in MFGComponents."
            )

        self.initial_mass = float(mass)
        self.initial_mass_measure = measure

        # TIER 2 -- report, always, WITH THE MEASURE. This is the tier that closes #1887's first
        # cost: a miscoded `m_initial` never announced itself, and the fix is to report the number
        # rather than to demand a particular one. Naming the measure is not decoration -- a bare
        # number would recreate the invisible convention the report exists to remove.
        from mfgarchon.utils.mfg_logging import get_logger

        get_logger(__name__).info(
            "initial density mass %.6g (%s measure); the library does not rescale it", mass, measure
        )

        # TIER 3 -- warn, and only here is a target of 1 assumed. Silence with
        # `warnings.filterwarnings("ignore", message="initial density mass")` when a sub-probability
        # density or a per-population share is intended.
        if abs(mass - 1.0) > self._INITIAL_MASS_TOLERANCE:
            import warnings

            warnings.warn(
                f"initial density mass {mass:.6g} on the {measure} measure, not 1. The library no "
                f"longer rescales it (#1887), so the solve runs with this mass and f(m) is "
                f"evaluated at these values. Intended for a sub-probability density or one "
                f"population's share? Silence this warning. Otherwise normalise before handing it "
                f"over: m_initial = raw / <integral of raw>.",
                UserWarning,
                stacklevel=3,
            )

    _INITIAL_MASS_TOLERANCE: ClassVar[float] = 1e-8

    def spatial_measure(self) -> SpatialMeasure:
        """The measure this problem integrates over space with, and its name: the one owner (#2555).

        ``initial_mass`` reads it, and so does the coupling iterators' convergence norm
        (``calculate_l2_convergence_metrics``), so the mass a problem reports and the change its outer
        tolerance bounds are measured alike. ``integrate`` reduces the trailing spatial axes.

        The name is returned rather than assumed because these branches are genuinely different
        objects, not fallbacks of one: a network has no cell volume at all, and an unstructured
        geometry has no quadrature. Reporting a number without saying which of them produced it is
        what #1887 calls the invisible convention.

        The network branch gates on ``self.is_network`` -- which reads ``geometry.geometry_type``
        -- and not on ``self.dimension`` (#2177). ``NetworkMFGProblem`` sets ``dimension =
        "network"`` AFTER ``super().__init__()``, and this runs inside it, so the old guard read
        ``dimension == 2``: a network problem describing itself as two-dimensional during its own
        construction. It fell through to ``point-average``, publishing ``1/N`` under the name
        "initial density mass" and warning that it was not 1, with a remedy -- divide by the
        integral -- that could not work because the density already summed to 1.
        The branch was NOT dead in general -- ``_init_network`` sets ``dimension = "network"`` at
        line ~1156, which runs from ``__init__`` BEFORE ``_initialize_functions``, so
        ``MFGProblem(network=<graph>)`` always reached it. What was unreachable is the
        *geometry-first* network path: ``NetworkMFGProblem`` and ``MFGProblem(geometry=<network>)``,
        both of which now measure on the nodes too. (#2177's own body says the branch was dead; that
        is true only of the path it was looking at.)

        ``node-sum`` is also the functional the network FP solver conserves --
        ``alg/numerical/network_solvers/fp_network.py`` uses ``float(np.sum(M[0, :]))`` as its own
        total mass -- so this puts ``problem.initial_mass`` on the same measure as the solve.

        Same lesson as #2157: gate on the thing you are about to use.
        """
        integrate = getattr(self.geometry, "integrate", None)
        if not self.is_network and callable(integrate):
            return SpatialMeasure(integrate, "grid")
        if self.is_network:
            return SpatialMeasure(lambda f: np.sum(f, axis=-1), "node-sum")
        if self.dimension == 1:
            dx = self._get_spacing() or 1.0
            return SpatialMeasure(lambda f: np.sum(f, axis=-1) * dx, "uniform-cell")
        from mfgarchon.geometry import GeometryType

        # A declared Cartesian grid without `integrate` is measured on its cells. This is the population that
        # had to report `spatial_discretization` before #1889 retired it, and the gate reads the same
        # declaration `__init__` dispatches on, not `CartesianGrid` inheritance, which the protocol does not ask for.
        if self.geometry is not None and self.geometry.geometry_type == GeometryType.CARTESIAN_GRID:
            get_spacing = getattr(self.geometry, "get_grid_spacing", None)
            spacing = get_spacing() if callable(get_spacing) else None
            if spacing is None:
                raise ValueError(
                    "measuring a field on this geometry needs the grid spacing, and this geometry "
                    f"({type(self.geometry).__name__}) declares a CARTESIAN_GRID with neither integrate() nor a "
                    "get_grid_spacing() that returns one."
                )
            cell = float(np.prod(spacing))
            axes = tuple(range(-len(spacing), 0))
            return SpatialMeasure(lambda f: np.sum(f, axis=axes) * cell, "uniform-cell")
        n = self.num_spatial_points
        return SpatialMeasure(lambda f: np.sum(f, axis=-1) / n, "point-average")

    def _measure_initial_density(self) -> tuple[float, str]:
        """The initial density's total mass, and the name of the measure that produced it."""
        measure = self.spatial_measure()
        try:
            return float(measure.integrate(np.asarray(self.m_initial))), measure.name
        except ValueError as exc:
            if measure.name != "grid":
                raise
            # The one case that reaches here is a single-node axis, which `quadrature_weights_1d`
            # refuses because a one-node axis has no measure -- returning 0 or dx would both be
            # inventions. That refusal is right, but its message names neither this problem nor
            # `m_initial`, so re-raise with both. Found by independent review of #2145: on a
            # 1-point grid `MFGProblem` used to construct and now did not, with a diagnostic a
            # caller could not act on.
            raise ValueError(
                f"cannot measure m_initial on this geometry: {exc}. A "
                f"{type(self.geometry).__name__} with a one-node axis has zero extent, so it "
                "carries no measure and no Fokker-Planck problem is posed on it. Give the axis "
                "at least two points."
            ) from exc

    # Issue #670: _setup_default_initial_density() removed - m_initial must be explicit

    # Methods inherited from HamiltonianMixin:
    # - H(), dH_dm(), get_hjb_hamiltonian_jacobian_contrib()
    # - get_hjb_residual_m_coupling_term(), _validate_hamiltonian_components()
    #
    # Methods inherited from ConditionsMixin:
    # - get_boundary_conditions()
    # - _setup_custom_initial_density(), _setup_custom_final_value()

    def get_u_terminal(self) -> np.ndarray:
        """Get terminal condition u(T, x). Issue #670: unified naming."""
        return self.u_terminal.copy()

    def get_m_initial(self) -> np.ndarray:
        """Get initial density m(0, x). Issue #670: unified naming."""
        return self.m_initial.copy()

    # Legacy aliases for backward compatibility
    def get_m_init(self) -> np.ndarray:
        """Legacy alias for get_m_initial()."""
        return self.get_m_initial()

    def get_final_u(self) -> np.ndarray:
        """Legacy alias for get_u_terminal()."""
        return self.get_u_terminal()

    def get_initial_m(self) -> np.ndarray:
        """Legacy alias for get_m_initial()."""
        return self.get_m_initial()

    def get_problem_info(self) -> dict[str, Any]:
        """Get information about the problem."""
        # Get domain info from geometry (modern API)
        bounds = self.geometry.get_bounds() if self.geometry is not None else None
        domain_info = {
            "dimension": self.dimension,
            "num_spatial_points": self.geometry.num_spatial_points if self.geometry is not None else None,
        }
        if bounds is not None and self.dimension == 1:
            domain_info["xmin"] = float(bounds[0][0])
            domain_info["xmax"] = float(bounds[1][0])
            domain_info["Nx"] = self._get_num_intervals()

        if self.is_custom and self.components is not None:
            return {
                "description": self.components.description,
                "problem_type": self.components.problem_type,
                "is_custom": True,
                "has_custom_hamiltonian": True,
                "has_custom_initial": self.components.m_initial is not None,
                "has_custom_final": self.components.u_terminal is not None,
                # Issue #673: jacobian_fd() always available on HamiltonianBase
                "has_jacobian": self.components._hamiltonian_class is not None,
                "parameters": self.components.parameters,
                "domain": domain_info,
                "time": {"T": self.T, "Nt": self.Nt},
                "coefficients": {"volatility": self._volatility, "coupling_coefficient": self.coupling_coefficient},
            }
        else:
            return {
                "description": "Default MFG Problem",
                "problem_type": "example",
                "is_custom": False,
                "has_custom_hamiltonian": False,
                "has_custom_initial": False,
                "has_custom_final": False,
                "has_jacobian": False,
                "has_coupling": False,
                "parameters": {},
                "domain": domain_info,
                "time": {"T": self.T, "Nt": self.Nt},
                "coefficients": {"volatility": self._volatility, "coupling_coefficient": self.coupling_coefficient},
            }

    # ============================================================================
    # Kwargs Validation - Fail Fast on Deprecated/Unrecognized Parameters
    # ============================================================================

    # Retired kwargs, each refused with the API that replaces it (Issue #666, #670, #2554)
    _DEPRECATED_KWARGS: ClassVar[dict[str, str]] = {
        "hamiltonian": "Model(hamiltonian=...), a Hamiltonian object such as SeparableHamiltonian(...)",
        "dH_dm": "the Hamiltonian's coupling_dm=, which takes f'(m); H carries -f, so dH/dm = -f'(m), e.g. "
        "SeparableHamiltonian(coupling=..., coupling_dm=...)",
        "dH_dp": "the Hamiltonian's control_cost=, whose dp() is dH/dp, e.g. QuadraticControlCost(...)",
        "potential": "SeparableHamiltonian(potential=...)",
        "running_cost": "the Hamiltonian's potential= (a cost of x) or coupling= (a cost of m)",
        "terminal_cost": "MFGComponents.u_terminal",
        # Issue #670: initial/terminal conditions now ONLY via MFGComponents
        "m_initial": "MFGComponents.m_initial",
        "u_final": "MFGComponents.u_terminal",
        "initial_density": "MFGComponents.m_initial",
        # Issue #1363: legacy 1D geometry kwargs removed (deprecated since v0.17.1);
        # the **kwargs signature would otherwise swallow them silently.
        "xmin": "geometry=TensorProductGrid(bounds=[(xmin, xmax)], Nx_points=[Nx + 1])",
        "xmax": "geometry=TensorProductGrid(bounds=[(xmin, xmax)], Nx_points=[Nx + 1])",
        "Nx": "geometry=TensorProductGrid(bounds=[(xmin, xmax)], Nx_points=[Nx + 1])",
        "Lx": "geometry=TensorProductGrid(bounds=[(0.0, Lx)], Nx_points=[Nx + 1])",
    }

    # Legacy 1D geometry kwargs removed in Issue #1363 (subset of _DEPRECATED_KWARGS)
    _GEOMETRY_KWARGS: ClassVar[set[str]] = {"xmin", "xmax", "Nx", "Lx"}

    # Known valid kwargs that are consumed by _initialize_functions or mixins
    _RECOGNIZED_KWARGS: ClassVar[set[str]] = {
        "boundary_conditions",  # BC object
    }

    def _validate_kwargs(self, kwargs: dict[str, Any]) -> None:
        """
        Validate kwargs - fail fast on deprecated or unrecognized parameters.

        Issue #666: Prevents silent fail where user-provided kwargs are ignored.
        Uses centralized validate_kwargs utility with MFGProblem-specific guidance.

        Raises:
            ValueError: If deprecated kwargs are passed (must use MFGComponents)
            UserWarning: If unrecognized kwargs are passed (probably a typo)
        """
        try:
            validate_kwargs(
                kwargs=kwargs,
                deprecated_kwargs=self._DEPRECATED_KWARGS,
                recognized_kwargs=self._RECOGNIZED_KWARGS,
                context="MFGProblem",
                error_on_deprecated=True,
                warn_on_unrecognized=True,
            )
        except ValueError as e:
            # Enhance error with the relevant MFGProblem-specific migration guide.
            if self._GEOMETRY_KWARGS & set(kwargs):
                # Issue #1363: legacy 1D geometry kwargs (xmin/xmax/Nx/Lx) removed.
                migration_guide = """

The legacy 1D geometry kwargs (xmin/xmax/Nx/Lx) are no longer supported.
Build the grid and pass it as the domain:

  from mfgarchon import Conditions, MFGProblem, Model
  from mfgarchon.geometry import TensorProductGrid
  from mfgarchon.geometry.boundary import no_flux_bc

  geometry = TensorProductGrid(
      bounds=[(xmin, xmax)],
      Nx_points=[Nx + 1],  # Nx intervals -> Nx + 1 grid points
      boundary_conditions=no_flux_bc(dimension=1),
  )
  problem = MFGProblem(
      model=Model(hamiltonian=my_hamiltonian, volatility=sigma),
      domain=geometry,
      conditions=Conditions(m_initial=my_m0, u_terminal=my_uT, T=T),
      Nt=Nt,
  )

See: docs/user/GEOMETRY_FIRST_API_GUIDE.md"""
            else:
                migration_guide = """

The kwargs-based Hamiltonian API is no longer supported.
Build the Hamiltonian as an object and pass it through Model:

  from mfgarchon import Conditions, MFGProblem, Model
  from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian

  hamiltonian = SeparableHamiltonian(
      control_cost=QuadraticControlCost(control_cost=1.0),
      potential=my_potential,  # V(t, x), cost-signed
      coupling=my_coupling,  # f(m)
      coupling_dm=my_coupling_dm,  # f'(m)
  )
  problem = MFGProblem(
      model=Model(hamiltonian=hamiltonian, volatility=sigma),
      domain=my_geometry,
      conditions=Conditions(m_initial=my_m0, u_terminal=my_uT, T=T),
      Nt=Nt,
  )

See: docs/user/CONVENTIONS.md, section 2 (the cost channels and their signs)"""
            raise ValueError(str(e) + migration_guide) from None

    # ============================================================================
    # Solve Method - Primary API for solving MFG problems
    # ============================================================================

    def solve(
        self,
        Nt: int | None = None,
        *,
        max_iterations: int | None = None,
        tolerance: float | None = None,
        absolute_tolerance: float | None = None,
        verbose: bool | None = None,
        config: Any | None = None,
        scheme: Any | None = None,
        hjb_solver: Any | None = None,
        fp_solver: Any | None = None,
    ) -> Any:
        """
        Solve this MFG problem using three-mode API (Issue #580).

        **Three Solving Modes:**

        1. **Safe Mode** (Recommended): Specify validated numerical scheme
            >>> result = problem.solve(scheme=NumericalScheme.FDM_UPWIND)
            Automatically creates dual HJB-FP solver pair with duality guarantee.

        2. **Expert Mode**: Manual solver injection for advanced users
            >>> hjb = HJBFDMSolver(problem)
            >>> fp = FPFDMSolver(problem)
            >>> result = problem.solve(hjb_solver=hjb, fp_solver=fp)
            Full control, but duality validation warnings if mismatched.

        3. **Auto Mode**: Intelligent automatic selection (default)
            >>> result = problem.solve()
            Analyzes geometry and selects appropriate scheme automatically.

        Args:
            Nt: Time discretization steps (Issue #875). If provided, overrides
                the Nt stored at construction time. This separates physics (T)
                from numerics (Nt) — different solvers may want different Nt
                for the same physical problem.
            max_iterations: Maximum fixed-point iterations (default: from config or 100)
            tolerance: Bound on the relative L2 change of one fixed-point sweep (default: from config
                or 1e-6; docs/user/CONVENTIONS.md § 9)
            absolute_tolerance: If given, the absolute L2 change must also fall below it (default:
                from config, or None: no absolute criterion; #2555)
            verbose: Show solver progress (default: from config or True)
            config: Optional MFGSolverConfig for advanced configuration.
                ``config.picard`` drives iteration parameters (max_iterations,
                tolerance, relaxation, anderson_memory).  ``config.hjb`` /
                ``config.fp`` non-default fields are translated to solver
                constructor kwargs (Issue #1155).  Fields with no mapping raise
                NotImplementedError (fail-loud, Refs #1155).  Use Expert Mode
                (hjb_solver/fp_solver) to bypass the translator entirely.
            scheme: NumericalScheme for Safe Mode (FDM_UPWIND, SL_LINEAR, GFDM, etc.)
            hjb_solver: Pre-initialized HJB solver for Expert Mode
            fp_solver: Pre-initialized FP solver for Expert Mode

        Returns:
            SolverResult with U (value function), M (density), convergence info

        Examples:
            >>> # Safe Mode: Automatic dual pairing
            >>> from mfgarchon.types import NumericalScheme
            >>> result = problem.solve(scheme=NumericalScheme.FDM_UPWIND)

            >>> # Auto Mode: Intelligent selection
            >>> result = problem.solve()

            >>> # Expert Mode: Full control
            >>> hjb = HJBSemiLagrangianSolver(problem, interpolation_method="cubic")
            >>> fp = FPSLSolver(problem)
            >>> result = problem.solve(hjb_solver=hjb, fp_solver=fp)

        Note:
            Cannot mix modes: specify either `scheme` OR (`hjb_solver` + `fp_solver`),
            not both. Omit all to use Auto Mode.
        """
        from mfgarchon.alg.numerical.coupling import FixedPointIterator
        from mfgarchon.config import MFGSolverConfig
        from mfgarchon.factory import create_paired_solvers, get_recommended_scheme
        from mfgarchon.utils import check_solver_duality

        # Issue #875: Nt override at solve() time
        if Nt is not None and Nt != self.Nt:
            # Nt deeply affects solver setup. For v1.0 API, reconstruct
            # with the correct Nt and delegate solve to the new problem.
            if self._v1_model is not None:
                new_problem = MFGProblem(
                    model=self._v1_model,
                    domain=self.geometry,
                    conditions=self._v1_conditions,
                    constraints=self._v1_constraints,
                    Nt=Nt,
                )
                return new_problem.solve(
                    max_iterations=max_iterations,
                    tolerance=tolerance,
                    absolute_tolerance=absolute_tolerance,
                    verbose=verbose,
                    config=config,
                    scheme=scheme,
                    hjb_solver=hjb_solver,
                    fp_solver=fp_solver,
                )
            else:
                raise ValueError(
                    f"solve(Nt={Nt}) differs from construction Nt={self.Nt}. "
                    "Nt override at solve time is only supported for v1.0 API problems "
                    "(created with model/domain/conditions). For legacy problems, "
                    "reconstruct with the desired Nt."
                )

        # Create or update config
        if config is None:
            config = MFGSolverConfig()

        # Override config only with explicitly passed parameters (not None)
        # This allows config values to be used when parameters are not specified
        if max_iterations is not None:
            config.picard.max_iterations = max_iterations
        if tolerance is not None:
            config.picard.tolerance = tolerance
        if absolute_tolerance is not None:
            config.picard.absolute_tolerance = absolute_tolerance
        if verbose is not None:
            config.picard.verbose = verbose

        # ═══════════════════════════════════════════════════════════════════════
        # Phase 3: Three-Mode API (Issue #580)
        # ═══════════════════════════════════════════════════════════════════════

        # Mode Detection
        safe_mode = scheme is not None
        expert_mode = hjb_solver is not None or fp_solver is not None

        # Mode Validation: Cannot mix modes
        if safe_mode and expert_mode:
            raise ValueError(
                "Cannot mix Safe Mode (scheme parameter) with Expert Mode "
                "(hjb_solver/fp_solver parameters). Use one mode at a time:\n"
                "  • Safe Mode: problem.solve(scheme=NumericalScheme.FDM_UPWIND)\n"
                "  • Expert Mode: problem.solve(hjb_solver=hjb, fp_solver=fp)\n"
                "  • Auto Mode: problem.solve() [no scheme/solver params]"
            )

        # ─────────────────────────────────────────────────────────────────────
        # Safe Mode: Automatic dual pairing via scheme selection
        # ─────────────────────────────────────────────────────────────────────
        if safe_mode:
            # Convert string to NumericalScheme if needed
            from mfgarchon.types import NumericalScheme

            if isinstance(scheme, str):
                try:
                    scheme = NumericalScheme(scheme)
                except ValueError:
                    raise ValueError(
                        f"Unknown scheme string: {scheme!r}. Valid schemes: {[s.value for s in NumericalScheme]}"
                    ) from None

            # Issue #1155: translate config.hjb / config.fp to solver kwargs.
            # Non-default fields that have clear mappings are threaded; unknown /
            # unsupported non-default fields raise NotImplementedError (fail-loud).
            from mfgarchon.config.translator import fp_config_to_kwargs, hjb_config_to_kwargs

            _hjb_ctor_kw = hjb_config_to_kwargs(config.hjb, scheme)
            _fp_ctor_kw = fp_config_to_kwargs(config.fp, scheme)

            # Create validated dual pair (Phase 2 factory)
            hjb_solver, fp_solver = create_paired_solvers(
                problem=self,
                scheme=scheme,
                hjb_config=_hjb_ctor_kw,
                fp_config=_fp_ctor_kw,
                validate_duality=True,  # Guaranteed dual by construction
            )

            if verbose:
                from mfgarchon.utils.mfg_logging import get_logger

                logger = get_logger(__name__)
                logger.info(f"Safe Mode: Created dual solver pair for {scheme.value}")

        # ─────────────────────────────────────────────────────────────────────
        # Expert Mode: Manual solver injection with duality validation
        # ─────────────────────────────────────────────────────────────────────
        elif expert_mode:
            # Both solvers must be provided
            if hjb_solver is None or fp_solver is None:
                raise ValueError(
                    "Expert Mode requires BOTH hjb_solver and fp_solver. "
                    "You provided only one. Either:\n"
                    "  • Provide both: problem.solve(hjb_solver=hjb, fp_solver=fp)\n"
                    "  • Use Safe Mode: problem.solve(scheme=NumericalScheme.FDM_UPWIND)\n"
                    "  • Use Auto Mode: problem.solve() [omit both]"
                )

            # Validate duality (educational warnings if mismatched)
            result = check_solver_duality(hjb_solver, fp_solver, warn_on_mismatch=True)

            if verbose and not result.is_valid_pairing():
                from mfgarchon.utils.mfg_logging import get_logger

                logger = get_logger(__name__)
                logger.warning(
                    f"Expert Mode: Non-dual solver pair detected!\n"
                    f"  HJB: {type(hjb_solver).__name__} ({result.hjb_family})\n"
                    f"  FP: {type(fp_solver).__name__} ({result.fp_family})\n"
                    f"  Status: {result.status.value}\n"
                    f"  Why: {result.message}\n"
                    f"This may lead to poor convergence or Nash gap issues.\n"
                    f"Consider using Safe Mode for guaranteed duality."
                )

        # ─────────────────────────────────────────────────────────────────────
        # Auto Mode: Intelligent scheme selection (Phase 3 future work)
        # ─────────────────────────────────────────────────────────────────────
        else:  # auto_mode
            # Phase 3 TODO: Implement geometry introspection
            # For now, get_recommended_scheme() returns FDM_UPWIND as safe default
            recommended_scheme = get_recommended_scheme(self)

            # Issue #1155: translate config.hjb / config.fp to solver kwargs.
            from mfgarchon.config.translator import fp_config_to_kwargs, hjb_config_to_kwargs

            _hjb_ctor_kw = hjb_config_to_kwargs(config.hjb, recommended_scheme)
            _fp_ctor_kw = fp_config_to_kwargs(config.fp, recommended_scheme)

            hjb_solver, fp_solver = create_paired_solvers(
                problem=self,
                scheme=recommended_scheme,
                hjb_config=_hjb_ctor_kw,
                fp_config=_fp_ctor_kw,
                validate_duality=True,
            )

            if verbose:
                from mfgarchon.utils.mfg_logging import get_logger

                logger = get_logger(__name__)
                logger.info(f"Auto Mode: Selected {recommended_scheme.value} (geometry-based recommendation)")

        # ─────────────────────────────────────────────────────────────────────
        # Create fixed-point iterator with selected/validated solvers
        # ─────────────────────────────────────────────────────────────────────
        # Issue #1155: thread anderson_memory and backend from config to iterator.
        from mfgarchon.config.translator import (
            backend_config_to_kwargs,
            check_logging_config,
            picard_config_to_iterator_kwargs,
        )

        _iterator_extra_kw = picard_config_to_iterator_kwargs(config.picard)
        _backend_kw = backend_config_to_kwargs(config.backend)
        check_logging_config(config.logging)

        solver = FixedPointIterator(
            problem=self,
            hjb_solver=hjb_solver,
            fp_solver=fp_solver,
            config=config,
            # No volatility= override: each solver reads this problem's own volatility, with its
            # declared kind, so both the HJB and FP solvers see the full volatility, as supplied
            # (#1248, #2376). Until #2378 part 2a this forwarded self._volatility without the kind,
            # and ten sites told it apart from a user override by identity. A pair built from another
            # problem's volatility is refused by the iterator's pairing guard (#2420 review).
            **_iterator_extra_kw,
            **_backend_kw,
        )

        return solver.solve(verbose=verbose)

    # ==========================================================================
    # API v1.0: Parameter Variation Helpers (Issue #875)
    # ==========================================================================

    def with_model(self, model: Any) -> MFGProblem:
        """Return new problem with different model (game rules).

        Requires that this problem was created with the v1.0 API.
        """
        if self._v1_model is None:
            raise ValueError("with_model() requires a problem created with API v1.0 (model/domain/conditions)")
        return MFGProblem(
            model=model,
            domain=self.geometry,
            conditions=self._v1_conditions,
            constraints=self._v1_constraints,
            Nt=self.Nt,
        )

    def with_domain(self, domain: Any) -> MFGProblem:
        """Return new problem with different domain (spatial geometry)."""
        if self._v1_model is None:
            raise ValueError("with_domain() requires a problem created with API v1.0 (model/domain/conditions)")
        return MFGProblem(
            model=self._v1_model,
            domain=domain,
            conditions=self._v1_conditions,
            constraints=self._v1_constraints,
            Nt=self.Nt,
        )

    def with_conditions(self, conditions: Any) -> MFGProblem:
        """Return new problem with different conditions (time + IC/TC)."""
        if self._v1_model is None:
            raise ValueError("with_conditions() requires a problem created with API v1.0 (model/domain/conditions)")
        return MFGProblem(
            model=self._v1_model,
            domain=self.geometry,
            conditions=conditions,
            constraints=self._v1_constraints,
            Nt=self.Nt,
        )

    # No with_volatility() or with_T() — these are premature convenience shortcuts
    # that break orthogonality. The volatility lives in Model, T lives in Conditions.
    # Use with_model(Model(hamiltonian=H, volatility=0.2)) or
    # with_conditions(Conditions(u_terminal=..., m_initial=..., T=2.0)) instead.
