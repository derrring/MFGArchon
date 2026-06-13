"""
Layer 2 -- Resolution: Equation-dependent BC resolution.

Translates physical boundary intent (BCType) into mathematical BC type
(MathBCType) based on which PDE equation is being solved. This is the
missing layer between specification (Layer 1) and enforcement (Layer 3)
in the 4-layer BC architecture.

The core problem: ``BCType.NO_FLUX`` means different things for different PDEs:

- HJB: du/dn = 0 (zero gradient -- field has no normal component at wall)
- FP: J*n = 0 (zero probability flux -- mass conservation at impermeable wall)

The resolution layer makes this mapping explicit via solver-specific resolvers
that implement the ``BCResolver`` protocol.

Architecture (Issue #848, see BC_ENFORCEMENT_ARCHITECTURE.md):

- **Layer 1 -- Specification**: types.py, conditions.py, providers.py
- **Layer 2 -- Resolution**: resolution.py (THIS MODULE)
- **Layer 3 -- Enforcement**: calculators.py, enforcement.py, ghost_cells.py
- **Layer 4 -- Application**: applicator_*.py, dispatch.py

References:
    - Issue #848 Phase 3
    - BC_ENFORCEMENT_ARCHITECTURE.md Section: The Resolution Problem
    - Achdou & Lauriere (2020): FP no-flux BC derivation
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from mfgarchon.utils.mfg_logging import get_logger

from .conditions import (
    BoundaryConditions,
    no_flux_bc,
)
from .providers import is_provider
from .types import BCSegment, BCType

logger = get_logger(__name__)

# =============================================================================
# Mathematical BC types (post-resolution, no ambiguity)
# =============================================================================


class MathBCType(Enum):
    """Mathematical boundary condition type (post-resolution).

    These represent the mathematical form of the BC as needed by calculators
    and enforcement functions. Each maps 1:1 to a calculator in calculators.py.

    Unlike ``BCType`` (Layer 1), this enum contains no physical intent types
    (NO_FLUX, REFLECTING) or ambiguous types. Those are resolved into one of
    these mathematical types by a ``BCResolver``.
    """

    DIRICHLET = "dirichlet"  # u = g
    NEUMANN = "neumann"  # du/dn = g
    ROBIN = "robin"  # alpha*u + beta*du/dn = g
    PERIODIC = "periodic"  # u(x_min) = u(x_max)
    ZERO_FLUX = "zero_flux"  # J*n = 0, needs drift+diffusion for calculator
    EXTRAPOLATION_LINEAR = "extrapolation_linear"  # d^2u/dx^2 = 0 at boundary
    EXTRAPOLATION_QUADRATIC = "extrapolation_quadratic"  # d^3u/dx^3 = 0


# =============================================================================
# Resolved BC dataclass
# =============================================================================


@dataclass(frozen=True)
class ResolvedBC:
    """A single resolved boundary condition segment.

    Contains everything a Layer 3 calculator needs: the mathematical BC type,
    concrete coefficient values, and traceability to the original segment.

    Produced by a ``BCResolver`` from a ``BCSegment``.
    """

    math_type: MathBCType
    value: float = 0.0
    alpha: float = 1.0  # Robin: weight on u
    beta: float = 0.0  # Robin: weight on du/dn
    segment_name: str = ""
    original_bc_type: BCType | None = None


# =============================================================================
# BCResolver protocol
# =============================================================================


@runtime_checkable
class BCResolver(Protocol):
    """Resolves physical boundary intent into mathematical BC for a specific PDE.

    Each PDE solver family has its own resolver that knows how to translate
    physical intent (e.g., "impermeable wall") into the correct mathematical
    BC (e.g., Neumann for HJB, zero-flux Robin for FP).

    See BC_ENFORCEMENT_ARCHITECTURE.md for the full design rationale.
    """

    def resolve(
        self,
        segment: BCSegment,
        solver_state: dict[str, Any],
    ) -> ResolvedBC:
        """Resolve a single BCSegment into a mathematical BC.

        Args:
            segment: Physical boundary specification (intent + location).
            solver_state: PDE coefficients and current solution state.
                Standard keys: 'drift', 'diffusion', 'density',
                'value_function', 'time', 'geometry'.

        Returns:
            ResolvedBC with concrete mathematical type and values.
        """
        ...


# =============================================================================
# Concrete resolvers
# =============================================================================

# Passthrough types that don't depend on the equation
_PASSTHROUGH_MAP: dict[BCType, MathBCType] = {
    BCType.DIRICHLET: MathBCType.DIRICHLET,
    BCType.NEUMANN: MathBCType.NEUMANN,
    BCType.ROBIN: MathBCType.ROBIN,
    BCType.PERIODIC: MathBCType.PERIODIC,
    BCType.EXTRAPOLATION_LINEAR: MathBCType.EXTRAPOLATION_LINEAR,
    BCType.EXTRAPOLATION_QUADRATIC: MathBCType.EXTRAPOLATION_QUADRATIC,
}


def _resolve_passthrough(seg: BCSegment) -> ResolvedBC | None:
    """Resolve BC types that are equation-independent.

    Returns None if the BCType requires equation-specific resolution.
    """
    math_type = _PASSTHROUGH_MAP.get(seg.bc_type)
    if math_type is None:
        return None

    value = float(seg.value) if isinstance(seg.value, (int, float)) else 0.0
    return ResolvedBC(
        math_type=math_type,
        value=value,
        alpha=getattr(seg, "alpha", 1.0),
        beta=getattr(seg, "beta", 0.0),
        segment_name=seg.name,
        original_bc_type=seg.bc_type,
    )


class HJBResolver:
    """Resolves BC intent for Hamilton-Jacobi-Bellman equations.

    Resolution rules:
        - NO_FLUX, REFLECTING -> NEUMANN(g=0): value function has zero normal
          gradient at reflecting walls (optimal control points inward).
        - BCValueProvider in segment.value -> resolve to concrete Robin BC
          (e.g., AdjointConsistentProvider for coupled MFG).
        - All other types: passthrough.
    """

    def resolve(
        self,
        segment: BCSegment,
        solver_state: dict[str, Any],
    ) -> ResolvedBC:
        """Resolve a BCSegment for HJB equation."""
        # Check passthrough types first
        result = _resolve_passthrough(segment)
        if result is not None:
            # For passthrough Robin with a provider value, resolve it
            if result.math_type == MathBCType.ROBIN and is_provider(segment.value):
                resolved_value = segment.value.compute(solver_state)
                return ResolvedBC(
                    math_type=MathBCType.ROBIN,
                    value=float(resolved_value),
                    alpha=getattr(segment, "alpha", 0.0),
                    beta=getattr(segment, "beta", 1.0),
                    segment_name=segment.name,
                    original_bc_type=segment.bc_type,
                )
            return result

        # Equation-specific: NO_FLUX and REFLECTING
        if segment.bc_type in (BCType.NO_FLUX, BCType.REFLECTING):
            # Check for dynamic provider (e.g., AdjointConsistentProvider)
            if is_provider(segment.value):
                resolved_value = segment.value.compute(solver_state)
                return ResolvedBC(
                    math_type=MathBCType.ROBIN,
                    value=float(resolved_value),
                    alpha=0.0,
                    beta=1.0,
                    segment_name=segment.name,
                    original_bc_type=segment.bc_type,
                )
            # Default: zero gradient (du/dn = 0)
            return ResolvedBC(
                math_type=MathBCType.NEUMANN,
                value=0.0,
                segment_name=segment.name,
                original_bc_type=segment.bc_type,
            )

        # Unknown BCType: default to Neumann(0) with warning
        logger.warning(
            "HJBResolver: unrecognized BCType %s on segment '%s', defaulting to NEUMANN(g=0)",
            segment.bc_type,
            segment.name,
        )
        return ResolvedBC(
            math_type=MathBCType.NEUMANN,
            value=0.0,
            segment_name=segment.name,
            original_bc_type=segment.bc_type,
        )


class FPResolver:
    """Resolves BC intent for Fokker-Planck equations.

    Resolution rules:
        - NO_FLUX, REFLECTING -> ZERO_FLUX: mass-conserving impermeable wall.
          The zero-flux condition J*n = 0 where J = v*m - D*grad(m) requires
          the ``ZeroFluxCalculator`` with drift and diffusion coefficients.
          Pure Neumann (dm/dn = 0) is incorrect when drift != 0 at boundary
          and violates the Lopatinski-Shapiro condition in the
          advection-dominated regime.
        - All other types: passthrough.

    References:
        - Achdou & Lauriere (2020): FP no-flux BC derivation
        - BC_ENFORCEMENT_ARCHITECTURE.md: L-S well-posedness analysis
    """

    def resolve(
        self,
        segment: BCSegment,
        solver_state: dict[str, Any],
    ) -> ResolvedBC:
        """Resolve a BCSegment for Fokker-Planck equation."""
        # Check passthrough types first
        result = _resolve_passthrough(segment)
        if result is not None:
            return result

        # Equation-specific: NO_FLUX and REFLECTING
        if segment.bc_type in (BCType.NO_FLUX, BCType.REFLECTING):
            return ResolvedBC(
                math_type=MathBCType.ZERO_FLUX,
                value=0.0,
                segment_name=segment.name,
                original_bc_type=segment.bc_type,
            )

        # Unknown BCType: default to zero flux with warning
        logger.warning(
            "FPResolver: unrecognized BCType %s on segment '%s', defaulting to ZERO_FLUX",
            segment.bc_type,
            segment.name,
        )
        return ResolvedBC(
            math_type=MathBCType.ZERO_FLUX,
            value=0.0,
            segment_name=segment.name,
            original_bc_type=segment.bc_type,
        )


# =============================================================================
# Main entry point
# =============================================================================


def resolve_bc(
    bc: BoundaryConditions,
    resolver: BCResolver,
    solver_state: dict[str, Any] | None = None,
) -> list[ResolvedBC]:
    """Resolve all BC segments using the given resolver.

    This is the main entry point for Layer 2. It takes a BoundaryConditions
    specification (Layer 1) and a solver-specific resolver, and produces a
    list of ResolvedBC (one per segment) for Layer 3 consumption.

    Args:
        bc: Boundary condition specification from Layer 1.
        resolver: Solver-specific resolver (e.g., HJBResolver, FPResolver).
        solver_state: PDE coefficients and current solution state.
            Passed to resolver.resolve() and to BCValueProviders.
            Standard keys: 'drift', 'diffusion', 'density', 'geometry', 'time'.

    Returns:
        List of ResolvedBC, one per segment in bc.segments.
    """
    state = solver_state or {}
    return [resolver.resolve(seg, state) for seg in bc.segments]


# =============================================================================
# Layer 2 -> Layer 3 bridge
# =============================================================================


def resolved_bc_to_calculator(
    resolved: ResolvedBC,
    shape: tuple[int, ...],
    grid_type: object | None = None,
    drift_velocity: float = 0.0,
    diffusion_coeff: float = 1.0,
) -> tuple[object, object | None]:
    """Convert a ResolvedBC to a (Topology, Calculator) pair for Layer 3.

    This parallels ``bc_to_topology_calculator()`` in applicator_fdm.py but
    works from ResolvedBC (unambiguous) rather than BoundaryConditions
    (potentially ambiguous). No flags needed.

    Args:
        resolved: A resolved BC from a BCResolver.
        shape: Grid shape (interior points).
        grid_type: Grid type for ghost cell formulas. Defaults to CELL_CENTERED.
        drift_velocity: Normal drift component (for ZERO_FLUX calculator).
        diffusion_coeff: Diffusion coefficient D = sigma^2/2 (for ZERO_FLUX).

    Returns:
        Tuple of (Topology, Calculator | None). Calculator is None for periodic.
    """
    from .calculators import (
        BoundedTopology,
        DirichletCalculator,
        LinearExtrapolationCalculator,
        NeumannCalculator,
        PeriodicTopology,
        QuadraticExtrapolationCalculator,
        RobinCalculator,
        ZeroFluxCalculator,
    )
    from .protocols import GridType as GridTypeEnum

    gt = grid_type if grid_type is not None else GridTypeEnum.CELL_CENTERED
    dimension = len(shape)

    match resolved.math_type:
        case MathBCType.PERIODIC:
            return PeriodicTopology(dimension, shape), None
        case MathBCType.DIRICHLET:
            return BoundedTopology(dimension, shape), DirichletCalculator(resolved.value, gt)
        case MathBCType.NEUMANN:
            return BoundedTopology(dimension, shape), NeumannCalculator(resolved.value, gt)
        case MathBCType.ROBIN:
            return BoundedTopology(dimension, shape), RobinCalculator(resolved.alpha, resolved.beta, resolved.value, gt)
        case MathBCType.ZERO_FLUX:
            return BoundedTopology(dimension, shape), ZeroFluxCalculator(drift_velocity, diffusion_coeff, gt)
        case MathBCType.EXTRAPOLATION_LINEAR:
            return BoundedTopology(dimension, shape), LinearExtrapolationCalculator()
        case MathBCType.EXTRAPOLATION_QUADRATIC:
            return (
                BoundedTopology(dimension, shape),
                QuadraticExtrapolationCalculator(),
            )
        case _:
            raise ValueError(f"Unknown MathBCType: {resolved.math_type}")


# =============================================================================
# Backward compatibility bridge
# =============================================================================

# Reverse mapping: MathBCType -> BCType for backward compat
_MATH_TO_BC_TYPE: dict[MathBCType, BCType] = {
    MathBCType.DIRICHLET: BCType.DIRICHLET,
    MathBCType.NEUMANN: BCType.NEUMANN,
    MathBCType.ROBIN: BCType.ROBIN,
    MathBCType.PERIODIC: BCType.PERIODIC,
    MathBCType.ZERO_FLUX: BCType.NO_FLUX,
    MathBCType.EXTRAPOLATION_LINEAR: BCType.EXTRAPOLATION_LINEAR,
    MathBCType.EXTRAPOLATION_QUADRATIC: BCType.EXTRAPOLATION_QUADRATIC,
}


def to_boundary_conditions(
    resolved: list[ResolvedBC],
    dimension: int | None = None,
) -> BoundaryConditions:
    """Convert resolved BCs back to BoundaryConditions for Layer 3/4 consumption.

    This is the backward-compatibility bridge. Existing applicators expect
    BoundaryConditions objects. This function creates one from resolved BCs,
    mapping MathBCType back to BCType.

    Note:
        ZERO_FLUX maps to BCType.NO_FLUX. The equation-awareness is lost in
        the round-trip, so callers needing full fidelity should use
        ``resolved_bc_to_calculator()`` instead.

    Args:
        resolved: List of ResolvedBC from a resolver.
        dimension: Spatial dimension (optional, inferred if possible).

    Returns:
        BoundaryConditions object with concrete values (no providers).
    """
    if not resolved:
        return no_flux_bc(dimension=dimension or 1)

    segments = []
    for rbc in resolved:
        bc_type = _MATH_TO_BC_TYPE.get(rbc.math_type, BCType.NEUMANN)
        seg = BCSegment(
            name=rbc.segment_name or f"resolved_{rbc.math_type.value}",
            bc_type=bc_type,
            value=rbc.value,
            alpha=rbc.alpha,
            beta=rbc.beta,
        )
        segments.append(seg)

    # Issue #1100: explicit NO_FLUX fallback for any boundary the resolved
    # segments do not cover (matches the empty-resolved no_flux_bc default above);
    # avoids the removed implicit PERIODIC default.
    return BoundaryConditions(segments=segments, dimension=dimension, default_bc=BCType.NO_FLUX)


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    from .conditions import dirichlet_bc, periodic_bc

    print("=== BC Resolution Layer Smoke Test ===\n")

    # --- HJB Resolver ---
    hjb = HJBResolver()
    state: dict[str, Any] = {}

    print("HJB Resolver:")
    # NO_FLUX -> NEUMANN(0)
    bc = no_flux_bc(dimension=1)
    results = resolve_bc(bc, hjb, state)
    for r in results:
        print(f"  NO_FLUX -> {r.math_type.value}(g={r.value})")
    assert results[0].math_type == MathBCType.NEUMANN
    assert results[0].value == 0.0

    # DIRICHLET -> DIRICHLET
    bc = dirichlet_bc(1.0, dimension=1)
    results = resolve_bc(bc, hjb, state)
    for r in results:
        print(f"  DIRICHLET -> {r.math_type.value}(g={r.value})")
    assert results[0].math_type == MathBCType.DIRICHLET
    assert results[0].value == 1.0

    # PERIODIC -> PERIODIC
    bc = periodic_bc(dimension=1)
    results = resolve_bc(bc, hjb, state)
    for r in results:
        print(f"  PERIODIC -> {r.math_type.value}")
    assert results[0].math_type == MathBCType.PERIODIC

    # --- FP Resolver ---
    fp = FPResolver()

    print("\nFP Resolver:")
    # NO_FLUX -> ZERO_FLUX
    bc = no_flux_bc(dimension=1)
    results = resolve_bc(bc, fp, state)
    for r in results:
        print(f"  NO_FLUX -> {r.math_type.value}")
    assert results[0].math_type == MathBCType.ZERO_FLUX

    # DIRICHLET -> DIRICHLET (passthrough)
    bc = dirichlet_bc(0.0, dimension=1)
    results = resolve_bc(bc, fp, state)
    for r in results:
        print(f"  DIRICHLET -> {r.math_type.value}(g={r.value})")
    assert results[0].math_type == MathBCType.DIRICHLET

    # --- Layer 2 -> 3 bridge ---
    print("\nLayer 2->3 bridge (resolved_bc_to_calculator):")
    for rbc in [
        ResolvedBC(MathBCType.NEUMANN, 0.0, segment_name="test_neum"),
        ResolvedBC(MathBCType.ZERO_FLUX, 0.0, segment_name="test_flux"),
        ResolvedBC(MathBCType.DIRICHLET, 1.0, segment_name="test_dir"),
    ]:
        topo, calc = resolved_bc_to_calculator(rbc, shape=(100,))
        print(f"  {rbc.math_type.value} -> {type(calc).__name__}")

    # --- Backward compat bridge ---
    print("\nBackward compat (to_boundary_conditions):")
    resolved_list = [
        ResolvedBC(MathBCType.NEUMANN, 0.0, segment_name="left"),
        ResolvedBC(MathBCType.DIRICHLET, 1.0, segment_name="right"),
    ]
    bc_back = to_boundary_conditions(resolved_list, dimension=1)
    print(f"  {len(bc_back.segments)} segments, dimension={bc_back.dimension}")
    assert len(bc_back.segments) == 2
    assert bc_back.segments[0].bc_type == BCType.NEUMANN

    print("\nAll smoke tests passed.")
