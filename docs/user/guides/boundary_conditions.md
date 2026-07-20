# Boundary Conditions Guide

This guide explains how to specify and apply boundary conditions in MFGArchon.

## Quick Start

```python
from mfgarchon.geometry.boundary import (
    dirichlet_bc,
    neumann_bc,
    periodic_bc,
    no_flux_bc,
    mixed_bc,
    BCSegment,
    BCType,
)

# Uniform BC (same on all boundaries)
bc = neumann_bc(dimension=2)  # Zero-flux everywhere

# Dirichlet with specific value
bc = dirichlet_bc(dimension=2, value=0.0)

# Periodic BC
bc = periodic_bc(dimension=2)

# Time-dependent BC
bc = dirichlet_bc(dimension=2, value=lambda t: np.sin(t))
```

## BC Types

A BC type is applied to **both** equations of the MFG system, and for two of them the
condition is *not the same equation on both sides*. This is the HJB/FP adjoint duality, not an
inconsistency: `u` is a value function and `m` is a density, and a reflecting wall for the density
is a Neumann condition for the value function.

| Type | HJB side (on `u`) | FP side (on `m`) | Typical use |
|------|-------------------|------------------|-------------|
| `DIRICHLET` | `u = g` | `m = g` | Fixed value at boundary |
| `NEUMANN` | `du/dn = g` | see note below | Prescribed boundary data |
| `NO_FLUX` | `du/dn = 0` | **`J.n = 0`**, where `J = v*m - D*grad(m)` | Reflecting / insulating wall; the mass-conserving choice |
| `ROBIN` | `alpha*u + beta*du/dn = g` | same form on `m` | Mixed condition |
| `PERIODIC` | `u(x_min) = u(x_max)` | `m(x_min) = m(x_max)` | Wrap-around domain |

**`NO_FLUX` on the FP side is zero *total* flux, not zero gradient.** With drift at the wall the
two differ: `J.n = 0` gives `D dm/dn = (v.n) m`, so `dm/dn` is generally **non-zero**. The FDM
divergence-form discretisation enforces the boundary flux as exactly zero
(`fp_fdm_alg_divergence_centered.py:273`), which is what conserves mass. Choosing `du/dn = 0` on
the density instead would leak mass wherever the drift is non-zero at the boundary.

The calculator classes name this distinction explicitly:

| class | condition |
|---|---|
| `ZeroGradientCalculator` | `du/dn = 0` |
| `ZeroFluxCalculator` | `J.n = 0` (mass conservation) |
| `NoFluxCalculator` | **deprecated since v0.16.11** — an alias for `ZeroGradientCalculator`. Pick one of the two above explicitly |

> **Note on `NEUMANN` with a non-zero value.** `neumann_bc(value=g)` with `g != 0` is honoured by
> the HJB side and **silently ignored by every FP solver** — see
> [Issue #1686](https://github.com/derrring/MFGArchon/issues/1686). Until that is fixed, only
> `g = 0` (equivalently `no_flux_bc()`) behaves consistently across the coupled system.

## Mixed Boundary Conditions

For different BC types on different boundaries:

```python
from mfgarchon.geometry.boundary import mixed_bc, BCSegment, BCType
import numpy as np

# Define domain
bounds = np.array([[0, 1], [0, 1]])  # Unit square

# Exit at top (Dirichlet), walls elsewhere (Neumann)
exit_bc = BCSegment(
    name="exit",
    bc_type=BCType.DIRICHLET,
    value=0.0,
    boundary="y_max",  # Top boundary
)
wall_bc = BCSegment(
    name="wall",
    bc_type=BCType.NEUMANN,
    value=0.0,
    # No boundary specified = default for remaining boundaries
)

bc = mixed_bc([exit_bc, wall_bc], dimension=2, domain_bounds=bounds)
```

## Region-Based Boundary Conditions

For complex geometries, you can define BCs using **regions** marked on the geometry rather than boundary identifiers:

```python
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import mixed_bc_from_regions, BCSegment, BCType

# Create geometry
geometry = TensorProductGrid(dimension=2, bounds=[(0, 2), (0, 1)], Nx_points=[41, 21])

# Mark regions using predicates
geometry.mark_region("inlet", predicate=lambda x: x[:, 0] < 0.1)  # Left 10%
geometry.mark_region("outlet", predicate=lambda x: x[:, 0] > 1.9)  # Right 10%
geometry.mark_region("walls", boundary="y_min")  # Bottom wall
geometry.mark_region("walls", boundary="y_max")  # Top wall (merged with bottom)

# Define BCs referencing regions
bc_config = {
    "inlet": BCSegment(name="inlet_bc", bc_type=BCType.DIRICHLET, value=1.0),
    "outlet": BCSegment(name="outlet_bc", bc_type=BCType.NEUMANN, value=0.0),
    "walls": BCSegment(name="wall_bc", bc_type=BCType.NO_FLUX),
    "default": BCSegment(name="periodic_bc", bc_type=BCType.PERIODIC),
}

bc = mixed_bc_from_regions(geometry, bc_config)

# Apply BCs (must pass geometry parameter)
from mfgarchon.geometry.boundary import FDMApplicator
applicator = FDMApplicator(dimension=2)
padded = applicator.apply(field, bc, domain_bounds=geometry.bounds, geometry=geometry)
```

### Region Marking Methods

Regions can be marked using boundaries or custom predicates:

```python
# Method 1: Mark using boundary identifier
geometry.mark_region("left_wall", boundary="x_min")

# Method 2: Mark using predicate function
geometry.mark_region("inlet", predicate=lambda x: (x[:, 0] < 0.1) & (x[:, 1] > 0.5))

# Method 3: Combine multiple boundaries in one region
geometry.mark_region("walls", boundary="y_min")
geometry.mark_region("walls", boundary="y_max")  # Adds to existing "walls" region
```

Predicates receive grid points as `(N, dimension)` array and must return boolean mask of length `N`.

### Priority Resolution for Overlapping Regions

When regions overlap, **priority** determines which BC wins (higher number = higher precedence):

```python
# Mark overlapping regions
geometry.mark_region("broad", predicate=lambda x: x[:, 0] < 0.5)
geometry.mark_region("narrow", predicate=lambda x: x[:, 0] < 0.1)

# Create segments with priorities
bc_narrow = BCSegment(
    name="narrow_bc",
    bc_type=BCType.DIRICHLET,
    value=1.0,
    region_name="narrow",
    priority=2,  # Higher number = higher precedence
)
bc_broad = BCSegment(
    name="broad_bc",
    bc_type=BCType.NEUMANN,
    value=0.0,
    region_name="broad",
    priority=1,  # Lower number = lower precedence
)

bc = BoundaryConditions(
    dimension=2,
    segments=[bc_narrow, bc_broad],  # Order doesn't matter
    default_bc=BCType.PERIODIC,
    domain_bounds=geometry.bounds,
)

# At x=0.05 (in both regions): narrow_bc wins due to higher priority
# At x=0.3 (only in broad): broad_bc applies
```

### Boundary-Based vs Region-Based

| Approach | Use When | Example |
|----------|----------|---------|
| **Boundary-based** | Simple geometries, entire boundaries | `boundary="x_min"` |
| **Region-based** | Complex geometries, partial boundaries | `predicate=lambda x: x[:, 0] < 0.1` |

Both approaches can be mixed in the same BC specification.

### Performance Notes

- Region lookup overhead: <5% compared to standard boundary-based BCs
- Region masks are cached after first call to `mark_region()`
- For best performance, mark all regions before solver iteration

## Boundary Naming Convention

| Dimension | Boundaries |
|-----------|------------|
| 1D | `x_min`, `x_max` |
| 2D | `x_min`, `x_max`, `y_min`, `y_max` |
| 3D | `x_min`, `x_max`, `y_min`, `y_max`, `z_min`, `z_max` |
| 4D+ | `dim0_min`, `dim0_max`, `dim1_min`, ... |

## Ghost Cell Method (FDM)

For finite difference methods, BCs are enforced via ghost cells:

```python
from mfgarchon.geometry.boundary import apply_boundary_conditions_2d

# field has shape (Ny, Nx) - interior points only
padded = apply_boundary_conditions_2d(field, bc, domain_bounds)
# padded has shape (Ny+2, Nx+2) - includes ghost cells
```

### Ghost Cell Formulas

For cell-centered grids with boundary at cell face:

- **Dirichlet** (u = g): `u_ghost = 2*g - u_interior`
- **Neumann** (du/dn = g): `u_ghost = u_interior + 2*dx*g` (outward normal)
- **No-flux** (du/dn = 0): `u_ghost = u_interior` -- this is the **zero-gradient** ghost, correct for `u` and for the default applicator path. It is *not* the FP mass-conserving wall; see [FP Solvers (FDM)](#fp-solvers-fdm) below.

## Corner Handling

When different BCs meet at corners (e.g., Dirichlet on one edge, Neumann on adjacent edge), the ghost cell value is computed using **averaging**:

```
corner_ghost = 0.5 * (adjacent_edge_ghost_1 + adjacent_edge_ghost_2)
```

### Why Averaging?

1. **Numerical stability**: Avoids discontinuities at corners
2. **BC agnostic**: Works for any combination of BC types
3. **Smooth solutions**: Produces well-behaved ghost values

### Corner Strategies (Advanced)

For special cases, alternative strategies may be more appropriate:

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| **Average** (default) | Mean of adjacent ghost values | General purpose |
| **Priority** | Dirichlet takes precedence over Neumann | Sharp BC interfaces |
| **Extrapolate** | From interior points | When BCs don't dominate |

The default averaging strategy is recommended for most MFG applications.

## Time-Dependent BCs

BCs can vary with time:

```python
# Time-varying Dirichlet
def inlet_profile(t):
    return 1.0 - np.exp(-t)

bc = dirichlet_bc(dimension=2, value=inlet_profile)

# Apply at specific time
padded = apply_boundary_conditions_2d(field, bc, bounds, time=0.5)
```

For spatially-varying BCs:

```python
# Function of position and time
def inlet_profile(point, time):
    x, y = point
    return np.sin(np.pi * y) * np.exp(-time)

exit_bc = BCSegment(
    name="exit",
    bc_type=BCType.DIRICHLET,
    value=inlet_profile,
    boundary="x_max",
)
```

## Lazy Dimension Binding

BCs can be created without specifying dimension, then bound when attached to geometry:

```python
# Create BC without dimension
bc = neumann_bc()  # dimension=None

# Dimension auto-detected when used with geometry
grid = TensorProductGrid(bounds=[[0,1], [0,1]], num_points=[51, 51])
grid.boundary_conditions = bc  # Automatically binds to dimension=2
```

## Solver-Specific Notes

### HJB Solvers (FDM, WENO, GFDM)

Ghost values are used for upwind gradient computation:
- At left boundary with rightward flow: backward difference uses ghost
- At right boundary with leftward flow: forward difference uses ghost

### FP Solvers (FDM)

**Mass conservation at a no-flux wall does not come from ghost values.** The divergence-form
discretisation writes a conservation row for the wall cell with the wall-face flux set to zero
(`fp_fdm_alg_divergence_centered.py:350-431`), which is what makes `J.n = 0` exact. That row does
not constrain `dm/dn`, and with drift at the wall the density gradient is large: measured with
`v = -1`, `D = 0.045` on 201 points, mass drifts `1.5e-13` while `dm/dx` at the wall is `-443`,
matching the analytic `exp(v*x/D)` profile to `1e-3`.

`ghost = interior` is the **zero-gradient** construction, and it is not mass-conserving for the FP
equation: it gives `J.n = v*m_interior`, which vanishes only when `v = 0`. It is what
`ZeroGradientCalculator` does, and what the default applicator path uses; `ZeroFluxCalculator`
(`ghost_cells.py:361`) is the one that constructs `ghost = interior*(2D + v*dx)/(2D - v*dx)` so
that the total flux vanishes.

- Dirichlet BC: ghost reflects the boundary value

### Particle Methods

Particles are reflected at boundaries:
- Normal reflection for no-flux
- Absorption for Dirichlet (particle removed or reset)

## Periodic BC: Model Compatibility

When using periodic boundary conditions, all physical quantities (potential, running cost, terminal cost) **must** be periodic functions with continuous derivatives. Violating this causes gradient discontinuities at the boundary.

### The Problem: Quadratic Potential on Periodic Domain

A standard quadratic potential $V(x) = C(x - L/2)^2$ has opposite gradients at opposite boundaries:

- At $x = 0$: $\partial V / \partial x = -CL$
- At $x = L$: $\partial V / \partial x = +CL$

Since periodic BC identifies $x = 0$ with $x = L$, this creates a gradient jump of $2CL$, causing numerical instability in HJB time-stepping and spurious oscillations in FP density.

### The Solution: Use Periodic-Compatible Functions

Replace non-periodic functions with their periodic counterparts:

```python
import numpy as np

# WRONG: Quadratic potential on periodic domain
V_quad = lambda x: C * (x - L/2)**2  # Gradient discontinuity at boundary!

# CORRECT: Cosine potential (periodic, smooth)
V_periodic = lambda x: C/2 * (1 + np.cos(2 * np.pi * x / L))
# V_min = 0 at center, V_max = C at boundaries
# Gradient is continuous: dV/dx = 0 at both x=0 and x=L
```

### Decision Guide

| Physical Setup | Recommended BC | Potential Form |
|----------------|----------------|----------------|
| Bounded domain with walls | No-flux (Neumann) | Quadratic $V \propto r^2$ |
| Infinite periodic lattice | Periodic | Cosine $V \propto \cos(kx)$ |
| Torus topology | Periodic | Cosine $V \propto \cos(kx)$ |

**Rule of thumb**: If any function in your MFG model is not periodic on the domain, use Neumann/no-flux BC instead.

## Performance Tips

1. **Pre-compute masks**: For mixed BCs, use `create_boundary_mask_2d()` to identify segments once
2. **Avoid callable BCs when possible**: Constant values are faster
3. **Use uniform BCs**: Simpler path, less overhead than mixed BCs

## Common Issues

### "BC dimension not set"

```python
# Wrong: using unbound BC
bc = neumann_bc()  # dimension=None
apply_boundary_conditions_2d(field, bc, bounds)  # Error!

# Correct: bind dimension first
bc = bc.bind_dimension(2)
apply_boundary_conditions_2d(field, bc, bounds)  # Works
```

### "domain_bounds required for mixed BC"

```python
# Wrong: mixed BC without bounds
bc = mixed_bc([seg1, seg2], dimension=2)
apply_boundary_conditions_2d(field, bc)  # Error!

# Correct: provide bounds
apply_boundary_conditions_2d(field, bc, domain_bounds=bounds)
```

## See Also

- `mfgarchon.geometry.boundary` module documentation
- `docs/user/advanced_boundary_conditions.md` - Variational inequalities, moving boundaries
