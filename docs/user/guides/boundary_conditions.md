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
| `DIRICHLET` | `u = g` | shared BC: **`m = 0`** (absorbing); a BC passed to the FP solver itself: `m = g` | Exit with exit cost `g`; fixed value |
| `NEUMANN` | `du/dn = g` | see note below | Prescribed boundary data |
| `NO_FLUX` | `du/dn = 0` | **`J.n = 0`**, where `J = v*m - D*grad(m)` | Reflecting / insulating wall; the mass-conserving choice |
| `ROBIN` | `alpha*u + beta*du/dn = g` | shared BC: refused; a BC passed to the FP solver itself: **`J.n = (D/beta)(alpha*m - g)`** (FP-FEM only) | Mixed condition; a finite-rate exit |
| `PERIODIC` | `u(x_min) = u(x_max)` | `m(x_min) = m(x_max)` | Wrap-around domain |

**`DIRICHLET` on a shared BC is an exit** (#2512, convention row 5). One `DIRICHLET(g)` on the
problem or geometry cannot mean `u = g` and `m = g` at once: the HJB reads the exit cost `u = g`,
and the FP solvers that accept DIRICHLET -- FDM, FEM, meshless Galerkin and particle -- read an
absorbing wall `m = 0` through `bc_utils.fp_view_of_shared_bc`. FP-FVM, FP-GFDM and FP-SL refuse
DIRICHLET at any `g`, shared or explicit. A BC passed to an FP solver explicitly is the FP's own,
where `DIRICHLET(g)` is a prescribed density `m = g`: `FPFDMSolver(boundary_conditions=...)` and
`FPFEMSolver(boundary_conditions=...)` impose one; the particle FP and meshless Galerkin refuse a nonzero
`g` there ([Issue #2532](https://github.com/derrring/MFGArchon/issues/2532)).

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

> **Note on `NEUMANN` on the FP side.** On the problem's shared BC, `neumann_bc(value=g)` is the HJB's
> `du/dn = g`. Every FP solver reads `neumann_bc()` (`g = 0`) as zero flux `J.n = 0`, and refuses a
> non-zero `g`, which says nothing about the agents' mass at the wall: keep `neumann_bc(value=g)` on the
> problem for the HJB, and give the FP solver its own `boundary_conditions=no_flux_bc(...)`
> ([Issue #2512](https://github.com/derrring/MFGArchon/issues/2512), row B3). A shared `ROBIN` is the
> HJB's too and is refused at the FP whatever its coefficients: an FP Robin is a condition on the total
> flux, `J.n = (D/beta)(alpha*m - g)` (see `docs/user/CONVENTIONS.md`). Handed to an FP solver
> explicitly, as in `FPFDMSolver(problem, boundary_conditions=...)`, a NEUMANN would mean `dm/dn = g`,
> which no FP solver implements, so it is refused: use `no_flux_bc()` for a reflecting FP wall. The
> semi-Lagrangian and meshless Galerkin HJB solvers refuse a non-zero `g`
> ([Issue #1686](https://github.com/derrring/MFGArchon/issues/1686)).

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
import numpy as np

from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, FDMApplicator, mixed_bc_from_regions, no_flux_bc

# Create geometry
geometry = TensorProductGrid(bounds=[(0, 2), (0, 1)], Nx_points=[41, 21], boundary_conditions=no_flux_bc(dimension=2))

# Mark regions on whole faces
geometry.mark_region("inlet", boundary="x_min")
geometry.mark_region("outlet", boundary="x_max")

# Define BCs referencing regions; "default" covers every face no region claims
bc_config = {
    "inlet": BCSegment(name="inlet_bc", bc_type=BCType.DIRICHLET, value=1.0),
    "outlet": BCSegment(name="outlet_bc", bc_type=BCType.NEUMANN, value=0.0),
    "default": BCSegment(name="wall_bc", bc_type=BCType.NO_FLUX),
}

bc = mixed_bc_from_regions(geometry, bc_config)

# Apply BCs (must pass geometry parameter)
field = np.zeros(tuple(geometry.Nx_points))
applicator = FDMApplicator(dimension=2)
padded = applicator.apply(field, bc, domain_bounds=np.array(geometry.bounds), geometry=geometry)
```

**A face-level path imposes one condition per face** (`FDMApplicator`, `pad_array_with_ghosts`, and
the FDM operators the solvers use). On a structured grid, `mixed_bc_from_regions` resolves a region
that covers whole faces, and no part of any other face, to those faces when it builds the BC, so every reader applies it without
needing the geometry (#2472). A face counts as covered when the region holds all of its interior, the
face without the edges it shares with other faces. So whether a strip covers a face depends on the
resolution.

A region that covers only part of a face keeps its region name, and a face-level reader refuses it
rather than stretching it to the whole face or dropping it. Such a region may not be named like a face
(`"top"`, `"x_max"`), because a reader without the geometry would read the name as that face. The
only region, on every face, is a uniform condition.

### Region Marking Methods

Regions can be marked using boundaries, custom predicates or masks:

```python
# Method 1: Mark using boundary identifier
geometry.mark_region("left_wall", boundary="x_min")

# Method 2: Mark using predicate function
geometry.mark_region("inlet_strip", predicate=lambda x: (x[:, 0] < 0.1) & (x[:, 1] > 0.5))

# Method 3: Combine whole faces in one region, through the grid's union of marked regions
geometry.mark_region("bottom", boundary="y_min")
geometry.mark_region("top", boundary="y_max")
geometry.mark_region("walls", mask=geometry.union_regions("bottom", "top"))
```

Predicates receive grid points as `(N, dimension)` array and must return boolean mask of length `N`.
A name is marked once: a second `mark_region` with the same name raises (`Region 'walls' already
exists`), so a region over several faces is marked whole, as in Method 3.

### Priority Resolution for Overlapping Regions

When regions overlap on a face, **priority** decides which segment governs it (higher number = higher
precedence). With the regions of Method 3, `walls` (both y faces) and `top` (y_max, a whole face, which
may carry a face's name):

```python
from mfgarchon.geometry.boundary import BoundaryConditions

bc_top = BCSegment(name="top_bc", bc_type=BCType.DIRICHLET, value=1.0, region_name="top", priority=2)
bc_walls = BCSegment(name="walls_bc", bc_type=BCType.NO_FLUX, region_name="walls", priority=1)

bc = BoundaryConditions(
    dimension=2,
    segments=[bc_top, bc_walls],  # Order doesn't matter
    default_bc=BCType.NO_FLUX,
    domain_bounds=np.array(geometry.bounds),
)

# y_max is in both regions: top_bc governs it, by priority
# y_min is in walls only: walls_bc governs it
# x_min and x_max are in neither: default_bc governs them
padded = applicator.apply(field, bc, domain_bounds=np.array(geometry.bounds), geometry=geometry)
```

The face-level path and the per-point reader, `bc.get_bc_at_point(point, geometry=geometry)`, give
the same answer here. Build such an overlap from `region_name` segments, as above:
`mixed_bc_from_regions` gives each face of `walls` its own segment, and a face-level reader then
refuses the BC, since the one on `y_max` governs nothing.

**Regions that overlap on part of a face**, such as two strips `x < 0.5` and `x < 0.1`, cannot be
imposed by a face-level path, which refuses them (#2472). On a `TensorProductGrid` they are read per
point, where the higher priority wins at a point both hold, by these solvers and for these conditions:

| a region on part of a face, placed by | `HJBGFDMSolver` | `FPParticleSolver`, array drift (as in a coupled solve) | `FPParticleSolver`, callable drift |
|---|---|---|---|
| `region=` or `sdf_region` | when no condition of the BC is PERIODIC, REFLECTING, or a ROBIN with nonzero `alpha` or with `beta` other than 1 | when no condition of the BC is ROBIN or a nonzero NEUMANN, and the region and the rest of its face are both PERIODIC or both not | when no condition of the BC is ROBIN or a nonzero NEUMANN |
| `mark_region` | as for `region=` | as for `region=`; it reads the marked nodes' cells, which reach half a grid spacing past them | as for `region=`, with the same reach |

A coupled `HJBGFDMSolver` + `FPParticleSolver` solve reads such a region when both of its columns
do. The conditions measured, each as the region's and as the rest's, are DIRICHLET 0, NO_FLUX,
NEUMANN 0 and 0.5, REFLECTING, ROBIN with `alpha` 1 and 0 (and `beta` 1 and 2), and PERIODIC.

The table is measured, and holds, for the problem's own BC (not one passed to `FPParticleSolver` as
`boundary_conditions=`, which refuses any NEUMANN and a nonzero DIRICHLET), on the grid's box (no
`implicit_domain=`, which refuses an exit placed by `region=` alone or by `sdf_region`), with
`corner_strategy="priority"`, constant condition values, no `flux_capacity`, the array drift given
as an array of U (not as velocities with `drift_is_precomputed=True`), and a region that holds at
least one grid node. Outside those the table is not measured; the refusals measured there are named
above. On a mesh, the FEM solvers read a `region=` placement per facet
(`BoundaryConditions.segments_for_facets`).

### Boundary-Based vs Region-Based

| Approach | Use When | Example |
|----------|----------|---------|
| **Boundary-based** | Simple geometries, entire boundaries | `boundary="x_min"` |
| **Region-based** | Complex geometries; part of a face only per point on a grid, or per facet on a mesh (above) | `region={"x": (0.0, 0.1)}` |

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
from mfgarchon.geometry.boundary import pad_array_with_ghosts

# field has shape (Ny, Nx) - interior points only.
# The domain bounds travel on `bc`; they are not a separate argument.
padded = pad_array_with_ghosts(field, bc, spacing=(dy, dx))
# padded has shape (Ny+2, Nx+2) - includes ghost cells
```

For a time-stepping loop, allocate once with `PreallocatedGhostBuffer` instead;
`pad_array_with_ghosts` allocates a new array per call.

### Ghost Cell Formulas

A `TensorProductGrid` puts a node on the wall, so the finite-difference ghost is node-centred: ghost layer `k` sits `(k+1)*dx` outside the wall node `u_b`, its mirror `u_m` the same distance inside, and the condition is imposed by the central difference about the wall node (#1935).

- **Robin** (alpha*u + beta*du/dn = g): `u_ghost = u_m + 2*(k+1)*dx*(g - alpha*u_b)/beta`. `du/dn` is the outward derivative and already carries the wall's direction, so there is no sign factor.
- **Neumann** (du/dn = g): Robin with alpha = 0, beta = 1, so `u_ghost = u_m + 2*(k+1)*dx*g`.
- **No-flux** and **reflecting** (du/dn = 0): `u_ghost = u_m`, the mirror about the wall node. This is the **zero-gradient** ghost, correct for `u` and for the default applicator path. It is *not* the FP mass-conserving wall; see [FP Solvers (FDM)](#fp-solvers-fdm) below.
- **Dirichlet** (u = g), and Robin with beta = 0: `u_ghost = 2*g - u_m` (with `g/alpha` for Robin), the odd reflection about the wall node (#1968). Until #1968 this was the cell-centred `2*g - u_b`, which #1935, concerned with the derivative conditions, had left alone.

Until #1935 the derivative conditions used the cell-centred mirror `u_ghost = u_b + dx*g`. On this grid that made the 3-point Laplacian's wall row return half the true value at every `dx`.

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

A callable value is `g(t, x)`, time first, like every callable the library accepts (see
`docs/user/CONVENTIONS.md`, § *Callable signatures*). It names what it reads, `(t, x)`, `(x)`, or `(t)` (or
`time`), and the library calls it by name; any other parameter needs a default, and keeps it. One that puts `x`
before `t`, or has a parameter without a default named otherwise, is refused when the segment is built (#2512).

BCs can vary with time:

```python
# Time-varying Dirichlet
def inlet_profile(t):
    return 1.0 - np.exp(-t)

bc = dirichlet_bc(dimension=2, value=inlet_profile)

# Apply at specific time
padded = pad_array_with_ghosts(field, bc, time=0.5, spacing=(dy, dx))
```

For spatially-varying BCs, `x` is one boundary point, an array of shape `(d,)`:

```python
# Function of time and position
def inlet_profile(t, x):
    return np.sin(np.pi * x[1]) * np.exp(-t)

exit_bc = BCSegment(
    name="exit",
    bc_type=BCType.DIRICHLET,
    value=inlet_profile,
    boundary="x_max",
)
```

Not every method can read every form. Each refuses, when it first reads the value, one that declares what it does
not have:
- a method that reads one value per face, such as the FDM ghost layer, has no point to pass, so it refuses a value
  that declares `x`, even with a default: write `g(t)` if the value does not vary along the face;
- FEM's Dirichlet DOFs and the meshless Galerkin path read `g(x)` at each boundary point but have no time to pass
  yet, so they refuse a value that declares `t`. FEM's Robin and Neumann loads read one constant per segment, with
  neither a time nor a point, so they refuse a callable that declares either;
- HJB-GFDM (through `BCSegment.get_value`) passes both, so it reads `inlet_profile(t, x)` above.

A boundary value is a function you write, a `def` or a `lambda` (or a `functools.partial` of one), since its
parameter names are what tell the library what it reads. Any other callable is refused when the segment is built:
a SciPy interpolant, a class instance, a bound method, a builtin or NumPy ufunc, a vectorized or jax-wrapped
function. Wrap it in a function that names what it reads. An interpolant over time data takes `(x, ...)` and would
read the time data at the position, so for `s = CubicSpline(times, values)` or `s = interp1d(times, values)` pass
`value=lambda t: s(t)`; for `np.sin`, `value=lambda t: np.sin(t)`.

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

- Dirichlet BC: the wall row is the identity, `m = value`, not a ghost -- 0 on a shared BC, where the
  wall is an exit (absorbing), and `g` on a BC passed to `FPFDMSolver(boundary_conditions=...)` itself.

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

1. **Reuse the buffer**: in a time-stepping loop use `PreallocatedGhostBuffer`, not `pad_array_with_ghosts` -- the latter allocates a padded array on every call
2. **Avoid callable BCs when possible**: Constant values are faster
3. **Use uniform BCs**: Simpler path, less overhead than mixed BCs

## Common Issues

Two entries that stood here documented errors of the removed
`apply_boundary_conditions_2d`, which took `domain_bounds` as a call argument.
Neither reproduces on the canonical path, measured:

- **An unbound BC does not raise.** `pad_array_with_ghosts` reads the dimension
  off the array it is padding, so `neumann_bc()` with `dimension=None` pads
  identically to `neumann_bc(dimension=2)`. `bind_dimension` still matters for
  code that queries the BC itself -- `bc._require_dimension` does raise -- but
  not for padding.
- **A mixed BC without `domain_bounds` does not raise.** Segments carrying a
  `boundary=` face name are matched symbolically through `parse_boundary_face`,
  so the bounds are never consulted; with and without them the padded array is
  element-wise identical. Bounds are needed only by segments that select a
  region by coordinate range (`region=`) or by SDF.

## See Also

- `mfgarchon.geometry.boundary` module documentation
- `docs/user/advanced_boundary_conditions.md` - Variational inequalities, moving boundaries
