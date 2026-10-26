"""
Particle Boundary Condition Utilities (1D, GPU/CPU). Deprecated (#2550).

These two functions were a second copy of `mfgarchon.utils.particle_utils.apply_boundary_conditions_gpu` /
`_numpy`, the pair the particle solver's 1-D GPU path calls. They now call that pair, which folds and wraps through
the owners (`mfgarchon.geometry.boundary.corner.reflect_positions`, `mfgarchon.geometry.boundary.periodic.wrap_positions`;
CONVENTIONS.md § *Particles at a wall*) and refuses "dirichlet" by name.

For n-D positions, use the owners directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mfgarchon.utils.deprecation import deprecated

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mfgarchon.backends.base_backend import BaseBackend


@deprecated(
    since="v0.22.0",
    replacement="mfgarchon.utils.particle_utils.apply_boundary_conditions_gpu",
    reason="a second copy of the particle solver's 1-D helper (#2550)",
)
def apply_boundary_conditions_gpu(
    particles,
    xmin: float,
    xmax: float,
    bc_type: str,
    backend: BaseBackend,
):
    """Apply 1-D boundary conditions to particles on a backend. Deprecated: it is
    `mfgarchon.utils.particle_utils.apply_boundary_conditions_gpu`, which refuses "dirichlet" by name."""
    from mfgarchon.utils.particle_utils import apply_boundary_conditions_gpu as owner

    return owner(particles, xmin, xmax, bc_type, backend)


@deprecated(
    since="v0.22.0",
    replacement="mfgarchon.utils.particle_utils.apply_boundary_conditions_numpy",
    reason="a second copy of the particle solver's 1-D helper (#2550)",
)
def apply_boundary_conditions_numpy(
    particles: NDArray,
    xmin: float,
    xmax: float,
    bc_type: str,
) -> NDArray:
    """Apply 1-D boundary conditions to particles with NumPy. Deprecated: it is
    `mfgarchon.utils.particle_utils.apply_boundary_conditions_numpy`, which folds and wraps through the owners and
    refuses "dirichlet" by name."""
    from mfgarchon.utils.particle_utils import apply_boundary_conditions_numpy as owner

    return owner(particles, xmin, xmax, bc_type)


__all__ = [
    "apply_boundary_conditions_gpu",
    "apply_boundary_conditions_numpy",
]
