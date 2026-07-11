"""Issue #1558 (defect 1): MeshfreeApplicator read Robin alpha/beta off the wrong object.

alpha/beta live on the BCSegment, not on BoundaryConditions, so ``getattr(bc, "beta", 0.0)``
always read 0.0 -- silently collapsing every Robin BC to pure Dirichlet: the field path forced a
hard ``u = g/alpha`` (never the penalty blend) and the particle path always absorbed (never
reflected). These pin the corrected reads (from the ROBIN segment). Off published numerics
(published adjoint-consistent Robin runs through the hjb_gfdm row builder, not MeshfreeApplicator).
"""

from __future__ import annotations

import numpy as np

from mfgarchon.geometry.boundary import robin_bc
from mfgarchon.geometry.boundary.applicator_meshfree import MeshfreeApplicator
from mfgarchon.geometry.implicit.hyperrectangle import Hyperrectangle


def test_field_robin_uses_penalty_blend_not_hard_dirichlet():
    """apply() with beta != 0 must use the penalty blend, not collapse to hard u = g/alpha."""
    domain = Hyperrectangle(bounds=[(0.0, 1.0)])
    applicator = MeshfreeApplicator(domain)

    g = 1.0
    bc = robin_bc(value=g, alpha=1.0, beta=1.0, dimension=1)  # penalty_weight = |alpha/beta| = 1
    points = np.array([[0.0], [0.5], [1.0]])  # x=0 and x=1 are on the boundary
    field = np.zeros(3)

    out = applicator.apply(field.copy(), bc, points)

    # penalty blend at a boundary point: (field_old + w*g) / (1 + w) = (0 + 1*1)/(1+1) = 0.5
    # hard Dirichlet (the collapsed bug) would give g/alpha = 1.0.
    assert np.isclose(out[0], 0.5), f"left boundary {out[0]} != penalty blend 0.5 (0.0/1.0 would be the bug)"
    assert np.isclose(out[2], 0.5), f"right boundary {out[2]} != penalty blend 0.5"
    assert np.isclose(out[1], 0.0), "interior point must be untouched"


def test_particles_robin_neumann_like_reflects_not_absorbs():
    """apply_particles() with alpha=0, beta!=0 (Neumann-like Robin) must reflect, not absorb."""
    domain = Hyperrectangle(bounds=[(0.0, 1.0)])
    applicator = MeshfreeApplicator(domain)

    bc = robin_bc(value=0.0, alpha=0.0, beta=1.0, dimension=1)  # alpha=0 -> reflecting
    particles = np.array([[1.2]])  # one particle stepped outside the right wall

    out = applicator.apply_particles(particles.copy(), bc)

    # Reflecting keeps the particle (projected back inside); absorbing (the bug) removes it.
    assert out.shape[0] == 1, f"reflecting must keep the particle, got {out.shape[0]} (0 == absorbed bug)"
    x = float(out.ravel()[0])
    assert 0.0 <= x <= 1.0, f"reflected particle must be inside the domain, got {x}"


def test_validate_bc_compatibility_emits_no_bc_type_verdict():
    """#1558 defect 3: validate_bc_compatibility no longer emits a per-discretization BC-type verdict.

    The old hardcoded table was a second, contradictory capability source: it flagged Robin as
    "limited" for GFDM while hjb_gfdm._SUPPORTED_BC_TYPES actually includes Robin (the #1456 single
    source). A uniform robin_bc has default_bc=ROBIN, so the old code returned that wrong verdict;
    now the function checks only dimension compatibility, so it returns [].
    """
    from mfgarchon.geometry import TensorProductGrid
    from mfgarchon.geometry.boundary import no_flux_bc, robin_bc
    from mfgarchon.geometry.boundary.dispatch import validate_bc_compatibility

    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=no_flux_bc(dimension=1))
    robin = robin_bc(dimension=1, alpha=1.0, beta=1.0)
    assert validate_bc_compatibility(robin, grid, discretization="GFDM") == []  # old table said "limited"
    assert validate_bc_compatibility(robin, grid, discretization="FDM") == []

    # The one real check still fires: a 2D BC on a 1D geometry is a genuine dimension mismatch.
    issues = validate_bc_compatibility(no_flux_bc(dimension=2), grid, discretization="FDM")
    assert any("Dimension mismatch" in m for m in issues)
