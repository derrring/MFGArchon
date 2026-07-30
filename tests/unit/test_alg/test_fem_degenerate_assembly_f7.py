"""Issue #1489 (F7): FEM basis creation fails loud on a degenerate (zero-measure) element instead of
silently assembling NaN stiffness/mass entries (only a numpy RuntimeWarning otherwise)."""

from __future__ import annotations

import pytest

skfem = pytest.importorskip("skfem", reason="scikit-fem required")


def test_create_basis_fails_loud_on_degenerate_element():
    from mfgarchon.alg.numerical.fem.assembly import create_basis

    m = skfem.MeshTri.init_sqsymmetric()
    p = m.p.copy()
    p[:, 1] = p[:, 0]  # collapse a vertex onto another -> zero-area (degenerate) triangles
    degenerate = skfem.MeshTri(p, m.t)
    with pytest.raises(ValueError, match="degenerate"):
        create_basis(degenerate, order=1)


def test_create_basis_accepts_valid_mesh():
    from mfgarchon.alg.numerical.fem.assembly import create_basis

    basis = create_basis(skfem.MeshTri.init_sqsymmetric().refined(1), order=2)  # no raise
    assert basis.N > 0


def test_a_degenerate_element_corrupts_stiffness_without_raising():
    """The guard's premise, measured: assembly does not fail, it returns NaN.

    Issue #1714: 47 of the 85 tests the fail-loud campaign added can only fail if the guard is
    deleted, and this file was one of the 20 with no numeric assertion. `pytest.raises` on the
    guard's own message records that the guard fires; it does not record why refusing is better
    than proceeding.

    Bypassing `create_basis` and assembling directly on a mesh with one collinear triangle shows
    exactly what the guard prevents, and two details sharper than the docstring's summary:

    - the corruption is in the STIFFNESS matrix, not the mass matrix -- laplace divides by detA,
      mass does not;
    - no warning is emitted during `asm` at all. The RuntimeWarnings appear earlier, when the
      basis builds its inverse affine map, so a caller who assembles from a basis handed to them
      sees nothing.
    """
    import warnings

    from skfem.models.poisson import laplace, mass

    import numpy as np

    # Node 3 is collinear with nodes 0 and 1, so the second triangle has zero area.
    points = np.array([[0.0, 1.0, 0.0, 2.0], [0.0, 0.0, 1.0, 0.0]])
    elements = np.array([[0, 0], [1, 1], [2, 3]])
    mesh = skfem.MeshTri(points, elements)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        basis = skfem.Basis(mesh, skfem.ElementTriP1())
        measures = np.asarray(basis.dx).sum(axis=1)

    assert measures[0] > 0.1, f"the healthy triangle must have real area, got {measures[0]}"
    assert measures[1] == pytest.approx(0.0, abs=1e-15), (
        f"the collinear triangle must have ~zero measure, got {measures[1]}"
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        stiffness = skfem.asm(laplace, basis)
        mass_matrix = skfem.asm(mass, basis)

    assert not caught, (
        f"assembly emitting no warning is what makes this silent; if it ever warns, the guard's "
        f"premise has changed. Got {[type(c.message).__name__ for c in caught]}"
    )
    assert not np.all(np.isfinite(stiffness.toarray())), (
        "the stiffness matrix must be corrupted -- that is the silent-wrong outcome the guard "
        "refuses. If it is finite, the guard is refusing a mesh that would have assembled."
    )
    assert np.all(np.isfinite(mass_matrix.toarray())), (
        "the mass matrix stays finite, so a caller checking only M would see nothing wrong"
    )
