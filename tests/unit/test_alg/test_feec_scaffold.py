"""FEEC infrastructure scaffold: the mixed RT0 x P0 structure-preserving building blocks assemble and
satisfy the MixedWeakFormDiscretization protocol; the research parts (velocity->flux projection) fail
loud rather than silently returning garbage."""

from __future__ import annotations

import pytest

import numpy as np

skfem = pytest.importorskip("skfem", reason="scikit-fem required")


def _disc():
    from mfgarchon.alg.numerical.feec import RaviartThomasDiscretization

    return RaviartThomasDiscretization(skfem.MeshTri.init_sqsymmetric().refined(2))


def test_mixed_discretization_building_blocks_assemble():
    from mfgarchon.alg.numerical.feec import MixedWeakFormDiscretization

    disc = _disc()
    assert isinstance(disc, MixedWeakFormDiscretization)
    assert disc.dim == 2
    assert disc.flux_dof > 0
    assert disc.density_dof > 0

    b = disc.divergence()
    assert b.shape == (disc.density_dof, disc.flux_dof)  # exact-divergence coupling P0 x RT0

    mf = disc.flux_mass()
    assert mf.shape == (disc.flux_dof, disc.flux_dof)
    assert np.allclose(mf.toarray(), mf.toarray().T)  # H(div) mass is symmetric

    md = disc.density_mass()
    assert md.shape == (disc.density_dof, disc.density_dof)
    # P0 density mass is diagonal (elementwise) -> the L2 pairing is block-local
    assert md.nnz == disc.density_dof


def test_velocity_projection_is_failloud_scaffold():
    disc = _disc()
    with pytest.raises(NotImplementedError, match="scaffold"):
        disc.project_velocity_to_flux(np.zeros((2, disc.density_dof)))


def test_non_simplicial_mesh_fails_loud():
    from mfgarchon.alg.numerical.feec import RaviartThomasDiscretization

    with pytest.raises(NotImplementedError, match="simplicial"):
        RaviartThomasDiscretization(skfem.MeshQuad())
