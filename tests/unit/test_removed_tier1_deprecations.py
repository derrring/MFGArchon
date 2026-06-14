"""Removed-deprecation pins for the Tier-1 past-window families.

These families had no prior equivalence test, so this file pins their removal:
the old parameter name must now fail loud (``TypeError`` -- unexpected keyword
argument) while the new name keeps working. Covers:

- Lowercase grid params: ``nx/ny/nz/nt`` -> ``Nx/Ny/Nz/Nt``
  (``MFGSystemBuilder.domain``, ``SparseMatrixOptimizer.create_laplacian_3d``)
- Mesh-IO param: ``format_type`` -> ``file_format`` (``Mesh1D.export_mesh``)

The FP-particle family pins live with their solver tests
(``tests/unit/test_alg/test_fp_particle*.py``).
"""

from __future__ import annotations

import pytest

from mfgarchon.geometry.meshes.mesh_1d import Mesh1D
from mfgarchon.meta.mathematical_dsl import MFGSystemBuilder
from mfgarchon.utils.performance.optimization import SparseMatrixOptimizer


class TestLowercaseGridParamsRemoved:
    """``nx/ny/nz/nt`` are gone; only ``Nx/Ny/Nz/Nt`` remain."""

    def test_domain_nx_removed(self):
        with pytest.raises(TypeError, match="unexpected keyword argument 'nx'"):
            MFGSystemBuilder().domain(0.0, 1.0, 1.0, nx=10)

    def test_domain_nt_removed(self):
        with pytest.raises(TypeError, match="unexpected keyword argument 'nt'"):
            MFGSystemBuilder().domain(0.0, 1.0, 1.0, nt=5)

    def test_domain_new_names_accepted(self):
        builder = MFGSystemBuilder().domain(0.0, 1.0, 1.0, Nx=33, Nt=7)
        assert builder.domain_info["Nx"] == 33
        assert builder.domain_info["Nt"] == 7

    def test_create_laplacian_3d_lowercase_removed(self):
        for kw in ("nx", "ny", "nz"):
            with pytest.raises(TypeError, match=f"unexpected keyword argument '{kw}'"):
                SparseMatrixOptimizer.create_laplacian_3d(**{kw: 3})

    def test_create_laplacian_3d_new_names_accepted(self):
        mat = SparseMatrixOptimizer.create_laplacian_3d(Nx=3, Ny=3, Nz=3)
        assert mat.shape == (27, 27)


class TestMeshFormatTypeRemoved:
    """``format_type`` is gone; only ``file_format`` remains on ``export_mesh``."""

    def test_export_mesh_format_type_removed(self):
        mesh = Mesh1D(bounds=(0.0, 1.0), num_elements=4)
        with pytest.raises(TypeError, match="unexpected keyword argument 'format_type'"):
            mesh.export_mesh(format_type="vtk", filename="unused.vtk")
