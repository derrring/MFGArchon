"""
Finite Element Method (FEM) solvers for MFG systems.

Uses scikit-fem (skfem) as the assembly backend for stiffness, mass, and
advection matrices on unstructured meshes. The coupling layer, boundary
condition handling, and MFG-specific logic remain in-house.

Requires: pip install scikit-fem

Issue #773: scikit-fem integration

Usage:
    >>> from mfgarchon.alg.numerical.fem import HJBFEMSolver, FPFEMSolver
    >>> hjb_solver = HJBFEMSolver(problem)
    >>> fp_solver = FPFEMSolver(problem)
"""

from mfgarchon.alg.numerical.fem.assembly import (
    apply_dirichlet_bc,
    assemble_advection,
    assemble_gradient_projection,
    assemble_mass,
    assemble_stiffness,
    create_basis,
)
from mfgarchon.alg.numerical.fem.discretization import FEMDiscretization
from mfgarchon.alg.numerical.fem.fp_fem_solver import FPFEMSolver
from mfgarchon.alg.numerical.fem.hjb_fem_solver import HJBFEMSolver
from mfgarchon.alg.numerical.fem.mesh_adapter import (
    meshdata_to_skfem,
    skfem_to_meshdata,
)

__all__ = [
    "HJBFEMSolver",
    "FPFEMSolver",
    "FEMDiscretization",
    "create_basis",
    "assemble_stiffness",
    "assemble_mass",
    "assemble_advection",
    "assemble_gradient_projection",
    "apply_dirichlet_bc",
    "meshdata_to_skfem",
    "skfem_to_meshdata",
]
