"""The weak-form FP adjoint step refuses a Dirichlet wall it cannot impose, as FP-FDM's does (#2608, #2531).

`WeakFormFPSolver.solve_fp_step_adjoint_mode` solves ``M/dt + A^T + D K`` with no condensation and no Nitsche term,
so a Dirichlet wall was dropped in silence. Measured at 375122e5 on the issue's fixture (FP-FEM on
`Mesh1D(bounds=(0, 1), num_elements=8)`, its own `dirichlet_bc(value=0.7)`, m0 = 1, zero drift): one adjoint step left
the walls at [1. 1.], where the forward step gives [0.7 0.7]. `BlockIterator(adjoint_mode="jacobian_transpose")` ran
that pairing to convergence with the walls at 1. It now raises at the first adjoint step and points to
``adjoint_mode='off'``. A BC with no Dirichlet wall, no-flux or Robin, is solved as before.
"""

from __future__ import annotations

import warnings

import pytest

import numpy as np
from scipy import sparse

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, dirichlet_bc, no_flux_bc, robin_bc

REFUSAL = r"solve_fp_step_adjoint_mode imposes no Dirichlet wall.*adjoint_mode='off'"


def _problem(geometry):
    zero = lambda m: 0.0 * np.asarray(m)  # noqa: E731
    hamiltonian = SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0), coupling=zero, coupling_dm=zero)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MFGProblem(
            model=Model(hamiltonian=hamiltonian, volatility=0.4),
            domain=geometry,
            conditions=Conditions(m_initial=lambda p: 1.0, u_terminal=lambda p: 0.0, T=0.5),
            Nt=10,
        )


def _fem(bc):
    from mfgarchon.alg.numerical.fem import FPFEMSolver
    from mfgarchon.geometry.meshes.mesh_1d import Mesh1D

    geometry = Mesh1D(bounds=(0.0, 1.0), num_elements=8)
    geometry.generate_mesh()
    geometry.boundary_conditions = no_flux_bc(dimension=1)
    return FPFEMSolver(_problem(geometry), order=1, boundary_conditions=bc)


def _meshless(bc):
    from mfgarchon.alg.numerical.meshless_galerkin.fp_solver import MeshlessGalerkinFPSolver
    from mfgarchon.geometry import TensorProductGrid

    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1))
    return MeshlessGalerkinFPSolver(
        _problem(grid), np.linspace(0.0, 1.0, 21).reshape(-1, 1), delta=0.35, boundary_conditions=bc
    )


def _adjoint_step(solver):
    n = solver.n_dof
    return solver.solve_fp_step_adjoint_mode(np.ones(n), sparse.csr_matrix((n, n)))


def _dirichlet_only_by_default():
    """A no-flux wall on x_min and nothing on x_max, which falls through to a DIRICHLET(0.7) default: FP-FEM imposes
    that default (#2595), so the wall is there with no DIRICHLET segment for a test on segment types to see."""
    wall = BCSegment(name="wall", bc_type=BCType.NO_FLUX, boundary="x_min")
    return BoundaryConditions(dimension=1, segments=[wall], default_bc=BCType.DIRICHLET, default_value=0.7)


@pytest.mark.parametrize(
    "build",
    [
        lambda: _fem(dirichlet_bc(dimension=1, value=0.7)),
        lambda: _fem(_dirichlet_only_by_default()),
        lambda: _meshless(dirichlet_bc(dimension=1, value=0.0)),
    ],
    ids=["FP-FEM, a wall at 0.7", "FP-FEM, the wall is the default", "meshless Galerkin, an absorbing wall"],
)
def test_the_adjoint_step_refuses_a_dirichlet_wall(build):
    with pytest.raises(NotImplementedError, match=REFUSAL):
        _adjoint_step(build())


@pytest.mark.parametrize(
    ("build", "bc"),
    [
        (_fem, no_flux_bc(dimension=1)),
        (_fem, robin_bc(dimension=1, alpha=1.0, beta=1.0, value=0.0)),
        (_meshless, no_flux_bc(dimension=1)),
    ],
    ids=["FP-FEM no-flux", "FP-FEM Robin", "meshless no-flux"],
)
def test_a_bc_with_no_dirichlet_wall_is_solved_as_before(build, bc):
    """Robin is carried by FP-FEM's own boundary mass (#1237), so it is not a wall this step drops. (A Dirichlet segment
    every face of which a higher priority overrides is refused by the face owner since #2490 B1c-2, and
    `test_meshless_nitsche_whole_faces_2490` pins that refusal.)"""
    solver = build(bc)
    m = _adjoint_step(solver)
    assert m.shape == (solver.n_dof,)
    assert np.isfinite(m).all()


def test_the_issue_pairing_refuses_rather_than_hold_the_walls_at_one():
    """The reach #2608 measured at 52219be2: HJB-FDM with FP-FEM under `adjoint_mode="jacobian_transpose"`, the FP
    with its own wall at 0.7, ran and converged with the FP walls at [1. 1.] at the last level."""
    from mfgarchon.alg.numerical.coupling.block_iterators import BlockIterator
    from mfgarchon.alg.numerical.fem import FPFEMSolver
    from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
    from mfgarchon.geometry import TensorProductGrid
    from mfgarchon.geometry.meshes.mesh_1d import Mesh1D

    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[9], boundary_conditions=no_flux_bc(dimension=1))
    mesh = Mesh1D(bounds=(0.0, 1.0), num_elements=8)
    mesh.generate_mesh()
    mesh.boundary_conditions = no_flux_bc(dimension=1)
    fp = FPFEMSolver(_problem(mesh), order=1, boundary_conditions=dirichlet_bc(dimension=1, value=0.7))
    hjb_problem = _problem(grid)
    iterator = BlockIterator(hjb_problem, HJBFDMSolver(hjb_problem), fp, adjoint_mode="jacobian_transpose")
    with pytest.raises(NotImplementedError, match=REFUSAL):
        iterator.solve(max_iterations=2, tolerance=1e-12)
