"""P1 FEM lumps the mass in its time derivative (#2845), so FP-FEM's implicit pure-diffusion step is an M-matrix step
wherever every off-diagonal of the stiffness is <= 0, and takes no density negative: the clip after it has nothing to add.

The oracle is the scheme's own invariant: a no-flux, pure-diffusion solve conserves mass exactly, so `solve_fp_system`'s
mass is pinned to round-off. With the consistent mass the step goes negative at this dt, and the clip injects 2e-2 to
6e-2 of the mass per step on a one-node spike (#2845's table). Each mesh first asserts the hypothesis, K's off-diagonals
<= 0: on P1 in 2-D it needs each interior edge's opposite angles to sum to at most pi and each boundary edge's opposite
angle to be at most pi/2, which a random Delaunay mesh does not always give, so a mesh failing it is no evidence either
way. The second mesh keeps its random points 0.03 off the boundary, which is how it passes; its seed is fixed.
"""

import pytest
import skfem

import numpy as np
from scipy.spatial import Delaunay

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.fem.fp_fem_solver import FPFEMSolver
from mfgarchon.alg.numerical.fem.mesh_adapter import skfem_to_meshdata
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions
from mfgarchon.geometry.meshes.mesh_2d import Mesh2D

DT, NT = 1e-4, 3


def _structured():
    return skfem.MeshTri.init_tensor(np.linspace(0.0, 1.0, 33), np.linspace(0.0, 1.0, 33))


def _delaunay():
    rng = np.random.default_rng(0)
    s = np.linspace(0.0, 1.0, 25)[:-1]
    wall = np.vstack([np.c_[s, 0 * s], np.c_[1 + 0 * s, s], np.c_[1 - s, 1 + 0 * s], np.c_[0 * s, 1 - s]])
    points = np.vstack([wall, rng.uniform(0.03, 0.97, size=(500, 2))])
    return skfem.MeshTri(points.T, Delaunay(points).simplices.T)


@pytest.mark.parametrize("build", [_structured, _delaunay], ids=["init_tensor", "delaunay"])
def test_a_no_flux_pure_diffusion_spike_keeps_its_mass_to_round_off(build):
    mesh = Mesh2D(domain_type="rectangle", bounds=(0.0, 1.0, 0.0, 1.0))
    mesh.mesh_data = skfem_to_meshdata(build())
    mesh.boundary_conditions = BoundaryConditions(
        dimension=2,
        segments=[BCSegment(name=f, bc_type=BCType.NO_FLUX, boundary=f) for f in ("x_min", "x_max", "y_min", "y_max")],
    )
    problem = MFGProblem(
        model=Model(hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0)), volatility=0.6),
        domain=mesh,
        conditions=Conditions(m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0, T=NT * DT),
        Nt=NT,
    )
    solver = FPFEMSolver(problem)
    stiffness = solver._K.tocoo()
    off_diagonal = stiffness.data[stiffness.row != stiffness.col]
    assert off_diagonal.max() <= 1e-12, off_diagonal.max()

    x = solver._disc.dof_coordinates
    m0 = np.zeros(len(x))
    m0[np.argmin(np.sum((x - 0.5) ** 2, axis=1))] = 1.0
    m = np.asarray(solver.solve_fp_system(M_initial=m0, potential_field=np.zeros((NT + 1, len(x)))))
    mass = np.asarray(solver._M.sum(axis=0)).ravel() @ m.T
    assert np.max(np.abs(mass / mass[0] - 1.0)) < 1e-12, mass
