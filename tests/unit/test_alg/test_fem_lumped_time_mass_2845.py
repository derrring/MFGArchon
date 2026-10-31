"""P1 FEM lumps the mass in its time derivative (#2845), so FP-FEM's implicit pure-diffusion step with no-flux or
Dirichlet walls is an M-matrix step wherever every off-diagonal of the stiffness is <= 0, and takes no density negative:
the clip after it has nothing to add. A Robin wall's consistent boundary mass is outside that condition.

The oracle is the scheme's own invariant: a no-flux, pure-diffusion solve conserves mass exactly, so `solve_fp_system`'s
mass is pinned to round-off. With the consistent mass the step goes negative at this dt, and the clip injects 1.2e-2 to
6.3e-2 of the mass per step on a one-node spike (#2845's table). Each mesh first asserts the hypothesis, K's
off-diagonals <= 0: in 1-D P1 always gives it; in 2-D it needs each interior edge's opposite angles to sum to at most pi
and each boundary edge's opposite angle to be at most pi/2, which a random Delaunay mesh does not always give, so a mesh
failing it is no evidence either way. The Delaunay mesh keeps its random points 0.03 off the boundary, which is how it
passes; its seed is fixed. The 1-D mesh is here because the pairing below cannot see a 1-D revert of both solvers at
once: on a uniform line P1's M = ML - (h^2/6) K, and <U, ML m> is then conserved under either mass.
"""

import pytest
import skfem

import numpy as np
from scipy import sparse
from scipy.spatial import Delaunay

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.fem.fp_fem_solver import FPFEMSolver
from mfgarchon.alg.numerical.fem.hjb_fem_solver import HJBFEMSolver
from mfgarchon.alg.numerical.fem.mesh_adapter import skfem_to_meshdata
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions
from mfgarchon.geometry.meshes.mesh_1d import Mesh1D
from mfgarchon.geometry.meshes.mesh_2d import Mesh2D
from mfgarchon.geometry.meshes.mesh_3d import Mesh3D

DT, NT = 1e-4, 3
FACES = {
    1: ("x_min", "x_max"),
    2: ("x_min", "x_max", "y_min", "y_max"),
    3: ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"),
}


def _unit_level(mesh) -> float:
    """The constant whose mass is 1 on the mesh's measure: Mesh1D's is (n + 1)/n for n cells, a 2-D or 3-D mesh's 1."""
    return mesh.num_elements / (mesh.num_elements + 1) if isinstance(mesh, Mesh1D) else 1.0


def _structured():
    return skfem.MeshTri.init_tensor(np.linspace(0.0, 1.0, 33), np.linspace(0.0, 1.0, 33))


def _delaunay():
    rng = np.random.default_rng(0)
    s = np.linspace(0.0, 1.0, 25)[:-1]
    wall = np.vstack([np.c_[s, 0 * s], np.c_[1 + 0 * s, s], np.c_[1 - s, 1 + 0 * s], np.c_[0 * s, 1 - s]])
    points = np.vstack([wall, rng.uniform(0.03, 0.97, size=(500, 2))])
    return skfem.MeshTri(points.T, Delaunay(points).simplices.T)


@pytest.mark.parametrize("build", [None, _structured, _delaunay], ids=["line", "init_tensor", "delaunay"])
def test_a_no_flux_pure_diffusion_spike_keeps_its_mass_to_round_off(build):
    if build is None:
        mesh = Mesh1D(bounds=(0.0, 1.0), num_elements=32)
        mesh.generate_mesh()
        dim = 1
    else:
        mesh = Mesh2D(domain_type="rectangle", bounds=(0.0, 1.0, 0.0, 1.0))
        mesh.mesh_data = skfem_to_meshdata(build())
        dim = 2
    mesh.boundary_conditions = BoundaryConditions(
        dimension=dim, segments=[BCSegment(name=f, bc_type=BCType.NO_FLUX, boundary=f) for f in FACES[dim]]
    )
    problem = MFGProblem(
        model=Model(hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0)), volatility=0.6),
        domain=mesh,
        conditions=Conditions(m_initial=lambda x, c=_unit_level(mesh): c, u_terminal=lambda x: 0.0, T=NT * DT),
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


def _pair_problem(dim: int, lambda_: float) -> MFGProblem:
    if dim == 1:
        mesh = Mesh1D(bounds=(0.0, 1.0), num_elements=16)
        mesh.generate_mesh()
    elif dim == 2:
        mesh = Mesh2D(domain_type="rectangle", bounds=(0.0, 1.0, 0.0, 1.0))
        mesh.mesh_data = skfem_to_meshdata(skfem.MeshTri.init_tensor(np.linspace(0, 1, 9), np.linspace(0, 1, 9)))
    else:
        axis = np.linspace(0, 1, 5)
        mesh = Mesh3D(domain_type="box", bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0))
        mesh.mesh_data = skfem_to_meshdata(skfem.MeshTet.init_tensor(axis, axis, axis))
    mesh.boundary_conditions = BoundaryConditions(
        dimension=dim, segments=[BCSegment(name=f, bc_type=BCType.NO_FLUX, boundary=f) for f in FACES[dim]]
    )
    return MFGProblem(
        model=Model(
            hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=lambda_)), volatility=0.5
        ),
        domain=mesh,
        conditions=Conditions(m_initial=lambda x, c=_unit_level(mesh): c, u_terminal=lambda x: 0.0, T=5 * 0.01),
        Nt=5,
    )


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_hjb_fem_and_fp_fem_step_on_one_time_mass(dim: int):
    """The pair's time terms share one mass (#2845). With no drift, and H read at U_coupling_prev = 0 where the
    quadratic H is 0, both steps solve (ML/dt + D K) x = (ML/dt) x_prev with a symmetric matrix, so <U^n, ML m^n> is
    the same at every n, to round-off. Setting either solver's time mass back to M moves it by 3.9e-4 (1-D), 1.6e-3
    (2-D) and 6.7e-3 to 6.8e-3 (3-D). A revert of HJB-FEM's time mass, or of FP-FEM's adjoint step, that keeps each
    step self-consistent reddened no other test at cf747a70, over the test files review 1 of #2848 listed as reaching
    FEM; a revert of FP-FEM's forward step also reddens the mass pins above. Three arms: HJB-FEM's Picard step; its
    Newton step, H made negligible (4.3e-9 to 4.8e-9 at the head); and FP-FEM's adjoint step, with no advection,
    which must equal its forward step."""
    for arm, lambda_, tolerance in (("picard", 1.0, 1e-12), ("newton", 1e8, 1e-7)):
        problem = _pair_problem(dim, lambda_)
        fp, hjb = FPFEMSolver(problem), HJBFEMSolver(problem)
        x = fp._disc.dof_coordinates
        n_dof = len(x)
        m0 = 1.0 + 0.5 * np.cos(np.pi * x[:, 0])
        m = np.asarray(fp.solve_fp_system(M_initial=m0, potential_field=np.zeros((6, n_dof))))
        u = np.asarray(
            hjb.solve_hjb_system(
                np.broadcast_to(m0, (6, n_dof)).copy(),
                np.cos(np.pi * x[:, 0]),
                np.zeros((6, n_dof)),
                use_newton=arm == "newton",
            )
        )
        lumped = np.asarray(fp._M.sum(axis=1)).ravel()
        pairing = np.array([u[n] @ (lumped * m[n]) for n in range(6)])
        assert np.max(np.abs(pairing / pairing[0] - 1.0)) < tolerance, (arm, pairing)
        if arm == "picard":
            adjoint = np.asarray(fp.solve_fp_step_adjoint_mode(m0, sparse.csr_matrix((n_dof, n_dof))))
            assert np.max(np.abs(adjoint - m[1])) < 1e-14
