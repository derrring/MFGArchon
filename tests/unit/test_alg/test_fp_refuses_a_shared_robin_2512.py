"""A shared ROBIN is the HJB's, and an FP solver refuses it (#2512, user ruling 2026-10-10).

On a BC the two equations share, ``ROBIN(alpha, beta, g)`` is the HJB's ``alpha*u + beta*du/dn = g``. An FP
Robin condition is on the total flux, ``J.n = (D/beta)(alpha*m - g)``, which is not
``alpha*m + beta*dm/dn = g`` where the drift crosses the wall, so the HJB's coefficients do not say what the
FP's are. `fp_view_of_shared_bc` refuses a shared ROBIN at every FP solver, as it refuses a shared
NEUMANN(g != 0), and advises the same split: the shared BC keeps the ROBIN for the HJB, and the FP solver is
given its own BC. Before this, FP-FEM assembled the shared coefficients as its own Robin; the grid FP solvers
refused ROBIN by type, and still would on their own BC.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, no_flux_bc, robin_bc

SIGMA, T, NT = 0.4, 0.5, 10
SHARED_ROBIN = r"the problem's shared boundary condition has a ROBIN condition"


def _model() -> Model:
    return Model(
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)), volatility=SIGMA
    )


def _problem(domain) -> MFGProblem:
    """``domain`` already carries the shared BC."""
    from mfgarchon.geometry.meshes.mesh_1d import Mesh1D

    # Unit mass on the domain's own measure: Mesh1D's uniform-cell measure gives a constant c the mass c (n+1)/n.
    c = 16 / 17 if isinstance(domain, Mesh1D) else 1.0
    return MFGProblem(
        model=_model(),
        domain=domain,
        conditions=Conditions(
            m_initial=lambda x: c + 0.0 * np.asarray(x, dtype=float)[..., 0], u_terminal=lambda x: 0.0, T=T
        ),
        Nt=NT,
    )


def _grid(bc: BoundaryConditions):
    return TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=bc)


def _mesh(bc: BoundaryConditions):
    from mfgarchon.geometry.meshes.mesh_1d import Mesh1D

    geometry = Mesh1D(bounds=(0.0, 1.0), num_elements=16)
    geometry.generate_mesh()
    geometry.boundary_conditions = bc
    return geometry


def _robin() -> BoundaryConditions:
    return robin_bc(value=0.3, alpha=1.0, beta=2.0, dimension=1)


def _builders():
    """Every in-library FP family that reads a segment BC, on the domain it runs on, and its class name."""
    from mfgarchon.alg.numerical.fem.fp_fem_solver import FPFEMSolver
    from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
    from mfgarchon.alg.numerical.fp_solvers.fp_fvm import FPFVMSolver
    from mfgarchon.alg.numerical.fp_solvers.fp_particle import FPParticleSolver
    from mfgarchon.alg.numerical.fp_solvers.fp_semi_lagrangian_adjoint import FPSLSolver
    from mfgarchon.alg.numerical.meshless_galerkin.fp_solver import MeshlessGalerkinFPSolver

    cloud = np.linspace(0.0, 1.0, 21).reshape(-1, 1)
    return {
        "FDM": (_grid, FPFDMSolver),
        "FVM": (_grid, FPFVMSolver),
        "SL": (_grid, FPSLSolver),
        "Particle": (_grid, lambda p: FPParticleSolver(p, num_particles=200, seed=0)),
        "Meshless": (_grid, lambda p: MeshlessGalerkinFPSolver(p, cloud, delta=0.35)),
        "FEM": (_mesh, lambda p: FPFEMSolver(p, order=1)),
    }


_CLASS = {
    "FDM": "FPFDMSolver",
    "FVM": "FPFVMSolver",
    "SL": "FPSLSolver",
    "Particle": "FPParticleSolver",
    "Meshless": "MeshlessGalerkinFPSolver",
    "FEM": "FPFEMSolver",
}


@pytest.mark.parametrize("family", list(_CLASS))
def test_every_family_refuses_a_shared_robin_naming_both_readings(family):
    domain, build = _builders()[family]
    with pytest.raises(NotImplementedError, match=SHARED_ROBIN) as excinfo:
        build(_problem(domain(_robin())))
    message = str(excinfo.value)
    assert "alpha*u + beta*du/dn = g" in message
    assert "J.n = (D/beta)(alpha*m - g), which is not alpha*m + beta*dm/dn = g" in message
    assert f"pass {_CLASS[family]} boundary_conditions=no_flux_bc(dimension=...) for reflected agents" in message
    assert "build both solvers yourself and pass them as hjb_solver= and fp_solver=" in message


def test_a_robin_fall_through_is_refused():
    """Segments covering both faces do not hide a ROBIN default_bc: it is the shared BC's value too (#1686)."""
    from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver

    covered = BoundaryConditions(
        dimension=1,
        segments=[
            BCSegment(name="left", bc_type=BCType.NO_FLUX, boundary="x_min"),
            BCSegment(name="right", bc_type=BCType.NO_FLUX, boundary="x_max"),
        ],
        default_bc=BCType.ROBIN,
    )
    with pytest.raises(NotImplementedError, match=SHARED_ROBIN) as excinfo:
        FPFDMSolver(_problem(_grid(covered)))
    assert "default_bc=ROBIN, the fall-through" in str(excinfo.value)


def test_the_robin_refusals_advice_runs():
    """Followed on the FEM pair, the one whose HJB takes a ROBIN: the shared BC keeps it, the FP gets no-flux.

    FP-FEM is also the solver whose behaviour this changes: it used to assemble the shared coefficients as
    its own Robin. Under a drift into x_max the no-flux wall holds and the mass stays, and HJB-FEM still
    reads the ROBIN with its coefficients. The advice's last clause is followed literally too: problem.solve
    with both solvers built by hand converges and keeps the mass (measured: 20 Picard iterations, 1.6e-15).
    """
    from mfgarchon.alg.numerical.fem.fp_fem_solver import FPFEMSolver
    from mfgarchon.alg.numerical.fem.hjb_fem_solver import HJBFEMSolver

    problem = _problem(_mesh(_robin()))
    solver = FPFEMSolver(problem, order=1, boundary_conditions=no_flux_bc(dimension=1))
    x = solver._disc.dof_coordinates[:, 0]
    M = np.asarray(
        solver.solve_fp_system(M_initial=np.ones(x.size), potential_field=np.broadcast_to(-x, (NT + 1, x.size)))
    )
    right, middle, left = (int(np.argmin(np.abs(x - at))) for at in (1.0, 0.5, 0.0))
    # Measured: 14.6 at x_max over the midpoint, as with a shared NEUMANN(0.7); a wall-less solve gives 1.0.
    assert M[-1, right] > 3.0 * M[-1, middle] > 3.0 * M[-1, left]
    order = np.argsort(x)
    assert abs(float(np.trapezoid(M[-1, order], x[order])) - 1.0) < 0.05

    hjb_bc = HJBFEMSolver(problem).get_boundary_conditions()
    assert [(seg.bc_type, seg.alpha, seg.beta, seg.value) for seg in hjb_bc.segments] == [(BCType.ROBIN, 1.0, 2.0, 0.3)]

    result = problem.solve(
        hjb_solver=HJBFEMSolver(problem),
        fp_solver=FPFEMSolver(problem, order=1, boundary_conditions=no_flux_bc(dimension=1)),
    )
    assert result.converged
    M_coupled = np.asarray(result.M)
    masses = [float(np.trapezoid(M_coupled[k, order], x[order])) for k in (0, -1)]
    assert abs(masses[1] - masses[0]) < 1e-10


def test_fp_fem_takes_a_robin_of_its_own():
    """Handed to FP-FEM as its own, a ROBIN is held as given and assembled, as J.n = (D/beta)(alpha*m - g)."""
    from mfgarchon.alg.numerical.fem.fp_fem_solver import FPFEMSolver

    solver = FPFEMSolver(_problem(_mesh(no_flux_bc(dimension=1))), order=1, boundary_conditions=_robin())
    assert [(seg.bc_type, seg.alpha, seg.beta, seg.value) for seg in solver._bc.segments] == [
        (BCType.ROBIN, 1.0, 2.0, 0.3)
    ]
    A_robin, rhs_robin = solver._robin_operator_terms(0.08)
    assert A_robin is not None
    assert rhs_robin is not None
