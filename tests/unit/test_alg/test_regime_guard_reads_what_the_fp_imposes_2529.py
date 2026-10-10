"""RegimeSwitchingIterator's FP-data guard reads the BC the FP solver imposes, through the owner (#2529, #2512 row B3).

The guard refuses inhomogeneous FP boundary data on a regime with outflow. For a solver with no
``boundary_conditions`` attribute (the weak-form family, which keeps its BC in ``_bc``), its fallback read the
geometry's BC raw. A shared Dirichlet exit is absorbing at the FP, its value dropped (row B3), so the FEM FP
imposes homogeneous data while the guard read ``[0.7]`` and refused. The fallback now asks the solver's
``get_boundary_conditions()``, which ends in ``BaseFPSolver._fp_view_of_shared``.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.coupling.regime_switching_iterator import RegimeSwitchingIterator
from mfgarchon.alg.numerical.fem.fp_fem_solver import FPFEMSolver
from mfgarchon.alg.numerical.fem.hjb_fem_solver import HJBFEMSolver
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.regime_switching import RegimeSwitchingConfig
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, no_flux_bc
from mfgarchon.geometry.boundary.bc_utils import describe_inhomogeneous_bc_data
from mfgarchon.geometry.meshes.mesh_1d import Mesh1D

_CONFIG = RegimeSwitchingConfig(transition_matrix=np.array([[-0.1, 0.1], [0.2, -0.2]]))
_EXIT = [
    BCSegment(name="L", bc_type=BCType.NO_FLUX, boundary="x_min"),
    BCSegment(name="R", bc_type=BCType.DIRICHLET, value=0.7, boundary="x_max"),
]


def _problem(domain, m0: float = 1.0) -> MFGProblem:
    return MFGProblem(
        model=Model(hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0)), volatility=0.8),
        domain=domain,
        conditions=Conditions(m_initial=lambda x: m0, u_terminal=lambda x: 0.0, T=0.5),
        Nt=5,
    )


def _mesh_with_shared_exit() -> Mesh1D:
    geo = Mesh1D(bounds=(0.0, 1.0), num_elements=10)
    geo.generate_mesh()
    geo.boundary_conditions = BoundaryConditions(dimension=1, segments=list(_EXIT))
    return geo


def test_an_fem_fp_with_a_shared_exit_is_not_refused():
    """At the base this raised "will impose carry data that is not verifiably zero: [0.7]"."""
    problems = [_problem(_mesh_with_shared_exit(), m0=1.0 / 1.1) for _ in range(2)]  # unit mass on 11 cells
    fps = [FPFEMSolver(p) for p in problems]
    assert describe_inhomogeneous_bc_data(fps[0]._bc, bc_types=None) == [], "the FEM FP imposes homogeneous data"
    RegimeSwitchingIterator(
        problems=problems, regime_config=_CONFIG, hjb_solvers=[HJBFEMSolver(p) for p in problems], fp_solvers=fps
    )


def test_data_the_fp_solver_really_imposes_is_still_refused():
    """The control: an FP solver given its own inhomogeneous Dirichlet imposes it, and the guard refuses that."""
    own = BoundaryConditions(dimension=1, segments=list(_EXIT))
    problems = [
        _problem(TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=no_flux_bc(dimension=1)))
        for _ in range(2)
    ]
    fps = [FPFDMSolver(p, boundary_conditions=own) for p in problems]
    with pytest.raises(ValueError, match=r"not verifiably zero: \[0\.7\]"):
        RegimeSwitchingIterator(
            problems=problems, regime_config=_CONFIG, hjb_solvers=[HJBFDMSolver(p) for p in problems], fp_solvers=fps
        )
