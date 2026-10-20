"""#2775: a coupling loop refuses an HJB or FP solver built for a geometry whose points are not those of the problem it
solves, through one owner, `assert_paired_solver_geometry`."""

from __future__ import annotations

import warnings

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.coupling import FixedPointIterator, MFGResidual, NewtonMFGSolver
from mfgarchon.alg.numerical.coupling.block_iterators import (
    BlockGaussSeidelIterator,
    BlockIterator,
    BlockJacobiIterator,
)
from mfgarchon.alg.numerical.coupling.fictitious_play import FictitiousPlayIterator
from mfgarchon.alg.numerical.coupling.graph_coupling import AdjacencyCoupling
from mfgarchon.alg.numerical.coupling.graph_mfg_solver import GraphMFGSolver
from mfgarchon.alg.numerical.coupling.multi_population_iterator import MultiPopulationIterator
from mfgarchon.alg.numerical.coupling.regime_switching_iterator import RegimeSwitchingIterator
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.multi_population import MultiPopulationProblem
from mfgarchon.core.regime_switching import RegimeSwitchingConfig
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

skfem = pytest.importorskip("skfem", reason="scikit-fem required for the mesh pair")

from mfgarchon.alg.numerical.fem import HJBFEMSolver  # noqa: E402
from mfgarchon.alg.numerical.fem.fp_fem_solver import FPFEMSolver  # noqa: E402
from mfgarchon.alg.numerical.fem.mesh_adapter import skfem_to_meshdata  # noqa: E402
from mfgarchon.geometry.meshes.mesh_2d import Mesh2D  # noqa: E402

# Each coupler, as (the name its refusal carries, how it is built from the problem solved and one HJB / FP pair).
_COUPLERS = [
    ("FixedPointIterator", lambda p, h, f: FixedPointIterator(p, h, f)),
    ("BlockIterator", lambda p, h, f: BlockIterator(p, h, f)),
    ("BlockIterator", lambda p, h, f: BlockGaussSeidelIterator(p, h, f)),
    ("BlockIterator", lambda p, h, f: BlockJacobiIterator(p, h, f)),
    ("FictitiousPlayIterator", lambda p, h, f: FictitiousPlayIterator(p, h, f)),
    ("NewtonMFGSolver", lambda p, h, f: NewtonMFGSolver(p, h, f)),
    ("MFGResidual", lambda p, h, f: MFGResidual(p, h, f)),
    (
        r"MultiPopulationIterator\[population 0\]",
        lambda p, h, f: MultiPopulationIterator(
            MultiPopulationProblem(populations=[p], population_names=["a"]), [h], [f]
        ),
    ),
    (
        r"GraphMFGSolver\[node 0\]",
        lambda p, h, f: GraphMFGSolver([p, p], AdjacencyCoupling(np.ones((2, 2)) - np.eye(2)), [h, h], [f, f]),
    ),
    (
        r"RegimeSwitchingIterator\[regime 0\]",
        lambda p, h, f: RegimeSwitchingIterator(
            [p, p], RegimeSwitchingConfig(transition_matrix=np.array([[-0.1, 0.1], [0.2, -0.2]])), [h, h], [f, f]
        ),
    ),
]


def _problem(domain):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # the density's mass on these measures is not this pin's subject
        return MFGProblem(
            model=Model(
                hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)), volatility=0.3
            ),
            domain=domain,
            conditions=Conditions(
                u_terminal=lambda x: float(np.sum((np.asarray(x, dtype=float) - 0.7) ** 2)),
                m_initial=lambda x: float(np.exp(-10 * np.sum((np.asarray(x, dtype=float) - 0.3) ** 2))),
                T=0.3,
            ),
            Nt=4,
        )


def _mesh(moved):
    """skfem's ``init_sqsymmetric().refined(2)`` on [0, 1]^2, 81 vertices; ``moved`` moves its interior by up to
    0.03."""
    mesh = skfem.MeshTri.init_sqsymmetric().refined(2)
    if moved:
        p = mesh.p.copy()
        inside = (p[0] > 1e-12) & (p[0] < 1 - 1e-12) & (p[1] > 1e-12) & (p[1] < 1 - 1e-12)
        p[:, inside] += 0.03 * np.vstack([np.sin(7 * p[1, inside]), np.cos(5 * p[0, inside])])
        mesh = skfem.MeshTri(p, mesh.t)
    geometry = Mesh2D(domain_type="rectangle", bounds=(0.0, 1.0, 0.0, 1.0))
    geometry.mesh_data = skfem_to_meshdata(mesh)
    geometry.boundary_conditions = no_flux_bc(dimension=2)
    return geometry


def test_a_pair_built_for_other_points_is_refused_by_every_coupler():
    """#2775's no-cost control. HJB-FEM is built on a mesh, and the problem solved is on the same mesh with its interior
    moved, 81 vertices each.

    At bbf94e5e the loops that run on a mesh solved it with nothing raised: three FixedPointIterator sweeps were
    3.711e-02 off the pair built on the solved mesh, and the others 2.5e-02 to 3.7e-02 off (FictitiousPlayIterator
    refuses every mesh). Each coupler, MFGResidual included, now refuses it by name when it is built, on either side,
    and so refuses a solver that carries no ``problem`` to compare, and P2 beside P1 on one mesh. The admitted side:
    HJB-FDM built from a grid equal to the solved problem's but built separately constructs every coupler, and at
    bbf94e5e that pair solved equal to the one built from the problem itself."""
    solved, built = _problem(_mesh(moved=True)), _problem(_mesh(moved=False))
    for name, make in _COUPLERS:
        with pytest.raises(ValueError, match=rf"^{name}: the HJB solver was built for a Mesh2D whose points are not"):
            make(solved, HJBFEMSolver(built), FPFEMSolver(solved))
    with pytest.raises(
        ValueError, match=r"^FixedPointIterator: the FP solver was built for a Mesh2D whose points are not"
    ):
        FixedPointIterator(solved, HJBFEMSolver(solved), FPFEMSolver(built))

    class NoProblem:
        def solve_hjb_system(self, *args, **kwargs):  # pragma: no cover - refused before any solve
            raise AssertionError("never reached")

    with pytest.raises(TypeError, match=r"^FixedPointIterator: the HJB solver, NoProblem, has no `problem`"):
        FixedPointIterator(solved, NoProblem(), FPFEMSolver(solved))
    # On one mesh, P2 beside P1: one geometry, two DOF layouts, refused by #1489's count, which every coupler now reads.
    with pytest.raises(ValueError, match=r"^FixedPointIterator: paired HJB and FP solvers report different DOF counts"):
        FixedPointIterator(solved, HJBFEMSolver(solved, order=2), FPFEMSolver(solved))

    def grid():
        return TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1))

    solved, twin = _problem(grid()), _problem(grid())
    for _, make in _COUPLERS:
        make(solved, HJBFDMSolver(twin), FPFDMSolver(solved))
