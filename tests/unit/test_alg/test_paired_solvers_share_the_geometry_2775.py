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
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver, FPParticleSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver, HJBGFDMSolver
from mfgarchon.alg.numerical.meshless_galerkin import MeshlessGalerkinFPSolver, MeshlessGalerkinHJBSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.multi_population import MultiPopulationProblem
from mfgarchon.core.regime_switching import RegimeSwitchingConfig
from mfgarchon.geometry import Hyperrectangle, PointCloudGeometry, TensorProductGrid
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


def _problem(domain, conditions=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # the density's mass on these measures is not this pin's subject
        return MFGProblem(
            model=Model(
                hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)), volatility=0.3
            ),
            domain=domain,
            conditions=conditions
            or Conditions(
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


def _grid(upper=1.0, coordinates=None):
    if coordinates is not None:
        return TensorProductGrid(
            bounds=[(0.0, upper)],
            Nx_points=[21],
            spacing_type="custom",
            custom_coordinates=[coordinates],
            boundary_conditions=no_flux_bc(dimension=1),
        )
    return TensorProductGrid(bounds=[(0.0, upper)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1))


def _box():
    problem = _problem(Hyperrectangle(np.array([[0.0, 1.0]])))
    problem.components.boundary_conditions = no_flux_bc(dimension=1)
    return problem


_CLOUD = np.linspace(0.0, 1.0, 21)[:, None]  # the grid's 21 nodes, bit for bit
_MOVED_CLOUD = _CLOUD.copy()
_MOVED_CLOUD[1:-1, 0] += 0.02 * np.sin(9 * _CLOUD[1:-1, 0])
_ULP_CLOUD = _CLOUD.copy()
_ULP_CLOUD[10, 0] = np.nextafter(_CLOUD[10, 0], 1.0)


class _CountOnly:
    """A solver to the protocol that reports ``n_dof`` and declares no ``dof_points``."""

    def __init__(self, problem, n_dof):
        self.problem, self.n_dof = problem, n_dof

    def solve_hjb_system(self, *args, **kwargs):  # pragma: no cover - refused before any solve
        raise AssertionError("never reached")

    solve_fp_system = solve_hjb_system


def test_a_pair_whose_halves_hold_their_values_at_other_points_is_refused():
    """#2775's no-cost control and its class, each refused by name when the loop is built.

    - HJB-FEM built on a mesh, the problem solved on the same mesh with its interior moved, 81 vertices each: every
      loop. At bbf94e5e the loops that run on a mesh solved it, FixedPointIterator 3.711e-02 off the pair built on the
      solved mesh (FictitiousPlayIterator refuses every mesh). The FP side too.
    - A meshless Galerkin pair on two clouds of 21 points, the interior of one moved by up to 1.959e-02: at bbf94e5e and
      at fc6c77ea it solved, M 7.675e-02 off the pair on one cloud. A meshless pair sharing one cloud that is not the
      grid's nodes, since the loop hands both halves the problem's data at its nodes: the nodes reversed, at bbf94e5e U
      4.000e-01 and M 8.433e-01 off the pair on the nodes once put back in their order; the moved cloud, U 1.468e-02 and
      M 4.766e-02 off. A meshless HJB on the moved cloud beside an FDM FP: at bbf94e5e, M 7.572e-02 off the HJB on the
      grid's nodes. HJB-FEM P2 beside FP-FEM P1 or P2 on one mesh, whose 289 DOFs are not its 81 vertices.
    - A grid whose coordinates differ from the solved one's by 1.1e-16, and a cloud one ulp off the nodes at one point:
      both comparisons are exact. An FP alone declaring a cloud off the nodes.
    - On an implicit domain, any pair with a declared side: its data have no fixed points to check them against (#2778).
      At bbf94e5e a meshless pair on one cloud there, and a meshless HJB beside a particle FP, solved, and fail the
      relabelling oracle by amounts that change from run to run. The refusal reads no points, so numpy's global random
      state, which the domain samples from, does not move.
    - An equal Hyperrectangle built separately, and a grid against a Hyperrectangle either way round: an implicit
      domain is one geometry only as one object.
    - Node, regime or population 1's pair built for node 0's geometry, not its own: each pair is read against its own
      problem.
    - A solver with no ``problem``, on either side; a protocol pair that reports ``n_dof`` only, in either order and as
      floats; and a declared ``dof_points`` that raises AttributeError, which is not read as no declaration.
    """
    solved, built = _problem(_mesh(moved=True)), _problem(_mesh(moved=False))
    for name, make in _COUPLERS:
        with pytest.raises(ValueError, match=rf"^{name}: the HJB solver was built for a Mesh2D whose points are not"):
            make(solved, HJBFEMSolver(built), FPFEMSolver(solved))
    with pytest.raises(
        ValueError, match=r"^FixedPointIterator: the FP solver was built for a Mesh2D whose points are not"
    ):
        FixedPointIterator(solved, HJBFEMSolver(solved), FPFEMSolver(built))

    grid, box = _problem(_grid()), _box()
    points = np.linspace(0.0, 1.0, box.geometry.num_spatial_points)[:, None]
    gfdm_on_box = HJBGFDMSolver(box, collocation_points=points, delta=0.3, monotonicity_scheme="none")
    for side, hjb, fp in (
        ("HJB", MeshlessGalerkinHJBSolver(box, points, delta=0.05), MeshlessGalerkinFPSolver(box, points, delta=0.05)),
        ("HJB", MeshlessGalerkinHJBSolver(box, points, delta=0.05), FPParticleSolver(box, seed=7)),
        ("FP", gfdm_on_box, MeshlessGalerkinFPSolver(box, points, delta=0.05)),
    ):
        state = np.random.get_state()
        with pytest.raises(
            ValueError,
            match=rf"^BlockIterator: the {side} solver declares the points its DOFs sit at, .* Hyperrectangle",
        ):
            BlockGaussSeidelIterator(box, hjb, fp)
        now = np.random.get_state()
        assert np.array_equal(state[1], now[1]), "the refusal read the implicit domain's points"
        assert state[2:] == now[2:], "the refusal read the implicit domain's points"
    reversed_nodes = np.ascontiguousarray(_CLOUD[::-1])
    for side, hjb, fp in (
        (
            "HJB",
            MeshlessGalerkinHJBSolver(grid, reversed_nodes, delta=0.18),
            MeshlessGalerkinFPSolver(grid, reversed_nodes, delta=0.18),
        ),
        (
            "HJB",
            MeshlessGalerkinHJBSolver(grid, _MOVED_CLOUD, delta=0.18),
            MeshlessGalerkinFPSolver(grid, _MOVED_CLOUD, delta=0.18),
        ),
        (
            "FP",
            MeshlessGalerkinHJBSolver(grid, _CLOUD, delta=0.18),
            MeshlessGalerkinFPSolver(grid, _MOVED_CLOUD, delta=0.18),
        ),
        (
            "FP",
            MeshlessGalerkinHJBSolver(grid, _CLOUD, delta=0.18),
            MeshlessGalerkinFPSolver(grid, _ULP_CLOUD, delta=0.18),
        ),
        ("HJB", MeshlessGalerkinHJBSolver(grid, _MOVED_CLOUD, delta=0.18), FPFDMSolver(grid)),
        ("FP", HJBFDMSolver(grid), MeshlessGalerkinFPSolver(grid, _MOVED_CLOUD, delta=0.18)),
        ("HJB", HJBFEMSolver(solved, order=2), FPFEMSolver(solved)),
        ("HJB", HJBFEMSolver(solved, order=2), FPFEMSolver(solved, order=2)),
    ):
        with pytest.raises(
            ValueError,
            match=rf"^FixedPointIterator: the {side} solver holds its values at points that are not those of",
        ):
            FixedPointIterator(hjb.problem, hjb, fp)

    with pytest.raises(
        ValueError, match=r"^FixedPointIterator: the HJB solver .* whose points are not .*other coordinates"
    ):
        FixedPointIterator(grid, HJBFDMSolver(_problem(_grid(coordinates=np.arange(21) / 20))), FPFDMSolver(grid))
    gfdm = HJBGFDMSolver(_box(), collocation_points=_CLOUD, delta=0.3, monotonicity_scheme="none")
    for at, hjb, fp in (
        (box, gfdm, FPParticleSolver(box, seed=7)),
        (grid, gfdm, FPFDMSolver(grid)),
        (box, HJBFDMSolver(grid), FPParticleSolver(box, seed=7)),
    ):
        with pytest.raises(
            ValueError,
            match=r"^FixedPointIterator: the HJB solver was built for a \w+ that is not the problem's own \w+ object",
        ):
            FixedPointIterator(at, hjb, fp)

    near, far = _problem(_grid()), _problem(_grid(2.0))
    q = RegimeSwitchingConfig(transition_matrix=np.array([[-0.1, 0.1], [0.2, -0.2]]))
    hjbs, fps = [HJBFDMSolver(far), HJBFDMSolver(far)], [FPFDMSolver(far), FPFDMSolver(far)]
    for name, make in (
        (
            r"GraphMFGSolver\[node 1\]",
            lambda: GraphMFGSolver([far, near], AdjacencyCoupling(np.ones((2, 2)) - np.eye(2)), hjbs, fps),
        ),
        (r"RegimeSwitchingIterator\[regime 1\]", lambda: RegimeSwitchingIterator([far, near], q, hjbs, fps)),
        (
            r"MultiPopulationIterator\[population 1\]",
            lambda: MultiPopulationIterator(
                MultiPopulationProblem(populations=[far, near], population_names=["a", "b"]), hjbs, fps
            ),
        ),
    ):
        with pytest.raises(
            ValueError, match=rf"^{name}: the HJB solver was built for a TensorProductGrid whose points"
        ):
            make()

    class NoProblem:
        def solve_hjb_system(self, *args, **kwargs):  # pragma: no cover - refused before any solve
            raise AssertionError("never reached")

        solve_fp_system = solve_hjb_system

    for side, hjb, fp in (("HJB", NoProblem(), FPFDMSolver(grid)), ("FP", HJBFDMSolver(grid), NoProblem())):
        with pytest.raises(TypeError, match=rf"^FixedPointIterator: the {side} solver, NoProblem, has no `problem`"):
            FixedPointIterator(grid, hjb, fp)
    for counts in ((21, 25), (25, 21), (21.0, 25.0)):
        with pytest.raises(
            ValueError, match=r"^FixedPointIterator: paired HJB and FP solvers report different DOF counts"
        ):
            FixedPointIterator(grid, _CountOnly(grid, counts[0]), _CountOnly(grid, counts[1]))

    class DeclaresAndFails(_CountOnly):
        @property
        def dof_points(self):
            raise AttributeError("a declared dof_points that fails to read")

    with pytest.raises(AttributeError, match="a declared dof_points that fails to read"):
        FixedPointIterator(grid, DeclaresAndFails(grid, None), FPFDMSolver(grid))


def test_a_pair_on_the_solved_points_or_undeclared_on_an_implicit_domain_is_admitted():
    """Each loop constructs where both halves hold their values at the solved problem's points, and on an implicit
    domain where neither half declares points of its own:

    - HJB-FDM built from an equal grid built separately, through every loop; at bbf94e5e that pair solved equal to the
      one built from the problem itself;
    - a meshless pair on one cloud, and a meshless HJB on the grid's nodes beside an FDM FP; and one whose caller moves
      its cloud after building it, which declares the points it was built on (#2779);
    - HJB-FEM P1 beside a particle FP. It does not solve, at bbf94e5e or here, for a reason outside this check: the
      particle FP cannot sample its initial density on a mesh (#2757);
    - HJB-GFDM at its own collocation points beside a particle FP on an implicit domain, in the Block Gauss-Seidel loop
      that solves on one, built from the problem solved or from a rebuild that shares its domain object: compared by
      identity only;
    - a meshless pair on a PointCloudGeometry of its cloud, which the implicit-domain refusal advises.
    """
    grid, twin = _problem(_grid()), _problem(_grid())
    for _, make in _COUPLERS:
        make(grid, HJBFDMSolver(twin), FPFDMSolver(grid))
    FixedPointIterator(
        grid, MeshlessGalerkinHJBSolver(grid, _CLOUD, delta=0.18), MeshlessGalerkinFPSolver(grid, _CLOUD, delta=0.18)
    )
    FixedPointIterator(grid, MeshlessGalerkinHJBSolver(grid, _CLOUD, delta=0.18), FPFDMSolver(grid))
    moved_later = _CLOUD.copy()
    hjb, fp = (
        MeshlessGalerkinHJBSolver(grid, moved_later, delta=0.18),
        MeshlessGalerkinFPSolver(grid, moved_later, delta=0.18),
    )
    moved_later[:] = _MOVED_CLOUD  # the caller moves its array after the solvers are built (#2779)
    for solver in (hjb, fp):
        assert np.array_equal(solver.dof_points, _CLOUD)
        assert not solver.dof_points.flags.writeable
    FixedPointIterator(grid, hjb, fp)
    mesh = _problem(_mesh(moved=False))
    FixedPointIterator(mesh, HJBFEMSolver(mesh), FPParticleSolver(mesh, seed=7))
    box = _box()
    points = np.linspace(0.0, 1.0, box.geometry.num_spatial_points)[:, None]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # the rebuild's density mass is not this pin's subject
        rebuild = box.with_conditions(
            Conditions(
                u_terminal=lambda x: 0.0, m_initial=lambda x: float(np.exp(-10 * (np.asarray(x)[0] - 0.6) ** 2)), T=0.3
            )
        )
    rebuild.components.boundary_conditions = no_flux_bc(dimension=1)
    assert rebuild.geometry is box.geometry
    for built in (box, rebuild):
        BlockGaussSeidelIterator(
            box,
            HJBGFDMSolver(built, collocation_points=points, delta=0.3, monotonicity_scheme="none"),
            FPParticleSolver(box, seed=7),
        )
    cloud = _problem(PointCloudGeometry(points.copy()))
    cloud.components.boundary_conditions = no_flux_bc(dimension=1)
    BlockGaussSeidelIterator(
        cloud, MeshlessGalerkinHJBSolver(cloud, points, delta=0.05), MeshlessGalerkinFPSolver(cloud, points, delta=0.05)
    )
