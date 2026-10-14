"""#2642: an HJB solve input the coupling supplies is required where the configured path reads it.

`WeakFormHJBSolver` (HJB-FEM, meshless Galerkin) defaulted a missing `M_density` to ``1/N``, `U_terminal` to zeros and
`U_coupling_prev` to zeros, and `HJBGFDMSolver` defaulted `M_density` to ones. Each default was read. On the weak-form
Picard branch H is evaluated at the gradient of `U_coupling_prev`, so a standalone call without it solved the backward
heat equation: 0.31 off the exact solution of the issue's Dirichlet problem at every resolution. HJB-FDM, SL and WENO
refuse the same call. `BaseHJBSolver.solve()` reached the same branch one layer up, passing the terminal condition as
the iterate.

1. Each such input is refused by name, with the FDM text "<name> is required".
2. The weak-form Newton branch does not read `U_coupling_prev`. With None it solves, bit-identical to Newton given an
   arbitrary iterate.
3. Control: an explicit zero iterate on the Picard branch still returns the heat-equation solution. So the refusal
   changes the interface, not the numerics.
4. GFDM's `U_coupling_prev` stays accepted: no GFDM path reads it (#1055).
"""

from __future__ import annotations

import logging
import warnings

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.fem.hjb_fem_solver import HJBFEMSolver
from mfgarchon.alg.numerical.hjb_solvers.hjb_gfdm import HJBGFDMSolver
from mfgarchon.alg.numerical.meshless_galerkin.hjb_solver import MeshlessGalerkinHJBSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import dirichlet_bc, no_flux_bc
from mfgarchon.geometry.meshes.mesh_1d import Mesh1D

T, SIGMA, G = 0.5, 0.4, 0.7


def _fem(n: int, nt: int, g: float = G) -> tuple[HJBFEMSolver, np.ndarray]:
    """HJB-FEM P1 on [0, 1], u = g at both ends, u(T) = 0, H = |p|^2 / 2, sigma = 0.4, T = 0.5: the issue's problem.
    m_initial has mass 1 on the n + 1 nodes' uniform cells; the HJB reads the density it is passed, not this."""
    mesh = Mesh1D(bounds=(0.0, 1.0), num_elements=n)
    mesh.generate_mesh()
    mesh.boundary_conditions = dirichlet_bc(value=g, dimension=1)
    model = Model(
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)), volatility=SIGMA
    )
    conditions = Conditions(
        m_initial=lambda x: n / (n + 1) + 0.0 * np.asarray(x, dtype=float)[..., 0], u_terminal=lambda x: 0.0, T=T
    )
    solver = HJBFEMSolver(MFGProblem(model=model, domain=mesh, conditions=conditions, Nt=nt), order=1)
    return solver, solver._disc.dof_coordinates[:, 0]


def _heat(x: np.ndarray, g: float = G) -> np.ndarray:
    """The backward heat equation's exact solution with u = g at both ends and u(T) = 0: H evaluated at p = 0."""
    k = np.arange(1, 8000, 2)
    return g - (g * 4 / (k * np.pi) * np.exp(-(SIGMA**2 / 2) * (k * np.pi) ** 2 * T)) @ np.sin(np.outer(k * np.pi, x))


def test_the_standalone_picard_call_is_refused_by_name():
    """The issue's call: U_coupling_prev omitted on the default (Picard) branch."""
    solver, x = _fem(32, 80)
    with pytest.raises(ValueError, match=r"^U_coupling_prev is required on the Picard branch"):
        solver.solve_hjb_system(M_density=np.ones((81, x.size)), U_terminal=np.zeros(x.size))


def test_an_explicit_zero_iterate_still_solves_the_heat_equation():
    """Control: the refusal changes the interface, not the numerics. 4.587e-04 at n = 32, Nt = 200 on main c8bedc09."""
    solver, x = _fem(32, 200)
    U = solver.solve_hjb_system(
        M_density=np.ones((201, x.size)), U_terminal=np.zeros(x.size), U_coupling_prev=np.zeros((201, x.size))
    )
    assert np.abs(np.asarray(U)[0] - _heat(x)).max() < 1e-3


def test_the_newton_branch_reads_no_iterate():
    solver, x = _fem(16, 40, g=0.1)
    M, U_T = np.ones((41, x.size)), np.zeros(x.size)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logging.disable(logging.WARNING)
        try:
            without = solver.solve_hjb_system(M_density=M, U_terminal=U_T, use_newton=True)
            arbitrary = np.random.default_rng(2642).normal(size=(41, x.size))
            given = solver.solve_hjb_system(M_density=M, U_terminal=U_T, use_newton=True, U_coupling_prev=arbitrary)
        finally:
            logging.disable(logging.NOTSET)
    assert np.array_equal(np.asarray(without), np.asarray(given))
    assert np.isfinite(np.asarray(without)).all()


def _grid_problem(n: int = 15) -> MFGProblem:
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[n], boundary_conditions=no_flux_bc(dimension=1))
    model = Model(hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)), volatility=0.5)
    conditions = Conditions(
        m_initial=lambda x: 1.0 + 0.0 * np.asarray(x, dtype=float)[..., 0], u_terminal=lambda x: 0.0, T=0.5
    )
    return MFGProblem(model=model, domain=grid, conditions=conditions, Nt=10)


def _meshless(use_newton: bool) -> MeshlessGalerkinHJBSolver:
    return MeshlessGalerkinHJBSolver(
        _grid_problem(), collocation_points=np.linspace(0.0, 1.0, 15)[:, None], delta=3.5 / 14, use_newton=use_newton
    )


def _gfdm() -> HJBGFDMSolver:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return HJBGFDMSolver(_grid_problem(), np.linspace(0.0, 1.0, 15)[:, None], monotonicity_scheme="none")


M_15, U_15, ITERATE_15 = np.ones((11, 15)) / 15, np.zeros(15), np.zeros((11, 15))


@pytest.mark.parametrize(
    ("build", "kwargs", "name"),
    [
        (lambda: _meshless(False), {"U_terminal": U_15, "U_coupling_prev": ITERATE_15}, "M_density"),
        (lambda: _meshless(False), {"M_density": M_15, "U_coupling_prev": ITERATE_15}, "U_terminal"),
        (lambda: _meshless(True), {"U_terminal": U_15}, "M_density"),
        (lambda: _meshless(True), {"M_density": M_15}, "U_terminal"),
        (lambda: _meshless(False), {"M_density": M_15, "U_terminal": U_15}, "U_coupling_prev"),
        (_gfdm, {"U_terminal": U_15}, "M_density"),
    ],
)
def test_an_input_the_configured_path_reads_is_refused_by_name(build, kwargs, name):
    """Each member: the weak-form pair's density and terminal condition on both branches, its iterate on Picard,
    and GFDM's density, which every GFDM path reads."""
    with pytest.raises(ValueError, match=rf"^{name} is required"):
        build().solve_hjb_system(**kwargs)


def test_gfdm_accepts_no_iterate_because_no_path_reads_it():
    """#1055: GFDM's U_coupling_prev is an interface slot of the coupling's call, read by no GFDM path."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        without = _gfdm().solve_hjb_system(M_density=M_15, U_terminal=U_15)
        given = _gfdm().solve_hjb_system(M_density=M_15, U_terminal=U_15, U_coupling_prev=np.full((11, 15), 7.0))
    assert np.array_equal(np.asarray(without), np.asarray(given))


def test_the_standalone_solve_is_refused_on_picard_and_runs_on_newton():
    """`BaseHJBSolver.solve` passes the terminal condition, tiled, as the iterate: the linearization point on the
    Picard branch, which nobody chose. The Newton branch does not read it."""
    with pytest.raises(ValueError, match=r"^MeshlessGalerkinHJBSolver\.solve\(\) runs the Picard branch"):
        _meshless(False).solve()
    U = np.asarray(_meshless(True).solve())
    assert U.shape == (11, 15)
    assert np.isfinite(U).all()
