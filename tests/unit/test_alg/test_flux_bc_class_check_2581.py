"""Every solver class meets a FLUX boundary one way or the other: honoured, or refused by name (#2581 PR 1's class check).

The population is derived, not listed: every concrete subclass of `BaseMFGSolver` the package defines, found by walking
the subclasses after importing every module of ``mfgarchon.alg``. A class added later joins it by itself.

- A class that declares ``_SUPPORTED_BC_TYPES`` declares FLUX exactly when it honours it -- FP-FDM's implicit step and
  FP-FEM -- and the base gate (`BaseMFGSolver._validate_bc_support`) refuses a FLUX for every other one. The planted
  rival, a subclass that declares FLUX with nothing behind it, must be caught.
- A class that declares none passes the gate, so each is recorded with what a FLUX meets there; a class that joins the
  population undeclared turns the record test red until its outcome is written down (the audit's ruling on #2581's plan,
  finding 2).
- The classes that take a BC of their own are handed a FLUX there and refuse it by name at construction,
  the weak-form base included. One that reads the shared BC meets its owner's refusal
  (`MFGProblem.get_boundary_conditions`, B-2).
"""

from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil

import pytest
import skfem

import numpy as np

import mfgarchon.alg
from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.base_solver import BaseMFGSolver
from mfgarchon.alg.numerical.fem.assembly import create_basis
from mfgarchon.alg.numerical.fem.discretization import FEMDiscretization
from mfgarchon.alg.numerical.fem.fp_fem_solver import FPFEMSolver
from mfgarchon.alg.numerical.fem.mesh_adapter import skfem_to_meshdata
from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
from mfgarchon.alg.numerical.fp_solvers.fp_fvm import FPFVMSolver
from mfgarchon.alg.numerical.fp_solvers.fp_particle import FPParticleSolver
from mfgarchon.alg.numerical.fp_solvers.fp_semi_lagrangian_adjoint import FPSLSolver
from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver
from mfgarchon.alg.numerical.hjb_solvers.hjb_gfdm import HJBGFDMSolver
from mfgarchon.alg.numerical.meshless_galerkin.fp_solver import MeshlessGalerkinFPSolver
from mfgarchon.alg.numerical.weak_form_fp_solver import WeakFormFPSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid, flux_bc, no_flux_bc
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions
from mfgarchon.geometry.meshes.mesh_2d import Mesh2D

#: The classes that honour FLUX.
HONOUR = {FPFDMSolver, FPFEMSolver}

#: Every concrete class that declares no ``_SUPPORTED_BC_TYPES``, with what a FLUX meets there.
UNDECLARED = {
    "PenaltyHJBSolver": "wraps an inner HJB solver, whose gate and the shared BC's owner refuse a FLUX",
    "WeakFormHJBSolver": "takes no BC of its own; the shared BC's owner refuses a FLUX",
    "SinkhornMFGSolver": "takes no BC of its own; the shared BC's owner refuses a FLUX",
    "WassersteinMFGSolver": "takes no BC of its own; the shared BC's owner refuses a FLUX",
    "PrimalDualMFGSolver": "takes no BC of its own; the shared BC's owner refuses a FLUX",
    "VariationalMFGSolver": "takes no BC of its own; the shared BC's owner refuses a FLUX",
    "FPNetworkSolver": "reads a graph's node BCs, an alphabet (GraphBCType) with no FLUX",
    "NetworkHJBSolver": "reads a graph's node BCs, an alphabet (GraphBCType) with no FLUX",
    "NetworkPolicyIterationHJBSolver": "reads a graph's node BCs, an alphabet (GraphBCType) with no FLUX",
    "WeakFormFPSolver": "takes a BC of its own and builds no prescribed-flux load, so refuses a FLUX at construction",
}


def _concrete_solver_classes() -> list[type]:
    for module in pkgutil.walk_packages(mfgarchon.alg.__path__, "mfgarchon.alg."):
        importlib.import_module(module.name)

    def walk(cls):
        for sub in cls.__subclasses__():
            yield sub
            yield from walk(sub)

    found = {
        cls for cls in walk(BaseMFGSolver) if cls.__module__.startswith("mfgarchon.") and not inspect.isabstract(cls)
    }
    return sorted(found, key=lambda c: c.__qualname__)


def _violations(classes) -> list[str]:
    """Where a declaring class's set or gate disagrees with whether it honours FLUX."""
    bc = flux_bc(0.2, dimension=1)
    out = []
    for cls in classes:
        declared = getattr(cls, "_SUPPORTED_BC_TYPES", None)
        if declared is None:
            continue
        honours = cls in HONOUR
        if (BCType.FLUX in declared) != honours:
            out.append(f"{cls.__qualname__} declares FLUX: {BCType.FLUX in declared}, honours it: {honours}")
        try:
            BaseMFGSolver._validate_bc_support(object.__new__(cls), bc)
            accepted = True
        except NotImplementedError as exc:
            accepted = False
            if "FLUX" not in str(exc):
                out.append(f"{cls.__qualname__} refused without naming FLUX: {exc}")
        if accepted != honours:
            out.append(f"{cls.__qualname__}'s gate accepted FLUX: {accepted}, honours it: {honours}")
    return out


@pytest.fixture(autouse=True)
def _quiet():
    logging.disable(logging.WARNING)
    yield
    logging.disable(logging.NOTSET)


def test_flux_is_declared_by_the_classes_that_honour_it_and_refused_by_every_other_gate():
    classes = _concrete_solver_classes()
    assert set(classes) >= HONOUR
    assert _violations(classes) == []


def test_a_class_that_declares_flux_with_nothing_behind_it_is_caught():
    class Planted(FPFVMSolver):
        _SUPPORTED_BC_TYPES = FPFVMSolver._SUPPORTED_BC_TYPES | {BCType.FLUX}

    assert any("Planted" in v for v in _violations([Planted]))


def test_every_class_that_declares_no_set_is_recorded_with_what_a_flux_meets_there():
    undeclared = {c.__qualname__ for c in _concrete_solver_classes() if getattr(c, "_SUPPORTED_BC_TYPES", None) is None}
    assert undeclared == set(UNDECLARED)


def _grid_problem(bc=None) -> MFGProblem:
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=bc or no_flux_bc(dimension=1))
    model = Model(hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0)), volatility=0.3)
    conditions = Conditions(m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0, T=0.1)
    return MFGProblem(model=model, domain=grid, conditions=conditions, Nt=4)


@pytest.mark.parametrize(
    "build",
    [
        lambda p, bc: FPFVMSolver(p, boundary_conditions=bc),
        lambda p, bc: FPParticleSolver(p, boundary_conditions=bc),
        lambda p, bc: FPSLSolver(p, boundary_conditions=bc),
        lambda p, bc: HJBGFDMSolver(
            p,
            collocation_points=np.linspace(0, 1, 11).reshape(-1, 1),
            boundary_conditions=bc,
            monotonicity_scheme="none",
        ),
        lambda p, bc: MeshlessGalerkinFPSolver(
            p, collocation_points=np.linspace(0, 1, 11).reshape(-1, 1), delta=0.35, boundary_conditions=bc
        ),
    ],
    ids=["FPFVMSolver", "FPParticleSolver", "FPSLSolver", "HJBGFDMSolver", "MeshlessGalerkinFPSolver"],
)
def test_a_class_that_takes_its_own_bc_and_does_not_honour_flux_refuses_it_at_construction(build):
    with pytest.raises(NotImplementedError, match=r"FLUX"):
        build(_grid_problem(), flux_bc(0.2, dimension=1))


def test_the_weak_form_base_refuses_a_flux_at_construction():
    mesh = Mesh2D(domain_type="rectangle", bounds=(0.0, 1.0, 0.0, 1.0))
    mesh.mesh_data = skfem_to_meshdata(skfem.MeshTri.init_tensor(np.linspace(0, 1, 5), np.linspace(0, 1, 5)))
    faces = ("x_min", "x_max", "y_min", "y_max")
    mesh.boundary_conditions = BoundaryConditions(
        dimension=2, segments=[BCSegment(name=f, bc_type=BCType.NO_FLUX, boundary=f) for f in faces]
    )
    model = Model(hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0)), volatility=0.3)
    problem = MFGProblem(
        model=model, domain=mesh, conditions=Conditions(m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0, T=0.1), Nt=4
    )
    own = BoundaryConditions(
        dimension=2,
        segments=[
            BCSegment(name=f, bc_type=BCType.FLUX if f == "x_max" else BCType.NO_FLUX, value=0.2, boundary=f)
            for f in faces
        ],
    )
    basis = create_basis(skfem.MeshTri.init_tensor(np.linspace(0, 1, 5), np.linspace(0, 1, 5)))
    with pytest.raises(NotImplementedError, match=r"WeakFormFPSolver builds no prescribed-flux load"):
        WeakFormFPSolver(problem, FEMDiscretization(basis), boundary_conditions=own)


def test_a_class_reading_the_shared_bc_meets_its_owners_refusal():
    with pytest.raises(ValueError, match=r"is the Fokker-Planck solver's alone"):
        HJBFDMSolver(_grid_problem(flux_bc(0.2, dimension=1)))
