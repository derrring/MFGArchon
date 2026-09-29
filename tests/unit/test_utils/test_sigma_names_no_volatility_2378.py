"""#2378 phase 4 part 2b: the MFG volatility has one name, `volatility`.

Ruling 16 (maintainer, 2026-09-28) hard-renamed every public `sigma=` that carried the agents' SDE
volatility; ruling 17 put the internal sites and their docs in the same part. What remains named
`sigma` is a different quantity, listed below with what it is. The retired spellings are refused by
name, so an old call fails where it is written rather than solving with a default.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil

import pytest

import numpy as np

import mfgarchon

#: Every function in the package still taking a `sigma` parameter, and what that `sigma` is. None of
#: them is the agents' SDE volatility.
_NOT_THE_VOLATILITY = {
    "mfgarchon.core.stochastic.noise_processes:OrnsteinUhlenbeckProcess.__init__": "the common-noise process's own volatility",
    "mfgarchon.core.stochastic.noise_processes:GeometricBrownianMotion.__init__": "the common-noise process's own volatility",
    "mfgarchon.core.stochastic.noise_processes:CoxIngersollRossProcess.__init__": "the common-noise process's own volatility",
    "mfgarchon.core.stochastic.noise_processes:JumpDiffusionProcess.__init__": "the common-noise process's own volatility",
    "mfgarchon.operators.integro_diff.levy_measures:GaussianJumps.__init__": "the Levy jump-size standard deviation",
    "mfgarchon.geometry.graph.maze_postprocessing:smooth_walls_gaussian": "a Gaussian blur width",
    "mfgarchon.geometry.graph.maze_utils:smooth_walls_gaussian": "a Gaussian blur width",
}


def _functions_taking_sigma() -> set[str]:
    found = set()
    for info in pkgutil.walk_packages(mfgarchon.__path__, "mfgarchon."):
        module = importlib.import_module(info.name)
        for obj in vars(module).values():
            if inspect.isclass(obj) and obj.__module__ == module.__name__:
                members = [
                    (f"{obj.__qualname__}.{k}", v.__func__ if isinstance(v, classmethod | staticmethod) else v)
                    for k, v in vars(obj).items()
                ]
            elif inspect.isfunction(obj) and obj.__module__ == module.__name__:
                members = [(obj.__qualname__, obj)]
            else:
                continue
            for qualname, func in members:
                if inspect.isfunction(func) and "sigma" in inspect.signature(inspect.unwrap(func)).parameters:
                    found.add(f"{module.__name__}:{qualname}")
    return found


def test_no_function_takes_sigma_for_the_volatility():
    """The population is every plain function defined at module level in the package and every
    function, classmethod and staticmethod defined in one of its classes, whatever its role; the
    property audited is a parameter spelt `sigma`. A new one must either be one of the listed
    quantities or be named `volatility`.

    Outside it: nested functions, `@jit`-compiled ones (a jax `PjitFunction` is not a function), and
    pydantic fields. At da768800 an AST scan, which sees the first two, found the same seven; the one
    pydantic field that carried the volatility has its own test below."""
    found = _functions_taking_sigma()
    assert "mfgarchon.core.stochastic.noise_processes:OrnsteinUhlenbeckProcess.__init__" in found, (
        "reach control: the walk no longer finds a known `sigma` parameter"
    )
    assert found == set(_NOT_THE_VOLATILITY), (
        f"new `sigma` parameters (name the volatility `volatility`): {sorted(found - set(_NOT_THE_VOLATILITY))}\n"
        f"listed but gone (drop from the list): {sorted(set(_NOT_THE_VOLATILITY) - found)}"
    )


def _renamed_public_api():
    from mfgarchon.alg.numerical.adjoint import operators as adjoint_ops
    from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
    from mfgarchon.alg.numerical.weak_form_fp_solver import WeakFormFPSolver
    from mfgarchon.alg.optimization import variational_problem as vp
    from mfgarchon.operators.differential.diffusion import DiffusionOperator
    from mfgarchon.utils import manufactured as mms
    from mfgarchon.utils import pde_coefficients as pc

    unbound = [
        adjoint_ops.build_diffusion_matrix,
        adjoint_ops.build_diffusion_matrix_1d,
        adjoint_ops.build_diffusion_matrix_2d,
        adjoint_ops.build_diffusion_matrix_from_geometry,
        vp.create_quadratic_variational_mfg,
        vp.create_obstacle_variational_mfg,
        pc.diffusion_from_volatility,
        pc.diffusion_from_volatility_torch,
        pc.check_adi_compatibility,
        mms.hjb_source,
        mms.fp_source,
        DiffusionOperator.from_volatility,
    ]
    methods = [
        vp.VariationalMFGProblem.__init__,
        FPFDMSolver.solve_fp_step_adjoint_mode,
        WeakFormFPSolver.solve_fp_step_adjoint_mode,
    ]
    return [(f, ()) for f in unbound] + [(f, (None,)) for f in methods]


def test_the_renamed_public_api_refuses_sigma_by_name():
    """Ruling 16: each public function part 2b renamed raises a TypeError naming `volatility=`. The
    refusal runs before the arguments bind, so a placeholder stands in for `self`. The manufactured
    sources also refuse their old `sigma_kind=`."""
    wrong = []
    for func, args in _renamed_public_api():
        try:
            func(*args, sigma=0.3)
        except TypeError as exc:
            if "no longer takes sigma=; pass volatility=" in str(exc):
                continue
            wrong.append(f"{func.__qualname__}: {exc}")
        except Exception as exc:  # any other outcome is the finding
            wrong.append(f"{func.__qualname__}: {type(exc).__name__}: {exc}")
        else:
            wrong.append(f"{func.__qualname__}: accepted sigma=")
    assert not wrong, "\n".join(wrong)

    from mfgarchon.utils.manufactured import fp_source, hjb_source

    for func in (hjb_source, fp_source):
        with pytest.raises(TypeError, match=r"no longer takes sigma_kind=; pass volatility_kind="):
            func(sigma_kind="tensor")


@pytest.mark.parametrize(
    ("owner", "old", "new"),
    [
        ("mfgarchon.alg.numerical.hjb_solvers.base_hjb:compute_hjb_residual", "sigma_at_n", "volatility_at_n"),
        ("mfgarchon.alg.numerical.hjb_solvers.base_hjb:compute_hjb_jacobian", "sigma_at_n", "volatility_at_n"),
        ("mfgarchon.alg.numerical.hjb_solvers.base_hjb:newton_hjb_step", "sigma_at_n", "volatility_at_n"),
        ("mfgarchon.alg.numerical.hjb_solvers.base_hjb:solve_hjb_timestep_newton", "sigma_at_n", "volatility_at_n"),
        (
            "mfgarchon.alg.numerical.gfdm_components.monotonicity_enforcer:MonotonicityEnforcer",
            "sigma_function",
            "volatility_function",
        ),
    ],
)
def test_the_part3_renames_refuse_the_old_keyword_by_name(owner, old, new):
    """Ruling 20 (2026-09-28): the volatility's sigma-prefixed public keywords are renamed, and the old
    spelling raises a TypeError naming the new one rather than failing on an unexpected keyword."""
    module, name = owner.split(":")
    func = getattr(importlib.import_module(module), name)
    with pytest.raises(TypeError, match=rf"no longer takes {old}=; pass {new}="):
        func(**{old: 0.3})


def test_the_grid_config_refusal_names_volatility():
    """`extra="forbid"` already rejects `sigma=` as an unknown field; the refusal makes the error name
    `volatility=`."""
    from mfgarchon.config.array_validation import MFGGridConfig

    assert MFGGridConfig(Nx=100, Nt=1000, volatility=0.3).volatility == 0.3
    with pytest.raises(ValueError, match=r"MFGGridConfig\(sigma=\.\.\.\) is retired .*pass volatility="):
        MFGGridConfig(Nx=100, Nt=1000, sigma=0.3)


def test_the_retired_attributes_name_the_replacement():
    from mfgarchon.alg.optimization.variational_problem import VariationalMFGProblem
    from mfgarchon.core.hamiltonian import HamiltonianValues

    values = HamiltonianValues(H=np.zeros(3), dH_dp=np.zeros((3, 1)), volatility=np.full(3, 0.2))
    problem = VariationalMFGProblem(volatility=0.2)
    assert problem.volatility == 0.2
    for owner in (values, problem):
        with pytest.raises(AttributeError, match=r"\.sigma is retired .*\.volatility"):
            _ = owner.sigma


def test_the_strict_adjoint_step_reads_a_tensor_by_its_kind_not_by_identity():
    """FP-FDM's strict-adjoint step refused the problem's tensor by checking `is problem.volatility`, so
    an equal tensor passed as an override was read as a per-point field. The kind decides now."""
    from scipy import sparse

    from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
    from mfgarchon.core.mfg_problem import MFGProblem
    from mfgarchon.core.model import Conditions, Model
    from mfgarchon.geometry import TensorProductGrid
    from mfgarchon.geometry.boundary import no_flux_bc

    tensor = np.array([[0.3, 0.1], [0.1, 0.2]])
    grid = TensorProductGrid(bounds=[(0.0, 1.0)] * 2, Nx_points=[6, 5], boundary_conditions=no_flux_bc(dimension=2))
    problem = MFGProblem(
        model=Model(
            hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
            volatility=tensor,
            volatility_kind="tensor",
        ),
        domain=grid,
        conditions=Conditions(m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0, T=0.2),
        Nt=4,
    )
    solver = FPFDMSolver(problem)
    m = np.ones((6, 5))
    a_t = sparse.csr_matrix((30, 30))
    with pytest.raises(NotImplementedError, match=r"volatility_kind='tensor'"):
        solver.solve_fp_step_adjoint_mode(m, a_t, volatility=tensor.copy(), volatility_kind="tensor")
    field = np.full((6, 5), 0.25)
    assert np.all(np.isfinite(solver.solve_fp_step_adjoint_mode(m, a_t, volatility=field, volatility_kind="field")))
