"""Issue #1783: the coupling kwarg gate must not be fooled by a ``**kwargs`` override.

``BaseCouplingIterator._build_hjb_kwargs`` / ``_build_fp_kwargs`` decide whether to forward
``volatility_field`` by asking ``inspect.signature`` whether the solver names the parameter. That
answers "does this callable name it", not "can this solver consume it", and a subclass declaring
``(self, *args, **kwargs)`` makes the two diverge.

Before the fix the field was dropped silently on the side that did not name it. Measured on the
meshless pair with ``problem.sigma = 0.3`` and a field of mean 0.7: the HJB side ran at D = 0.045
while the paired FP side ran at D = 0.245 -- a 5.4x mismatch, no warning, and a converged density
for a problem nobody posed.

Three lines below, the same function already raised for exactly this situation with
``source_term`` (#1424). Two adjacent branches of one function, opposite policies.
"""

from __future__ import annotations

import inspect

import pytest

import numpy as np

from mfgarchon.alg.numerical.coupling.base_mfg import BaseCouplingIterator


def test_a_var_keyword_override_does_not_count_as_accepting_the_parameter():
    """The gate must refuse, not guess, when the name is absent.

    Treating ``VAR_KEYWORD`` as accept-anything was the other candidate fix. It assumes a solver
    consumes what it swallows -- which is precisely the assumption that produced #1316, where
    three HJB solvers declared ``volatility_field`` and ignored it.
    """
    params = inspect.signature(lambda self, *args, **kwargs: None).parameters
    assert "volatility_field" not in params
    assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()), (
        "the fixture must actually have **kwargs, or it tests nothing about the hole"
    )

    with pytest.raises(NotImplementedError, match="does not accept 'volatility_field'"):
        BaseCouplingIterator._require_kwarg(
            params, "volatility_field", "FakeSolver", "solve_hjb_system", "Consequence."
        )

    # And it does NOT refuse when the parameter is named.
    named = inspect.signature(lambda self, volatility_field=None: None).parameters
    BaseCouplingIterator._require_kwarg(named, "volatility_field", "FakeSolver", "solve_hjb_system", "Consequence.")


def test_the_message_names_the_signature_and_the_consequence():
    """A refusal that does not say what would have gone wrong sends the reader to the wrong fix.

    The natural but wrong response to "solver does not accept volatility_field" is to add
    ``**kwargs`` to the solver -- which is what caused the defect. The message has to say that
    explicitly.
    """
    params = inspect.signature(lambda self, *args, **kwargs: None).parameters
    with pytest.raises(NotImplementedError) as exc:
        BaseCouplingIterator._require_kwarg(
            params,
            "volatility_field",
            "MeshlessGalerkinHJBSolver",
            "solve_hjb_system",
            "Dropping it would leave the two equations on different diffusion.",
        )
    message = str(exc.value)
    assert "MeshlessGalerkinHJBSolver.solve_hjb_system" in message
    assert "different diffusion" in message, "the consequence must be stated, not just the refusal"
    assert "**kwargs does not count" in message, (
        "without this the reader's natural fix is to widen the signature, which is the defect"
    )


def test_both_coupling_sides_route_through_one_owner():
    """HJB and FP had two copies of this gate. A fix to one is a fix to one.

    The repo's dominant defect class is a convention with a private copy on a neighbouring path;
    this pins that the two sides share an owner rather than agreeing by coincidence.
    """
    source = inspect.getsource(BaseCouplingIterator)
    assert source.count("_require_kwarg") >= 3, (
        "expected one definition and two call sites; a lower count means one side has drifted back to an inline check"
    )
    assert source.count('"volatility_field" in params') == 0, (
        "an inline membership test has reappeared -- that is the form that dropped the value silently"
    )


def test_the_meshless_pair_is_refused_end_to_end():
    """The real configuration, through the public constructor.

    `volatility_field` is a CONSTRUCTOR argument of the iterator, not a `solve()` kwarg -- passing
    it to `solve()` is swallowed by **kwargs and never reaches the gate, which is how a first
    attempt at this test measured nothing while appearing to pass.
    """
    from mfgarchon.alg.numerical.coupling.fixed_point_iterator import FixedPointIterator
    from mfgarchon.alg.numerical.meshless_galerkin.fp_solver import MeshlessGalerkinFPSolver
    from mfgarchon.alg.numerical.meshless_galerkin.hjb_solver import MeshlessGalerkinHJBSolver
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
    from mfgarchon.core.mfg_components import MFGComponents
    from mfgarchon.core.mfg_problem import MFGProblem
    from mfgarchon.geometry import TensorProductGrid
    from mfgarchon.geometry.boundary import no_flux_bc

    n = 21
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[n], boundary_conditions=no_flux_bc(dimension=1))
    problem = MFGProblem(
        geometry=grid,
        T=0.2,
        Nt=5,
        sigma=0.3,
        components=MFGComponents(
            m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2),
            u_terminal=lambda x: 0.0,
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0),
                coupling=lambda m: m,
                coupling_dm=lambda m: 1.0,
            ),
        ),
    )
    points = np.linspace(0.0, 1.0, n).reshape(-1, 1)
    delta = 2.6 / np.sqrt(n)

    def pair():
        return (
            MeshlessGalerkinHJBSolver(problem, points, delta=delta),
            MeshlessGalerkinFPSolver(problem, points, delta=delta),
        )

    hjb, _fp = pair()
    assert "volatility_field" not in inspect.signature(hjb.solve_hjb_system).parameters, (
        "this solver must be the **kwargs shape, or the test does not exercise the hole"
    )

    # Without the field, nothing changes -- the refusal must not break ordinary use.
    FixedPointIterator(problem, *pair()).solve(max_iterations=2, verbose=False)

    with pytest.raises(NotImplementedError, match="does not accept 'volatility_field'"):
        FixedPointIterator(problem, *pair(), volatility_field=np.linspace(0.5, 0.9, n)).solve(
            max_iterations=2, verbose=False
        )
