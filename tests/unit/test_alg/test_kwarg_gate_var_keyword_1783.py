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
import pathlib
from dataclasses import dataclass
from typing import Any

import pytest

import numpy as np

from mfgarchon.alg.numerical.coupling import base_mfg
from mfgarchon.alg.numerical.coupling.base_mfg import resolve_volatility_kwarg


@dataclass
class _Problem:
    """Only the two attributes the owner reads."""

    sigma: Any
    volatility_field: Any = None


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
        resolve_volatility_kwarg(params, 0.7, _Problem(0.3), "FakeSolver", "solve_hjb_system", "HJB")

    # And it does NOT refuse when the parameter is named.
    named = inspect.signature(lambda self, volatility_field=None: None).parameters
    assert resolve_volatility_kwarg(named, 0.7, _Problem(0.3), "FakeSolver", "solve_hjb_system", "HJB") == {
        "volatility_field": 0.7
    }


def test_the_message_names_the_signature_and_the_consequence():
    """A refusal that does not say what would have gone wrong sends the reader to the wrong fix.

    The natural but wrong response to "solver does not accept volatility_field" is to add
    ``**kwargs`` to the solver -- which is what caused the defect. The message has to say that
    explicitly.
    """
    params = inspect.signature(lambda self, *args, **kwargs: None).parameters
    with pytest.raises(NotImplementedError) as exc:
        resolve_volatility_kwarg(params, 0.7, _Problem(0.3), "MeshlessGalerkinHJBSolver", "solve_hjb_system", "HJB")
    message = str(exc.value)
    assert "MeshlessGalerkinHJBSolver.solve_hjb_system" in message
    assert "different diffusion" in message, "the consequence must be stated, not just the refusal"
    assert "**kwargs does not count" in message, (
        "without this the reader's natural fix is to widen the signature, which is the defect"
    )


def test_every_coupling_path_routes_through_one_owner():
    """Four call sites, one decision. Scanned across the package, not one class.

    The first version of this test read ``inspect.getsource(BaseCouplingIterator)`` and was green
    while ``mfg_residual`` -- the Newton coupling path, in the same package -- carried two live
    inline copies that dropped the field silently. A single-owner guard scoped to one class cannot
    see the neighbouring path, which is the whole shape it exists to catch.
    """
    package = pathlib.Path(base_mfg.__file__).parent
    modules = sorted(package.glob("*.py"))
    assert len(modules) >= 4, f"expected the coupling package, found {len(modules)} files"

    inline = {m.name: m.read_text(encoding="utf-8").count('"volatility_field" in ') for m in modules}
    assert sum(inline.values()) == 1, (
        f"exactly one membership test may exist -- the one inside resolve_volatility_kwarg. Found "
        f"{ {k: v for k, v in inline.items() if v} }. An inline copy is the form that dropped the "
        f"value silently on both Picard (#1783) and Newton (found reviewing the #1783 fix)."
    )
    assert inline["base_mfg.py"] == 1, "the surviving one must be the owner's"

    # Per file, not a total: a total lets one path lose a call site while another gains one.
    uses = {m.name: m.read_text(encoding="utf-8").count("resolve_volatility_kwarg(") for m in modules}
    assert uses["base_mfg.py"] == 3, f"Picard: expected the definition + both sides, found {uses['base_mfg.py']}"
    assert uses["mfg_residual.py"] == 2, f"Newton: expected both sides, found {uses['mfg_residual.py']}"


def test_an_exempt_scalar_is_still_forwarded_when_the_solver_names_it():
    """Indistinguishable from sigma is a reason not to REFUSE, not a reason not to FORWARD.

    The first fix of #1783 put the forward inside the hazard branch, so a scalar equal to
    ``problem.sigma`` stopped being forwarded at all. That is silent-wrong in the same way the
    original defect was: ``problem.volatility_field`` is not always ``problem.sigma`` -- construct
    with an array sigma and the field is the array while ``sigma`` is its mean -- so a solver
    falling back through ``get_diffusion_coefficient_field(None)`` picks up the array instead of
    the constant the caller asked for.
    """
    named = inspect.signature(lambda self, volatility_field=None: None).parameters
    array_sigma = _Problem(0.3, volatility_field=np.full(9, 0.3))
    assert resolve_volatility_kwarg(named, 0.3, array_sigma, "FDM", "solve_hjb_system", "HJB") == {
        "volatility_field": 0.3
    }, "an explicit override must reach the solver even when it equals problem.sigma"

    # The exemption still applies where it was needed: a solver that cannot take it, and a field
    # whose loss changes nothing. Without this branch every ordinary solve would refuse.
    kw_only = inspect.signature(lambda self, *args, **kwargs: None).parameters
    assert resolve_volatility_kwarg(kw_only, 0.3, _Problem(0.3), "X", "solve_hjb_system", "HJB") == {}


def test_a_numpy_scalar_is_not_a_hazard():
    """``np.float32(0.3)`` is neither ``int`` nor ``float``; refusing it refuses an identical solve."""
    kw_only = inspect.signature(lambda self, *args, **kwargs: None).parameters
    p = _Problem(np.float32(0.3))
    assert resolve_volatility_kwarg(kw_only, np.float32(0.3), p, "X", "solve_hjb_system", "HJB") == {}


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
