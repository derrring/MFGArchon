"""The translator threads what the user SET, not what differs from a default (Issue #2266).

`config/translator.py` threaded a field only when its value differed from the Pydantic
default. That rule is sound if and only if ``Pydantic default == solver constructor
default`` for every threaded field. Where the two disagreed, a user who wrote the config's
own documented default silently got the solver's different value:

* ``SLConfig.interpolation_method`` defaults to ``'cubic'`` while
  ``HJBSemiLagrangianSolver`` defaults to ``'linear'``, so an explicit ``'cubic'`` was
  dropped. Whether that was *observable* depended on the scheme:
  ``factory/scheme_factory.py`` re-supplies the value by ``setdefault``, so ``SL_CUBIC`` +
  ``'cubic'`` ran cubic anyway and only ``SL_LINEAR`` + ``'cubic'`` silently ran linear.
  (#2266's own matrix reports SL_CUBIC as running linear; measured, it does not. The
  accidental masking is the point: the same dropped field is visible under one scheme and
  invisible under another.)
* ``NewtonConfig.max_iterations`` defaults to 10 while ``HJBFDMSolver`` resolves ``None`` to
  ``DEFAULT_NEWTON_MAX_ITERATIONS = 30`` — asking for 10 got a 3x budget.

The rule is now ``field in model_fields_set``: an explicitly-set field is threaded whatever
its value. #2250 is the loud member of the same class — it was found only because it
happened to raise, which is the argument for fixing the rule rather than the three fields.

Retirement condition: these trip if the translator returns to comparing values, or if a
threaded field's two defaults are allowed to diverge again.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.config import MFGSolverConfig
from mfgarchon.config.mfg_methods import NewtonConfig, SLConfig
from mfgarchon.config.translator import hjb_config_to_kwargs
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.types import NumericalScheme


def _problem():
    """v1.0 API, and a density whose grid-measure mass is exactly 1.

    Both matter to the warnings ratchet: the legacy ``MFGProblem(geometry=, components=)``
    form emits a DeprecationWarning, and an unnormalised ``m_initial`` emits the #1887
    "mass is not 1" UserWarning. A new test should not be the thing that teaches either.
    """
    return MFGProblem(
        model=Model(
            hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0), coupling=lambda m: m),
            sigma=0.3,
        ),
        domain=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1)),
        conditions=Conditions(
            u_terminal=lambda x: np.squeeze(0.5 * (np.asarray(x) - 0.5) ** 2),
            m_initial=lambda x: 1.0,  # uniform on [0, 1]: grid-measure mass is exactly 1
            T=0.1,
        ),
        Nt=4,
    )


class TestAnExplicitlySetFieldReachesTheSolver:
    def test_an_explicitly_asked_for_interpolation_changes_the_answer(self):
        """The end-to-end oracle, on the ONE fixture where the defect is expressible.

        The scheme is ``SL_LINEAR`` and that is load-bearing, not incidental.
        ``factory/scheme_factory.py`` does ``hjb_config.setdefault("interpolation_method",
        ...)`` -- ``"linear"`` for ``SL_LINEAR``, ``"cubic"`` for ``SL_CUBIC``. So a dropped
        field is masked whenever the scheme's own default happens to be the value the user
        asked for, and under ``SL_CUBIC`` it did: measured pre-fix, ``SL_CUBIC`` + ``'cubic'``
        ran cubic, and only ``SL_LINEAR`` + ``'cubic'`` ran linear.

        A version of this test written on ``SL_CUBIC`` passes on the UNFIXED tree, because
        explicit ``'linear'`` was threaded there and overrode the scheme while ``'cubic'``
        fell through to the same cubic. It compares "explicit vs scheme default", not "asked
        for vs got". Measured, and the reason this fixture is the one used.

        Under ``SL_LINEAR`` both values collapse to linear before the fix, so the two solves
        are bit-identical and this assertion cannot pass; after it they differ.
        """
        problem = _problem()
        u = {}
        for value in ("linear", "cubic"):
            cfg = MFGSolverConfig()
            cfg.hjb.sl = SLConfig(interpolation_method=value)
            u[value] = problem.solve(config=cfg, scheme=NumericalScheme.SL_LINEAR, max_iterations=3).U
        assert np.max(np.abs(u["linear"] - u["cubic"])) > 0.0, (
            "SL_LINEAR ran identical solves for linear and cubic interpolation, so the config "
            "field is not reaching the solver (#2266)"
        )

    @pytest.mark.parametrize("value", [10, 30, 5])
    def test_the_newton_budget_asked_for_is_the_one_threaded(self, value):
        """10 is the Pydantic default and was therefore dropped, yielding the solver's 30."""
        cfg = MFGSolverConfig()
        cfg.hjb.newton = NewtonConfig(max_iterations=value)
        kwargs = hjb_config_to_kwargs(cfg.hjb, NumericalScheme.FDM_UPWIND)
        assert kwargs.get("max_newton_iterations") == value

    @pytest.mark.parametrize("value", ["linear", "cubic"])
    def test_the_interpolation_asked_for_is_the_one_threaded(self, value):
        cfg = MFGSolverConfig()
        cfg.hjb.sl = SLConfig(interpolation_method=value)
        kwargs = hjb_config_to_kwargs(cfg.hjb, NumericalScheme.SL_CUBIC)
        assert kwargs.get("interpolation_method") == value

    @pytest.mark.parametrize("scheme", [NumericalScheme.FDM_UPWIND, NumericalScheme.SL_LINEAR, NumericalScheme.GFDM])
    def test_an_untouched_config_still_threads_nothing(self, scheme):
        """The control, and the compatibility guarantee.

        The whole point of the old rule was that callers relying on solver defaults are
        unaffected. A config nobody has touched has an empty ``model_fields_set``, so it must
        still thread nothing — if this ever returned kwargs, the new rule would be silently
        overriding every solver default in the package.
        """
        assert hjb_config_to_kwargs(MFGSolverConfig().hjb, scheme) == {}
