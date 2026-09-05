"""Both coupling iterators refuse a backend they cannot run on (Issue #2250).

`FixedPointIterator` and `FictitiousPlayIterator` annotated ``backend`` as ``str | None``
and documented it as a name, then used it as an object in exactly one place -- the
cold-start allocation ``self.U = self.backend.zeros(...)``. Every non-``None`` value raised
``AttributeError: 'str' object has no attribute 'zeros'``, *including* ``"numpy"``.

The defect was invisible because the default is ``None``, which took the ``np.zeros``
branch: the whole suite and every example ran the working path, and no test in the
repository constructed either iterator with a non-``None`` backend.

Retirement condition: these tests trip when the coupling loop learns to run on a real
backend (#1922). At that point the refusal is wrong and this file should be replaced by a
test that the chosen backend is actually *used* -- not merely accepted, which is the
failure #2250 records.

Both iterators are covered deliberately: they held the same three lines, so a fix applied
to one leaves the other, and only a test naming both can fail on that.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.alg.numerical.coupling.base_mfg import resolve_supported_backend
from mfgarchon.alg.numerical.coupling.fictitious_play import FictitiousPlayIterator
from mfgarchon.alg.numerical.coupling.fixed_point_iterator import FixedPointIterator
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_problem import MFGComponents, MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

ITERATORS = [FixedPointIterator, FictitiousPlayIterator]


def _problem():
    return MFGProblem(
        geometry=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=no_flux_bc(dimension=1)),
        T=0.1,
        Nt=4,
        sigma=0.3,
        components=MFGComponents(
            m_initial=lambda x: np.exp(-30 * (x - 0.5) ** 2),
            u_terminal=lambda x: 0.0 * x,
            hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0), coupling=lambda m: m),
        ),
    )


class TestBackendSelectionIsRefusedNotIgnored:
    @pytest.mark.parametrize("iterator_cls", ITERATORS, ids=lambda c: c.__name__)
    @pytest.mark.parametrize("backend", ["jax", "torch", "numba"])
    def test_an_unsupported_backend_is_refused_at_construction(self, iterator_cls, backend):
        """Refused where it is passed, not as an AttributeError deep inside solve()."""
        problem = _problem()
        with pytest.raises(NotImplementedError, match="#2250"):
            iterator_cls(problem, hjb_solver=None, fp_solver=None, backend=backend)

    @pytest.mark.parametrize("iterator_cls", ITERATORS, ids=lambda c: c.__name__)
    @pytest.mark.parametrize("backend", [None, "numpy", "NumPy"])
    def test_the_backend_the_loop_actually_runs_on_is_accepted_and_normalised(self, iterator_cls, backend):
        """``None`` and ``"numpy"`` both describe what runs, so both normalise to ``None``.

        Storing the *name* is what re-arms the original AttributeError, so asserting the
        stored value is ``None`` -- not merely that construction succeeded -- is the half
        that would catch a fix which only stopped raising.
        """
        it = iterator_cls(_problem(), hjb_solver=None, fp_solver=None, backend=backend)
        assert it.backend is None

    def test_a_non_name_non_backend_value_is_refused(self):
        """An int is neither a name nor a backend; the message names #1922, the capability."""
        with pytest.raises(NotImplementedError, match="#1922"):
            resolve_supported_backend(42, "T")
