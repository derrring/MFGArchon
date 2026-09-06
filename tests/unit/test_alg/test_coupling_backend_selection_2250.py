"""A backend name resolves, an object passes through, and an unusable one is refused (#2250).

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

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.coupling.base_mfg import allocate_state_arrays, resolve_backend
from mfgarchon.alg.numerical.coupling.fictitious_play import FictitiousPlayIterator
from mfgarchon.alg.numerical.coupling.fixed_point_iterator import FixedPointIterator
from mfgarchon.backends import create_backend
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

ITERATORS = [FixedPointIterator, FictitiousPlayIterator]


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


class TestBackendSelectionIsResolvedOrRefused:
    """A name resolves, an object passes through, and one that cannot carry a solve is refused."""

    @pytest.mark.parametrize("iterator_cls", ITERATORS, ids=lambda c: c.__name__)
    def test_a_backend_name_is_resolved_to_an_object(self, iterator_cls):
        """Storing the NAME is what raised `'str' object has no attribute 'zeros'`."""
        it = iterator_cls(_problem(), hjb_solver=None, fp_solver=None, backend="numpy")
        assert not isinstance(it.backend, str)
        assert hasattr(it.backend, "zeros")

    @pytest.mark.parametrize("iterator_cls", ITERATORS, ids=lambda c: c.__name__)
    def test_a_backend_object_is_passed_through_unchanged(self, iterator_cls):
        """A NumPyBackend object solved end to end before this change, so refusing it would be
        a capability regression rather than a fix -- `acceleration_comparison.py` passes one."""
        backend = create_backend("numpy")
        it = iterator_cls(_problem(), hjb_solver=None, fp_solver=None, backend=backend)
        assert it.backend is backend

    @pytest.mark.parametrize("iterator_cls", ITERATORS, ids=lambda c: c.__name__)
    def test_none_stays_none(self, iterator_cls):
        assert iterator_cls(_problem(), hjb_solver=None, fp_solver=None, backend=None).backend is None

    def test_a_non_backend_value_is_refused_by_type(self):
        with pytest.raises(TypeError, match="#2250"):
            resolve_backend(42, "T")

    def test_a_backend_that_accepts_only_its_own_type_is_refused(self):
        """The asymmetric case, and the one a self-assignment probe cannot see.

        A torch tensor accepts its own element back -- ``U[0] = U[0]`` succeeds -- and rejects
        the numpy array the coupling loop actually assigns. Measured on shape (3, 4): the
        symmetric probe reads ok / TypeError / **ok** for numpy / jax / torch, while assigning
        a numpy row reads ok / TypeError / TypeError. So a probe written the obvious way passes
        torch and fails jax while both are equally unusable, and torch reached the solve and
        died there instead. This fake reproduces exactly that asymmetry.
        """

        class _OwnTypeOnly:
            class _Arr(np.ndarray):
                def __setitem__(self, key, value):
                    if not isinstance(value, _OwnTypeOnly._Arr):
                        raise TypeError("can't assign a numpy.ndarray to this array")
                    super().__setitem__(key, value)

            def zeros(self, shape):
                return np.zeros(shape).view(self._Arr)

        with pytest.raises(NotImplementedError, match="#1922"):
            allocate_state_arrays(_OwnTypeOnly(), (3, 4), "T")

    def test_an_unwritable_backend_is_refused_at_allocation_with_a_reason(self):
        """The check lives at the allocation, not the constructor, and that is load-bearing.

        ``self.backend`` is a plain public attribute and this repository's own example
        (``examples/basic/solvers/acceleration_comparison.py``) assigns to it AFTER
        construction. A constructor-only guard cannot see that path, and with the allocation
        branch present it would silently ignore the assigned backend -- the exact
        wrong-config-silently-ignored failure #2250 exists to remove.
        """

        class _Immutable:
            def zeros(self, shape):
                a = np.zeros(shape)
                a.flags.writeable = False
                return a

        with pytest.raises(NotImplementedError, match="#1922"):
            allocate_state_arrays(_Immutable(), (3, 4), "T")

    def test_a_writable_backend_allocates_normally(self):
        """The control for the test above: same call, a backend that CAN be written."""
        U, M = allocate_state_arrays(create_backend("numpy"), (3, 4), "T")
        U[0, 0] = 1.0
        assert U.shape == (3, 4)
        assert M.shape == (3, 4)

    def test_no_backend_allocates_numpy(self):
        U, M = allocate_state_arrays(None, (3, 4), "T")
        assert isinstance(U, np.ndarray)
        assert isinstance(M, np.ndarray)
