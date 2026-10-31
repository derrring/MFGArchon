"""The backend-aware helpers read ``hasattr(x, "device")`` as "a torch tensor"; NumPy >= 2 arrays have it (#2825).

Defect pins, one per site the owner ``compat._is_torch_tensor`` serves: putting the old predicate back at
``backend_aware_zeros_like``, ``backend_aware_ones_like`` or ``backend_aware_assign`` turns a test here red, and
so does the module-name predicate at either ``*_like`` site. ``backend_aware_assign`` converts a value only for
an MPS target, so its pin runs on Apple silicon only: in the local gate, never on CI's runner. Not caught: an
ndarray assigned into a CPU tensor, which that function does not convert at all.
"""

import pytest

import numpy as np

from mfgarchon.backends.compat import backend_aware_assign, backend_aware_ones_like, backend_aware_zeros_like


def test_numpy_arrays_keep_numpy_for_ones_and_zeros_like():
    template = np.array([1.0, 2.0])

    zeros = backend_aware_zeros_like(template)
    ones = backend_aware_ones_like(template)

    assert type(zeros) is np.ndarray
    assert type(ones) is np.ndarray
    np.testing.assert_array_equal(zeros, np.array([0.0, 0.0]))
    np.testing.assert_array_equal(ones, np.array([1.0, 1.0]))


def test_a_tensor_subclass_defined_outside_torch_stays_a_tensor():
    torch = pytest.importorskip("torch")

    class OutsideTensor(torch.Tensor):  # its module is this file's, as torchvision's tensors carry torchvision's
        pass

    template = torch.ones(2).as_subclass(OutsideTensor)

    assert type(backend_aware_zeros_like(template)) is OutsideTensor
    assert type(backend_aware_ones_like(template)) is OutsideTensor


def test_an_ndarray_assigned_into_an_mps_tensor_is_converted():
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("backend_aware_assign converts a value only for an MPS target")
    target = torch.zeros((2, 3), device="mps")

    backend_aware_assign(target, (1, slice(None)), np.ones(3))

    assert target.cpu().tolist() == [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
