import numpy as np

from mfgarchon.backends.compat import backend_aware_ones_like, backend_aware_zeros_like


def test_numpy_arrays_keep_numpy_for_ones_and_zeros_like():
    template = np.array([1.0, 2.0])

    zeros = backend_aware_zeros_like(template)
    ones = backend_aware_ones_like(template)

    np.testing.assert_array_equal(zeros, np.array([0.0, 0.0]))
    np.testing.assert_array_equal(ones, np.array([1.0, 1.0]))
