"""`dp` and `dm` each compute the same central difference twice; pin them together.

`HamiltonianBase.dp` evaluates `H` on a stacked `p` and divides by `2*eps` inline (the batch path,
2 calls total); on `TypeError`/`ValueError` it falls back to `_finite_diff_dp`, which does the same
arithmetic per point (2*d calls each). `dm` has the identical shape. Neither is wrong and they are
not redundant -- one is vectorised over points and the other is not, which is a real structural
distinction, not duplication. What IS restated is the difference quotient itself, and numerical
code does not crash when two copies of that drift: it converges to a self-consistent wrong answer.

A path-A-vs-path-B comparison is the right pin **while the fork is open** -- it goes tautological
only after a consolidation routes both through one owner, which is not what happened here.

Measured at c98a9c5f: the two paths agree to `0.000e+00` on this fixture, and both sit `2.7e-10`
from the analytic derivative. Positive control: tripling `eps` in the loop path alone moves the
comparison to `3.7e-10`, so this test can see a disagreement.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.core.hamiltonian import HamiltonianBase


class _SmoothProbe(HamiltonianBase):
    """No analytic `dp`/`dm`, so both finite-difference paths are reachable."""

    @property
    def is_hamiltonian(self) -> bool:
        return True

    def __call__(self, x, m, p, t):
        p_arr = np.atleast_2d(np.asarray(p, dtype=float))
        m_arr = np.asarray(m, dtype=float)
        v = 0.5 * np.sum(p_arr**2, axis=-1) + 0.3 * np.sum(p_arr**3, axis=-1) + 0.7 * m_arr.ravel() ** 2
        return v if np.ndim(p) > 1 and p_arr.shape[0] > 1 else float(np.ravel(v)[0])


_N = 5
_X = np.linspace(0.0, 1.0, _N).reshape(-1, 1)
_M = np.linspace(0.5, 1.5, _N)
_P = np.linspace(-2.0, 2.0, _N).reshape(-1, 1)


@pytest.mark.parametrize(
    ("batched", "per_point", "exact"),
    [
        ("dp", "_finite_diff_dp", lambda: _P.ravel() + 0.9 * _P.ravel() ** 2),
        ("dm", "_finite_diff_dm", lambda: 1.4 * _M),
    ],
)
def test_the_two_finite_difference_paths_agree(batched, per_point, exact):
    h = _SmoothProbe()
    b = np.asarray(getattr(h, batched)(_X, _M, _P, 0.0)).reshape(_N, -1)
    loop = np.stack([np.atleast_1d(getattr(h, per_point)(_X[i], float(_M[i]), _P[i], 0.0)) for i in range(_N)]).reshape(
        _N, -1
    )

    assert np.array_equal(b, loop), (
        f"{batched}: batch and per-point finite differences disagree by "
        f"{np.abs(b - loop).max():.3e}; they compute the same quotient"
    )
    # An external oracle, so agreement alone cannot pass over two identically-wrong copies.
    assert np.abs(b.ravel() - exact()).max() < 1e-6, f"{batched}: both paths are off the analytic derivative"
