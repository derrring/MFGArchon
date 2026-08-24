"""Issue #2066 / #2081: Howard's operator builders close the stencil row unconditionally.

`_build_dlap_from_socp` / `_build_dgrad_central` wrote stencil weights verbatim and never closed
the row. That was correct only because their names asserted SOCP input: SOCP weights satisfy the
sum rule by construction -- `build_taylor_matrix_*` sets `A[:, 0] = 1.0` and the consistency
constraint is `e_lap == A.T @ L` with `e_lap[0] == 0.0`, so row 0 of that equality IS
`sum_j L_j == 0`.

Once operators can come from elsewhere (#2066) that guarantee is gone. `TaylorOperator` and
`UpwindOperator` return weights multiplying deviations `u_j - u_i`, whose rows do not sum to zero;
written verbatim they produce a Laplacian wrong by O(1e+2).

The closure is mandatory for deviation weights and idempotent for sum-rule ones, so it is applied
unconditionally with no branch on provenance. These tests pin both halves.
"""

from __future__ import annotations

import numpy as np

from mfgarchon.alg.numerical.gfdm_components.gfdm_strategies import create_operator
from mfgarchon.alg.numerical.hjb_solvers.hjb_howard import (
    _build_dgrad_central_from_weights,
    _build_dlap_from_weights,
)


def _cloud(n: int = 41):
    pts = np.linspace(0.0, 1.0, n).reshape(-1, 1)
    return pts, np.arange(4, n - 4)


def _stencils(op, n: int) -> dict:
    """The per-point weight dicts, in the shape both builders consume."""
    out = {}
    for i in range(n):
        w = op.get_derivative_weights(i)
        if w is not None:
            out[i] = w
    return out


def test_a_deviation_convention_operator_is_exact_once_the_row_is_closed() -> None:
    """`TaylorOperator` weights need the closure; without it the Laplacian is wrong by O(1e+2)."""
    pts, interior = _cloud()
    n = len(pts)
    op = create_operator(pts, delta=0.1, method="taylor")
    stencils = _stencils(op, n)

    d_lap = _build_dlap_from_weights(stencils, n)
    q = 0.5 * pts[:, 0] ** 2  # Lap q == 1 exactly

    err = np.abs((d_lap @ q)[interior] - 1.0).max()
    assert err < 1e-11, f"closed Laplacian should be exact on a quadratic, got {err:.3e}"

    row_sums = np.abs(np.asarray(d_lap.sum(axis=1)).ravel()[interior]).max()
    assert row_sums < 1e-10, f"closed rows must sum to zero, worst {row_sums:.3e}"

    # Two-sided: the unclosed matrix must be wrong by O(1), which is what makes the closure
    # load-bearing rather than cosmetic. Built here rather than asserted from memory.
    from scipy.sparse import lil_matrix

    open_mat = lil_matrix((n, n))
    for i, w in stencils.items():
        nbr = np.asarray(w["neighbor_indices"])
        lap_w = np.asarray(w["lap_weights"], dtype=float)
        for k, j in enumerate(nbr):
            if j >= 0:
                open_mat[i, int(j)] += float(lap_w[k])
    err_open = np.abs((open_mat.tocsr() @ q)[interior] - 1.0).max()
    assert err_open > 1.0, (
        f"the unclosed Laplacian must be wrong by O(1) -- got {err_open:.3e}. If this is now "
        f"small, TaylorOperator's weight convention changed and #2081's contract needs revisiting"
    )


def test_the_gradient_builder_closes_its_row_at_the_boundary() -> None:
    """The gradient closure is load-bearing only where the stencil is one-sided.

    A symmetric interior stencil gives antisymmetric first-derivative weights, which already sum
    to zero -- measured 4.796e-14 over interior rows against 5.134e+01 over boundary rows on this
    cloud. So an interior-only assertion cannot see the closure removed, and this test would have
    been vacuous restricted the way the Laplacian one is. The Laplacian has no such exemption:
    its weights sum to 1.301e+03 in the interior.
    """
    pts, _ = _cloud()
    n = len(pts)
    op = create_operator(pts, delta=0.1, method="taylor")
    stencils = _stencils(op, n)

    d_grad = _build_dgrad_central_from_weights(stencils, n, 0)
    lin = pts[:, 0]  # d/dx == 1 exactly, and a one-sided stencil reproduces it too

    err = np.abs((d_grad @ lin) - 1.0).max()
    assert err < 1e-11, f"closed gradient should be exact on a linear field, got {err:.3e}"

    row_sums = np.abs(np.asarray(d_grad.sum(axis=1)).ravel()).max()
    assert row_sums < 1e-10, f"closed rows must sum to zero, worst {row_sums:.3e}"


def test_the_closure_is_a_no_op_on_sum_rule_weights() -> None:
    """SOCP weights already sum to zero; closing them must not move the operator.

    This is why the builders need no branch on where the weights came from. `LocalRBFOperator`
    stands in for SOCP here -- both satisfy the sum rule, and it needs no cvxpy.
    """
    pts, interior = _cloud()
    n = len(pts)
    op = create_operator(pts, delta=0.1, method="rbf")
    stencils = _stencils(op, n)

    for i, w in stencils.items():
        if i in interior:
            total = float(np.asarray(w["lap_weights"], dtype=float).sum())
            assert abs(total) < 1e-8, f"point {i}: these weights are not sum-rule, sum={total:.3e}"

    d_lap = _build_dlap_from_weights(stencils, n)
    q = 0.5 * pts[:, 0] ** 2
    err = np.abs((d_lap @ q)[interior] - 1.0).max()
    assert err < 1e-9, f"closing a sum-rule operator must leave it exact, got {err:.3e}"


def test_a_repeated_column_accumulates_rather_than_overwrites() -> None:
    """`neighbor_indices` is not guaranteed unique -- a periodic geometry maps ghost images back.

    Assignment would keep the last weight and silently drop the rest; the row would then not sum
    to zero and the operator would be wrong at exactly the points a periodic wrap touches.
    """
    n = 5
    stencils = {
        2: {
            "neighbor_indices": np.array([1, 3, 1]),  # column 1 twice, different weights
            "lap_weights": np.array([2.0, 3.0, 5.0]),
            "grad_weights": np.array([[1.0, -1.0, 4.0]]),
        }
    }
    d_lap = _build_dlap_from_weights(stencils, n)
    assert d_lap[2, 1] == 7.0, f"the repeated column must accumulate 2+5, got {d_lap[2, 1]}"
    assert d_lap[2, 3] == 3.0
    assert d_lap[2, 2] == -10.0, f"diagonal must close the full row sum, got {d_lap[2, 2]}"
    assert abs(d_lap.sum(axis=1)[2, 0]) < 1e-12
