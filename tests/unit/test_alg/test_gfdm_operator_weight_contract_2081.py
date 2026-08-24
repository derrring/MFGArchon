"""Issue #2081: the operator weight contract, and the closure that reconciles it.

``DifferentialOperator.get_derivative_weights`` documents a **raw-value**
(sensitivity) map, under which a row must sum to zero because a constant field
has zero derivative. ``LocalRBFOperator`` satisfies that. ``TaylorOperator`` —
the default, ``method="direct"`` — and ``UpwindOperator`` do not: their weights
multiply *deviations* ``b_j = u_j - u_center``, so a row sums to something far
from zero. Two conventions behind one method name.

Nothing pinned any of this: before this file, ``grep -rl "create_operator"
tests/`` returned zero files. A change flipping either operator's convention
passed the suite in both directions, and #2077 fed deviation weights to a
raw-value builder in ``hjb_howard.py`` with nothing to catch it.

These tests pin:
  (a) every operator in the ``create_operator`` registry is exact on a
      quadratic (Laplacian) and a linear field (gradient), in 1D and 2D,
      through the production API — the operators themselves are correct;
  (b) each operator's *convention*, so a silent flip in either direction fails;
  (c) the invariant the whole stack rests on: the row closure is **mandatory**
      for a deviation-convention operator and **idempotent** for a sum-rule
      one, so a consumer may apply it unconditionally;
  (d) the three closure forms that appear at the four live consumer sites are
      the same algebra;
  (e) the pre-assembled production matrices carry the closure — this is the
      assertion that fails if ``_preassemble_sparse_matrices`` loses it.

Tolerances come from measurement on ``origin/main`` (``4f0c90d0``), not from a
guess: Taylor/Upwind reach 5.2e-14 on the Laplacian and 2.2e-15 on the
gradient; RBF reaches 4.6e-12 and 2.8e-13. The bounds below sit two to three
orders above those, which still leaves them far below any real discretisation
error — a genuinely inexact operator fails by O(1), as test (c) demonstrates.
"""

from __future__ import annotations

import pytest

import numpy as np
from scipy.sparse import lil_matrix

from mfgarchon.alg.numerical.gfdm_components.gfdm_strategies import create_operator

# Registry of `create_operator` methods, with the tolerance each reaches.
# "deviation" operators need the row closed; "sum_rule" ones arrive closed.
OPERATORS = {
    "taylor": {"convention": "deviation", "tol_lap": 1e-11, "tol_grad": 1e-11},
    "upwind": {"convention": "deviation", "tol_lap": 1e-11, "tol_grad": 1e-11},
    "rbf": {"convention": "sum_rule", "tol_lap": 1e-9, "tol_grad": 1e-9},
}


def _cloud_1d(n: int = 41) -> tuple[np.ndarray, float, np.ndarray]:
    pts = np.linspace(0.0, 1.0, n).reshape(-1, 1)
    interior = np.arange(4, n - 4)
    return pts, 0.1, interior


def _cloud_2d(n: int = 11) -> tuple[np.ndarray, float, np.ndarray]:
    g = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(g, g, indexing="ij")
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    inside = (pts[:, 0] > 0.25) & (pts[:, 0] < 0.75) & (pts[:, 1] > 0.25) & (pts[:, 1] < 0.75)
    return pts, 0.35, inside.nonzero()[0]


CLOUDS = {"1d": _cloud_1d, "2d": _cloud_2d}


def _build(method: str, pts: np.ndarray, delta: float):
    kwargs = {}
    if method == "upwind":
        # UpwindOperator biases stencils along the flow; the direction is
        # irrelevant to polynomial exactness, but the argument is required.
        kwargs["velocity_field"] = np.ones_like(pts)
    return create_operator(pts, delta=delta, method=method, **kwargs)


def _exact_fields(pts: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """`q = |x|^2/2` has `Lap q == dim`; `lin = x_0` has `d/dx_0 == 1`.

    Both are analytic truths, not values recorded from a previous run.
    """
    dim = pts.shape[1]
    q = 0.5 * (pts**2).sum(axis=1)
    lin = pts[:, 0]
    return q, float(dim), lin


@pytest.mark.parametrize("method", sorted(OPERATORS))
@pytest.mark.parametrize("cloud", sorted(CLOUDS))
def test_operator_is_exact_on_quadratic_and_linear(method: str, cloud: str) -> None:
    """(a) Every registry operator differentiates polynomials exactly."""
    pts, delta, interior = CLOUDS[cloud]()
    spec = OPERATORS[method]
    op = _build(method, pts, delta)
    q, lap_exact, lin = _exact_fields(pts)

    err_lap = np.abs(op.laplacian(q)[interior] - lap_exact).max()
    grad = np.asarray(op.gradient(lin)).reshape(len(pts), -1)
    err_grad = np.abs(grad[interior, 0] - 1.0).max()

    assert err_lap < spec["tol_lap"], f"{method} {cloud}: Lap error {err_lap:.3e}"
    assert err_grad < spec["tol_grad"], f"{method} {cloud}: grad error {err_grad:.3e}"


@pytest.mark.parametrize("method", sorted(OPERATORS))
def test_weight_convention_is_the_documented_one(method: str) -> None:
    """(b) Pin each operator's convention, so a silent flip fails either way.

    The contract is the *sum rule*, not the key set — two operators can return
    the same documented keys and mean different things by them.
    """
    pts, delta, interior = _cloud_1d()
    op = _build(method, pts, delta)
    centre = int(interior[len(interior) // 2])
    weights = op.get_derivative_weights(centre)
    assert weights is not None
    row_sum = float(np.asarray(weights["lap_weights"], dtype=float).sum())

    if OPERATORS[method]["convention"] == "deviation":
        # Weights multiply `u_j - u_center`; the row does NOT sum to zero, and
        # the caller owes the closure. Measured magnitude is O(1e2)-O(1e3).
        assert abs(row_sum) > 1.0, (
            f"{method} used to return deviation weights (sum(lap_w) far from 0) "
            f"and now sums to {row_sum:.3e}. If this operator was converted to "
            f"the sum-rule convention, every consumer applying the closure now "
            f"needs re-checking — see #2081."
        )
    else:
        assert abs(row_sum) < 1e-8, (
            f"{method} used to satisfy the ABC's raw-value contract (sum(lap_w) == 0) and now sums to {row_sum:.3e}."
        )


# --- the three closure forms found at the four live consumer sites -----------
# A: TaylorOperator._preassemble_sparse_matrices — `w[k] + diag_val` at centre
# B: HJBGFDMSolver._build_differentiation_matrices — `=` then `+= -sum(all)`
# C: DirectCollocationHandler.apply_to_matrix and
#    HJBGFDMSolver._build_neumann_bc_row — skip centre, `diag = -sum(j != c)`


def _assemble(op, n: int, form: str) -> lil_matrix:
    mat = lil_matrix((n, n))
    for i in range(n):
        weights = op.get_derivative_weights(i)
        if weights is None:
            continue
        nbr = np.asarray(weights["neighbor_indices"])
        lap_w = np.asarray(weights["lap_weights"], dtype=float)

        if form == "none":
            for k, j in enumerate(nbr):
                if j >= 0:
                    mat[i, int(j)] = lap_w[k]
        elif form == "A":
            diag_val = -lap_w.sum()
            for k, j in enumerate(nbr):
                if j >= 0:
                    mat[i, int(j)] = lap_w[k] + diag_val if int(j) == i else lap_w[k]
            if i not in nbr:
                mat[i, i] = diag_val
        elif form == "B":
            total = 0.0
            for k, j in enumerate(nbr):
                if j >= 0:
                    mat[i, int(j)] = lap_w[k]
                    total += lap_w[k]
            mat[i, i] += -total
        elif form == "C":
            centre_w = 0.0
            for k, j in enumerate(nbr):
                if j >= 0 and int(j) != i:
                    mat[i, int(j)] = lap_w[k]
                    centre_w -= lap_w[k]
            mat[i, i] = centre_w
        else:  # pragma: no cover - guards the parametrisation itself
            raise ValueError(f"unknown closure form {form!r}")
    return mat.tocsr()


@pytest.mark.parametrize("method", sorted(OPERATORS))
def test_closure_is_mandatory_for_deviation_and_idempotent_for_sum_rule(method: str) -> None:
    """(c) The invariant that lets every consumer close the row unconditionally.

    Two-sided: for a deviation operator the *unclosed* matrix must be wrong by
    O(1) or more, which is what makes the closed result meaningful. For a
    sum-rule operator both must be right, and identically so.
    """
    pts, delta, interior = _cloud_1d()
    op = _build(method, pts, delta)
    q, lap_exact, _ = _exact_fields(pts)

    open_mat = _assemble(op, len(pts), "none")
    closed = _assemble(op, len(pts), "A")
    err_open = np.abs((open_mat @ q)[interior] - lap_exact).max()
    err_closed = np.abs((closed @ q)[interior] - lap_exact).max()

    assert err_closed < OPERATORS[method]["tol_lap"], (
        f"{method}: closing the row must give the exact Laplacian, got {err_closed:.3e}"
    )

    if OPERATORS[method]["convention"] == "deviation":
        assert err_open > 1.0, (
            f"{method}: the unclosed matrix must be wrong — the closure is "
            f"mandatory, not cosmetic. Got {err_open:.3e}; if this is now small, "
            f"the operator's convention changed (see #2081)."
        )
    else:
        assert err_open < OPERATORS[method]["tol_lap"], (
            f"{method}: a sum-rule operator needs no closure, got {err_open:.3e}"
        )
        # Idempotence, the property consumers rely on. The closure shifts the
        # diagonal by exactly `-sum(weights)`, so the change it makes IS the
        # unclosed row's own residual — not a tolerance chosen to pass, but the
        # mechanism: a row that already sums to zero is left where it was.
        residual = np.abs(np.asarray(open_mat.sum(axis=1)).ravel()).max()
        shift = abs(closed - open_mat).max()
        assert shift <= residual + 1e-15, (
            f"{method}: the closure moved the matrix by {shift:.3e}, more than "
            f"the unclosed row-sum residual {residual:.3e} it should be bounded "
            f"by — it is not acting as a no-op on a sum-rule operator"
        )
        assert residual < 1e-10, (
            f"{method}: a sum-rule operator's rows must already sum to zero, largest residual {residual:.3e}"
        )


@pytest.mark.parametrize("method", sorted(OPERATORS))
def test_the_three_live_closure_forms_are_the_same_algebra(method: str) -> None:
    """(d) The four production sites spell the closure three ways; all agree."""
    pts, delta, interior = _cloud_1d()
    op = _build(method, pts, delta)
    q, lap_exact, _ = _exact_fields(pts)

    mats = {form: _assemble(op, len(pts), form) for form in ("A", "B", "C")}
    for form, mat in mats.items():
        err = np.abs((mat @ q)[interior] - lap_exact).max()
        assert err < OPERATORS[method]["tol_lap"], f"{method} form {form}: {err:.3e}"
        row_sums = np.abs(np.asarray(mat.sum(axis=1)).ravel()[interior]).max()
        assert row_sums < 1e-10, f"{method} form {form}: row sum {row_sums:.3e}"

    for other in ("B", "C"):
        gap = abs(mats["A"] - mats[other]).max()
        assert gap < 1e-10, f"{method}: closure forms A and {other} differ by {gap:.3e}"


def test_preassembled_matrices_carry_the_closure() -> None:
    """(e) The production matrices, not a re-implementation, hold the closure.

    ``TaylorOperator`` pre-assembles ``_laplacian_matrix`` / ``_gradient_matrices``
    at construction and ``laplacian`` / ``gradient`` are sparse matmuls against
    them. This is the assertion that fails if that assembly drops the closure —
    the re-implemented forms above cannot see a production edit.
    """
    pts, delta, interior = _cloud_1d()
    op = _build("taylor", pts, delta)

    assert op._laplacian_matrix is not None, "TaylorOperator must pre-assemble"
    assert op._gradient_matrices is not None

    lap_rows = np.abs(np.asarray(op._laplacian_matrix.sum(axis=1)).ravel()[interior]).max()
    assert lap_rows < 1e-10, f"pre-assembled Laplacian row sum {lap_rows:.3e} != 0"
    for dim, grad_mat in enumerate(op._gradient_matrices):
        grad_rows = np.abs(np.asarray(grad_mat.sum(axis=1)).ravel()[interior]).max()
        assert grad_rows < 1e-10, f"gradient[{dim}] row sum {grad_rows:.3e} != 0"


def test_sparse_path_and_per_point_fallback_agree() -> None:
    """``laplacian`` has two implementations; they must not have drifted.

    The sparse matmul is the live one and the per-point loop is documented as
    "should not be reached after init", which is exactly the condition under
    which a second implementation rots unobserved.
    """
    pts, delta, interior = _cloud_1d()
    op = _build("taylor", pts, delta)
    q, lap_exact, _ = _exact_fields(pts)

    sparse = op.laplacian(q).copy()
    op._laplacian_matrix = None
    op._gradient_matrices = None
    fallback = op.laplacian(q)

    assert np.abs(sparse - fallback)[interior].max() < 1e-10
    assert np.abs(fallback[interior] - lap_exact).max() < 1e-11


@pytest.mark.parametrize("method", sorted(OPERATORS))
def test_assembling_from_weights_reproduces_the_operator(method: str) -> None:
    """The weights API and the operator API must be the *same* operator.

    This is the assertion a consumer actually depends on: whatever
    ``get_derivative_weights`` returns, closed per the invariant above, must
    equal what ``op.laplacian`` computes. Checked on a non-polynomial field, so
    both sides are inexact — agreeing here means they are the same operator,
    not merely both exact on the test polynomial.
    """
    pts, delta, interior = _cloud_1d()
    op = _build(method, pts, delta)
    u = np.sin(3.0 * pts[:, 0])  # not in the reproduced polynomial space

    assembled = _assemble(op, len(pts), "A") @ u
    direct = op.laplacian(u)
    gap = np.abs(assembled[interior] - direct[interior]).max()
    scale = np.abs(direct[interior]).max()

    assert gap < 1e-9 * max(scale, 1.0), (
        f"{method}: assembling from get_derivative_weights gives a different "
        f"operator than op.laplacian — gap {gap:.3e} against scale {scale:.3e}. "
        f"Consumers assemble from the weights; they must agree (#2081)."
    )
