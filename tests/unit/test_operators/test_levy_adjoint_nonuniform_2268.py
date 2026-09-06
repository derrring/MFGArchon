"""The Lévy operator's adjoint is W-weighted, and only a non-uniform grid can see it.

`levy_integro_diff.py` states a design constraint in live code:

    - Adjoint: J* = W^{-1} J^T W where W = grid integration weights
    - Tests MUST include non-uniform grid adjoint consistency

The test satisfying it was deleted by `18d8cc80` (#2227), leaving the constraint asserting
something untrue. This restores the binding half only -- not the deleted file, whose other
cases were happy-path assertions the #2227 admission bar removed on purpose.

Admission class: **labelled defect pin** (AGENTS.md class 3), not an external oracle.

I first admitted this as class 2 and that was wrong. The identity under test is

    <J[v], m>_W = <v, J*[m]>_W,    <f, g>_W = sum_i W_i f_i g_i

and `apply_adjoint` reaches `J^T` through `as_sparse()`, which is assembled column by column
from `_matvec`. So the identity reduces to `sum_i v_i (J^T W m)_i == sum_j W_j m_j (Jv)_j`,
which is **true for every linear J and every positive W**. Nothing external is computed, and
this file cannot see a wrong J at all. Measured, by mutating the operator itself:

    scale the whole operator by 3          -> 5 passed
    flip the compensator sign (-= to +=)   -> 5 passed
    drop the -v(x) term                    -> 5 passed
    drop the Levy density from the weight  -> 5 passed

What it does pin is `apply_adjoint` against `_matvec`: dropping the `W^{-1}...W` conjugation,
either half of it, or assuming a constant W each fail 3 of 5. That is the defect #2268 names
and the reason this file earns its place -- but the claim to be verifying the operator's
mathematics was false, and would have kept passing while J was arbitrarily wrong.

Retirement condition: if `as_sparse()` ever stops being derived from `_matvec`, this file
becomes a genuine cross-implementation oracle and this note should be rewritten, not deleted.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.operators.integro_diff.levy_integro_diff import LevyIntegroDiffOperator
from mfgarchon.operators.integro_diff.levy_measures import GaussianJumps


def _nonuniform_grid() -> np.ndarray:
    """Dense near 0, coarse near 2*pi. Weight ratio must be far from 1."""
    return np.sort(np.unique(np.concatenate([np.linspace(0.0, 1.0, 30), np.linspace(1.0, 2 * np.pi, 71)])))


def _uniform_grid() -> np.ndarray:
    return np.linspace(0.0, 2 * np.pi, 101)


def _operator(grid: np.ndarray) -> LevyIntegroDiffOperator:
    return LevyIntegroDiffOperator(
        grid_points=grid,
        levy_measure=GaussianJumps(mu=0.0, sigma=0.3, truncate_at=3.0),
        intensity=1.0,
        compensate=True,
    )


def _fields(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two fields that are not proportional.

    They are also asymmetric under x -> 2*pi - x, but that is incidental and NOT what makes
    the test work: review measured a genuinely symmetric pair separating `J^T` from
    `W^{-1} J^T W` at rel err 1.08, better than the 1.00 this pair gives. An earlier version
    of this docstring claimed the asymmetry was load-bearing. It is not.
    """
    v = np.sin(grid) + 0.3 * np.cos(3.0 * grid)
    m = np.exp(-2.0 * (grid - 1.0) ** 2) + 0.1
    return v, m


def _pairing(w: np.ndarray, f: np.ndarray, g: np.ndarray) -> float:
    return float(np.sum(w * f * g))


class TestLevyAdjointIsWeighted:
    def test_duality_holds_on_a_nonuniform_grid(self):
        """<J[v], m>_W == <v, J*[m]>_W -- the constraint the source file states."""
        grid = _nonuniform_grid()
        op = _operator(grid)
        w = op.integration_weights
        v, m = _fields(grid)

        lhs = _pairing(w, (op @ v), m)
        rhs = _pairing(w, v, op.apply_adjoint(m))

        scale = max(abs(lhs), abs(rhs), 1e-30)
        assert abs(lhs - rhs) / scale < 1e-10, (
            f"L2(W) duality violated on a non-uniform grid: <J v, m>_W={lhs!r} vs <v, J* m>_W={rhs!r}"
        )

    def test_the_nonuniform_grid_varies_the_INTERIOR_weights(self):
        """Guards the fixture, on the quantity that actually discriminates.

        The whole-array weight ratio is the wrong guard: trapezoidal weights give the two
        endpoints half weight on *any* grid, so even `np.linspace` scores 2.0 and the check
        would pass on a grid with a perfectly uniform interior. What a non-uniform grid buys
        is interior variation, and that is what `W^{-1} J^T W` differs from `J^T` on.
        """
        interior_ratio = {}
        for name, grid in (("uniform", _uniform_grid()), ("nonuniform", _nonuniform_grid())):
            w = _operator(grid).integration_weights[1:-1]
            interior_ratio[name] = w.max() / w.min()

        assert interior_ratio["uniform"] == pytest.approx(1.0, abs=1e-12), (
            f"a linspace grid should have constant interior weights, got {interior_ratio['uniform']!r}"
        )
        assert interior_ratio["nonuniform"] > 2.0, (
            f"fixture no longer discriminates: interior weight ratio {interior_ratio['nonuniform']:.3f} "
            "is near 1, so this file stops testing the constraint it exists for"
        )

    def test_dropping_the_weights_breaks_the_duality(self):
        """Discrimination: the W-free transpose must be separable from the real adjoint.

        `J^T` without the `W^{-1}...W` conjugation is the natural wrong implementation, and it
        fails here by order 1 -- so this file can see the defect it exists to catch.

        Two things I checked and will not claim, because both are false and both look
        plausible. It is NOT the case that `J^T` is correct on a uniform grid: trapezoidal
        weights give the endpoints half weight, so `W` is not proportional to the identity
        even for `np.linspace`. Nor is it rescued by pairing only over interior indices --
        `(W^{-1} J^T W m)_i` sums over every `j`, so the endpoint weights reach interior rows
        too. The uniform-grid error is recorded below rather than asserted to vanish.

        What the non-uniform grid buys is therefore not "the only case that fails" but the
        only case where `W` varies in the INTERIOR, which is the part an implementation
        assuming constant spacing would get wrong while still handling endpoints.
        """

        def rel_err(grid):
            op = _operator(grid)
            w = op.integration_weights
            v, m = _fields(grid)
            lhs = _pairing(w, (op @ v), m)
            rhs = _pairing(w, v, op.as_sparse().T @ m)  # J^T m, no W^{-1}...W
            return abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-30)

        nonuniform = rel_err(_nonuniform_grid())
        uniform = rel_err(_uniform_grid())

        assert nonuniform > 1e-3, (
            f"the W-free transpose is indistinguishable on this fixture (rel err {nonuniform:.3e}); "
            "the test cannot see the defect it exists to catch"
        )
        # Recorded, not a contrast: the endpoint half-weights alone already break it.
        assert uniform > 1e-3, (
            f"unexpected: J^T agreed with the W-weighted adjoint on a uniform grid "
            f"(rel err {uniform:.3e}). If trapezoidal endpoint weighting changed, the reasoning "
            "in this docstring needs re-deriving."
        )


@pytest.mark.parametrize("grid_fn", [_uniform_grid, _nonuniform_grid])
def test_adjoint_preserves_the_weighted_mass_pairing(grid_fn):
    """<J[v], 1>_W == <v, J*[1]>_W: the constant field, on both grid kinds."""
    grid = grid_fn()
    op = _operator(grid)
    w = op.integration_weights
    v, _ = _fields(grid)
    ones = np.ones_like(grid)

    lhs = _pairing(w, (op @ v), ones)
    rhs = _pairing(w, v, op.apply_adjoint(ones))
    scale = max(abs(lhs), abs(rhs), 1e-30)
    assert abs(lhs - rhs) / scale < 1e-10
