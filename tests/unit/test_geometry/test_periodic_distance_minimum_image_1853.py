"""Issue #1853: compute_periodic_distance must use the true minimum image.

Both ``Hyperrectangle`` and ``TensorProductGrid`` implemented the rule inline as
``min(|d|, L - |d|)``, which is correct only for ``|d| <= L``. On a unit torus they
agreed with each other and disagreed with the truth: 0.9 for a separation of 1.9
(true 0.1), and 6.7 for 7.7 -- larger than the domain. Both now delegate to
``wrap_displacement``, the single owner of the rule (#1841, #1847).

What pins what, deliberately:

- ``REFERENCE`` holds output captured by **executing the pre-change implementations**,
  for ``|d| <= L`` only -- the regime the fix must not disturb. Float noise is part of
  the literal (0.30000000000000004, not 0.3).
- ``test_large_displacement_*`` carries the values that must CHANGE, against the
  closed form ``|d - L*round(d/L)|``.
- Agreement *between* the two classes is NOT used as a pin. Both now route through one
  owner, so agreement is tautological and would pass over a broken owner.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc, periodic_bc
from mfgarchon.geometry.implicit.hyperrectangle import Hyperrectangle

L = 1.0

# (name, delta, distance) -- captured from the pre-change implementations, |d| <= L.
REFERENCE = [
    ("zero", 0.0, 0.0),
    ("small", 0.1, 0.1),
    ("quarter", 0.25, 0.25),
    ("half", 0.5, 0.5),
    ("over_half", 0.7, 0.30000000000000004),
    ("near_L", 0.9, 0.09999999999999998),
    ("exact_L", 1.0, 0.0),
    ("negative", -0.3, 0.3),
]


def _periodic_hyperrectangle() -> Hyperrectangle:
    return Hyperrectangle(bounds=np.array([[0.0, L]]), periodic_dims=(0,))


def _periodic_grid() -> TensorProductGrid:
    return TensorProductGrid(bounds=[(0.0, L)], Nx_points=[11], boundary_conditions=periodic_bc(dimension=1))


GEOMETRIES = [
    pytest.param(_periodic_hyperrectangle, id="hyperrectangle"),
    pytest.param(_periodic_grid, id="tensor_grid"),
]


@pytest.mark.parametrize("factory", GEOMETRIES)
@pytest.mark.parametrize(("name", "delta", "expected"), REFERENCE, ids=[c[0] for c in REFERENCE])
def test_short_displacement_is_byte_identical_to_pre_change(factory, name, delta, expected):
    """|d| <= L was already correct; the consolidation must not move it by one ULP."""
    geom = factory()
    got = geom.compute_periodic_distance(np.array([delta]), np.array([0.0]))
    assert float(got) == expected, f"{name}: {got!r} != captured {expected!r}"


@pytest.mark.parametrize("factory", GEOMETRIES)
@pytest.mark.parametrize(
    ("delta", "expected"),
    [(1.5, 0.5), (1.9, 0.1), (2.5, 0.5), (7.7, 0.3), (-1.9, 0.1), (-7.7, 0.3), (100.4, 0.4)],
)
def test_large_displacement_uses_the_minimum_image(factory, delta, expected):
    """The values #1853 reported wrong. Old rule gave 0.9 at 1.9 and 6.7 at 7.7."""
    geom = factory()
    got = geom.compute_periodic_distance(np.array([delta]), np.array([0.0]))
    assert float(got) == pytest.approx(expected, abs=1e-12), f"delta={delta}: got {float(got)}"


@pytest.mark.parametrize("factory", GEOMETRIES)
def test_distance_never_exceeds_half_the_period(factory):
    """Structural invariant the old rule violated, and the reason this was a bug.

    On a circle of circumference L no two points are more than L/2 apart. The old rule
    returned values up to and beyond L. Swept across many periods so the failure cannot
    hide between sample points.
    """
    geom = factory()
    deltas = np.linspace(-10 * L, 10 * L, 977)
    got = np.array([float(geom.compute_periodic_distance(np.array([d]), np.array([0.0]))) for d in deltas])
    assert np.all(got <= L / 2 + 1e-12), f"max {got.max()} exceeds L/2 = {L / 2}"
    # Measured: 830 of these 977 points violate the bound under the old rule (max 9.0).
    # A companion `got >= 0` assertion was dropped -- it cannot fail after a norm(), so it
    # was unfalsifiable rather than weak.


@pytest.mark.parametrize("factory", GEOMETRIES)
def test_matches_closed_form_minimum_image(factory):
    """Closed form |d - L*round(d/L)|, evaluated here rather than fetched from the library.

    Not fully independent, and deliberately labelled as such: it is the same closed form the
    owner implements, transcribed. That makes it a strong pin against the OLD rule and a weak
    one against a rounding or sign error shared with the transcription -- which is what the
    invariant test above, and the period sweep below, are for.
    """
    geom = factory()
    deltas = np.linspace(-8.3 * L, 8.3 * L, 401)
    for d in deltas:
        expected = abs(d - L * np.round(d / L))
        got = float(geom.compute_periodic_distance(np.array([d]), np.array([0.0])))
        assert got == pytest.approx(expected, abs=1e-12), f"delta={d}"


def test_periodicity_of_the_distance_function():
    """d(x, y) is invariant under shifting either point by a whole period."""
    geom = _periodic_hyperrectangle()
    base = float(geom.compute_periodic_distance(np.array([0.3]), np.array([0.0])))
    for k in (-3, -1, 1, 2, 5):
        shifted = float(geom.compute_periodic_distance(np.array([0.3 + k * L]), np.array([0.0])))
        assert shifted == pytest.approx(base, abs=1e-12), f"k={k}: {shifted} != {base}"


def test_non_periodic_axis_is_plain_euclidean():
    """The empty-periods path must return the ordinary distance, unwrapped."""
    geom = Hyperrectangle(bounds=np.array([[0.0, L]]), periodic_dims=())
    batched = geom.compute_periodic_distance(np.array([[0.9]]), np.array([[0.1]]))
    assert batched.shape == (1,)
    assert batched[0] == 0.8
    # 1-D input keeps returning a scalar, not a length-1 array (captured pre-change).
    scalar = geom.compute_periodic_distance(np.array([0.9]), np.array([0.1]))
    assert scalar.ndim == 0
    assert float(scalar) == 0.8


def test_non_periodic_grid_does_not_wrap():
    """A no-flux grid has no periodic axes: 1.9 apart stays 1.9, it is not a torus."""
    grid = TensorProductGrid(bounds=[(0.0, L)], Nx_points=[11], boundary_conditions=no_flux_bc(dimension=1))
    got = grid.compute_periodic_distance(np.array([1.9]), np.array([0.0]))
    assert float(got) == pytest.approx(1.9, abs=1e-12)


def test_mixed_periodic_and_bounded_axes():
    """Only the periodic axis wraps; the bounded one contributes its raw separation."""
    geom = Hyperrectangle(bounds=np.array([[0.0, L], [0.0, 2.0]]), periodic_dims=(0,))
    # axis 0: 1.9 -> 0.1 (wraps).  axis 1: 0.4 -> 0.4 (does not).
    got = float(geom.compute_periodic_distance(np.array([1.9, 0.5]), np.array([0.0, 0.1])))
    assert got == pytest.approx(np.hypot(0.1, 0.4), abs=1e-12)


@pytest.mark.parametrize("period", [0.3, 1.0, 2.0, 2 * np.pi, 10.5])
@pytest.mark.parametrize("cls", ["hyperrectangle", "tensor_grid"])
def test_rule_is_period_sensitive(cls, period):
    """The formula scales with L, so the pin must vary L -- every other case here uses L = 1.

    Without this, hardcoding ``length = 1.0`` inside ``wrap_displacement`` passes this whole
    file (measured: 40/40). The defect is period-dependence, and a period-blind suite cannot
    see it.
    """
    if cls == "hyperrectangle":
        geom = Hyperrectangle(bounds=np.array([[0.0, period]]), periodic_dims=(0,))
    else:
        geom = TensorProductGrid(bounds=[(0.0, period)], Nx_points=[11], boundary_conditions=periodic_bc(dimension=1))
    for factor in (0.1, 0.5, 0.9, 1.4, 2.6, 7.3, -3.2):
        delta = factor * period
        expected = abs(delta - period * np.round(delta / period))
        got = float(geom.compute_periodic_distance(np.array([delta]), np.array([0.0])))
        assert got == pytest.approx(expected, abs=1e-12 * max(1.0, period)), (
            f"L={period}, delta={delta}: got {got}, expected {expected}"
        )
        assert got <= period / 2 + 1e-12 * max(1.0, period)


@pytest.mark.parametrize("factory", GEOMETRIES)
def test_batch_shape_and_values(factory):
    """Batched input returns one distance per row, mixing wrapping and non-wrapping rows."""
    geom = factory()
    p1 = np.array([[0.1], [0.6], [1.9], [7.7]])
    p2 = np.zeros((4, 1))
    got = geom.compute_periodic_distance(p1, p2)
    assert got.shape == (4,)
    np.testing.assert_allclose(got, [0.1, 0.4, 0.1, 0.3], atol=1e-12)


def test_out_of_contract_shapes_behave_as_before_1853():
    """Shapes outside the documented (N, d) / (d,) contract are unchanged by #1853.

    Before #1853 the no-periodic-dims case returned early via ``norm(..., axis=-1)`` without
    reshaping, so a higher-rank stack and a ``(d,)``-against-``(N, d)`` broadcast both worked.
    The consolidation removed that early return; reducing on the last axis and reshaping only a
    1-D *result* keeps both. Nothing in the library calls these shapes -- this exists so the
    source comment asserting it is pinned rather than merely asserted.
    """
    geom = Hyperrectangle(bounds=np.array([[0.0, L], [0.0, L]]), periodic_dims=())
    rng = np.random.default_rng(0)

    stacked1, stacked2 = rng.random((3, 5, 2)), rng.random((3, 5, 2))
    got = geom.compute_periodic_distance(stacked1, stacked2)
    assert got.shape == (3, 5)
    np.testing.assert_allclose(got, np.linalg.norm(stacked1 - stacked2, axis=-1), atol=0)

    one, many = rng.random(2), rng.random((5, 2))
    broadcast = geom.compute_periodic_distance(one, many)
    assert broadcast.shape == (5,)
    np.testing.assert_allclose(broadcast, np.linalg.norm(one - many, axis=-1), atol=0)
