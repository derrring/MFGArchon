"""The ghost buffer must use the grid's spacing, not fall back to dx = 1.0.

`PreallocatedGhostBuffer` derived its spacing only from `domain_bounds`, and almost no caller
passes those: `pad_array_with_ghosts(..., geometry=None)` left `_grid_spacing = None` and every
consumer read `dx = 1.0`. An inhomogeneous Neumann condition was therefore applied as `g/h`
instead of `g`, so the recovered `du/dn` DIVERGED as `1/h` -- measured through the solver's own
gradient path at a requested 2.0:

    Nx = 21 / 41 / 81   ->   11.83 / 23.71 / 47.44

Robin read the same fallback, in the denominator `alpha + beta/dx`.

Found by review of #1902 (which is blocked on #1904, the node-centring half of the same function).
`HJBFDMSolver.honors_inhomogeneous_neumann` is True and pinned, so the declaration was live and
wrong -- the existing applicator test builds its buffer WITH `domain_bounds`, validating the
instrument on a path the solver never takes, which is why this passed.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.alg.numerical.hjb_solvers.base_hjb import _compute_gradient_array_1d, _compute_laplacian_1d
from mfgarchon.geometry.boundary import neumann_bc, no_flux_bc, robin_bc
from mfgarchon.geometry.boundary.applicator_fdm import pad_array_with_ghosts


@pytest.mark.parametrize("nx", [21, 41, 81, 161])
def test_inhomogeneous_neumann_recovers_the_requested_flux_at_every_resolution(nx: int):
    """The law: what the caller asked for is what the ghost encodes, independent of h."""
    g = 2.0
    dx = 1.0 / (nx - 1)
    u = 0.5 * np.cos(2 * np.pi * np.linspace(0.0, 1.0, nx))
    padded = pad_array_with_ghosts(u, neumann_bc(dimension=1, value=g), ghost_depth=1, time=0.0, spacing=dx)

    # Outward normal is -x at the low wall, +x at the high wall.
    low = -(padded[1] - padded[0]) / dx
    high = (padded[-1] - padded[-2]) / dx
    assert low == pytest.approx(g, abs=1e-12), f"nx={nx}: du/dn = {low} at the low wall, requested {g}"
    assert high == pytest.approx(g, abs=1e-12), f"nx={nx}: du/dn = {high} at the high wall, requested {g}"


def test_the_fallback_would_diverge_which_is_why_the_spacing_is_threaded():
    """Control: without spacing the ghost is spacing-independent, so the flux scales as 1/h.

    Asserts the DEFECT is still reachable through the un-threaded call, so the test above is
    measuring the threading rather than something that was always true.
    """
    u = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    bc = neumann_bc(dimension=1, value=2.0)
    without = pad_array_with_ghosts(u, bc, ghost_depth=1, time=0.0)
    with_small = pad_array_with_ghosts(u, bc, ghost_depth=1, time=0.0, spacing=0.05)
    assert without[0] == pytest.approx(3.0), "the un-threaded fallback is no longer dx = 1.0"
    assert with_small[0] == pytest.approx(1.1), "the threaded call does not use the spacing given"
    assert without[0] != with_small[0], "spacing makes no difference; the threading is inert"


@pytest.mark.parametrize("scalar_or_sequence", [0.05, [0.05], (0.05,)])
def test_spacing_accepts_a_scalar_or_one_value_per_axis(scalar_or_sequence):
    u = np.array([1.0, 2.0, 3.0])
    bc = neumann_bc(dimension=1, value=1.0)
    padded = pad_array_with_ghosts(u, bc, ghost_depth=1, time=0.0, spacing=scalar_or_sequence)
    assert padded[0] == pytest.approx(1.05)


def test_a_wrong_length_spacing_raises_rather_than_broadcasting_silently():
    """Fail loud: a 2-entry spacing for a 1-D array is a caller bug, not something to guess at."""
    u = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="one per axis"):
        pad_array_with_ghosts(u, neumann_bc(dimension=1, value=1.0), ghost_depth=1, time=0.0, spacing=[0.1, 0.2])


def test_zero_flux_is_unaffected_because_it_multiplies_the_spacing_by_zero():
    """Scope control: no-flux must be byte-identical, or this change is wider than claimed.

    The node-centring defect in the same function (#1904) is NOT addressed here and this test
    would not see it -- it compares the two spellings against each other, not against the truth.
    """
    u = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    for bc in (no_flux_bc(dimension=1), neumann_bc(dimension=1, value=0.0)):
        a = pad_array_with_ghosts(u, bc, ghost_depth=1, time=0.0)
        b = pad_array_with_ghosts(u, bc, ghost_depth=1, time=0.0, spacing=0.05)
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("nx", [21, 81])
def test_the_solver_gradient_path_now_carries_the_spacing(nx: int):
    """End-to-end through `_compute_gradient_array_1d`, which is what the residual calls."""
    dx = 1.0 / (nx - 1)
    u = 0.5 * np.cos(2 * np.pi * np.linspace(0.0, 1.0, nx))
    tight = _compute_gradient_array_1d(u, dx, bc=neumann_bc(dimension=1, value=2.0), upwind=False, time=0.0)
    loose = _compute_gradient_array_1d(u, dx, bc=neumann_bc(dimension=1, value=0.0), upwind=False, time=0.0)
    # A non-zero requested flux must change the wall rows, and by an amount that does not blow up.
    assert abs(tight[0] - loose[0]) < 5.0, f"nx={nx}: wall gradient moved by {abs(tight[0] - loose[0]):.3e}"
    assert abs(tight[0] - loose[0]) > 1e-9, "the requested flux reached nothing"


def test_robin_reads_the_same_spacing():
    """Robin used the fallback too, in the denominator `alpha + beta/dx`."""
    u = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    bc = robin_bc(dimension=1, alpha=1.0, beta=1.0, value=0.5)
    coarse = pad_array_with_ghosts(u, bc, ghost_depth=1, time=0.0, spacing=0.5)
    fine = pad_array_with_ghosts(u, bc, ghost_depth=1, time=0.0, spacing=0.01)
    assert coarse[0] != fine[0], "Robin ghosts are spacing-independent; the threading did not reach it"


def test_the_laplacian_path_threads_its_own_spacings_parameter():
    """`laplacian_with_bc` already took `spacings` and dropped it at the padding call."""
    u = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    bc = neumann_bc(dimension=1, value=3.0)
    coarse = _compute_laplacian_1d(u, 0.5, bc=bc, time=0.0)
    fine = _compute_laplacian_1d(u, 0.01, bc=bc, time=0.0)
    # The wall row must respond to the spacing; interior rows scale as 1/h^2 either way.
    assert not np.allclose(coarse[0] * 0.5**2, fine[0] * 0.01**2), (
        "the wall Laplacian is spacing-independent, so the flux term is still applied as g/h"
    )
