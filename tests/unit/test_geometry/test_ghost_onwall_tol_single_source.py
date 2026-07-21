"""Pinning tests for Issue #1101 tail: ghost.py on-wall tolerance single-sourcing.

Two migration sites in geometry/boundary/ghost.py:
  1. compute_normal_from_bounds default tol=1e-10  →  ONWALL_TOL
  2. create_ghost_stencil interior-side mask -1e-10  →  -ONWALL_TOL

Tests assert:
  (A) ghost module imports from tolerances (structural: import present)
  (B) compute_normal_from_bounds default matches ONWALL_TOL exactly (VALUE only)
  (C) create_ghost_stencil classification is unchanged on a known 2-D cloud
      (behavior-invariant: byte-identical result before/after)
  source-grep (:154/:170) the actual re-fork guard -- the module source of each site
      must not contain the bare literal '1e-10'.

What catches a revert (Issue #1569): the import-vs-literal re-fork is caught by the
source-grep tests (``test_no_hardcoded_1e10_in_*``, :154/:170), NOT by (B). A signature
default is the *evaluated* value, so ``tol=1e-10`` and ``tol=ONWALL_TOL`` are both ``1e-10``
to introspection -- (B) is a value equality that a bare-literal revert PASSES. (B) exists
only to pin that ONWALL_TOL keeps that float value (a change to the constant would surface),
and (A) that the import exists; the grep is the discriminating single-source guard.
"""

from __future__ import annotations

import importlib
import inspect
import textwrap

import numpy as np

from mfgarchon.geometry.boundary.tolerances import ONWALL_TOL

# ---------------------------------------------------------------------------
# (A) Structural: ghost module must import from tolerances
# ---------------------------------------------------------------------------


def test_ghost_imports_onwall_tol():
    """ghost.py must import ONWALL_TOL (or tolerances) — no magic literal in source."""
    import mfgarchon.geometry.boundary.ghost as ghost_mod

    # Reload to get fresh source (not cached bytecode)
    importlib.reload(ghost_mod)

    source = inspect.getsource(ghost_mod)
    assert "from mfgarchon.geometry.boundary.tolerances import" in source or ("from .tolerances import" in source), (
        "ghost.py does not import from tolerances — hardcoded literal not yet replaced.\n"
        f"First 1000 chars of ghost.py imports section:\n{source[:1000]}"
    )


# ---------------------------------------------------------------------------
# (B) Default parameter of compute_normal_from_bounds must equal ONWALL_TOL
# ---------------------------------------------------------------------------


def test_compute_normal_from_bounds_default_tol_is_onwall_tol():
    """The default tol= argument of compute_normal_from_bounds must equal ONWALL_TOL by VALUE.

    Scope (Issue #1569): a signature default is the evaluated value, so this equality does NOT
    distinguish ``tol=1e-10`` from ``tol=ONWALL_TOL`` (both introspect as 1e-10) -- a bare-literal
    revert PASSES here. The import-vs-literal re-fork is caught by ``test_no_hardcoded_1e10_in_
    compute_normal_from_bounds`` (source grep). This test pins only that ONWALL_TOL keeps that
    float value, so a change to the constant surfaces as a mismatch.
    """
    from mfgarchon.geometry.boundary.ghost import compute_normal_from_bounds

    sig = inspect.signature(compute_normal_from_bounds)
    default_tol = sig.parameters["tol"].default

    assert default_tol == ONWALL_TOL, (
        f"compute_normal_from_bounds default tol={default_tol!r} != ONWALL_TOL={ONWALL_TOL!r}. "
        "The default's float value has drifted from the single source (ONWALL_TOL)."
    )


# ---------------------------------------------------------------------------
# (C) Behavior-invariant: create_ghost_stencil classifications unchanged
# ---------------------------------------------------------------------------


def test_create_ghost_stencil_interior_mask_unchanged():
    """Interior-side mask in create_ghost_stencil must produce the same result
    regardless of whether -1e-10 or -ONWALL_TOL is used (same float value).

    This test verifies the classification is numerically correct on a known 2-D
    cloud: 2 interior neighbors, 2 on-boundary neighbors (normal component = 0),
    and 1 ghost from a prior call (exact negative offset).
    """
    from mfgarchon.geometry.boundary.ghost import create_ghost_stencil

    boundary_pt = np.array([0.0, 0.5])
    normal = np.array([-1.0, 0.0])  # Left wall: outward = -x

    # Interior neighbors: positive x → component (x-0)*(-1) = -x < 0 → interior
    # Boundary neighbors: x=0 → component = 0 → NOT interior (>= -ONWALL_TOL)
    neighbors = np.array(
        [
            [0.1, 0.4],  # interior: offset·n = -0.1 < 0
            [0.1, 0.6],  # interior: offset·n = -0.1 < 0
            [0.0, 0.4],  # boundary: offset·n = 0.0 → NOT interior
            [0.0, 0.6],  # boundary: offset·n = 0.0 → NOT interior
        ]
    )

    ghosts, _augmented, interior_mask = create_ghost_stencil(boundary_pt, neighbors, normal, return_interior_mask=True)

    # 2 interior neighbors should be identified
    assert int(np.sum(interior_mask)) == 2, (
        f"Expected 2 interior neighbors, got {int(np.sum(interior_mask))}. Interior mask: {interior_mask}"
    )

    # 2 ghost points should be created (reflections of the 2 interior neighbors)
    assert len(ghosts) == 2, f"Expected 2 ghost points, got {len(ghosts)}"

    # Ghost points must be reflections across x=0 wall
    expected_ghosts = np.array([[-0.1, 0.4], [-0.1, 0.6]])
    np.testing.assert_allclose(
        ghosts, expected_ghosts, atol=1e-14, err_msg="Ghost point positions differ from expected reflections"
    )


def test_compute_normal_from_bounds_behavior_unchanged():
    """compute_normal_from_bounds must return correct outward normals on a 2-D box.

    Checks exact on-wall (tol=ONWALL_TOL) and just-inside (tol→0) detection.
    """
    from mfgarchon.geometry.boundary.ghost import compute_normal_from_bounds

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])

    # Left boundary: outward normal = [-1, 0]
    n_left = compute_normal_from_bounds(np.array([0.0, 0.5]), bounds)
    np.testing.assert_allclose(n_left, [-1.0, 0.0], atol=1e-14)

    # Right boundary: outward normal = [+1, 0]
    n_right = compute_normal_from_bounds(np.array([1.0, 0.5]), bounds)
    np.testing.assert_allclose(n_right, [1.0, 0.0], atol=1e-14)

    # Bottom: outward normal = [0, -1]
    n_bot = compute_normal_from_bounds(np.array([0.5, 0.0]), bounds)
    np.testing.assert_allclose(n_bot, [0.0, -1.0], atol=1e-14)

    # Corner (0, 0): outward normal = [-1, -1] / sqrt(2)
    n_corner = compute_normal_from_bounds(np.array([0.0, 0.0]), bounds)
    expected_corner = np.array([-1.0, -1.0]) / np.sqrt(2)
    np.testing.assert_allclose(n_corner, expected_corner, atol=1e-14)

    # Interior point: no boundary face matched → zero vector (not normalised)
    n_interior = compute_normal_from_bounds(np.array([0.5, 0.5]), bounds)
    np.testing.assert_allclose(n_interior, [0.0, 0.0], atol=1e-14)


# ---------------------------------------------------------------------------
# (D) Source-level check: no bare 1e-10 literal in compute_normal_from_bounds
# ---------------------------------------------------------------------------


def test_no_hardcoded_1e10_in_compute_normal_from_bounds():
    """The source of compute_normal_from_bounds must not contain the bare literal '1e-10'.

    This is the negative-evidence complement to test_ghost_imports_onwall_tol.
    It directly fails if the default-parameter literal was not replaced.
    """
    from mfgarchon.geometry.boundary.ghost import compute_normal_from_bounds

    src = inspect.getsource(compute_normal_from_bounds)
    assert "1e-10" not in src, (
        "compute_normal_from_bounds still contains the hardcoded literal '1e-10'.\n"
        "Replace it with ONWALL_TOL imported from geometry/boundary/tolerances.py.\n"
        f"Current source:\n{textwrap.indent(src, '  ')}"
    )


def test_no_hardcoded_1e10_in_create_ghost_stencil():
    """The source of create_ghost_stencil must not contain the bare literal '1e-10'.

    Verifies the interior-mask line was migrated.
    """
    from mfgarchon.geometry.boundary.ghost import create_ghost_stencil

    src = inspect.getsource(create_ghost_stencil)
    assert "1e-10" not in src, (
        "create_ghost_stencil still contains the hardcoded literal '1e-10'.\n"
        "Replace '-1e-10' with '-ONWALL_TOL' imported from geometry/boundary/tolerances.py.\n"
        f"Current source:\n{textwrap.indent(src, '  ')}"
    )
