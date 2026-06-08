"""Single-source boundary-detection tolerances (Issue #1101).

These values were previously scattered as magic literals across the boundary / geometry
on-wall classifiers. They are intentionally **not** a single number: the "is this point on the
wall" judgment has genuinely different scales depending on how membership is measured.

- ``BOUNDARY_TOL`` (1e-6): bounds-comparison on-wall band for scattered / collocation geometries,
  where boundary points are *placed* at ~1e-6 of the wall (GFDM). Used with a closed ``<=``
  comparison; this is the value the 2D scattered-cloud GFDM path uses.
- ``ONWALL_TOL`` (1e-10): tight exact-membership band for analytic / grid geometries, where a
  point is "on the wall" only at ~machine precision for O(1) coordinates (strict ``<``).
- ``SDF_BOUNDARY_TOL`` (1e-8): signed-distance band (``|phi(x)| < tol``), and the
  ``conditions.py`` bounds-comparison variant that shares this scale.
- ``BOUNDARY_REL_TOL`` (1e-12): relative-coordinate augmentation factor added to an absolute
  band to absorb floating-point subtraction error at large coordinates
  (``tol + |bound| * BOUNDARY_REL_TOL``).

Each is a **tunable default**: pass an explicit ``tolerance=`` to override per call, or retune
globally here. The values are unchanged from the prior scattered literals (byte-identical
single-sourcing); they do NOT collapse the distinct on-wall scales into one (that would loosen
analytic-geometry detection by four orders of magnitude — see Issue #1101 discussion).
"""

from __future__ import annotations

BOUNDARY_TOL: float = 1e-6
ONWALL_TOL: float = 1e-10
SDF_BOUNDARY_TOL: float = 1e-8
BOUNDARY_REL_TOL: float = 1e-12

__all__ = ["BOUNDARY_TOL", "ONWALL_TOL", "SDF_BOUNDARY_TOL", "BOUNDARY_REL_TOL"]
