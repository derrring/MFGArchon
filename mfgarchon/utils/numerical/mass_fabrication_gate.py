"""One owner for "would clipping this density fabricate mass?" (Issue #1683).

Nine FP solve paths clip a negative density to zero, and several then renormalise to the
pre-step total. That combination makes a diverging solve **indistinguishable from a
healthy one**: the returned array is finite, non-negative, and exactly mass-conserving,
so every cheap invariant a caller might check is satisfied by the repair rather than by
the physics. Measured on the #1507 configuration, the clip discarded 8.39% of the mass
and the renormalised output still reported exact conservation.

## What is tested, and what is not

The invariant is **the mass the clip would create, relative to the mass present**:

    fabricated = |sum of negatives| / sum of positives

Not `min(density)`, not the returned total, not whether a warning was emitted. Those are
the three quantities the repair itself fixes. Issue #1671 is the case in point: total mass
grew 1.0 -> 6378.77 while `min(M)` read +0.037 -- non-negative, finite, and entirely
wrong.

Round-off gives ~1e-15. A scheme that has genuinely failed gives O(1). The default
threshold sits between them.

**That is not universal, and one caller has already falsified it.** Measured on 144
configurations of the network graph scheme, the fabricated fraction forms a continuous ladder
from 1e-9 to O(1), tracking the true final drift monotonically the whole way -- so the default
rejected nine solves whose honest answer was a drift between 2e-7 and 5.8e-5. That scheme
passes its own `threshold=`, justified by a measured 2.1-order gap between its discretisation
noise and its failures (`fp_network.py`, `_MAX_NETWORK_CLIP_FABRICATION`).

So: the default is a default, not a law. Before adopting it at a new site, measure where that
site's honest solves sit. What is single-sourced here is the **invariant** -- how fabricated
mass is defined and compared. A tolerance is not an invariant; it belongs to the scheme whose
discretisation error it has to clear, the same way a solver tolerance does.

## Why weights are optional, and when they are not

For a **uniform** quadrature the cell measure cancels in the ratio, so the same numbers
come out whether or not you multiply by dV -- and passing them is wasted work. For a
non-uniform one (GFDM quadrature, a weak-form mass matrix, graph node masses) it does
**not** cancel, and omitting the weights silently measures a different quantity than the
solver's own mass functional. Hence `weights=None` means "uniform, and I have checked
that it is", not "I did not think about it".

## Not for input validation

A non-finite or negative *initial* density is a caller error and belongs in a
construction-time check with its own message. This gate is for repair during
time-stepping, and conflating the two produces an error that blames a timestep for
something the caller supplied.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["MAX_CLIP_MASS_FABRICATION", "clip_nonnegative_or_raise", "mass_fabricated_by_clip"]

# The largest fraction of the present mass a non-negativity clip may create before the
# solve is stopped. Round-off is ~1e-15; a failed scheme is O(1) (Issue #1671).
MAX_CLIP_MASS_FABRICATION = 1e-8


def mass_fabricated_by_clip(
    density: NDArray[np.floating],
    *,
    weights: NDArray[np.floating] | None = None,
) -> float:
    """Fraction of the present mass that clipping ``density`` to zero would create.

    Returns 0.0 when nothing is negative, and ``inf`` when there is no positive mass to
    measure against -- a density that is entirely non-positive is not a small error.
    """
    negatives = density < 0
    if not negatives.any():
        return 0.0
    if weights is None:
        negative_mass = -float(density[negatives].sum())
        positive_mass = float(density[density > 0].sum())
    else:
        w = np.asarray(weights)
        if w.shape != density.shape:
            raise ValueError(f"weights shape {w.shape} does not match density shape {density.shape}")
        negative_mass = -float((density * w)[negatives].sum())
        positive_mass = float((density * w)[density > 0].sum())
    if positive_mass <= 0:
        return float("inf")
    return negative_mass / positive_mass


def clip_nonnegative_or_raise(
    density: NDArray[np.floating],
    *,
    context: str,
    remedy: str,
    weights: NDArray[np.floating] | None = None,
    threshold: float = MAX_CLIP_MASS_FABRICATION,
) -> NDArray[np.floating]:
    """Clip round-off negatives to zero, or stop the solve if the clip would matter.

    Args:
        density: the array about to be clipped.
        context: where this happened, in the caller's own terms -- scheme, timestep,
            whichever coordinates make the message actionable. It is quoted verbatim.
        remedy: what the caller should change. A diagnostic that names a defect without
            naming a next step gets read as noise.
        weights: the quadrature the caller's own mass functional uses. ``None`` asserts
            the quadrature is uniform; see the module docstring.
        threshold: fraction of present mass the clip may create.

    Returns:
        The clipped array. **Not renormalised** -- restoring the pre-clip total is what
        made this class of defect invisible. Absorbing or source boundaries change mass
        legitimately and are the BC layer's business, not this function's.
    """
    fabricated = mass_fabricated_by_clip(density, weights=weights)
    if fabricated > threshold:
        raise ValueError(
            f"{context}: density went to {float(np.min(density)):.3e}. Clipping it to zero "
            f"would fabricate {fabricated:.3%} of the total mass, so the solve is stopped "
            f"rather than reporting a conserved density it did not compute. {remedy}"
        )
    return np.maximum(density, 0.0)
