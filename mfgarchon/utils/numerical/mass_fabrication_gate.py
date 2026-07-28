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

Round-off gives ~1e-15. A scheme that has genuinely failed gives O(1). There is no
interesting régime between them, which is why a single threshold works.

## What this cannot see, and why no threshold fixes it

The ratio is **scale-invariant** and evaluated **per step**. Both matter, and the second is
the serious one: refining the timestep shrinks what any one step can fabricate, so the
observable can vanish under dt-refinement whether or not the answer improves. Measured on the
GFDM path -- and, so far, only there (sigma=0.1, drift=25, Issue #1752):

    Nt        dt        max fabricated      final mass     this gate
    10     5.00e-02        9.591e-02          8.40e+02       raises
    20     2.50e-02        5.961e-03          5.18e+04       raises
   160     3.13e-03        3.646e-04          4.17e+08       raises
   640     7.81e-04        3.396e-05          1.70e+09       raises
  2560     1.95e-04        0.000e+00          2.55e+09       PASSES
 10240     4.88e-05        0.000e+00          2.83e+09       PASSES

The observable falls five orders and then to exactly zero -- nothing goes negative at all --
while the end-to-end error climbs seven. At Nt=2560 the solve is maximally wrong and
maximally clean by this function's own criterion.

**Scope of that measurement.** It is one site. The obvious generalisation -- "any fixed
threshold is defeatable by refining dt at any caller" -- was asserted during review and then
withdrawn, because the attempt to reproduce it at the FDM time-stepping site produced a null
with no working positive control. So: demonstrated at the GFDM site, open elsewhere. Do not
cite it as a general property of this function until a second site shows it.

What is general, and does not depend on that table: this gate rules out a step that repairs a
sign violation large enough to matter. It does not certify that a solve converged. A caller
must not read a passing gate as "healthy".

Where magnitude matters, something must **stop**, and this function is not it. `fp_gfdm.py`
reports its whole-solve drift, but reports is the accurate verb -- it is a `logger.warning`,
and the solve returns. That is the only thing separating a GFDM configuration whose honest
answer is a 0.27% drift from one that returns a final mass of 1.06e+23, and by this campaign's
own standard ("a diagnostic nobody reads is the same failure as no diagnostic") a log is not a
gate. Recorded rather than fixed: what should stop a divergent-but-positive solve is a
different invariant than this one, not a stricter threshold.

Worth stating plainly because the failure is adversarial in shape wherever it does occur: the
natural response to this gate firing is to refine the timestep, and on a scheme whose spatial
operator is the real problem that silences the gate and makes the answer worse.

## Why weights are optional, and when they are not

For a **uniform** quadrature the cell measure cancels in the ratio, so the same numbers
come out whether or not you multiply by dV -- and passing them is wasted work. For a
non-uniform one (a weak-form mass matrix is the real case) it does **not** cancel, and
omitting the weights silently measures a different quantity than the solver's own mass
functional. Hence `weights=None` means "uniform, and I have checked that it is", not "I
did not think about it".

The check is mechanical: read what the caller itself compares to decide mass changed, and
match the gate to that. Worked example -- GFDM compares `np.sum(M)` against `np.sum(m_init)`
and carries no cell measure anywhere (the only weights under `gfdm_components/` are
least-squares stencil and interpolation weights), so `weights=None` is not merely the default
there, it is the only correct value.

An earlier version of this paragraph offered "GFDM quadrature" as the paradigm non-uniform
case. That was wrong, and wrong in a specific way worth naming: it reasoned from the family
name rather than from the code, so the first caller it warned about was one where the warning
did not apply. The replacement then over-corrected by counting callers it had not read. Check
each site; do not generalise from the scheme family, and do not generalise from this file.

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
