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

Round-off gives ~1e-15 **where it occurs at all**. A scheme that has genuinely failed gives
O(1). The default threshold sits between them.

Whether that picture holds is a per-site empirical question, and the two sites measured answer
it opposite ways -- so measure before adopting the default at a third.

- **GFDM** (576 configurations): nothing lands in `(0, 1.86e-06)`. The distribution is
  `{exactly 0}` union `[1.86e-06, O(1)]`, so every value in `[0, 1.9e-06]` is behaviourally
  identical there. The default is right, and no tighter value buys anything.
- **The network graph scheme** (3704 configurations): the fabricated fraction runs continuously
  from 1e-9 to O(1), and the default rejected solves whose honest answer was a drift of 5.8e-5.
  It passes its own `threshold=` (`fp_network.py`, `_MAX_NETWORK_CLIP_FABRICATION`), chosen for
  zero false negatives out of 2941 broken solves at a cost of 31 false positives out of 533
  honest ones. No value separates the two populations there -- on scale-free topologies they
  abut at a ratio of 1.011 -- because the per-step observable is a discordant proxy for the
  drift it protects (16% of pairs rank-inverted). #1758 tracks that.

What is single-sourced here is the **invariant** -- how fabricated mass is defined and compared.
A tolerance is not an invariant: it belongs to the scheme whose discretisation error it has to
clear, the same way a solver tolerance does.

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

**Demonstrated at GFDM only.** The general form -- "any fixed threshold is defeatable by
refining dt at any caller" -- is not established; an attempt to reproduce it at the FDM
time-stepping site produced a null with no working positive control. Do not cite it as a
property of this function until a second site shows it.

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

The check is mechanical: read what the caller itself compares to decide mass changed, and match
the gate to that. Worked example -- GFDM compares `np.sum(M)` against `np.sum(m_init)` and
carries no cell measure anywhere (the only weights under `gfdm_components/` are least-squares
stencil and interpolation weights), so `weights=None` is not merely the default there, it is
the only correct value.

Read the site. The scheme family does not tell you: "GFDM" names a stencil construction, not a
quadrature, and reasoning from the name gets this backwards.

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

__all__ = [
    "GROSS_MASS_CHANGE_BAND",
    "MAX_CLIP_MASS_FABRICATION",
    "MAX_CONSERVED_MASS_DRIFT",
    "clip_nonnegative_or_raise",
    "gross_mass_excursion",
    "mass_fabricated_by_clip",
    "stop_on_gross_mass_change",
    "stop_on_mass_drift",
]

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
            f"would fabricate {fabricated:.3%} (fraction {fabricated:.3e}, threshold {threshold:.0e}) of the total "
            f"mass, so the solve is stopped rather than reporting a conserved density it did not compute. {remedy}"
        )
    return np.maximum(density, 0.0)


#: Default tolerance on |mass_t / mass_0 - 1| for a scheme that opts into `stop_on_mass_drift`.
MAX_CONSERVED_MASS_DRIFT = 1e-2


def stop_on_mass_drift(
    mass: float,
    mass_initial: float,
    *,
    step: int,
    tolerance: float,
    context: str,
    remedy: str,
    source_added: float = 0.0,
    source_scale: float = 0.0,
) -> float:
    """Stop a solve whose mass has left its budget (#2512 S5).

    `clip_nonnegative_or_raise` cannot see a solve that blows up while staying positive: it measures
    fabricated mass as a ratio of the mass present, so it is scale-invariant (#1752 measured a finite,
    non-negative density of mass 2.55e+09 that it passed). This checks the other invariant, the total.

    The law is d/dt (integral of m) = integral of S. The budget is ``mass_initial + source_added``, where
    ``source_added`` is the source the caller's integrator actually added, summed by the integrator's own
    rule (time level, dt); with no source it is the initial mass. The drift is measured against
    ``mass_initial + source_scale``, the same sum over |S|, so a source that drains the mass toward zero
    is not divided by a vanishing number.

    Call it only where nothing else moves the mass -- zero-flux walls, nothing absorbing; deciding that
    is the caller's, which knows its BCs. ``mass`` must be the integral with the domain's measure, not an
    unweighted sum, whose drift on an exact solution can exceed any useful tolerance (#2512 S5). With a
    scale <= 0 there is nothing to compare against and the check returns 0.0. Returns the drift.
    """
    scale = float(mass_initial) + float(source_scale)
    if not scale > 0:
        return 0.0
    budget = float(mass_initial) + float(source_added)
    drift = abs(float(mass) - budget) / scale
    if not drift <= tolerance:
        if source_scale:
            what = f"mass {float(mass):.6g} against its budget {budget:.6g} (initial mass plus the source added)"
        else:
            what = f"mass ratio {float(mass) / float(mass_initial):.6g}"
        raise ValueError(
            f"{context}: {what} at step {step} (|drift| {drift:.3e} of {scale:.6g} > tolerance {tolerance:.3g}), "
            f"where only the source moves the mass, so the solve is stopped rather than returning a density "
            f"with that mass. {remedy}"
        )
    return drift


#: The band for `stop_on_gross_mass_change`: a factor of 10 each way. For a non-negative density an
#: unweighted ratio can differ from the true one by the cloud's quadrature-weight spread (max w / min w),
#: which adaptive clouds put well above 2; the check exists for catastrophes (its pinned specimen reaches
#: 2.53e+09), and a factor of 10 still catches those at the first step past it.
GROSS_MASS_CHANGE_BAND = (0.1, 10.0)


def gross_mass_excursion(
    total: float,
    total_initial: float,
    *,
    band: tuple[float, float] = GROSS_MASS_CHANGE_BAND,
    source_added: float | None = None,
) -> float | None:
    """The factor by which sum|m| left its band, or None while it is inside.

    Without a source (``source_added is None``) the band is two-sided around ``total_initial``. Under a
    source the mass may legitimately drain, so only the upper bound applies, against ``total_initial +
    source_added`` -- the sum of dt * sum|S_k| the solve added. ``sum|m|`` and not ``sum m``, because a
    sign-indefinite density can cancel in a plain sum and hide growth (#1683).
    """
    reference = float(total_initial) + (float(source_added) if source_added is not None else 0.0)
    if not reference > 0:
        return None
    ratio = float(total) / reference
    low, high = band
    if ratio > high or (source_added is None and ratio < low):
        return ratio
    return None


def stop_on_gross_mass_change(
    total: float,
    total_initial: float,
    *,
    step: int,
    context: str,
    remedy: str,
    band: tuple[float, float] = GROSS_MASS_CHANGE_BAND,
    source_added: float | None = None,
) -> None:
    """Stop a solve whose density blows up -- or, with no source, vanishes -- where no measure exists to
    check the mass with. A gross check, not a conservation check: see `gross_mass_excursion`."""
    ratio = gross_mass_excursion(total, total_initial, band=band, source_added=source_added)
    if ratio is None:
        return
    low, high = band
    if source_added is None:
        what = f"sum|m| changed by a factor {ratio:.6g} at step {step}, outside the gross band [{low:.6g}, {high:.6g}]"
    else:
        what = (
            f"sum|m| reached {ratio:.6g} times sum|m_0| plus the |source| added at step {step}, above the gross "
            f"bound {high:.6g}"
        )
    raise ValueError(
        f"{context}: {what}. This is a gross blow-up check, not a conservation check: there is no measure on "
        f"these points to check the mass with. {remedy}"
    )
