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
reported its whole-solve drift, but reported is the accurate verb -- it was a `logger.warning`,
and the solve returned. That was the only thing separating a GFDM configuration whose honest
answer was a 0.27% drift from one that returned a final mass of 1.06e+23, and by this campaign's
own standard ("a diagnostic nobody reads is the same failure as no diagnostic") a log is not a
gate. FP-GFDM has since been withdrawn (#2583), so that solve no longer runs. The point stands for any
scheme: what should stop a divergent-but-positive solve is a different invariant than this one,
not a stricter threshold.

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

from mfgarchon.utils.numerical.quadrature import integrate_on_weights

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

__all__ = [
    "MAX_CLIP_MASS_FABRICATION",
    "clip_nonnegative_or_raise",
    "leave_out_or_raise",
    "mass_fabricated_by_clip",
    "refuse_what_a_flux_takes_negative",
]

# The largest fraction of the present mass a non-negativity clip may create before the
# solve is stopped. Round-off is ~1e-15; a failed scheme is O(1) (Issue #1671).
MAX_CLIP_MASS_FABRICATION = 1e-8


def _part_mass(part: NDArray[np.floating], where: NDArray[np.bool_], weights: NDArray[np.floating] | None) -> float:
    """The mass of ``part`` on the nodes ``where`` selects, on the caller's measure: the plain sum over those nodes for a
    uniform quadrature (``weights=None``), else the integral on ``weights`` through its one sum (#1642). The one form
    `mass_fabricated_by_clip` and B-3's owner measure with, so the owner's ``s0 <= threshold`` is the gate's own."""
    if weights is None:
        return float(part[where].sum())
    return float(integrate_on_weights(np.where(where, part, 0.0), np.asarray(weights)))


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
    if weights is not None and np.asarray(weights).shape != density.shape:
        raise ValueError(f"weights shape {np.asarray(weights).shape} does not match density shape {density.shape}")
    negative_mass = _part_mass(-density, negatives, weights)
    positive_mass = _part_mass(density, density > 0, weights)
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


def refuse_what_a_flux_takes_negative(
    m0: NDArray[np.floating],
    m_q: NDArray[np.floating],
    *,
    context: str,
    time: float | None,
    unit: str,
    locate: Callable[[int], str],
    remedy: str,
    remedy_past_gate: str,
    weights: NDArray[np.floating] | None = None,
    threshold: float = MAX_CLIP_MASS_FABRICATION,
) -> NDArray[np.floating]:
    """A step with a prescribed flux, ``m0 + m_q``, or its refusal where clipping it would fabricate, past this gate, mass
    the flux decides: B-3 of #2581 (user rulings 2026-10-09 and 2026-10-30), as the approver ruled it on #2581's plan
    (Q3) and on #2840. Never clipped past the gate, on every clause.

    The step is linear in its right-hand side, so ``m0`` is the step without the flux and ``m_q`` the flux's own part.
    With ``F`` the mass of the negatives the flux alone makes (at the nodes ``m0`` leaves non-negative), ``D`` the depth
    it adds at ``m0``'s negative nodes (``sum over m0_i < 0 of w_i max(-m_q_i, 0)``, none netted against a node the
    flux fills), ``s0`` and ``s_step`` this gate's fraction of ``m0`` and of ``m0 + m_q``, the step is refused when

    1. ``F / pos(m0 + m_q) > threshold``, whatever the rest of the step does: the flux alone takes those nodes
       negative; round-off within the gate is clipped, as any clip's is; or
    2. ``s0 <= threshold`` and either (a) ``s0 + D / pos(m0 + m_q) > threshold``: the flux deepens the step's own
       negatives; or (b) ``s_step > threshold``: the step with the flux fails the gate itself, as a flux that drains the
       positive mass makes it, F, D and the step's own negatives counted together.

    Neither refuses a step with no negative (``s_step = 0`` leaves ``F = D = 0``). ``F`` and ``D`` are measured on the
    step that holds them, a zero ``F`` or ``D`` as 0 and any other on a step with no positive mass as infinite;
    ``pos(m0)`` enters only through ``s0``, which decides whether (2) applies. ``F`` is not added to (2a); ``s_step``
    counts it with the step's other negatives. So neither depends on how little mass the step without the flux holds:
    an inflow's round-off negatives in a nearly empty domain are measured against the mass the inflow brings.
    Otherwise a step whose own clip is past the threshold is the scheme's: the caller's clip decides the step with the
    flux as it decides any step.
    ``weights`` are those of that clip, so that ``s0 <= threshold`` is its verdict on the step without the flux.
    ``remedy`` follows a refusal; ``remedy_past_gate`` follows clause (1)'s when the step without the flux is itself
    past the gate, where the caller's clip, not the flux, decides what a change of this step's q does. The state the
    step starts from holds what the flux did on the earlier steps. The step is not clipped.
    """
    step = m0 + m_q
    step_positive = _part_mass(step, step > 0, weights)
    own = mass_fabricated_by_clip(m0, weights=weights)
    taken = (step < 0) & (m0 >= 0)
    made = _part_mass(-step, taken, weights) if np.any(taken) else 0.0
    alone = 0.0 if made == 0 else (made / step_positive if step_positive > 0 else float("inf"))
    if alone > threshold:
        worst = int(np.argmin(np.where(taken, step, np.inf)))
        if own > threshold:
            tail = (
                f" The step without the flux would itself fabricate {own:.3e}, past the threshold {threshold:.0e}: the "
                f"scheme's own step from this state fails there before this step's flux is added. {remedy_past_gate}"
            )
        else:
            tail = f" {remedy}"
        raise ValueError(
            f"{context}: the prescribed flux's part of the step would take the density negative at {int(taken.sum())} "
            f"{unit}(s) in the step to t={time}, the most at {locate(worst)} (m = {step[worst]:.3e}), where the step "
            f"without it is non-negative ({m0[worst]:.3e} there). Clipping that would fabricate {alone:.3e} of the "
            f"mass, past the threshold {threshold:.0e}, which the flux alone decides, so the step is refused (#2581, "
            f"user rulings 2026-10-09 and 2026-10-30, B-3).{tail}"
        )
    if own <= threshold:
        deepened = _part_mass(np.maximum(-m_q, 0.0), m0 < 0, weights)
        added = 0.0 if deepened == 0 else (deepened / step_positive if step_positive > 0 else float("inf"))
        if own + added > threshold:
            raise ValueError(
                f"{context}: in the step to t={time}, clipping the step without the prescribed flux would fabricate "
                f"{own:.3e} of the mass, within the threshold {threshold:.0e}, and the flux's part deepens its "
                f"negatives by {added:.3e} more, past it. The flux decides the clip, so the step is refused rather "
                f"than clipped (#2581, user rulings 2026-10-09 and 2026-10-30, B-3). {remedy}"
            )
        clipped = mass_fabricated_by_clip(step, weights=weights)
        if clipped > threshold:
            raise ValueError(
                f"{context}: in the step to t={time}, clipping the step without the prescribed flux would fabricate "
                f"{own:.3e} of the mass, within the threshold {threshold:.0e}, and clipping the step with it would "
                f"fabricate {clipped:.3e}, past it. The flux decides the clip, so the step is refused rather than "
                f"clipped (#2581, user rulings 2026-10-09 and 2026-10-30, B-3). {remedy}"
            )
    return step


def leave_out_or_raise(
    probs: NDArray[np.floating],
    blocked: NDArray[np.bool_],
    *,
    context: str,
    why: str,
    remedy: str,
    threshold: float = MAX_CLIP_MASS_FABRICATION,
) -> NDArray[np.floating]:
    """The non-negative weights ``probs`` with the ``blocked`` entries left out and the rest renormalised to sum to 1,
    or a refusal when the blocked share of the total is more than a clip may change the caller's measure.

    The counterpart of `clip_nonnegative_or_raise` for mass the library cannot place rather than mass a clip would
    create: the same threshold, compared the same way, and a message of the same shape -- the share, why it cannot be
    kept, and the caller's choices (#2706). At or below the threshold the mass is left out silently, as a clip of
    that size is. The callers compute ``blocked`` only; leaving out and renormalising are here (#2706, review 2).
    """
    total = float(probs.sum())
    fraction = float(probs[blocked].sum()) / total if total > 0 else float("inf")
    if fraction > threshold:
        raise ValueError(
            f"{context}: {why} Leaving it out would drop a fraction {fraction:.3e} of the total mass (threshold "
            f"{threshold:.0e}), so the solve is stopped rather than run on a measure the caller did not give. The "
            f"total is the net sum, so a negative entry lowers it and raises this fraction: a density must also be "
            f"non-negative. {remedy}"
        )
    kept = np.where(blocked, 0.0, probs)
    return kept / kept.sum()
