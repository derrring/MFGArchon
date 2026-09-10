"""Both resolvers refuse a BC member they cannot map. One protocol, one failure contract.

The contract is #1471's, written down there as "total-or-fail-loud (no best-effort collapse)".
`HJBResolver` and `FPResolver` live in one file and implement one protocol, so they owe the same
answer to a member with no image in `MathBCType`.

WAS A RECORDED DEFECT PIN (#2293), retired 2026-09-10 by its own stated condition. `HJBResolver`
logged a WARNING and returned `ResolvedBC(NEUMANN, alpha=1, beta=0, g=0)` — a well-formed
homogeneous Neumann wall the caller could not distinguish from a real resolution — while `FPResolver`
raised. Giving HJB the raise turned the `[HJB]` param XPASS(strict) and failed the two tests that
asserted the defect, each printing the instruction that said to delete it. Both are deleted; this is
what the file's own docstring said it would become.

What it defends now: a caller handed a member outside `BCType` gets an exception from either
resolver, never a wall. The failure mode this rules out is the silent one — a default reflecting
boundary is well-posed and convergent, so a solve built on it runs to completion and answers a
question nobody asked.
"""

from __future__ import annotations

import pytest

from mfgarchon.geometry.boundary import BCSegment
from mfgarchon.geometry.boundary.applicator_graph import GraphBCType
from mfgarchon.geometry.boundary.resolution import FPResolver, HJBResolver, MathBCType

#: A real member of a real alphabet in this package with no `BCType` counterpart, so it is the shape
#: a caller actually reaches this path with rather than a synthetic sentinel. `GraphBCType.SOURCE`
#: and `.CUSTOM` would do as well; `ABSORBING` is used because #2292 records it as the member that
#: reaches `NodeBC` today.
_NO_IMAGE = GraphBCType.ABSORBING

#: `FPResolver` needs these to resolve a real member at all, so passing them proves the refusal
#: below is about the *member* and not about missing state — see the positive control.
_STATE = {"drift": 0.5, "diffusion": 0.1}


def _segment(bc_type):
    return BCSegment(name="s", bc_type=bc_type, value=0.0)


def test_the_state_is_sufficient_for_a_member_that_does_have_an_image():
    """Positive control for the refusal below.

    Without it, `FPResolver` raising on `_NO_IMAGE` proves nothing: that resolver also raises when
    `solver_state` lacks `drift`, with a different message, and the two are indistinguishable from a
    test that only asserts "it raised". The same now goes for `HJBResolver`, which is why the
    control covers both.
    """
    from mfgarchon.geometry.boundary import no_flux_bc

    segment = no_flux_bc(dimension=1).segments[0]

    assert HJBResolver().resolve(segment, _STATE).math_type is MathBCType.NEUMANN
    assert FPResolver().resolve(segment, _STATE).math_type is MathBCType.ROBIN


@pytest.mark.parametrize("resolver", [HJBResolver(), FPResolver()], ids=["HJB", "FP"])
def test_both_resolvers_refuse_a_member_they_cannot_map(resolver):
    """The contract #1471 already wrote down: "total-or-fail-loud (no best-effort collapse)"."""
    with pytest.raises((ValueError, NotImplementedError, TypeError), match="unrecognized"):
        resolver.resolve(_segment(_NO_IMAGE), _STATE)
