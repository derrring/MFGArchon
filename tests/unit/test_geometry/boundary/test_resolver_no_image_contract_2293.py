"""Both resolvers must refuse a BC member they cannot map. One of them returns a default instead.

RECORDED DEFECT, not a contract (#2293). `HJBResolver` and `FPResolver` live in one file, implement
one protocol, and disagree on what happens to a member with no image in `MathBCType`: FP raises,
HJB logs a WARNING and returns `ResolvedBC(NEUMANN, alpha=1, beta=0, g=0)` — a well-formed
homogeneous Neumann wall the caller cannot distinguish from a real resolution.

Retirement: give `HJBResolver` the `else: raise` its sibling already has, and
`test_the_hjb_resolver_still_returns_a_default_instead_of_refusing` fails with an instruction. Delete
that test then, and `test_both_resolvers_refuse_a_member_they_cannot_map` becomes the whole file.

Why a pin rather than a fix here: choosing the failure mode is a contract decision (#1471 already
wrote it down as "total-or-fail-loud"), and #2295 is why nothing caught the divergence — the layer
is not type-checked and 0 of its 9 functions execute in a 60-configuration census.
"""

from __future__ import annotations

import logging

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
    """Positive control for every refusal in this file.

    Without it, `FPResolver` raising on `_NO_IMAGE` proves nothing: that resolver also raises when
    `solver_state` lacks `drift`, with a different message, and the two are indistinguishable from a
    test that only asserts "it raised".
    """
    from mfgarchon.geometry.boundary import no_flux_bc

    segment = no_flux_bc(dimension=1).segments[0]

    assert HJBResolver().resolve(segment, _STATE).math_type is MathBCType.NEUMANN
    assert FPResolver().resolve(segment, _STATE).math_type is MathBCType.ROBIN


@pytest.mark.parametrize(
    "resolver",
    [
        # The mark goes on the PARAM, not inside the body. An imperative `pytest.xfail()` aborts
        # before the assertion runs, so the cell can never report XPASS and the exemption becomes
        # permanent -- measured under this repository's `xfail_strict = true`, the marker form on a
        # passing test gives FAILED [XPASS(strict)] and the imperative form gives a silent `xfailed`.
        # The first draft of this file used the imperative form while its docstring promised the
        # marker's behaviour, which is the same defect #1948's table carried.
        pytest.param(
            HJBResolver(),
            marks=pytest.mark.xfail(strict=True, reason="#2293: HJBResolver defaults instead of refusing"),
        ),
        pytest.param(FPResolver()),
    ],
    ids=["HJB", "FP"],
)
def test_both_resolvers_refuse_a_member_they_cannot_map(resolver):
    """The contract #1471 already wrote down: "total-or-fail-loud (no best-effort collapse)"."""
    with pytest.raises((ValueError, NotImplementedError, TypeError)):
        resolver.resolve(_segment(_NO_IMAGE), _STATE)


def test_the_hjb_resolver_still_returns_a_default_instead_of_refusing(caplog):
    """RECORDED DEFECT (#2293). This asserts the WRONG behaviour on purpose.

    Fixing #2293 trips it, and the message below is the instruction. It pins three things a partial
    fix could each leave in place: that a value comes back at all, that the value is the homogeneous
    Neumann wall specifically, and that the original member survives on the returned object — the
    last being the only thing that lets a consumer notice after the fact.
    """
    with caplog.at_level(logging.WARNING, logger="mfgarchon.geometry.boundary.resolution"):
        resolved = HJBResolver().resolve(_segment(_NO_IMAGE), _STATE)

    assert resolved is not None, (
        "HJBResolver now refuses an unmappable member. That is the #2293 fix: delete this test and "
        "remove the xfail from test_both_resolvers_refuse_a_member_they_cannot_map."
    )
    assert resolved.math_type is MathBCType.NEUMANN
    assert (resolved.value, resolved.alpha, resolved.beta) == (0.0, 1.0, 0.0)
    assert resolved.original_bc_type is _NO_IMAGE

    assert caplog.records, "the default is now silent as well as wrong; #2293 got worse rather than fixed"


def test_the_two_resolvers_disagree_and_that_is_the_defect():
    """The pair, asserted together, because neither half alone shows the contract is not shared.

    A reader who sees only the FP raise concludes the contract is fail-loud; a reader who sees only
    the HJB default concludes it is best-effort. The file states which it is by asserting both on
    one input.
    """
    segment = _segment(_NO_IMAGE)

    hjb_returned = HJBResolver().resolve(segment, _STATE)

    with pytest.raises(ValueError, match="unrecognized"):
        FPResolver().resolve(segment, _STATE)

    assert hjb_returned is not None, (
        "both resolvers now refuse: the contract is shared and #2293 is fixed. Delete this test."
    )
