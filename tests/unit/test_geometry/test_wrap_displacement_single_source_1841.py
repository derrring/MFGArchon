"""The minimum-image wrap must not move when its owner does. Issue #1841.

`wrap_displacement` was a method on `Hyperrectangle` with exactly one caller. #1841 moved the
arithmetic to `geometry.boundary.periodic`, routed that caller through the protocol's `get_periods()`
so any `SupportsPeriodic` geometry works, and deleted the method. Implementations went 2 -> 1.

The pin for a consolidation cannot be path-A-vs-path-B: once both encodings route through one owner,
their agreement is tautological and passes over a broken owner. So these are BYTE-IDENTITY cases
captured from the pre-change implementation at `origin/main` and embedded as literals -- the only
form that still discriminates after the fork is closed.

Without them the owner had no discriminating coverage at all: independent review mutated
`np.round` -> `np.floor`, which flips the sign convention for every negative displacement, and the
entire marker-filtered suite stayed green at 5856 passed. The wrap is arithmetically inert on the
paths the suite reaches -- neighbours come from the ghost-augmented cloud, so the displacement is
already the minimum image -- which is exactly why it needs a direct pin rather than an integration
test.

Coverage is deliberate about the cases where an implementation can drift: round-half-to-even ties at
L/2 and 1.5L, displacements many periods long, an offset (non-zero-based) domain, a subset of axes,
axes given in DESCENDING order (the old loop iterated a tuple, the new one a dict), 3-D, the
no-periodic-dims identity path, and the 1-D single-vector branch.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.geometry.boundary.periodic import wrap_displacement

# (label, bounds, periodic_dims, delta, expected) -- `expected` produced by
# `Hyperrectangle.wrap_displacement` at origin/main, before it was deleted.
REFERENCE: list[tuple[str, list, tuple, list, list]] = [
    (
        "unit-1d-small",
        [[0.0, 1.0]],
        (0,),
        [[0.1], [-0.9], [0.0]],
        [[0.1], [0.09999999999999998], [0.0]],
    ),
    (
        "unit-1d-ties",
        [[0.0, 1.0]],
        (0,),
        [[0.5], [-0.5], [1.5], [2.5], [-1.5]],
        [[0.5], [-0.5], [-0.5], [0.5], [0.5]],
    ),
    (
        "unit-1d-huge",
        [[0.0, 1.0]],
        (0,),
        [[1000000.3], [-1000000.3]],
        [[0.30000000004656613], [-0.30000000004656613]],
    ),
    (
        "offset-1d",
        [[-2.0, 3.5]],
        (0,),
        [[2.75], [-2.75], [7.7]],
        [[2.75], [-2.75], [2.2]],
    ),
    (
        "2d-both",
        [[-1.0, 3.0], [0.0, 1.0]],
        (0, 1),
        [
            [-1.2780808160096675, -7.916648193178071],
            [-3.213968021433959, 1.2182901642096475],
            [3.3636139815802686, 7.2918189138922305],
            [-4.480977606486634, -5.304746482623764],
            [-6.682908026456461, -1.5937908523025182],
            [7.929504265123851, -5.613788770813995],
        ],
        [
            [-1.2780808160096675, 0.08335180682192878],
            [0.786031978566041, 0.21829016420964753],
            [-0.6363860184197314, 0.29181891389223047],
            [-0.48097760648663357, -0.3047464826237638],
            [1.3170919735435387, 0.40620914769748184],
            [-0.07049573487614857, 0.3862112291860047],
        ],
    ),
    (
        "2d-second-axis-only",
        [[-1.0, 3.0], [0.0, 1.0]],
        (1,),
        [
            [5.09030781716905, -4.718544065939266],
            [-5.474539101043742, 4.04338868045852],
            [-8.908169239492473, -2.1389053009552326],
            [1.6221354769785281, 0.8314052065743427],
            [0.03333820760525397, -5.701619056288758],
        ],
        [
            [5.09030781716905, 0.2814559340607339],
            [-5.474539101043742, 0.04338868045851996],
            [-8.908169239492473, -0.13890530095523257],
            [1.6221354769785281, -0.1685947934256573],
            [0.03333820760525397, 0.29838094371124235],
        ],
    ),
    (
        "2d-reversed-order",
        [[-1.0, 3.0], [0.0, 1.0]],
        (1, 0),
        [
            [-3.361553472086714, 4.16994937597276],
            [6.904950942209215, 8.972463686510158],
            [3.3237568227028187, 7.720533737847077],
            [-7.5646755659393365, 0.8469216314794283],
            [0.13951731154422875, -7.003103997362217],
        ],
        [
            [0.638446527913286, 0.16994937597276039],
            [-1.095049057790785, -0.027536313489841646],
            [-0.6762431772971813, -0.2794662621529227],
            [0.43532443406066346, -0.15307836852057166],
            [0.13951731154422875, -0.0031039973622171146],
        ],
    ),
    (
        "3d",
        [[0.0, 2.0], [0.0, 1.0], [-1.0, 1.0]],
        (0, 2),
        [
            [6.217999846695465, -6.178937521674154, 5.389898643988186],
            [1.631577705882292, 3.236544287010334, -1.2328615544041215],
            [2.062672677550483, 1.9899315367914652, 5.355509591043985],
            [5.613699362055087, 2.2234722851819173, -3.898601223973963],
            [-4.706337974591535, -6.8395720010007945, 1.1874189606593606],
        ],
        [
            [0.21799984669546468, -6.178937521674154, -0.610101356011814],
            [-0.368422294117708, 3.236544287010334, 0.7671384455958785],
            [0.06267267755048289, 1.9899315367914652, -0.6444904089560151],
            [-0.3863006379449132, 2.2234722851819173, 0.10139877602603686],
            [-0.706337974591535, -6.8395720010007945, -0.8125810393406394],
        ],
    ),
    (
        "none-periodic",
        [[0.0, 1.0], [0.0, 1.0]],
        (),
        [
            [0.30455534725633515, 0.1490804207118861],
            [0.41577309417224173, 1.996582577356231],
            [1.738761981013141, 0.6144008098554237],
            [0.9094611902609546, -2.208752658105409],
        ],
        [
            [0.30455534725633515, 0.1490804207118861],
            [0.41577309417224173, 1.996582577356231],
            [1.738761981013141, 0.6144008098554237],
            [0.9094611902609546, -2.208752658105409],
        ],
    ),
    (
        "single-vector-1d",
        [[0.0, 1.0]],
        (0,),
        [0.9],
        [-0.09999999999999998],
    ),
    (
        "single-vector-2d",
        [[0.0, 2.0], [0.0, 1.0]],
        (0, 1),
        [3.3, -2.2],
        [-0.7000000000000002, -0.20000000000000018],
    ),
]


@pytest.mark.parametrize(
    ("label", "bounds", "periodic_dims", "delta", "expected"), REFERENCE, ids=[c[0] for c in REFERENCE]
)
def test_the_wrap_is_byte_identical_to_the_implementation_it_replaced(label, bounds, periodic_dims, delta, expected):
    periods = {d: bounds[d][1] - bounds[d][0] for d in periodic_dims}
    got = wrap_displacement(np.array(delta, dtype=float), periods)
    want = np.array(expected, dtype=float)
    assert got.shape == want.shape, f"{label}: shape moved"
    assert got.tobytes() == want.tobytes(), (
        f"{label}: the wrap no longer reproduces the implementation it replaced\n  got  {got}\n  want {want}"
    )


def test_no_periodic_dimensions_returns_the_input_object_untouched():
    """The `if not periods` early exit is a no-copy identity, and callers rely on not paying for it."""
    delta = np.array([[1.5, -2.5]])
    assert wrap_displacement(delta, {}) is delta


def test_the_wrap_does_not_mutate_its_argument():
    """The old method copied before writing; a consolidation that started writing in place would be
    a silent aliasing bug for any caller reusing the buffer."""
    delta = np.array([[1.9, -0.2]])
    before = delta.copy()
    wrap_displacement(delta, {0: 1.0, 1: 1.0})
    np.testing.assert_array_equal(delta, before)
