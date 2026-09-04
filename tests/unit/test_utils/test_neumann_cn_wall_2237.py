"""What each wall treatment conserves, and what it costs. Issue #2237.

The library had six implementations of the Crank-Nicolson diffusion step at a zero-flux wall. Five
were the same operator and one was not, and no file said which was which -- each carried its own
comment justifying a choice, and three of those comments described a choice the code did not make.
`utils.numerical.implicit_diffusion` is now the single owner and the difference is a parameter.

This file does NOT check that there are six call sites; a hand-written list of today's call sites
confirms itself and says nothing about tomorrow's. It pins the pairing that made the duplication
dangerous: **each wall conserves a different measure, and neither conserves both.** On an
endpoint-inclusive grid the end node owns h/2, so the mass is the trapezoid integral -- the point
#2145 established for `operators.differential.laplacian`, which the SL/CN family never followed.
A check written against uniform weights passes on the wall that does not conserve the mass.

Both oracles are independent of the scheme: exact quadrature invariance, and an exact heat solution.
"""

import pytest

import numpy as np

from mfgarchon.utils.numerical.implicit_diffusion import cn_alpha, neumann_cn_stencil, neumann_cn_step

SIGMA = 0.4
DT = 0.01


def _operator(n: int, dx: float, treatment: str) -> np.ndarray:
    """The matrix the step actually applies, reconstructed from its action on the standard basis.

    Reading the code produced three wrong conclusions during #2237; acting on the basis produced
    the right one, so the test uses the instrument that worked.
    """
    return np.column_stack([neumann_cn_step(np.eye(n)[k], DT, SIGMA, dx, treatment=treatment) for k in range(n)])


def _weights(n: int, kind: str) -> np.ndarray:
    """Uniform weights, or the trapezoid weights an endpoint-inclusive grid actually carries."""
    w = np.ones(n)
    if kind == "trapezoid":
        w[0] = w[-1] = 0.5
    return w


@pytest.mark.parametrize(
    ("treatment", "conserved", "broken"),
    [("half_wall", "uniform", "trapezoid"), ("mirror", "trapezoid", "uniform")],
)
@pytest.mark.parametrize("n", [9, 21, 81])
def test_each_wall_conserves_exactly_one_of_the_two_measures(treatment, conserved, broken, n):
    """`w @ A == w` for its own weighting, and measurably not for the other.

    Asserting only the first half would pass on both walls under whichever weighting the test
    happened to pick -- which is how a wall that does not conserve the mass survived in five
    places. The `broken` half is what makes the check discriminating.
    """
    a = _operator(n, 1.0 / (n - 1), treatment)
    kept, lost = _weights(n, conserved), _weights(n, broken)
    assert np.max(np.abs(kept @ a - kept)) < 1e-13, f"{treatment} lost its own measure ({conserved})"
    assert np.max(np.abs(lost @ a - lost)) > 1e-4, f"{treatment} unexpectedly also conserved {broken}"


def test_the_trapezoid_is_the_measure_this_grid_carries():
    """The reason the pairing above matters, as an assertion rather than a comment.

    Diffusing a density with a non-zero wall gradient, `mirror` holds the trapezoid integral to
    roundoff while `half_wall` drifts it -- and the drift halves as n doubles, so it is the wall's
    first-order error and not an accumulation artefact.
    """
    drift = {}
    for treatment in ("half_wall", "mirror"):
        for n in (41, 81):
            x = np.linspace(0.0, 1.0, n)
            m = np.exp(-40 * (x - 0.35) ** 2) + 0.1
            mass_0 = np.trapezoid(m, x)
            for _ in range(300):
                m = neumann_cn_step(m, 0.002, SIGMA, x[1] - x[0], treatment=treatment)
            drift[treatment, n] = abs(np.trapezoid(m, x) - mass_0) / mass_0
    assert drift["mirror", 41] < 1e-12, f"mirror drifted: {drift}"
    assert drift["half_wall", 41] > 1e-3, f"half_wall did not drift: {drift}"
    ratio = drift["half_wall", 41] / drift["half_wall", 81]
    assert 1.7 < ratio < 2.3, f"drift is not first order in h: ratio {ratio:.3f} from {drift}"


def test_the_treatments_differ_only_in_the_wall_row():
    """If they ever differ in the interior, one of them has stopped being the theta-scheme."""
    st_h = neumann_cn_stencil(cn_alpha(DT, SIGMA, 0.125), treatment="half_wall")
    st_m = neumann_cn_stencil(cn_alpha(DT, SIGMA, 0.125), treatment="mirror")
    assert (st_h.implicit_main, st_h.implicit_off) == (st_m.implicit_main, st_m.implicit_off)
    assert (st_h.explicit_main, st_h.explicit_off) == (st_m.explicit_main, st_m.explicit_off)
    assert st_m.implicit_wall_off == pytest.approx(2.0 * st_h.implicit_wall_off)


@pytest.mark.parametrize(("treatment", "expected_order"), [("half_wall", 1.0), ("mirror", 2.0)])
def test_order_at_the_wall(treatment, expected_order):
    """Against an exact solution, computed without reference to the scheme.

    `u(0, x) = cos(pi x)` on [0, 1] satisfies zero flux at both walls, and diffusion carries it to
    `exp(-D pi^2 t) cos(pi x)`. The wall is where the two differ, so the error there is what
    separates them: `half_wall` is first order, `mirror` second.
    """
    d = SIGMA**2 / 2.0
    t_end = 0.05
    errors = []
    for n in (21, 41, 81, 161):
        x = np.linspace(0.0, 1.0, n)
        dx = x[1] - x[0]
        dt = t_end / 200
        u = np.cos(np.pi * x)
        for _ in range(200):
            u = neumann_cn_step(u, dt, SIGMA, dx, treatment=treatment)
        exact = np.exp(-d * np.pi**2 * t_end) * np.cos(np.pi * x)
        errors.append(np.max(np.abs(u - exact)))
    eocs = [np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
    assert eocs[-1] == pytest.approx(expected_order, abs=0.25), f"EOC {eocs} from errors {errors}"


def test_an_unknown_treatment_is_refused_by_name():
    """Silently falling back to a default would make the wall a coin flip, which is the whole bug."""
    with pytest.raises(ValueError, match="half_wall"):
        neumann_cn_step(np.ones(5), DT, SIGMA, 0.25, treatment="ghost")
