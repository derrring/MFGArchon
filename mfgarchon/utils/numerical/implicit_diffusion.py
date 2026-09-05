"""One owner for the implicit (theta-scheme) diffusion step at a Neumann wall.

WHY THIS EXISTS (#2237). Six implementations of this step were in the library. Five produced the
SAME operator and one produced a different one, with no single place recording which was which.
Measured by reconstructing each one's operator from its action on the standard basis -- not by
reading the code, which had already produced three wrong conclusions:

    implementation                                        vs #1        1^T A = 1^T ?
    hjb_sl_adi.solve_crank_nicolson_diffusion_1d          --           yes (1.1e-16)
    hjb_sl_adi.solve_1d_diffusion_along_axis (ADI)        1.11e-16     yes (1.1e-16)
    fp_semi_lagrangian_adjoint (FPSLSolver, inline)       0.000e+00    yes (1.1e-16)
    adjoint.operators.build_diffusion_matrix_1d           2.22e-16     yes (1.1e-16)
    fp_semi_lagrangian (FPSLJacobianSolver, inline)       2.72e-02     no  (0.972 to 1.027)

at N = 7, sigma = 0.4, dt = 0.01, alpha = 0.0288. `adjoint.operators.build_diffusion_matrix_2d` is
the sixth and is not in that table: the census probed the 1D path, so a 2D assembly carrying its
own copy of the same wall was invisible to it. It was found afterwards by sweeping the tree for a
`dt/dx^2` expression beside a tridiagonal assembly, with the five above as the control.

THE TWO WALLS, and which measure each conserves. On an endpoint-inclusive grid the wall lies ON the
end node and that node owns h/2, so the mass is the TRAPEZOID integral, weights
``w = (1/2, 1, ..., 1, 1/2)`` -- not the uniform sum. The two walls each conserve one of them
exactly and neither conserves both. Measured on the step operator A, n = 5/9/21:

                    max|1^T A - 1^T|          max|w^T A - w^T|
    ``half_wall``   0 / 4.4e-16 / 2.2e-16     6.3e-03 / 2.4e-02 / 1.2e-01
    ``mirror``      1.3e-02 / 4.8e-02 / 2.2e-01   2.2e-16 / 2.2e-16 / 3.3e-16

- ``half_wall`` -- the wall row carries HALF the interior coefficient. Conserves ``sum(m)``.
- ``mirror`` -- the ghost-point reflection ``u_{-1} = u_1``, the full ``2*theta*alpha``. Conserves
  the trapezoid integral. Second order at the wall: EOC 2.00, 2.00, 2.00 against an exact heat
  solution, against 0.73, 0.87, 0.94 for ``half_wall``, and 1.7e3 times more accurate at nx = 161.

THIS IS NOT AN OPEN TRADE-OFF; #2145 SETTLED IT. `operators.differential.laplacian` carries the
argument in full: on this grid ``w^T L = 0`` is the statement that holds, ``1^T L = 0`` is column
conservation under the wrong weights, and the half wall's accuracy cost "was never the price of
conservation; it was the price of the wrong measure". #2145 moved both of that operator's branches
onto the mirror stencil.

#2243 MOVED THE REST. Every call site OF THIS MODULE now names ``mirror``; ``half_wall`` has none.
Measured in the PR that closed #2243 -- the wall EOC 0.73/0.87/0.94 -> 2.00/2.00/2.00 through the
shipped routines, and `FPSLSolver` landing exactly on `FPSLJacobianSolver`, which had ``mirror``
from the start.

**That is a statement about this module's consumers and NOT about the library.** A separate family
implements the same wall through ghost padding rather than through a stencil, and is still on the
half wall: `geometry.boundary` ZeroGradientCalculator, `operators.stencils.laplacian_with_bc`,
`operators.differential.LaplacianOperator.__call__`/`_matvec` (whose own `as_scipy_sparse` is on
the mirror, so that class answers differently by route), `operators.differential.DiffusionOperator`,
and `base_hjb._compute_laplacian_1d` -- which is HJB-FDM's residual path. They were consistent with
the SL/CN family before #2243 and are not now. Filed separately; do not read the paragraph above as
covering them.

``half_wall`` REMAINS A NAME, with no caller, and deliberately: it is what makes the pin in
`tests/unit/test_utils/test_neumann_cn_wall_2237.py` discriminating. That test asserts each wall
conserves its own measure AND measurably not the other, so deleting the loser would leave a check
that passes on any wall. Keeping it costs one dict entry; it is not an offer, and `treatment` has
no default so nothing acquires it by omission.

One caution that outlives the switch: the library carries two mass conventions, and they pair with
the two walls. `utils.numerical.flux_diagnostics.compute_mass_conservation_error` uses
``sum(M) * cell_volume``, which ``half_wall`` conserved; the ``np.trapezoid`` sites are the ones
``mirror`` conserves. Since #2243 the solvers are all on ``mirror``, so a mass check written with
the rectangle rule will report drift that is the convention's and not the solver's -- which is
exactly the shape of the eight failures #2233 hit and #2189 recorded after #2145.

What each wall does is pinned in `tests/unit/test_utils/test_neumann_cn_wall_2237.py`, against both
weightings and against an exact heat solution -- oracles independent of the scheme.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.linalg import solve_banded

if TYPE_CHECKING:
    from numpy.typing import NDArray

from mfgarchon.utils.pde_coefficients import diffusion_from_volatility

__all__ = [
    "NeumannCNStencil",
    "WallTreatment",
    "cn_alpha",
    "neumann_cn_step",
    "neumann_cn_stencil",
    "wall_factor",
]

WallTreatment = Literal["half_wall", "mirror"]

_WALL_FACTOR: dict[str, float] = {"half_wall": 1.0, "mirror": 2.0}


def wall_factor(treatment: WallTreatment) -> float:
    """How many interior coefficients the wall row carries: 1.0 for ``half_wall``, 2.0 for ``mirror``.

    The whole difference between the two treatments is this number, so it has one definition. It is
    exposed separately from `neumann_cn_stencil` because a consumer may hold the wall decision
    without holding a diffusion number: `adjoint.operators._build_1d_laplacian` assembles the bare
    negative Laplacian (interior row ``[-1, 2, -1]``, wall row ``[factor, -factor]``) and ``alpha``
    enters later, at the caller. Before #2243 that site carried its own ``1.0``, which is how it
    stayed outside #2237's census.
    """
    factor = _WALL_FACTOR.get(treatment)
    if factor is None:
        raise ValueError(
            f"treatment must be one of {sorted(_WALL_FACTOR)}, got {treatment!r}. The two are not "
            f"interchangeable and neither is a safe default: 'half_wall' conserves sum(m) and is "
            f"first order at the wall; 'mirror' conserves the trapezoid integral -- the mass on an "
            f"endpoint-inclusive grid, per #2145 -- and is second order."
        )
    return factor


@dataclass(frozen=True)
class NeumannCNStencil:
    """Every coefficient of the theta-scheme at a zero-flux wall, derived in one place.

    The six implementations #2237 found each recomputed all of these, and agreed on all but
    ``factor``. They are exposed as numbers rather than as an assembled operator because the
    consumers want different products from them -- a stepped field, the diagonals of a batched
    Thomas solve, a sparse matrix -- and forcing one product on all three would replace a
    duplicated constant with a worse coupling.

    ``implicit`` names the operator on ``u^{n+1}``, ``explicit`` the one on ``u^n``:

        (implicit) u^{n+1} = (explicit) u^n
    """

    alpha: float
    theta: float
    factor: float
    """1.0 for ``half_wall`` (wall row at half the interior coefficient), 2.0 for ``mirror``."""

    implicit_main: float
    implicit_off: float
    implicit_diag_term: float
    """This axis's contribution to the diagonal, ``2*theta*alpha``. In 1D ``implicit_main`` is
    ``1 + implicit_diag_term``; in nD the diagonal is ``1`` plus one such term per axis, and a wall
    replaces only its own axis's term. Exposed because that is what an nD assembly consumes."""

    implicit_wall_diag_term: float
    """The same contribution at a wall on this axis, ``factor*theta*alpha``."""

    implicit_wall_main: float
    implicit_wall_off: float
    explicit_main: float
    explicit_off: float
    explicit_wall_main: float
    explicit_wall_off: float


def cn_alpha(dt: float, sigma: float, dx: float) -> float:
    """The diffusion number ``alpha = D dt / dx^2``, with ``D = sigma^2 / 2`` (#811, one owner)."""
    return diffusion_from_volatility(sigma) * dt / dx**2


def neumann_cn_stencil(
    alpha: float,
    *,
    treatment: WallTreatment,
    theta: float = 0.5,
) -> NeumannCNStencil:
    """Derive the stencil for one theta-step at the diffusion number ``alpha`` (see `cn_alpha`).

    ``treatment`` selects the wall row and nothing else; the interior is identical either way. It
    is required rather than defaulted: the defect this module was written for is six call sites
    each carrying a wall nobody had to state, and a default would let a seventh do the same. There
    WAS a seventh -- `adjoint.operators._build_1d_laplacian`, found while doing #2243, invisible to
    #2237's census because it holds the wall without holding ``alpha``; it now shares this module's
    `wall_factor`. See the module docstring for what each treatment conserves, measured. Callers
    holding ``alpha`` already -- the ADI sweep does -- pass it straight in; nothing here needs
    ``dt``, ``sigma`` or ``dx``.
    """
    factor = wall_factor(treatment)
    implicit_off = -theta * alpha
    explicit_off = (1.0 - theta) * alpha
    return NeumannCNStencil(
        alpha=alpha,
        theta=theta,
        factor=factor,
        implicit_main=1.0 + 2.0 * theta * alpha,
        implicit_off=implicit_off,
        implicit_diag_term=2.0 * theta * alpha,
        implicit_wall_diag_term=factor * theta * alpha,
        implicit_wall_main=1.0 + factor * theta * alpha,
        implicit_wall_off=factor * implicit_off,
        explicit_main=1.0 - 2.0 * explicit_off,
        explicit_off=explicit_off,
        explicit_wall_main=1.0 - factor * explicit_off,
        explicit_wall_off=factor * explicit_off,
    )


def neumann_cn_step(
    u: NDArray[np.floating],
    dt: float,
    sigma: float,
    dx: float,
    *,
    treatment: WallTreatment,
    theta: float = 0.5,
) -> NDArray[np.floating]:
    """One Crank-Nicolson diffusion step on ``[0, L]`` with a zero-flux wall at each end.

    Solves ``(I - theta*alpha*L) u^{n+1} = (I + (1-theta)*alpha*L) u^n`` with
    ``alpha = D dt / dx^2`` and ``D = sigma^2 / 2`` taken from the single owner of that conversion.

    ``treatment`` selects the wall row and nothing else; the interior is identical either way, and
    it has no default -- see `neumann_cn_stencil`. What each conserves, and the accuracy each
    costs, is in the module docstring, measured.
    """
    values = np.asarray(u, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"neumann_cn_step: u must be 1-D; got shape {values.shape}.")
    n = values.size
    if n < 3:
        raise ValueError(f"neumann_cn_step: need at least 3 nodes for a wall and an interior; got {n}.")

    st = neumann_cn_stencil(cn_alpha(dt, sigma, dx), treatment=treatment, theta=theta)

    ab = np.zeros((3, n))
    ab[0, 1:] = st.implicit_off
    ab[1, :] = st.implicit_main
    ab[2, :-1] = st.implicit_off

    rhs = np.zeros(n)
    rhs[1:-1] = st.explicit_main * values[1:-1] + st.explicit_off * (values[:-2] + values[2:])

    # The wall rows, and the only place `treatment` acts.
    rhs[0] = st.explicit_wall_main * values[0] + st.explicit_wall_off * values[1]
    ab[1, 0] = st.implicit_wall_main
    ab[0, 1] = st.implicit_wall_off

    rhs[-1] = st.explicit_wall_main * values[-1] + st.explicit_wall_off * values[-2]
    ab[1, -1] = st.implicit_wall_main
    ab[2, -2] = st.implicit_wall_off

    return solve_banded((1, 1), ab, rhs)
