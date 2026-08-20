"""Issue #2019: `source_term` had two incompatible calling conventions across the FP solvers.

`FPFVMSolver` passed `geometry.meshgrid()` — a `(d, *shape)` tuple — and accepted only a
grid-shaped return. `FPFDMSolver` and the whole HJB side pass `geometry.get_spatial_grid()`, an
`(N, d)` point array, and ravel/reshape the return. So one callback could not serve both: a source
written against the convention `BaseHJBSolver.solve_hjb_system` documents ("x has shape (N, d)")
returned an `(N,)` array to FVM and hit

    ValueError: operands could not be broadcast together with shapes (21,21) (882,)

where 882 = 2 x 441 is the two coordinate planes flattened together.

WHY IT SURVIVED, AND WHY THE TESTS BELOW ARE 2-D
------------------------------------------------
In 1-D the two conventions coincide: `meshgrid()` gives one `(21,)` array and `get_spatial_grid()`
gives `(21, 1)`, so a callback that ravels its input accepts both and the fork is invisible. It is
expressible only in `d >= 2` — the rule `AGENTS.md` states as "the dimension must be able to express
the property under test".
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

_L, _T, _SIGMA = 1.0, 0.2, 0.4
_D = 0.5 * _SIGMA**2


def _problem(dim, n, nt):
    grid = TensorProductGrid(
        bounds=[(0.0, _L)] * dim, Nx_points=[n] * dim, boundary_conditions=no_flux_bc(dimension=dim)
    )
    return MFGProblem(
        geometry=grid,
        T=_T,
        Nt=nt,
        sigma=_SIGMA,
        coupling_coefficient=0.0,
        components=MFGComponents(
            # Per-point callables returning a float: what the 2-D validator accepts.
            m_initial=lambda p: 1.0,
            u_terminal=lambda p: 0.0,
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0),
                coupling=lambda m: np.asarray(m) * 0.0,
                coupling_dm=lambda m: np.asarray(m) * 0.0,
            ),
        ),
    )


def _solvers():
    from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
    from mfgarchon.alg.numerical.fp_solvers.fp_fvm import FPFVMSolver

    return (("FPFDMSolver", FPFDMSolver), ("FPFVMSolver", FPFVMSolver))


@pytest.mark.parametrize(("name", "cls"), _solvers(), ids=[n for n, _ in _solvers()])
def test_the_documented_point_array_convention_works_in_2d(name, cls):
    """One callback, both solvers, `d = 2`. This is the assertion #2019 is about."""
    n = 11
    problem = _problem(2, n, 4)
    solver = cls(problem)
    seen: dict[str, tuple] = {}

    def source(_t, x):
        a = np.asarray(x, dtype=float)
        seen["shape"] = a.shape
        # The documented convention: (N, d) in, one value per point out.
        return np.full(a.shape[0], 1.0)

    m0 = np.ones(n * n) / (n * n)
    out = np.asarray(solver.solve_fp_system(m0.reshape(n, n), potential_field=None, source_term=source))
    assert seen.get("shape") == (n * n, 2), (
        f"{name} called the source with shape {seen.get('shape')}, not the documented (N, d). "
        f"A (d, *shape) meshgrid is the #2019 fork."
    )
    assert np.all(np.isfinite(out))


@pytest.mark.parametrize(("name", "cls"), _solvers(), ids=[n for n, _ in _solvers()])
def test_the_source_actually_moves_the_answer(name, cls):
    """Accepting the convention is not the same as using what it delivers. Constant source S over
    time T adds S*T to the total, which is checkable in closed form."""
    n, nt = 11, 8
    problem = _problem(2, n, nt)
    solver = cls(problem)
    m0 = (np.ones(n * n) / (n * n)).reshape(n, n)

    base = np.asarray(solver.solve_fp_system(m0.copy(), potential_field=None))
    forced = np.asarray(
        solver.solve_fp_system(
            m0.copy(), potential_field=None, source_term=lambda _t, x: np.full(np.asarray(x).shape[0], 2.0)
        )
    )
    delta = float(np.abs(forced[-1] - base[-1]).max())
    assert delta == pytest.approx(2.0 * _T, rel=0.25), (
        f"{name}: a constant source of 2.0 over T={_T} should raise the density by ~{2.0 * _T}; measured {delta:.4e}"
    )


def test_the_fork_is_invisible_in_1d_which_is_why_it_survived():
    """Recorded rather than asserted about the library: in 1-D both conventions produce an input a
    raveling callback accepts, so no 1-D test could have caught this."""
    grid1 = TensorProductGrid(bounds=[(0.0, _L)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1))
    grid2 = TensorProductGrid(bounds=[(0.0, _L)] * 2, Nx_points=[21] * 2, boundary_conditions=no_flux_bc(dimension=2))
    assert np.asarray(grid1.get_spatial_grid()).ravel().size == np.asarray(grid1.meshgrid()).ravel().size
    assert np.asarray(grid2.get_spatial_grid()).ravel().size == np.asarray(grid2.meshgrid()).ravel().size
    # Same total size in both dimensions -- so size alone never discriminates. The SHAPE does, and
    # only in 2-D does a raveling callback produce a length the other convention rejects.
    assert np.asarray(grid1.get_spatial_grid()).shape[0] == 21
    assert np.asarray(grid2.get_spatial_grid()).shape[0] == 441
    assert np.asarray(grid2.meshgrid()).shape == (2, 21, 21)
