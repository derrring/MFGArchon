"""Under Engquist-Osher the FDM HJB linearisation is the transpose of the FDM FP operator, local maxima included (#2313).

The default FDM pair, `HJBFDMSolver` with `FPFDMSolver(advection_scheme="divergence_upwind")`, is advertised as
discretely adjoint (#580, #622), and `build_linearized_operator` is what the strict-adjoint coupling hands the FP side
as ``A_fp = J^T`` (#707). At 6c0610d2 the HJB took the Rouy-Tourin momentum, whose linearisation at a local maximum
uses one one-sided difference while the FP face upwinding pairs with their sum: max|A_fp - J^T| was 17.0 there in 1-D
and 0 elsewhere. Engquist-Osher, the default since #2313, is ACD's Example 1 form, whose linearisation is that sum.

Oracle: ``A_fp`` is the library's own FP assembly (`add_interior_entries_divergence_upwind`, dt -> infinity,
sigma = 0) and ``J`` the HJB solver's own `build_linearized_operator`; neither is written here. The states have local
maxima along every axis, and Rouy-Tourin is the control that shows the fixture reaches them.
"""

from __future__ import annotations

import warnings

import pytest

import numpy as np

from mfgarchon import MFGProblem
from mfgarchon.alg.numerical.fp_solvers.fp_fdm_alg_divergence_upwind import add_interior_entries_divergence_upwind
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def _transpose_gap(shape: tuple[int, ...], numerical_hamiltonian: str) -> float:
    dimension = len(shape)
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0)] * dimension, Nx_points=list(shape), boundary_conditions=no_flux_bc(dimension=dimension)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        problem = MFGProblem(
            geometry=grid,
            Nt=10,
            T=1.0,
            sigma=0.0,
            components=MFGComponents(
                m_initial=lambda x: 1.0,
                u_terminal=lambda x: 0.0,
                hamiltonian=SeparableHamiltonian(
                    control_cost=QuadraticControlCost(control_cost=1.0), coupling=lambda m: m, coupling_dm=lambda m: 1.0
                ),
            ),
        )
    x = np.meshgrid(*grid.coordinates, indexing="ij")
    if dimension == 1:
        u = 0.3 * np.cos(2 * np.pi * x[0]) + 0.2 * np.cos(6 * np.pi * x[0] + 0.4)
    else:
        u = (
            0.3 * np.cos(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1] + 0.4)
            + 0.2 * np.sin(6 * np.pi * x[0] + 0.3)
            + 0.15 * np.cos(4 * np.pi * x[1])
        )
    total = int(np.prod(shape))
    spacing = tuple(grid.get_grid_spacing())

    def indices(margin: int) -> list[int]:
        return [
            k
            for k in range(total)
            if all(margin <= i < n - margin for i, n in zip(np.unravel_index(k, shape), shape, strict=True))
        ]

    interior, deep = indices(1), indices(2)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for k in interior:
        multi = tuple(int(i) for i in np.unravel_index(k, shape))
        add_interior_entries_divergence_upwind(
            rows, cols, vals, k, multi, shape, dimension, 1e300, 0.0, 1.0, spacing, u.ravel(), grid,
            no_flux_bc(dimension=dimension),
        )  # fmt: skip
    fp = np.zeros((total, total))
    np.add.at(fp, (rows, cols), vals)
    hjb = HJBFDMSolver(problem, numerical_hamiltonian=numerical_hamiltonian).build_linearized_operator(
        u, np.ones(shape)
    )
    return float(np.abs(fp - hjb.toarray().T)[np.ix_(interior, deep)].max())


@pytest.mark.parametrize("shape", [(41,), (17, 15)], ids=["1d", "2d"])
def test_engquist_osher_linearises_to_the_transpose_of_the_fp_operator(shape):
    """Measured at the #2313 branch: 2.8e-14 (1-D) and 4.4e-14 (2-D), against 17.0 and 27.0 for Rouy-Tourin."""
    assert _transpose_gap(shape, "engquist_osher") < 1e-10
    assert _transpose_gap(shape, "rouy_tourin") > 1.0, "no local maximum reaches the operator; nothing is tested"
