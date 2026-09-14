"""PDE reinitialisation upwinds by the sign of the initial level set (#2310).

``phi_tau + S (|grad phi| - 1) = 0`` with ``S = sign(phi_0)``. The Godunov magnitude depends on the sign
of ``S``; `reinitialize(method="pde")` used the ``S > 0`` form everywhere, which is downwind where
``phi_0 < 0``. External oracle: a signed distance function already satisfies ``|grad phi| = 1``, so it is
a fixed point of the evolution at any iteration count, and its zero level set stays where it is.

Measured at 6c0610d2, before this fix: the exact ``x - 0.5`` drifted by 2.99e+01 after 100 iterations,
and ``|x - 0.5| - 0.25`` (a ``phi_0 < 0`` minimum on a node) moved by 3.73e-02 on the default call.

The sign selection is per axis, so each axis gets a plane of its own on a 2-D grid with unequal node
counts: a selection wrong on axis 1 only leaves ``x - 0.5`` fixed. The Dirichlet case pins that the
``S < 0`` form negates the padded array rather than padding ``-phi``, which would impose the wall value
``-0.25`` on ``-phi`` instead of on ``phi``.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.geometry import TensorProductGrid, dirichlet_bc, no_flux_bc
from mfgarchon.geometry.level_set.reinitialization import reinitialize

_LINE = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[101], boundary_conditions=no_flux_bc(dimension=1))
_X = _LINE.coordinates[0]
_PLANE = TensorProductGrid(
    bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[41, 31], boundary_conditions=no_flux_bc(dimension=2)
)
_PX, _PY = np.meshgrid(*_PLANE.coordinates, indexing="ij")
_WALLED = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[101], boundary_conditions=dirichlet_bc(-0.25, dimension=1))
_LONG_RUN = {"max_iterations": 400, "tolerance": 0.0}


@pytest.mark.parametrize(
    ("label", "grid", "phi", "kwargs"),
    [
        ("linear_distance_long_run", _LINE, _X - 0.5, _LONG_RUN),
        ("negative_minimum_default_call", _LINE, np.abs(_X - 0.5) - 0.25, {}),
        ("plane_across_axis_0", _PLANE, _PX - 0.5, _LONG_RUN),
        ("plane_across_axis_1", _PLANE, _PY - 0.5, _LONG_RUN),
        ("negative_walls_under_dirichlet", _WALLED, 0.25 - np.abs(_X - 0.5), _LONG_RUN),
    ],
)
def test_a_signed_distance_function_is_a_fixed_point(label: str, grid, phi: np.ndarray, kwargs: dict):
    result = reinitialize(phi.copy(), grid, method="pde", **kwargs)
    interior = tuple(slice(5, -5) for _ in range(phi.ndim))
    drift = float(np.abs(result - phi)[interior].max())
    assert drift < 1e-12, f"{label}: an exact signed distance moved by {drift:.3e} (#2310)"
