"""PDE reinitialisation upwinds by the sign of the initial level set (#2310).

``phi_tau + S (|grad phi| - 1) = 0`` with ``S = sign(phi_0)``. The Godunov magnitude depends on the sign
of ``S``; `reinitialize(method="pde")` used the ``S > 0`` form everywhere, which is downwind where
``phi_0 < 0``. External oracle: a signed distance function already satisfies ``|grad phi| = 1``, so it is
a fixed point of the evolution at any iteration count, and its zero level set stays where it is.

Measured by #2310's reproducer at e1517262, before this fix: the exact ``x - 0.5`` drifted to 1.10e+01 after 100
iterations, and ``|x - 0.5| - 0.25`` (a ``phi_0 < 0`` minimum on a node) moved by 3.73e-02 on the
default call. After #2310 both stay at rounding.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.geometry import TensorProductGrid, no_flux_bc
from mfgarchon.geometry.level_set.reinitialization import reinitialize

_GRID = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[101], boundary_conditions=no_flux_bc(dimension=1))
_X = _GRID.coordinates[0]


@pytest.mark.parametrize(
    ("label", "phi", "kwargs"),
    [
        ("linear_distance_long_run", _X - 0.5, {"max_iterations": 400, "tolerance": 0.0}),
        ("negative_minimum_default_call", np.abs(_X - 0.5) - 0.25, {}),
    ],
)
def test_a_signed_distance_function_is_a_fixed_point(label: str, phi: np.ndarray, kwargs: dict):
    result = reinitialize(phi.copy(), _GRID, method="pde", **kwargs)
    drift = float(np.abs(result - phi)[5:-5].max())
    assert drift < 1e-12, f"{label}: an exact signed distance moved by {drift:.3e} (#2310)"
