"""The FEM branch of `get_applicator_for_geometry` refuses and names the import to use instead; that import runs.

The message used to advise `from mfgarchon.geometry.boundary.bc_adapter import apply_fem_bc`: neither the module
nor the function exists, so following it raised `ModuleNotFoundError`. The advice is executed here, and the test
asserts what it imports, not merely that it imports.
"""

from __future__ import annotations

import inspect
import re

import pytest

from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.geometry.boundary.dispatch import get_applicator_for_geometry
from mfgarchon.geometry.boundary.protocols import DiscretizationType


def test_the_fem_refusal_names_an_import_that_runs_and_gives_the_fem_bc_adapter():
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[5], boundary_conditions=no_flux_bc(dimension=1))
    with pytest.raises(NotImplementedError) as refusal:
        get_applicator_for_geometry(grid, DiscretizationType.FEM)

    advice = re.search(r"Use: (from \S+ import (\w+))", str(refusal.value))
    assert advice, f"the refusal names no import: {refusal.value}"
    namespace: dict = {}
    exec(advice.group(1), namespace)

    from mfgarchon.alg.numerical.fem import bc_adapter

    imported = namespace[advice.group(2)]
    assert imported is bc_adapter.apply_bc_to_fem_system
    assert list(inspect.signature(imported).parameters)[:4] == ["A", "rhs", "basis", "bc"]
