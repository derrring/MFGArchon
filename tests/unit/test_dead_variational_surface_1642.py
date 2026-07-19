"""
Structural pins for Issue #1642 Stage-1 unit B (A5 + D1): narrow declared-but-unhonored
surface around the experimental variational stack and the functional-calculus config.

These are surface pins, not numerical tests. Each one fails if the removed declaration
comes back:

- ``test_no_ghost_variational_module_reference`` fails if any source file under
  ``mfgarchon/`` mentions the never-existing ``mfgarchon.solvers.variational`` again.
- ``test_ghost_package_is_not_importable`` fails if someone creates that package to
  satisfy an import instead of removing the import.
- ``test_empty_optimization_packages_are_gone`` fails if the three zero-byte solver
  packages are restored.
- ``test_functional_derivative_config_declares_no_autodiff_backend`` fails if the
  JAX/PyTorch selector (or either deprecated boolean alias) is re-added to
  ``FunctionalDerivativeConfig``, which has no autodiff implementation behind it.
- ``test_functional_calculus_does_not_import_autodiff_backend`` fails if the module
  re-acquires the unused ``AutoDiffBackend`` dependency.

Refs #1642 (capabilities A5, D1), #1342.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
from pathlib import Path

import pytest

import mfgarchon
from mfgarchon.utils.functional_calculus import FunctionalDerivativeConfig

GHOST_MODULE = "mfgarchon.solvers.variational"

PACKAGE_ROOT = Path(mfgarchon.__file__).resolve().parent

# Zero-byte packages removed by D1: each contained only an empty __init__.py.
REMOVED_OPTIMIZATION_PACKAGES = (
    "alg/optimization/augmented_lagrangian",
    "alg/optimization/primal_dual",
    "alg/optimization/variational_methods",
)

# Autodiff selector removed by A5, plus the deprecated boolean aliases that only
# existed to write into it.
REMOVED_CONFIG_FIELDS = ("backend", "use_jax", "use_pytorch")


def test_no_ghost_variational_module_reference() -> None:
    """No source file under mfgarchon/ may reference the nonexistent variational module."""
    offenders = [
        str(path.relative_to(PACKAGE_ROOT))
        for path in sorted(PACKAGE_ROOT.rglob("*.py"))
        if GHOST_MODULE in path.read_text(encoding="utf-8")
    ]
    assert offenders == [], (
        f"{GHOST_MODULE} does not exist; referencing it turns into a ModuleNotFoundError "
        f"instead of a diagnostic. Offending files: {offenders}"
    )


def test_ghost_package_is_not_importable() -> None:
    """The ghost module must stay absent -- the fix is removing the import, not adding a stub."""
    assert importlib.util.find_spec("mfgarchon.solvers") is None, (
        "mfgarchon.solvers was created. D1 narrows the dead reference away; it does not "
        "sanction resurrecting the package to satisfy a stale import."
    )


@pytest.mark.parametrize("relative_path", REMOVED_OPTIMIZATION_PACKAGES)
def test_empty_optimization_packages_are_gone(relative_path: str) -> None:
    """The three zero-byte optimization solver packages must not come back.

    Pins the absence of Python modules, not of the directory. Git leaves a directory
    behind when it still holds untracked content, so any working copy that imported a
    pre-#1642 commit keeps an empty ``<package>/__pycache__/`` after checking this
    branch out -- with a clean ``git status``. Asserting ``package_dir.exists()`` would
    therefore fail on the maintainer's tree while CI's fresh clone stayed green.

    Globbing ``*.py`` rather than checking ``__init__.py`` alone: without an
    ``__init__.py`` the directory is still a PEP 420 namespace package, so a bare
    ``primal_dual/solver.py`` imports fine on an editable install (verified) while
    ``setuptools.find_packages()`` omits it from a wheel -- resurrection that works for
    the author and breaks for every installed user.
    """
    package_dir = PACKAGE_ROOT / relative_path
    modules = sorted(path.name for path in package_dir.glob("*.py"))
    assert modules == [], (
        f"{relative_path} carries Python modules again: {modules}. That directory is an "
        f"empty package advertising a solver family that does not exist (Issue #1342); "
        f"#1642 capability D1 removed it. Add the solver implementation somewhere real, "
        f"not the empty package. (A leftover __pycache__/ alone is not a failure.)"
    )


@pytest.mark.parametrize("field_name", REMOVED_CONFIG_FIELDS)
def test_functional_derivative_config_declares_no_autodiff_backend(field_name: str) -> None:
    """FunctionalDerivativeConfig must not advertise a JAX/PyTorch selector it cannot honor."""
    declared = {field.name for field in dataclasses.fields(FunctionalDerivativeConfig)}
    assert field_name not in declared, (
        f"FunctionalDerivativeConfig.{field_name} promises autodiff that "
        f"mfgarchon/utils/functional_calculus.py does not implement -- the module ships "
        f"finite-difference and particle derivatives only. If autodiff lands (#1642 "
        f"capability E2), add the selector together with the code that reads it."
    )


def test_functional_calculus_does_not_import_autodiff_backend() -> None:
    """The autodiff enum has no consumer left in functional_calculus."""
    module = importlib.import_module("mfgarchon.utils.functional_calculus")
    assert getattr(module, "AutoDiffBackend", None) is None, (
        "functional_calculus re-imported AutoDiffBackend without a reader; that is the "
        "dead-declaration pattern #1642 capability A5 removed."
    )
