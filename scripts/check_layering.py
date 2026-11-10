#!/usr/bin/env python3
"""A lower layer of mfgarchon does not import an upper one (#2938).

Usage:
    python scripts/check_layering.py [--root TREE]   # check TREE/mfgarchon (default: this repository)
    python scripts/check_layering.py --self-test     # the parser and mechanism cases, on temporary packages

**It observes.** Neither CI nor the pre-push gate runs the check of the package. The gate runs only
``--self-test``, in ``scripts/local_ci.sh``'s self-test loop: that guards the parser and enforces nothing
about the package. When the check itself enters CI is the user's call (#2938).

This file is the one owner of the layer table (``LAYERS``) and of the exception lists (``EDGE_EXCEPTIONS``,
``UNRESOLVED_EXCEPTIONS``, ``DYNAMIC_SITES``). A PR that removes an edge removes its line in the same diff.

**What is counted** (#2938, "What is counted"):

- An edge is (importing module, upper target module, position). Position is one of:
  - ``module``: the import stands outside any function, including the ``else`` of ``if TYPE_CHECKING:``,
    the body of ``if not TYPE_CHECKING:``, ``if __name__ == "__main__":``, class bodies, ``try``/``except``,
    ``with`` and ``match``. The body of ``if False:`` counts here too: the rule is about the direction of
    source dependencies, not about what runs when a module is loaded.
  - ``function``: inside a ``def`` or a ``lambda``;
  - ``type``: the body of ``if TYPE_CHECKING:``.
- An import has two targets, the module on its path and the owner of each name it binds. Either one in an
  upper layer is an edge, so a re-export of an upper name is an edge.
- Owners are followed in a static scope only:
  - module-level statements are followed, not function or class bodies;
  - the last module-level binding of a name wins;
  - a last binding inside ``if``, ``try`` or ``with`` is decided at run time and is unresolved (ambiguous);
  - ``from x import *`` exports ``x``'s static ``__all__`` (unconditional, a list or tuple of string
    literals), or, without one, the names with no leading underscore;
  - an ``__all__`` changed under a condition or by a method call is ambiguous;
  - a name whose binding is not found is unresolved (not found);
  - an unresolved owner is never taken to be the importing module.
- The path of a star import is an edge of its own, whatever the names expand to.

**What it reports.** Each report names a location, and none prints a total:

- an upward edge outside ``EDGE_EXCEPTIONS``, at the importer's file and line;
- a line of ``EDGE_EXCEPTIONS`` whose edge no longer exists;
- an unresolved owner outside ``UNRESOLVED_EXCEPTIONS``, at the importer's file and line;
- a line of ``UNRESOLVED_EXCEPTIONS`` whose import no longer exists;
- a dynamic import site outside ``DYNAMIC_SITES``, at its file and line;
- a line of ``DYNAMIC_SITES`` whose site no longer exists;
- a ``LAYERS`` prefix that matches no module;
- a module that no ``LAYERS`` prefix places.

It exits 1 if it reports anything.

**What a static check cannot see:** ``importlib.import_module``, ``__import__`` and a module-level
``def __getattr__``. Their sites are listed in ``DYNAMIC_SITES``, so a new one is reported, but the
imports they make are not followed.

**A second owner of one slice.** ``tests/unit/test_utils_does_not_import_alg.py`` ratchets the module-level
imports of ``mfgarchon.alg`` from ``mfgarchon/utils/`` and runs in the gate. The two disagree on one edge.
That test counts ``utils/adjoint_validation.py`` -> ``alg.base_solver`` as the last violation, because the
file sits in ``utils/``. ``LAYERS`` places ``utils.adjoint_validation`` in the solve layer, beside ``alg``,
because of what it is. The ratchet is deleted in the PR that puts this check into the gate (#2938).
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

PACKAGE = "mfgarchon"

# Dotted prefix under `mfgarchon` -> layer; the longest prefix wins. "(root)" is the top-level package.
# `meta`, `extensions` and `hooks` are placed whole only so that edges can be counted (#2938).
LAYERS: dict[str, int] = {
    "types": 0,
    "config": 0,
    "backends": 0,
    "utils.exceptions": 0,
    "utils.mfg_logging": 0,
    "utils.dependencies": 0,
    "utils.numpy_compat": 0,
    "utils.progress": 0,
    "utils.memory_management": 0,
    "utils.solver_result": 0,
    "utils": 1,
    "utils.numerical": 1,
    "utils.acceleration": 1,
    "geometry": 2,
    "utils.geometry": 2,
    "utils.particle_utils": 2,
    "operators": 3,
    "utils.numerical.tensor_calculus": 3,
    "utils.numerical.implicit_diffusion": 3,
    "core": 4,
    "extensions": 4,
    "meta": 4,
    "utils.pde_coefficients": 4,
    "utils.drift_helpers": 4,
    "utils.manufactured": 4,
    "utils.validation": 4,
    "alg": 5,
    "hooks": 5,
    "backends.strategies": 5,
    "utils.numerical.hjb_policy_iteration": 5,
    "utils.adjoint_validation": 5,
    "utils.solver_decorators": 5,
    "utils.numerical.flux_diagnostics": 5,
    "utils.numerical.mass_fabrication_gate": 5,
    "factory": 6,
    "workflow": 6,
    "visualization": 6,
    "utils.cli": 6,
    "utils.experiment_manager": 6,
    "(root)": 6,
}
LAYER_NAMES = {0: "base", 1: "numerics", 2: "geometry", 3: "operators", 4: "model", 5: "solve", 6: "entry"}

# (importing module, upper target module, position, the stage that removes it, reason). Stages are #2938's.
EDGE_EXCEPTIONS: tuple[tuple[str, str, str, str, str], ...] = (
    # base -> numerics
    ("mfgarchon.backends.jax_backend", "mfgarchon.utils.acceleration.jax_precision", "function", "#2938 stage 2b", "the jax backend sets the x64 policy through an acceleration helper"),
    ("mfgarchon.backends.numpy_backend", "mfgarchon.utils.numerical.integration", "module", "#2938 stage 2b", "imports trapezoid through a layer-1 path; its owner is utils.numpy_compat"),
    ("mfgarchon.config.array_validation", "mfgarchon.utils.numerical.integration", "module", "#2938 stage 2b", "imports trapezoid through a layer-1 path; its owner is utils.numpy_compat"),
    ("mfgarchon.utils.solver_result", "mfgarchon.utils.io.hdf5_utils", "function", "#2938 stage 2b", "the result's save method calls the HDF5 writer"),
    ("mfgarchon.utils.solver_result", "mfgarchon.utils.notebooks.reporting", "function", "#2938 stage 2b", "the result's report method calls the notebook reporter"),
    # numerics -> geometry
    ("mfgarchon.utils", "mfgarchon.geometry.implicit.hyperrectangle", "module", "#2938 stage 2a", "utils/__init__ re-exports a geometry name"),
    ("mfgarchon.utils", "mfgarchon.geometry.implicit.hypersphere", "module", "#2938 stage 2a", "utils/__init__ re-exports a geometry name"),
    ("mfgarchon.utils", "mfgarchon.geometry.implicit.implicit_domain", "module", "#2938 stage 2a", "utils/__init__ re-exports a geometry name"),
    ("mfgarchon.utils", "mfgarchon.geometry.implicit.sdf_utils", "module", "#2938 stage 2a", "utils/__init__ re-exports a geometry name"),
    ("mfgarchon.utils", "mfgarchon.utils.geometry", "module", "#2938 stage 2a", "utils/__init__ re-exports a geometry name"),
    ("mfgarchon.utils.numerical", "mfgarchon.geometry.implicit.sdf_utils", "module", "#2938 stage 2a", "utils/numerical/__init__ re-exports a geometry name"),
    ("mfgarchon.utils.numerical", "mfgarchon.geometry.meshes.mesh_distances", "module", "#2938 stage 2a", "utils/numerical/__init__ re-exports a geometry name"),
    ("mfgarchon.utils.numerical.particle.sampling", "mfgarchon.geometry.boundary.periodic", "function", "#2938 stage 2a", "sampling wraps positions with the periodic boundary's helpers"),
    ("mfgarchon.utils.sparse_operations", "mfgarchon.geometry.grids.tensor_grid", "type", "#2938 stage 2a", "an annotation names TensorProductGrid"),
    # numerics -> model
    ("mfgarchon.utils", "mfgarchon.utils.drift_helpers", "module", "#2938 stage 2a", "utils/__init__ re-exports a model-layer name"),
    ("mfgarchon.utils", "mfgarchon.utils.pde_coefficients", "module", "#2938 stage 2a", "utils/__init__ re-exports a model-layer name"),
    # numerics -> solve
    ("mfgarchon.utils", "mfgarchon.utils.adjoint_validation", "module", "#2938 stage 2a", "utils/__init__ re-exports a solve-layer name"),
    ("mfgarchon.utils.numerical", "mfgarchon.utils.numerical.flux_diagnostics", "module", "#2938 stage 2a", "utils/numerical/__init__ re-exports a solve-layer name"),
    ("mfgarchon.utils.numerical", "mfgarchon.utils.numerical.hjb_policy_iteration", "module", "#2938 stage 2a", "utils/numerical/__init__ re-exports a solve-layer name"),
    ("mfgarchon.utils.numerical", "mfgarchon.utils.numerical.mass_fabrication_gate", "module", "#2938 stage 2a", "utils/numerical/__init__ re-exports a solve-layer name"),
    ("mfgarchon.utils.numerical.particle.sampling", "mfgarchon.utils.numerical.mass_fabrication_gate", "function", "#2938 stage 2a", "sampling refuses a fabricated mass through the solve-layer gate"),
    # numerics -> entry
    ("mfgarchon.utils", "mfgarchon.utils.cli", "module", "#2938 stage 2a", "utils/__init__ re-exports an entry-layer name"),
    ("mfgarchon.utils", "mfgarchon.utils.experiment_manager", "module", "#2938 stage 2a", "utils/__init__ re-exports an entry-layer name"),
    ("mfgarchon.utils.data.recorder", "mfgarchon", "function", "#2938 stage 2a", "the recorder reads the package's __version__"),
    # geometry -> operators
    ("mfgarchon.geometry", "mfgarchon.operators", "module", "#2938 stage 3", "geometry/__init__ re-exports GeometryProjector and ProjectionRegistry"),
    ("mfgarchon.geometry", "mfgarchon.operators.interpolation.projection", "module", "#2938 stage 3", "geometry/__init__ re-exports GeometryProjector and ProjectionRegistry"),
    ("mfgarchon.geometry.boundary.types", "mfgarchon.operators.differential.function_gradient", "function", "#2938 stage 3", "a boundary type differentiates a function with an operator"),
    ("mfgarchon.geometry.collocation", "mfgarchon.operators.differential.function_gradient", "function", "#2938 stage 3", "collocation differentiates a function with an operator"),
    ("mfgarchon.geometry.grids.tensor_grid", "mfgarchon.operators", "function", "#2938 stage 3", "the grid builds its own operators"),
    ("mfgarchon.geometry.grids.tensor_grid", "mfgarchon.operators.differential.advection", "function", "#2938 stage 3", "the grid builds its own operators"),
    ("mfgarchon.geometry.grids.tensor_grid", "mfgarchon.operators.differential.divergence", "function", "#2938 stage 3", "the grid builds its own operators"),
    ("mfgarchon.geometry.grids.tensor_grid", "mfgarchon.operators.differential.gradient", "function", "#2938 stage 3", "the grid builds its own operators"),
    ("mfgarchon.geometry.grids.tensor_grid", "mfgarchon.operators.differential.laplacian", "function", "#2938 stage 3", "the grid builds its own operators"),
    ("mfgarchon.geometry.grids.tensor_grid", "mfgarchon.operators.interpolation.interpolation", "function", "#2938 stage 3", "the grid builds its own operators"),
    ("mfgarchon.geometry.implicit.implicit_domain", "mfgarchon.operators.differential.function_gradient", "function", "#2938 stage 3", "an implicit domain differentiates its level-set function with an operator"),
    ("mfgarchon.geometry.level_set.core", "mfgarchon.operators.reconstruction.weno", "function", "#2938 stage 3", "level-set evolution uses an operator-layer stencil"),
    ("mfgarchon.geometry.level_set.core", "mfgarchon.operators.stencils.finite_difference", "function", "#2938 stage 3", "level-set evolution uses an operator-layer stencil"),
    ("mfgarchon.geometry.level_set.reinitialization", "mfgarchon.operators.stencils.finite_difference", "function", "#2938 stage 3", "reinitialisation uses an operator-layer stencil"),
    # geometry -> solve
    ("mfgarchon.geometry.base", "mfgarchon.alg.numerical.fem.assembly", "function", "#2938 stage 2c", "a geometry method reaches FEM assembly in the solve layer"),
    ("mfgarchon.geometry.base", "mfgarchon.alg.numerical.fem.mesh_adapter", "function", "#2938 stage 2c", "a geometry method reaches the FEM mesh adapter in the solve layer"),
    # operators -> solve
    ("mfgarchon.operators.interpolation.projection", "mfgarchon.alg.numerical.density_estimation", "function", "#2938 stage 2c", "projection estimates a density with the solve layer's KDE"),
    # operators -> model
    ("mfgarchon.operators.differential.diffusion", "mfgarchon.utils.pde_coefficients", "function", "#2938 stage 2d", "the diffusion operator reads model-layer coefficient helpers"),
    ("mfgarchon.operators.interaction.convolution", "mfgarchon.core.mfg_problem", "type", "#2938 stage 2d", "an annotation names MFGProblem: building an operator from a problem is an adapter"),
    ("mfgarchon.operators.interaction.coupling_costs", "mfgarchon.core.mfg_problem", "type", "#2938 stage 2d", "coupling_costs is the model's cost specification"),
    ("mfgarchon.operators.interaction.coupling_costs", "mfgarchon.core.model", "function", "#2938 stage 2d", "coupling_costs is the model's cost specification"),
    ("mfgarchon.operators.interaction.coupling_costs", "mfgarchon.core.model", "type", "#2938 stage 2d", "coupling_costs is the model's cost specification"),
    ("mfgarchon.utils.numerical.implicit_diffusion", "mfgarchon.utils.pde_coefficients", "module", "#2938 stage 2d", "the implicit diffusion step reads model-layer coefficient helpers"),
    # model -> solve
    ("mfgarchon.core.measure", "mfgarchon.utils.numerical.mass_fabrication_gate", "module", "#2938 stage 4", "the measure clips through the solve-layer mass gate"),
    ("mfgarchon.core.mfg_problem", "mfgarchon.alg.numerical.coupling", "function", "#2938 stage 4", "problem.solve() orchestrates the solve"),
    ("mfgarchon.core.mfg_problem", "mfgarchon.alg.numerical.coupling.fixed_point_iterator", "function", "#2938 stage 4", "problem.solve() orchestrates the solve"),
    ("mfgarchon.core.mfg_problem", "mfgarchon.alg.numerical.coupling.source_composition", "type", "#2938 stage 4", "an annotation names CouplingBinding"),
    ("mfgarchon.core.mfg_problem", "mfgarchon.utils.adjoint_validation", "function", "#2938 stage 4", "problem.solve() orchestrates the solve"),
    # model -> entry
    ("mfgarchon.core.mfg_problem", "mfgarchon.factory", "function", "#2938 stage 4", "problem.solve() orchestrates the solve"),
    ("mfgarchon.core.mfg_problem", "mfgarchon.factory.scheme_factory", "function", "#2938 stage 4", "problem.solve() orchestrates the solve"),
    ("mfgarchon.core.plugin_system", "mfgarchon", "function", "#2938 stage 4", "plugin_system imports the package; stage 4 deletes the module"),
)  # fmt: skip

# (importing module, module imported from, name, position, the condition that binds the name or the loader that
# serves it). Each owner is decided at run time or served by `__getattr__`, so the check cannot place it.
UNRESOLVED_EXCEPTIONS: tuple[tuple[str, str, str, str, str], ...] = (
    ("mfgarchon.utils", "mfgarchon.utils.data.polars_integration", "POLARS_AVAILABLE", "module", "bound in a try on importing polars"),
    ("mfgarchon.utils.data", "mfgarchon.utils.data.polars_integration", "POLARS_AVAILABLE", "module", "bound in a try on importing polars"),
    ("mfgarchon.utils.acceleration", "mfgarchon.utils.acceleration.jax_utils", "HAS_JAX", "module", "bound in utils/acceleration/__init__'s try on importing jax"),
    ("mfgarchon.utils.acceleration", "mfgarchon.utils.acceleration.jax_utils", "HAS_GPU", "function", "bound in utils/acceleration/__init__'s try on importing jax"),
    ("mfgarchon.utils.acceleration", "mfgarchon.utils.acceleration.jax_utils", "HAS_JAX", "function", "bound in utils/acceleration/__init__'s try on importing jax"),
    ("mfgarchon.utils.acceleration", "mfgarchon.utils.acceleration.torch_utils", "HAS_CUDA", "type", "bound in torch_utils' try on importing torch"),
    ("mfgarchon.utils.acceleration", "mfgarchon.utils.acceleration.torch_utils", "HAS_MPS", "type", "bound in torch_utils' try on importing torch"),
    ("mfgarchon.utils.acceleration", "mfgarchon.utils.acceleration.torch_utils", "HAS_TORCH", "type", "bound in torch_utils' try on importing torch"),
    ("mfgarchon.utils.acceleration", "mfgarchon.utils.acceleration.torch_utils", "HAS_CUDA", "function", "bound in torch_utils' try on importing torch"),
    ("mfgarchon.utils.acceleration", "mfgarchon.utils.acceleration.torch_utils", "HAS_MPS", "function", "bound in torch_utils' try on importing torch"),
    ("mfgarchon.utils.acceleration", "mfgarchon.utils.acceleration.torch_utils", "HAS_TORCH", "function", "bound in torch_utils' try on importing torch"),
    ("mfgarchon.utils.acceleration.jax_utils", "mfgarchon.utils.acceleration", "DEFAULT_DEVICE", "module", "bound in utils/acceleration/__init__'s try on importing jax"),
    ("mfgarchon.utils.acceleration.jax_utils", "mfgarchon.utils.acceleration", "HAS_GPU", "module", "bound in utils/acceleration/__init__'s try on importing jax"),
    ("mfgarchon.utils.acceleration.jax_utils", "mfgarchon.utils.acceleration", "HAS_JAX", "module", "bound in utils/acceleration/__init__'s try on importing jax"),
    ("mfgarchon.utils.acceleration.jax_utils", "mfgarchon.types.solver_types", "JAXArray", "type", "bound in solver_types' try on importing jax"),
    ("mfgarchon.utils.numerical.nonlinear_solvers", "mfgarchon.utils.acceleration", "HAS_JAX", "module", "bound in utils/acceleration/__init__'s try on importing jax"),
    ("mfgarchon.utils.io", "mfgarchon.utils.io.hdf5_utils", "HDF5_AVAILABLE", "module", "bound in a try on importing h5py, a required package (#2930)"),
    ("mfgarchon.utils.data.recorder", "mfgarchon", "__version__", "function", "bound in a try on the installed metadata"),
    ("mfgarchon.geometry", "mfgarchon.geometry.boundary", "MixedBoundaryConditions", "module", "served by geometry/boundary/__init__'s __getattr__"),
    ("mfgarchon.geometry", "mfgarchon.geometry.meshes", "Mesh1D", "module", "served by geometry/meshes/__init__'s __getattr__"),
    ("mfgarchon.geometry", "mfgarchon.geometry.meshes", "Mesh2D", "module", "served by geometry/meshes/__init__'s __getattr__"),
    ("mfgarchon.geometry", "mfgarchon.geometry.meshes", "Mesh3D", "module", "served by geometry/meshes/__init__'s __getattr__"),
    ("mfgarchon.geometry", "mfgarchon.geometry.meshes", "MeshManager", "module", "served by geometry/meshes/__init__'s __getattr__"),
    ("mfgarchon.geometry", "mfgarchon.geometry.meshes", "MeshPipeline", "module", "served by geometry/meshes/__init__'s __getattr__"),
    ("mfgarchon.alg.optimization", "mfgarchon.alg.optimization.optimal_transport", "POT_AVAILABLE", "module", "bound in a try on importing ot"),
    ("mfgarchon.alg.optimization", "mfgarchon.alg.optimization.optimal_transport", "SCIPY_AVAILABLE", "module", "bound in a try on importing scipy, a required package (#2930)"),
    ("mfgarchon.alg.optimization", "mfgarchon.alg.optimization.optimal_transport", "SinkhornMFGSolver", "module", "bound under `if SCIPY_AVAILABLE and POT_AVAILABLE`"),
    ("mfgarchon.alg.optimization", "mfgarchon.alg.optimization.optimal_transport", "SinkhornSolverConfig", "module", "bound under `if SCIPY_AVAILABLE and POT_AVAILABLE`"),
    ("mfgarchon.alg.optimization", "mfgarchon.alg.optimization.optimal_transport", "SinkhornSolverResult", "module", "bound under `if SCIPY_AVAILABLE and POT_AVAILABLE`"),
    ("mfgarchon.alg.optimization", "mfgarchon.alg.optimization.optimal_transport", "WassersteinMFGSolver", "module", "bound under `if SCIPY_AVAILABLE and POT_AVAILABLE`"),
    ("mfgarchon.alg.optimization", "mfgarchon.alg.optimization.optimal_transport", "WassersteinSolverConfig", "module", "bound under `if SCIPY_AVAILABLE and POT_AVAILABLE`"),
    ("mfgarchon.alg.optimization", "mfgarchon.alg.optimization.optimal_transport", "WassersteinSolverResult", "module", "bound under `if SCIPY_AVAILABLE and POT_AVAILABLE`"),
)  # fmt: skip

# (module, mechanism): the sites whose imports a static check cannot follow, by module.
DYNAMIC_SITES: tuple[tuple[str, str], ...] = (
    ("mfgarchon.core.plugin_system", "import_module"),
    ("mfgarchon.geometry", "__import__"),
    ("mfgarchon.geometry.boundary", "__getattr__"),
    ("mfgarchon.geometry.boundary", "import_module"),
    ("mfgarchon.geometry.meshes", "__getattr__"),
    ("mfgarchon.utils.acceleration", "__getattr__"),
    ("mfgarchon.utils.acceleration", "import_module"),
    ("mfgarchon.utils.dependencies", "__import__"),
)  # fmt: skip

POSITIONS = ("module", "function", "type")


def make_layer_of(table: dict[str, int], package: str) -> Callable[[str], int | None]:
    """The layer of a dotted module name under `package`: the longest prefix in `table` wins."""

    def layer_of(module: str) -> int | None:
        parts = module.split(".")
        if parts[0] != package:
            return None
        if len(parts) == 1:
            return table.get("(root)")
        rest = parts[1:]
        for k in range(len(rest), 0, -1):
            key = ".".join(rest[:k])
            if key in table:
                return table[key]
        return None

    return layer_of


def is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def position_of(stack: list[tuple[ast.AST, str]]) -> str:
    """Position of an import, given the (parent node, field) pairs from the module down to it."""
    for node, fieldname in stack:
        if isinstance(node, ast.If) and fieldname == "body" and is_type_checking(node.test):
            return "type"
    if any(isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) for node, _ in stack):
        return "function"
    return "module"


AMBIGUOUS = object()
CYCLE = object()


class Tree:
    """Static owners of names in one package on disk."""

    def __init__(self, root: str, package: str = PACKAGE) -> None:
        self.root = root
        self.package = package
        self.cache: dict[str, tuple[ast.Module | None, bool]] = {}
        self.event_cache: dict[str, list[tuple[str, str, object, bool]]] = {}

    def path_of(self, module: str) -> tuple[str | None, bool]:
        base = os.path.join(self.root, *module.split("."))
        if os.path.isfile(os.path.join(base, "__init__.py")):
            return os.path.join(base, "__init__.py"), True
        if os.path.isfile(base + ".py"):
            return base + ".py", False
        return None, False

    def parsed(self, module: str) -> tuple[ast.Module | None, bool]:
        if module not in self.cache:
            path, is_package = self.path_of(module)
            tree = ast.parse(Path(path).read_text(encoding="utf-8")) if path else None
            self.cache[module] = (tree, is_package)
        return self.cache[module]

    @staticmethod
    def absolute(module: str, is_package: bool, node: ast.ImportFrom) -> str:
        """The absolute module an ImportFrom names, from the importing module and whether it is a package."""
        if not node.level:
            return node.module or ""
        base = module.split(".") if is_package else module.split(".")[:-1]
        base = base[: len(base) - (node.level - 1)]
        return ".".join(base + ([node.module] if node.module else []))

    def events(self, module: str) -> list[tuple[str, str, object, bool]]:
        """Module-level binding events in source order: (name, kind, payload, conditional).

        The body of `if TYPE_CHECKING:` binds nothing at run time and is skipped. Any other if/try/with marks
        a binding conditional. Function and class bodies are not entered."""
        if module in self.event_cache:
            return self.event_cache[module]
        tree, is_package = self.parsed(module)
        out: list[tuple[str, str, object, bool]] = []

        def walk(body: list[ast.stmt], conditional: bool, runtime: bool) -> None:
            for n in body:
                if isinstance(n, ast.If):
                    walk(n.body, True, runtime and not is_type_checking(n.test))
                    walk(n.orelse, True, runtime)
                    continue
                if isinstance(n, ast.Try | ast.TryStar | ast.With):
                    for name in ("body", "orelse", "finalbody"):
                        walk(getattr(n, name, []) or [], True, runtime)
                    for handler in getattr(n, "handlers", []):
                        walk(handler.body, True, runtime)
                    continue
                if not runtime:
                    continue
                if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                    out.append((n.name, "def", None, conditional))
                elif isinstance(n, ast.Assign):
                    for target in n.targets:
                        for x in ast.walk(target):
                            if isinstance(x, ast.Name):
                                out.append((x.id, "def", n, conditional))
                elif isinstance(n, ast.AnnAssign | ast.AugAssign) and isinstance(n.target, ast.Name):
                    if not (isinstance(n, ast.AnnAssign) and n.value is None):
                        out.append((n.target.id, "def", n, conditional))
                elif isinstance(n, ast.TypeAlias):
                    out.append((n.name.id, "def", None, conditional))
                elif isinstance(n, ast.ImportFrom):
                    source = self.absolute(module, is_package, n)
                    for alias in n.names:
                        if alias.name == "*":
                            out.append(("*", "star", source, conditional))
                        else:
                            out.append((alias.asname or alias.name, "import", (source, alias.name), conditional))
                elif isinstance(n, ast.Import):
                    for alias in n.names:
                        bound = alias.asname or alias.name.split(".")[0]
                        target = alias.name if alias.asname else alias.name.split(".")[0]
                        out.append((bound, "module", target, conditional))
                elif isinstance(n, ast.Expr) and isinstance(n.value, ast.Call):
                    func = n.value.func
                    if (
                        isinstance(func, ast.Attribute)
                        and isinstance(func.value, ast.Name)
                        and func.value.id == "__all__"
                    ):
                        out.append(("__all__", "allmut", n, conditional))

        if tree is not None:
            walk(tree.body, False, True)
        self.event_cache[module] = out
        return out

    def star_exports(self, module: str, depth: int = 0) -> set[str] | None:
        """The names `from module import *` binds, or None when that cannot be decided statically."""
        if depth > 16:
            return None
        events = self.events(module)
        alls = [e for e in events if e[0] == "__all__"]
        if alls:
            if any(e[3] for e in alls) or any(e[1] == "allmut" for e in alls):
                return None
            node = alls[-1][2]
            value = getattr(node, "value", None)
            if not isinstance(node, ast.Assign) or not isinstance(value, ast.List | ast.Tuple):
                return None
            if not all(isinstance(x, ast.Constant) and isinstance(x.value, str) for x in value.elts):
                return None
            return {x.value for x in value.elts}
        names = {e[0] for e in events if e[1] != "star" and not e[0].startswith("_")}
        for e in events:
            if e[1] == "star":
                sub = self.star_exports(e[2], depth + 1)
                if sub is None:
                    return None
                names |= sub
        return names

    def owner(self, module: str, name: str) -> object:
        """The module that defines `name` as reached through `module`.

        None if it is bound nowhere statically visible; AMBIGUOUS if its binding depends on a run-time
        condition or on a non-literal `__all__`. A re-export ring is followed to the binding that ends it: a
        binding that leads back to a (module, name) already on the path is skipped in favour of the binding
        before it, which is what the partially initialised module held when the ring ran. A ring no binding
        ends is ambiguous."""
        found = self._owner(module, name, 0, ())
        return AMBIGUOUS if found is CYCLE else found

    def _owner(self, module: str, name: str, depth: int, stack: tuple[tuple[str, str], ...]) -> object:
        if depth > 16:
            return AMBIGUOUS
        if (module, name) in stack:
            return CYCLE
        events = self.events(module)
        if self.path_of(module + "." + name)[0] and not any(e[0] == name for e in events):
            return module + "." + name
        hits = []
        for e in events:
            if e[0] == name:
                hits.append(e)
            elif e[1] == "star":
                exported = self.star_exports(e[2])
                if exported is None:
                    hits.append((name, "ambiguous", None, e[3]))
                elif name in exported:
                    hits.append((name, "star", e[2], e[3]))
        if not hits:
            return None
        inner = (*stack, (module, name))
        for last in reversed(hits):
            if last[3] or last[1] == "ambiguous":
                return AMBIGUOUS
            kind, payload = last[1], last[2]
            if kind == "def":
                return module
            if kind == "module":
                return payload
            if kind == "star":
                found = self._owner(payload, name, depth + 1, inner)
            else:
                source, original = payload
                if (source, original) == (module, name) and self.path_of(source + "." + original)[0]:
                    return source + "." + original  # `from . import submodule` in a package __init__
                if not source.startswith(self.package):
                    return source
                found = self._owner(source, original, depth + 1, inner)
            if found is not CYCLE:
                return found
        return CYCLE


@dataclass
class Scan:
    """What one walk of a package found. Locations are (file relative to the root, line)."""

    modules: dict[str, str] = field(default_factory=dict)  # module -> file
    edges: dict[tuple[str, str, str], list[tuple[str, int]]] = field(default_factory=dict)
    unresolved: dict[tuple[str, str, str, str], list[tuple[str, int, str]]] = field(default_factory=dict)
    dynamic: dict[tuple[str, str], list[tuple[str, int]]] = field(default_factory=dict)
    owners: dict[tuple[str, str], tuple[object, str]] = field(default_factory=dict)  # (importer, name) -> owner


def python_files(root: str, package: str) -> Iterator[tuple[str, str, bool]]:
    """(module, file relative to root, is a package __init__) for every .py file under root/package."""
    for directory, dirnames, filenames in os.walk(os.path.join(root, package)):
        dirnames.sort()
        for filename in sorted(filenames):
            if not filename.endswith(".py"):
                continue
            rel = os.path.relpath(os.path.join(directory, filename), root)
            dotted = rel[:-3].replace(os.sep, ".")
            module = dotted[: -len(".__init__")] if dotted.endswith(".__init__") else dotted
            yield module, rel, filename == "__init__.py"


def scan(root: str, package: str, layer_of: Callable[[str], int | None]) -> Scan:
    tree = Tree(root, package)
    result = Scan()
    for module, rel, is_package in python_files(root, package):
        scan_module(tree, result, layer_of, module, rel, is_package)
    return result


def scan_module(
    tree: Tree, result: Scan, layer_of: Callable[[str], int | None], module: str, rel: str, is_package: bool
) -> None:
    """Add one module's edges, unresolved owners and dynamic sites to `result`."""
    package = tree.package
    result.modules[module] = rel
    source_layer = layer_of(module)
    parsed = ast.parse(Path(tree.root, rel).read_text(encoding="utf-8"))
    for n in parsed.body:
        if isinstance(n, ast.FunctionDef) and n.name == "__getattr__":
            result.dynamic.setdefault((module, "__getattr__"), []).append((rel, n.lineno))

    def add_edge(target: str, position: str, line: int) -> None:
        if source_layer is None:
            return
        target_layer = layer_of(target)
        if target_layer is not None and target_layer > source_layer:
            result.edges.setdefault((module, target, position), []).append((rel, line))

    def add_from(node: ast.ImportFrom, position: str) -> None:
        source = Tree.absolute(module, is_package, node)
        if not (source == package or source.startswith(package + ".")):
            return
        add_edge(source, position, node.lineno)  # the path is a dependency of its own
        for alias in node.names:
            if alias.name == "*":
                exported = tree.star_exports(source)
                if exported is None:
                    key = (module, source, "*", position)
                    result.unresolved.setdefault(key, []).append((rel, node.lineno, "ambiguous"))
                    continue
                names = sorted(exported)
            else:
                names = [alias.name]
            for name in names:
                owner = tree.owner(source, name)
                result.owners[(module, name)] = (owner, position)
                if owner is None or owner is AMBIGUOUS:
                    kind = "ambiguous" if owner is AMBIGUOUS else "not found"
                    result.unresolved.setdefault((module, source, name, position), []).append((rel, node.lineno, kind))
                else:
                    add_edge(str(owner), position, node.lineno)

    def visit(node: ast.AST, stack: list[tuple[ast.AST, str]]) -> None:
        for fieldname, value in ast.iter_fields(node):
            for child in value if isinstance(value, list) else [value]:
                if not isinstance(child, ast.AST):
                    continue
                here = [*stack, (node, fieldname)]
                if isinstance(child, ast.Call):
                    func = child.func
                    called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                    if called in ("import_module", "__import__"):
                        result.dynamic.setdefault((module, called), []).append((rel, child.lineno))
                if isinstance(child, ast.Import):
                    for alias in child.names:
                        if alias.name == package or alias.name.startswith(package + "."):
                            add_edge(alias.name, position_of(here), child.lineno)
                elif isinstance(child, ast.ImportFrom):
                    add_from(child, position_of(here))
                visit(child, here)

    visit(parsed, [])


def check(
    root: str,
    package: str,
    table: dict[str, int],
    edge_exceptions: tuple[tuple[str, ...], ...],
    unresolved_exceptions: tuple[tuple[str, ...], ...],
    dynamic_sites: tuple[tuple[str, str], ...],
) -> list[tuple[str, tuple[str, ...], str]]:
    """Every report, as (kind, key, message), sorted."""
    layer_of = make_layer_of(table, package)
    found = scan(root, package, layer_of)
    reports: list[tuple[str, tuple[str, ...], str]] = []

    def name(layer: int | None) -> str:
        return LAYER_NAMES.get(layer, str(layer)) if layer is not None else "none"

    listed_edges = {e[:3] for e in edge_exceptions}
    for key, sites in sorted(found.edges.items()):
        if key not in listed_edges:
            importer, target, position = key
            where = ", ".join(f"{f}:{ln}" for f, ln in sites)
            reports.append(
                (
                    "edge",
                    key,
                    (
                        f"{where}: {importer} imports {target} ({position}), {name(layer_of(importer))} -> "
                        f"{name(layer_of(target))}, an upward import not in EDGE_EXCEPTIONS. Import from the owner's "
                        f"layer or below, or add a line naming the stage that removes it."
                    ),
                )
            )
    for key in sorted(listed_edges - set(found.edges)):
        reports.append(("stale edge", key, f"EDGE_EXCEPTIONS: {key} no longer exists in the package; delete its line."))

    listed_unresolved = {u[:4] for u in unresolved_exceptions}
    for key, sites in sorted(found.unresolved.items()):
        if key not in listed_unresolved:
            importer, source, imported, position = key
            where = ", ".join(f"{f}:{ln} ({kind})" for f, ln, kind in sites)
            reports.append(
                (
                    "unresolved",
                    key,
                    (
                        f"{where}: {importer} imports {imported} from {source} ({position}), and its owner cannot be "
                        f"placed statically. It is not in UNRESOLVED_EXCEPTIONS: bind it unconditionally, or add a line "
                        f"naming the condition or the loader that binds it."
                    ),
                )
            )
    for key in sorted(listed_unresolved - set(found.unresolved)):
        reports.append(("stale unresolved", key, f"UNRESOLVED_EXCEPTIONS: {key} no longer exists; delete its line."))

    listed_dynamic = set(dynamic_sites)
    for key, sites in sorted(found.dynamic.items()):
        if key not in listed_dynamic:
            where = ", ".join(f"{f}:{ln}" for f, ln in sites)
            reports.append(
                (
                    "dynamic",
                    key,
                    (f"{where}: {key[1]} in {key[0]}, an import this check cannot follow, not in DYNAMIC_SITES."),
                )
            )
    for key in sorted(listed_dynamic - set(found.dynamic)):
        reports.append(("stale dynamic", key, f"DYNAMIC_SITES: {key} no longer exists; delete its line."))

    for prefix in sorted(table):
        if prefix == "(root)":
            continue
        dotted = f"{package}.{prefix}"
        if not any(m == dotted or m.startswith(dotted + ".") for m in found.modules):
            reports.append(("stale prefix", (prefix,), f"LAYERS: the prefix {prefix!r} matches no module; delete it."))
    for module, rel in sorted(found.modules.items()):
        if layer_of(module) is None:
            reports.append(("no layer", (module,), f"{rel}: {module} has no layer; add its prefix to LAYERS."))
    return reports


# --------------------------------------------------------------------------------------------- self-test

# owners7's seventeen parser cases (#2938, stage 1), and an eighteenth, on a package `pk` with `pk.low` below `pk.high`.
PARSER_FILES = {
    "pk/__init__.py": "",
    "pk/low/__init__.py": "",
    "pk/high/__init__.py": "",
    "pk/high/impl.py": "def direct():\n    pass\ndef starred():\n    pass\n",
    "pk/high/facade.py": "from pk.high.impl import *\n",
    "pk/high/lazy.py": "def f():\n    from pk.high.impl import direct\n",
    "pk/low/a_direct.py": "from pk.high.impl import direct\n",
    "pk/low/b_star.py": "from pk.high.facade import starred\n",
    "pk/low/c_funcimport.py": "from pk.high.lazy import direct\n",
    "pk/low/d_else.py": "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    pass\nelse:\n    from pk.high.impl import direct\n",
    "pk/low/e_not.py": "from typing import TYPE_CHECKING\nif not TYPE_CHECKING:\n    from pk.high.impl import direct\n",
    "pk/low/f_type.py": "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from pk.high.impl import direct\n",
    "pk/high/empty_all.py": "__all__ = []\ndef hidden():\n    pass\n",
    "pk/high/facade_empty.py": "from pk.high.empty_all import *\n",
    "pk/low/g_emptyall.py": "from pk.high.facade_empty import hidden\n",
    "pk/high/listed_all.py": "__all__ = ['shown']\ndef shown():\n    pass\n",
    "pk/high/facade_listed.py": "from pk.high.listed_all import *\n",
    "pk/low/h_listed.py": "from pk.high.facade_listed import shown\n",
    "pk/high/rebind.py": "from pk.high.impl import direct\ndef direct():\n    pass\n",
    "pk/low/i_rebind.py": "from pk.high.rebind import direct\n",
    "pk/high/condbind.py": "try:\n    from pk.high.impl import direct\nexcept ImportError:\n    direct = None\n",
    "pk/low/j_cond.py": "from pk.high.condbind import direct\n",
    "pk/low/facade_low.py": "from pk.high.impl import direct\n",
    "pk/low/k_consumer_star.py": "from pk.low.facade_low import *\n",
    "pk/high/empty_star.py": "__all__ = []\n",
    "pk/low/l_emptystar.py": "from pk.high.empty_star import *\n",
    "pk/high/sub/__init__.py": "from . import leaf\n",
    "pk/high/sub/leaf.py": "",
    "pk/low/m_submodule.py": "from pk.high.sub import leaf\n",
    "pk/high/ring/__init__.py": "try:\n    FLAG = True\nexcept ImportError:\n    FLAG = False\nfrom .part import FLAG\n",
    "pk/high/ring/part.py": "from . import FLAG\n",
    "pk/low/n_ring_pkg.py": "from pk.high.ring import FLAG\n",
    "pk/low/o_ring_part.py": "from pk.high.ring.part import FLAG\n",
    "pk/high/loop_a.py": "from pk.high.loop_b import spin\n",
    "pk/high/loop_b.py": "from pk.high.loop_a import spin\n",
    "pk/low/p_pure_loop.py": "from pk.high.loop_a import spin\n",
    "pk/high/ring2/__init__.py": "FLAG = True\nfrom .part import FLAG\n",
    "pk/high/ring2/part.py": "from . import FLAG\n",
    "pk/low/q_ring_def.py": "from pk.high.ring2.part import FLAG\n",
    "pk/high/typeonly.py": "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from pk.high.impl import direct\n",
    "pk/low/r_typeonly.py": "from pk.high.typeonly import direct\n",
}
PK_TABLE = {"low": 0, "high": 1, "(root)": 0}
# (importing module, name) -> (expected owner, position). An unresolved owner is "ambiguous" or "not found".
PARSER_CASES = {
    ("pk.low.a_direct", "direct"): ("pk.high.impl", "module"),  # 1: a direct definition (the positive control)
    ("pk.low.b_star", "starred"): ("pk.high.impl", "module"),  # 2: a star re-export, followed to the definer
    ("pk.low.c_funcimport", "direct"): ("not found", "module"),  # 3: a name imported only inside a function
    ("pk.low.d_else", "direct"): ("pk.high.impl", "module"),  # 4: the else of TYPE_CHECKING runs
    ("pk.low.e_not", "direct"): ("pk.high.impl", "module"),  # 5: `if not TYPE_CHECKING:` runs
    ("pk.low.f_type", "direct"): ("pk.high.impl", "type"),  # 6: the body of TYPE_CHECKING
    ("pk.low.g_emptyall", "hidden"): ("not found", "module"),  # 7: a star-imported module with `__all__ = []`
    ("pk.low.h_listed", "shown"): ("pk.high.listed_all", "module"),  # 8: a name in a static __all__
    ("pk.low.i_rebind", "direct"): ("pk.high.rebind", "module"),  # 9: imported and then redefined
    ("pk.low.j_cond", "direct"): ("ambiguous", "module"),  # 10: a conditional binding in try
    ("pk.low.k_consumer_star", "direct"): ("pk.high.impl", "module"),  # 11: a star import of a re-exporting module
    # 12 is a path edge, below.
    ("pk.low.m_submodule", "leaf"): ("pk.high.sub.leaf", "module"),  # 13: `from . import submodule`
    ("pk.low.n_ring_pkg", "FLAG"): ("ambiguous", "module"),  # 14: a ring ended by a conditional binding, package side
    ("pk.low.o_ring_part", "FLAG"): ("ambiguous", "module"),  # 15: the same ring, submodule side
    ("pk.low.p_pure_loop", "spin"): ("ambiguous", "module"),  # 16: a ring no binding ends; it terminates
    ("pk.low.q_ring_def", "FLAG"): ("pk.high.ring2", "module"),  # 17: a ring ended by an unconditional definition
    ("pk.low.r_typeonly", "direct"): ("not found", "module"),  # 18: bound only under TYPE_CHECKING: never at run time
}

# The statement shapes of tests/unit/test_utils_does_not_import_alg.py, with the position this check gives each.
# One differs from that test: it calls the body of `if False:` dead, while this check measures the direction of
# source dependencies, not loading, so the import counts at module level.
_I = "from pk.high.impl import direct"
STATEMENT_SHAPES = (
    ("try body", f"try:\n    {_I}\nexcept ImportError:\n    pass\n", "module"),
    ("except handler", f"try:\n    pass\nexcept ImportError:\n    {_I}\n", "module"),
    ("try finally", f"try:\n    pass\nfinally:\n    {_I}\n", "module"),
    ("with", f"import contextlib\nwith contextlib.suppress(ImportError):\n    {_I}\n", "module"),
    ("module-level for", f"for _ in (1,):\n    {_I}\n", "module"),
    ("if not TYPE_CHECKING", f"from typing import TYPE_CHECKING\nif not TYPE_CHECKING:\n    {_I}\n", "module"),
    ("version guard", f"import sys\nif sys.version_info >= (3, 12):\n    {_I}\n", "module"),
    ("match case", f"match 1:\n    case 1:\n        {_I}\n", "module"),
    (
        "else of TYPE_CHECKING",
        f"from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    pass\nelse:\n    {_I}\n",
        "module",
    ),
    ("class body", f"class C:\n    {_I}\n", "module"),
    ("__main__ block", f"if __name__ == '__main__':\n    {_I}\n", "module"),
    ("if TYPE_CHECKING", f"from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    {_I}\n", "type"),
    ("function body", f"def f():\n    {_I}\n", "function"),
    ("if False", f"if False:\n    {_I}\n", "module"),  # the old test: dead
    ("else of if False", f"if False:\n    pass\nelse:\n    {_I}\n", "module"),
)

# (importing module, is a package, level, module named, expected absolute module): the relative cases of
# tests/unit/test_utils_does_not_import_alg.py, checked there against importlib.util.resolve_name.
RELATIVE_CASES = (
    ("mfgarchon.utils.numerical", True, 3, "alg.base_solver", "mfgarchon.alg.base_solver"),
    ("mfgarchon.utils.adjoint_validation", False, 2, "alg.base_solver", "mfgarchon.alg.base_solver"),
    ("mfgarchon.utils.convergence.convergence_monitors", False, 3, "alg.base_solver", "mfgarchon.alg.base_solver"),
    ("mfgarchon.utils.numerical.particle.sampling", False, 4, "alg.base_solver", "mfgarchon.alg.base_solver"),
    ("mfgarchon.utils.numerical", True, 1, "particle", "mfgarchon.utils.numerical.particle"),
)


def write_package(root: str, files: dict[str, str]) -> None:
    for rel, text in files.items():
        path = Path(root, rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def report(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'} {label}{': ' + detail if detail and not ok else ''}")
    return ok


def self_test() -> bool:
    ok = True
    layer_of = make_layer_of(PK_TABLE, "pk")

    print("parser cases:")
    with tempfile.TemporaryDirectory() as root:
        write_package(root, PARSER_FILES)
        found = scan(root, "pk", layer_of)
        for (importer, name), (owner, position) in PARSER_CASES.items():
            got_owner, got_position = found.owners.get((importer, name), ("<no row>", "<no row>"))
            got_owner = "ambiguous" if got_owner is AMBIGUOUS else "not found" if got_owner is None else got_owner
            ok &= report(
                f"{importer} {name}",
                (got_owner, got_position) == (owner, position),
                f"expected {(owner, position)}, got {(got_owner, got_position)}",
            )
        ok &= report(
            "pk.low.l_emptystar: an empty __all__ still imports the module (path edge, case 12)",
            ("pk.low.l_emptystar", "pk.high.empty_star", "module") in found.edges,
        )
        ok &= report(
            "pk.low.k_consumer_star: the star-expanded name is an edge to pk.high.impl",
            ("pk.low.k_consumer_star", "pk.high.impl", "module") in found.edges,
        )

    print("statement shapes:")
    with tempfile.TemporaryDirectory() as root:
        files = {
            "pk/__init__.py": "",
            "pk/low/__init__.py": "",
            "pk/high/__init__.py": "",
            "pk/high/impl.py": "def direct():\n    pass\n",
        }
        for i, (_, text, _) in enumerate(STATEMENT_SHAPES):
            files[f"pk/low/s{i:02d}.py"] = text
        write_package(root, files)
        found = scan(root, "pk", layer_of)
        for i, (label, _, position) in enumerate(STATEMENT_SHAPES):
            got = sorted(k[2] for k in found.edges if k[0] == f"pk.low.s{i:02d}" and k[1] == "pk.high.impl")
            ok &= report(f"{label} -> {position}", got == [position], f"got {got}")

    print("relative imports:")
    for importer, is_package, level, named, expected in RELATIVE_CASES:
        node = ast.ImportFrom(module=named, names=[ast.alias(name="X")], level=level)
        got = Tree.absolute(importer, is_package, node)
        ok &= report(f"{importer} level {level} {named}", got == expected, f"got {got}")

    print("mechanism cases:")
    base = {
        "pk/__init__.py": "",
        "pk/low/__init__.py": "",
        "pk/low/listed.py": "from pk.high.impl import direct\n",
        "pk/low/flagged.py": "from pk.low.cond import FLAG\n",
        "pk/low/dyn.py": "import importlib\nimportlib.import_module('pk.high.impl')\n",
        "pk/high/__init__.py": "",
        "pk/high/impl.py": "def direct():\n    pass\n",
        "pk/low/cond.py": "try:\n    import zz\n    FLAG = True\nexcept ImportError:\n    FLAG = False\n",
    }
    edges = (("pk.low.listed", "pk.high.impl", "module", "stage", "reason"),)
    unresolved = (("pk.low.flagged", "pk.low.cond", "FLAG", "module", "condition"),)
    dynamic = (("pk.low.dyn", "import_module"),)

    def run(changes: dict[str, str], table: dict[str, int]) -> list[tuple[str, tuple[str, ...], str]]:
        with tempfile.TemporaryDirectory() as root:
            write_package(root, {**base, **changes})
            return check(root, "pk", table, edges, unresolved, dynamic)

    # (label, files changed, table, expected (kind, key) reports, the location the one report must name)
    cases = (
        ("everything listed: nothing reported (the control)", {}, PK_TABLE, [], None),
        (
            "a planted upward import is reported",
            {"pk/low/planted.py": "from pk.high.impl import direct\n"},
            PK_TABLE,
            [("edge", ("pk.low.planted", "pk.high.impl", "module"))],
            "pk/low/planted.py:1",
        ),
        (
            "a listed edge whose import is deleted is reported stale",
            {"pk/low/listed.py": ""},
            PK_TABLE,
            [("stale edge", ("pk.low.listed", "pk.high.impl", "module"))],
            "EDGE_EXCEPTIONS",
        ),
        (
            "an unresolved owner outside the list is reported",
            {"pk/low/planted.py": "def f():\n    from pk.low.cond import FLAG\n"},
            PK_TABLE,
            [("unresolved", ("pk.low.planted", "pk.low.cond", "FLAG", "function"))],
            "pk/low/planted.py:2",
        ),
        (
            "a listed unresolved owner whose import is deleted is reported stale",
            {"pk/low/flagged.py": ""},
            PK_TABLE,
            [("stale unresolved", ("pk.low.flagged", "pk.low.cond", "FLAG", "module"))],
            "UNRESOLVED_EXCEPTIONS",
        ),
        (
            "a dynamic site outside the list is reported",
            {"pk/low/lazy.py": "def __getattr__(name):\n    raise AttributeError(name)\n"},
            PK_TABLE,
            [("dynamic", ("pk.low.lazy", "__getattr__"))],
            "pk/low/lazy.py:1",
        ),
        (
            "a listed dynamic site that is deleted is reported stale",
            {"pk/low/dyn.py": ""},
            PK_TABLE,
            [("stale dynamic", ("pk.low.dyn", "import_module"))],
            "DYNAMIC_SITES",
        ),
        (
            "a layer-table prefix that matches no module is reported",
            {},
            {**PK_TABLE, "gone": 1},
            [("stale prefix", ("gone",))],
            "LAYERS",
        ),
        (
            "a module no prefix places is reported",
            {"pk/side.py": ""},
            {"low": 0, "high": 1},
            [("no layer", ("pk",)), ("no layer", ("pk.side",))],
            "pk/side.py",
        ),
    )
    for label, changes, table, expected, location in cases:
        got = run(changes, table)
        kinds = [(kind, key) for kind, key, _ in got]
        named = location is None or (bool(got) and location in got[-1][2])
        ok &= report(label, kinds == expected and named, f"expected {expected} naming {location}, got {got}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parent.parent),
        help="the directory holding mfgarchon/ (default: this repository)",
    )
    parser.add_argument("--self-test", action="store_true", help="run the parser and mechanism cases")
    args = parser.parse_args()
    if args.self_test:
        passed = self_test()
        print("self-test passed" if passed else "self-test FAILED")
        return 0 if passed else 1
    if not Path(args.root, PACKAGE).is_dir():
        print(f"{args.root} holds no {PACKAGE}/ directory")
        return 1
    reports = check(args.root, PACKAGE, LAYERS, EDGE_EXCEPTIONS, UNRESOLVED_EXCEPTIONS, DYNAMIC_SITES)
    for _, _, message in reports:
        print(message)
    if reports:
        return 1
    print("OK: every upward import, unresolved owner and dynamic site is listed, and every listed one exists.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
