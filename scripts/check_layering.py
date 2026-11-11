#!/usr/bin/env python3
"""A lower layer of mfgarchon does not import an upper one (#2938).

Usage:
    python scripts/check_layering.py [--root TREE]   # check TREE/mfgarchon (default: this repository)
    python scripts/check_layering.py --self-test     # its own cases, on temporary packages

**It is enforced by the gate.** ``scripts/local_ci.sh`` runs ``--self-test`` in its self-test loop and then the check
of the package as a step of its own, and a non-zero exit fails the gate (user ruling, 2026-11-11, #2938): the
exception lists below are the ratchet. CI does not run it.

This file holds the layer table (``LAYERS``) and the exception lists (``EDGE_EXCEPTIONS``,
``UNRESOLVED_EXCEPTIONS``, ``DYNAMIC_SITES``). A PR that removes an edge removes its line in the same diff.

**It does not simulate the interpreter** (#2938). It resolves a closed list of module-level binding forms, and any
other module-level statement that binds a name in its syntax makes that name ambiguous: reported unless listed. It
may over-report. Within its scope it does not miss: module-level statements outside ``def`` and ``class`` bodies, for
the forms on the closed list, apart from the forms stated as not read with that list (#2945).

**What is counted:**

- An edge is (importing module, upper target module, position). Position is one of:
  - ``type``: inside the body of ``if TYPE_CHECKING:`` or ``if X.TYPE_CHECKING:``, at any depth. No other test
    is read as type-only: an alias of the name, ``TYPE_CHECKING and ...`` or ``if not TYPE_CHECKING:`` is an
    ordinary condition;
  - ``function``: otherwise inside a ``def`` or an ``async def``;
  - ``module``: otherwise. That includes class bodies, ``if __name__ == "__main__":``, and the body of
    ``if False:``, since the rule is about the direction of source dependencies, not about what runs when a
    module is loaded.
- An import has two targets: the module on its path, and the owner of each name it binds. Either one in an
  upper layer is an edge, so a re-export of an upper name is an edge. The path of ``from P import ...`` is an edge
  unless every name it binds is, in ``P``, the submodule object (``Tree.binds_submodule``, #2938's path-target
  ruling): that statement is the dependency ``import P.name`` makes, and the owner edge to each submodule stands. A
  name the ``__init__`` binds otherwise -- a function re-exported from a same-named submodule, or a star import that
  exports it -- keeps the path edge, and so does one whose binding is conditional, a star import whose exports
  cannot be read (its source outside the tree, or a non-literal ``__all__``), and any module-level ``__getattr__``
  in the ``__init__``. A star import's path is always an edge.
- **The closed list.** At module level these bind names, unconditionally, and conditionally inside ``if``,
  ``try`` (body, ``else``, ``finally`` and handlers) or ``with``:
  - ``def``, ``async def`` and ``class``;
  - an assignment whose targets are all names, chained or not;
  - an annotated assignment with a value to a name; ``NAME: T`` alone binds nothing;
  - ``type NAME = ...``;
  - ``import a.b`` (binds ``a``) and ``import a.b as x``;
  - ``from M import n [as x]``;
  - ``from M import *``;
  - ``__all__``, assigned or annotated, as a list or tuple of string literals.
  - The body of ``if TYPE_CHECKING:`` binds nothing at run time, and its ``else`` binds as its parent does.
- **Everything else is ambiguous.** Any other module-level statement or expression that binds a name in its
  syntax gives that name a conditional binding: a tuple, list or starred target, an augmented assignment, a walrus
  (in a lambda or a comprehension too), ``del``, ``for``, ``while`` and ``match`` with everything inside them,
  ``with ... as`` and ``except ... as``, and a ``def``'s or ``class``'s decorators, defaults, annotations and
  bases. Any other module-level statement that mentions ``__all__``, other than a ``def`` or ``class`` statement,
  makes the star exports ambiguous.
- **Not read** (#2945). A name bound in one of these ways keeps a wrong owner, so an upward edge through it is
  missed:
  - a ``def`` or ``class`` statement that mentions ``__all__``, as the export decorator
    ``def export(fn): __all__.append(fn.__name__); return fn`` does; and the annotation of an annotated
    assignment, which Python evaluates, so ``X: (direct := f) = 1`` binds ``direct``;
  - ``global``: a ``def`` or ``class`` body that declares a name ``global`` and binds it, as
    ``def _load(): global direct; from pk.high.impl import direct`` does when the module calls ``_load()``. On main
    bfe6de4d, 9 names in 6 modules are declared ``global``, and no import in the package can bind one; #2945 has
    the command;
  - a submodule load: the first load of ``P.x``, from any module, sets ``P``'s name ``x`` to the submodule, while
    the check keeps ``P``'s own binding, such as ``def leaf()`` in ``P/__init__.py`` beside ``P/leaf.py``;
  - two forms on the list: an ``__all__`` bound in a chained assignment and changed through its other name,
    ``__all__ = _names = []`` and then ``_names.append("x")``; and ``from . import x`` in ``P/__init__.py`` after
    ``P`` bound ``x``, which the check takes for the submodule while Python keeps the earlier binding;
  - a binding that is not syntax, such as ``globals()["x"] = ...``, and the loaders named under *What a static
    check cannot follow*.
- **Owners.** The last binding of a name wins, and a conditional last binding is ambiguous. An attribute or
  subscript target binds no name. A binding that leads back into a re-export ring is skipped for the binding
  before it; a ring no binding ends is ambiguous, and so is a ring of star imports. ``from x import *`` exports
  ``x``'s static ``__all__``, or, when ``x`` binds no ``__all__``, its names with no leading underscore. A name
  ``P`` does not bind is the submodule ``P.name`` when that exists and no star import in ``P`` could bind it. A
  name bound nowhere is unresolved (not found). An unresolved owner is never taken to be the importing module.

**What it reports.** Each report names a location, and none prints a total:

- an upward edge outside ``EDGE_EXCEPTIONS``, at the importer's file and line;
- a line of ``EDGE_EXCEPTIONS`` whose edge the scan no longer finds;
- an unresolved owner outside ``UNRESOLVED_EXCEPTIONS``, at the importer's file and line;
- a line of ``UNRESOLVED_EXCEPTIONS`` that the scan no longer finds unresolved;
- a dynamic import mechanism in a module where ``DYNAMIC_SITES`` does not list it, at its file and line;
- a line of ``DYNAMIC_SITES`` whose mechanism the scan no longer finds in that module;
- a ``LAYERS`` prefix that matches no module;
- a module that no ``LAYERS`` prefix places.

It exits 1 if it reports anything.

**What a static check cannot follow.** It recognises, widely, the places a module could import by a computed
name: any name, attribute, import alias or string constant equal to ``import_module`` or ``__import__``, which
includes a stored reference, ``getattr(importlib, "import_module")`` and an attribute that shares the name, such as
a call of a method named ``import_module``; and any module-level binding of ``__getattr__`` in syntax, outside the
bodies of defs and classes. Their sites are listed in ``DYNAMIC_SITES`` by module and mechanism: a mechanism new to a
module is reported, a further site of a listed mechanism in that module is not, and no site's imports are followed.
Other loaders are not detected (#2945), such as ``pkgutil.resolve_name``, ``runpy``, ``exec`` of import text, or a
``__getattr__`` bound through ``globals()``. The package's own code uses two of them, ``entry_point.load()`` and
``spec_from_file_location`` with ``exec_module``, both in ``core/plugin_system.py``: they load plugin code from
outside the package, and #2938's stage 4 deletes that module.

**Limits.** A ``type``-position import of a name its source binds only under ``TYPE_CHECKING`` is reported as
unresolved (not found), not followed to the type binding. A symlinked directory under the package is not walked.
A ``foo.py`` beside ``foo/__init__.py`` is scanned under the package's name.

**The one owner.** This check is the rule's one owner. ``tests/unit/test_utils_does_not_import_alg.py`` held a
second count of one slice, the module-level imports of ``mfgarchon.alg`` from every file under
``mfgarchon/utils/``, and its count was deleted when this check entered the gate (#2938). Where the two disagreed,
the layer table's reading stands: the seven ``utils`` modules ``LAYERS`` places in or above ``alg``'s layer (the
five in the solve layer, and ``utils.cli`` and ``utils.experiment_manager`` in the entry layer) may import ``alg``
at module level, so ``utils/adjoint_validation.py`` -> ``alg.base_solver``, which that count held as its last
violation, lies within one layer and is not an edge. Six of the seven, all but ``utils.solver_decorators``, load
with ``mfgarchon.utils`` through its re-exports until #2938's stage 2a.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import importlib.util
import io
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
    ("mfgarchon.utils.solver_result", "mfgarchon.utils.io.hdf5_utils", "function", "#2938 stage 2b", "the result's save_hdf5 and load_hdf5 call the HDF5 writer and reader"),
    ("mfgarchon.utils.solver_result", "mfgarchon.utils.notebooks.reporting", "function", "#2938 stage 2b", "the result's report method calls the notebook reporter"),
    # numerics -> geometry
    ("mfgarchon.utils", "mfgarchon.geometry.implicit.hyperrectangle", "module", "#2938 stage 2a", "utils/__init__ re-exports a geometry name"),
    ("mfgarchon.utils", "mfgarchon.geometry.implicit.hypersphere", "module", "#2938 stage 2a", "utils/__init__ re-exports a geometry name"),
    ("mfgarchon.utils", "mfgarchon.geometry.implicit.implicit_domain", "module", "#2938 stage 2a", "utils/__init__ re-exports a geometry name"),
    ("mfgarchon.utils", "mfgarchon.geometry.implicit.sdf_utils", "module", "#2938 stage 2a", "utils/__init__ re-exports a geometry name"),
    ("mfgarchon.utils", "mfgarchon.utils.geometry", "module", "#2938 stage 2a", "utils/__init__ re-exports a geometry name"),
    ("mfgarchon.utils.numerical", "mfgarchon.geometry.implicit.sdf_utils", "module", "#2938 stage 2a", "utils/numerical/__init__ re-exports a geometry name"),
    ("mfgarchon.utils.numerical", "mfgarchon.geometry.meshes.mesh_distances", "module", "#2938 stage 2a", "utils/numerical/__init__ re-exports a geometry name"),
    ("mfgarchon.utils.numerical.particle.sampling", "mfgarchon.geometry.boundary.periodic", "function", "#2938 stage 2a", "sampling builds periodic ghost points and wraps positions with the periodic boundary's helpers"),
    ("mfgarchon.utils.sparse_operations", "mfgarchon.geometry.grids.tensor_grid", "type", "#2938 stage 2a", "an annotation names TensorProductGrid"),
    # numerics -> model
    ("mfgarchon.utils", "mfgarchon.utils.drift_helpers", "module", "#2938 stage 2a", "utils/__init__ re-exports a model-layer name"),
    ("mfgarchon.utils", "mfgarchon.utils.pde_coefficients", "module", "#2938 stage 2a", "utils/__init__ re-exports a model-layer name"),
    # numerics -> solve
    ("mfgarchon.utils", "mfgarchon.utils.adjoint_validation", "module", "#2938 stage 2a", "utils/__init__ re-exports a solve-layer name"),
    ("mfgarchon.utils.numerical", "mfgarchon.utils.numerical.flux_diagnostics", "module", "#2938 stage 2a", "utils/numerical/__init__ re-exports a solve-layer name"),
    ("mfgarchon.utils.numerical", "mfgarchon.utils.numerical.hjb_policy_iteration", "module", "#2938 stage 2a", "utils/numerical/__init__ re-exports a solve-layer name"),
    ("mfgarchon.utils.numerical", "mfgarchon.utils.numerical.mass_fabrication_gate", "module", "#2938 stage 2a", "utils/numerical/__init__ re-exports a solve-layer name"),
    ("mfgarchon.utils.numerical.particle.sampling", "mfgarchon.utils.numerical.mass_fabrication_gate", "function", "#2938 stage 2a", "sampling leaves out, or refuses, the mass of cells with no navigable point through the gate's leave_out_or_raise"),
    # numerics -> entry
    ("mfgarchon.utils", "mfgarchon.utils.cli", "module", "#2938 stage 2a", "utils/__init__ re-exports an entry-layer name"),
    ("mfgarchon.utils", "mfgarchon.utils.experiment_manager", "module", "#2938 stage 2a", "utils/__init__ re-exports an entry-layer name"),
    ("mfgarchon.utils.data.recorder", "mfgarchon", "function", "#2938 stage 2a", "the recorder reads the package's __version__"),
    # geometry -> operators
    ("mfgarchon.geometry", "mfgarchon.operators", "module", "#2938 stage 3", "geometry/__init__ re-exports GeometryProjector and ProjectionRegistry"),
    ("mfgarchon.geometry", "mfgarchon.operators.interpolation.projection", "module", "#2938 stage 3", "geometry/__init__ re-exports GeometryProjector and ProjectionRegistry"),
    ("mfgarchon.geometry.boundary.types", "mfgarchon.operators.differential.function_gradient", "function", "#2938 stage 3", "the module-level _compute_sdf_gradient differentiates a function with an operator"),
    ("mfgarchon.geometry.collocation", "mfgarchon.operators.differential.function_gradient", "function", "#2938 stage 3", "collocation differentiates a function with an operator"),
    ("mfgarchon.geometry.grids.tensor_grid", "mfgarchon.operators", "function", "#2938 stage 3", "the grid builds its own operators"),
    ("mfgarchon.geometry.grids.tensor_grid", "mfgarchon.operators.differential.advection", "function", "#2938 stage 3", "the grid builds its own operators"),
    ("mfgarchon.geometry.grids.tensor_grid", "mfgarchon.operators.differential.divergence", "function", "#2938 stage 3", "the grid builds its own operators"),
    ("mfgarchon.geometry.grids.tensor_grid", "mfgarchon.operators.differential.gradient", "function", "#2938 stage 3", "the grid builds its own operators"),
    ("mfgarchon.geometry.grids.tensor_grid", "mfgarchon.operators.differential.laplacian", "function", "#2938 stage 3", "the grid builds its own operators"),
    ("mfgarchon.geometry.grids.tensor_grid", "mfgarchon.operators.interpolation.interpolation", "function", "#2938 stage 3", "the grid builds its own operators"),
    ("mfgarchon.geometry.implicit.implicit_domain", "mfgarchon.operators.differential.function_gradient", "function", "#2938 stage 3", "an implicit domain differentiates its level-set function with an operator"),
    ("mfgarchon.geometry.level_set.core", "mfgarchon.operators.reconstruction.weno", "function", "#2938 stage 3", "level-set evolution uses an operator-layer WENO5 reconstruction"),
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
    ("mfgarchon.core.measure", "mfgarchon.utils.numerical.mass_fabrication_gate", "module", "#2938 stage 2c", "the measure clips through mass_fabrication_gate, which imports only numpy and utils.numerical.quadrature and is read from layers 1, 4 and 5: its place in the table is the question"),
    ("mfgarchon.core.mfg_problem", "mfgarchon.alg.numerical.coupling", "function", "#2938 stage 4", "problem.solve() orchestrates the solve"),
    ("mfgarchon.core.mfg_problem", "mfgarchon.alg.numerical.coupling.fixed_point_iterator", "function", "#2938 stage 4", "problem.solve() orchestrates the solve"),
    ("mfgarchon.core.mfg_problem", "mfgarchon.alg.numerical.coupling.source_composition", "type", "#2938 stage 2c", "an annotation names CouplingBinding: which side owns the type it needs is the question (rule 3)"),
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
    ("mfgarchon.utils.acceleration.jax_utils", "mfgarchon.types.solver_types", "JAXArray", "type", "bound in solver_types' try, whose body imports jax only under TYPE_CHECKING: at run time both arms bind Any"),
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

# (module, mechanism): the sites whose imports a static check cannot follow. Keyed by module and mechanism, with no
# line and no count (#2938): a mechanism new to a module is reported, but a further site of a listed mechanism in the
# same module is not, and no site's imports are followed.
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
    """`TYPE_CHECKING`, or any `X.TYPE_CHECKING`. Any other test is an ordinary condition, since the check does not
    simulate the interpreter (#2938): an alias of the name or a compound test gives ambiguous bindings and `module`
    position."""
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def position_of(stack: list[tuple[ast.AST, str]]) -> str:
    """Position of an import, given the (parent node, field) pairs from the module down to it."""
    for node, fieldname in stack:
        if isinstance(node, ast.If) and fieldname == "body" and is_type_checking(node.test):
            return "type"
    if any(isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) for node, _ in stack):
        return "function"
    return "module"


SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def binders(node: ast.AST, into_scopes: bool) -> Iterator[tuple[str, ast.AST]]:
    """Every (name, node) that a piece of code can bind: a Name stored or deleted (by any assignment, augmented
    assignment, walrus, `for`, `with ... as`, `del` or comprehension), an `except ... as` name, a match capture, an
    import's names ("*" for a star import), and the name of a `def` or `class`. With `into_scopes` False, the bodies
    of a `def` or `class` are not entered; their decorators, defaults, annotations and bases are."""
    stack: list[ast.AST] = [node]
    while stack:
        x = stack.pop()
        if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store | ast.Del):
            yield x.id, x
        elif isinstance(x, ast.ExceptHandler | ast.MatchAs | ast.MatchStar) and x.name:
            yield x.name, x
        elif isinstance(x, ast.MatchMapping) and x.rest:
            yield x.rest, x
        elif isinstance(x, ast.Import):
            for alias in x.names:
                yield alias.asname or alias.name.split(".")[0], x
        elif isinstance(x, ast.ImportFrom):
            for alias in x.names:
                yield alias.asname or alias.name, x
        elif isinstance(x, SCOPES):
            yield x.name, x
            if not into_scopes:
                for fieldname, value in ast.iter_fields(x):
                    if fieldname != "body":
                        stack.extend(
                            v for v in (value if isinstance(value, list) else [value]) if isinstance(v, ast.AST)
                        )
                continue
        stack.extend(ast.iter_child_nodes(x))


def read_source(path: Path) -> ast.Module:
    """Parse from bytes, so that a BOM or a coding cookie is honoured as Python honours it."""
    return ast.parse(path.read_bytes(), filename=str(path))


AMBIGUOUS = object()
CYCLE = object()


class Tree:
    """Static owners of names in one package on disk."""

    def __init__(self, root: str, package: str = PACKAGE) -> None:
        self.root = root
        self.package = package
        self.event_cache: dict[str, list[tuple[str, str, object, bool]]] = {}

    def path_of(self, module: str) -> tuple[str | None, bool]:
        base = os.path.join(self.root, *module.split("."))
        if os.path.isfile(os.path.join(base, "__init__.py")):
            return os.path.join(base, "__init__.py"), True
        if os.path.isfile(base + ".py"):
            return base + ".py", False
        return None, False

    def parsed(self, module: str) -> tuple[ast.Module | None, bool]:
        """The module's syntax tree, None when no file holds it, and whether it is a package. `events` caches it."""
        path, is_package = self.path_of(module)
        return (read_source(Path(path)) if path else None), is_package

    @staticmethod
    def absolute(module: str, is_package: bool, node: ast.ImportFrom) -> str:
        """The absolute module an ImportFrom names, from the importing module and whether it is a package."""
        if not node.level:
            return node.module or ""
        base = module.split(".") if is_package else module.split(".")[:-1]
        if node.level - 1 >= len(base):
            return ""  # beyond the top-level package: Python refuses it, and there is no module to name
        base = base[: len(base) - (node.level - 1)]
        return ".".join(base + ([node.module] if node.module else []))

    def events(self, module: str) -> list[tuple[str, str, object, bool]]:
        """Module-level binding events in source order: (name, kind, payload, conditional).

        The closed list of resolved forms: `def`, `async def` and `class`; an assignment whose targets are all names;
        an annotated assignment with a value to a name (`NAME: T` alone binds nothing); `type NAME = ...`; `import`;
        `from M import n`; and `from M import *`. Each is unconditional at module level and conditional inside `if`,
        `try` or `with`. The body of `if TYPE_CHECKING:` binds nothing at run time, and its `else` runs as its parent
        does. Any other statement or expression that binds a name in its syntax gives that name a conditional
        binding, which resolves as ambiguous; and any other statement that mentions `__all__`, other than a `def` or
        `class` statement, makes the star exports ambiguous. The forms #2945 lists are not read."""
        if module in self.event_cache:
            return self.event_cache[module]
        tree, is_package = self.parsed(module)
        out: list[tuple[str, str, object, bool]] = []

        def ambiguous(node: ast.AST, conditional: bool) -> None:
            for name, binder in binders(node, into_scopes=True):
                if name == "*" and isinstance(binder, ast.ImportFrom):
                    out.append(("*", "star", self.absolute(module, is_package, binder), True))
                else:
                    out.append((name, "def", binder, True))
            if any(isinstance(x, ast.Name) and x.id == "__all__" for x in ast.walk(node)):
                out.append(("__all__", "allmut", node, conditional))

        def walk(body: list[ast.stmt], conditional: bool) -> None:
            for n in body:
                if isinstance(n, ast.If):
                    ambiguous(n.test, conditional)
                    if is_type_checking(n.test):
                        walk(n.orelse, conditional)
                    else:
                        walk(n.body, True)
                        walk(n.orelse, True)
                elif isinstance(n, ast.Try | ast.TryStar):
                    for name in ("body", "orelse", "finalbody"):
                        walk(getattr(n, name), True)
                    for handler in n.handlers:
                        if handler.name:
                            out.append((handler.name, "def", handler, True))
                        if handler.type is not None:
                            ambiguous(handler.type, True)
                        walk(handler.body, True)
                elif isinstance(n, ast.With):
                    for item in n.items:
                        ambiguous(item, True)
                    walk(n.body, True)
                elif isinstance(n, SCOPES):
                    out.append((n.name, "def", None, conditional))
                    for name, binder in binders(n, into_scopes=False):
                        if binder is not n:
                            out.append((name, "def", binder, True))
                elif isinstance(n, ast.Assign) and all(isinstance(t, ast.Name) for t in n.targets):
                    out.extend((t.id, "def", n, conditional) for t in n.targets)
                    ambiguous(n.value, conditional)
                elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
                    if n.value is not None:
                        out.append((n.target.id, "def", n, conditional))
                        ambiguous(n.value, conditional)
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
                else:
                    ambiguous(n, conditional)

        if tree is not None:
            walk(tree.body, False)
        self.event_cache[module] = out
        return out

    def star_exports(self, module: str, seen: tuple[str, ...] = ()) -> set[str] | None:
        """The names `from module import *` binds, or None when that cannot be decided statically, a ring of star
        imports among them."""
        if module in seen:
            return None
        events = self.events(module)
        alls = [e for e in events if e[0] == "__all__"]
        if alls:
            if any(e[3] for e in alls) or any(e[1] == "allmut" for e in alls):
                return None
            node = alls[-1][2]
            value = getattr(node, "value", None)
            if not isinstance(node, ast.Assign | ast.AnnAssign) or not isinstance(value, ast.List | ast.Tuple):
                return None
            if not all(isinstance(x, ast.Constant) and isinstance(x.value, str) for x in value.elts):
                return None
            return {x.value for x in value.elts}
        names = {e[0] for e in events if e[1] != "star" and not e[0].startswith("_")}
        for e in events:
            if e[1] == "star":
                sub = self.star_exports(e[2], (*seen, module))
                if sub is None:
                    return None
                names |= sub
        return names

    def binds_submodule(self, package: str, name: str) -> object:
        """Whether `from package import name` binds the submodule object `package.name`: True, False or AMBIGUOUS
        (#2938's path-target ruling).

        True when `package` is a package, `package.name` is a module on disk, no star import in its `__init__` may
        bind `name`, and its `__init__`'s module-level bindings of `name` are none, or only `from . import name`: the
        import system then hands back the submodule, the dependency `import package.name` makes. False when the
        `__init__` binds `name` otherwise -- a function re-exported from a same-named submodule
        (`operators.differential.function_gradient`), a `def`, a class, an assignment -- or a star import exports it,
        shadowing the submodule. AMBIGUOUS when a binding of `name` is conditional; when a star import's exports
        cannot be read, its source being no module in the tree (`from os.path import *` binds `join`) or its
        `__all__` not a literal; and when the `__init__` binds a module-level `__getattr__`, by any binding form or
        through a star import. Python's import consults that `__getattr__` before it loads the submodule, so one bound
        ahead of `from . import name` answers that import itself; the check does not follow which comes first. The
        `__init__`'s static bindings are read; a submodule load that rebinds the package's name at run time is #2945's
        (B3)."""
        _path, is_package = self.path_of(package)
        if not is_package or not self.path_of(package + "." + name)[0]:
            return False
        events = self.events(package)
        if any(e[0] == "__getattr__" for e in events):
            return AMBIGUOUS
        for e in events:
            if e[1] == "star":
                exported = self.star_exports(e[2]) if self.path_of(e[2])[0] else None
                if exported is None or "__getattr__" in exported:
                    return AMBIGUOUS
                if name in exported:
                    return False
        hits = [e for e in events if e[0] == name and e[1] != "allmut"]
        if not hits:
            return True
        if any(e[3] for e in hits):
            return AMBIGUOUS
        return all(e[1] == "import" and e[2] == (package, name) for e in hits)

    def owner(self, module: str, name: str) -> object:
        """The module that defines `name` as reached through `module`.

        None if it is bound nowhere statically visible; AMBIGUOUS if its binding depends on a run-time
        condition or on a non-literal `__all__`. A re-export ring is followed to the binding that ends it: a
        binding that leads back to a (module, name) already on the path is skipped in favour of the binding
        before it, which is what the partially initialised module held when the ring ran. A ring no binding
        ends is ambiguous."""
        found = self._owner(module, name, ())
        return AMBIGUOUS if found is CYCLE else found

    def _owner(self, module: str, name: str, stack: tuple[tuple[str, str], ...]) -> object:
        if (module, name) in stack:
            return CYCLE
        events = self.events(module)
        stars = [e for e in events if e[1] == "star"]
        star_may_bind = any((exported := self.star_exports(e[2])) is None or name in exported for e in stars)
        if self.path_of(module + "." + name)[0] and not star_may_bind and not any(e[0] == name for e in events):
            return module + "." + name
        hits = []
        for e in events:
            if e[0] == name and e[1] != "allmut":  # a mention of `__all__` binds nothing
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
                found = self._owner(payload, name, inner)
            else:
                source, original = payload
                if (source, original) == (module, name) and self.path_of(source + "." + original)[0]:
                    return source + "." + original  # `from . import submodule` in a package __init__
                if not source.startswith(self.package):
                    return source
                found = self._owner(source, original, inner)
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


DYNAMIC_NAMES = ("import_module", "__import__")


def dynamic_sites(tree: ast.Module) -> Iterator[tuple[str, int]]:
    """(mechanism, line) for every place a module could import by a computed name, matched widely rather than by
    following aliases: any name, attribute, import alias or string constant equal to `import_module` or
    `__import__`; and any module-level binding of `__getattr__` in syntax, outside the bodies of defs and
    classes."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            found = node.id
        elif isinstance(node, ast.Attribute):
            found = node.attr
        elif isinstance(node, ast.alias):
            found = node.name
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            found = node.value
        else:
            continue
        if found in DYNAMIC_NAMES:
            yield found, node.lineno
    for stmt in tree.body:
        for name, binder in binders(stmt, into_scopes=False):
            if name == "__getattr__":
                yield "__getattr__", getattr(binder, "lineno", stmt.lineno)


def scan_module(
    tree: Tree, result: Scan, layer_of: Callable[[str], int | None], module: str, rel: str, is_package: bool
) -> None:
    """Add one module's edges, unresolved owners and dynamic sites to `result`."""
    package = tree.package
    result.modules[module] = rel
    source_layer = layer_of(module)
    parsed = read_source(Path(tree.root, rel))
    for mechanism, line in dynamic_sites(parsed):
        result.dynamic.setdefault((module, mechanism), []).append((rel, line))

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
        # The path is a dependency of its own, unless every name the statement binds is, in the package it names, the
        # submodule object: then the statement is the dependency `import P.name` makes (#2938's path-target ruling).
        # A star import's path is always an edge, since no submodule is named `*`.
        if not all(tree.binds_submodule(source, alias.name) is True for alias in node.names):
            add_edge(source, position, node.lineno)
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

    def name(layer: int) -> str:
        return LAYER_NAMES.get(layer, str(layer))

    listed_edges = {e[:3] for e in edge_exceptions}
    for key, sites in sorted(found.edges.items()):
        if key not in listed_edges:
            importer, target, position = key
            where = ", ".join(f"{f}:{ln}" for f, ln in sorted(set(sites)))
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
        reports.append(
            ("stale edge", key, f"EDGE_EXCEPTIONS: {key}: the scan no longer finds this edge; delete its line.")
        )

    listed_unresolved = {u[:4] for u in unresolved_exceptions}
    for key, sites in sorted(found.unresolved.items()):
        if key not in listed_unresolved:
            importer, source, imported, position = key
            where = ", ".join(f"{f}:{ln} ({kind})" for f, ln, kind in sorted(set(sites)))
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
        reports.append(
            (
                "stale unresolved",
                key,
                f"UNRESOLVED_EXCEPTIONS: {key}: the scan no longer finds this unresolved; delete its line.",
            )
        )

    listed_dynamic = set(dynamic_sites)
    for key, sites in sorted(found.dynamic.items()):
        if key not in listed_dynamic:
            where = ", ".join(f"{f}:{ln}" for f, ln in sorted(set(sites)))
            reports.append(
                (
                    "dynamic",
                    key,
                    (f"{where}: {key[1]} in {key[0]}, an import this check cannot follow, not in DYNAMIC_SITES."),
                )
            )
    for key in sorted(listed_dynamic - set(found.dynamic)):
        reports.append(
            (
                "stale dynamic",
                key,
                f"DYNAMIC_SITES: {key}: the scan no longer finds this in the module; delete its line.",
            )
        )

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

# #2938's seventeen parser cases and further ones, on a package `pk` with `pk.low` below `pk.high`.
PARSER_FILES: dict[str, str] = {
    "pk/__init__.py": "",
    "pk/low/__init__.py": "",
    "pk/high/__init__.py": "",
    "pk/high/impl.py": "def direct():\n    pass\ndef starred():\n    pass\nclass Klass:\n    pass\nTABLE = {}\ndef _private():\n    pass\n",
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
    "pk/high/listed_all.py": "__all__ = ['shown']\ndef shown():\n    pass\ndef unlisted():\n    pass\n",
    "pk/high/facade_listed.py": "from pk.high.listed_all import *\n",
    "pk/low/h_listed.py": "from pk.high.facade_listed import shown, unlisted\n",
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
    "pk/low/s_private.py": "from pk.high.facade import _private\n",
    "pk/low/t_class.py": "from pk.high.impl import Klass\n",
    "pk/high/ifbind.py": "import sys\nif sys.platform:\n    def direct():\n        pass\n",
    "pk/low/u_ifbind.py": "from pk.high.ifbind import direct\n",
    "pk/high/elsebind.py": "import sys\nif sys.platform:\n    pass\nelse:\n    def direct():\n        pass\n",
    "pk/low/v_elsebind.py": "from pk.high.elsebind import direct\n",
    "pk/high/trybind.py": "try:\n    from pk.high.impl import direct\nexcept ImportError:\n    pass\n",
    "pk/low/w_trybind.py": "from pk.high.trybind import direct\n",
    "pk/high/withbind.py": "import contextlib\nwith contextlib.suppress(ImportError):\n    from pk.high.impl import direct\n",
    "pk/low/x_withbind.py": "from pk.high.withbind import direct\n",
    "pk/high/badall.py": "NAME = 'other'\n__all__ = ['direct', NAME]\ndef direct():\n    pass\n",
    "pk/high/facade_badall.py": "from pk.high.badall import *\n",
    "pk/low/y_badall.py": "from pk.high.facade_badall import direct\n",
    "pk/low/y2_starbad.py": "from pk.high.badall import *\n",
    "pk/high/condall.py": "import sys\nif sys.platform:\n    __all__ = ['direct']\ndef direct():\n    pass\n",
    "pk/high/facade_condall.py": "from pk.high.condall import *\n",
    "pk/low/z_condall.py": "from pk.high.facade_condall import direct\n",
    "pk/high/sub2/__init__.py": "",
    "pk/high/sub2/leaf2.py": "",
    "pk/low/aa_unbound_sub.py": "from pk.high.sub2 import leaf2\n",
    "pk/high/ext.py": "from os import path\n",
    "pk/low/ab_ext.py": "from pk.high.ext import path\n",
    "pk/high/dotted.py": "import os.path\n",
    "pk/low/ac_dotted.py": "from pk.high.dotted import os\n",
    "pk/high/sr_a.py": "from pk.high.sr_b import *\n",
    "pk/high/sr_b.py": "from pk.high.sr_a import *\n",
    "pk/low/ad_starring.py": "from pk.high.sr_a import spin\n",
    "pk/high/attr.py": "from pk.high.impl import Klass\nKlass.extra = None\n",
    "pk/low/ae_attr.py": "from pk.high.attr import Klass\n",
    "pk/high/subs.py": "from pk.high.impl import TABLE, direct\nTABLE[direct] = 1\n",
    "pk/low/af_subs.py": "from pk.high.subs import TABLE, direct\n",
    "pk/high/forbody.py": "def direct():\n    pass\nfor _ in (1,):\n    from pk.high.impl import direct\n",
    "pk/low/ag_for.py": "from pk.high.forbody import direct\n",
    "pk/high/whilebody.py": "def direct():\n    pass\nwhile False:\n    from pk.high.impl import direct\n",
    "pk/low/ah_while.py": "from pk.high.whilebody import direct\n",
    "pk/high/matchbody.py": "def direct():\n    pass\nmatch 1:\n    case 1:\n        from pk.high.impl import direct\n",
    "pk/low/ai_match.py": "from pk.high.matchbody import direct\n",
    "pk/high/withas.py": "import contextlib\nfrom pk.high.impl import direct\nwith contextlib.nullcontext() as direct:\n    pass\n",
    "pk/low/aj_withas.py": "from pk.high.withas import direct\n",
    "pk/high/walrus.py": "from pk.high.impl import direct\n(direct := 1)\n",
    "pk/low/ak_walrus.py": "from pk.high.walrus import direct\n",
    "pk/high/fortarget.py": "from pk.high.impl import direct\nfor direct in (1,):\n    pass\n",
    "pk/low/al_fortarget.py": "from pk.high.fortarget import direct\n",
    "pk/high/tcelse.py": "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    pass\nelse:\n    def direct():\n        pass\n",
    "pk/low/am_tcelse.py": "from pk.high.tcelse import direct\n",
    "pk/high/annall.py": "__all__: list[str] = ['shown2']\ndef shown2():\n    pass\ndef other2():\n    pass\n",
    "pk/high/facade_annall.py": "from pk.high.annall import *\n",
    "pk/low/an_annall.py": "from pk.high.facade_annall import shown2, other2\n",
    "pk/high/augall.py": "__all__ = ['a1']\n__all__ += ['a2']\ndef a1():\n    pass\ndef a2():\n    pass\n",
    "pk/high/facade_augall.py": "from pk.high.augall import *\n",
    "pk/low/ao_augall.py": "from pk.high.facade_augall import a1\n",
    "pk/high/augall2.py": "__all__ = ['a1']\n__all__ += ['a2']\n__all__ = ['a1']\ndef a1():\n    pass\n",
    "pk/high/facade_augall2.py": "from pk.high.augall2 import *\n",
    "pk/low/ao2_augall.py": "from pk.high.facade_augall2 import a1\n",
    "pk/high/tuplebind.py": "from pk.high.impl import direct\na, direct = 1, 2\n",
    "pk/low/aq_tuple.py": "from pk.high.tuplebind import direct\n",
    "pk/high/listbind.py": "from pk.high.impl import direct\n[direct] = [1]\n",
    "pk/low/ar_list.py": "from pk.high.listbind import direct\n",
    "pk/high/starbind.py": "from pk.high.impl import direct\nfirst, *direct = (1, 2)\n",
    "pk/low/as_starred.py": "from pk.high.starbind import direct\n",
    "pk/high/augbind.py": "from pk.high.impl import direct\ndirect += 1\n",
    "pk/low/at_aug.py": "from pk.high.augbind import direct\n",
    "pk/high/lambdawalrus.py": "from pk.high.impl import direct\nf = lambda: (direct := 1)\n",
    "pk/low/au_lambda.py": "from pk.high.lambdawalrus import direct\n",
    "pk/high/compwalrus.py": "from pk.high.impl import direct\n[(direct := x) for x in ()]\n",
    "pk/low/av_comp.py": "from pk.high.compwalrus import direct\n",
    "pk/high/testwalrus.py": "from pk.high.impl import direct\nif (direct := 1):\n    pass\n",
    "pk/low/aw_test.py": "from pk.high.testwalrus import direct\n",
    "pk/high/defaultwalrus.py": "from pk.high.impl import direct\ndef f(x=(direct := 1)):\n    pass\n",
    "pk/low/ax_default.py": "from pk.high.defaultwalrus import direct\n",
    "pk/high/delbind.py": "from pk.high.impl import direct\ndel direct\n",
    "pk/low/ay_del.py": "from pk.high.delbind import direct\n",
    "pk/high/matchcap.py": "from pk.high.impl import direct\nmatch 1:\n    case direct:\n        pass\n",
    "pk/low/az_matchcap.py": "from pk.high.matchcap import direct\n",
    "pk/high/chained.py": "a = direct = 1\n",
    "pk/low/ba_chained.py": "from pk.high.chained import direct\n",
    "pk/high/annonly.py": "from pk.high.impl import direct\ndirect: int\n",
    "pk/low/bb_annonly.py": "from pk.high.annonly import direct\n",
    "pk/high/typealias.py": "type direct = int\n",
    "pk/low/bc_typealias.py": "from pk.high.typealias import direct\n",
    "pk/high/allslice.py": "__all__ = ['direct']\n__all__[:] = ['direct']\ndef direct():\n    pass\n",
    "pk/high/facade_allslice.py": "from pk.high.allslice import *\n",
    "pk/low/bd_allslice.py": "from pk.high.facade_allslice import direct\n",
    "pk/high/allsort.py": "__all__ = ['direct']\n__all__.sort()\ndef direct():\n    pass\n",
    "pk/high/facade_allsort.py": "from pk.high.allsort import *\n",
    "pk/low/be_allsort.py": "from pk.high.facade_allsort import direct\n",
    "pk/high/ss/__init__.py": "from .impl import *\n",
    "pk/high/ss/impl.py": "def sub():\n    pass\n",
    "pk/high/ss/sub.py": "",
    "pk/low/bf_starshadow.py": "from pk.high.ss import sub\n",
    "pk/high/ss2/__init__.py": "from .bad import *\n",
    "pk/high/ss2/bad.py": "NAME = 'x'\n__all__ = [NAME]\n",
    "pk/high/ss2/sub.py": "",
    "pk/low/bg_starshadow_unread.py": "from pk.high.ss2 import sub\n",
    "pk/high/tcand.py": "from typing import TYPE_CHECKING\nX = 1\nif TYPE_CHECKING and X:\n    def direct():\n        pass\n",
    "pk/low/bh_tcand.py": "from pk.high.tcand import direct\n",
    "pk/high/maprest.py": "from pk.high.impl import direct\nmatch {}:\n    case {**direct}:\n        pass\n",
    "pk/low/bi_maprest.py": "from pk.high.maprest import direct\n",
    "pk/high/forimport.py": "def os():\n    pass\nfor _ in (1,):\n    import os.path\n",
    "pk/low/bj_forimport.py": "from pk.high.forimport import os\n",
    "pk/high/forstar.py": "def direct():\n    pass\nfor _ in (1,):\n    from pk.high.impl import *\n",
    "pk/low/bk_forstar.py": "from pk.high.forstar import direct\n",
    "pk/high/forprivate.py": "def _private():\n    pass\nfor _ in (1,):\n    from pk.high.impl import _private\n",
    "pk/low/bl_forprivate.py": "from pk.high.forprivate import _private\n",
    "pk/high/exctype.py": "from pk.high.impl import direct\ntry:\n    pass\nexcept (direct := ValueError):\n    pass\n",
    "pk/low/bm_exctype.py": "from pk.high.exctype import direct\n",
    "pk/high/nameall.py": "NAMES = ['direct']\n__all__ = NAMES\ndef direct():\n    pass\n",
    "pk/high/facade_nameall.py": "from pk.high.nameall import *\n",
    "pk/low/bn_nameall.py": "from pk.high.facade_nameall import direct\n",
    "pk/high/sub3/__init__.py": "OTHER = 1\n",
    "pk/high/sub3/leaf3.py": "",
    "pk/low/bo_sub3.py": "from pk.high.sub3 import leaf3\n",
    "pk/high/selfimp/__init__.py": "FLAG = 1\nfrom . import FLAG\n",
    "pk/low/bp_selfimp.py": "from pk.high.selfimp import FLAG\n",
    "pk/high/fordef.py": "from pk.high.impl import direct\nfor _ in (1,):\n    def direct():\n        pass\n",
    "pk/low/bq_fordef.py": "from pk.high.fordef import direct\n",
    "pk/low/br_missing.py": "from pk.high.missing import thing\n",
    "pk/high/bareexc.py": "try:\n    from pk.high.impl import direct\nexcept:\n    direct = None\n",
    "pk/low/bs_bareexc.py": "from pk.high.bareexc import direct\n",
    "pk/low/bt_starchain.py": "from pk.high.facade_badall import *\n",
    "pk/low/bu_starchain_ok.py": "from pk.high.facade import *\n",
    "pk/high/trystartc.py": "def direct():\n    pass\ntry:\n    from typing import TYPE_CHECKING\n    if TYPE_CHECKING:\n        from pk.high.impl import direct\nexcept* ValueError:\n    pass\n",
    "pk/low/bv_trystartc.py": "from pk.high.trystartc import direct\n",
    "pk/high/excas.py": "from pk.high.impl import direct\ntry:\n    pass\nexcept Exception as direct:\n    pass\n",
    "pk/low/ap_excas.py": "from pk.high.excas import direct\n",
    "pk/high/sortall.py": "__all__ = ['direct']\nfrom pk.high.impl import direct\n__all__ = sorted(__all__)\n",
    "pk/low/bw_sortall.py": "from pk.high.sortall import __all__\n",
    "pk/high/assertall.py": "__all__ = ['direct']\nfrom pk.high.impl import direct\nassert __all__\n",
    "pk/low/bx_assertall.py": "from pk.high.assertall import __all__\n",
    "pk/low/asfrom.py": "def renamed():\n    pass\nfrom pk.high.impl import direct as renamed\n",
    "pk/low/by_asfrom.py": "from pk.low.asfrom import renamed\n",
    "pk/low/asimport.py": "def hi():\n    pass\nimport pk.high.impl as hi\n",
    "pk/low/bz_asimport.py": "from pk.low.asimport import hi\n",
    "pk/low/finally_.py": "def direct():\n    pass\ntry:\n    pass\nfinally:\n    from pk.high.impl import direct\n",
    "pk/low/ca_finally.py": "from pk.low.finally_ import direct\n",
}  # fmt: skip
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
    ("pk.low.h_listed", "shown"): ("pk.high.listed_all", "module"),  # 8: a name in a static __all__ ...
    ("pk.low.h_listed", "unlisted"): ("not found", "module"),  # ... and a public name outside it is not exported
    ("pk.low.i_rebind", "direct"): ("pk.high.rebind", "module"),  # 9: imported and then redefined
    ("pk.low.j_cond", "direct"): ("ambiguous", "module"),  # 10: a conditional binding in try
    ("pk.low.k_consumer_star", "direct"): ("pk.high.impl", "module"),  # 11: a star import of a re-exporting module
    # 12 is a path edge, below.
    ("pk.low.m_submodule", "leaf"): ("pk.high.sub.leaf", "module"),  # 13: `from . import submodule`
    ("pk.low.n_ring_pkg", "FLAG"): ("ambiguous", "module"),  # 14: a ring ended by a conditional binding, package side
    ("pk.low.o_ring_part", "FLAG"): ("ambiguous", "module"),  # 15: the same ring, submodule side
    ("pk.low.p_pure_loop", "spin"): ("ambiguous", "module"),  # 16: a ring no binding ends; it terminates
    ("pk.low.q_ring_def", "FLAG"): ("pk.high.ring2", "module"),  # 17: a ring ended by an unconditional definition
    ("pk.low.r_typeonly", "direct"): ("not found", "module"),  # bound only under TYPE_CHECKING: never at run time
    ("pk.low.s_private", "_private"): ("not found", "module"),  # an underscore name is not star-exported
    ("pk.low.t_class", "Klass"): ("pk.high.impl", "module"),  # a class binds its name
    ("pk.low.u_ifbind", "direct"): ("ambiguous", "module"),  # bound only in an `if` body
    ("pk.low.v_elsebind", "direct"): ("ambiguous", "module"),  # bound only in an `else`
    ("pk.low.w_trybind", "direct"): ("ambiguous", "module"),  # bound only in a `try` body
    ("pk.low.x_withbind", "direct"): ("ambiguous", "module"),  # bound only in a `with` body
    ("pk.low.y_badall", "direct"): ("ambiguous", "module"),  # an `__all__` that is not all string literals
    ("pk.low.z_condall", "direct"): ("ambiguous", "module"),  # an `__all__` bound under a condition
    ("pk.low.aa_unbound_sub", "leaf2"): ("pk.high.sub2.leaf2", "module"),  # a submodule its package never binds
    ("pk.low.ab_ext", "path"): ("os", "module"),  # an owner outside the package is not followed
    ("pk.low.ac_dotted", "os"): ("os", "module"),  # `import os.path` binds `os`
    ("pk.low.ad_starring", "spin"): ("ambiguous", "module"),  # a ring of star imports terminates
    ("pk.low.ae_attr", "Klass"): ("pk.high.impl", "module"),  # an attribute target binds no name
    ("pk.low.af_subs", "TABLE"): ("pk.high.impl", "module"),  # a subscript target binds no name ...
    ("pk.low.af_subs", "direct"): ("pk.high.impl", "module"),  # ... and neither does a name read in its slice
    ("pk.low.ag_for", "direct"): ("ambiguous", "module"),  # bound in the body of a module-level `for`
    ("pk.low.ah_while", "direct"): ("ambiguous", "module"),  # ... of a `while`
    ("pk.low.ai_match", "direct"): ("ambiguous", "module"),  # ... of a `match` case
    ("pk.low.aj_withas", "direct"): ("ambiguous", "module"),  # rebound by `with ... as`
    ("pk.low.ak_walrus", "direct"): ("ambiguous", "module"),  # a walrus outside the closed list: ambiguous
    ("pk.low.al_fortarget", "direct"): ("ambiguous", "module"),  # rebound by a `for` target
    ("pk.low.am_tcelse", "direct"): ("pk.high.tcelse", "module"),  # the else of TYPE_CHECKING always runs
    ("pk.low.an_annall", "shown2"): ("pk.high.annall", "module"),  # an annotated static `__all__` is read ...
    ("pk.low.an_annall", "other2"): ("not found", "module"),  # ... and limits the export
    ("pk.low.ao_augall", "a1"): ("ambiguous", "module"),  # `__all__ +=` changes it: ambiguous
    ("pk.low.ap_excas", "direct"): ("ambiguous", "module"),  # rebound by `except ... as`
    ("pk.low.ao2_augall", "a1"): ("ambiguous", "module"),  # ... even when a plain assignment follows, as for `.extend`
    ("pk.low.aq_tuple", "direct"): ("ambiguous", "module"),  # a tuple target
    ("pk.low.ar_list", "direct"): ("ambiguous", "module"),  # a list target
    ("pk.low.as_starred", "direct"): ("ambiguous", "module"),  # a starred target
    ("pk.low.at_aug", "direct"): ("ambiguous", "module"),  # an augmented assignment
    ("pk.low.au_lambda", "direct"): ("ambiguous", "module"),  # a walrus inside a lambda: an over-report
    ("pk.low.av_comp", "direct"): ("ambiguous", "module"),  # a walrus inside a comprehension
    ("pk.low.aw_test", "direct"): ("ambiguous", "module"),  # a walrus in an `if` test
    ("pk.low.ax_default", "direct"): ("ambiguous", "module"),  # a walrus in a def's default
    ("pk.low.ay_del", "direct"): ("ambiguous", "module"),  # `del`
    ("pk.low.az_matchcap", "direct"): ("ambiguous", "module"),  # a match capture
    ("pk.low.ba_chained", "direct"): ("pk.high.chained", "module"),  # a chained assignment of names: closed list
    ("pk.low.bb_annonly", "direct"): ("pk.high.impl", "module"),  # an annotation with no value binds nothing
    ("pk.low.bc_typealias", "direct"): ("pk.high.typealias", "module"),  # `type NAME = ...`: closed list
    ("pk.low.bd_allslice", "direct"): ("ambiguous", "module"),  # `__all__[:] = ...` changes it
    ("pk.low.be_allsort", "direct"): ("ambiguous", "module"),  # `__all__.sort()` changes it
    ("pk.low.bf_starshadow", "sub"): ("pk.high.ss.impl", "module"),  # a star import shadows a submodule's name
    ("pk.low.bg_starshadow_unread", "sub"): ("ambiguous", "module"),  # ... with an unreadable __all__: ambiguous
    ("pk.low.bh_tcand", "direct"): ("ambiguous", "module"),  # `TYPE_CHECKING and X`: an ordinary condition
    ("pk.low.bi_maprest", "direct"): ("ambiguous", "module"),  # a match mapping's rest
    ("pk.low.bj_forimport", "os"): ("ambiguous", "module"),  # `import os.path` inside a `for` rebinds `os`
    ("pk.low.bk_forstar", "direct"): ("ambiguous", "module"),  # a star import inside a `for`
    ("pk.low.bl_forprivate", "_private"): ("ambiguous", "module"),  # an underscore name imported inside a `for`
    ("pk.low.bm_exctype", "direct"): ("ambiguous", "module"),  # a walrus in an `except` clause's type
    ("pk.low.bn_nameall", "direct"): ("ambiguous", "module"),  # an `__all__` bound to a name, not a literal
    ("pk.low.bo_sub3", "leaf3"): ("pk.high.sub3.leaf3", "module"),  # an unbound submodule beside other bindings
    ("pk.low.bp_selfimp", "FLAG"): ("pk.high.selfimp", "module"),  # `from . import FLAG` of an earlier binding
    ("pk.low.bq_fordef", "direct"): ("ambiguous", "module"),  # a def inside a `for` body
    ("pk.low.br_missing", "thing"): ("not found", "module"),  # an import from a module that does not exist
    ("pk.low.bs_bareexc", "direct"): ("ambiguous", "module"),  # a bare `except` handler
    ("pk.low.bu_starchain_ok", "starred"): ("pk.high.impl", "module"),  # a star import of a star-importing module
    ("pk.low.bv_trystartc", "direct"): (
        "pk.high.trystartc",
        "module",
    ),  # TYPE_CHECKING inside `try ... except*` binds nothing
    ("pk.low.bw_sortall", "__all__"): ("pk.high.sortall", "module"),  # `__all__` imported from a module that reads it
    ("pk.low.bx_assertall", "__all__"): ("pk.high.assertall", "module"),  # ... in a statement that binds nothing
    ("pk.low.by_asfrom", "renamed"): ("pk.high.impl", "module"),  # `from M import n as x` binds x
    ("pk.low.bz_asimport", "hi"): ("pk.high.impl", "module"),  # `import a.b as x` binds x to a.b
    ("pk.low.ca_finally", "direct"): ("ambiguous", "module"),  # bound in a `finally`
}

# The fourteen statement shapes the scanner of tests/unit/test_utils_does_not_import_alg.py held until it was deleted
# (#2938), then further ones, with the position this check gives each. One differs from that scanner: it called the
# body of `if False:` dead, while this check measures the direction of source dependencies, not loading, so the import
# counts at module level.
_I = "from pk.high.impl import direct"
_TC = "from typing import TYPE_CHECKING\n"
STATEMENT_SHAPES = (
    ("try body", f"try:\n    {_I}\nexcept ImportError:\n    pass\n", "module"),
    ("except handler", f"try:\n    pass\nexcept ImportError:\n    {_I}\n", "module"),
    ("try finally", f"try:\n    pass\nfinally:\n    {_I}\n", "module"),
    ("with", f"import contextlib\nwith contextlib.suppress(ImportError):\n    {_I}\n", "module"),
    ("module-level for", f"for _ in (1,):\n    {_I}\n", "module"),
    ("if not TYPE_CHECKING", f"{_TC}if not TYPE_CHECKING:\n    {_I}\n", "module"),
    ("version guard", f"import sys\nif sys.version_info >= (3, 12):\n    {_I}\n", "module"),
    ("match case", f"match 1:\n    case 1:\n        {_I}\n", "module"),
    ("else of TYPE_CHECKING", f"{_TC}if TYPE_CHECKING:\n    pass\nelse:\n    {_I}\n", "module"),
    ("class body", f"class C:\n    {_I}\n", "module"),
    ("if TYPE_CHECKING", f"{_TC}if TYPE_CHECKING:\n    {_I}\n", "type"),
    ("function body", f"def f():\n    {_I}\n", "function"),
    ("if False", f"if False:\n    {_I}\n", "module"),  # the old test: dead
    ("else of if False", f"if False:\n    pass\nelse:\n    {_I}\n", "module"),
    ("__main__ block", f"if __name__ == '__main__':\n    {_I}\n", "module"),
    ("an `if` inside TYPE_CHECKING", f"{_TC}X = 1\nif TYPE_CHECKING:\n    if X:\n        {_I}\n", "type"),
    ("an import statement in a function", "def f():\n    import pk.high.impl\n", "function"),
    ("async def body", f"async def f():\n    {_I}\n", "function"),
    ("TYPE_CHECKING inside a function", f"def f():\n    {_TC.strip()}\n    if TYPE_CHECKING:\n        {_I}\n", "type"),
    ("typing.TYPE_CHECKING", f"import typing\nif typing.TYPE_CHECKING:\n    {_I}\n", "type"),
    ("TYPE_CHECKING and X: an ordinary condition", f"{_TC}X = 1\nif TYPE_CHECKING and X:\n    {_I}\n", "module"),
    ("TYPE_CHECKING is True: an ordinary condition", f"{_TC}if TYPE_CHECKING is True:\n    {_I}\n", "module"),
    (
        "an aliased TYPE_CHECKING: an ordinary condition",
        f"from typing import TYPE_CHECKING as TC\nif TC:\n    {_I}\n",
        "module",
    ),
    (
        "else of if not TYPE_CHECKING: an ordinary condition",
        f"{_TC}if not TYPE_CHECKING:\n    pass\nelse:\n    {_I}\n",
        "module",
    ),
)

# (importing module, is a package, level, module named, expected absolute module): the five relative cases the deleted
# scanner of tests/unit/test_utils_does_not_import_alg.py held, each also checked against importlib.util.resolve_name
# below, and two beyond the top-level package, which Python refuses.
RELATIVE_CASES = (
    ("mfgarchon.utils.numerical", True, 3, "alg.base_solver", "mfgarchon.alg.base_solver"),
    ("mfgarchon.utils.adjoint_validation", False, 2, "alg.base_solver", "mfgarchon.alg.base_solver"),
    ("mfgarchon.utils.convergence.convergence_monitors", False, 3, "alg.base_solver", "mfgarchon.alg.base_solver"),
    ("mfgarchon.utils.numerical.particle.sampling", False, 4, "alg.base_solver", "mfgarchon.alg.base_solver"),
    ("mfgarchon.utils.numerical", True, 1, "particle", "mfgarchon.utils.numerical.particle"),
    ("pk.low.deep", False, 4, None, ""),
    ("pk.low.deep", False, 3, "x", ""),
)

# The longest prefix wins, on dotted boundaries.
NESTED_TABLE = {"low": 0, "low.cli": 1, "high": 1, "(root)": 0}
LAYER_CASES = (
    ("pk.low.cli", 1),
    ("pk.low.cli.sub", 1),
    ("pk.low.client", 0),
    ("pk.low", 0),
    ("pk", 0),
    ("pk.other", None),
    ("pkx.low", None),
)


def write_package(root: str, files: dict[str, str | bytes]) -> None:
    for rel, content in files.items():
        path = Path(root, rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")


def report(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'} {label}{': ' + detail if detail and not ok else ''}")
    return ok


def self_test() -> bool:
    ok = True
    for label, section in (
        ("parser cases", parser_section),
        ("path targets", path_target_section),
        ("statement shapes", shapes_section),
        ("relative imports", relative_section),
        ("layer table", layer_section),
        ("mechanism cases", mechanism_section),
        ("command line", command_line_section),
    ):
        print(f"{label}:")
        try:
            ok &= section()
        except Exception as exc:  # a crash is a failed case, reported as one
            ok &= report(f"the {label} section", False, f"crashed: {exc!r}")
    return ok


def parser_section() -> bool:
    ok = True
    layer_of = make_layer_of(PK_TABLE, "pk")
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
        ok &= report(
            "pk.low.y2_starbad: a star import whose __all__ cannot be read is an unresolved owner",
            ("pk.low.y2_starbad", "pk.high.badall", "*", "module") in found.unresolved,
        )
        ok &= report(
            "pk.low.bt_starchain: ... and so is a star import of a module that star-imports one",
            ("pk.low.bt_starchain", "pk.high.facade_badall", "*", "module") in found.unresolved,
        )
    return ok


# #2938's path-target ruling: `from P import name` adds the path edge (importer, P) unless every name is, in P, the
# submodule object. A package `pt` with `pt.low` below `pt.high`; each `pt.high.pN` a package with a submodule `sub`.
PATH_FILES: dict[str, str] = {
    "pt/__init__.py": "",
    "pt/low/__init__.py": "",
    "pt/high/__init__.py": "",
    "pt/high/p1/__init__.py": "",
    "pt/high/p1/sub.py": "X = 1\n",
    "pt/low/i_sub.py": "from pt.high.p1 import sub\n",
    "pt/high/p2/__init__.py": "from .sub import sub\n",
    "pt/high/p2/sub.py": "def sub():\n    pass\n",
    "pt/low/ii_func.py": "from pt.high.p2 import sub\n",
    "pt/high/p3/__init__.py": "from .other import *\n",
    "pt/high/p3/other.py": "sub = 1\n",
    "pt/high/p3/sub.py": "",
    "pt/low/iii_star.py": "from pt.high.p3 import sub\n",
    "pt/high/p4/__init__.py": "from . import sub\n",
    "pt/high/p4/sub.py": "",
    "pt/low/iv_from_dot.py": "from pt.high.p4 import sub\n",
    "pt/high/p5/__init__.py": "import os\nif os.environ.get('X'):\n    from . import sub\n",
    "pt/high/p5/sub.py": "",
    "pt/low/v_cond.py": "from pt.high.p5 import sub\n",
    "pt/low/base.py": "def other():\n    pass\n",
    "pt/high/p6/__init__.py": "from pt.low.base import other\n",
    "pt/high/p6/sub.py": "",
    "pt/high/p6/sub2.py": "",
    "pt/low/vi_mixed.py": "from pt.high.p6 import sub, other, sub2\n",
    "pt/high/p7/__init__.py": "def __getattr__(name):\n    raise AttributeError(name)\n",
    "pt/high/p7/sub.py": "",
    "pt/low/vii_getattr.py": "from pt.high.p7 import sub\n",
    "pt/high/p8/__init__.py": "from .bad import *\n",
    "pt/high/p8/bad.py": "NAME = 'x'\n__all__ = [NAME]\n",
    "pt/high/p8/sub.py": "",
    "pt/low/viii_unread.py": "from pt.high.p8 import sub\n",
    "pt/high/p9/__init__.py": "from os.path import *\n",
    "pt/high/p9/join.py": "",
    "pt/low/ix_outside.py": "from pt.high.p9 import join\n",
    "pt/high/p10/__init__.py": "def __getattr__(name):\n    return 42\nfrom . import sub\n",
    "pt/high/p10/sub.py": "",
    "pt/low/x_getattr_first.py": "from pt.high.p10 import sub\n",
    "pt/high/p11/__init__.py": "from .lazy import *\n",
    "pt/high/p11/lazy.py": "__all__ = ['__getattr__']\ndef __getattr__(name):\n    return 42\n",
    "pt/high/p11/sub.py": "",
    "pt/low/xi_star_getattr.py": "from pt.high.p11 import sub\n",
    "pt/high/p12/__init__.py": "def sub():\n    pass\nfrom . import sub\n",
    "pt/high/p12/sub.py": "",
    "pt/low/xii_def_first.py": "from pt.high.p12 import sub\n",
    "pt/low/xiii_star_package.py": "from pt.high.p1 import *\n",
    "pt/high/p14/__init__.py": "from .lazy import __getattr__\n",
    "pt/high/p14/lazy.py": "def __getattr__(name):\n    return 42\n",
    "pt/high/p14/sub.py": "",
    "pt/low/xiv_getattr_imported.py": "from pt.high.p14 import sub\n",
    "pt/high/p15/__init__.py": "from .a import *\nfrom .b import *\n",
    "pt/high/p15/a.py": "sub = 1\n",
    "pt/high/p15/b.py": "X = 1\n",
    "pt/high/p15/sub.py": "",
    "pt/low/xv_first_star.py": "from pt.high.p15 import sub\n",
    "pt/high/p16/__init__.py": "import os\nif os.environ.get('X'):\n    def __getattr__(name):\n        return 42\n",
    "pt/high/p16/sub.py": "",
    "pt/low/xvi_getattr_cond.py": "from pt.high.p16 import sub\n",
    "pt/high/p17/__init__.py": "from . import other as sub\n",
    "pt/high/p17/other.py": "",
    "pt/high/p17/sub.py": "",
    "pt/low/xvii_as.py": "from pt.high.p17 import sub\n",
    "pt/high/p18.py": "X = 1\n",
    "pt/high/p18/sub.py": "",
    "pt/low/xviii_module.py": "from pt.high.p18 import sub\n",
}  # fmt: skip
# (importer, package on its path, whether the path is an edge, the row)
PATH_CASES = [
    ("pt.low.i_sub", "pt.high.p1", False, "(i) a submodule the __init__ does not bind"),
    ("pt.low.ii_func", "pt.high.p2", True, "(ii) a function re-exported from a same-named submodule"),
    ("pt.low.iii_star", "pt.high.p3", True, "(iii) a star import in the __init__ that shadows the submodule"),
    ("pt.low.iv_from_dot", "pt.high.p4", False, "(iv) `from . import sub` in the __init__"),
    ("pt.low.v_cond", "pt.high.p5", True, "(v) a conditional binding: ambiguous"),
    ("pt.low.vi_mixed", "pt.high.p6", True, "(vi) three names, the middle one not a submodule"),
    ("pt.low.vii_getattr", "pt.high.p7", True, "(vii) a module-level __getattr__ and no static binding: ambiguous"),
    ("pt.low.viii_unread", "pt.high.p8", True, "(viii) a star import whose __all__ is not a literal: ambiguous"),
    ("pt.low.ix_outside", "pt.high.p9", True, "(ix) a star import from outside the tree: ambiguous"),
    ("pt.low.x_getattr_first", "pt.high.p10", True, "(x) a __getattr__ ahead of `from . import sub`: ambiguous"),
    ("pt.low.xi_star_getattr", "pt.high.p11", True, "(xi) a __getattr__ a star import exports: ambiguous"),
    ("pt.low.xii_def_first", "pt.high.p12", True, "(xii) a def ahead of `from . import sub`"),
    ("pt.low.xiii_star_package", "pt.high.p1", True, "(xiii) a star import of a package"),
    ("pt.low.xiv_getattr_imported", "pt.high.p14", True, "(xiv) a __getattr__ bound by an import: ambiguous"),
    ("pt.low.xv_first_star", "pt.high.p15", True, "(xv) two star imports, the first exporting the name"),
    ("pt.low.xvi_getattr_cond", "pt.high.p16", True, "(xvi) a __getattr__ under a module-level `if`: ambiguous"),
    ("pt.low.xvii_as", "pt.high.p17", True, "(xvii) `from . import other as sub` in the __init__"),
    ("pt.low.xviii_module", "pt.high.p18", True, "(xviii) a module P.py beside a directory P/ with no __init__.py"),
]
# (package, name, what `Tree.binds_submodule` answers), one per name of each row: the three-way answer itself, which
# the edge alone cannot tell apart, since a False and an AMBIGUOUS name both keep the path edge.
BINDS_CASES = [
    ("pt.high.p1", "sub", True, "(i)"),
    ("pt.high.p2", "sub", False, "(ii)"),
    ("pt.high.p3", "sub", False, "(iii)"),
    ("pt.high.p4", "sub", True, "(iv)"),
    ("pt.high.p5", "sub", "ambiguous", "(v)"),
    ("pt.high.p6", "sub", True, "(vi)"),
    ("pt.high.p6", "other", False, "(vi)"),
    ("pt.high.p6", "sub2", True, "(vi)"),
    ("pt.high.p7", "sub", "ambiguous", "(vii)"),
    ("pt.high.p8", "sub", "ambiguous", "(viii)"),
    ("pt.high.p9", "join", "ambiguous", "(ix)"),
    ("pt.high.p10", "sub", "ambiguous", "(x)"),
    ("pt.high.p11", "sub", "ambiguous", "(xi)"),
    ("pt.high.p12", "sub", False, "(xii)"),
    ("pt.high.p1", "*", False, "(xiii)"),
    ("pt.high.p14", "sub", "ambiguous", "(xiv)"),
    ("pt.high.p15", "sub", False, "(xv)"),
    ("pt.high.p16", "sub", "ambiguous", "(xvi)"),
    ("pt.high.p17", "sub", False, "(xvii)"),
    ("pt.high.p18", "sub", False, "(xviii)"),
]


def path_target_section() -> bool:
    ok = True
    layer_of = make_layer_of({"low": 0, "high": 1, "(root)": 0}, "pt")
    with tempfile.TemporaryDirectory() as root:
        write_package(root, PATH_FILES)
        found = scan(root, "pt", layer_of)
        for importer, package, is_edge, row in PATH_CASES:
            got = (importer, package, "module") in found.edges
            ok &= report(f"{row}: the path is {'an' if is_edge else 'not an'} edge", got == is_edge, f"got {got}")
        ok &= report(
            "(i) the owner edge to the submodule stays",
            ("pt.low.i_sub", "pt.high.p1.sub", "module") in found.edges,
        )
        tree = Tree(root, "pt")
        for package, name, expected, row in BINDS_CASES:
            got = tree.binds_submodule(package, name)
            shown = "ambiguous" if got is AMBIGUOUS else got
            ok &= report(f"{row} binds_submodule({package}, {name!r}) is {expected}", shown == expected, f"got {shown}")
    return ok


def shapes_section() -> bool:
    ok = True
    layer_of = make_layer_of(PK_TABLE, "pk")
    with tempfile.TemporaryDirectory() as root:
        files: dict[str, str | bytes] = {
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
    return ok


def relative_section() -> bool:
    ok = True
    for importer, is_package, level, named, expected in RELATIVE_CASES:
        node = ast.ImportFrom(module=named, names=[ast.alias(name="X")], level=level)
        got = Tree.absolute(importer, is_package, node)
        ok &= report(f"{importer} level {level} {named}", got == expected, f"got {got!r}")
        if expected:
            python = importlib.util.resolve_name(
                "." * level + (named or ""), importer if is_package else importer.rpartition(".")[0]
            )
            ok &= report(f"{importer} level {level} {named}: importlib agrees", python == expected, f"got {python!r}")
    return ok


def layer_section() -> bool:
    ok = True
    nested = make_layer_of(NESTED_TABLE, "pk")
    for module, expected in LAYER_CASES:
        ok &= report(f"{module} -> {expected}", nested(module) == expected, f"got {nested(module)}")
    return ok


def mechanism_section() -> bool:
    ok = True
    base: dict[str, str | bytes] = {
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

    def run(changes: dict[str, str | bytes], table: dict[str, int]) -> list[tuple[str, tuple[str, ...], str]]:
        with tempfile.TemporaryDirectory() as root:
            write_package(root, {**base, **changes})
            return check(root, "pk", table, edges, unresolved, dynamic)

    planted = "from pk.high.impl import direct\n"
    getattr_def = "def __getattr__(name):\n    raise AttributeError(name)\n"
    # (label, files changed, table, expected (kind, key) reports, the location the last report must name)
    cases = (
        ("everything listed: nothing reported (the control)", {}, PK_TABLE, [], None),
        (
            "a planted upward import is reported, its location once",
            {"pk/low/planted.py": planted},
            PK_TABLE,
            [("edge", ("pk.low.planted", "pk.high.impl", "module"))],
            "pk/low/planted.py:1: pk.low.planted imports pk.high.impl (module), base -> numerics",
        ),
        (
            "a relative import is resolved from the importing module",
            {"pk/low/rel.py": "from ..high.impl import direct\n"},
            PK_TABLE,
            [("edge", ("pk.low.rel", "pk.high.impl", "module"))],
            "pk/low/rel.py:1",
        ),
        (
            "an import of the root package is an edge when the root is upper",
            {"pk/low/rootimp.py": "import pk\n"},
            {**PK_TABLE, "(root)": 1},
            [("edge", ("pk.low.rootimp", "pk", "module"))],
            "pk/low/rootimp.py:1",
        ),
        ("a file that is not .py is not read", {"pk/low/notes.txt": "not python ("}, PK_TABLE, [], None),
        (
            "an `import` statement is reported",
            {"pk/low/planted.py": "import pk.high.impl\n"},
            PK_TABLE,
            [("edge", ("pk.low.planted", "pk.high.impl", "module"))],
            "pk/low/planted.py:1",
        ),
        ("a downward import is not reported", {"pk/high/down.py": "import pk.low.cond\n"}, PK_TABLE, [], None),
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
            "pk/low/planted.py:2 (ambiguous)",
        ),
        (
            "a name imported twice on one line is reported at that line once",
            {"pk/low/planted.py": "from pk.low.cond import FLAG, FLAG\n"},
            PK_TABLE,
            [("unresolved", ("pk.low.planted", "pk.low.cond", "FLAG", "module"))],
            "pk/low/planted.py:1 (ambiguous)",
        ),
        (
            "an unresolved owner listed only at another position is reported",
            {"pk/low/flagged.py": "from pk.low.cond import FLAG\ndef f():\n    from pk.low.cond import FLAG\n"},
            PK_TABLE,
            [("unresolved", ("pk.low.flagged", "pk.low.cond", "FLAG", "function"))],
            "pk/low/flagged.py:3",
        ),
        (
            "a listed unresolved owner whose import is deleted is reported stale",
            {"pk/low/flagged.py": ""},
            PK_TABLE,
            [("stale unresolved", ("pk.low.flagged", "pk.low.cond", "FLAG", "module"))],
            "UNRESOLVED_EXCEPTIONS",
        ),
        (
            "a module-level __getattr__ outside the list is reported",
            {"pk/low/lazy.py": getattr_def},
            PK_TABLE,
            [("dynamic", ("pk.low.lazy", "__getattr__"))],
            "pk/low/lazy.py:1",
        ),
        (
            "a __getattr__ under a module-level `if` is reported",
            {"pk/low/lazy.py": "import sys\nif sys.platform:\n    " + getattr_def.replace("\n    ", "\n        ")},
            PK_TABLE,
            [("dynamic", ("pk.low.lazy", "__getattr__"))],
            "pk/low/lazy.py:3",
        ),
        (
            "a __getattr__ under a module-level `try` is reported",
            {
                "pk/low/lazy.py": "try:\n    "
                + getattr_def.replace("\n    ", "\n        ")
                + "except Exception:\n    pass\n"
            },
            PK_TABLE,
            [("dynamic", ("pk.low.lazy", "__getattr__"))],
            "pk/low/lazy.py:2",
        ),
        (
            "a __getattr__ bound by assignment is reported",
            {"pk/low/lazy.py": "def _lazy(name):\n    raise AttributeError(name)\n__getattr__ = _lazy\n"},
            PK_TABLE,
            [("dynamic", ("pk.low.lazy", "__getattr__"))],
            "pk/low/lazy.py:3",
        ),
        (
            "an `__import__` call is reported",
            {"pk/low/dunder.py": "x = __import__('pk.high.impl')\n"},
            PK_TABLE,
            [("dynamic", ("pk.low.dunder", "__import__"))],
            "pk/low/dunder.py:1",
        ),
        (
            "an aliased import_module is reported",
            {"pk/low/alias.py": "from importlib import import_module as im\nim('pk.high.impl')\n"},
            PK_TABLE,
            [("dynamic", ("pk.low.alias", "import_module"))],
            "pk/low/alias.py:1",
        ),
        (
            "a stored reference to import_module is reported",
            {"pk/low/stored.py": "import importlib\n_load = importlib.import_module\n"},
            PK_TABLE,
            [("dynamic", ("pk.low.stored", "import_module"))],
            "pk/low/stored.py:2",
        ),
        (
            "import_module reached through getattr is reported",
            {"pk/low/viagetattr.py": "import importlib\ngetattr(importlib, 'import_module')('x')\n"},
            PK_TABLE,
            [("dynamic", ("pk.low.viagetattr", "import_module"))],
            "pk/low/viagetattr.py:2",
        ),
        (
            "builtins.__import__ is reported",
            {"pk/low/builtin.py": "import builtins\nbuiltins.__import__('x')\n"},
            PK_TABLE,
            [("dynamic", ("pk.low.builtin", "__import__"))],
            "pk/low/builtin.py:2",
        ),
        (
            "a __getattr__ bound by an import is reported",
            {"pk/low/lazy.py": "from pk.low.cond import __getattr__\n"},
            PK_TABLE,
            [
                ("unresolved", ("pk.low.lazy", "pk.low.cond", "__getattr__", "module")),
                ("dynamic", ("pk.low.lazy", "__getattr__")),
            ],
            "pk/low/lazy.py:1",
        ),
        (
            "a class method named __getattr__ is not reported",
            {"pk/low/klass.py": "class K:\n    " + getattr_def.replace("\n    ", "\n        ")},
            PK_TABLE,
            [],
            None,
        ),
        (
            "a method named import_module is reported: the match is wide, and over-reports",
            {
                "pk/low/method.py": "class L:\n    def import_module(self, n):\n        return n\nL().import_module('x')\n"
            },
            PK_TABLE,
            [("dynamic", ("pk.low.method", "import_module"))],
            "pk/low/method.py:4",
        ),
        (
            "a listed dynamic site that is deleted is reported stale",
            {"pk/low/dyn.py": ""},
            PK_TABLE,
            [("stale dynamic", ("pk.low.dyn", "import_module"))],
            "DYNAMIC_SITES",
        ),
        (
            "a file with a BOM is read",
            {"pk/low/bom.py": "\ufeff" + planted},
            PK_TABLE,
            [("edge", ("pk.low.bom", "pk.high.impl", "module"))],
            "pk/low/bom.py:1",
        ),
        (
            "a file with a latin-1 coding cookie is read",
            {"pk/low/latin.py": ("# -*- coding: latin-1 -*-\nNAME = '\xe9'\n" + planted).encode("latin-1")},
            PK_TABLE,
            [("edge", ("pk.low.latin", "pk.high.impl", "module"))],
            "pk/low/latin.py:3",
        ),
        (
            "a layer-table prefix that matches no module is reported",
            {},
            {**PK_TABLE, "gone": 1},
            [("stale prefix", ("gone",))],
            "LAYERS",
        ),
        (
            "a prefix kept alive only by a longer name is reported, on a dotted boundary",
            {"pk/low/client.py": ""},
            NESTED_TABLE,
            [("stale prefix", ("low.cli",))],
            "LAYERS",
        ),
        (
            "a module no prefix places is reported",
            {"pk/side.py": "from pk.high.impl import direct\n"},
            {"low": 0, "high": 1},
            [("no layer", ("pk",)), ("no layer", ("pk.side",))],
            "pk/side.py",
        ),
    )
    for label, changes, table, expected, location in cases:
        try:
            got = run(changes, table)
        except Exception as exc:  # a crash is a failed case, reported as one
            ok &= report(label, False, f"crashed: {exc!r}")
            continue
        kinds = [(kind, key) for kind, key, _ in got]
        named = location is None or (bool(got) and got[-1][2].count(location) == 1)
        ok &= report(label, kinds == expected and named, f"expected {expected} naming {location}, got {got}")
    return ok


def command_line_section() -> bool:
    """`main` on a root with no package, and on a root whose package reports. Its OK branch needs a tree that
    satisfies the repository's own lists, which a temporary tree does not: the run on the repository covers it."""
    ok = True
    with tempfile.TemporaryDirectory() as root:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["--root", root])
        ok &= report(
            "a root with no mfgarchon/ exits 1 and says so",
            code == 1 and "holds no" in out.getvalue(),
            f"exit {code}, printed {out.getvalue()!r}",
        )
        Path(root, PACKAGE).mkdir()
        Path(root, PACKAGE, "__init__.py").write_text("")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["--root", root])
        printed = out.getvalue().splitlines()
        ok &= report(
            "a root whose package reports exits 1 and prints each report",
            code == 1 and any("no longer" in line for line in printed) and not any(x.startswith("OK") for x in printed),
            f"exit {code}, printed {printed[:3]}",
        )
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parent.parent),
        help="the directory holding mfgarchon/ (default: this repository)",
    )
    parser.add_argument("--self-test", action="store_true", help="run the parser and mechanism cases")
    args = parser.parse_args(argv)
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
