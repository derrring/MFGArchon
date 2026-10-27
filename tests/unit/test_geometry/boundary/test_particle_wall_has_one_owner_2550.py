"""C6's structural guard: a box wall rule on positions has one owner per rule (#2550; CONVENTIONS.md § *Particles at a
wall*).

The owners are `corner.reflect_positions` (the fold), `periodic.wrap_positions` (the wrap) and `corner.absorb_at_walls`
with `corner.walls_reached` (the absorbing wall). This guard fails if a function applies a box wall rule to positions
with its own arithmetic, as `Hyperrectangle.apply_boundary_conditions` and both 1-D helper pairs did before #2810 and
#2811 (the manifest rule, `scripts/oracle_pins.txt`; #2512 comment 6073690080).

**Population.** Every function and lambda under `mfgarchon/`. Module-level code is not scanned, so a module's
``if __name__ == "__main__":`` self-test is outside it.

**A wall rule's arithmetic** is one of the shapes the census of #2550's class A found:
- P1, a modulo of a difference: ``(x - lo) % m``, or ``s % m`` where ``s`` was assigned a difference in the same
  function. A difference with a literal on either side is not one, so ``(i - 1) % n`` is not;
- P2, a ``mod``, ``remainder`` or ``fmod`` call, from any module, whose first argument is such a difference;
- P3, a mirror about a bound, ``2 * <bound> - x``;
- P4, a shift by a span, ``x -= <span>`` or ``x = x + <span>``: the single-shift wrap, and the per-period loop;
- P5, a clip to bounds, ``clip(x, <bound>, <bound>)`` or ``clamp(...)``, or a nested ``minimum(maximum(x, <bound>),
  <bound>)`` (``min``/``max`` too), as the 1-D GPU helper clamped on a backend with no ``clip``: a clamp standing for
  absorption.
A *bound* is a name, attribute or subscript whose name says lo, hi, min, max, low, high, lower, upper, left, right,
wall, bound(s) or box, alone or as a ``_``-separated part. A *span* is one named span, width, length, period, L, Lx,
Ly, extent or box/domain size, or a difference of two bounds.

**What they keep out, by construction:** an index wrap ``(i + 1) % n`` or ``(i - 1) % n``; a counter ``k % 10``; an
angle ``theta % (2 * np.pi)``; a mirror about a name outside the bound vocabulary. The contact loop, the path rule's
owner, mirrors about a box bound at a corner, so it is a site, and is allowed by name.

**Allowed**, each by file and qualified name (``Class.method``, as ``__qualname__`` spells it), so a moved line does not
reopen the question and a function of the same name elsewhere in the file is not covered. An allowance that covers no
site fails the guard, so a function routed to an owner cannot leave its name behind for the next one.

**Known blind spots** (the census's shapes are the criterion, as for #2512's guards, since an AST guard cannot be
complete over Python syntax):
- an absorber that drops the rows past the box through a mask, as `ParticleApplicator`'s and `MeshfreeApplicator`'s
  own copies did: it has no shape apart from a membership test. The behavioural pins of
  `test_particle_wall_owners_2550.py` hold it;
- a wrap through ``floor`` (``x - L * floor((x - lo) / L)``) or ``divmod``, and a shift inside a call,
  ``np.where(x >= hi, x - (hi - lo), x)``: the consumers' behavioural pins hold these;
- a modulo of the position itself, ``x % L`` on a box at 0 or ``x % (xmax - xmin)``: a rule for it would fire on the
  index wraps whose counts are named span or width. Also the in-place spelling, ``x -= lo`` then ``x %= L``;
- a mirror spelled ``lo + (lo - x)``, ``hi - (x - hi)`` or ``-x + 2 * hi``, a fold through ``abs``
  (``lo + abs(x - lo)``), and an assignment to the opposite face, as
  `ImplicitApplicator._wrap_to_opposite_boundary` places a field's periodic read point;
- a clip to offset bounds, ``clip(x, lo + eps, hi - eps)``, and a clip by keyword (``np.clip(x, a_min=lo, a_max=hi)``);
- a difference carried through a second name, a walrus or tuple unpacking;
- a bound or a span whose name is outside the vocabulary above, and ``<bound> * 2 - x``;
- module-level code.

**Class line B** (a reader that gives one action to every axis at a periodic corner) has no syntactic shape. It is held
by `test_a_periodic_corner_acts_per_axis` and `test_a_full_exit_turns_a_particle_back_per_axis`.
"""

from __future__ import annotations

import ast
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.structural_guard import FUNCTIONS, REPO, name_of, own_nodes, planted_scan

if TYPE_CHECKING:
    from collections.abc import Mapping

PACKAGE = Path("mfgarchon")
_BOUND = re.compile(
    r"(^|_)(lo|hi|min|max|low|high|lower|upper|left|right|wall|bound|bounds|box)(_|$|\d)|^[xyz]?(min|max)$", re.I
)
_SPAN = re.compile(r"^(span|width|length|period|periods|L|Lx|Ly|L_i|extent|box_size|domain_size)$")
_MODS = frozenset({"mod", "remainder", "fmod"})
_CLAMPS = {"minimum": "maximum", "maximum": "minimum", "min": "max", "max": "min"}

_SL = "a field's departure point, not a particle: its Dirichlet arm reads g (#2815)"
_READ_POINT = "the nearest point of the box, read for its segment; the end itself is moved by the owners"
_KDE = "images for a density estimator; no particle moves (#2573 comment 6419124057)"
ALLOWED = {
    # The owners, and the owners' device copy (#2811), which `test_particle_wall_owners_2550.py` pins against them.
    ("mfgarchon/geometry/boundary/corner/position.py", "reflect_positions"): "the fold's owner",
    ("mfgarchon/geometry/boundary/periodic.py", "wrap_positions"): "the wrap's owner",
    ("mfgarchon/utils/particle_utils.py", "apply_boundary_conditions_gpu"): "the owners' device copy",
    # The path rule's owner, the contact loop (#2733): at a corner it mirrors the step's end about the box's face.
    ("mfgarchon/alg/numerical/fp_solvers/fp_particle_bc.py", "reflect_at_first_contacts"): "the path rule's owner",
    # The clamp #2810 deprecated; it goes with its deprecation.
    ("mfgarchon/geometry/boundary/corner/position.py", "absorb_positions"): "deprecated (#2810)",
    # The semi-Lagrangian departure-point folds.
    ("mfgarchon/alg/numerical/hjb_solvers/hjb_sl_characteristics.py", "apply_boundary_conditions_1d"): _SL,
    ("mfgarchon/alg/numerical/hjb_solvers/hjb_sl_characteristics.py", "apply_boundary_conditions_nd"): _SL,
    ("mfgarchon/alg/numerical/hjb_solvers/hjb_sl_characteristics.py", "fold_into_domain"): _SL,
    ("mfgarchon/alg/numerical/hjb_solvers/hjb_sl_characteristics.py", "reflect_into_domain"): _SL,
    # Reflection ghosts for a KDE.
    ("mfgarchon/geometry/boundary/ghost.py", "create_ghost_points_for_kde"): _KDE,
    ("mfgarchon/geometry/boundary/periodic.py", "create_reflection_ghost_points"): _KDE,
    ("mfgarchon/utils/numerical/particle/kde_boundary.py", "create_ghost_particles"): _KDE,
    # The reflection KDE's images of an evaluation point across a wall (#2556), the same images for an estimator.
    ("mfgarchon/utils/numerical/particle/kde_boundary.py", "reflection_kde"): _KDE,
    # A field's ghost value about its wall value, not a position.
    ("mfgarchon/geometry/boundary/applicator_fdm.py", "_write_wall_ghosts"): "a field ghost value",
    # A point's nearest point of the box, read for the segment that holds it, not a particle moved.
    ("mfgarchon/geometry/boundary/applicator_particle.py", "ParticleApplicator._segment_at"): _READ_POINT,
    ("mfgarchon/geometry/boundary/applicator_particle.py", "ParticleApplicator._act_per_face"): _READ_POINT,
    ("mfgarchon/geometry/boundary/conditions.py", "BoundaryConditions._segments_holding"): _READ_POINT,
    # Clips to a bound that is not a box wall rule on particles: a control's bounds, a field's bounds, a query point
    # held to the grid, a step factor, a maze door's width, and the geometry API's projections.
    ("mfgarchon/core/hamiltonian.py", "LagrangianBase.conjugate_argmax"): "a control's bounds",
    ("mfgarchon/core/hamiltonian.py", "LagrangianBase.proximal"): "a control's bounds",
    ("mfgarchon/core/hamiltonian.py", "DualHamiltonian.__call__"): "a control's bounds",
    ("mfgarchon/core/hamiltonian.py", "DualHamiltonian._find_optimal_alpha"): "a control's bounds",
    ("mfgarchon/core/hamiltonian.py", "DualLagrangian.__call__"): "a control's bounds",
    ("mfgarchon/core/hamiltonian.py", "DualLagrangian._find_optimal_p"): "a control's bounds",
    ("mfgarchon/geometry/boundary/constraints.py", "BilateralConstraint.project"): "a field's obstacle bounds",
    ("mfgarchon/geometry/boundary/applicator_interpolation.py", "InterpolationApplicator.clamp_query_points"): (
        "an interpolation query"
    ),
    ("mfgarchon/utils/acceleration/jax_utils.py", "adaptive_time_step"): "a time-step factor",
    ("mfgarchon/geometry/graph/maze_postprocessing.py", "adaptive_door_carving"): "a maze door's width",
    ("mfgarchon/geometry/graph/maze_utils.py", "compute_adaptive_door_width"): "a maze door's width",
    ("mfgarchon/geometry/base.py", "nearest_point_on_box_boundary"): "geometry API",
    ("mfgarchon/geometry/base.py", "Geometry.project_to_interior"): "geometry API",
    ("mfgarchon/geometry/graph/network_geometry.py", "NetworkGeometry.project_to_interior"): "geometry API",
    ("mfgarchon/geometry/implicit/point_cloud.py", "PointCloudGeometry.project_to_interior"): "geometry API",
}


def _tail(node: ast.AST) -> str | None:
    return _tail(node.value) if isinstance(node, ast.Subscript) else name_of(node)


def _is_bound(node: ast.AST) -> bool:
    name = _tail(node)
    return bool(name and _BOUND.search(name))


def _is_difference(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Sub)
        and not isinstance(node.left, ast.Constant)
        and not isinstance(node.right, ast.Constant)
    )


def _is_span(node: ast.AST) -> bool:
    name = _tail(node)
    if name and _SPAN.match(name):
        return True
    return _is_difference(node) and _is_bound(node.left) and _is_bound(node.right)


def _is_shift(node: ast.AST) -> bool:
    """``x -= <span>``, ``x += <span>``, or ``x = x -/+ <span>``."""
    if isinstance(node, ast.AugAssign):
        return isinstance(node.op, (ast.Add, ast.Sub)) and _is_span(node.value)
    return (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.value, ast.BinOp)
        and isinstance(node.value.op, (ast.Add, ast.Sub))
        and _is_span(node.value.right)
        and _tail(node.targets[0]) is not None
        and _tail(node.targets[0]) == _tail(node.value.left)
    )


def _is_nested_clamp(node: ast.AST) -> bool:
    """``minimum(maximum(x, <bound>), <bound>)``, or ``maximum(minimum(...))``, ``min(max(...))``, from any module."""
    outer = name_of(node.func) if isinstance(node, ast.Call) else None
    if outer not in _CLAMPS or len(node.args) != 2:
        return False
    inner, bound = node.args
    if isinstance(bound, ast.Call):
        inner, bound = bound, inner
    return (
        isinstance(inner, ast.Call)
        and name_of(inner.func) == _CLAMPS[outer]
        and len(inner.args) == 2
        and _is_bound(bound)
        and any(_is_bound(arg) for arg in inner.args)
    )


def scopes(tree: ast.Module) -> list[tuple[str, ast.AST]]:
    """Every function and lambda in ``tree`` with its qualified name, as ``__qualname__`` spells it."""
    found: list[tuple[str, ast.AST]] = []

    def visit(node: ast.AST, prefix: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, [*prefix, child.name])
            elif isinstance(child, FUNCTIONS):
                name = [*prefix, getattr(child, "name", "<lambda>")]
                found.append((".".join(name), child))
                visit(child, [*name, "<locals>"])
            else:
                visit(child, prefix)

    visit(tree, [])
    return found


def wall_rules(scope: ast.AST) -> list[tuple[str, int]]:
    """``(predicate, line)`` for each wall rule's arithmetic in ``scope`` itself."""
    nodes = list(own_nodes(scope))
    carried = {
        node.targets[0].id
        for node in nodes
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and _is_difference(node.value)
    }

    def difference(node: ast.AST) -> bool:
        return _is_difference(node) or (isinstance(node, ast.Name) and node.id in carried)

    hits = []
    for node in nodes:
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod) and difference(node.left):
            hits.append(("P1", node.lineno))
        elif isinstance(node, ast.Call) and name_of(node.func) in _MODS and node.args and difference(node.args[0]):
            hits.append(("P2", node.lineno))
        elif (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Sub)
            and isinstance(node.left, ast.BinOp)
            and isinstance(node.left.op, ast.Mult)
            and isinstance(node.left.left, ast.Constant)
            and node.left.left.value == 2
            and _is_bound(node.left.right)
        ):
            hits.append(("P3", node.lineno))
        elif _is_shift(node):
            hits.append(("P4", node.lineno))
        elif (
            isinstance(node, ast.Call)
            and name_of(node.func) in ("clip", "clamp")
            and len(node.args) >= 3
            and _is_bound(node.args[1])
            and _is_bound(node.args[2])
        ) or _is_nested_clamp(node):
            hits.append(("P5", node.lineno))
    return hits


def scan(root: Path, allowed: Mapping[tuple[str, str], str] = ALLOWED) -> tuple[list[str], list[str]]:
    """``(population, sites)`` for the package under ``root``; a site is ``path:line: in qualname [P#]``."""
    population, sites = [], []
    for path in sorted((root / PACKAGE).rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        population.append(rel)
        for qualname, node in scopes(ast.parse(path.read_text())):
            if (rel, qualname) not in allowed:
                sites.extend(f"{rel}:{line}: in {qualname} [{kind}]" for kind, line in wall_rules(node))
    return population, sorted(set(sites))


@pytest.fixture(scope="module")
def package_copy(tmp_path_factory) -> Path:
    """A copy of `mfgarchon/` to plant rivals in; each test adds a module and removes it."""
    root = tmp_path_factory.mktemp("c6_guard")
    shutil.copytree(REPO / PACKAGE, root / PACKAGE, ignore=shutil.ignore_patterns("__pycache__"))
    return root


def _planted(package_copy: Path, planted: str) -> list[str]:
    target = package_copy / PACKAGE / "geometry" / "boundary" / "_planted_rival.py"
    return [site for site in planted_scan(scan, package_copy, target, planted) if "_planted_rival.py" in site]


def test_every_wall_rule_on_positions_is_its_owners():
    population, sites = scan(REPO)
    assert "mfgarchon/geometry/boundary/periodic.py" in population, population[:5]
    assert len(population) > 100, f"the population looks truncated: {len(population)} modules"
    assert sites == [], "a box wall rule applied to positions outside its owner (#2550, C6):\n" + "\n".join(sites)


def test_every_allowance_still_covers_a_site():
    """An allowance for a function that has been routed, or is gone, would hide the next one given its name."""
    _, sites = scan(REPO, allowed={})
    covered = {(site.split(":")[0], site.split(" in ", 1)[1].split(" [")[0]) for site in sites}
    assert set(ALLOWED) <= covered, sorted(set(ALLOWED) - covered)


@pytest.mark.parametrize(
    "planted",
    [
        "def _rival(x, lo, L):\n    return lo + (x - lo) % L\n",
        "def _rival(x, lo, L):\n    s = x - lo\n    return lo + s % L\n",
        "import numpy as np\n\n\ndef _rival(x, lo, L):\n    return lo + np.mod(x - lo, L)\n",
        "import numpy as np\n\n\ndef _rival(x, lo, L):\n    return lo + np.remainder(x - lo, L)\n",
        "import numpy as np\n\n\ndef _rival(x, lo, L):\n    return lo + np.fmod(x - lo, L)\n",
        "def _rival(xp, x, lo, L):\n    return lo + xp.remainder(x - lo, L)\n",
        "import torch\n\n\ndef _rival(x, lo, L):\n    return lo + torch.remainder(x - lo, L)\n",
        "import jax.numpy as jnp\n\n\ndef _rival(x, lo, L):\n    return lo + jnp.mod(x - lo, L)\n",
        "import math\n\n\ndef _rival(x, lo, L):\n    return lo + math.fmod(x - lo, L)\n",
        "def _rival(x, lo, L):\n    y = (x - lo) % (2 * L)\n    return lo + L - abs(y - L)\n",
        "def _rival(x, xmax):\n    return 2 * xmax - x\n",
        "def _rival(x, width):\n    x -= width\n    return x\n",
        "def _rival(x, xmin, xmax):\n    while x > xmax:\n        x = x - (xmax - xmin)\n    return x\n",
        "import numpy as np\n\n\ndef _rival(particles, xmin, xmax):\n    return np.clip(particles, xmin, xmax)\n",
        "def _rival(xp, particles, xmin, xmax):\n    return xp.minimum(xp.maximum(particles, xmin), xmax)\n",
        "def _rival(x, lo, hi):\n    return min(max(x, lo), hi)\n",
        "def _rival(stop, box):\n    return 2.0 * box[0, 1] - stop[0]\n",
    ],
    ids=[
        "mod-difference",
        "mod-carried",
        "np.mod",
        "np.remainder",
        "np.fmod",
        "xp.remainder",
        "torch.remainder",
        "jnp.mod",
        "math.fmod",
        "triangle-wave",
        "mirror",
        "shift",
        "shift-by-bounds",
        "clip",
        "minimum-maximum",
        "min-max",
        "mirror-about-box",
    ],
)
def test_a_planted_rival_is_found(package_copy, planted):
    assert _planted(package_copy, planted), planted


def test_an_allowance_does_not_cover_a_namesake(package_copy):
    """An allowance names a qualified name, so a new ``__call__`` in `core/hamiltonian.py`, a file whose two
    control-bound ``__call__`` methods are allowed, is a site."""
    target = package_copy / PACKAGE / "core" / "hamiltonian.py"
    planted = (
        "class _RivalHamiltonian:\n    def __call__(self, x, xmin, xmax):\n        return np.clip(x, xmin, xmax)\n"
    )
    sites = planted_scan(scan, package_copy, target, planted)
    assert any("in _RivalHamiltonian.__call__ [P5]" in site for site in sites), sites


# Class A's members as they stood at 95b097fe, before #2810 and #2811 routed them: the shapes the guard was written for.
_OLD_NUMPY_HELPER = """import numpy as np


def apply_boundary_conditions_numpy(particles, xmin, xmax, bc_type):
    Lx = xmax - xmin
    if bc_type == "periodic":
        particles = xmin + ((particles - xmin) % Lx)
    elif bc_type == "no_flux":
        Lx = xmax - xmin
        if Lx > 1e-14:
            shifted = particles - xmin
            period = 2 * Lx
            pos_in_period = shifted % period
            in_second_half = pos_in_period > Lx
            pos_in_period[in_second_half] = period - pos_in_period[in_second_half]
            particles = xmin + pos_in_period
    elif bc_type == "dirichlet":
        particles = np.clip(particles, xmin, xmax)
    return particles
"""
_OLD_GPU_HELPER = """def apply_boundary_conditions_gpu(particles, xmin, xmax, bc_type, backend):
    xp = backend.array_module
    Lx = xmax - xmin
    if bc_type == "periodic":
        particles = xmin + ((particles - xmin) % Lx)
    elif bc_type == "no_flux":
        if Lx > 1e-14:
            shifted = particles - xmin
            period = 2 * Lx
            pos_in_period = shifted % period
            in_second_half = pos_in_period > Lx
            particles = xp.where(in_second_half, xmin + period - pos_in_period, xmin + pos_in_period)
    elif bc_type == "dirichlet":
        if hasattr(xp, "clip"):
            particles = xp.clip(particles, xmin, xmax)
        else:
            particles = xp.minimum(xp.maximum(particles, xmin), xmax)
    return particles
"""
_OLD_DOMAIN_METHOD = """import numpy as np


def apply_boundary_conditions(self, particles, bc_type="reflecting"):
    particles = particles.copy()
    if bc_type == "reflecting":
        for dim in range(self.dimension):
            xmin = self.bounds[dim, 0]
            xmax = self.bounds[dim, 1]
            Lx = xmax - xmin
            if Lx > 1e-14:
                shifted = particles[:, dim] - xmin
                period = 2 * Lx
                pos_in_period = shifted % period
                in_second_half = pos_in_period > Lx
                pos_in_period[in_second_half] = period - pos_in_period[in_second_half]
                particles[:, dim] = xmin + pos_in_period
        return particles
    elif bc_type == "absorbing":
        inside = np.all((particles >= self.bounds[:, 0]) & (particles <= self.bounds[:, 1]), axis=1)
        return particles[inside]
    elif bc_type == "periodic":
        for dim in range(self.dimension):
            width = self.bounds[dim, 1] - self.bounds[dim, 0]
            below = particles[:, dim] < self.bounds[dim, 0]
            particles[below, dim] += width
            above = particles[:, dim] > self.bounds[dim, 1]
            particles[above, dim] -= width
        return particles
"""


@pytest.mark.parametrize(
    ("planted", "kinds"),
    [
        (_OLD_NUMPY_HELPER, {"P1": 2, "P5": 1}),
        (_OLD_GPU_HELPER, {"P1": 2, "P5": 2}),
        (_OLD_DOMAIN_METHOD, {"P1": 1, "P4": 2}),
    ],
    ids=["the 1-D NumPy helper", "the 1-D GPU helper", "Hyperrectangle.apply_boundary_conditions"],
)
def test_a_removed_member_of_class_a_is_found(package_copy, planted, kinds):
    """Each line of each wall rule, by its predicate: the GPU helper's clamp on a backend with no ``clip`` (a nested
    minimum and maximum) is a site of its own, and the domain method's periodic arm (a shift by the width) is P4's
    control from code the library shipped. Its absorbing arm, a mask, is a blind spot."""
    sites = _planted(package_copy, planted)
    found: dict[str, int] = {}
    for site in sites:
        kind = site.rsplit("[", 1)[1].rstrip("]")
        found[kind] = found.get(kind, 0) + 1
    assert found == kinds, sites


@pytest.mark.parametrize(
    "planted",
    [
        "def _not_a_rival(i, n):\n    return (i + 1) % n\n",
        "def _not_a_rival(i, n):\n    return (i - 1) % n\n",
        "def _not_a_rival(k):\n    return k % 10 == 0\n",
        "import numpy as np\n\n\ndef _not_a_rival(theta):\n    return theta % (2 * np.pi)\n",
        # A mirror about a name outside the bound vocabulary. The library's own field ghost, ``2 * wall_value - u``,
        # is a site, and is allowed by name.
        "def _not_a_rival(u_b, u_i):\n    return 2 * u_b - u_i\n",
    ],
    ids=["index-wrap", "index-wrap-back", "counter", "angle", "mirror-about-a-value"],
)
def test_a_shape_that_is_not_a_wall_rule_is_not_found(package_copy, planted):
    assert not _planted(package_copy, planted), planted
