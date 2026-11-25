"""The coupled iterate's one owner (#2997, stage 1).

A coupling route iterates on U and M, each of shape ``(Nt + 1, *shape)``: ``shape`` is the grid's shape on a
Cartesian grid, and ``(n_dof,)`` on an unstructured mesh. `IterationState` decides, once for every route that holds
one:

- that shape;
- the cold start: U is u(T) at the initial density in every row, and M is the initial density at level 0 and
  `cold_start_density_at_level` at each later level (#2991);
- the warm start, checked against that shape;
- the write-back after each sweep: level 0 to the initial density, and the terminal row to the u(T) that sweep's HJB
  solve was given;
- the packing of (U, M) into one vector, for Newton;
- the order in which a level's values sit at the points a solver hands a source (`level_at_points`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .base_mfg import resolve_backend
from .fixed_point_utils import cold_start_density_at_level, preserve_initial_condition, preserve_terminal_condition

if TYPE_CHECKING:
    from mfgarchon.alg.numerical.fp_solvers.base_fp import BaseFPSolver
    from mfgarchon.core.mfg_problem import CouplingBinding, MFGProblem


def _state_shape(
    problem: MFGProblem, hjb_solver: Any, fp_solver: Any, *, reader: str, unstructured_mesh: bool
) -> tuple[int, ...]:
    """The spatial shape of one level of the iterate.

    On an unstructured mesh it is the solver's DOF count, not ``num_spatial_points`` (the vertex count): P2 FEM has more
    DOFs than vertices, and a state sized on the vertices fails deep in the update with an opaque broadcast error
    (#1489 S6). A route that has not been shown to run on a mesh passes ``unstructured_mesh=False`` and is refused here.
    """
    from mfgarchon.geometry import GeometryType
    from mfgarchon.geometry.base import CartesianGrid

    geometry = problem.geometry
    if isinstance(geometry, CartesianGrid):
        return tuple(int(n) for n in geometry.get_grid_shape())
    if unstructured_mesh and getattr(geometry, "geometry_type", None) == GeometryType.UNSTRUCTURED_MESH:
        fp_n = getattr(fp_solver, "n_dof", None)
        hjb_n = getattr(hjb_solver, "n_dof", None)
        n = fp_n if fp_n is not None else (hjb_n if hjb_n is not None else int(problem.num_spatial_points))
        return (int(n),)
    accepted = "a CartesianGrid or an unstructured mesh" if unstructured_mesh else "a CartesianGrid"
    raise ValueError(f"{reader}: the problem's geometry must be {accepted}, got {type(geometry).__name__}")


def allocate_iterate(backend: Any, shape: tuple[int, ...], reader: str) -> tuple[Any, Any]:
    """``(U, M)`` of ``shape``, refusing a backend whose arrays the coupling loop cannot write (#2250).

    The loop assigns NumPy rows into U and M by index. A JAX array is immutable and a torch tensor rejects a NumPy
    right-hand side, so allocate, then probe one write of a NumPy row: a self-assignment probe is symmetric, and torch
    passes it while failing the real write. ``backend`` is resolved again here because ``backend`` is a public
    attribute and a name assigned to it after construction reaches this call unresolved (#1922 is the capability).
    """
    backend = resolve_backend(backend, reader)
    if backend is None:
        return np.zeros(shape), np.zeros(shape)
    U = backend.zeros(shape)
    M = backend.zeros(shape)
    try:
        U[0] = np.zeros(shape[1:])
    except (TypeError, ValueError) as exc:
        raise NotImplementedError(
            f"{reader}: backend {type(backend).__name__} allocates arrays the coupling loop cannot write into "
            f"({type(exc).__name__}: {exc}). The loop assigns U and M by index at every step, so this backend cannot "
            f"carry a solve. Pass backend=None or a NumPy backend. Backend selection is tracked in #1922; this refusal "
            f"is #2250."
        ) from exc
    return U, M


class IterationState:
    """One component's iterate: its layout, cold and warm start, write-back and packing (#2997).

    Args:
        problem: The problem the route solves.
        hjb_solver, fp_solver: The route's pair. The FP solver is asked what its step prescribes at a later level
            (`cold_start_density_at_level`); the pair's DOF count sizes the state on a mesh.
        reader: The route's name, which every refusal names.
        terminal_binding: The HJB solver's binding of a terminal coupling, for ``problem.terminal_value``.
        unstructured_mesh: Whether the route runs on an unstructured mesh.
    """

    def __init__(
        self,
        problem: MFGProblem,
        hjb_solver: Any,
        fp_solver: BaseFPSolver,
        *,
        reader: str,
        terminal_binding: CouplingBinding | None,
        unstructured_mesh: bool = False,
    ) -> None:
        self.problem = problem
        self.fp_solver = fp_solver
        self.reader = reader
        self.terminal_binding = terminal_binding
        self.shape = _state_shape(problem, hjb_solver, fp_solver, reader=reader, unstructured_mesh=unstructured_mesh)
        self.solution_shape = (problem.Nt + 1, *self.shape)
        self._warm_start: tuple[np.ndarray, np.ndarray] | None = None
        # m(0) as `start` read it, for the solve that follows: the FP's initial datum and the write-back's level 0.
        self.m0: np.ndarray | None = None

    def initial_density(self) -> np.ndarray:
        """m(0): the problem's initial density, in this state's shape."""
        return np.asarray(self.problem.get_m_initial()).reshape(self.shape)

    def terminal_at(self, m_terminal: np.ndarray) -> np.ndarray:
        """u(T) at the density ``m_terminal`` (the m(T) of the HJB solve it is the terminal of), in this state's shape:
        the terminal datum's one owner, ``problem.terminal_value`` (#1642)."""
        return np.asarray(
            self.problem.terminal_value(m_terminal, binding=self.terminal_binding, reader=self.reader)
        ).reshape(self.shape)

    def set_warm_start(self, U: np.ndarray, M: np.ndarray) -> None:
        """Start the next solve from ``(U, M)``, copied, each of this state's shape ``(Nt + 1, *shape)``."""
        for name, array in (("U", U), ("M", M)):
            if np.shape(array) != self.solution_shape:
                raise ValueError(
                    f"{self.reader}: the warm start's {name} has shape {np.shape(array)}, and this problem's iterate "
                    f"has shape {self.solution_shape} (Nt + 1 levels of the state's shape {self.shape})"
                )
        self._warm_start = (np.array(U, dtype=float, copy=True), np.array(M, dtype=float, copy=True))

    def warm_start(self) -> tuple[np.ndarray, np.ndarray] | None:
        """The warm start set, or None."""
        return self._warm_start

    def clear_warm_start(self) -> None:
        self._warm_start = None

    def start(self, backend: Any = None) -> tuple[Any, Any]:
        """The first sweep's ``(U, M)``: a copy of the warm start if one is set, else the cold start.

        The cold start's U is u(T) at the initial density in every row; its M is the initial density at level 0 and
        `cold_start_density_at_level` at each later level, which holds the FP step's value at a node the step
        prescribes (#2991). Either way it reads m(0) for the solve that follows (`m0`).
        """
        m0 = self.initial_density()
        self.m0 = m0
        if self._warm_start is not None:
            U_warm, M_warm = self._warm_start
            return U_warm.copy(), M_warm.copy()
        u_terminal = self.terminal_at(m0)
        U, M = allocate_iterate(backend, self.solution_shape, self.reader)
        for n in range(self.solution_shape[0]):
            U[n] = u_terminal
        M[0] = m0
        for n in range(1, self.solution_shape[0]):
            M[n] = cold_start_density_at_level(m0, self.fp_solver, n)
        return U, M

    def write_back(self, U: Any, M: Any, u_terminal: np.ndarray) -> tuple[Any, Any]:
        """After a sweep's update: level 0 of M to the initial density, and U's terminal row to ``u_terminal``, the
        u(T) that sweep's HJB solve was given. Both in place; the arrays are returned. `start` comes first."""
        if self.m0 is None:
            raise RuntimeError(f"{self.reader}: write_back before start: the state has not read m(0) for this solve")
        return preserve_terminal_condition(U, u_terminal), preserve_initial_condition(M, self.m0)

    @staticmethod
    def level_at_points(level: Any) -> np.ndarray:
        """One time level of an iterate, of a state's shape, as its values at the geometry's points, shape ``(N,)``: for
        the coupling closures that combine levels pointwise into a source (#2997).

        The order is the geometry's enumeration of its points: a tensor grid lists them in the C order of its shape,
        axis 0 slowest (`TensorProductGrid.flatten`, as `get_multi_index` reads an index). HJB-FDM, the base HJB
        solver's timestep and FP-FDM evaluate a source at ``geometry.get_spatial_grid()`` and read its values in that
        order. On a mesh the shape is ``(n_dof,)`` and the level is already in it. A solver that evaluates a source at
        its own points, as HJB-GFDM does at its collocation points, still reads these values by index, which is #2765.
        """
        return np.reshape(level, -1, order="C")

    def pack(self, U: np.ndarray, M: np.ndarray) -> np.ndarray:
        """``(U, M)`` as one vector, U's entries first, each in C order."""
        return np.concatenate([np.ravel(U), np.ravel(M)])

    def unpack(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The inverse of `pack`."""
        size = int(np.prod(self.solution_shape))
        if np.shape(x) != (2 * size,):
            raise ValueError(f"{self.reader}: a packed state has {2 * size} entries, got shape {np.shape(x)}")
        return x[:size].reshape(self.solution_shape), x[size:].reshape(self.solution_shape)
