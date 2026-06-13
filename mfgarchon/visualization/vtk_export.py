"""
VTK export for FEM/unstructured-mesh MFG solutions via meshio (Issue #895).

Writes a mesh plus nodal field values (e.g. the value function $u(t,x)$ and the
density $m(t,x)$) to ParaView-readable ``.vtu`` (XML VTK) files:

- :func:`export_mesh_solution_vtk` — one timestep (mesh + ``{name: array(N,)}``).
- :func:`export_time_series_vtk` — an MFG solution ``(Nt+1, N)`` as a sequence of
  ``<stem>_<k>.vtu`` files plus a ParaView ``.pvd`` collection indexing them by
  time $t_k$.

The mesh argument is the FEM solution's mesh: a MFGarchon :class:`MeshData`
(``problem.geometry.mesh_data``) or, for convenience, a ``skfem.Mesh`` (converted
via the existing :func:`skfem_to_meshdata` adapter). Nodal fields must be indexed
by mesh vertex, so this targets P1 Lagrange / vertex-DOF fields; higher-order DOF
vectors (P2 edge midpoints) do not match the vertex count and fail loud.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

    from mfgarchon.geometry.meshes.mesh_data import MeshData

# MeshData.element_type -> meshio cell type. MeshData uses "tetrahedron"/"hexahedron"
# where meshio uses "tetra"/"hexahedron"; the rest coincide.
_MESHIO_CELL_TYPE: dict[str, str] = {
    "line": "line",
    "triangle": "triangle",
    "quad": "quad",
    "tetrahedron": "tetra",
    "hexahedron": "hexahedron",
}


def _import_meshio():
    """Import meshio with a clear error message (declared dependency)."""
    try:
        import meshio
    except ImportError as err:  # pragma: no cover - meshio is a hard dependency
        raise ImportError("meshio is required for VTK export. Install with: pip install meshio") from err
    return meshio


def _as_meshdata(mesh) -> MeshData:
    """Coerce a ``MeshData`` or ``skfem.Mesh`` to ``MeshData`` (fail loud otherwise)."""
    from mfgarchon.geometry.meshes.mesh_data import MeshData

    if isinstance(mesh, MeshData):
        return mesh

    try:
        import skfem
    except ImportError:
        skfem = None
    if skfem is not None and isinstance(mesh, skfem.Mesh):
        from mfgarchon.alg.numerical.fem.mesh_adapter import skfem_to_meshdata

        return skfem_to_meshdata(mesh)

    raise TypeError(f"mesh must be a MeshData or skfem.Mesh, got {type(mesh).__name__}")


def _meshio_mesh(mesh_data: MeshData, fields: Mapping[str, NDArray]):
    """Build a ``meshio.Mesh`` from ``MeshData`` + nodal fields, validating both."""
    meshio = _import_meshio()

    cell_type = _MESHIO_CELL_TYPE.get(mesh_data.element_type)
    if cell_type is None:
        raise ValueError(
            f"Unsupported element_type '{mesh_data.element_type}' for VTK export. "
            f"Supported: {sorted(_MESHIO_CELL_TYPE)}"
        )

    n_points = mesh_data.num_vertices
    point_data: dict[str, NDArray] = {}
    for name, values in fields.items():
        arr = np.asarray(values)
        if arr.shape[0] != n_points:
            raise ValueError(
                f"Nodal field '{name}' has length {arr.shape[0]} but the mesh has {n_points} "
                f"vertices. Nodal fields must be indexed by mesh vertex (P1/vertex DOFs); "
                f"higher-order (e.g. P2 edge-midpoint) DOF vectors are not supported."
            )
        point_data[name] = arr

    # VTK is intrinsically 3D; pad 2D/1D coordinates with zero columns so meshio does
    # not emit an implicit-padding warning and ParaView reads explicit 3D points.
    points = mesh_data.vertices
    if points.shape[1] < 3:
        points = np.column_stack([points, np.zeros((points.shape[0], 3 - points.shape[1]))])

    return meshio.Mesh(
        points=points,
        cells=[(cell_type, mesh_data.elements)],
        point_data=point_data,
    )


def export_mesh_solution_vtk(mesh, fields: Mapping[str, NDArray], path: str | Path) -> Path:
    """Export one mesh + nodal fields to a ParaView ``.vtu`` file.

    Args:
        mesh: The solution mesh — a MFGarchon ``MeshData`` (e.g.
            ``problem.geometry.mesh_data``) or a ``skfem.Mesh``.
        fields: Nodal fields ``{name: array}``; each array's first axis must have
            length ``mesh.num_vertices`` (e.g. ``{"U": U[-1], "M": M[-1]}``).
        path: Output path (``.vtu`` XML VTK recommended).

    Returns:
        The written file path.

    Raises:
        ValueError: Unsupported ``element_type`` or a field whose length does not
            match the vertex count.
        TypeError: ``mesh`` is neither ``MeshData`` nor ``skfem.Mesh``.
    """
    mesh_data = _as_meshdata(mesh)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    _meshio_mesh(mesh_data, fields).write(out)
    return out


def export_time_series_vtk(
    mesh,
    fields: Mapping[str, NDArray],
    times: Sequence[float] | NDArray,
    path: str | Path,
) -> Path:
    """Export a time-resolved MFG solution as per-timestep ``.vtu`` files + a ``.pvd``.

    For ``Nt+1`` timesteps this writes ``<stem>_<k>.vtu`` (``k = 0 .. Nt``) next to
    ``path`` and a ParaView ``.pvd`` collection at ``path`` indexing each file by its
    time $t_k$, so ParaView animates the series.

    Args:
        mesh: The solution mesh (``MeshData`` or ``skfem.Mesh``).
        fields: Time-resolved nodal fields ``{name: array(Nt+1, N)}``; the first axis
            is time (length ``len(times)``), the second is the mesh vertices
            (e.g. ``{"U": U, "M": M}`` with ``U, M`` of shape ``(Nt+1, N)``).
        times: The ``Nt+1`` time values $t_k$.
        path: Output ``.pvd`` path; the per-timestep ``.vtu`` stem is derived from it.

    Returns:
        The written ``.pvd`` collection path.

    Raises:
        ValueError: Empty ``fields``, a field that is not ``(n_times, N)``, or a field
            whose time length disagrees with ``len(times)``.
    """
    mesh_data = _as_meshdata(mesh)
    times = np.asarray(times, dtype=float)
    n_times = times.shape[0]

    if not fields:
        raise ValueError("export_time_series_vtk requires at least one field.")
    for name, values in fields.items():
        arr = np.asarray(values)
        if arr.ndim != 2:
            raise ValueError(f"Time-series field '{name}' must be 2D (n_times, n_vertices), got shape {arr.shape}.")
        if arr.shape[0] != n_times:
            raise ValueError(f"Time-series field '{name}' has {arr.shape[0]} timesteps but {n_times} times were given.")

    pvd_path = Path(path)
    pvd_path.parent.mkdir(parents=True, exist_ok=True)
    stem = pvd_path.stem

    datasets: list[tuple[float, str]] = []
    for k in range(n_times):
        vtu_name = f"{stem}_{k}.vtu"
        snapshot = {name: np.asarray(values)[k] for name, values in fields.items()}
        # Per-vertex length validation happens inside _meshio_mesh.
        _meshio_mesh(mesh_data, snapshot).write(pvd_path.parent / vtu_name)
        datasets.append((float(times[k]), vtu_name))

    _write_pvd(pvd_path, datasets)
    return pvd_path


def _write_pvd(pvd_path: Path, datasets: Sequence[tuple[float, str]]) -> None:
    """Write a ParaView ``.pvd`` collection referencing ``(time, vtu_filename)`` pairs.

    File names are stored relative to the ``.pvd`` (same directory), so the collection
    is relocatable as a unit.
    """
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        "  <Collection>",
    ]
    for time, vtu_name in datasets:
        lines.append(f'    <DataSet timestep="{time}" group="" part="0" file="{vtu_name}"/>')
    lines.append("  </Collection>")
    lines.append("</VTKFile>")
    pvd_path.write_text("\n".join(lines) + "\n")
