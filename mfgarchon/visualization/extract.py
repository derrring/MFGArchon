"""Rendering-free extraction of plot-ready data from MFG solve results (Issue #1362).

These functions return numpy arrays / pandas ``DataFrame`` (tidy long-form), importing **no** renderer
(no matplotlib / plotly / pyvista / meshio). Plot with your own backend, or build paper figures,
without re-deriving the shaping logic (slice selection, mass series, tidy reshaping, per-node
unpacking) that the helpers in ``convergence_plots.py`` otherwise embed.

Fail-fast on shape/length mismatch and unknown slice times — no silent nearest-time snapping, no
silent fallbacks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

import numpy as np

if TYPE_CHECKING:
    from mfgarchon.alg.numerical.coupling.graph_mfg_solver import GraphMFGResult
    from mfgarchon.utils.solver_result import SolverResult


def extract_convergence_history(result: SolverResult) -> pd.DataFrame:
    r"""Per-iteration convergence residuals as a tidy frame — columns ``iteration, error_U, error_M``.

    ``error_U`` / ``error_M`` are the residual histories $\|U^{k}-U^{k-1}\|$ and $\|M^{k}-M^{k-1}\|$.

    >>> df = extract_convergence_history(result)
    >>> df.plot(x="iteration", y=["error_U", "error_M"], logy=True)  # your own matplotlib
    """
    eu = np.asarray(result.error_history_U, dtype=float)
    em = np.asarray(result.error_history_M, dtype=float)
    if eu.shape != em.shape:
        raise ValueError(f"error_history_U {eu.shape} and error_history_M {em.shape} length mismatch")
    return pd.DataFrame({"iteration": np.arange(len(eu)), "error_U": eu, "error_M": em})


def _slices(field: np.ndarray, x_grid, t_grid, times, name: str) -> pd.DataFrame:
    """Shared tidy-long-form slicer for U/M. Validates ``field.shape == (len(t_grid), len(x_grid))``
    and that every requested time is in ``t_grid`` (fail-loud, no nearest-snap)."""
    f = np.asarray(field, dtype=float)
    x = np.asarray(x_grid, dtype=float)
    t = np.asarray(t_grid, dtype=float)
    if f.shape != (len(t), len(x)):
        raise ValueError(f"{name} shape {f.shape} != (len(t_grid), len(x_grid)) = ({len(t)}, {len(x)})")
    if times is None:
        t_idx = np.arange(len(t))
    else:
        t_idx = []
        for tv in np.atleast_1d(np.asarray(times, dtype=float)):
            hits = np.where(np.isclose(t, tv))[0]
            if len(hits) == 0:
                raise ValueError(
                    f"{name}: requested time {tv} not in t_grid (range [{t[0]}, {t[-1]}], {len(t)} points)"
                )
            t_idx.append(int(hits[0]))
        t_idx = np.asarray(t_idx, dtype=int)
    return pd.DataFrame(
        {
            "t": np.repeat(t[t_idx], len(x)),
            "x": np.tile(x, len(t_idx)),
            "value": f[t_idx].ravel(),
        }
    )


def extract_density_slices(result: SolverResult, x_grid, t_grid, times=None) -> pd.DataFrame:
    r"""Density $m(t,x)$ as tidy long-form — columns ``t, x, value``. ``times=None`` returns all time
    knots; otherwise each requested time must be in ``t_grid`` (raises, no silent snap).

    >>> extract_density_slices(result, x_grid, t_grid, times=[0.0, 0.5, 1.0])
    """
    return _slices(result.M, x_grid, t_grid, times, "M")


def extract_value_slices(result: SolverResult, x_grid, t_grid, times=None) -> pd.DataFrame:
    r"""Value function $u(t,x)$ as tidy long-form — columns ``t, x, value`` (see
    :func:`extract_density_slices`)."""
    return _slices(result.U, x_grid, t_grid, times, "U")


def extract_mass_history(result: SolverResult, x_grid) -> pd.DataFrame:
    r"""Total mass $\int m(t,x)\,dx$ per time step (trapezoidal over ``x_grid``) — columns
    ``step, total_mass``. Exposes the series behind the scalar ``mass_conservation_error``.

    >>> extract_mass_history(result, x_grid).plot(x="step", y="total_mass")  # your own matplotlib
    """
    M = np.asarray(result.M, dtype=float)
    x = np.asarray(x_grid, dtype=float)
    if M.ndim != 2 or M.shape[1] != len(x):
        raise ValueError(f"M shape {M.shape} incompatible with x_grid of length {len(x)}")
    mass = np.trapezoid(M, x, axis=1)
    return pd.DataFrame({"step": np.arange(M.shape[0]), "total_mass": mass})


def extract_graph_trajectories(result: GraphMFGResult, x_grid=None, field: str = "density") -> pd.DataFrame:
    r"""Per-node trajectories from a ``GraphMFGResult`` (list-of-arrays) as tidy long-form — columns
    ``node, step, x, value``. ``field`` selects ``"density"`` ($m$) or ``"value"`` ($u$). ``x_grid``
    defaults to integer indices when omitted.

    >>> extract_graph_trajectories(graph_result, field="density")
    """
    if field == "density":
        arrays = result.densities
    elif field == "value":
        arrays = result.values
    else:
        raise ValueError(f"field must be 'density' or 'value', got {field!r}")
    if not arrays:
        raise ValueError("GraphMFGResult has no per-node arrays")

    frames = []
    for node, arr in enumerate(arrays):
        a = np.asarray(arr, dtype=float)
        if a.ndim != 2:
            raise ValueError(f"node {node} array must be 2D (Nt+1, Nx), got shape {a.shape}")
        nt, nx = a.shape
        xg = np.arange(nx) if x_grid is None else np.asarray(x_grid, dtype=float)
        if len(xg) != nx:
            raise ValueError(f"node {node}: x_grid length {len(xg)} != Nx {nx}")
        frames.append(
            pd.DataFrame(
                {
                    "node": node,
                    "step": np.repeat(np.arange(nt), nx),
                    "x": np.tile(xg, nt),
                    "value": a.ravel(),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)
