"""
Lions derivative correction bridge for MFG source term injection.

Connects the functional calculus infrastructure (FunctionalDerivative)
to the HJB source term pipeline (MFGProblem.source_term_hjb).

For a coupling energy F[m], the Lions correction adds the first variation
delta F / delta m[m](x) as a cost-signed source term S on the right-hand side of the HJB equation
(docs/user/CONVENTIONS.md § The source term):

    -du/dt + H(t, x, Du, m) - (sigma^2/2) Laplacian(u) = delta F / delta m[m](x)

This is the "measure-dependent" coupling that goes beyond local f(m(x))
coupling (which is already handled by the Hamiltonian).

Issue #956: Part of Layer 2 (Measure-Dependent MFG).

Mathematical background:
    For nonlocal coupling F[m] = (1/2) int int W(x,y) m(x) m(y) dx dy,
    the first variation is:
        delta F / delta m[m](x) = int W(x,y) m(y) dy = (W * m)(x)

    For local coupling F[m] = int f(m(x)) dx:
        delta F / delta m[m](x) = f'(m(x))

    The FunctionalDerivative infrastructure computes this via finite
    differences or particle approximation, making this bridge agnostic
    to the specific coupling form.

Convention
----------
The HJB source slot holds the **pointwise** flat derivative
``delta F / delta m``, and ``F[m]`` is the **physical** (mesh-independent)
energy. Both paths of :func:`create_lions_source` return that same object; see
:mod:`mfgarchon.operators.interaction.energy_functionals` for the discretization
convention ``integral g m dx = sum_k w_k g_k m_k``. In particular the double
integral above discretizes with **two** quadrature factors,
``F[m] = (1/2) sum_ij w_i w_j W_ij m_i m_j``, and its flat derivative is
``(delta F / delta m)_i = sum_j W_ij w_j m_j`` -- one factor, not zero and not
two.

Usage:
    >>> from mfgarchon.utils.functional_calculus import FiniteDifferenceFunctionalDerivative
    >>> from mfgarchon.alg.numerical.coupling.lions_correction import create_lions_source
    >>>
    >>> # A nonlocal energy on the grid's measure: both quadrature factors are its weights (#1642)
    >>> grid = TensorProductGrid(...)
    >>> w = grid.measure_weights().ravel()  # problem.spatial_measure().weights on this grid, flattened
    >>> W = ...  # interaction kernel matrix W(x_i, x_j), shape (Nx, Nx), on the grid's nodes
    >>> def energy(m):
    ...     return 0.5 * np.sum(w * m * (W @ (w * m)))
    >>>
    >>> fd = FiniteDifferenceFunctionalDerivative(epsilon=1e-4)
    >>> problem = MFGProblem(..., domain=grid, source_term_hjb=create_lions_source(energy, fd, weights=w))
    >>> result = problem.solve()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from mfgarchon.operators.interaction.energy_functionals import EnergyFunctional
    from mfgarchon.utils.functional_calculus import FunctionalDerivative, FunctionalOnMeasures


@runtime_checkable
class _AppliedWithMatmul(Protocol):
    """A kernel ``create_nonlocal_source`` applies as it is: it applies itself with ``@``. This check finds
    ``__matmul__`` statically, on the type or on the instance, while Python's ``@`` uses the type's alone. Its shape is
    read apart, as an attribute, since a wrapper can forward one through ``__getattr__``, which this check does not see
    (#2726)."""

    def __matmul__(self, other: Any) -> Any: ...


def _spatial_density(m: NDArray, n: int | None = None) -> NDArray:
    """Return the flat spatial density, rejecting (Nt+1, Nx) trajectories.

    The composed source pipeline (:mod:`source_composition`) time-slices the
    density to the current backward time step before calling the source, so a
    per-time source always receives one spatial slice: shape (Nx,) in 1-D, the
    grid's shape in n-D. ``n`` is the number of nodes the functional is on,
    known from its weights: a slice holding exactly ``n`` values is flattened in
    the grid's C order whatever its shape (#1642). Without ``n``, a 2-D array is
    read as a ``(Nt+1, Nx)`` trajectory and refused: silently collapsing it to
    the terminal slice ``m[-1]`` reintroduces the Issue #1285 wrong-slice bug.
    """
    m = np.asarray(m)
    if n is not None and m.size == n:
        return m.ravel()
    if m.ndim == 2:
        raise ValueError(
            f"Lions source expects a 1-D spatial density (Nx,), got a 2-D array "
            f"with shape {m.shape}. Pass the time-t density slice, not the full "
            "(Nt+1, Nx) trajectory (see source_composition.compose_hjb_source; "
            "Issue #1285)."
        )
    return m.ravel()


def create_lions_source(
    energy_functional: FunctionalOnMeasures | EnergyFunctional,
    functional_derivative: FunctionalDerivative | None = None,
    *,
    weights: NDArray | float | None = None,
) -> Callable[[float, NDArray, NDArray, NDArray], NDArray]:
    """Create a source_term_hjb from a measure-dependent energy functional.

    Bridges the functional calculus infrastructure to the MFGProblem
    source_term_hjb interface by computing delta F / delta m[m](x)
    at each Picard iteration.

    Two paths, selected from the first argument. **Both return the same object**:
    the pointwise flat derivative ``delta F / delta m`` of the physical energy
    ``F[m]``.

    1. **Analytic** (Issue #1023, Phase 2): if ``energy_functional`` is an
       :class:`~mfgarchon.operators.interaction.energy_functionals.EnergyFunctional`,
       its exact derivative is used and the FD path is skipped.
       ``functional_derivative`` is then ignored and ``weights`` must not be
       passed -- the functional already carries them. The per-slice ``t`` is
       forwarded to ``flat_derivative``.
    2. **Finite difference** (original path): if ``energy_functional`` is a plain
       ``F[m] -> float`` callable, both ``functional_derivative`` **and**
       ``weights`` are required.

    ``weights`` is required on the FD path because the FD engine perturbs an
    *unweighted* Dirac ``m_k += epsilon``, so it returns the entry gradient
    ``d F / d m_k = w_k * (delta F / delta m)_k`` -- a factor ``w_k`` away from
    the pointwise object the HJB slot needs. Dividing it out is the *only* way
    the two paths agree, and it has one owner,
    :func:`~mfgarchon.operators.interaction.energy_functionals.flat_derivative_from_energy_gradient`
    (Issue #1642 A2), applied here. There is no default: silently assuming
    ``w = 1`` makes the interaction coupling scale with the mesh and switch off
    under refinement, with no error.

    An object that provides *some* but not all of the ``EnergyFunctional``
    members is refused loudly rather than downgraded to the FD path, where its
    exact ``flat_derivative`` would never be called (Issue #1642 D-6).

    Args:
        energy_functional: Either an ``EnergyFunctional`` (analytic path) or a
            plain ``F[m] -> float`` callable returning the **physical** energy
            (FD path). Density arrays have shape (Nx,).
        functional_derivative: FunctionalDerivative instance for the FD path
            (FiniteDifferenceFunctionalDerivative or
            ParticleApproximationFunctionalDerivative). Required only when
            ``energy_functional`` is a plain callable; ignored for an
            ``EnergyFunctional``.
        weights: Quadrature weights of the measure representation for the FD
            path — cell volumes on a grid, particle masses for an empirical
            measure. Either shape ``(Nx,)`` or a scalar broadcast to the density
            length at call time. Strictly positive. Required on the FD path;
            rejected on the analytic path.

    Returns:
        source_term_hjb(t, x, v, m) -> NDArray compatible with
        MFGProblem.source_term_hjb field. The returned array has shape
        matching x (one value per spatial grid point).

    Example:
        >>> import numpy as np
        >>> from mfgarchon.utils.functional_calculus import FiniteDifferenceFunctionalDerivative
        >>>
        >>> # Quadratic interaction: F[m] = (1/2) int m(x)^2 dx = (1/2) sum_k w_k m_k^2
        >>> dx = 0.02
        >>> def energy(m):
        ...     return 0.5 * np.sum(m**2) * dx
        >>>
        >>> fd = FiniteDifferenceFunctionalDerivative(epsilon=1e-4)
        >>> source = create_lions_source(energy, fd, weights=dx)
        >>>
        >>> # source(t, x, v, m) returns delta F / delta m = m(x)
        >>> m = np.ones(50) / 50
        >>> result = source(t=0.0, x=np.linspace(0, 1, 50), v=np.zeros(50), m=m)
    """
    # Analytic path: EnergyFunctional carries its own exact flat derivative.
    from mfgarchon.operators.interaction.energy_functionals import (
        EnergyFunctional,
        energy_functional_members,
        flat_derivative_from_energy_gradient,
        missing_energy_functional_members,
        validate_weights,
    )

    if isinstance(energy_functional, EnergyFunctional):
        if weights is not None:
            raise ValueError(
                "create_lions_source: weights= is for the finite-difference path only. "
                f"{type(energy_functional).__name__} is an EnergyFunctional and carries its "
                "own quadrature weights, so a second set here would be a silent second source "
                "of the same quantity."
            )
        analytic = energy_functional

        def source_term_hjb_analytic(
            t: float,
            x: NDArray,
            v: NDArray,
            m: NDArray,
        ) -> NDArray:
            """Evaluate the analytic correction delta F / delta m[m](x)."""
            m_flat = _spatial_density(m, np.size(analytic.weights))
            return np.asarray(analytic.flat_derivative(m_flat, t=t)).ravel()

        return source_term_hjb_analytic

    # Not an EnergyFunctional. Distinguish "almost one" from "not one at all":
    # a near-miss that happens to be callable would otherwise take the FD path
    # with its exact flat_derivative never called, and no diagnostic.
    missing = missing_energy_functional_members(energy_functional)
    if missing and len(missing) < len(energy_functional_members()):
        provided = [n for n in energy_functional_members() if n not in missing]
        raise TypeError(
            f"create_lions_source: {type(energy_functional).__name__} provides "
            f"{provided} but is missing {list(missing)}, so it is not an EnergyFunctional. "
            "Refusing rather than falling through to the finite-difference path, which would "
            "never call its exact flat_derivative. Implement the missing members -- or "
            "subclass EnergyFunctional to inherit the second_variation default -- or pass a "
            "plain F[m] -> float callable together with functional_derivative= and weights=."
        )
    if not callable(energy_functional):
        raise TypeError(
            f"create_lions_source: expected an EnergyFunctional or a plain F[m] -> float "
            f"callable, got {type(energy_functional).__name__}, which provides none of "
            f"{list(energy_functional_members())} and is not callable."
        )

    if functional_derivative is None:
        raise ValueError(
            "functional_derivative is required when energy_functional is a plain "
            "callable (FD path); pass an EnergyFunctional to use the analytic path"
        )
    if weights is None:
        raise ValueError(
            "create_lions_source: weights= is required on the finite-difference path. "
            "The FD engine perturbs an unweighted Dirac (m_k += epsilon), so it returns the "
            "entry gradient dF/dm_k = w_k * (delta F / delta m)_k, while the HJB source slot "
            "holds the pointwise delta F / delta m. Pass the quadrature weights of the measure "
            "representation (cell volumes on a grid, particle masses for an empirical measure) "
            "so the conversion can be applied. There is no default: assuming w = 1 makes the "
            "interaction coupling scale with the mesh and switch off under refinement, with no "
            "error (Issue #1642 A2). Pass an EnergyFunctional to use the analytic path instead."
        )

    w_input = np.asarray(weights, dtype=float)
    w_scalar = float(w_input) if w_input.ndim == 0 else None
    # Validate eagerly: a non-positive or non-finite weight is a broken
    # discretization, and it should fail at wiring time, not mid-Picard.
    w_array = validate_weights(
        np.full(1, w_scalar) if w_scalar is not None else w_input,
        "create_lions_source (finite-difference path)",
    )

    def source_term_hjb(
        t: float,
        x: NDArray,
        v: NDArray,
        m: NDArray,
    ) -> NDArray:
        """Evaluate the correction delta F / delta m[m](x).

        Args:
            t: Current time (unused for time-independent F[m])
            x: Spatial grid points, shape (Nx,) or (Nx, d)
            v: Current value function (unused — correction depends on m only)
            m: Current spatial density, one time-t slice: shape (Nx,), or the
                grid's shape in n-D when the weights number its nodes. The
                source pipeline time-slices before calling (a 2-D (Nt+1, Nx)
                trajectory is rejected — see Issue #1285).

        Returns:
            Source term values, shape (Nx,)
        """
        m_flat = _spatial_density(m, None if w_scalar is not None else w_array.size)
        Nx = len(m_flat)

        # Compute dF/dm_k at each grid point
        # y_points = all grid indices (perturb at each point)
        y_indices = np.arange(Nx)

        gradient = functional_derivative.compute(
            energy_functional,
            m_flat,
            x_points=x,
            y_points=y_indices,
        )

        w = np.full(Nx, w_scalar) if w_scalar is not None else w_array
        return flat_derivative_from_energy_gradient(gradient, w)

    return source_term_hjb


def create_nonlocal_source(
    interaction_kernel: Any,
    weights: NDArray,
) -> Callable[[float, NDArray, NDArray, NDArray], NDArray]:
    """Create source_term_hjb for nonlocal interaction coupling.

    Optimized path for the common case F[m] = (1/2) int int W(x,y) m(x) m(y) dx dy,
    where delta F / delta m[m](x) = int W(x,y) m(y) dy = W @ (w * m), with ``w`` the
    per-node weights of the problem's measure. ``W`` need not be translation-invariant;
    for a kernel K(x - y), ``ConvolutionCouplingOperator.from_problem`` is the same
    integral with an FFT path.

    This avoids the overhead of finite-difference functional derivatives
    by computing the convolution directly.

    Args:
        interaction_kernel: The unweighted kernel W(x_i, x_j), shape (N, N) where N is the
            total number of spatial grid points (N = Nx for 1D, Nx*Ny for 2D, etc.):
            anything with that shape that applies itself with ``@`` -- a dense array,
            a sparse matrix, a ``LinearOperator``, a matrix-free object of the
            caller's -- applied as ``W @ v``, or anything ``np.asarray`` makes a numeric
            matrix. Symmetric for undirected interaction. Operates on flattened
            density arrays. ``W`` must not carry the measure's weights itself: those
            are ``weights``, and a weighted form is weighted twice. The library's
            weighted operators, a ``ConvolutionCouplingOperator`` or a
            ``GraphonCouplingOperator`` passed as built, are refused; a form derived
            from one (``op.T``, ``op.H``, ``2 * op``, ``op.as_dense()``, a wrapper of
            one) cannot be told from an unweighted kernel and is not.
        weights: The measure's per-node weights, ``problem.spatial_measure().weights``:
            shape (N,), or the grid's shape. A scalar is refused. One cell volume for
            every node is the rectangle rule, which gives each end node a full cell
            past the domain (#1642).

    Returns:
        source_term_hjb(t, x, v, m) -> NDArray.

    Example:
        >>> # Gaussian interaction kernel on a problem's grid
        >>> x = problem.geometry.get_spatial_grid().ravel()
        >>> W = np.exp(-((x[:, None] - x[None, :]) ** 2) / (2 * 0.1**2))
        >>> source = create_nonlocal_source(W, problem.spatial_measure().weights)
    """
    from mfgarchon.operators.interaction.energy_functionals import validate_weights

    if np.ndim(weights) == 0:
        raise ValueError(
            f"create_nonlocal_source: weights is the scalar {weights!r}, one cell volume for every node. That is the "
            "rectangle rule, which gives each end node a full cell past the domain. Pass the measure's per-node "
            "weights, problem.spatial_measure().weights (#1642)."
        )
    from mfgarchon.operators.integro_diff.graphon_coupling import GraphonCouplingOperator
    from mfgarchon.operators.interaction.convolution import ConvolutionCouplingOperator

    # The library's own weighted operators, passed as built: their @ is the integral, weights included. A form derived
    # from one is not refused, since whether a matrix carries the measure's weights cannot be read from it (#2726).
    if isinstance(interaction_kernel, ConvolutionCouplingOperator):
        raise ValueError(
            "create_nonlocal_source: the interaction kernel is a ConvolutionCouplingOperator, whose @ already carries "
            "its weights, so W @ (w * m) would weight m twice. Its own channel to the HJB source is "
            "create_lions_source(QuadraticInteractionEnergy(op)); or pass its unweighted matrix, op.kernel_matrix() "
            "(#1642)."
        )
    if isinstance(interaction_kernel, GraphonCouplingOperator):
        raise ValueError(
            "create_nonlocal_source: the interaction kernel is a GraphonCouplingOperator, whose @ is already the "
            "integral over [0, 1], quadrature weights included, so W @ (w * m) would weight m twice. Pass the "
            "unweighted matrix of the graphon kernel it was built with, kernel.evaluate(x, x) on the grid's nodes x "
            "(#1642)."
        )
    w = validate_weights(weights, "create_nonlocal_source")
    applied_as_given = isinstance(interaction_kernel, _AppliedWithMatmul)
    if applied_as_given:
        # Applied as it is -- an ndarray, a sparse matrix, a LinearOperator, a matrix-free object of the caller's, a
        # wrapper of one; np.asarray would wrap the matrix-free ones in a 0-d object array.
        W = interaction_kernel
        try:
            shape = tuple(interaction_kernel.shape)
        except AttributeError:
            raise ValueError(
                f"create_nonlocal_source: the interaction kernel, a {type(W).__name__}, applies itself with @ but has "
                f"no shape, so it cannot be checked against the weights' {w.size} nodes. Give it a shape "
                f"({w.size}, {w.size}), as a LinearOperator has (#2726)."
            ) from None
    else:
        W = np.asarray(interaction_kernel)
        shape = W.shape
        if W.dtype == object:
            raise ValueError(
                f"create_nonlocal_source: the interaction kernel, a {type(interaction_kernel).__name__}, neither "
                f"applies itself with @ nor converts to a numeric array (np.asarray gives an object array of shape "
                f"{W.shape}) (#2726)."
            )
    if shape != (w.size, w.size):
        raise ValueError(
            f"create_nonlocal_source: the interaction kernel has shape {shape} and the weights {w.size} nodes; it "
            f"must be ({w.size}, {w.size})"
        )

    def source_term_hjb(
        t: float,
        x: NDArray,
        v: NDArray,
        m: NDArray,
    ) -> NDArray:
        """Evaluate (W * m)(x) = int W(x,y) m(y) dy."""
        m_spatial = _spatial_density(m, w.size)
        return W @ (w * m_spatial)

    return source_term_hjb
