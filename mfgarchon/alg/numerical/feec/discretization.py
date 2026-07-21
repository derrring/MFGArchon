r"""Finite Element Exterior Calculus (FEEC) — mixed structure-preserving discretization scaffold.

Infrastructure foundation for structure-preserving Mean Field Games (a stepping stone toward
**symplectic MFG**), deliberately a SCAFFOLD — not a finished solver.

Motivation
----------
The primal nodal Galerkin scheme (:class:`WeakFormFPSolver` / meshless-Galerkin) conserves mass
GLOBALLY (partition of unity ``sum_i phi_i = 1`` makes the operator's column sums vanish) but not
LOCALLY, and its advection is not an M-matrix — the positivity clip injects mass in the coupled regime.
A **mixed FEEC** formulation represents the transport flux ``F = v m`` in a Raviart-Thomas ``H(div)``
space and the density ``m`` in an ``L2`` (``P0``) space, so the discrete divergence is exact
elementwise::

    int_K (div F) dx = oint_{dK} F . n ds      for every element K

i.e. **exact local conservation** — the structure-preserving core the de Rham complex provides.

Scope (what this module IS and IS NOT)
--------------------------------------
PROVIDED (assembles, tested) — the structure-preserving building blocks:
  * the mixed ``RT0 x P0`` spaces (flux in ``H(div)``, density in ``L2``);
  * the divergence operator ``B[i,j] = int q_i (div F_j)`` (``density_dof x flux_dof``), exact per element;
  * the ``H(div)`` flux mass ``int F_i . F_j`` and the ``L2`` density mass ``int m_i m_j``;
  * an ``L2`` projection of a velocity field into the flux space (a convenience, not the commuting proj).

NOT provided (research — fail loud, never silent):
  * the coupled saddle-point HJB-FP solve (a block/Uzawa/Schur system, not the scalar ``M/dt + D K + C``);
  * positivity / limiting — an ``RT/L2`` density is piecewise-constant and still NOT sign-preserving under
    advection, so the mass-injecting clip is NOT removed by FEEC (the binding constraint; needs
    DG+limiters / FV-upwind / streamline diffusion);
  * the nonlinear-Hamiltonian coupling (value in ``H1`` -> flux velocity in ``H(div)`` welded through ``H``);
  * a symplectic time integrator (the current base is implicit Euler).

Design guardrails (from the FEEC variant study, 2026-07-04; Joplin Dev "FEEC + symplectic MFG roadmap")
-------------------------------------------------------------------------------------------------------
  * This is a **sibling family**, not an extension of the scalar-nodal :class:`WeakFormDiscretization`:
    a mixed system has block operators + vector flux DOFs (no per-DOF coordinate) + no nodal density, so
    it does NOT satisfy that protocol. Keep it a clean, separate, TYPED family; do NOT fork the WeakForm
    base into a divergent parallel path (the single-source-of-truth failure class this codebase guards).
  * FEEC **complements**, does not subsume, the primal adjoint-consistency (``A_FP = A_HJB^T``) thesis,
    and it does NOT fix the ``O(h)`` advective gap (that is nonlinear-``H`` linearization + gradient
    recovery, not a flux-space defect). Do not claim otherwise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import skfem

    from numpy.typing import NDArray
    from scipy import sparse


def _import_skfem():
    try:
        import skfem
    except ImportError:
        raise ImportError(
            "scikit-fem is required for the FEEC discretization. Install: pip install scikit-fem"
        ) from None
    return skfem


@runtime_checkable
class MixedWeakFormDiscretization(Protocol):
    """Block operators a mixed (structure-preserving / FEEC) MFG solver needs.

    Unlike :class:`WeakFormDiscretization` (one scalar-nodal space, every operator ``(n_dof, n_dof)``),
    a mixed FEEC discretization carries TWO spaces — an ``H(div)`` flux and an ``L2`` density — and block
    operators between them. Concrete implementations (e.g. :class:`RaviartThomasDiscretization`) assemble
    these; the coupled solve lives in a mixed solver family, not in the scalar ``WeakForm*`` base.
    """

    @property
    def flux_dof(self) -> int:
        """Number of ``H(div)`` flux DOFs (edges in 2D / faces in 3D)."""

    @property
    def density_dof(self) -> int:
        """Number of ``L2`` density DOFs (elements)."""

    @property
    def dim(self) -> int:
        """Spatial dimension."""

    def divergence(self) -> sparse.csr_matrix:
        r"""Divergence coupling ``B[i,j] = int_Omega q_i (div F_j) dx``, shape ``(density_dof, flux_dof)``.

        This is the structure-preserving core: with ``q_i`` the ``P0`` density basis and ``F_j`` the
        ``RT`` flux basis, ``B`` realises ``int_K div F`` EXACTLY per element (Stokes), so the discrete
        mass balance ``dm/dt + div F = ...`` conserves mass locally, not just globally.
        """

    def flux_mass(self) -> sparse.csr_matrix:
        r"""``H(div)`` flux mass ``int_Omega F_i . F_j dx``, shape ``(flux_dof, flux_dof)`` (symmetric)."""

    def density_mass(self) -> sparse.csr_matrix:
        r"""``L2`` density mass ``int_Omega m_i m_j dx``, shape ``(density_dof, density_dof)`` (diagonal for P0)."""


class RaviartThomasDiscretization:
    """Lowest-order mixed FEEC discretization: ``RT0`` flux (``H(div)``) x ``P0`` density (``L2``).

    On a triangular (2D) or tetrahedral (3D) ``skfem`` mesh. Implements the structure-preserving building
    blocks of :class:`MixedWeakFormDiscretization`; the coupled MFG solve is intentionally NOT here — see
    the module docstring. Satisfies ``isinstance(disc, MixedWeakFormDiscretization)``.
    """

    def __init__(self, mesh: skfem.Mesh) -> None:
        skfem = _import_skfem()
        if isinstance(mesh, skfem.MeshTri):
            flux_elem, density_elem = skfem.ElementTriRT0(), skfem.ElementTriP0()
        elif isinstance(mesh, skfem.MeshTet):
            flux_elem, density_elem = skfem.ElementTetRT0(), skfem.ElementTetP0()
        else:
            raise NotImplementedError(
                f"RaviartThomasDiscretization supports simplicial meshes (MeshTri / MeshTet) only; got "
                f"{type(mesh).__name__}. Lowest-order RT is defined on simplices; quad/hex need RTC/NC "
                f"elements (a future extension)."
            )
        self._mesh = mesh
        self._flux_basis = skfem.Basis(mesh, flux_elem)
        self._density_basis = skfem.Basis(mesh, density_elem)
        self._divergence: sparse.csr_matrix | None = None
        self._flux_mass: sparse.csr_matrix | None = None
        self._density_mass: sparse.csr_matrix | None = None

    @property
    def flux_dof(self) -> int:
        return self._flux_basis.N

    @property
    def density_dof(self) -> int:
        return self._density_basis.N

    @property
    def dim(self) -> int:
        return self._mesh.p.shape[0]

    def divergence(self) -> sparse.csr_matrix:
        if self._divergence is None:
            skfem = _import_skfem()
            from skfem import BilinearForm
            from skfem.helpers import div

            @BilinearForm
            def b(flux, q, w):
                return div(flux) * q

            # asm(form, trial_basis, test_basis): trial = flux (RT0), test = density (P0)
            self._divergence = skfem.asm(b, self._flux_basis, self._density_basis).tocsr()
        return self._divergence

    def flux_mass(self) -> sparse.csr_matrix:
        if self._flux_mass is None:
            skfem = _import_skfem()
            from skfem import BilinearForm
            from skfem.helpers import dot

            @BilinearForm
            def m(u, v, w):
                return dot(u, v)

            self._flux_mass = skfem.asm(m, self._flux_basis).tocsr()
        return self._flux_mass

    def density_mass(self) -> sparse.csr_matrix:
        if self._density_mass is None:
            skfem = _import_skfem()
            from skfem.models import mass

            self._density_mass = skfem.asm(mass, self._density_basis).tocsr()
        return self._density_mass

    def project_velocity_to_flux(self, velocity: NDArray) -> NDArray:
        r"""``L2``-project a nodal velocity field into the ``RT0`` flux space (a convenience for the
        coupling scaffold; NOT the commuting/canonical projection the full FEEC theory would use).

        ``velocity`` is ``(dim, n_points)`` at the density quadrature points; returns ``RT0`` DOFs.
        The load-bearing MFG coupling (mapping the ``H1`` value gradient into ``H(div)`` through the
        nonlinear Hamiltonian) is a research step and is NOT solved here.
        """
        raise NotImplementedError(
            "project_velocity_to_flux is a scaffold hook: the canonical (commuting) RT projection of the "
            "MFG drift v = -coupling*grad(U) — welding the H1 value gradient into the H(div) flux through "
            "the nonlinear Hamiltonian — is the research step, not yet implemented. The building-block "
            "operators (divergence / flux_mass / density_mass) ARE available. See the module docstring."
        )
