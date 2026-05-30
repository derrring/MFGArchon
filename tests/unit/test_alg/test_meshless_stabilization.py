"""Issue #1145 (Bug B): opt-in streamline-diffusion stabilization + Newton-inner for the
meshless-Galerkin scheme.

The central (Bubnov-)Galerkin FP advection is not an M-matrix, so undamped it produces
negative density undershoots that the positivity clip rectifies into injected mass and
blows up the coupled solve; and the Picard inner path treats the quadratic Hamiltonian
explicitly and self-amplifies. The recipe adds (opt-in) a symmetric streamline-diffusion
block to BOTH operators (preserving A_FP = A_HJB^T) and an opt-in Newton inner solver.
Both default OFF, so existing behaviour is byte-identical.

Validated end-to-end in mfg-research/scripts/{prototype_tight,decoupled_lq_anchor}.py.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.alg.numerical.meshless_galerkin.discretization import MeshlessGalerkinDiscretization
from mfgarchon.alg.numerical.meshless_galerkin.quadrature import tensor_gauss
from mfgarchon.alg.numerical.weak_form_hjb_solver import WeakFormHJBSolver


def _disc(d: int, n_per: int):
    ax = np.linspace(0.0, 1.0, n_per)
    mesh = np.meshgrid(*([ax] * d), indexing="ij")
    nodes = np.stack([m.ravel() for m in mesh], axis=1)
    h = 1.0 / (n_per - 1)
    rho = 3.5 * h if d == 1 else 2.6 * h
    pts, wts = tensor_gauss([(0.0, 1.0)] * d, n_cells=n_per - 1, n_gauss=4)
    return MeshlessGalerkinDiscretization(nodes, rho, 2, pts, wts, backend="numpy"), nodes


# --- streamline-diffusion operator invariants --------------------------------
@pytest.mark.parametrize(("d", "n_per"), [(1, 11), (2, 7)])
def test_streamline_diffusion_symmetric_psd_mass_conserving(d, n_per):
    """S is symmetric (=> preserves A_FP=A_HJB^T when added to both), PSD (SUPG tau>=0),
    and S@1=0 (=> keeps the FP mass-conserving)."""
    disc, _ = _disc(d, n_per)
    velocity = np.ones((d, disc.n_dof))  # constant unit drift
    s = disc.streamline_diffusion(velocity, diffusion=0.045).toarray()
    assert np.max(np.abs(s - s.T)) < 1e-12
    assert np.max(np.abs(s @ np.ones(disc.n_dof))) < 1e-10
    assert np.linalg.eigvalsh(0.5 * (s + s.T)).min() > -1e-10


def test_streamline_diffusion_vanishes_without_drift_or_strength():
    disc, _ = _disc(1, 11)
    assert disc.streamline_diffusion(np.zeros((1, disc.n_dof)), 0.045).nnz == 0
    assert disc.streamline_diffusion(np.ones((1, disc.n_dof)), 0.045, c_scale=0.0).nnz == 0


# --- the SD block is added IDENTICALLY to FP and HJB (duality preservation) ----
def _meshless_pair(n=21, sd_scale=1.0):
    import numpy as _np

    from mfgarchon import MFGProblem
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
    from mfgarchon.core.mfg_problem import MFGComponents
    from mfgarchon.factory.scheme_factory import create_paired_solvers
    from mfgarchon.geometry import TensorProductGrid
    from mfgarchon.geometry.boundary import no_flux_bc
    from mfgarchon.types.schemes import NumericalScheme

    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[n], boundary_conditions=no_flux_bc(dimension=1))
    H = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0), coupling=lambda m: m, coupling_dm=lambda m: 0.0 * m + 1.0
    )
    comp = MFGComponents(
        hamiltonian=H, m_initial=lambda x: _np.exp(-40 * (x - 0.35) ** 2), u_terminal=lambda x: 0.5 * (x - 0.5) ** 2
    )
    problem = MFGProblem(geometry=grid, components=comp, T=0.5, Nt=20, sigma=0.3, coupling_coefficient=0.5)
    cloud = _np.linspace(0.0, 1.0, n)[:, None]
    return create_paired_solvers(
        problem,
        NumericalScheme.MESHLESS_GALERKIN,
        hjb_config={
            "collocation_points": cloud,
            "delta": 3.5 / (n - 1),
            "streamline_diffusion_scale": sd_scale,
            "use_newton": True,
        },
    )


def test_sd_block_identical_in_fp_and_hjb():
    """The SD matrix the FP advection adds equals the one the HJB Newton Jacobian adds
    (same velocity b=-coupling*grad(U), same D), so A_FP = A_HJB^T survives stabilization.
    Uses a D that is NOT 0.5*sigma^2 to confirm the diffusion coefficient is THREADED into
    the FP SD (volatility-aware), not hardcoded -- otherwise the duality breaks when a
    volatility_field != sigma is supplied."""
    hjb, fp = _meshless_pair(n=21, sd_scale=1.0)
    x = hjb._disc.dof_coordinates[:, 0]
    u = 0.5 * (x - 0.5) ** 2
    d_test = 0.1  # deliberately != 0.5 * sigma^2 (= 0.045)
    s_hjb = hjb._stabilization_terms(u, d_test).toarray()

    with_sd = fp._build_advection(u, d_test).toarray()
    fp._sd_scale = 0.0
    without_sd = fp._build_advection(u, d_test).toarray()
    fp._sd_scale = 1.0
    s_fp = with_sd - without_sd

    assert np.max(np.abs(s_hjb - s_fp)) < 1e-12, "FP and HJB must add the SAME S (duality)"
    assert np.max(np.abs(s_fp - s_fp.T)) < 1e-12


def test_stabilization_requires_newton_fails_fast():
    """streamline_diffusion_scale > 0 with use_newton=False is rejected at construction: S
    enters only the HJB Newton Jacobian, so with Picard the HJB omits S while the FP carries
    it (duality break). The two knobs are duality-coupled and must not be set independently."""
    import numpy as _np

    from mfgarchon import MFGProblem
    from mfgarchon.alg.numerical.meshless_galerkin.hjb_solver import MeshlessGalerkinHJBSolver
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
    from mfgarchon.core.mfg_problem import MFGComponents
    from mfgarchon.geometry import TensorProductGrid
    from mfgarchon.geometry.boundary import no_flux_bc

    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=no_flux_bc(dimension=1))
    H = SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0))
    comp = MFGComponents(hamiltonian=H, m_initial=lambda x: _np.ones_like(x), u_terminal=lambda x: x * 0)
    problem = MFGProblem(geometry=grid, components=comp, T=0.5, Nt=5, sigma=0.3)
    cloud = _np.linspace(0.0, 1.0, 11)[:, None]
    with pytest.raises(ValueError, match="streamline_diffusion_scale > 0 requires use_newton=True"):
        MeshlessGalerkinHJBSolver(
            problem, collocation_points=cloud, delta=0.35, streamline_diffusion_scale=1.0, use_newton=False
        )


def test_factory_threads_sd_scale_to_both():
    """streamline_diffusion_scale passed only in hjb_config propagates to the FP solver, so
    the pair cannot be half-stabilized (which would break the duality)."""
    hjb, fp = _meshless_pair(n=15, sd_scale=1.0)  # _meshless_pair sets the scale only in hjb_config
    assert hjb._sd_scale == 1.0
    assert fp._sd_scale == 1.0


def test_recipe_defaults_off_byte_identical():
    """A DEFAULT meshless pair (no recipe kwargs) has streamline_diffusion_scale=0 and
    use_newton off, so existing behaviour is byte-identical."""
    import numpy as _np

    from mfgarchon import MFGProblem
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
    from mfgarchon.core.mfg_problem import MFGComponents
    from mfgarchon.factory.scheme_factory import create_paired_solvers
    from mfgarchon.geometry import TensorProductGrid
    from mfgarchon.geometry.boundary import no_flux_bc
    from mfgarchon.types.schemes import NumericalScheme

    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[15], boundary_conditions=no_flux_bc(dimension=1))
    H = SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0))
    comp = MFGComponents(hamiltonian=H, m_initial=lambda x: _np.ones_like(x), u_terminal=lambda x: x * 0)
    problem = MFGProblem(geometry=grid, components=comp, T=0.5, Nt=10, sigma=0.3)
    cloud = _np.linspace(0.0, 1.0, 15)[:, None]
    hjb, fp = create_paired_solvers(
        problem, NumericalScheme.MESHLESS_GALERKIN, hjb_config={"collocation_points": cloud, "delta": 3.5 / 14}
    )
    assert fp._sd_scale == 0.0
    assert hjb._sd_scale == 0.0
    assert hjb._use_newton_default is False
    assert hjb._stabilization_terms(np.zeros(hjb.n_dof), 0.045) is None


def test_base_stabilization_hook_is_noop():
    """The base hook returns None so FEM (which inherits it) adds no SD -- byte-identical."""

    class _Dummy(WeakFormHJBSolver):
        def __init__(self):
            pass

    assert _Dummy()._stabilization_terms(np.zeros(3), 0.1) is None


# --- end-to-end: the recipe converges and matches a conservative reference -----
@pytest.mark.integration
def test_recipe_coupled_matches_fdm_on_wellposed_problem():
    """On a decoupled (unique) LQ problem all schemes must agree. The meshless recipe
    (Newton + SD) must converge and track FDM-upwind (the mass-conserving reference)."""
    import numpy as _np

    from mfgarchon import MFGProblem
    from mfgarchon.config import MFGSolverConfig
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
    from mfgarchon.core.mfg_problem import MFGComponents
    from mfgarchon.factory.scheme_factory import create_paired_solvers
    from mfgarchon.geometry import TensorProductGrid
    from mfgarchon.geometry.boundary import no_flux_bc
    from mfgarchon.types.schemes import NumericalScheme

    def mk():
        grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1))
        H = SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=1.0),
            potential=lambda x, t: 0.5 * 4.0 * (x[0] - 0.5) ** 2,
            coupling=None,
        )
        comp = MFGComponents(
            hamiltonian=H, m_initial=lambda x: _np.exp(-40 * (x - 0.35) ** 2), u_terminal=lambda x: 0.5 * (x - 0.5) ** 2
        )
        return MFGProblem(geometry=grid, components=comp, T=0.5, Nt=20, sigma=0.3, coupling_coefficient=0.5)

    x = _np.linspace(0.0, 1.0, 21)
    dx = x[1] - x[0]

    def mean_std(row):
        m = _np.maximum(row, 0.0)
        mass = m.sum() * dx
        mean = (x * m).sum() * dx / mass
        return mean, _np.sqrt(((x - mean) ** 2 * m).sum() * dx / mass)

    ref = _np.asarray(
        mk().solve(scheme=NumericalScheme.FDM_UPWIND, max_iterations=200, tolerance=1e-6, verbose=False).M
    )
    ref_mean, ref_std = mean_std(ref[-1])

    prob = mk()
    cloud = x[:, None].copy()
    hjb, fp = create_paired_solvers(
        prob,
        NumericalScheme.MESHLESS_GALERKIN,
        hjb_config={
            "collocation_points": cloud,
            "delta": 3.5 / 20,
            "use_newton": True,
            "streamline_diffusion_scale": 1.0,
        },
    )
    res = prob.solve(
        hjb_solver=hjb, fp_solver=fp, config=MFGSolverConfig(), max_iterations=200, tolerance=1e-6, verbose=False
    )
    M = _np.asarray(res.M)
    assert _np.all(_np.isfinite(M)), "recipe blew up"
    assert M[-1].min() > -1e-9, "density positive (SD suppresses undershoots)"
    mean, std = mean_std(M[-1])
    assert abs(mean - ref_mean) < 2e-2, f"mean {mean:.4f} vs FDM {ref_mean:.4f}"
    assert abs(std - ref_std) < 2e-2, f"std {std:.4f} vs FDM {ref_std:.4f}"
