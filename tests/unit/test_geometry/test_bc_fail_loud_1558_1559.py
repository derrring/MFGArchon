"""Fail-loud guards for silent-wrong BC handling (Issues #1558, #1559).

Each converts a silent-wrong (a fabricated normal, a wrong-BC default, a silently
no-flux-coerced dirichlet) into an explicit raise. All three paths are off published
numerics (no experiment / shipped example reaches them), so these pin the fail-loud
behavior rather than a numeric result.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def test_sdf_vanishing_gradient_normal_fails_loud():
    """#1558: SDFParticleBCHandler._compute_normal fabricated an arbitrary [1,0,...] normal when
    the finite-difference SDF gradient vanished -- reflecting a particle along a geometry-independent
    direction (silent-wrong). A constant SDF has zero gradient everywhere, so the normal is undefined;
    it must raise (mirroring project_to_domain's #1047 raise)."""
    from mfgarchon.geometry.boundary import SDFParticleBCHandler

    handler = SDFParticleBCHandler(lambda pts: -0.5 * np.ones(np.asarray(pts).shape[0]), dimension=2)
    with pytest.raises(RuntimeError, match="vanishing SDF gradient"):
        handler._compute_normal(np.array([0.3, 0.4]))


def test_tensor_grid_unknown_bc_type_fails_loud():
    """#1558: get_boundary_handler(bc_type) silently defaulted an unrecognized bc_type to periodic
    (1D) / neumann (nD) -- and its docstring advertised periodic_x/periodic_both/mixed keys that were
    never in either factory, so all of them silently became the default BC. An unrecognized key must
    raise, not substitute a different BC. The bc_type factory is reached only when no BC is stored
    (Priority 1 short-circuits otherwise), so clear it first via the documented 'None to clear' setter."""
    grid1d = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=no_flux_bc(dimension=1))
    grid1d.set_boundary_conditions(None)
    with pytest.raises(ValueError, match="Unsupported 1D bc_type"):
        grid1d.get_boundary_handler("bogus")

    grid2d = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[11, 11], boundary_conditions=no_flux_bc(dimension=2)
    )
    grid2d.set_boundary_conditions(None)
    with pytest.raises(ValueError, match="Unsupported 2D bc_type"):
        grid2d.get_boundary_handler("periodic_x")  # advertised in the old docstring, never implemented

    # A supported key must still resolve (the raise is scoped to unknown keys only).
    assert grid1d.get_boundary_handler("periodic") is not None
    assert grid2d.get_boundary_handler("no_flux") is not None


def _small_1d_problem(n=11):
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[n], boundary_conditions=no_flux_bc(dimension=1))
    comps = MFGComponents(
        m_initial=lambda x: np.ones_like(np.asarray(x, dtype=float)),
        u_terminal=lambda x: 0.0 * np.asarray(x, dtype=float),
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
    )
    return MFGProblem(geometry=grid, T=0.1, Nt=2, sigma=0.3, components=comps)


def test_legacy_mishandled_bc_fails_loud():
    """#1559: the FP-FDM time-stepping assembly treated ANY legacy fdm_bc_1d BoundaryConditions as
    no-flux (the except-AttributeError branch). _is_dirichlet_at_point can't see a legacy BC (no
    is_uniform -> returns False), so a legacy dirichlet was silently assembled as no-flux; a legacy
    'periodic' likewise gets NO wrap (byte-identical to legacy no_flux, O(1) off canonical periodic
    once mass reaches the wall). It must raise for legacy dirichlet/robin/periodic; only legacy
    neumann/no_flux (which ARE no-flux) still assemble."""
    from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
    from mfgarchon.geometry.boundary.fdm_bc_1d import BoundaryConditions as LegacyBC

    prob = _small_1d_problem()
    n = 11
    m0 = np.ones(n)
    drift = np.zeros((prob.Nt + 1, n))
    solver = FPFDMSolver(prob)

    for legacy_type in ("dirichlet", "periodic"):
        solver.boundary_conditions = LegacyBC(type=legacy_type, left_value=0.0, right_value=0.0)
        with pytest.raises(NotImplementedError, match="1559"):
            solver.solve_fp_system(m0.copy(), drift_field=drift, volatility_field=0.3)

    # Legacy neumann/no_flux ARE no-flux -> still assemble (finite, no raise).
    solver.boundary_conditions = LegacyBC(type="neumann", left_value=0.0, right_value=0.0)
    M = solver.solve_fp_system(m0.copy(), drift_field=drift, volatility_field=0.3)
    assert np.all(np.isfinite(M))


def test_periodic_and_no_flux_diverge_by_o1_which_is_why_coercion_is_not_harmless():
    """The guard refuses to coerce a legacy periodic BC to no-flux. This measures the difference.

    Issue #1714: a fail-loud test that only asserts `pytest.raises` records that the guard fires,
    never what it prevents. The mutation that reddens such a test is "delete the guard", and its
    output is `DID NOT RAISE` — a symptom with no diagnosis.

    The guard's own comment states the claim and says it was "verified with an off-center bump",
    but that verification was never committed. Reproduced here: a bump at x=0.12, sigma=0.35,
    T=0.4 on 61 nodes, solved once under each canonical BC.

        mass in the last five nodes, no-flux    3.24e-04
        mass in the last five nodes, periodic   1.10e-01     340x
        relative L1 difference over the field   55%

    Coercing periodic to no-flux does not perturb the answer, it replaces it: mass that should
    wrap to the far side is held against the near wall instead. The mutation that reddens THIS
    test is a change to the `divergence_upwind` periodic wrap in assembly, not the deletion of a
    guard. Scope: the wrap is copy-pasted into all four advection-scheme interior handlers and
    only the default one is covered here -- disabling it in the gradient_upwind,
    gradient_centered or divergence_centered handlers leaves this green.
    """
    import numpy as np

    from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
    from mfgarchon.core.mfg_problem import MFGComponents, MFGProblem
    from mfgarchon.geometry import TensorProductGrid
    from mfgarchon.geometry.boundary import no_flux_bc, periodic_bc

    n_points, n_steps = 61, 20
    xs = np.linspace(0.0, 1.0, n_points)

    def final_density(boundary_conditions):
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[n_points], boundary_conditions=boundary_conditions)
        components = MFGComponents(
            m_initial=lambda x: 1.0,
            u_terminal=lambda x: 0.0,
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0),
                coupling=lambda m: m,
                coupling_dm=lambda m: 1.0,
            ),
        )
        problem = MFGProblem(geometry=geometry, T=0.4, Nt=n_steps, sigma=0.35, components=components)
        # Off-centre so the near wall is reached long before the far one; a centred bump would
        # reach both walls together and the two BCs would look alike.
        m_initial = np.exp(-((xs - 0.12) ** 2) / (2 * 0.05**2))
        m_initial /= m_initial.sum()
        return FPFDMSolver(problem).solve_fp_system(m_initial, potential_field=np.zeros((n_steps + 1, n_points)))[-1]

    reflecting = final_density(no_flux_bc(dimension=1))
    wrapping = final_density(periodic_bc(dimension=1))

    far_wall = slice(-5, None)
    assert wrapping[far_wall].sum() > 100 * reflecting[far_wall].sum(), (
        f"periodic must carry mass to the far wall that no-flux does not: got "
        f"{wrapping[far_wall].sum():.3e} against {reflecting[far_wall].sum():.3e}. If these ever "
        f"agree, coercing one BC to the other is harmless and the guard is refusing nothing."
    )

    relative_l1 = np.abs(wrapping - reflecting).sum() / reflecting.sum()
    assert relative_l1 > 0.2, (
        f"the two boundary conditions must differ by O(1) over the field, not at the margin; got {relative_l1:.1%}"
    )

    for name, density in (("no-flux", reflecting), ("periodic", wrapping)):
        assert np.all(np.isfinite(density)), f"{name} produced a non-finite density"
        assert density.min() >= -1e-12, f"{name} produced a negative density: {density.min():.3e}"
        # Two-sided. Without this the assertions above are one-sided -- ANY corruption that makes
        # the two BCs differ MORE passes. Measured: routing the no-flux wall to the interior
        # handler leaks 54% of its mass (sum 0.461 instead of 1.0) and makes the test greener,
        # ratio 340 -> 617. Residuals here are 3.9e-15 and 4.4e-16, so this has five orders of
        # margin over the tolerance.
        assert abs(density.sum() - 1.0) < 1e-10, (
            f"{name} did not conserve mass: sum {density.sum():.6f}. A leak makes the two BCs "
            f"differ more, so the comparison above would read it as success."
        )
