"""Conservation tests for the FP FDM solver.

Verifies the production Fokker-Planck implicit assembly conserves mass and stays
positive -- the linear-algebraic root of MFG mass conservation (Achdou &
Capuzzo-Dolcetta 2010, Issue #486).

History: an earlier ``TestFPMatrixConservation`` asserted column-sum = 1/dt on a
*test-local re-implementation* of the matrix (a shadow that could pass even if the
production ``FPFDMSolver`` assembly diverged) and a no-op placeholder test. Those are
replaced by ``TestFPProductionConservation``, which runs the real solver: the
machine-precision mass invariant is the signal a wrong boundary stencil (the
retrospect's non-conservation-at-boundary bug class) would break, and the loose
``rtol=0.1`` end-to-end checks in ``test_fp_fdm_solver.py`` would miss.
``TestLinearConstraintMatrixAssembly`` below already points at production
(``_build_diffusion_matrix_with_bc``) and is unchanged.
"""

from __future__ import annotations

import pytest

import numpy as np
import scipy.sparse as sparse

from mfgarchon import MFGProblem
from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import neumann_bc, no_flux_bc, periodic_bc, robin_bc
from mfgarchon.geometry.boundary.invariants import mass_drift
from mfgarchon.utils.numerical.quadrature import quadrature_weights_1d


def _w(n: int, dx: float) -> np.ndarray:
    """The control volumes on an endpoint-inclusive axis: the trapezoid weights (#2145).

    `sum(m) * dx` gives both wall nodes a full cell on a grid whose wall lies ON the node, so it is
    a different functional from the mass -- and it is the one the wall rows used to telescope
    against, which is why a scheme losing a quarter of the density could report machine zero.
    """
    return quadrature_weights_1d(np.arange(n, dtype=float) * dx)


def _cos_density(xx):
    return 1.0 + 0.4 * np.cos(2 * np.pi * np.asarray(xx))


def _zero_drift_density_evolution(bc, n=41, nt=40, T=0.2, sigma=0.5, m_init=_cos_density):
    """Evolve an initial density under the production FP implicit step, zero drift.

    Default state is m = 1 + 0.4 cos(2 pi x). Zero drift isolates pure diffusion, where
    every conservative scheme must preserve total mass exactly (no advective boundary
    subtlety). Returns the full M history.
    """
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[n], boundary_conditions=bc)
    H = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: 0.0 * np.asarray(m),
        coupling_dm=lambda m: 0.0 * np.asarray(m),
    )
    x = np.linspace(0.0, 1.0, n)
    comps = MFGComponents(
        m_initial=m_init,
        u_terminal=lambda xx: 0.0 * np.asarray(xx),
        hamiltonian=H,
    )
    prob = MFGProblem(geometry=grid, T=T, Nt=nt, sigma=sigma, components=comps)
    solver = FPFDMSolver(prob)
    m0 = np.asarray(m_init(x), dtype=float)
    # drift_field as an (nt+1, n) array routes through the implicit per-point assembly
    M = solver.solve_fp_system(m0.copy(), drift_field=np.zeros((nt + 1, n)), volatility_field=sigma)
    return M, 1.0 / (n - 1)


class TestFPProductionConservation:
    """Mass + positivity of the real FPFDMSolver implicit step (not a shadow matrix)."""

    def test_mass_conserved_periodic_zero_drift(self):
        """Quadrature is `trapezoid`, not `sum`, because the grid is endpoint-inclusive (#1822).

        `M.sum(axis=1) * dx` counts the shared node twice -- it read 1.035 for a datum of mass 1
        even at t=0. That was consistent while the solver treated the repeat as its own cell on a
        torus of length 1 + dx; now that it does not, the rectangle rule is measuring a quantity
        the solve does not conserve, and the drift it reports is one cell's worth of density
        (verified: the gap equals `m[0]*dx` to ten digits). Under the owner's quadrature the same
        solve conserves mass to 1.2e-14.
        """
        M, _dx = _zero_drift_density_evolution(periodic_bc(dimension=1))
        x = np.linspace(0.0, 1.0, M.shape[1])
        relerr = mass_drift(M, x)
        assert relerr < 1e-12, f"periodic zero-drift mass not conserved: relerr {relerr:.2e}"

    def test_mass_conserved_no_flux_zero_drift(self):
        """Same quadrature as the periodic test above, for the same reason. It now PASSES.

        History, because the number it reads went from 5.77e-03 to 1e-14 without this file moving.
        This asserted `M.sum(axis=1) * dx` directly below a docstring explaining why that
        quadrature is wrong on this grid (#1822). The rectangle rule over-reports by exactly the
        endpoint half-weights: `dx*(m[0]+m[-1])/2 = 0.035` at t=0, matched to 1e-12, so it calls a
        datum of mass 1 a datum of mass 1.035 before anything has evolved. It then reads
        1.29e-14 drift, because the live wall assembly
        (`fp_fdm_bc.add_boundary_no_flux_entries_conservative`) is designed to hold column sum =
        1/dt, which conserves that rectangle sum exactly.

        Under the owner's quadrature the same solve loses 5.77e-3 of its mass, and the loss is
        O(h): 5.77e-3 / 3.00e-3 / 1.53e-3 / 7.71e-4 at n = 41 / 81 / 161 / 321.

        The wall Laplacian is the cause and is separately measured: on `u = cos(pi x)`, where
        `u'(0) = 0` holds exactly, the column-conservative wall row converges to HALF of
        `u''(0)` -- order 0.00 -- while the node-centred row `[-2, 2]/h^2` gives order 2.00
        (interior control 6.8e-16). That is #1904 / #1935.

        It was marked `xfail(strict=True)` rather than pinned to 5.77e-3, because pinning the wrong
        number would have made the defect a specification. #2145 gave the wall node the half cell it
        owns, the test flipped to `XPASS(strict)`, and the marker is gone -- which is the retirement
        condition working as designed, with the failure message as the instruction.

        It has also INHERITED the #1250 signal. A rectangle-sum guard used to sit beside this one
        and catch the absorbing-wall regression by a ~3% drop over 5 steps; that guard asserted a
        functional the corrected wall no longer conserves, so it is retired. An absorbing wall loses
        real mass, so it moves this check too -- and this one cannot be satisfied by a wall that
        conserves the wrong thing, which is what let #1250's defect class survive beside a green
        conservation test in the first place.
        """
        M, _dx = _zero_drift_density_evolution(no_flux_bc(dimension=1))
        x = np.linspace(0.0, 1.0, M.shape[1])
        relerr = mass_drift(M, x)
        assert relerr < 1e-12, f"no-flux zero-drift mass not conserved: relerr {relerr:.2e}"

    def test_density_stays_positive(self):
        # an M-matrix implicit step keeps a positive initial density positive
        M, _ = _zero_drift_density_evolution(periodic_bc(dimension=1))
        assert np.all(M[-1] > -1e-12), "production FP step produced a negative density"

        # The line above cannot fail on this fixture: m0 = 1 + 0.4 cos(2 pi x) has min 0.6 and
        # pure diffusion contracts toward the mean, so the whole sweep stays above 0.6 and the
        # -1e-12 bound is unreachable by construction. Repeat on a state that actually reaches
        # the floor: exp(-500 (x-0.5)^2) spans 54 orders of magnitude at t=0, and one implicit
        # step still leaves tail values at 2.6e-08 beside a peak of ~1. That is the regime where
        # a sign error in an off-diagonal (i.e. a loss of the M-matrix property this test is
        # named for) produces a negative entry rather than a merely inaccurate one.
        M_peaked, _ = _zero_drift_density_evolution(
            periodic_bc(dimension=1), m_init=lambda xx: np.exp(-500.0 * (np.asarray(xx) - 0.5) ** 2)
        )
        assert M_peaked.min() > -1e-12, "production FP step produced a negative density from a peaked state"

        # The default (cos) solve has a machine-precision closed form, and nothing in this file
        # used it. The k=1 Fourier mode is an eigenvector of the periodic three-point Laplacian
        # with eigenvalue -(2 - 2 cos(2 pi h))/h^2, so under backward Euler its amplitude is
        # multiplied by 1/(1 + dt*lam) per step, exactly.
        n, nt, T, sigma = 41, 40, 0.2, 0.5
        h, dt = 1.0 / (n - 1), T / nt
        lam = (sigma**2 / 2) * (2 - 2 * np.cos(2 * np.pi * h)) / h**2
        expected_amplitude = 0.4 * (1 + dt * lam) ** (-nt)
        measured_amplitude = (M[-1].max() - M[-1].min()) / 2
        # Measured relative error 6.4e-15; 1e-12 is a ~156x margin. This pins the sigma -> D
        # convention and the time discretization together: D = sigma gives 0.00934 and D = sigma^2
        # gives 0.0585 against the true 0.15118, and forward Euler differs by 2.4e-2 relative --
        # every one of them 10 orders of magnitude above this threshold.
        assert abs(measured_amplitude - expected_amplitude) / expected_amplitude < 1e-12


class TestNeumannBCImplicitFP:
    """Issue #1250 pinning tests: uniform 'neumann' BC must conserve mass in implicit FDM.

    Before the fix, neumann was routed to the interior handler (is_no_flux=False,
    is_uniform=True → neither branch of the dispatch), producing a ghost-zero absorbing
    wall row (+D/dx² diagonal, missing coupling → mass drains 3-7% / 5 steps).
    After the fix, neumann routes to the same no-flux boundary handler as no_flux.
    """

    def test_mass_conserved_neumann_zero_value_zero_drift(self):
        """The twin of the no-flux test above; `neumann_bc(0)` is the same physical wall.

        `test_neumann_matches_no_flux_implicit` below already asserts the two produce byte-identical
        histories, so these two cannot diverge: whatever fixes one fixes both, and if only one moves,
        that test goes red first. Both were `xfail(strict=True)` until #2145 corrected the wall;
        both flipped together, as that identity predicts.
        """
        M, _dx = _zero_drift_density_evolution(neumann_bc(dimension=1))
        x = np.linspace(0.0, 1.0, M.shape[1])
        relerr = mass_drift(M, x)
        assert relerr < 1e-12, f"neumann zero-drift mass not conserved: relerr {relerr:.2e}"

    def test_neumann_matches_no_flux_implicit(self):
        """neumann_bc(0) and no_flux_bc must produce byte-identical mass histories.

        Both are homogeneous-Neumann; the operator paths already treat them identically
        (laplacian.py:307, advection.py:316).  After the fix the implicit assembly must
        as well.
        """
        M_nf, _dx = _zero_drift_density_evolution(no_flux_bc(dimension=1))
        M_ne, _ = _zero_drift_density_evolution(neumann_bc(dimension=1))
        max_diff = np.max(np.abs(M_nf - M_ne))
        assert max_diff < 1e-14, (
            f"neumann and no_flux produce different density histories (Issue #1250): "
            f"max|diff|={max_diff:.2e} (should be 0 to floating-point)"
        )

    def test_robin_bc_fails_loud_implicit(self):
        """Uniform 'robin' BC must raise NotImplementedError (not silently absorb mass).

        Issue #1250: until a correct Robin stencil is implemented, failing loud is
        correct (fail-fast doctrine); silently assembling an absorbing wall is not.
        Issue #1456: now caught at construction by the BC-capability gate (message says ROBIN).
        """
        with pytest.raises(NotImplementedError, match=r"(?i)robin"):
            _zero_drift_density_evolution(robin_bc(dimension=1))


class TestConservativeAdvection:
    """Conservative finite-volume divergence at no-flux walls (Issue #1184).

    The default node-based upwind divergence is conservative in the interior but leaks
    +-(v*m) through no-flux walls; under strong drift into a wall the explicit-drift FP
    solve loses (or even negates) mass. The opt-in ``mass_conservative=True`` FV
    flux-difference zeroes the wall flux so the assembled operator is column-conservative.
    """

    @staticmethod
    def _adv_matrix(drift_val, n, conservative, bc):
        from mfgarchon.alg.numerical.fp_solvers.fp_fdm_advection import compute_advection_from_drift_nd

        dx = 1.0 / (n - 1)
        drift = np.full(n, drift_val)
        cols = []
        for j in range(n):
            e = np.zeros(n)
            e[j] = 1.0
            cols.append(compute_advection_from_drift_nd(e, drift, (dx,), 1, bc=bc, mass_conservative=conservative))
        return np.array(cols).T, dx

    def test_conservative_operator_column_sum_zero_no_flux(self):
        """1^T A = 0 (mass conserved) for the conservative operator at no-flux walls."""
        for drift_val in (0.3, 0.8):
            A, dx = self._adv_matrix(drift_val, 41, conservative=True, bc=no_flux_bc(dimension=1))
            col_sums = _w(41, dx) @ A
            assert np.max(np.abs(col_sums)) < 1e-12, (
                f"conservative operator leaks at drift={drift_val}: max|colsum|={np.max(np.abs(col_sums)):.2e}"
            )

    def test_conservative_operator_column_sum_zero_periodic(self):
        """1^T A = 0 for the conservative operator under periodic wrap."""
        A, dx = self._adv_matrix(0.5, 41, conservative=True, bc=None)
        assert np.max(np.abs(A.sum(axis=0))) * dx < 1e-12

    def test_default_leaks_but_conservative_holds_at_wall(self):
        """Direct contrast in a pure-advection wall-piling loop: the default node-based
        upwind divergence loses mass at the no-flux wall, the conservative FV form does not.
        (The default is state-dependent/non-linear, so it is exercised in a real solve loop
        rather than via a meaningless basis-vector matrix probe.)"""
        from mfgarchon.alg.numerical.fp_solvers.fp_fdm_advection import compute_advection_from_drift_nd

        n = 51
        x = np.linspace(0.0, 1.0, n)
        dx = x[1] - x[0]
        bc = no_flux_bc(dimension=1)
        drift = np.full(n, -0.8)
        dt = 0.2 * dx / 0.8  # CFL-stable for pure explicit advection

        w = _w(n, dx)

        def run(conservative):
            M = np.exp(-200.0 * (x - 0.18) ** 2)
            M /= float(w @ M)
            for _ in range(400):
                M = M - dt * compute_advection_from_drift_nd(M, drift, (dx,), 1, bc=bc, mass_conservative=conservative)
            return float(w @ M)

        mass_default = run(False)
        mass_conservative = run(True)
        assert abs(mass_default - 1.0) > 1e-3, f"expected default to leak at wall, got mass={mass_default:.6f}"
        assert abs(mass_conservative - 1.0) < 1e-12, (
            f"conservative form must preserve mass, got {mass_conservative:.8f}"
        )

    def test_explicit_drift_fp_conserves_mass_into_wall(self):
        """The explicit-drift FP step keeps mass=1 and density>=0 when strong drift piles
        density against a no-flux wall (pre-fix this leaked to negative mass)."""
        from mfgarchon.alg.numerical.fp_solvers.fp_fdm_time_stepping import solve_timestep_explicit_with_drift

        n = 51
        x = np.linspace(0.0, 1.0, n)
        dx = x[1] - x[0]
        dt = 5e-4
        bc = no_flux_bc(dimension=1)
        w = _w(n, dx)
        M = np.exp(-200.0 * (x - 0.18) ** 2)
        M /= float(w @ M)
        drift = np.full(n, -0.8)  # strong drift into the left wall
        for _ in range(800):
            M = solve_timestep_explicit_with_drift(M, drift, dt, 0.1, (dx,), 1, boundary_conditions=bc)
        mass = float(w @ M)
        assert abs(mass - 1.0) < 1e-9, f"explicit-drift FP leaked mass at wall: mass={mass:.8f}"
        assert M.min() > -1e-12, f"explicit-drift FP produced negative density: min={M.min():.2e}"

    def test_mass_conservative_requires_upwind_divergence(self):
        """mass_conservative is only valid for form='divergence', scheme='upwind'."""
        from mfgarchon.operators import AdvectionOperator

        v = np.zeros((1, 5))
        with pytest.raises(ValueError, match="mass_conservative"):
            AdvectionOperator(v, [0.25], (5,), scheme="centered", form="divergence", mass_conservative=True)
        with pytest.raises(ValueError, match="mass_conservative"):
            AdvectionOperator(v, [0.25], (5,), scheme="upwind", form="gradient", mass_conservative=True)

    def test_conservative_rejects_unsupported_bc(self):
        """The conservative path supports only no-flux/Neumann/reflecting/periodic (zero wall
        flux); other BCs (e.g. Dirichlet inflow) raise rather than silently mishandling the wall."""
        from mfgarchon.alg.numerical.fp_solvers.fp_fdm_advection import compute_advection_from_drift_nd
        from mfgarchon.geometry.boundary import dirichlet_bc

        m = np.ones(11)
        drift = np.full(11, 0.4)
        with pytest.raises(NotImplementedError, match="conservative"):
            compute_advection_from_drift_nd(
                m, drift, (0.1,), 1, bc=dirichlet_bc(value=0.0, dimension=1), mass_conservative=True
            )

    def test_default_path_byte_identical_golden(self):
        """Pin the DEFAULT (mass_conservative=False) divergence to frozen values, so a future
        change to the node-based path is caught. The opt-in EOC-safety story depends on the
        default staying byte-identical for HJB/geometry/non-opted-in consumers (Issue #1184)."""
        from mfgarchon.alg.numerical.fp_solvers.fp_fdm_advection import compute_advection_from_drift_nd

        x = np.linspace(0.0, 1.0, 11)
        m = np.exp(-10.0 * (x - 0.5) ** 2)
        drift = np.full(11, 0.4)
        r = compute_advection_from_drift_nd(m, drift, (x[1] - x[0],), 1, bc=no_flux_bc(dimension=1))
        expected = np.array([0.0, 0.818693, 0.938069, -0.938069, -0.818693, 0.0])
        np.testing.assert_allclose(r[::2], expected, atol=1e-6)

    def test_tensor_explicit_path_records_its_wall_drift(self):
        """This path does NOT conserve mass, and the body below records how much (#2145 / #1904).

        The name and this docstring said "also conserves mass" over a body that pins a defect. Its
        advection is conservative -- `compute_advection_from_drift_nd(mass_conservative=True)`, wall
        cells on the h/2 control volume -- but its diffusion goes through `apply_diffusion`, i.e.
        the ghost path, whose wall row is half the mirror stencil. The two-sided band below brackets
        the measured 1.30e-03 and carries its own retirement condition.
        """
        from mfgarchon.alg.numerical.fp_solvers.fp_fdm_time_stepping import solve_timestep_tensor_explicit

        n = 51
        x = np.linspace(0.0, 1.0, n)
        dx = x[1] - x[0]
        bc = no_flux_bc(dimension=1)
        grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[n], boundary_conditions=bc)
        H = SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=1.0),
            coupling=lambda m: 0.0 * np.asarray(m),
            coupling_dm=lambda m: 0.0 * np.asarray(m),
        )
        comps = MFGComponents(
            m_initial=lambda xx: np.exp(-((np.asarray(xx) - 0.18) ** 2) / 0.005),
            u_terminal=lambda xx: 0.0 * np.asarray(xx),
            hamiltonian=H,
        )
        prob = MFGProblem(geometry=grid, T=0.4, Nt=50, sigma=0.05, components=comps)
        w = _w(n, dx)
        M = np.exp(-200.0 * (x - 0.18) ** 2)
        M /= float(w @ M)
        tensor = np.array([[0.05**2]])  # 1x1 diffusion tensor
        drift = np.full(n, -0.8)  # strong drift into the left wall
        for k in range(800):
            M = solve_timestep_tensor_explicit(
                M, None, prob, 5e-4, tensor, 1.0, (dx,), grid, 1, (n,), bc, k, drift=drift
            )
        # RECORDED DEFECT, not a contract (#2145 / #1904). This path's ADVECTION is conservative --
        # `compute_advection_from_drift_nd(..., mass_conservative=True)`, whose wall cells now use
        # the h/2 control volume -- but its DIFFUSION goes through `apply_diffusion`, i.e.
        # `DiffusionOperator` -> `laplacian_with_bc` -> `pad_array_with_ghosts`, the ghost path.
        # That path writes `ghost = u[0] + h*g`, so for g = 0 the wall row is (u[1] - u[0])/h^2 --
        # exactly HALF the mirror stencil's 2(u[1] - u[0])/h^2. Measured on u = cos(pi x), where
        # u''(0) = -9.86960: the sparse export gives -9.86453 and the ghost matvec -4.93227.
        #
        # Neither half is new. `test_ghost_spacing_1904` states in its own docstring that the ghost
        # certifies only what the caller asked for and "not that any derivative the solver goes on
        # to form is exact", and names the gap "the node-centring half of #1904", with #1902 blocked
        # on it. `test_laplacian_bc_equivalence` documents the matvec/sparse split as intentional
        # and declares `as_scipy_sparse()` the correct one. What is a defect is the ROUTING: an FP
        # step that must conserve mass is taking its diffusion from the path the repository itself
        # marks as boundary-incorrect.
        #
        # Not fixed here. The correct operator is scalar/diagonal-coefficient and this path carries
        # a full tensor, so the fix is a tensor-capable conservative assembly, not a redirect.
        #
        # Retirement: route the tensor diffusion through a conservative assembly and this fails at
        # 1e-6. Replace both bounds with `abs(mass - 1.0) < 1e-9`.
        mass = float(w @ M)
        assert abs(mass - 1.0) < 1e-2, f"tensor-explicit drift {abs(mass - 1.0):.3e} exceeds the recorded 1.30e-03"
        assert abs(mass - 1.0) > 1e-6, (
            f"tensor-explicit drift {abs(mass - 1.0):.3e} is below the recorded defect: the tensor "
            "diffusion now conserves. Tighten to `< 1e-9` and delete this pin (#2145 / #1904)."
        )
        assert M.min() > -1e-12, f"tensor-explicit path produced negative density: min={M.min():.2e}"


class TestLinearConstraintMatrixAssembly:
    """
    Test LinearConstraint-based matrix assembly.

    These tests verify the implementation of the matrix assembly protocol
    from docs/development/matrix_assembly_bc_protocol.md.

    The key principle is Algebraic-Geometric Equivalence:
    - Explicit solver (ghost cells) and implicit solver (matrix folding)
      must produce identical numerical results.
    """

    def test_neumann_bc_via_linear_constraint(self):
        """Test Neumann BC using LinearConstraint coefficient folding."""
        from mfgarchon.alg.numerical.fp_solvers.fp_fdm_time_stepping import (
            _build_diffusion_matrix_with_bc,
        )
        from mfgarchon.geometry.boundary.applicator_base import LinearConstraint

        # Setup
        Nx = 11
        dx = 0.1
        dt = 0.01
        D = 0.05  # diffusion coefficient

        shape = (Nx,)
        spacing = (dx,)
        ndim = 1

        # Neumann BC: du/dn = 0 -> u_ghost = u_inner (Tier 2)
        neumann_constraint = LinearConstraint(weights={0: 1.0}, bias=0.0)

        A, b_bc = _build_diffusion_matrix_with_bc(
            shape=shape,
            spacing=spacing,
            D=D,
            dt=dt,
            ndim=ndim,
            bc_constraint_min=neumann_constraint,
            bc_constraint_max=neumann_constraint,
        )

        # Check matrix shape
        assert A.shape == (Nx, Nx), f"Expected ({Nx}, {Nx}), got {A.shape}"

        # Check b_bc is zero for homogeneous Neumann
        np.testing.assert_allclose(b_bc, 0.0, atol=1e-14)

        # Check row sums (should equal 1/dt for implicit scheme)
        row_sums = np.array(A.sum(axis=1)).flatten()
        np.testing.assert_allclose(
            row_sums,
            1.0 / dt,
            rtol=1e-10,
            err_msg="Row sums should equal 1/dt",
        )

    def test_dirichlet_bc_via_linear_constraint(self):
        """Test Dirichlet BC using LinearConstraint coefficient folding."""
        from mfgarchon.alg.numerical.fp_solvers.fp_fdm_time_stepping import (
            _build_diffusion_matrix_with_bc,
        )
        from mfgarchon.geometry.boundary.applicator_base import LinearConstraint

        # Setup
        Nx = 11
        dx = 0.1
        dt = 0.01
        D = 0.05

        shape = (Nx,)
        spacing = (dx,)
        ndim = 1

        # Dirichlet BC: u = g at boundary (Tier 1)
        # For cell-centered: u_ghost = 2*g - u_inner
        # LinearConstraint: weights={0: -1.0}, bias=2*g
        g_left = 1.0
        g_right = 2.0
        dirichlet_left = LinearConstraint(weights={0: -1.0}, bias=2 * g_left)
        dirichlet_right = LinearConstraint(weights={0: -1.0}, bias=2 * g_right)

        A, b_bc = _build_diffusion_matrix_with_bc(
            shape=shape,
            spacing=spacing,
            D=D,
            dt=dt,
            ndim=ndim,
            bc_constraint_min=dirichlet_left,
            bc_constraint_max=dirichlet_right,
        )

        # Check matrix shape
        assert A.shape == (Nx, Nx)

        # Check b_bc has non-zero entries at boundaries
        # The bias terms should contribute to b_bc[0] and b_bc[-1]
        assert b_bc[0] != 0.0, "Left boundary should have BC contribution"
        assert b_bc[-1] != 0.0, "Right boundary should have BC contribution"

    def test_linear_extrapolation_bc_via_linear_constraint(self):
        """Test linear extrapolation BC using LinearConstraint (Tier 4)."""
        from mfgarchon.alg.numerical.fp_solvers.fp_fdm_time_stepping import (
            _build_diffusion_matrix_with_bc,
        )
        from mfgarchon.geometry.boundary.applicator_base import LinearConstraint

        # Setup
        Nx = 11
        dx = 0.1
        dt = 0.01
        D = 0.05

        shape = (Nx,)
        spacing = (dx,)
        ndim = 1

        # Linear extrapolation: u_ghost = 2*u[0] - u[1]
        # For left boundary: weights={0: 2.0, 1: -1.0}
        # For right boundary: weights={0: 2.0, 1: -1.0} (relative to boundary-adjacent)
        linear_extrap = LinearConstraint(weights={0: 2.0, 1: -1.0}, bias=0.0)

        A, b_bc = _build_diffusion_matrix_with_bc(
            shape=shape,
            spacing=spacing,
            D=D,
            dt=dt,
            ndim=ndim,
            bc_constraint_min=linear_extrap,
            bc_constraint_max=linear_extrap,
        )

        # Check matrix shape
        assert A.shape == (Nx, Nx)

        # Check b_bc is zero for extrapolation (no bias)
        np.testing.assert_allclose(b_bc, 0.0, atol=1e-14)

        # Matrix should be non-singular (solvable)
        m_test = np.ones(Nx)
        b_test = m_test / dt
        m_result = sparse.linalg.spsolve(A, b_test)
        assert not np.any(np.isnan(m_result)), "Matrix should be solvable"

    def test_2d_neumann_bc_via_linear_constraint(self):
        """Test 2D Neumann BC using LinearConstraint."""
        from mfgarchon.alg.numerical.fp_solvers.fp_fdm_time_stepping import (
            _build_diffusion_matrix_with_bc,
        )
        from mfgarchon.geometry.boundary.applicator_base import LinearConstraint

        # Setup
        Nx, Ny = 6, 6
        dx, dy = 0.1, 0.1
        dt = 0.01
        D = 0.05

        shape = (Nx, Ny)
        spacing = (dx, dy)
        ndim = 2

        # Neumann BC
        neumann = LinearConstraint(weights={0: 1.0}, bias=0.0)

        A, b_bc = _build_diffusion_matrix_with_bc(
            shape=shape,
            spacing=spacing,
            D=D,
            dt=dt,
            ndim=ndim,
            bc_constraint_min=neumann,
            bc_constraint_max=neumann,
        )

        N_total = Nx * Ny
        assert A.shape == (N_total, N_total)

        # Check b_bc is zero
        np.testing.assert_allclose(b_bc, 0.0, atol=1e-14)

        # Check row sums
        row_sums = np.array(A.sum(axis=1)).flatten()
        np.testing.assert_allclose(
            row_sums,
            1.0 / dt,
            rtol=1e-10,
            err_msg="Row sums should equal 1/dt in 2D",
        )

    def test_mass_conservation_with_linear_constraint_neumann(self):
        """Test mass conservation using LinearConstraint-based matrix assembly."""
        from mfgarchon.alg.numerical.fp_solvers.fp_fdm_time_stepping import (
            _build_diffusion_matrix_with_bc,
        )
        from mfgarchon.geometry.boundary.applicator_base import LinearConstraint

        # Setup
        Nx = 21
        dx = 1.0 / (Nx - 1)
        dt = 0.01
        sigma = 0.5
        D = sigma**2 / 2.0

        shape = (Nx,)
        spacing = (dx,)
        ndim = 1

        # Initial Gaussian density
        x = np.linspace(0, 1, Nx)
        m0 = np.exp(-((x - 0.5) ** 2) / 0.1)
        m0 = m0 / (np.sum(m0) * dx)  # Normalize

        initial_mass = np.sum(m0) * dx

        # Neumann BC for mass conservation
        neumann = LinearConstraint(weights={0: 1.0}, bias=0.0)

        A, b_bc = _build_diffusion_matrix_with_bc(
            shape=shape,
            spacing=spacing,
            D=D,
            dt=dt,
            ndim=ndim,
            bc_constraint_min=neumann,
            bc_constraint_max=neumann,
        )

        # Evolve for multiple timesteps
        m = m0.copy()
        for _ in range(10):
            b_rhs = m / dt + b_bc
            m = sparse.linalg.spsolve(A, b_rhs)

        final_mass = np.sum(m) * dx

        np.testing.assert_allclose(
            final_mass,
            initial_mass,
            rtol=1e-10,
            err_msg="Mass should be conserved with Neumann BC",
        )


class TestStrictAdjointPerPointSigma:
    """Issue #1183: the strict-adjoint FP step honors a per-point sigma (conservative
    variable-coefficient diffusion) instead of collapsing it to the mean."""

    @staticmethod
    def _solver(n):
        grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[n], boundary_conditions=no_flux_bc(dimension=1))
        H = SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=1.0),
            coupling=lambda m: 0.0 * np.asarray(m),
            coupling_dm=lambda m: 0.0 * np.asarray(m),
        )
        comps = MFGComponents(
            m_initial=lambda x: np.exp(-((np.asarray(x) - 0.25) ** 2) / 0.01),
            u_terminal=lambda x: 0.0 * np.asarray(x),
            hamiltonian=H,
        )
        prob = MFGProblem(geometry=grid, T=0.3, Nt=30, sigma=0.1, components=comps)
        return FPFDMSolver(prob)

    def test_array_sigma_conserves_and_per_point(self):
        """Pure-diffusion strict-adjoint step (zero advection matrix) with a non-uniform sigma:
        mass conserved, density non-negative, and the low-sigma region retains a higher peak than
        the mean-collapse would (the per-point fidelity the issue asks for)."""
        n = 41
        x = np.linspace(0.0, 1.0, n)
        dx = x[1] - x[0]
        solver = self._solver(n)
        a_t_zero = sparse.csr_matrix((n, n))  # no advection -> isolate diffusion
        w = _w(n, dx)
        m0 = np.exp(-((x - 0.25) ** 2) / 0.01)
        m0 /= float(w @ m0)
        sigma_field = np.where(x < 0.5, 0.05, 0.30)  # bump sits in the low-sigma region

        m_pp = m0.copy()
        for _ in range(30):
            m_pp = solver.solve_fp_step_adjoint_mode(m_pp, a_t_zero, sigma=sigma_field)
        m_mc = m0.copy()
        for _ in range(30):
            m_mc = solver.solve_fp_step_adjoint_mode(m_mc, a_t_zero, sigma=float(np.mean(sigma_field)))

        assert abs(float(w @ m_pp) - 1.0) < 1e-9, f"per-point strict-adjoint leaked mass: {float(w @ m_pp):.8f}"
        assert np.all(m_pp >= -1e-12), "per-point strict-adjoint produced a negative density"
        assert m_pp.max() > m_mc.max() * 1.02, (
            f"per-point did not under-diffuse the low-sigma bump: {m_pp.max():.4f} vs mean {m_mc.max():.4f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
