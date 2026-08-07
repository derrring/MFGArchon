"""A periodic FP solve must reproduce the torus the grid describes. Issue #1822.

The seam invariant in ``test_periodic_capability_invariant_1822.py`` asks whether ``m[0]`` and
``m[-1]`` agree. That is necessary and it is not sufficient: under a datum symmetric about the
midpoint, ``FPFDMSolver`` and ``FPFVMSolver`` both returned a seam of **2e-15** while the mode's
amplitude came out **8.7% off** the analytic value -- a decay rate 9.4% low. They wrapped cell
``N-1`` to cell ``0`` on an endpoint-inclusive grid, which solves the problem on a torus one cell too long, and symmetry keeps the duplicated pair
equal all the way to the end. A seam test cannot see that, so this file measures against laws
computed independently of any discretisation:

- **the heat kernel** -- with zero drift the FP equation is ``dm/dt = D m_xx``, ``D = sigma^2/2``,
  and the mode ``cos(2 pi x)`` decays by exactly ``exp(-D (2 pi)^2 T)`` on the unit torus;
- **rigid translation** -- with zero diffusion and a constant velocity ``v``, ``m(T, .)`` is
  ``m0(. - vT)``, wrapped.

Both are stated by the PDE, so neither can go tautological when the two solvers are later
consolidated onto shared machinery -- which is the failure mode that kills path-A-vs-path-B
agreement tests once the consolidation they were written for succeeds.

Grid, and why only this one is measured. ``TensorProductGrid(bounds=[(0, 1)], Nx_points=[N])``
always builds ``linspace(0, 1, N)``: endpoint-INCLUSIVE, ``dx = 1/(N-1)``, ``N-1`` distinct cells,
torus length exactly 1. Asking for an exclusive layout by passing ``x[-1] = 0.95`` does NOT produce
one -- it produces an inclusive grid over ``[0, 0.95]``, whose 21-cell torus comes out at 0.9975
and matches the oracle by accident. That accident is how the first version of this measurement
certified the unfixed solvers; the exclusive layout belongs to the operator layer, which builds its
own coordinates and is covered in ``test_operators/test_laplacian.py``.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
from mfgarchon.alg.numerical.fp_solvers.fp_fvm import FPFVMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import periodic_bc

T_FINAL = 0.5
NT = 200  # fine in time on purpose: what is under test is the SPACE wrap, not the time error
AMP = 0.3
SIGMA = 0.3
D = SIGMA**2 / 2

SOLVERS = {"FPFDMSolver": FPFDMSolver, "FPFVMSolver": FPFVMSolver}


def _datum(z, k=1.0):
    """Exactly 1-periodic and strictly positive, so it is a legal density on the unit torus."""
    return 1.0 + AMP * np.cos(2 * np.pi * k * np.asarray(z))


def _problem(nx, sigma, k=1.0):
    return MFGProblem(
        geometry=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[nx], boundary_conditions=periodic_bc(dimension=1)),
        T=T_FINAL,
        Nt=NT,
        sigma=sigma,
        components=MFGComponents(
            m_initial=lambda z: _datum(z, k),
            u_terminal=lambda z: np.zeros_like(np.asarray(z)),
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0),
                coupling=lambda m: m,
                coupling_dm=lambda m: 1.0,
            ),
        ),
    )


def _diffuse(name, nx, sigma=SIGMA, k=1.0, scheme=None):
    """Zero drift: the optimal control vanishes and FP reduces to the heat equation."""
    kwargs = {"advection_scheme": scheme} if scheme else {}
    solver = SOLVERS[name](_problem(nx, sigma, k), **kwargs)
    x = np.linspace(0.0, 1.0, nx)
    return np.asarray(solver.solve_fp_system(_datum(x, k), np.zeros((NT + 1, nx)))), x


def _mode_amplitude(m, x, k=1.0):
    """Projection onto ``cos(2 pi k x)``, with the repeated endpoint weighted once.

    ``x[-1]`` IS ``x[0]``; counting it twice biases ``<phi, phi>`` by 1/(N-1) and would show up as
    a solver error of the same size.
    """
    phi = np.cos(2 * np.pi * k * x)
    w = np.ones_like(x)
    w[-1] = 0.0
    return float(np.sum(w * m * phi) / np.sum(w * phi * phi))


@pytest.mark.parametrize("name", sorted(SOLVERS))
def test_a_periodic_solve_reproduces_the_heat_kernel(name):
    """The decay rate of a Fourier mode is fixed by the PDE, and by the torus length.

    Wrapping N-1 to 0 on this grid stretches the torus from 1 to 1 + dx, which slows the decay --
    measured 8.7e-02 of relative error at Nx=21 before the fix, where the number below is 9.3e-03.
    The failure is silent: nothing raises, the density stays positive, the seam stays at round-off.
    """
    oracle = AMP * np.exp(-D * (2 * np.pi) ** 2 * T_FINAL)
    errors = []
    for nx in (21, 41, 81):
        m, x = _diffuse(name, nx)
        assert np.isfinite(m).all(), f"{name} produced non-finite values"
        errors.append(abs(_mode_amplitude(m[-1], x) - oracle) / oracle)

    trend = ", ".join(f"{e:.3e}" for e in errors)
    assert errors[0] < 3e-2, (
        f"{name} decayed cos(2 pi x) at the wrong rate on the unit torus: relative error "
        f"{errors[0]:.3e} at Nx=21 ({trend} at 21/41/81). A wrap that treats the repeated endpoint "
        f"as its own cell puts the solve on a torus of length 1 + dx and lands near 8.7e-02 here"
    )
    assert errors[2] < errors[0], f"{name} heat-kernel error did not improve under refinement: {trend}"


@pytest.mark.parametrize("name", sorted(SOLVERS))
def test_zero_diffusion_and_zero_drift_do_not_move_the_density(name):
    """Positive control for the test above: with nothing driving it, the datum must be untouched.

    Without this, `test_a_periodic_solve_reproduces_the_heat_kernel` has two ways to pass -- the
    scheme is right, or the solve did nothing and the amplitude never changed. This separates them:
    here doing nothing is the correct answer and any evolution is a defect.
    """
    m, x = _diffuse(name, 21, sigma=0.0)
    assert _mode_amplitude(m[-1], x) == pytest.approx(AMP, abs=1e-12)
    assert np.max(np.abs(m[-1] - m[0])) < 1e-12, f"{name} moved the density with no diffusion and no drift"


@pytest.mark.parametrize("name", sorted(SOLVERS))
def test_the_heat_kernel_oracle_can_fail(name):
    """Negative control: a datum that is NOT 1-periodic must miss the unit-torus oracle badly.

    An oracle nothing can fail measures nothing. ``k = 1.5`` is discontinuous across the seam, so
    the solve cannot reproduce a clean modal decay, and the assertion in the first test would fire.
    """
    m, x = _diffuse(name, 21, k=1.5)
    oracle = AMP * np.exp(-D * (2 * np.pi * 1.5) ** 2 * T_FINAL)
    error = abs(_mode_amplitude(m[-1], x, 1.5) - oracle) / oracle
    assert error > 0.5, (
        f"{name}: a datum with a jump at the seam matched the smooth-torus oracle to {error:.3e}. "
        f"The oracle cannot distinguish anything if this passes"
    )


@pytest.mark.parametrize("name", sorted(SOLVERS))
def test_a_periodic_solve_translates_rigidly(name):
    """The advection half, which a zero-drift oracle cannot reach at all.

    The wrap face is only exercised when mass actually crosses it. The velocity is handed over
    directly rather than derived from a potential: a constant drift cannot come from a periodic U
    (``U = -v x`` jumps at the seam), so deriving it would measure the fixture's own artefact.
    """
    velocity = 0.4
    errors = []
    for nx in (41, 81):
        solver = SOLVERS[name](_problem(nx, sigma=1e-8))
        x = np.linspace(0.0, 1.0, nx)
        m = np.asarray(solver.solve_fp_system(_datum(x), drift_field=np.full((NT + 1, nx), velocity)))
        exact = _datum(x - velocity * T_FINAL)  # the datum is 1-periodic, so no wrapping is needed
        errors.append(float(np.max(np.abs(m[-1] - exact))))

    assert errors[0] < 5e-2, (
        f"{name} did not translate rigidly on the torus: max error {errors[0]:.3e} at Nx=41. "
        f"A wrap face at the wrong cell shows up here as the profile arriving displaced"
    )
    assert errors[1] < errors[0], f"{name} translation error did not improve under refinement: {errors}"


@pytest.mark.parametrize("scheme", ["divergence_upwind", "divergence_centered", "gradient_upwind", "gradient_centered"])
def test_every_fdm_advection_scheme_wraps_on_the_same_torus(scheme):
    """All four schemes carried their own copy of the wrap, so all four are measured.

    They now route through one owner, which is exactly why this cannot be one test on the default:
    a copy left behind in a non-default scheme is invisible to a caller who never selects it, and
    ``FPFDMSolver`` reaches all four through ``advection_scheme``.

    Note what this does NOT show. Under zero drift the four reduce to the same diffusion operator,
    so agreement here says they were fixed together, not that each wrap is separately exercised;
    the drift case above is what puts mass across the wrap face, and it runs on the default.
    """
    oracle = AMP * np.exp(-D * (2 * np.pi) ** 2 * T_FINAL)
    m, x = _diffuse("FPFDMSolver", 41, scheme=scheme)
    error = abs(_mode_amplitude(m[-1], x) - oracle) / oracle
    assert error < 2e-2, f"advection_scheme={scheme!r} decayed the mode at the wrong rate: {error:.3e}"


@pytest.mark.parametrize("name", sorted(SOLVERS))
def test_every_channel_that_supplies_a_bc_gets_the_same_torus(name):
    """The answer must not depend on WHICH object handed the solver its boundary condition.

    Both solvers resolve the BC themselves, shadowing ``BaseMFGSolver.boundary_conditions`` -- the
    property that binds the grid's node layout onto a periodic BC. Fixing only the geometry channel
    left a caller-supplied BC wrapping the historical way, so one problem and one library gave
    8.7e-02 or 9.3e-03 of heat-kernel error depending on the channel. Both were wrong before this
    issue, so the divergence did not exist until the fix; a per-channel pin is what keeps it closed.
    """
    oracle = AMP * np.exp(-D * (2 * np.pi) ** 2 * T_FINAL)
    x = np.linspace(0.0, 1.0, 21)
    errors = {}
    for channel in ("geometry", "constructor", "components"):
        problem = _problem(21, SIGMA)
        kwargs = {}
        if channel == "constructor":
            kwargs["boundary_conditions"] = periodic_bc(dimension=1)
        elif channel == "components":
            # The third channel each solver's own resolution consults, ahead of the geometry.
            problem.components.boundary_conditions = periodic_bc(dimension=1)
        solver = SOLVERS[name](problem, **kwargs)
        m = np.asarray(solver.solve_fp_system(_datum(x), np.zeros((NT + 1, 21))))
        errors[channel] = abs(_mode_amplitude(m[-1], x) - oracle) / oracle

    assert errors["constructor"] == pytest.approx(errors["geometry"], rel=1e-12), (
        f"{name} gave different answers depending on which channel supplied the periodic BC: "
        f"{errors}. A BC that never met the grid must still be completed with the grid's layout"
    )
    assert errors["components"] == pytest.approx(errors["geometry"], rel=1e-12), (
        f"{name} disagreed between the components and geometry channels: {errors}"
    )
    assert errors["geometry"] < 3e-2, f"{name} missed the heat kernel on every channel: {errors}"


def test_a_drift_that_disagrees_with_itself_at_the_seam_still_gives_one_density_there():
    """The repeated node must be single-valued even when the field driving it is not.

    This is the case the constraint row exists for, and it took a mutation to find: with a
    PERIODIC drift, the stencil rows for node 0 and node N-1 both reach neighbours {N-2, 1} with
    equal coefficients, so they are the same equation and ``m[N-1] == m[0]`` falls out whether or
    not anything enforces it. Every other test here has a periodic drift, so all of them passed
    with the constraint deleted. Under a drift that disagrees across the seam the two rows differ
    and the pair splits: measured 2.36e-01.

    Not a synthetic input. During Picard the drift comes from an HJB solve, and HJBFDMSolver's own
    periodic seam is 7.4e-01 (#1834) -- so an FP solver is routinely handed exactly this.
    """
    nx, nt = 21, 20
    x = np.linspace(0.0, 1.0, nx)
    solver = FPFDMSolver(_problem(nx, sigma=SIGMA))
    drift = np.tile(0.4 * np.sin(2 * np.pi * x), (nt + 1, 1))
    drift[:, -1] = -0.9  # node N-1 is node 0; this says otherwise, and the solver must not believe it

    m = np.asarray(solver.solve_fp_system(_datum(x), drift_field=drift))
    seam = abs(m[-1][0] - m[-1][-1])
    assert seam < 1e-12, (
        f"a drift with a jump at the repeated node produced two different densities at one physical "
        f"point: seam {seam:.4e}. x[0] and x[-1] are the same place regardless of what the drift says"
    )


def test_fdm_and_fvm_converge_to_the_same_field_under_a_varying_periodic_drift():
    """Two independently written schemes on the same torus must agree in the limit.

    A VARYING drift on purpose: every other FVM drift in the suite is a constant, and a constant
    cannot distinguish the wrap face from its neighbour. Asserted as CONVERGENCE, not as a
    tolerance -- FDM is first-order upwind and FVM is minmod-limited second order, so at any fixed
    grid they differ by O(h) and an absolute bound here would be a number chosen to pass. Measured
    2.80e-01, 1.59e-01, 8.84e-02, 5.10e-02 over Nx = 41/81/161/321: ratios ~1.75, converging.
    """
    gaps = []
    for nx in (41, 81, 161):
        x = np.linspace(0.0, 1.0, nx)
        drift = np.tile(0.5 * np.sin(2 * np.pi * x) + 0.2, (NT + 1, 1))  # varying, exactly periodic
        fields = {}
        for name in ("FPFDMSolver", "FPFVMSolver"):
            solver = SOLVERS[name](_problem(nx, sigma=0.15))
            fields[name] = np.asarray(solver.solve_fp_system(_datum(x), drift_field=drift))[-1]
            assert abs(fields[name][0] - fields[name][-1]) < 1e-9, f"{name} left a seam under a varying periodic drift"
        gaps.append(float(np.max(np.abs(fields["FPFDMSolver"] - fields["FPFVMSolver"]))))

    trend = ", ".join(f"{g:.3e}" for g in gaps)
    assert gaps[2] < gaps[0] / 2.0, (
        f"FDM and FVM did not converge to the same field on the periodic torus: {trend} at "
        f"Nx=41/81/161. Two schemes advecting across different faces disagree at O(1), not O(h)"
    )


def test_the_fvm_wrap_face_velocity_is_the_face_between_the_last_cell_and_the_first():
    """Pinned on the kernel, because end-to-end nothing can see it.

    On an inclusive axis the wrap face is `alpha_int[span-1]` -- the face between cell `span-1` and
    the node that IS cell 0. The caller's `alpha_wrap` is built from nodes `N-1` and `0`, which are
    one point, so it is the velocity AT that node rather than averaged across the face: an O(h)
    error at exactly one face.

    It survives every end-to-end check, and that is not an oversight to fix by widening them.
    Both choices are *consistent* -- each face velocity is shared by the two cells that touch it --
    so the flux still telescopes and mass is conserved either way (measured 2.2e-14 with the line
    and 2.2e-14 without), and the seam closes either way because the repeated cell copies cell 0.
    A direct call is the only thing that separates them.
    """
    from mfgarchon.alg.numerical.fp_solvers.fp_fvm_flux import axis_flux_divergence

    n_full, span = 6, 5
    m = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 1.0])  # cell 5 repeats cell 0
    alpha_int = np.array([0.1, 0.2, 0.3, 0.4, 0.9])  # the last entry IS the wrap face
    wrong_wrap = np.array(-7.0)  # what an exclusive-layout caller would supply; must be ignored

    div = axis_flux_divergence(m, alpha_int, 0, 1.0, "upwind", "periodic", wrong_wrap, span=span)

    # Upwind, all velocities positive: F_{i+1/2} = alpha_i * m_i, and the wrap face carries
    # alpha_int[span-1] * m[span-1] into cell 0.
    f_wrap = alpha_int[span - 1] * m[span - 1]
    assert div[0] == pytest.approx(alpha_int[0] * m[0] - f_wrap), (
        "cell 0's inflow must come across the face between cell span-1 and itself"
    )
    assert div[span - 1] == pytest.approx(f_wrap - alpha_int[span - 2] * m[span - 2])
    assert div[n_full - 1] == pytest.approx(div[0]), "the repeated cell must carry cell 0's divergence"
    assert float(np.sum(div[:span])) == pytest.approx(0.0, abs=1e-14), (
        "the flux must telescope over the distinct cells, or the scheme is not conservative"
    )
    """The last node IS the first one, so the assembled system must say so, not merely end up agreeing.

    Checked on the operator rather than on the output: a solver that solved for N cells and copied
    ``m[0]`` into ``m[-1]`` afterwards would show a clean seam while still having stepped the wrong
    torus -- which is the state this issue started in.
    """
    from mfgarchon.geometry.boundary.conditions import periodic_axis_span, repeated_endpoint_mirror

    bc = periodic_bc(dimension=1)
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=bc)
    bound = grid.boundary_conditions

    assert periodic_axis_span(bound, 21) == 20, "an inclusive periodic axis of 21 nodes has 20 distinct cells"
    assert repeated_endpoint_mirror(bound, (20,), (21,)) == (0,), "node 20 must be identified with node 0"
    assert repeated_endpoint_mirror(bound, (19,), (21,)) is None, "node 19 is its own cell"

    # And an unstated convention keeps the historical exclusive layout, so no caller that never
    # met a grid has its numbers moved by any of this.
    assert periodic_axis_span(periodic_bc(dimension=1), 21) == 21
