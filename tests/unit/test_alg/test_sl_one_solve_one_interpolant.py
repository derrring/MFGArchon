"""One solve must not mix interpolants (#1811, #1664, #1809).

`interpolation_method="cubic"` names four different interpolants across the SL family, and
which one runs is decided **per timestep**, not per configuration: `_compute_cfl_and_substeps`
routes a CFL<=1 timestep to the batch path and a CFL>1 timestep to the pointwise path, and
those two paths build different cubics. Measured on `main` before the consolidation, one solve
at `Nx=41, Nt=8`, terminal `0.05*exp(-300(x-0.5)^2)`:

    CubicSpline(not-a-knot) constructed : 7
    PchipInterpolator constructed       : 81

The two disagree by ~1.9e-2 on coarse grids (PCHIP 8.96 against not-a-knot 7.68 on
`U = [0, 10, 0, 10, 0]` at `x = 0.30`), and they differ in the property that matters:
`CubicSpline` is non-monotone and is what #583/#1033 replaced with PCHIP to stop the
Towel-on-Beach blow-up. Three of the four sites took that fix; the default path did not.

This is the acceptance criterion for giving the SL family one interpolation owner. It is not a
convergence check and not an accuracy bound -- it asserts that a single solve commits to a
single backend, which is a property no tolerance can express.
"""

from __future__ import annotations

import warnings

import pytest

import numpy as np
import scipy.interpolate as scipy_interpolate

import mfgarchon.alg.numerical.hjb_solvers.hjb_semi_lagrangian as solver_mod
from mfgarchon import MFGProblem
from mfgarchon.alg.numerical.hjb_solvers import HJBSemiLagrangianSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

# Interpolant constructors any SL path could reach. Counting CONSTRUCTIONS rather than
# sabotaging a body: the call sites in this family swallow exceptions, so a sabotaged path
# reports as passing (recorded in #1664).
_BACKENDS = ("CubicSpline", "PchipInterpolator", "RegularGridInterpolator", "RBFInterpolator")


@pytest.fixture
def backend_counter(monkeypatch):
    """Count every backend construction across scipy and the modules that import it by name."""
    import mfgarchon.alg.numerical.hjb_solvers.hjb_semi_lagrangian as solver_mod
    import mfgarchon.alg.numerical.hjb_solvers.hjb_sl_interpolation as interp_mod

    counts: dict[str, int] = dict.fromkeys(_BACKENDS, 0)

    for name in _BACKENDS:
        original = getattr(scipy_interpolate, name)

        def make(_name=name, _original=original):
            def counting(*args, **kwargs):
                counts[_name] += 1
                return _original(*args, **kwargs)

            return counting

        counting_fn = make()
        monkeypatch.setattr(scipy_interpolate, name, counting_fn, raising=False)
        for module in (solver_mod, interp_mod):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, counting_fn, raising=False)

    return counts


def _steep_1d_problem(nx: int = 41, nt: int = 8) -> MFGProblem:
    """Steep terminal data, so some timesteps exceed CFL and substep and some do not.

    That spread is the point: it is what routes different timesteps of ONE solve down the
    batch and pointwise paths.
    """
    return MFGProblem(
        geometry=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[nx], boundary_conditions=no_flux_bc(dimension=1)),
        T=0.4,
        Nt=nt,
        sigma=0.2,
        components=MFGComponents(
            m_initial=lambda x: 1.0,
            u_terminal=lambda x: 0.05 * np.exp(-300 * (np.asarray(x) - 0.5) ** 2),
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0),
                coupling=lambda m: m,
                coupling_dm=lambda m: 1.0,
            ),
        ),
    )


def _solver_for(dimension: int, method: str, **solver_kwargs) -> HJBSemiLagrangianSolver:
    """A solver at the given dimension, on a grid large enough for every honoured method."""
    problem = MFGProblem(
        geometry=TensorProductGrid(
            bounds=[(0.0, 1.0)] * dimension,
            Nx_points=[11] * dimension,
            boundary_conditions=no_flux_bc(dimension=dimension),
        ),
        T=0.4,
        Nt=4,
        sigma=0.2,
        components=MFGComponents(
            m_initial=lambda x: 1.0,
            u_terminal=lambda x: 0.0,
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0),
                coupling=lambda m: m,
                coupling_dm=lambda m: 1.0,
            ),
        ),
    )
    return HJBSemiLagrangianSolver(problem, interpolation_method=method, **solver_kwargs)


def _solve(problem: MFGProblem, **solver_kwargs) -> None:
    solver = HJBSemiLagrangianSolver(problem, **solver_kwargs)
    nx = problem.geometry.get_grid_shape()[0]
    m = np.ones((problem.Nt + 1, nx))
    u = np.zeros((problem.Nt + 1, nx))
    u[-1] = problem.get_u_terminal()
    solver.solve_hjb_system(m, u[-1], u)


def test_a_cubic_solve_commits_to_one_cubic_backend(backend_counter):
    """The headline. On main this fails with 7 CubicSpline and 81 PCHIP in one solve."""
    _solve(_steep_1d_problem(), interpolation_method="cubic", characteristic_solver="explicit_euler")

    cubic_backends = {
        name: n for name, n in backend_counter.items() if name in ("CubicSpline", "PchipInterpolator") and n
    }
    assert len(cubic_backends) <= 1, (
        f"one solve used more than one cubic backend: {cubic_backends}. Which one runs is decided "
        "per timestep by the CFL number rather than by configuration, and the two differ in "
        "monotonicity -- the property #583/#1033 replaced CubicSpline for."
    )


def test_the_cubic_backend_is_the_monotone_one(backend_counter):
    """Not merely consistent -- consistent with the #583 fix.

    Three of the four sites already use PCHIP and say why. Consolidating onto the
    non-monotone `CubicSpline` would satisfy the test above and reintroduce the blow-up.
    """
    _solve(_steep_1d_problem(), interpolation_method="cubic", characteristic_solver="explicit_euler")

    assert backend_counter["CubicSpline"] == 0, (
        "the solve constructed CubicSpline(not-a-knot), which is non-monotone and is what "
        "Issue #583 replaced with PchipInterpolator to stop the Issue #1033 blow-up"
    )
    assert backend_counter["PchipInterpolator"] > 0, "no cubic interpolant was built at all"


def test_substepping_does_not_change_which_backend_runs(backend_counter):
    """The mechanism, isolated: with substepping off, only the batch path runs.

    If the two paths agree on the backend, turning substepping off cannot change which one is
    built. On main it does, which is the defect stated as a difference rather than a count.
    """
    _solve(
        _steep_1d_problem(),
        interpolation_method="cubic",
        characteristic_solver="explicit_euler",
        enable_adaptive_substepping=False,
    )
    without = {k: v for k, v in backend_counter.items() if v}

    for key in backend_counter:
        backend_counter[key] = 0
    _solve(
        _steep_1d_problem(),
        interpolation_method="cubic",
        characteristic_solver="explicit_euler",
        enable_adaptive_substepping=True,
    )
    with_sub = {k: v for k, v in backend_counter.items() if v}

    assert set(without) == set(with_sub), (
        f"substepping changed the interpolant backend: without={without}, with={with_sub}. "
        "The CFL number is selecting the numerical method."
    )


class TestTheFoldOvershootsAndTheInterpolantMustSurviveIt:
    """`[0, 1]` is the one domain where this cannot fire, and every fixture above uses it.

    `reflect_into_domain` computes ``xmin + span - |((x - xmin) mod 2*span) - span|``, which
    rounds OUTWARD when the endpoints are not exactly representable:

        bounds (0.0, 1.0)   reflect(xmin) = 0.0                  exact
        bounds (-0.3, 1.7)  reflect(xmin) = -0.30000000000000004  2.8e-17 BELOW xmin
        bounds (0.1, 0.9)   reflect(xmin) = 0.09999999999999998   below xmin

    Both interpolants the batch path used to build extrapolated, so that overshoot cost ~1e-17
    of error. `PchipInterpolator(extrapolate=False)` returns NaN instead, and `solve_banded`
    rejects it -- so consolidating onto PCHIP turned a rounding artefact into an aborted solve.
    Caught in review; the acceptance tests above are all on `[0, 1]` and are structurally
    incapable of seeing it.
    """

    @pytest.mark.parametrize("bounds", [(-0.3, 1.7), (0.1, 0.9), (0.0, 1.0)], ids=str)
    @pytest.mark.parametrize("method", ["cubic", "linear"])
    def test_a_solve_survives_a_non_representable_domain(self, bounds, method):
        problem = MFGProblem(
            geometry=TensorProductGrid(bounds=[bounds], Nx_points=[41], boundary_conditions=no_flux_bc(dimension=1)),
            T=0.4,
            Nt=8,
            sigma=0.2,
            components=MFGComponents(
                m_initial=lambda x: 1.0,
                u_terminal=lambda x: 0.05 * np.exp(-300 * (np.asarray(x) - 0.5) ** 2),
                hamiltonian=SeparableHamiltonian(
                    control_cost=QuadraticControlCost(control_cost=1.0),
                    coupling=lambda m: m,
                    coupling_dm=lambda m: 1.0,
                ),
            ),
        )
        solver = HJBSemiLagrangianSolver(problem, interpolation_method=method, characteristic_solver="explicit_euler")
        u = np.zeros((9, 41))
        u[-1] = problem.get_u_terminal()
        result = solver.solve_hjb_system(np.ones((9, 41)), u[-1], u)
        assert np.all(np.isfinite(result)), (
            f"non-finite value function on bounds={bounds}, method={method}: the reflect fold "
            "overshoots by ~1 ULP and the interpolant returned NaN rather than clamping"
        )

    def test_the_fold_really_does_overshoot(self):
        """Pin the mechanism, not just the symptom -- otherwise a later 'fix' to the interpolant
        can hide it while the fold still returns points outside the domain."""
        from mfgarchon.alg.numerical.hjb_solvers.hjb_sl_characteristics import reflect_into_domain

        lo, hi = -0.3, 1.7
        folded = reflect_into_domain(np.array([[lo]]), np.array([lo]), np.array([hi]))[0, 0]
        assert folded < lo, "fixture is stale: this domain no longer exercises the outward rounding"


@pytest.mark.parametrize("method", ["linear", "cubic"])
def test_the_stochastic_path_consults_the_owner(method, monkeypatch):
    """Coverage gap this file had: every case above uses explicit_euler + adi.

    The stochastic path has its own dispatch line, and when it was first routed through the
    owner the import was function-local, so that line raised NameError at runtime while all
    three tests above stayed green. Lint caught it; this test is what should have.

    The assertion is a POSITIVE CONTROL, and the first attempt at it was not. Counting backend
    constructions reported `{}` here and looked like a pass, because 1D+linear takes `np.interp`
    -- which is not a counted backend. An empty count had two causes (the line did not run, or
    the measurement could not see it) and could not separate them. Counting consultations of
    `sl_backend` itself distinguishes them: 0 means the line never ran.
    """
    import mfgarchon.alg.numerical.hjb_solvers.hjb_semi_lagrangian as solver_mod

    # Construct BEFORE patching. The constructor's disclosure also consults the owner, and with
    # `monotone_required=False` among its calls -- so recording construction too made an `all(...)`
    # assertion here false, and the version of this test that passed did so only because
    # `_disclose_monotone_override` re-imported `sl_backend` function-locally and thereby shadowed
    # the patch. Deleting that redundant import, a pure no-op, turned 18 passed into 2 failed: the
    # test was pinning an import style rather than a dispatch property. Patching after construction
    # isolates the solve-time dispatch, which is the thing under test.
    problem = _steep_1d_problem(nx=21, nt=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # the cubic case warns by design
        solver = HJBSemiLagrangianSolver(problem, interpolation_method=method, diffusion_method="stochastic")

    calls: list[tuple] = []
    original = solver_mod.sl_backend

    def recording(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(solver_mod, "sl_backend", recording)

    nx = problem.geometry.get_grid_shape()[0]
    u = np.zeros((problem.Nt + 1, nx))
    u[-1] = problem.get_u_terminal()
    solver.solve_hjb_system(np.ones((problem.Nt + 1, nx)), u[-1], u)

    assert calls, "the stochastic dispatch never consulted sl_backend -- this test cannot see a NameError there"
    assert all(kw.get("monotone_required") is True for _, kw in calls), (
        f"the stochastic path must demand a monotone interpolant (Carlini-Silva 2014): {calls}"
    )


class TestTheOverrideIsDisclosedRatherThanApplied:
    """A monotone scheme silently replacing the caller's interpolant is the #1810 hole.

    The disclosure previously named a hardcoded `("cubic", "quintic")` pair -- a third
    restatement of the monotone policy, which missed `nearest`/`slinear`. Those are honoured at
    nD, are equally non-monotone, and were remapped to linear with no warning. Measured spread
    on one 7x7 profile: nearest 0.0688 against linear 0.2732.
    """

    def test_the_owner_reports_what_the_monotone_scheme_runs(self):
        from mfgarchon.alg.numerical.hjb_solvers.hjb_sl_interpolation import sl_backend

        forced = {m: sl_backend(m, 2, monotone_required=True) for m in ("nearest", "slinear", "quintic", "linear")}
        assert forced == {"nearest": "linear", "slinear": "linear", "quintic": "pchip", "linear": "linear"}

    def test_one_d_cubic_is_pchip_with_or_without_the_monotone_flag(self):
        """The case that broke the first version of the disclosure.

        Keyed on "does the monotone flag change anything", 1D cubic answers no -- PCHIP runs
        either way -- and the warning vanished, silently dropping a disclosure the code had
        given correctly before the consolidation. The predicate that survives is "is what runs
        the Q1 interpolant the proof covers".
        """
        from mfgarchon.alg.numerical.hjb_solvers.hjb_sl_interpolation import sl_backend

        assert sl_backend("cubic", 1, monotone_required=True) == "pchip"
        assert sl_backend("cubic", 1, monotone_required=False) == "pchip"

    @pytest.mark.parametrize(
        ("method", "expected"),
        [("cubic", "PchipInterpolator"), ("nearest", "is not the interpolant you selected")],
        ids=["cubic-is-replaced-by-pchip", "nearest-is-replaced-by-linear"],
    )
    def test_a_downgraded_method_warns(self, method, expected):
        """`nearest` is the case that used to pass in silence."""
        dimension = 1 if method == "cubic" else 2
        with pytest.warns(UserWarning, match="Carlini-Silva 2014") as record:
            _solver_for(dimension, method, diffusion_method="stochastic")
        assert any(expected in str(w.message) for w in record), (
            f"the warning must say what actually runs instead: {[str(w.message) for w in record]}"
        )

    def test_a_monotone_method_does_not_warn(self):
        """Negative control. A disclosure that fires unconditionally passes every test above."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            _solver_for(1, "linear", diffusion_method="stochastic")

    def test_a_non_monotone_scheme_does_not_warn(self):
        """Second negative control: the override only happens under the monotone scheme."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            _solver_for(1, "cubic", diffusion_method="explicit_euler")


class TestTheLinearPathIsUNCHANGEDByTheConsolidation:
    """The other half of this change's stated invariant, and the half nothing was pinning.

    The claim is "only the backend choice changes": both interpolants the batch path built
    extrapolated out-of-domain feet, and it still does. The cubic half is pinned by
    `test_a_solve_survives_a_non_representable_domain[cubic-(-0.3, 1.7)]`. The linear half was
    pinned by nothing -- and during this change the batch site's
    `interp1d(fill_value="extrapolate")` was silently swapped for `np.interp`, which CLAMPS.
    Reapplying that one-line swap passes 613 SL/HJB tests while moving a periodic linear solve by
    5.26e-3, eleven times the 4.77e-4 the intended cubic backend swap moves it. The `[linear-*]`
    fold cases assert only finiteness, and a clamped value is finite.

    Periodic BC is what makes this reachable: its fold is dead (#1739), so feet leave the domain
    at every step and the out-of-bounds policy is what decides the answer.

    The values below are captured from `main` BEFORE the consolidation. That is deliberate -- an
    agreement test between the two paths goes tautological the moment they share an owner, which
    is exactly when this change lands, so the only pin that survives is against the
    pre-consolidation output. Update it only alongside a stated, measured intent to change the
    linear path.
    """

    # Captured on main @ fecbfa1d, before sl_backend existed.
    MAIN_U0_FIRST5 = (
        -0.40983509012196784,
        -0.404058809261078,
        -0.3982725404248194,
        -0.3923195952518894,
        -0.3860645384176972,
    )
    MAIN_SUM = -75.51161537275942

    def _periodic_linear_solve(self) -> np.ndarray:
        from mfgarchon.geometry.boundary import periodic_bc

        problem = MFGProblem(
            geometry=TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[41], boundary_conditions=periodic_bc(dimension=1)
            ),
            T=0.4,
            Nt=8,
            sigma=0.05,
            components=MFGComponents(
                m_initial=lambda x: 1.0,
                u_terminal=lambda x: 0.05 * np.sin(2 * np.pi * np.asarray(x)),
                hamiltonian=SeparableHamiltonian(
                    control_cost=QuadraticControlCost(control_cost=1.0),
                    coupling=lambda m: m,
                    coupling_dm=lambda m: 1.0,
                ),
            ),
        )
        solver = HJBSemiLagrangianSolver(problem, interpolation_method="linear", characteristic_solver="explicit_euler")
        u = np.zeros((9, 41))
        u[-1] = problem.get_u_terminal()
        return solver.solve_hjb_system(np.ones((9, 41)), u[-1], u)

    # Relative, not bit-exact. Bit-exactness made this a scipy-internals pin wearing a policy
    # pin's name: on scipy 1.13.1 -- which pyproject declares supported -- `interp1d`'s
    # `_call_linear` differs by 1 ULP on 16 of 42 query points and the pin failed while claiming
    # the boundary policy had changed. The margin is not close: M8, the mutation this exists for,
    # moves these values by rel 3.6e-2; the scipy-version noise is rel 5.4e-16. Any tolerance in
    # [1e-12, 1e-4] keeps ~10 orders of discrimination and drops the library coupling.
    RTOL = 1e-9

    def test_a_periodic_linear_solve_matches_pre_consolidation(self):
        result = self._periodic_linear_solve()
        assert tuple(float(v) for v in result[0][:5]) == pytest.approx(self.MAIN_U0_FIRST5, rel=self.RTOL), (
            "the linear batch path moved. This change is supposed to alter only which CUBIC "
            "backend runs; a moved linear result means the out-of-bounds policy moved with it "
            "(clamp vs extrapolate), which is the boundary policy #1739 owns and this PR defers."
        )
        assert float(result.sum()) == pytest.approx(self.MAIN_SUM, rel=self.RTOL)

    def test_the_fixture_actually_sends_feet_out_of_the_domain(self, monkeypatch):
        """Positive control: MEASURE the feet, do not infer them from a dispatch string.

        The first version asserted `bc_type_to_geometric_operation("periodic") != "wrap"`, which
        is a claim about the producer. #1739 is a defect in the CONSUMER -- three HJB sites compare
        `bc_op == "wrap"` against a function that returns `"reflect" | "clamp" | "periodic"`, and
        the FP adjoint already compares against `"periodic"`, so the available fix is to change
        those three comparisons, not the producer. Constructed, that fix takes out-of-domain feet
        from 8/328 to 0/328 -- the fixture stops exercising the policy, which is exactly what this
        control exists to catch -- and the string assertion still PASSED.

        Counting the feet cannot go inert that way: it fails when they stop leaving, whatever the
        reason. That matters because the pin above is relative, and on a wrapped fixture M8 moves
        the values by only ~1 ULP, so the pin would be blind and this control is what says so.
        """
        import scipy.interpolate as scipy_interpolate

        # Counted through BOTH backends, so this measures where the feet are rather than which
        # interpolant is built. Counting only interp1d made the control fail under a mutation that
        # merely switched to np.interp -- a failure about the implementation, not the fixture.
        outside = []

        def _tally(x, q):
            lo, hi = float(np.min(x)), float(np.max(x))
            q_arr = np.atleast_1d(q)
            outside.append(int(np.sum((q_arr < lo) | (q_arr > hi))))

        original_interp1d = scipy_interpolate.interp1d

        def counting_interp1d(x, y, **kwargs):
            f = original_interp1d(x, y, **kwargs)

            def wrapped(q):
                _tally(x, q)
                return f(q)

            return wrapped

        original_np_interp = np.interp

        def counting_np_interp(q, xp, fp, **kwargs):
            _tally(xp, q)
            return original_np_interp(q, xp, fp, **kwargs)

        monkeypatch.setattr(scipy_interpolate, "interp1d", counting_interp1d)
        monkeypatch.setattr(solver_mod, "interp1d", counting_interp1d, raising=False)
        monkeypatch.setattr(solver_mod.np, "interp", counting_np_interp)
        self._periodic_linear_solve()

        assert sum(outside) > 0, (
            "no departure foot left the domain, so the pin above measures nothing about the "
            "out-of-bounds policy. Most likely #1739 was fixed and the periodic fold now wraps: "
            "re-capture the reference on a fixture that still exercises extrapolation, rather "
            "than keeping a pin that has quietly stopped testing its subject."
        )
