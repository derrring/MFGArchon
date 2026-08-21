"""Which solvers a manufactured source can actually reach (#2020, #1991).

An MMS convergence-order claim is only about the solvers a manufactured source **reaches**.
Both prior counts of that were signature audits, and a signature is a coarser predicate than
the hazard: ``HJBWENOSolver`` names ``source_term`` and, before #2029, still refused a 2-D
problem with ``source_term is threaded through the 1D path only``. A census keyed on "accepts
the kwarg" reports such a solver reachable when the source reaches one branch of it.

So the predicate here is behavioural and finite: **solve twice on one fixture, once with a zero
source and once with a strong one, and ask whether the answer moved.** Four outcomes, and the
third is the one worth having a test for:

``THREADS_IT``
    The answer moved. The source is genuinely applied.
``REFUSED``
    The solver is out of reach and says so. Not a defect -- an honest refusal.

    Two distinct mechanisms produce it, and this probe reaches only the first:

    - **The signature refuses.** Every solver measured REFUSED here raises a bare
      ``TypeError: got an unexpected keyword argument``. That is loud *by design*, not by
      accident: ``BaseFPSolver.__init_subclass__`` says so in as many words -- *"No
      ``**kwargs``: an unnamed parameter raises TypeError at the call site, which is loud"* --
      and the gate exists only to stop a ``**kwargs`` signature from swallowing the argument
      silently instead.
    - **The coupler refuses**, with guidance. On the documented route -- a problem that defines
      a source, passed through ``compose_fp_source`` -- ``mfg_residual.py`` raises
      ``NotImplementedError: ... does not accept 'source_term', but the problem defines an FP
      source term``, naming the remedy. This probe calls the solver directly, so it never gets
      there.

    Both are honest. Conflating them is not: an earlier draft of this file called the
    ``TypeError`` "the fail-loud guard working", which named the wrong guard.
``ACCEPTS_AND_IGNORES``
    Took the argument, returned a byte-identical answer. **A silent wrong result**: an MMS built
    on such a solver measures the order of the wrong equation and reports a clean number.
    ``test_no_solver_accepts_a_source_and_ignores_it`` exists for this class alone.
``INDISTINGUISHABLE_FROM_ITS_OWN_NOISE``
    The answer moved, but not by more than the solver moves when handed the SAME source twice.
    ``FPParticleSolver`` differs by 1.3e-01 on byte-identical inputs, so for a stochastic
    solver "the answer moved" is not evidence that the source moved it. The noise floor is
    measured per solver, from a repeat, before the comparison is read.
``NOT_PROBED``
    The fixture cannot construct it -- an unstructured mesh, a network problem, a mixed
    discretization. Says nothing about the solver, only about this fixture, and is listed by
    name in ``NOT_PROBED_EXPECTED`` because a population predicate is itself a claim about scope.

Population: every concrete subclass of ``BaseHJBSolver`` / ``BaseFPSolver`` under
``mfgarchon.alg.numerical``, by ``walk_packages`` + ``issubclass``. Discovery must not be keyed
on the property under audit, so no signature takes part in choosing it -- naming ``source_term``
is a recorded column, never a filter. An earlier draft of this file walked only the
``hjb_solvers`` and ``fp_solvers`` packages and found 5 and 7; the correct roots give **10 and
11**, which is where #1991's "ten" comes from. The narrow predicate silently halved the
population and nothing in its output said so.

**THREADS_IT is liveness, not correctness, and the distinction is not a quibble.**
``tests/integration/test_weak_form_source_mms_2020.py`` states it for the weak-form family and it
governs this table too: *"the answer moves when a source is passed" is a liveness check, not a
correctness one; what a source channel owes is an ORDER against an exact solution.* A solver can
apply a source at the wrong sign, the wrong time level or the wrong quadrature point and still
move the answer by O(1). Everything in the THREADS_IT column has passed the cheapest possible
test and nothing more; only ``WeakFormHJBSolver`` / ``WeakFormFPSolver`` currently carry order
studies, and those two are ``NOT_PROBED`` here because this fixture cannot build their
discretization. So the two files are complementary and neither subsumes the other: this one says
which solvers a source reaches, that one says whether the source it reaches is right.

A solver newly wired under #2020 owes an order study, not a green row here.

What the noise floor does NOT yet do: it is **one** repeat, so it estimates a stochastic
solver's spread from a single sample. That is enough today because every solver reaching the
difference test is deterministic -- ``FPFDMSolver`` and the rest return byte-identical arrays on
a repeat, and the file passes identically over five consecutive runs. The moment a stochastic
solver becomes reachable (``FPParticleSolver`` is the live candidate, currently ``REFUSED``), the
floor needs several repeats and the ``10x`` margin needs justifying against the resulting spread
rather than being asserted. Recorded here rather than discovered later, because a threshold that
has never been stressed reads exactly like one that has.

``HJBHowardSolver`` is deliberately absent: it is an iterator, not a solver, and
``scripts/capability_census.py`` already lists it under ``OUTSIDE_EVERY_PREDICATE``.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil

import pytest

import numpy as np

import mfgarchon.alg.numerical as _numerical
from mfgarchon.alg.numerical.fp_solvers.base_fp import BaseFPSolver
from mfgarchon.alg.numerical.hjb_solvers.base_hjb import BaseHJBSolver
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.geometry.grids import TensorProductGrid

THREADS_IT = "THREADS_IT"
REFUSED = "REFUSED"
ACCEPTS_AND_IGNORES = "ACCEPTS_AND_IGNORES"
NOT_PROBED = "NOT_PROBED"
INDISTINGUISHABLE = "INDISTINGUISHABLE_FROM_ITS_OWN_NOISE"

#: Measured 2026-08-21. Leaving REFUSED for THREADS_IT is #2020 progress and this set must be
#: updated; the reverse is a regression. ACCEPTS_AND_IGNORES may never appear here.
EXPECTED = {
    "HJBFDMSolver": THREADS_IT,
    "HJBGFDMSolver": THREADS_IT,
    "HJBWENOSolver": THREADS_IT,
    "HJBSemiLagrangianSolver": REFUSED,
    "FPFDMSolver": THREADS_IT,
    "FPFVMSolver": THREADS_IT,
    "FPGFDMSolver": REFUSED,
    "FPParticleSolver": REFUSED,
    "FPSLSolver": REFUSED,
    "FPSLAdjointSolver": REFUSED,
    "FPSLJacobianSolver": REFUSED,
}

#: What THIS fixture cannot reach, named rather than discovered. Each needs a problem this one
#: does not build -- not evidence about the solver.
NOT_PROBED_EXPECTED = {
    "HJBFEMSolver": "requires an unstructured mesh",
    "FPFEMSolver": "requires an unstructured mesh",
    "MeshlessGalerkinHJBSolver": "MLS moment matrix ill-conditioned on this node set",
    "MeshlessGalerkinFPSolver": "MLS moment matrix ill-conditioned on this node set",
    "NetworkHJBSolver": "requires a network problem",
    "NetworkPolicyIterationHJBSolver": "requires a network problem",
    "FPNetworkSolver": "requires a network problem",
    "WeakFormHJBSolver": "requires a WeakFormDiscretization",
    "WeakFormFPSolver": "requires a WeakFormDiscretization",
    "PenaltyHJBSolver": "wrapper -- requires an inner solver and an obstacle",
}

NX = 19


def _population():
    hjb, fp = {}, {}
    for info in pkgutil.walk_packages(_numerical.__path__, _numerical.__name__ + "."):
        try:
            module = importlib.import_module(info.name)
        except Exception:
            continue
        for obj in vars(module).values():
            if not inspect.isclass(obj):
                continue
            if issubclass(obj, BaseHJBSolver) and obj is not BaseHJBSolver:
                hjb[obj.__name__] = obj
            elif issubclass(obj, BaseFPSolver) and obj is not BaseFPSolver:
                fp[obj.__name__] = obj
    return hjb, fp


@pytest.fixture(scope="module")
def fixture():
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], boundary_conditions=no_flux_bc(dimension=1), Nx=[NX])
    problem = MFGProblem(geometry=grid, T=0.2, Nt=5, sigma=0.2, components=_components())
    npts = problem.geometry.get_grid_shape()[0]
    nt = problem.Nt_points
    x = np.asarray(grid.coordinates[0]).ravel()
    m0 = np.exp(-30.0 * (x - 0.5) ** 2)
    return {
        "problem": problem,
        "x": x,
        "M": np.full((nt, npts), 1.0 / npts),
        "U_terminal": (x - 0.5) ** 2,
        "U_zero": np.zeros((nt, npts)),
        "m0": m0 / m0.sum(),
    }


def _components():
    from tests.integration.test_hjb_with_obstacle import _default_components

    return _default_components()


def _flat(xs):
    a = np.asarray(xs)
    return a[:, 0] if a.ndim == 2 else a.ravel()


def _zero_source(t, xs):
    return np.zeros_like(_flat(xs))


def _strong_source(t, xs):
    # Sign-definite and O(1) against a density of O(1/Nx): a solver that applies it at all
    # cannot return a byte-identical answer, so ACCEPTS_AND_IGNORES is unambiguous.
    return 5.0 * np.sin(np.pi * _flat(xs))


def _classify(name, cls, kind, f):
    params = inspect.signature(cls.__init__).parameters
    kwargs = {}
    if "collocation_points" in params:
        kwargs["collocation_points"] = f["x"].reshape(-1, 1)
    required = [
        n
        for n, p in params.items()
        if n not in ("self", "problem")
        and p.default is inspect.Parameter.empty
        and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        and n not in kwargs
    ]
    if required:
        return NOT_PROBED, f"requires {required}"
    try:
        solver = cls(f["problem"], **kwargs)
    except Exception as exc:
        return NOT_PROBED, f"construction raised {type(exc).__name__}"

    method = solver.solve_hjb_system if kind == "hjb" else solver.solve_fp_system

    def _solve(src):
        args = (f["M"], f["U_terminal"], f["U_zero"]) if kind == "hjb" else (f["m0"].copy(), f["U_zero"])
        return np.asarray(method(*args, source_term=src))

    try:
        quiet = _solve(_zero_source)
        # Noise floor FIRST, from a repeat of the SAME source. Without it, "the answer moved"
        # is not evidence that the source moved it: `FPParticleSolver` differs by 1.3e-01 on
        # byte-identical inputs, so a stochastic solver that ignores the source entirely would
        # be scored THREADS_IT. Found by mutation -- an ACCEPTS_AND_IGNORES mutant of that
        # solver went undetected until this check existed.
        noise = float(np.abs(quiet - _solve(_zero_source)).max())
        loud = _solve(_strong_source)
    except (TypeError, NotImplementedError) as exc:
        return REFUSED, f"{type(exc).__name__}: {str(exc)[:60]}"
    except Exception as exc:
        return NOT_PROBED, f"solve raised {type(exc).__name__}"

    signal = float(np.abs(quiet - loud).max())
    detail = f"signal = {signal:.3e}, noise = {noise:.3e}"
    if signal <= 1e-12:
        return ACCEPTS_AND_IGNORES, detail
    if noise > 1e-12 and signal <= 10.0 * noise:
        return INDISTINGUISHABLE, detail
    return THREADS_IT, detail


@pytest.fixture(scope="module")
def measured(fixture):
    logging.disable(logging.CRITICAL)
    try:
        hjb, fp = _population()
        out = {}
        for kind, pop in (("hjb", hjb), ("fp", fp)):
            for name in sorted(pop):
                out[name] = _classify(name, pop[name], kind, fixture)
    finally:
        logging.disable(logging.NOTSET)
    return out


def test_the_population_is_ten_hjb_and_eleven_fp():
    """Discovery must reach both hierarchies wherever they live under `alg.numerical`.

    Pinned because the first draft walked only `hjb_solvers`/`fp_solvers` and silently found 5
    and 7 -- half the population, with nothing in the output to say so.
    """
    hjb, fp = _population()
    assert len(hjb) == 10, sorted(hjb)
    assert len(fp) == 11, sorted(fp)
    assert "HJBHowardSolver" not in hjb, "Howard is an iterator, not a solver"


def test_no_solver_accepts_a_source_and_ignores_it(measured):
    """The class this file exists for.

    REFUSED is honest and THREADS_IT is correct. A solver that takes `source_term` and returns
    a byte-identical answer is neither: an MMS built on it measures the order of an equation
    without the source and reports a clean, wrong number.
    """
    offenders = {n: why for n, (v, why) in measured.items() if v == ACCEPTS_AND_IGNORES}
    assert not offenders, f"solvers that silently drop a source: {offenders}"


def test_measured_reachability_matches_the_recorded_table(measured):
    """Ratchet. REFUSED -> THREADS_IT is progress and belongs in EXPECTED; the reverse is a bug."""
    actual = {n: v for n, (v, _) in measured.items() if v != NOT_PROBED}
    assert actual == EXPECTED, {
        n: (EXPECTED.get(n), actual.get(n), measured.get(n, (None, ""))[1])
        for n in set(actual) | set(EXPECTED)
        if EXPECTED.get(n) != actual.get(n)
    }


def test_what_this_fixture_cannot_reach_is_named_not_discovered(measured):
    """A population predicate is a claim about scope, so its gaps are listed, not inferred.

    If a solver leaves NOT_PROBED it should leave this dict too -- silently shrinking the
    unreachable set is how a fixture starts looking more comprehensive than it is.
    """
    not_probed = {n for n, (v, _) in measured.items() if v == NOT_PROBED}
    assert not_probed == set(NOT_PROBED_EXPECTED), {
        "newly unreachable": sorted(not_probed - set(NOT_PROBED_EXPECTED)),
        "now reachable": sorted(set(NOT_PROBED_EXPECTED) - not_probed),
    }


def test_the_probe_would_notice_a_source_that_is_applied(fixture):
    """Positive control on the instrument.

    `test_no_solver_accepts_a_source_and_ignores_it` passes when nothing is in that class, which
    is also what it does if the two solves were never actually different. This asserts the
    fixture's own strong source is strong enough to move a solver known to thread it.
    """
    from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver

    solver = FPFDMSolver(fixture["problem"])
    quiet = np.asarray(solver.solve_fp_system(fixture["m0"].copy(), fixture["U_zero"], source_term=_zero_source))
    loud = np.asarray(solver.solve_fp_system(fixture["m0"].copy(), fixture["U_zero"], source_term=_strong_source))
    assert np.abs(quiet - loud).max() > 1e-3, "the fixture's source is too weak to discriminate"
