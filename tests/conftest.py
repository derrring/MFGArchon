"""
Pytest configuration and shared fixtures for MFGarchon test suite.

This module provides common fixtures, test configuration, and utilities
used across the entire test suite.
"""

import contextlib
import logging
import os
import shutil
import tempfile
from pathlib import Path

import pytest

import numpy as np

# Issue #2090: select a headless matplotlib backend before pyplot is imported.
#
# `ConvergenceInfo.plot_convergence()` ends in a bare `plt.show()` and a unit test calls it. On a
# GUI backend with a foreground session that blocks until the window is dismissed, so the suite
# does not fail -- it waits. Measured here: that test exits 124 under a 60s cap, and passes in
# 0.03s under Agg. (On a machine with no foreground GUI session it completes slowly instead of
# blocking, so the symptom is environment-dependent; the cause is not.)
#
# Both interpreters used with this repository default to `macosx` -- the conda env the gate runs
# under and `uv run --extra dev` alike -- so a headless backend is not something either
# environment supplies.
#
# `MPLBACKEND` is the mechanism: it is read when matplotlib is first imported, and this file is
# loaded at `pytest_load_initial_conftests`, before any test module and separately in every xdist
# worker. `use(..., force=True)` is belt-and-braces for a future plugin that imports matplotlib
# ahead of this file. No such plugin exists here today -- traced over a full-tree collection, the
# first matplotlib import in the process is this file -- so treat it as insurance, not as a
# measured necessity.
#
# `setdefault` so a backend can be forced deliberately; the `use()` call then follows whatever
# that resolved to rather than overriding it.
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib  # must follow the MPLBACKEND assignment above

try:
    matplotlib.use(os.environ["MPLBACKEND"], force=True)
except ValueError as _exc:  # an unusable MPLBACKEND must name itself, not kill the session
    raise RuntimeError(
        f"MPLBACKEND={os.environ['MPLBACKEND']!r} is not a backend matplotlib accepts. "
        f"The test suite needs a non-interactive one (Agg, pdf, svg, ps, pgf, template)."
    ) from _exc

# Import main package components
from mfgarchon import MFGProblem
from mfgarchon.config import MFGSolverConfig
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.factory import lq_mfg_initial_density, lq_mfg_terminal_cost
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

# =============================================================================
# Default Components for Testing (Issue #670, #673: explicit specification required)
# =============================================================================


def _default_hamiltonian():
    """Default class-based Hamiltonian for tests (Issue #673)."""
    return SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: m,
        coupling_dm=lambda m: 1.0,
    )


def _default_test_components(Lx: float = 1.0) -> MFGComponents:
    """
    Default MFGComponents for shared test fixtures.

    Uses LQ MFG problem components for well-tested behavior.

    Args:
        Lx: Domain length for terminal cost scaling

    Returns:
        MFGComponents with Gaussian initial density and quadratic terminal cost
    """
    return MFGComponents(
        hamiltonian=_default_hamiltonian(),
        m_initial=lq_mfg_initial_density(),
        u_terminal=lq_mfg_terminal_cost(Lx=Lx),
    )


# =============================================================================
# Test Configuration
# =============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    # Core test type markers
    config.addinivalue_line("markers", "unit: Unit tests (fast, isolated)")
    config.addinivalue_line("markers", "integration: Integration tests (slower, cross-component)")
    config.addinivalue_line("markers", "performance: Performance tests (may be slow)")
    config.addinivalue_line("markers", "mathematical: Mathematical property validation tests")
    config.addinivalue_line("markers", "slow: Slow tests (may take >30 seconds)")

    # Test tier markers (for CI pipeline control)
    config.addinivalue_line("markers", "tier1: Fast unit tests (<1s) - run on every commit")
    config.addinivalue_line("markers", "tier2: Medium tests (1-30s) - run on PRs")
    config.addinivalue_line("markers", "tier3: Slow integration tests (>30s) - run on merge to main")
    config.addinivalue_line("markers", "tier4: Performance/stress tests - run weekly or manually")

    # Domain-specific markers
    config.addinivalue_line("markers", "network: Tests requiring network/graph geometry")
    config.addinivalue_line("markers", "stochastic: Tests for stochastic MFG solvers")
    config.addinivalue_line("markers", "numerical: Tests for numerical algorithms")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test paths."""
    for item in items:
        # Add markers based on test file paths
        test_path = str(item.fspath)

        if "/unit/" in test_path:
            item.add_marker(pytest.mark.unit)
        elif "/integration/" in test_path:
            item.add_marker(pytest.mark.integration)
        elif "/performance/" in test_path:
            item.add_marker(pytest.mark.performance)
        elif "/mathematical/" in test_path:
            item.add_marker(pytest.mark.mathematical)

        # Mark slow tests based on name patterns
        if "large" in item.name or "slow" in item.name or "benchmark" in item.name:
            item.add_marker(pytest.mark.slow)


# =============================================================================
# Problem Fixtures
# =============================================================================


@pytest.fixture
def tiny_problem():
    """Very small problem for quick tests."""
    geometry = TensorProductGrid(
        bounds=[(0.0, 1.0)], Nx_points=[6], boundary_conditions=no_flux_bc(dimension=1)
    )  # Nx=5 -> 6 points
    return MFGProblem(
        geometry=geometry,
        Nt=3,
        T=0.1,
        components=_default_test_components(Lx=1.0),
    )


@pytest.fixture
def small_problem():
    """Small problem for unit tests."""
    geometry = TensorProductGrid(
        bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=no_flux_bc(dimension=1)
    )  # Nx=10 -> 11 points
    return MFGProblem(
        geometry=geometry,
        Nt=5,
        T=0.5,
        components=_default_test_components(Lx=1.0),
    )


@pytest.fixture
def medium_problem():
    """Medium problem for integration tests."""
    geometry = TensorProductGrid(
        bounds=[(0.0, 1.0)], Nx_points=[26], boundary_conditions=no_flux_bc(dimension=1)
    )  # Nx=25 -> 26 points
    return MFGProblem(
        geometry=geometry,
        Nt=12,
        T=1.0,
        components=_default_test_components(Lx=1.0),
    )


@pytest.fixture
def large_problem():
    """Large problem for performance tests."""
    geometry = TensorProductGrid(
        bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1)
    )  # Nx=50 -> 51 points
    return MFGProblem(
        geometry=geometry,
        Nt=25,
        T=2.0,
        components=_default_test_components(Lx=1.0),
    )


@pytest.fixture(
    params=[
        {"Nx_points": 11, "Nt": 5, "T": 0.5},  # Nx=10 -> 11 points
        {"Nx_points": 16, "Nt": 8, "T": 1.0},  # Nx=15 -> 16 points
        {"Nx_points": 21, "Nt": 10, "T": 1.5},  # Nx=20 -> 21 points
    ]
)
def parametrized_problem(request):
    """Parametrized problem fixture for testing multiple configurations."""
    params = request.param
    geometry = TensorProductGrid(
        bounds=[(0.0, 1.0)], Nx_points=[params["Nx_points"]], boundary_conditions=no_flux_bc(dimension=1)
    )
    return MFGProblem(
        geometry=geometry,
        Nt=params["Nt"],
        T=params["T"],
        components=_default_test_components(Lx=1.0),
    )


@pytest.fixture(params=[0.1, 0.5, 1.0, 2.0])
def diffusion_coefficient(request):
    """Parametrized diffusion coefficient for testing."""
    return request.param


# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture
def default_config():
    """Default MFGSolverConfig for testing."""
    return MFGSolverConfig()


# =============================================================================
# Data Fixtures
# =============================================================================


@pytest.fixture
def deterministic_arrays():
    """Deterministic arrays for reproducible tests."""
    np.random.seed(42)  # Fixed seed for reproducibility
    # Note: shapes match small_problem (Nx=10→11 points) and medium_problem (Nx=25→26 points)
    return {
        "U_small": np.random.rand(6, 11),  # (Nt+1, Nx+1) for small problem
        "M_small": np.random.rand(6, 11),
        "U_medium": np.random.rand(13, 26),  # (Nt+1, Nx+1) for medium problem
        "M_medium": np.random.rand(13, 26),
    }


@pytest.fixture
def valid_test_matrices():
    """Valid test matrices with proper physical properties."""
    np.random.seed(123)

    # Create density matrix with mass conservation
    M = np.random.rand(11, 21)
    M = np.maximum(M, 0)  # Ensure non-negativity
    # Normalize each time slice to conserve mass
    for t in range(M.shape[0]):
        if np.sum(M[t, :]) > 0:
            M[t, :] /= np.sum(M[t, :])

    # Create value function matrix
    U = np.random.rand(11, 21) * 10 - 5  # Range [-5, 5]

    return {"U": U, "M": M}


@pytest.fixture
def boundary_conditions():
    """Standard boundary condition configurations."""
    return {
        "dirichlet": {"type": "dirichlet", "value": 0.0},
        "neumann": {"type": "neumann", "derivative": 0.0},
        "periodic": {"type": "periodic"},
        "mixed": {"type": "mixed", "left": "dirichlet", "right": "neumann"},
    }


# =============================================================================
# File System Fixtures
# =============================================================================


@pytest.fixture
def temp_directory():
    """Temporary directory for file operations."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_output_dir(temp_directory):
    """Test output directory with subdirectories."""
    output_dir = temp_directory / "test_output"
    output_dir.mkdir()

    # Create subdirectories
    (output_dir / "results").mkdir()
    (output_dir / "plots").mkdir()
    (output_dir / "reports").mkdir()

    return output_dir


# =============================================================================
# Solver Fixtures
# =============================================================================


@pytest.fixture(params=["fixed_point"])
def solver_type(request):
    """Parametrized solver type for testing."""
    return request.param


@pytest.fixture
def solver_factory():
    """Factory function for creating solvers."""

    def _create_solver(problem, config=None):
        from mfgarchon.factory import create_solver

        return create_solver(problem, config=config)

    return _create_solver


# =============================================================================
# Mathematical Properties Fixtures
# =============================================================================


@pytest.fixture
def tolerance_levels():
    """Different tolerance levels for testing convergence."""
    return {"strict": 1e-8, "normal": 1e-6, "relaxed": 1e-4, "loose": 1e-2}


@pytest.fixture
def convergence_criteria():
    """Different convergence criteria configurations."""
    return {
        "standard": {"relative_tolerance": True, "absolute_tolerance": False},
        "absolute": {"relative_tolerance": False, "absolute_tolerance": True},
        "combined": {"relative_tolerance": True, "absolute_tolerance": True},
    }


# =============================================================================
# Utility Fixtures
# =============================================================================


@pytest.fixture
def performance_timer():
    """Timer utility for performance testing."""
    import time

    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def start(self):
            self.start_time = time.time()

        def stop(self):
            self.end_time = time.time()
            return self.elapsed

        @property
        def elapsed(self):
            if self.start_time is None:
                return 0
            end = self.end_time or time.time()
            return end - self.start_time

    return Timer()


@pytest.fixture
def memory_tracker():
    """Memory usage tracker for performance testing."""
    import psutil

    class MemoryTracker:
        def __init__(self):
            self.process = psutil.Process()
            self.start_memory = None
            self.peak_memory = 0

        def start(self):
            self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            self.peak_memory = self.start_memory

        def update(self):
            current_memory = self.process.memory_info().rss / 1024 / 1024
            self.peak_memory = max(self.peak_memory, current_memory)
            return current_memory

        @property
        def current_mb(self):
            return self.process.memory_info().rss / 1024 / 1024

        @property
        def increase_mb(self):
            if self.start_memory is None:
                return 0
            return self.current_mb - self.start_memory

    return MemoryTracker()


# =============================================================================
# Mock and Stub Fixtures
# =============================================================================


@pytest.fixture
def mock_convergence_result():
    """Mock convergence result for testing."""
    return {
        "converged": True,
        "iterations": 15,
        "final_error": 1e-7,
        "error_history": [1e-1, 3e-2, 8e-3, 2e-3, 5e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7],
        "convergence_rate": 0.3,
        "execution_time": 2.5,
    }


@pytest.fixture
def mock_failed_convergence():
    """Mock failed convergence result for testing error handling."""
    return {
        "converged": False,
        "iterations": 50,
        "final_error": 1e-3,
        "error_history": [1e-1, 5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3],
        "convergence_rate": None,
        "execution_time": 10.0,
        "failure_reason": "Maximum iterations reached",
    }


# =============================================================================
# Parameterized Test Data
# =============================================================================


@pytest.fixture(
    params=[
        (10, 5, 0.5),  # Small problem
        (20, 10, 1.0),  # Medium problem
        (30, 15, 1.5),  # Large problem
    ]
)
def problem_dimensions(request):
    """Parametrized problem dimensions (Nx, Nt, T)."""
    return request.param


@pytest.fixture(
    params=[
        {"max_iterations": 10, "tolerance": 1e-4},
        {"max_iterations": 20, "tolerance": 1e-6},
        {"max_iterations": 50, "tolerance": 1e-8},
    ]
)
def newton_parameters(request):
    """Parametrized Newton solver parameters."""
    return request.param


# =============================================================================
# Session-Scoped Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def reference_solutions():
    """Reference solutions for validation (computed once per session)."""
    # This would load or compute reference solutions
    # For now, return empty dict - implement as needed
    return {}


@pytest.fixture(scope="session")
def performance_baselines():
    """Performance baselines for regression testing."""
    return {
        "small_problem_time": 1.0,  # seconds
        "medium_problem_time": 5.0,  # seconds
        "memory_per_dof": 1e-6,  # MB per degree of freedom
    }


# =============================================================================
# Log capture for mfgarchon loggers (Issue #2083)
# =============================================================================


class _RecordCollector(logging.Handler):
    """Appends every record it is handed to a list owned by the capture object.

    It carries no level of its own: ``at_level`` sets the level on the logger, which is the
    one owner for the threshold (#2083).

    The measurement first offered for this was a null result with no positive control --
    loosening the handler to level 0 killed no test, which it could not have done, since the
    logger's own level already filtered everything below. The control that discriminates runs
    the other way: *tightening* the collector to CRITICAL kills 10 of the 16 tests here. So the
    handler level was reachable, and the honest reason to drop it is single-ownership, not
    inertness. It is also not universally true that the logger's level decides: a *propagating*
    descendant delivers its records to this handler without the target logger's level being
    consulted. No logger under an mfgarchon name propagates today -- they are all created
    through ``get_logger`` -- so this is a limit to know, not a live hole.
    """

    def __init__(self, records: list[logging.LogRecord]):
        super().__init__()
        self._records = records

    def emit(self, record: logging.LogRecord) -> None:
        self._records.append(record)


class MFGLogCapture:
    """Capture records from an mfgarchon logger, where plain ``caplog`` is unreliable (#2083).

    ``MFGLogger._setup_logger`` sets ``propagate = False`` on every logger it configures
    (``mfgarchon/utils/mfg_logging/logger.py:211``), so no mfgarchon record reaches the root
    logger. What ``caplog`` does about that differs by pytest version, and the two versions in
    use here disagree -- which is why six test modules each grew their own collector:

    - pytest 8.4.1 (``uv run --extra dev``): ``catching_logs.__enter__`` attaches the capture
      handler to the root logger only. **No mfgarchon record is visible, ever**, whatever the
      logger's creation site. On this version ``propagate = False`` is the whole story.
    - pytest 9.1.1 (the gate interpreter): it also attaches to every non-propagating logger
      **that already exists** when ``catching_logs.__enter__`` runs. That sweep runs once per
      test PHASE (setup / call / teardown), so the discriminator is not "module level vs
      inside a function" -- a logger created in a *fixture* is visible in the test body,
      measured -- it is whether the logger existed before this phase's sweep. A logger born
      mid-solve is not, which is the ``fp_gfdm.py:575`` case. pytest's own comment names this
      gap: the sweep "will miss loggers that *become* non-propagating after the ``__enter__``",
      and ``MFGLogger`` sets ``propagate`` at creation.

    So on 9.1.1 the same test passes or fails on whether an earlier test in the same worker
    created the logger first. 34 of the package's 104 ``get_logger`` calls are inside a
    function; 15 of those are consumer-side and 12 have ``fp_gfdm``'s shape, the rest being
    ``mfg_logging``'s own plumbing.

    This attaches to the emitting logger on demand, so it depends on neither the version nor
    the order.

    The API mirrors ``caplog`` so uses read the same, with one difference: ``logger=`` is
    required. There is no root to fall back to, and the no-argument form would capture nothing
    silently. Note what that does **not** cover: a name that is merely *wrong* captures nothing
    just as silently, and this fixture does **not** catch it -- see ``at_level``'s docstring for
    the two guards that were tried, why both were removed, and the discipline that replaces them.

    ```python
    def test_the_drift_is_reported(mfg_caplog):
        with mfg_caplog.at_level(logging.WARNING, logger="mfgarchon.alg....fp_gfdm"):
            solver.solve_fp_system(m0, drift)
        assert mfg_caplog.records
    ```

    ``records`` holds what arrived at or above the level ``at_level`` was given, in order,
    across every ``at_level`` block since the last ``clear()``.
    """

    def __init__(self) -> None:
        self.records: list[logging.LogRecord] = []
        self._active: set[str] = set()

    @property
    def messages(self) -> list[str]:
        """The formatted message of each captured record."""
        return [record.getMessage() for record in self.records]

    def clear(self) -> None:
        self.records.clear()

    @contextlib.contextmanager
    def at_level(self, level: int, logger: str):
        """Collect records of `level` and above emitted through the logger named `logger`.

        The logger is the object production code emits on -- ``MFGLogger.get_logger`` is
        ``logging.getLogger`` plus a cache, so both return the same instance. This takes it the
        plain way, which leaves the logger's configuration untouched.

        **A wrong name captures nothing, silently, and this fixture does not stop it.** That is
        the shape a typo takes: `assert not mfg_caplog.records` is satisfied by a misspelt
        logger exactly as it is by a solve that did not warn, and nothing here can tell them
        apart. The discipline that does: **pair every absence assertion with a presence
        assertion on the same logger name**, so a typo fails the presence half loudly. Five of
        the six absence assertions in this repository already do.

        Two guards against it were built and both were removed, because each re-created the
        order-dependence this fixture exists to remove (#2083, and the issue tracking a sound
        design):

        - *Refuse a name the package has never handed out, if the block captured nothing.*
          Fires on a correct absence assertion over a logger created inside a function --
          `fp_gfdm` and `mfg_problem` have no module-level `get_logger` -- so the verdict moved
          with whether an earlier test in the same worker had run a solve. Its cleanup also
          popped the name out of `logging.Logger.manager.loggerDict`, orphaning live loggers.
        - *Refuse a name that is neither a module path nor already registered.* Measured, **10 of
          11** logger names this package actually uses are not module paths (`MFGSolver`,
          `mfgarchon.performance`, `mfgarchon.solvers`, `mfgarchon.solvers.<class>`,
          `__name__ + ".PluginManager"`, ...) -- the one that is comes from a demo function -- so
          they fell through to the registry arm and the verdict moved with test order again. And `importlib.util.find_spec`
          imports every parent package to answer: one refused name under
          `mfgarchon.geometry.level_set.*` left **+10** loggers and **+8** registry entries
          behind, first-call-only.
        """
        if not isinstance(logger, str) or not logger:
            raise ValueError(
                "mfg_caplog.at_level needs the name of the logger to capture: "
                "at_level(logging.WARNING, logger='mfgarchon.alg.numerical....'). "
                "There is no root fallback -- mfgarchon loggers do not propagate (#2083)."
            )
        if logger in self._active:
            raise RuntimeError(
                f"mfg_caplog is already capturing {logger!r} in an enclosing block. Two "
                f"collectors on one logger append the same record twice, which silently "
                f"doubles any count assertion. Use one block, or capture a different logger."
            )

        # `logging.getLogger`, NOT the package's `get_logger`. They return the SAME object
        # (`MFGLogger.get_logger` is `logging.getLogger` plus a cache -- verified), but
        # `get_logger` also CONFIGURES a logger it has not seen: `_setup_logger` clears handlers,
        # sets the level, attaches a StreamHandler and sets `propagate = False`
        # (`utils/mfg_logging/logger.py:189-211`). A capture helper has no business doing that,
        # and undoing it took three revisions and three review rounds to get wrong three ways.
        # Not calling it removes the problem instead of restoring from it: nothing is configured,
        # nothing is cached in `MFGLogger._loggers`, and after the `finally` below the logger is
        # byte-for-byte as it was found. Measured on a name the package had never used:
        # `handlers=0 level=0 propagate=True` after the block, and a later production
        # `get_logger` still configures it normally.
        target = logging.getLogger(logger)
        handler = _RecordCollector(self.records)
        previous_level = target.level
        # Level first: an unusable level must raise before anything has been mutated.
        target.setLevel(level)
        target.addHandler(handler)
        self._active.add(logger)
        try:
            yield self
        finally:
            self._active.discard(logger)
            target.removeHandler(handler)
            target.setLevel(previous_level)


@pytest.fixture
def mfg_caplog() -> MFGLogCapture:
    """``caplog`` for mfgarchon loggers, independent of pytest version and logger creation
    order (Issue #2083). See :class:`MFGLogCapture`."""
    return MFGLogCapture()


# =============================================================================
# Cleanup Utilities
# =============================================================================


@pytest.fixture(autouse=True)
def cleanup_numpy_state():
    """Automatically cleanup numpy random state after each test."""
    yield
    # Reset numpy random state to avoid test interference
    np.random.seed(None)


@pytest.fixture(autouse=True)
def suppress_warnings():
    """Suppress known warnings during testing."""
    import warnings

    # Suppress specific warnings that are expected during testing
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="mfgarchon.*")
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

    yield

    # Reset warning filters
    warnings.resetwarnings()


# =============================================================================
# Test Data Validation
# =============================================================================


def validate_mfg_solution(U, M, problem):
    """Validate that U and M arrays represent a valid MFG solution."""
    # Check dimensions
    Nx_points = problem.geometry.get_grid_shape()[0]
    expected_shape = (problem.Nt + 1, Nx_points)
    assert U.shape == expected_shape, f"U shape {U.shape} != expected {expected_shape}"
    assert M.shape == expected_shape, f"M shape {M.shape} != expected {expected_shape}"

    # Check for NaN/Inf
    assert not np.any(np.isnan(U)), "U contains NaN values"
    assert not np.any(np.isnan(M)), "M contains NaN values"
    assert not np.any(np.isinf(U)), "U contains Inf values"
    assert not np.any(np.isinf(M)), "M contains Inf values"

    # Check physical properties
    assert np.all(M >= -1e-10), f"M contains negative values: min={np.min(M)}"

    # Check mass conservation (approximately)
    dx = problem.geometry.get_grid_spacing()[0]
    initial_mass = np.sum(problem.m_initial) * dx  # Issue #670: unified naming
    for t in range(problem.Nt + 1):
        current_mass = np.sum(M[t, :]) * dx
        mass_error = abs(current_mass - initial_mass)
        assert mass_error < 0.1, f"Mass conservation violated at t={t}: error={mass_error}"


# Export validation function for use in tests
__all__ = ["validate_mfg_solution"]
