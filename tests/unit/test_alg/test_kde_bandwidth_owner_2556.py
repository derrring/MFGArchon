"""A KDE bandwidth has one owner, and the library's own readers read it through it (#2556, C3-1).

The rule (CONVENTIONS.md § 9): a `bandwidth` is a factor of the sample's spread, as in `scipy.stats.gaussian_kde`:
"scott" and "silverman" are scipy's rules, and a positive number is the factor itself. The kernel is a Gaussian with
covariance factor^2 * cov(sample, ddof 1), full in every dimension. The owner is
`utils.numerical.particle.interpolation`: `estimate_kde_bandwidth` (the rules), `kde_factor`, `kde_kernel_covariance`.

**Oracles**, each from the rule: scipy's own `gaussian_kde` (its `factor`, its `covariance`, its density), and, for the
reflection estimator, the zero derivative normal to the wall on the wall, which the method of images gives for any
sample covariance (a particle's image carries the reflected kernel R Sigma R).

**Readers pinned here:** the two Gaussian estimators of `density_estimation`, the particle-to-grid projector, the
reflection and renormalization estimators, the particle solver's beta and GPU paths, `FPParticleResult`'s query, and the
solver's default path on a point-mass initial density. The fuzz over n, d, sample shape, bandwidth and refused inputs,
with a rival for each misreading, is the PR's, in scratch.
"""

from __future__ import annotations

import logging
import warnings

import pytest

import numpy as np
from scipy.stats import gaussian_kde

from mfgarchon.alg.numerical import density_estimation as de
from mfgarchon.backends import create_backend
from mfgarchon.utils.numerical.particle import kde_boundary as kb
from mfgarchon.utils.numerical.particle.interpolation import estimate_kde_bandwidth, kde_factor, kde_kernel_covariance

BANDS = [0.5, "scott", "silverman"]
RNG = np.random.default_rng(2556)
CORRELATED = RNG.multivariate_normal([0.5, 0.5], [[0.01, 0.0094], [0.0094, 0.0144]], size=800)  # rho 0.78
ONE_D = RNG.normal(0.5, 0.05, 1000)


@pytest.mark.parametrize("d", [1, 2, 3])
@pytest.mark.parametrize("rule", ["scott", "silverman"])
def test_the_rules_are_scipys(rule, d):
    """`estimate_kde_bandwidth` is scipy's factor (its equivalence pin), and `kde_factor` reaches it for a rule and
    passes a positive number through."""
    sample = RNG.normal(size=(300, d))
    assert estimate_kde_bandwidth(sample, method=rule) == pytest.approx(
        gaussian_kde(sample.T, bw_method=rule).factor, rel=1e-14
    )
    assert kde_factor(rule, sample) == estimate_kde_bandwidth(sample, method=rule)
    assert kde_factor(0.5, sample) == 0.5


@pytest.mark.parametrize("bandwidth", [0, -1.0, np.nan, np.inf, True, "foo", np.array([0.1, 0.2])])
def test_a_bandwidth_that_is_not_a_factor_is_refused(bandwidth):
    with pytest.raises(ValueError, match=r"positive factor|Unknown method"):
        kde_factor(bandwidth, ONE_D)


@pytest.mark.parametrize("bandwidth", BANDS)
@pytest.mark.parametrize("sample", [ONE_D, CORRELATED], ids=["1-D", "2-D correlated"])
def test_the_kernel_is_scipys(sample, bandwidth):
    np.testing.assert_allclose(
        kde_kernel_covariance(sample, bandwidth),
        gaussian_kde(np.atleast_2d(sample.T), bw_method=bandwidth).covariance,
        rtol=1e-12,
    )


@pytest.mark.parametrize(
    "sample",
    [np.full(50, 0.3), np.column_stack([np.linspace(0, 1, 50), np.linspace(0, 1, 50)]), np.array([[0.2, 0.4]])],
    ids=["one point", "a line in 2-D", "one particle"],
)
def test_a_sample_with_no_spread_is_refused(sample):
    with pytest.raises(ValueError, match="spread in every direction"):
        kde_kernel_covariance(sample, "scott")


@pytest.mark.parametrize("bandwidth", BANDS)
def test_the_gaussian_estimators_are_scipys_kde(bandwidth):
    """`gaussian_kde_gpu` (1-D), `gaussian_kde_gpu_nd` on a correlated 2-D sample (scipy's full covariance), and the
    projector in 2-D, which handed a rule's absolute width to a reader of a factor."""
    backend = create_backend("numpy")
    points_1d = np.linspace(0.3, 0.7, 41)
    np.testing.assert_allclose(
        de.gaussian_kde_gpu(ONE_D, points_1d, bandwidth, backend),
        gaussian_kde(ONE_D, bw_method=bandwidth)(points_1d),
        rtol=1e-10,
    )
    points = RNG.multivariate_normal([0.5, 0.5], [[0.02, 0.0], [0.0, 0.02]], size=60)
    np.testing.assert_allclose(
        de.gaussian_kde_gpu_nd(CORRELATED, points, bandwidth, backend),
        gaussian_kde(CORRELATED.T, bw_method=bandwidth)(points.T),
        rtol=1e-10,
    )

    import types

    from mfgarchon.geometry import TensorProductGrid
    from mfgarchon.geometry.boundary import no_flux_bc
    from mfgarchon.operators.interpolation.projection import GeometryProjector

    grid = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[21, 21], boundary_conditions=no_flux_bc(dimension=2)
    )
    fake = types.SimpleNamespace(
        hjb_geometry=grid, fp_geometry=types.SimpleNamespace(get_spatial_grid=lambda: CORRELATED)
    )
    projected = GeometryProjector._project_particles_to_grid_kde(fake, np.ones(len(CORRELATED)), bandwidth, None)
    nodes = np.stack(np.meshgrid(*grid.coordinates, indexing="ij"), axis=-1).reshape(-1, 2)
    np.testing.assert_allclose(
        projected.ravel(), gaussian_kde(CORRELATED.T, bw_method=bandwidth)(nodes.T), rtol=1e-10, atol=1e-300
    )


@pytest.mark.parametrize("bandwidth", BANDS)
@pytest.mark.parametrize("where", ["wall", "corner"])
def test_reflection_has_no_slope_across_a_wall(where, bandwidth):
    """A correlated sample against the wall x = 0, or the corner (0, 0): the estimate's derivative normal to the wall,
    on the wall, is zero to the stencil's error, measured from inside with the second-order one-sided stencil. The same
    Sigma at the particles' images gave (audit session) +0.26, -0.44 and -1.64 in units of f / h on this geometry."""
    centre = [0.1, 0.5] if where == "wall" else [0.1, 0.1]
    sample = np.abs(RNG.multivariate_normal(centre, [[0.01, 0.0094], [0.0094, 0.0144]], size=1500))
    h = np.sqrt(np.diag(kde_kernel_covariance(sample, bandwidth)))
    eps = 1e-5
    for y in (0.4, 0.5, 0.6) if where == "wall" else (0.0, 0.03, 0.08):
        f0, f1, f2 = kb.reflection_kde(
            sample, np.array([[0.0, y], [eps, y], [2 * eps, y]]), bandwidth, [(0.0, 1.0), (0.0, 1.0)], normalize=False
        )
        slope = (-3 * f0 + 4 * f1 - f2) / (2 * eps)
        assert abs(slope) / (f0 / h[0]) < 1e-3, (where, y, slope, f0)


@pytest.mark.parametrize("bandwidth", BANDS)
def test_the_wall_estimators_read_the_rule_far_from_a_wall(bandwidth):
    """Ten standard deviations from either wall, the reflection and renormalization estimators are scipy's KDE.
    Before, a number was the reflection's absolute ghost zone and a factor of the spread of sample plus ghosts (0.153 of
    scipy at the centre), and the renormalization raised TypeError on a rule."""
    centre = np.array([0.5])
    want = gaussian_kde(ONE_D, bw_method=bandwidth)(centre)[0]
    assert kb.reflection_kde(ONE_D, centre, bandwidth, [(0.0, 1.0)], normalize=False)[0] == pytest.approx(
        want, rel=1e-6
    )
    assert kb.renormalization_kde(ONE_D, centre, bandwidth, [(0.0, 1.0)])[0] == pytest.approx(want, rel=1e-6)


def _problem(nx=21):
    from mfgarchon import Conditions, MFGProblem, Model
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
    from mfgarchon.geometry import TensorProductGrid
    from mfgarchon.geometry.boundary import no_flux_bc

    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[nx], boundary_conditions=no_flux_bc(dimension=1))
    model = Model(hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0)), volatility=0.3)
    conditions = Conditions(m_initial=lambda z: 1.0, u_terminal=lambda z: 0.0, T=0.1)
    return MFGProblem(model=model, domain=grid, conditions=conditions, Nt=4)


def _gaussian_density(nx, std=0.05):
    x = np.linspace(0.0, 1.0, nx)
    m = np.exp(-0.5 * ((x - 0.5) / std) ** 2)
    return m / np.trapezoid(m, x)


@pytest.mark.parametrize("bandwidth", ["scott", "silverman"])
def test_the_beta_path_reads_a_rule(bandwidth):
    """`kde_method="beta"` with a rule raised TypeError inside the estimator (#2804's crash row): the solver hands it
    the kernel's width."""
    from mfgarchon.alg.numerical.fp_solvers.fp_particle import FPParticleSolver

    solver = FPParticleSolver(_problem(), num_particles=400, seed=3, kde_method="beta", kde_bandwidth=bandwidth)
    M = np.asarray(solver.solve_fp_system(_gaussian_density(21), drift_field=np.zeros((5, 21)), show_progress=False))
    assert np.isfinite(M).all()


@pytest.mark.parametrize("bandwidth", BANDS)
def test_the_gpu_path_reads_each_steps_spread(monkeypatch, bandwidth):
    """The GPU path computed its width once from X0 and used it at every step, so a factor read std(X0). Every width it
    hands its kernel is scipy's for that step's sample."""
    pytest.importorskip("torch")
    from mfgarchon.alg.numerical.fp_solvers.fp_particle import FPParticleSolver

    handed = []
    kernel = de.gaussian_kde_gpu_internal

    def recording(particles, grid, width, backend, kernel_name="gaussian"):
        handed.append((np.asarray(backend.to_numpy(particles), dtype=float).copy(), width))
        return kernel(particles, grid, width, backend, kernel_name)

    monkeypatch.setattr(de, "gaussian_kde_gpu_internal", recording)
    nx = 101
    solver = FPParticleSolver(_problem(nx), num_particles=1500, seed=7, backend="torch", kde_bandwidth=bandwidth)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logging.disable(logging.WARNING)
        try:
            solver.solve_fp_system(_gaussian_density(nx, 0.03), drift_field=np.zeros((5, nx)), show_progress=False)
        finally:
            logging.disable(logging.NOTSET)
    assert len(handed) == 5, len(handed)
    for positions, width in handed:
        assert width == pytest.approx(
            gaussian_kde(positions, bw_method=bandwidth).factor * np.std(positions, ddof=1), rel=1e-12
        )
    assert handed[-1][1] > 1.5 * handed[0][1], "the sample spread under the diffusion, and the width with it"


def test_the_particle_result_queries_with_the_factor(monkeypatch):
    """`FPParticleResult(bandwidth=b)` holds the solver's factor, and its query was handed b as an absolute width. It
    hands the query b times that timestep's spread."""
    from mfgarchon.alg.numerical.fp_solvers import particle_result
    from mfgarchon.alg.numerical.fp_solvers.particle_result import FPParticleResult

    widths = []
    real = particle_result.ParticleDensityQuery

    def recording(particles, bandwidth=None, bandwidth_rule="fixed", **kwargs):
        widths.append((np.asarray(particles), bandwidth, bandwidth_rule))
        return real(particles, bandwidth=bandwidth, bandwidth_rule=bandwidth_rule, **kwargs)

    monkeypatch.setattr(particle_result, "ParticleDensityQuery", recording)
    history = [RNG.normal(0.5, s, (500, 1)) for s in (0.05, 0.1)]
    result = FPParticleResult(np.zeros((2, 11)), np.array([0.0, 0.1]), particle_history=history, bandwidth=0.5)
    for t in (0, 1):
        result.query_density(np.array([[0.5]]), timestep=t, method="kernel")
    for (_positions, width, rule), sample in zip(widths, history, strict=True):
        assert rule == "fixed"
        assert width == pytest.approx(0.5 * np.std(sample, ddof=1), rel=1e-12)


def test_a_point_mass_initial_density_still_solves_on_the_default_path():
    """The default path (reflection, "scott") on a point-mass initial density: M[0] is the density handed in, and the
    particles have spread by the next step, so no kernel is built on a sample with no spread (C3's refusal)."""
    from mfgarchon.alg.numerical.fp_solvers.fp_particle import FPParticleSolver

    point = np.where(np.abs(np.linspace(0.0, 1.0, 21) - 0.5) < 1e-9, 20.0, 0.0)
    M = np.asarray(
        FPParticleSolver(_problem(), num_particles=400, seed=5).solve_fp_system(
            point, drift_field=np.zeros((5, 21)), show_progress=False
        )
    )
    assert np.isfinite(M).all()
    assert M[-1].max() < M[0].max()
