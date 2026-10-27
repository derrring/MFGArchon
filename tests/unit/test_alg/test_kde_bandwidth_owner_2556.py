"""A KDE bandwidth has one owner, and the library's own readers read it through it (#2556, C3-1).

The rule (CONVENTIONS.md § 9): a `bandwidth` is a factor of the sample's spread, as in `scipy.stats.gaussian_kde`:
"scott" and "silverman" are scipy's rules, and a positive number is the factor itself. The kernel is a Gaussian with
covariance factor^2 * cov(sample, ddof 1), full in every dimension. The owner is
`utils.numerical.particle.interpolation`: `estimate_kde_bandwidth` (the rules), `kde_factor`, `kde_kernel_covariance`,
`kde_kernel_widths` (its diagonal's square roots, which exist on a line), `kde_sample_has_spread`.

**Oracles**, each from the rule: scipy's own `gaussian_kde` (its `factor`, its `covariance`, its density), and, for the
reflection estimator, the zero derivative normal to the wall on the wall, which the method of images gives for any
sample covariance (a particle's image carries the reflected kernel R Sigma R); for the renormalization estimator, the
density of a uniform sample one kernel width from a wall.

**Readers pinned here:** the two Gaussian estimators of `density_estimation`, the particle-to-grid projector, the
reflection and renormalization estimators, the particle solver's beta, GPU and n-D default paths, and `FPParticleResult`'s
query. The fuzz over n, d, sample shape, bandwidth and refused inputs, with a rival for each misreading, is the PR's, in
scratch.

**The no-kernel member** (C3-2): a grid reader reads the kernel only where the grid's lattice can, and deposits the
sample on the nodes elsewhere. Its oracles: B, the Poisson sum over the lattice of |the kernel's transform| |the
sample's characteristic function| at k = 2 pi m / h, m != 0, which bounds the estimate's mass error on the nodes and is
decided against 10%; and the tent deposit from its definition, the hat functions of each particle's cell divided by the
trapezoid weights. Pinned: the gate on both sides of B = 10%, its tail bound, the deposit on a graded axis, and the
solver and projector depositing.
"""

from __future__ import annotations

import logging
import warnings

import pytest

import numpy as np
from scipy.stats import gaussian_kde

from mfgarchon.alg.numerical import density_estimation as de
from mfgarchon.backends import create_backend
from mfgarchon.utils.numerical.particle import interpolation as kde_owner
from mfgarchon.utils.numerical.particle import kde_boundary as kb
from mfgarchon.utils.numerical.particle.interpolation import (
    KDE_LATTICE_ALIASING_TOL,
    deposit_on_nodes,
    estimate_kde_bandwidth,
    kde_factor,
    kde_kernel_covariance,
    kde_kernel_widths,
    kde_lattice_aliasing,
    kde_resolves_on_grid,
    kde_sample_has_spread,
)

BANDS = [0.5, "scott", "silverman"]
_SAMPLES = np.random.default_rng(2556)
CORRELATED = _SAMPLES.multivariate_normal([0.5, 0.5], [[0.01, 0.0094], [0.0094, 0.0144]], size=800)  # rho 0.78
ONE_D = _SAMPLES.normal(0.5, 0.05, 1000)


def _scipy_factor(bandwidth, n, d):
    """scipy's factor for a sample of n points in d dimensions (it depends on nothing else)."""
    return gaussian_kde(np.random.default_rng(0).normal(size=(d, n)), bw_method=bandwidth).factor


@pytest.mark.parametrize("d", [1, 2, 3])
@pytest.mark.parametrize("rule", ["scott", "silverman"])
def test_the_rules_are_scipys(rule, d):
    """`estimate_kde_bandwidth` is scipy's factor (its equivalence pin), and `kde_factor` reaches it for a rule and
    passes a positive number through."""
    sample = np.random.default_rng(d).normal(size=(300, d))
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
    want = gaussian_kde(np.atleast_2d(sample.T), bw_method=bandwidth).covariance
    np.testing.assert_allclose(kde_kernel_covariance(sample, bandwidth), want, rtol=1e-12)
    np.testing.assert_allclose(kde_kernel_widths(sample, bandwidth), np.sqrt(np.diag(want)), rtol=1e-12)


_t = np.random.default_rng(6).uniform(size=(60, 2))
_u = np.random.default_rng(25).uniform(size=1000)
NO_SPREAD = {
    "one point": np.full(50, 0.3),
    "copies of a point in 2-D": np.tile([0.3, 0.7], (50, 1)),
    "a line on the diagonal": np.column_stack([np.linspace(0, 1, 50), np.linspace(0, 1, 50)]),
    "a line off the axes": np.column_stack([0.2 + 0.3 * _t[:, 0], 0.7 - 0.5 * _t[:, 0]]),
    "a short line far from the origin": np.column_stack([1.0 + 1e-10 * _t[:, 0], 1.0 + 0.7e-10 * _t[:, 0]]),
    "a short line far from the origin on one axis": np.column_stack([1e6 + 1e-6 * _t[:, 0], 0.7e-6 * _t[:, 0]]),
    "a long line": np.column_stack([0.1 + 0.8 * _u, 0.2 + 0.55 * _u]),
    "a line on an axis, 100000 particles": np.column_stack([np.linspace(0, 1, 100000), np.full(100000, 0.53)]),
    "a plane in 3-D": np.array([0.1, 0.2, 0.3])
    + np.outer(_t[:, 0], [0.3, -0.2, 0.5])
    + np.outer(_t[:, 1], [0.1, 0.4, -0.3]),
    "n = d": np.array([[0.1, 0.2], [0.4, 0.9]]),
    "one particle": np.array([[0.2, 0.4]]),
}


@pytest.mark.parametrize("sample", NO_SPREAD.values(), ids=NO_SPREAD.keys())
def test_a_sample_with_no_spread_is_refused(sample):
    """No kernel exists on these. On a line or a plane `eigvalsh` returns a smallest eigenvalue of rounding size, a few
    1e-15 of the largest at most (5.6e-16 in the correlation's units, about the first particle, on the long line here
    of 1000 points). On a short line far from the origin it is of the size of the positions' rounding, and the floor
    refuses it; on a line far from the origin on one axis only, the thin direction carries half that axis's floor.
    `np.cov` gives the axis of 100000 particles at 0.53 a variance over its floor, from its mean's accumulated rounding,
    which in the correlation's units is a spread."""
    assert not kde_sample_has_spread(sample)
    with pytest.raises(ValueError, match="spread in every direction"):
        kde_kernel_covariance(sample, "scott")


_s = np.random.default_rng(7)
_a, _v = np.random.default_rng(10).uniform(-0.5, 0.5, 400), 1e-7 * np.random.default_rng(11).normal(size=400)
SMALL_SPREAD = {
    "1e6 + 1e-5 N(0, 1)": 1e6 + 1e-5 * _s.normal(size=400),
    "0.5 + 1e-6 N(0, I) in 2-D": 0.5 + 1e-6 * _s.normal(size=(400, 2)),
    "correlation 0.999": _s.multivariate_normal([0.5, 0.5], [[1.0, 0.999], [0.999, 1.0]], size=400),
    "thin: U(0, 1) by 0.5 + 1e-7 N(0, 1)": np.column_stack([_s.uniform(size=400), 0.5 + 1e-7 * _s.normal(size=400)]),
    "U(0, 1e-7) by U(0, 1)": np.column_stack([1e-7 * _s.uniform(size=400), _s.uniform(size=400)]),
    "thin, rotated 30 degrees": np.column_stack([0.5 + 0.866 * _a - 0.5 * _v, 0.5 + 0.5 * _a + 0.866 * _v]),
}


@pytest.mark.parametrize("sample", SMALL_SPREAD.values(), ids=SMALL_SPREAD.keys())
def test_a_small_spread_is_a_spread(sample):
    """The permission side of the refusal: a small spread, far from the origin, nearly on a line, 0.5 + 1e-7 N(0, 1)
    across U(0, 1) (a standard deviation 3.4e-7 of the other), on a domain 1e-7 wide, or 1e-7 thin across a rotated
    line, is scipy's kernel."""
    assert kde_sample_has_spread(sample)
    np.testing.assert_allclose(
        kde_kernel_covariance(sample, "scott"), gaussian_kde(np.atleast_2d(sample.T)).covariance, rtol=1e-9
    )


def test_an_axis_near_its_floor_leaves_a_thin_direction_without_it_alone():
    """The rounding floor is read per direction. x at 1e6 spreads 1e-11 of its magnitude, so its floor is 9.4e-5 of its
    variance, above the correlation's smallest eigenvalue, 9.8e-6; but that thin direction is y - z, which does not
    involve x, and the sample is scipy's kernel. #2819's review-3 correction took the largest axis's floor in every
    direction and refused it."""
    rng = np.random.default_rng(12)
    y = rng.normal(size=400)
    z = (1 - 1e-5) * y + np.sqrt(1 - (1 - 1e-5) ** 2) * rng.normal(size=400)
    sample = np.column_stack([1e6 + 1e-5 * rng.normal(size=400), y, z])
    assert kde_sample_has_spread(sample)
    want = gaussian_kde(sample.T).covariance
    scale = np.sqrt(np.outer(np.diag(want), np.diag(want)))
    np.testing.assert_allclose(kde_kernel_covariance(sample, "scott") / scale, want / scale, rtol=0, atol=1e-9)


def test_a_sample_with_a_non_finite_position_is_refused():
    """A NaN row (an absorbed particle in a `preserve_indices` history) is refused as such, not as a point."""
    sample = np.array([[0.1, 0.2], [np.nan, np.nan], [0.5, 0.4], [0.3, 0.9]])
    for owner in (kde_kernel_covariance, kde_kernel_widths):
        with pytest.raises(ValueError, match="finite positions"):
            owner(sample, "scott")


def test_the_widths_exist_on_a_line_and_not_at_a_point():
    """A reader of one isotropic width needs the kernel's per-axis widths, which a sample on a line has (its covariance
    is singular): the factor times each axis's standard deviation. A sample at one point has none."""
    line = NO_SPREAD["a line off the axes"]
    for bandwidth in BANDS:
        np.testing.assert_allclose(
            kde_kernel_widths(line, bandwidth),
            _scipy_factor(bandwidth, *line.shape) * np.std(line, axis=0, ddof=1),
            rtol=1e-12,
        )
    many = np.tile([0.3, 0.7], (20000, 1))  # `np.cov` gives each axis a variance over its floor
    for point in (NO_SPREAD["copies of a point in 2-D"], many, NO_SPREAD["one point"], NO_SPREAD["one particle"]):
        with pytest.raises(ValueError, match="no kernel width"):
            kde_kernel_widths(point, "scott")


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
    points = np.random.default_rng(1).multivariate_normal([0.5, 0.5], [[0.02, 0.0], [0.0, 0.02]], size=60)
    np.testing.assert_allclose(
        de.gaussian_kde_gpu_nd(CORRELATED, points, bandwidth, backend),
        gaussian_kde(CORRELATED.T, bw_method=bandwidth)(points.T),
        rtol=1e-10,
    )
    projector, grid = _projector(CORRELATED, 2)
    projected = projector(np.ones(len(CORRELATED)), bandwidth, None)
    nodes = np.stack(np.meshgrid(*grid.coordinates, indexing="ij"), axis=-1).reshape(-1, 2)
    np.testing.assert_allclose(
        projected.ravel(), gaussian_kde(CORRELATED.T, bw_method=bandwidth)(nodes.T), rtol=1e-10, atol=1e-300
    )


def _projector(particles, d):
    import functools
    import types

    from mfgarchon.geometry import TensorProductGrid
    from mfgarchon.geometry.boundary import no_flux_bc
    from mfgarchon.operators.interpolation.projection import GeometryProjector

    grid = TensorProductGrid(bounds=[(0.0, 1.0)] * d, Nx_points=[21] * d, boundary_conditions=no_flux_bc(dimension=d))
    fake = types.SimpleNamespace(
        hjb_geometry=grid,
        fp_geometry=types.SimpleNamespace(get_spatial_grid=lambda: particles.reshape(len(particles), -1)),
    )
    return functools.partial(GeometryProjector._project_particles_to_grid_kde, fake), grid


@pytest.mark.parametrize("sample", [ONE_D, CORRELATED], ids=["1-D", "2-D"])
def test_the_projector_refuses_an_unknown_rule(sample):
    projector, _grid = _projector(sample, np.atleast_2d(sample.T).shape[0])
    with pytest.raises(ValueError, match="Unknown method"):
        projector(np.ones(len(sample)), "foo", None)


@pytest.mark.parametrize("bandwidth", BANDS)
@pytest.mark.parametrize("where", ["wall", "corner"])
def test_reflection_has_no_slope_across_a_wall(where, bandwidth):
    """A correlated sample against the wall x = 0, or the corner (0, 0): the estimate's derivative normal to the wall,
    on the wall, is zero to the stencil's error, measured from inside with the second-order one-sided stencil. The same
    Sigma at the particles' images gave (audit session) +0.26, -0.44 and -1.64 in units of f / h on this geometry."""
    centre = [0.1, 0.5] if where == "wall" else [0.1, 0.1]
    sample = np.abs(np.random.default_rng(2).multivariate_normal(centre, [[0.01, 0.0094], [0.0094, 0.0144]], size=1500))
    h = np.sqrt(np.diag(kde_kernel_covariance(sample, bandwidth)))
    eps = 1e-5
    for y in (0.4, 0.5, 0.6) if where == "wall" else (0.0, 0.03, 0.08):
        f0, f1, f2 = kb.reflection_kde(
            sample, np.array([[0.0, y], [eps, y], [2 * eps, y]]), bandwidth, [(0.0, 1.0), (0.0, 1.0)], normalize=False
        )
        slope = (-3 * f0 + 4 * f1 - f2) / (2 * eps)
        assert abs(slope) / (f0 / h[0]) < 1e-3, (where, y, slope, f0)


@pytest.mark.parametrize("bandwidth", BANDS)
def test_an_integer_evaluation_point_is_read_as_a_float(bandwidth):
    """An integer point's image across a wall at 1.25 was built in the point's own dtype, so 2 * 1.25 - 1 read 1, and
    the estimate at (1, 1) came out 2.09 against 0.63."""
    sample = np.random.default_rng(5).uniform([0.0, 0.0], [1.25, 1.25], size=(4000, 2))
    points = np.array([[1, 1], [0, 1], [1, 0]])
    bounds = [(0.0, 1.25), (0.0, 1.25)]
    np.testing.assert_allclose(
        kb.reflection_kde(sample, points, bandwidth, bounds, normalize=False),
        kb.reflection_kde(sample, points.astype(float), bandwidth, bounds, normalize=False),
        rtol=1e-12,
    )


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


def test_renormalization_reads_each_axis_width_near_a_wall():
    """A uniform sample on [0, 1] x [0, 4], whose kernel is four times wider along y: one kernel width from the wall
    x = 0, and from the wall y = 0, the renormalized estimate is the density 1/4, since each axis's mass correction
    uses that axis's width. With the mean of the two widths the estimate comes out 31% high at the first and 10% low at
    the second. One point per call: on several, the estimator rescales its output to unit mass over them as over a
    grid."""
    sample = np.random.default_rng(4).uniform([0.0, 0.0], [1.0, 4.0], size=(50000, 2))
    h = kde_kernel_widths(sample, "scott")
    bounds = [(0.0, 1.0), (0.0, 4.0)]
    near_x = [kb.renormalization_kde(sample, np.array([[h[0], y]]), "scott", bounds)[0] for y in (1.5, 2.0, 2.5)]
    near_y = [kb.renormalization_kde(sample, np.array([[x, h[1]]]), "scott", bounds)[0] for x in (0.4, 0.5, 0.6)]
    assert np.mean(near_x) == pytest.approx(0.25, rel=0.05)
    assert np.mean(near_y) == pytest.approx(0.25, rel=0.05)


def _problem(nx=21, d=1, volatility=0.3):
    from mfgarchon import Conditions, MFGProblem, Model
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
    from mfgarchon.geometry import TensorProductGrid
    from mfgarchon.geometry.boundary import no_flux_bc

    grid = TensorProductGrid(bounds=[(0.0, 1.0)] * d, Nx_points=[nx] * d, boundary_conditions=no_flux_bc(dimension=d))
    model = Model(
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0)), volatility=volatility
    )
    conditions = Conditions(m_initial=lambda z: 1.0, u_terminal=lambda z: 0.0, T=0.1)
    return MFGProblem(model=model, domain=grid, conditions=conditions, Nt=4)


def _gaussian_density(nx, std=0.05, d=1):
    x = np.linspace(0.0, 1.0, nx)
    m1 = np.exp(-0.5 * ((x - 0.5) / std) ** 2)
    m = m1 if d == 1 else np.outer(m1, m1)
    return m / (np.trapezoid(m, x) if d == 1 else np.trapezoid(np.trapezoid(m, x), x))


@pytest.mark.parametrize("bandwidth", BANDS)
@pytest.mark.parametrize("d", [1, 2])
def test_the_beta_path_reads_the_kernel_widths(monkeypatch, d, bandwidth):
    """`kde_method="beta"` with a rule raised TypeError inside the estimator (#2804's crash row), and a number reached
    it as an absolute width. Every width the solver hands it is the kernel's for the sample it hands with it."""
    from mfgarchon.alg.numerical.fp_solvers.fp_particle import FPParticleSolver

    handed = []
    beta = kb.beta_kde

    def recording(particles, eval_points, bandwidth, bounds):
        handed.append((np.array(particles, dtype=float), np.atleast_1d(bandwidth)))
        return beta(particles, eval_points, bandwidth, bounds)

    monkeypatch.setattr(kb, "beta_kde", recording)
    nx = 21 if d == 1 else 11
    solver = FPParticleSolver(_problem(nx, d), num_particles=400, seed=3, kde_method="beta", kde_bandwidth=bandwidth)
    M = np.asarray(
        solver.solve_fp_system(
            _gaussian_density(nx, 0.1, d), drift_field=np.zeros((5,) + (nx,) * d), show_progress=False
        )
    )
    assert np.isfinite(M).all()
    assert len(handed) == 5, len(handed)
    for particles, widths in handed:
        want = gaussian_kde(np.atleast_2d(particles.T), bw_method=bandwidth).covariance
        np.testing.assert_allclose(widths, np.sqrt(np.diag(want)), rtol=1e-12)


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


@pytest.mark.parametrize("bandwidth", BANDS)
def test_the_particle_result_queries_with_the_solvers_bandwidth(monkeypatch, bandwidth):
    """`FPParticleResult` holds the solver's `kde_bandwidth`, a factor or a rule. The solver handed it None for a rule
    (the query's own Scott), and the result handed the query a factor as an absolute width. The query is handed the mean
    of the kernel's per-axis widths on that timestep's sample, which exist on a sample on a line."""
    from mfgarchon.alg.numerical.fp_solvers import particle_result
    from mfgarchon.alg.numerical.fp_solvers.fp_particle import FPParticleSolver
    from mfgarchon.alg.numerical.fp_solvers.particle_result import FPParticleResult

    widths = []
    real = particle_result.ParticleDensityQuery

    def recording(particles, bandwidth=None, bandwidth_rule="fixed", **kwargs):
        widths.append((np.asarray(particles, dtype=float), bandwidth, bandwidth_rule))
        return real(particles, bandwidth=bandwidth, bandwidth_rule=bandwidth_rule, **kwargs)

    monkeypatch.setattr(particle_result, "ParticleDensityQuery", recording)
    solver = FPParticleSolver(_problem(), num_particles=500, seed=3, density_mode="hybrid", kde_bandwidth=bandwidth)
    result = solver.solve_fp_system(_gaussian_density(21), drift_field=np.zeros((5, 21)), show_progress=False)
    assert result.bandwidth == bandwidth
    for t in (0, 2):
        result.query_density(np.array([[0.5]]), timestep=t, method="kernel")
    line = NO_SPREAD["a line off the axes"]
    FPParticleResult(
        np.zeros((1, 11, 11)), np.array([0.0]), particle_history=[line], bandwidth=bandwidth
    ).query_density(np.array([[0.3, 0.45]]), timestep=0, method="kernel")
    assert len(widths) == 3
    for sample, width, rule in widths:
        n, d = sample.shape
        assert rule == "fixed"
        assert width == pytest.approx(
            _scipy_factor(bandwidth, n, d) * np.mean(np.std(sample, axis=0, ddof=1)), rel=1e-12
        )


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


def _tent_deposit(particles, axes):
    """The tent deposit from its definition, the oracle for the solver's: node (i, j) takes each particle's
    prod_axis max(0, 1 - |x - node| / h) of its mass 1/N, divided by the node's trapezoid weight (half a cell on a wall,
    a cell inside, per axis)."""
    tents = [
        np.maximum(0.0, 1.0 - np.abs(particles[:, k, None] - a[None, :]) / (a[1] - a[0])) for k, a in enumerate(axes)
    ]
    weights = [
        np.where((np.arange(len(a)) == 0) | (np.arange(len(a)) == len(a) - 1), 0.5, 1.0) * (a[1] - a[0]) for a in axes
    ]
    return np.einsum("pi,pj->ij", *tents) / len(particles) / np.outer(*weights)


@pytest.mark.parametrize("kde_method", ["reflection", "standard", "renormalization", "beta", "cic"])
def test_a_sample_on_a_line_is_deposited_on_the_nodes_by_every_kde_method(kde_method, mfg_caplog):
    """Particles that stay on the line x = 0.53 (no diffusion) have no kernel, and every kde_method gives their tent
    deposit on the grid's nodes, split between the nodes 0.5 and 0.6 (#2556). A kernel method records the levels it
    deposited and the solve logs them once; cic deposits every level and records none. At 654848bc the kernel methods
    refused this sample by name (C3-1) and cic read a cell-centred deposit at the nodes (mass n/(n-1))."""
    import logging

    from mfgarchon.alg.numerical.fp_solvers.fp_particle import FPParticleSolver

    particles = np.column_stack([np.full(300, 0.53), np.random.default_rng(8).uniform(size=300)])
    solver = FPParticleSolver(_problem(11, 2, volatility=0.0), num_particles=300, seed=5, kde_method=kde_method)
    with mfg_caplog.at_level(logging.WARNING, logger="mfgarchon.alg.numerical.fp_solvers.fp_particle"):
        M = np.asarray(
            solver.solve_fp_system(
                M_initial=None,
                drift_field=lambda t, x, m: np.zeros_like(x),
                initial_particles=particles,
                show_progress=False,
            )
        )
    axes = [np.linspace(0.0, 1.0, 11)] * 2
    want = _tent_deposit(particles, axes)
    w = np.outer(*[np.where((np.arange(11) == 0) | (np.arange(11) == 10), 0.05, 0.1)] * 2)
    assert np.sum(want * w) == pytest.approx(1.0, abs=1e-12)
    assert np.sum(want * w * axes[0][:, None]) == pytest.approx(0.53, abs=1e-12)
    for slice_ in M[1:]:
        np.testing.assert_allclose(slice_, want, rtol=1e-12, atol=1e-12)
        assert slice_[5].sum() > 0
        assert slice_[6].sum() > 0
    deposited = [r for r in mfg_caplog.messages if "deposited on the grid's nodes" in r]
    if kde_method == "cic":
        assert solver.deposited_levels == []
        assert deposited == []
    else:
        assert solver.deposited_levels == list(range(1, len(M)))
        assert len(deposited) == 1, mfg_caplog.messages


@pytest.mark.parametrize("kde_method", ["reflection", "standard", "renormalization"])
def test_a_thin_domain_is_a_rescaled_unit_one(kde_method):
    """Scaling an axis leaves scipy's KDE equivariant, and so the spread predicate must leave a sample alone: on
    [0, 1e-7] x [0, 1] the default path's density is the unit square's, with the same particles rescaled, times 1e7.
    #2819's first bound read the eigenvalue ratio in absolute units and refused this sample as one on a line."""
    from mfgarchon import Conditions, MFGProblem, Model
    from mfgarchon.alg.numerical.fp_solvers.fp_particle import FPParticleSolver
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
    from mfgarchon.geometry import TensorProductGrid
    from mfgarchon.geometry.boundary import no_flux_bc

    unit = np.random.default_rng(9).uniform(size=(600, 2))
    densities = {}
    for width in (1.0, 1e-7):
        grid = TensorProductGrid(
            bounds=[(0.0, width), (0.0, 1.0)], Nx_points=[11, 11], boundary_conditions=no_flux_bc(dimension=2)
        )
        model = Model(hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0)), volatility=0.0)
        conditions = Conditions(m_initial=lambda z, width=width: 1.0 / width, u_terminal=lambda z: 0.0, T=0.1)
        solver = FPParticleSolver(
            MFGProblem(model=model, domain=grid, conditions=conditions, Nt=4),
            num_particles=600,
            seed=5,
            kde_method=kde_method,
        )
        densities[width] = np.asarray(
            solver.solve_fp_system(
                M_initial=None,
                drift_field=lambda t, x, m: np.zeros_like(x),
                initial_particles=unit * [width, 1.0],
                show_progress=False,
            )
        )[1:]
    np.testing.assert_allclose(densities[1e-7] * 1e-7, densities[1.0], rtol=1e-12, atol=1e-12)


def _poisson_bound(sample, bandwidth, h):
    """B from its definition: sum over the lattice m != 0 of |scipy's kernel transform| |the sample's characteristic
    function| at k = 2 pi m / h, in shells of max |m_i| = r until a shell adds under 1e-16."""
    H = gaussian_kde(sample.T, bw_method=bandwidth).covariance
    d = sample.shape[1]
    total = 0.0
    for r in range(1, 400):
        span = np.arange(-r, r + 1)
        m = np.array(np.meshgrid(*([span] * d), indexing="ij")).reshape(d, -1).T
        k = 2.0 * np.pi * m[np.abs(m).max(axis=1) == r] / h
        g = np.exp(-0.5 * np.einsum("ij,jk,ik->i", k, H, k))
        total += float(np.sum(g * np.abs(np.exp(1j * (sample @ k.T)).mean(axis=0))))
        if g.sum() < 1e-16:
            return total
    raise AssertionError("the oracle's sum did not converge")


_WIDE = 0.5 + 0.1 * np.random.default_rng(3).standard_normal((3000, 2))
_NARROW = np.array([0.53, 0.47]) + 0.005 * np.random.default_rng(3).standard_normal((300, 2))
_ANISO = np.random.default_rng(0).multivariate_normal([0.53, 0.47], np.diag([0.05, 0.2]) ** 2, size=300)


@pytest.mark.parametrize(
    ("sample", "bandwidth", "resolved"),
    [(_WIDE, "scott", True), (_NARROW, "scott", False), (_ANISO, "scott", True), (_ANISO, 0.15, False)],
    ids=["wide", "narrow off a node", "anisotropic", "anisotropic, factor 0.15"],
)
def test_the_gate_reads_the_aliasing_bound(sample, bandwidth, resolved):
    """The gate's decision is B's against 10%, with B from its definition. A value at or under 10% bounds B from above
    (the partial sum plus the tail bound), and one over it from below (a partial sum)."""
    bound = _poisson_bound(sample, bandwidth, 0.1)
    assert (bound <= KDE_LATTICE_ALIASING_TOL) is resolved
    assert kde_resolves_on_grid(sample, bandwidth, [0.1, 0.1]) is resolved
    value = kde_lattice_aliasing(sample, bandwidth, [0.1, 0.1])
    if resolved:
        assert bound - 1e-12 <= value <= KDE_LATTICE_ALIASING_TOL
    else:
        assert KDE_LATTICE_ALIASING_TOL < value <= bound + 1e-12


def test_the_gate_decides_either_side_of_its_tolerance():
    """Factors either side of the one where B is 10%, found by bisection on the oracle, are decided as B is."""
    lo, hi = 0.05, 3.0
    for _ in range(50):
        mid = np.sqrt(lo * hi)
        lo, hi = (lo, mid) if _poisson_bound(_ANISO, mid, 0.1) <= KDE_LATTICE_ALIASING_TOL else (mid, hi)
    for factor, resolved in ((hi * (1 + 1e-6), True), (lo * (1 - 1e-6), False)):
        assert (_poisson_bound(_ANISO, factor, 0.1) <= KDE_LATTICE_ALIASING_TOL) is resolved
        assert kde_resolves_on_grid(_ANISO, factor, [0.1, 0.1]) is resolved


def test_the_tail_bound_is_over_the_kernels_tail():
    """The tail the gate adds is at least the kernel transform's sum outside the cube max |m_i| <= r, on a correlated,
    anisotropic kernel (correlation 0.9, widths 0.3 and 0.05 cells), summed here over the lattice to |m_i| <= 120."""
    h = np.array([0.1, 0.1])
    H = np.array([[1.0, 0.9], [0.9, 1.0]]) * np.outer([0.03, 0.005], [0.03, 0.005])
    span = np.arange(-120, 121)
    m = np.array(np.meshgrid(span, span, indexing="ij")).reshape(2, -1).T
    k = 2.0 * np.pi * m / h
    g = np.exp(-0.5 * np.einsum("ij,jk,ik->i", k, H, k))
    assert g[np.abs(m).max(axis=1) == 120].max() < 1e-30
    a = kde_owner._tail_rates(H, h)
    inner = np.ones(2)
    for r in range(1, 30):
        inner += 2.0 * np.exp(-0.5 * a * r**2)
        assert kde_owner._kernel_tail_bound(a, inner, r) >= float(g[np.abs(m).max(axis=1) > r].sum())


def _trapezoid(axis):
    w = np.empty_like(axis)
    w[1:-1] = (axis[2:] - axis[:-2]) / 2
    w[0], w[-1] = (axis[1] - axis[0]) / 2, (axis[-1] - axis[-2]) / 2
    return w


def _hats(x, axis):
    """(particles, nodes): each node's hat function at x, rising over the cell to its left and falling over the one to
    its right."""
    out = np.zeros((len(x), len(axis)))
    for i in range(len(axis)):
        if i > 0:
            on = (x >= axis[i - 1]) & (x <= axis[i])
            out[on, i] = (x[on] - axis[i - 1]) / (axis[i] - axis[i - 1])
        if i < len(axis) - 1:
            on = (x >= axis[i]) & (x <= axis[i + 1])
            out[on, i] = np.maximum(out[on, i], (axis[i + 1] - x[on]) / (axis[i + 1] - axis[i]))
    return out


def test_the_deposit_is_the_tents_of_a_graded_grid():
    """On a graded axis beside a uniform one, particles on nodes, on walls, at a corner and inside are deposited by their
    cells' hat functions over the trapezoid weights: mass 1 and the sample's mean on those weights. A position 1e-12
    outside the box, and an axis of one node, are refused by name."""
    graded = np.concatenate([np.linspace(0.0, 0.2, 6), np.linspace(0.3, 1.0, 8)])
    uniform = np.linspace(0.0, 1.0, 6)
    rng = np.random.default_rng(5)
    x = np.concatenate([graded[[0, 3, 5, 6, 13]], [0.0, 1.0, 0.25, 0.97], rng.uniform(0, 1, 40)])
    y = np.concatenate([uniform[[0, 2, 5, 1, 4]], [1.0, 1.0, 0.0, 0.33], rng.uniform(0, 1, 40)])
    weights = np.outer(_trapezoid(graded), _trapezoid(uniform))
    want = np.einsum("pi,pj->ij", _hats(x, graded), _hats(y, uniform)) / len(x) / weights
    got = deposit_on_nodes(np.column_stack([x, y]), [graded, uniform], weights)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=0)
    assert np.sum(got * weights) == pytest.approx(1.0, abs=1e-13)
    assert np.sum(got * weights * graded[:, None]) == pytest.approx(x.mean(), abs=1e-13)
    assert np.sum(got * weights * uniform[None, :]) == pytest.approx(y.mean(), abs=1e-13)
    with pytest.raises(ValueError, match="outside the grid on axis 0"):
        deposit_on_nodes(np.array([[1.0 + 1e-12, 0.5]]), [graded, uniform], weights)
    with pytest.raises(ValueError, match="at least two strictly increasing nodes"):
        deposit_on_nodes(np.array([0.0]), [np.array([0.0])], np.ones(1))


def test_the_projector_deposits_a_sample_its_grid_cannot_read():
    """The particle-to-grid projector on a uniform grid: the narrow sample is its tent deposit, mass 1 on the grid's
    weights, where the KDE at the nodes had mass 0; the wide one is still scipy's KDE at the nodes."""
    from mfgarchon.geometry import TensorProductGrid
    from mfgarchon.geometry.boundary import no_flux_bc
    from mfgarchon.geometry.implicit.point_cloud import PointCloudGeometry
    from mfgarchon.operators.interpolation.projection import GeometryProjector

    grid = TensorProductGrid(bounds=[(0.0, 1.0)] * 2, Nx_points=[11, 11], boundary_conditions=no_flux_bc(dimension=2))
    axis = np.linspace(0.0, 1.0, 11)
    weights = np.outer(_trapezoid(axis), _trapezoid(axis))
    for sample in (_NARROW, np.clip(_WIDE, 0.0, 1.0)):
        projector = GeometryProjector(grid, PointCloudGeometry(sample), projection_method="kde")
        got = np.asarray(projector.project_fp_to_hjb(np.ones(len(sample)), bandwidth="scott"))
        if sample is _NARROW:
            want = np.einsum("pi,pj->ij", _hats(sample[:, 0], axis), _hats(sample[:, 1], axis)) / len(sample) / weights
            assert (
                np.sum(
                    gaussian_kde(sample.T)(np.array(np.meshgrid(axis, axis, indexing="ij")).reshape(2, -1))
                    * weights.ravel()
                )
                < 1e-10
            )
        else:
            want = gaussian_kde(sample.T)(np.array(np.meshgrid(axis, axis, indexing="ij")).reshape(2, -1)).reshape(
                11, 11
            )
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)
