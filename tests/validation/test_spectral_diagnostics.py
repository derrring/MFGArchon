"""Spectral diagnostics for BC-applied operators (Issue #1859).

Replaces test_gks_conditions.py. That file had ten tests, all of which fed a VALID operator and
asserted ``stable is True``; none fed a known-bad operator and asserted the checker said bad. A
criterion that cannot fail passes such a suite indistinguishably from a correct one, and that is
exactly what happened -- the hyperbolic branch compared max|Im(lambda)| against 10*max|lambda|,
which holds for every complex number.

Two things are pinned here that the old file pinned neither of:

1. **An external oracle.** The Dirichlet Laplacian has the closed-form spectrum
   ``lambda_j = -(4/h^2) sin^2(j*pi*h/2)``, so the measured numbers are checked against
   mathematics rather than against another call into the library. (This is the pattern
   scikit-fem uses for its Orr-Sommerfeld example, pinning against a published table -- and its
   first reference eigenvalue has a POSITIVE growth rate, i.e. it pins instability *detection*.)
2. **The theorem the module claims.** ``||exp(tL)||_2 <= exp(t*omega)`` for the numerical
   abscissa omega is asserted directly against a computed matrix exponential, including on a
   non-normal operator where the spectral abscissa is negative and the propagator norm still
   grows by orders of magnitude. That case is the reason this module reports numbers instead of
   a boolean: the old API would have called it stable.
"""

from __future__ import annotations

import pytest

import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix, diags, eye

from mfgarchon.geometry.boundary.validation import SpectralDiagnostics, spectral_diagnostics


def _dirichlet_laplacian(n: int) -> csr_matrix:
    """1D Dirichlet Laplacian. Spectrum is closed-form; see _analytic_spectrum."""
    h = 1.0 / (n + 1)
    return (diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(n, n)) / h**2).tocsr()


def _analytic_spectrum(n: int) -> np.ndarray:
    """lambda_j = -(4/h^2) sin^2(j*pi*h/2), j = 1..n. The external oracle."""
    h = 1.0 / (n + 1)
    j = np.arange(1, n + 1)
    return -(4.0 / h**2) * np.sin(j * np.pi * h / 2.0) ** 2


class TestAgainstAnAnalyticOracle:
    def test_full_spectrum_matches_the_closed_form(self):
        n = 60
        diag = spectral_diagnostics(_dirichlet_laplacian(n), full_spectrum=True)
        exact = np.sort(_analytic_spectrum(n))
        got = np.sort(diag.eigenvalues.real)
        np.testing.assert_allclose(got, exact, rtol=1e-10)
        assert diag.full_spectrum
        assert len(diag.eigenvalues) == n

    def test_spectral_abscissa_matches_the_closed_form_on_the_sparse_path(self):
        """The truncated solve must still find the true maximum -- that is what 'LR' is for."""
        n = 400
        diag = spectral_diagnostics(_dirichlet_laplacian(n))
        assert not diag.full_spectrum, "expected the truncated path at this size"
        assert diag.spectral_abscissa == pytest.approx(_analytic_spectrum(n).max(), rel=1e-8)


class TestItCanDetectGrowth:
    """The must-detect controls. Their absence is why a tautology survived for months."""

    @pytest.mark.parametrize("n", [40, 400])
    @pytest.mark.parametrize("delta", [0.1, 5.0])
    def test_a_growing_operator_has_a_positive_spectral_abscissa(self, n, delta):
        """Shifted so the abscissa lands exactly on +delta, at both sizes.

        The shift is measured from the analytic spectrum rather than assumed: the Dirichlet
        Laplacian's largest eigenvalue tends to -pi^2, not to 0, so a small additive delta leaves
        it comfortably negative. (The Neumann operator is the one with an eigenvalue at 0 -- its
        constant null vector. Confusing the two is what made the first draft of this test fail.)
        """
        top = _analytic_spectrum(n).max()
        operator = (_dirichlet_laplacian(n) + (delta - top) * eye(n)).tocsr()
        diag = spectral_diagnostics(operator, description=f"laplacian shifted to +{delta}")
        assert diag.spectral_abscissa > 0
        assert diag.spectral_abscissa == pytest.approx(delta, rel=1e-6)

    def test_a_dissipative_operator_has_a_negative_spectral_abscissa(self):
        """The converse, so the detector is not simply always-positive."""
        diag = spectral_diagnostics(_dirichlet_laplacian(200))
        assert diag.spectral_abscissa < 0


class TestTheNumericalAbscissaTheorem:
    """omega = lambda_max((L + L^H)/2) must satisfy ||exp(tL)||_2 <= exp(t*omega)."""

    def test_equals_the_spectral_abscissa_for_a_symmetric_operator(self):
        """For symmetric L the two coincide -- a consistency check on the implementation."""
        diag = spectral_diagnostics(_dirichlet_laplacian(50), full_spectrum=True)
        assert diag.numerical_abscissa == pytest.approx(diag.spectral_abscissa, rel=1e-10)

    def test_bounds_the_propagator_norm_on_a_non_normal_operator(self):
        """The case that makes a `stable: bool` on max Re(lambda) wrong.

        L = [[-1, 30], [0, -2]] has eigenvalues -1 and -2, so the spectral abscissa is -1 < 0 and
        the retired API would have reported it STABLE. Its propagator nonetheless grows by a
        large factor before decaying, and the numerical abscissa sees that while the spectral
        abscissa cannot.
        """
        operator = csr_matrix(np.array([[-1.0, 30.0], [0.0, -2.0]]))
        diag = spectral_diagnostics(operator, full_spectrum=True)

        assert diag.spectral_abscissa == pytest.approx(-1.0, abs=1e-12)
        assert diag.numerical_abscissa > 0, "numerical abscissa must see the transient growth"

        dense = operator.toarray()
        peak = max(np.linalg.norm(expm(t * dense), 2) for t in np.linspace(0.0, 5.0, 200))
        assert peak > 5.0, f"control is not actually transiently growing (peak {peak:.2f})"

        for t in np.linspace(0.0, 5.0, 50):
            norm = np.linalg.norm(expm(t * dense), 2)
            bound = np.exp(t * diag.numerical_abscissa)
            assert norm <= bound * (1 + 1e-9), f"theorem violated at t={t}: {norm} > {bound}"

    def test_bounds_the_propagator_norm_on_a_dissipative_operator(self):
        """Same theorem where it is tight: omega <= 0 must imply the norm never exceeds 1."""
        operator = _dirichlet_laplacian(30)
        diag = spectral_diagnostics(operator, full_spectrum=True)
        assert diag.numerical_abscissa < 0

        dense = operator.toarray()
        for t in (0.0, 1e-5, 1e-4, 1e-3):
            assert np.linalg.norm(expm(t * dense), 2) <= 1.0 + 1e-9


class TestItReportsWhatItActuallySaw:
    """A truncated solve must not be presentable as a whole-spectrum statement."""

    def test_truncated_solve_declares_itself_truncated(self):
        n = 400
        diag = spectral_diagnostics(_dirichlet_laplacian(n))
        assert not diag.full_spectrum
        assert len(diag.eigenvalues) < n
        assert "of 400" in str(diag)

    def test_full_spectrum_declares_itself_complete(self):
        diag = spectral_diagnostics(_dirichlet_laplacian(120), full_spectrum=True)
        assert diag.full_spectrum
        assert len(diag.eigenvalues) == 120
        assert "all 120" in str(diag)

    def test_full_spectrum_refuses_above_the_cap_rather_than_truncating(self):
        with pytest.raises(ValueError, match="full_spectrum"):
            spectral_diagnostics(_dirichlet_laplacian(300), full_spectrum=True, max_dense_size=200)

    def test_min_real_part_from_a_truncated_solve_is_not_the_true_minimum(self):
        """Documents the limit rather than hiding it: 'LR' samples one end, so min is not global.

        This is the trap the retired elliptic branch fell into -- it asked whether ALL eigenvalues
        share a sign, from a one-ended sample.
        """
        n = 400
        truncated = spectral_diagnostics(_dirichlet_laplacian(n))
        exact_min = _analytic_spectrum(n).min()
        assert not truncated.full_spectrum
        assert truncated.min_real_part > exact_min * 0.5, (
            "expected the truncated subset to miss the far end of the spectrum entirely"
        )


class TestNoVerdictIsExposed:
    def test_result_carries_no_boolean_stability_field(self):
        """The retired API's `stable: bool` must not come back by habit."""
        diag = spectral_diagnostics(_dirichlet_laplacian(30), full_spectrum=True)
        assert isinstance(diag, SpectralDiagnostics)
        fields = set(SpectralDiagnostics.__dataclass_fields__)
        assert not (fields & {"stable", "is_stable", "verdict", "ok", "passed"}), (
            f"a verdict field reappeared: {fields}"
        )

    def test_str_reports_numbers_and_what_they_prove(self):
        diag = spectral_diagnostics(_dirichlet_laplacian(30), full_spectrum=True)
        text = str(diag)
        assert "spectral abscissa" in text
        assert "numerical abscissa" in text
        assert "STABLE" not in text.upper().replace("INSTABILITY", "")
