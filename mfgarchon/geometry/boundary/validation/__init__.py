"""
Boundary Condition Validation Tools.

Developer tools for validating BC implementations, not user-facing runtime checks.

``spectral_diagnostics`` reports measured spectral quantities of a discretized operator and
returns **no stability verdict**. It replaces ``check_gks_stability``, which claimed to implement
GKS (Gustafsson-Kreiss-Sundström) analysis, did not, and whose three ``pde_type`` branches were
each defective -- one of them a tautology that reported every operator stable (Issue #1859).
See ``spectral.py`` for what the reported quantities do and do not prove.

Boundary-condition correctness is validated in this repository by exactness against an external
field, discrete conservation, and measured order of accuracy under refinement -- which is what
the field does, and what catches the defects that actually occur.

Created: 2026-01-18 (Issue #593 Phase 4.2). Rescoped 2026-08-08 (Issue #1859).
"""

from mfgarchon.geometry.boundary.validation.spectral import (
    SpectralDiagnostics,
    spectral_diagnostics,
)

__all__ = [
    "SpectralDiagnostics",
    "spectral_diagnostics",
]
