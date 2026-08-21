"""Issue #1981, and a correction to what it reported.

#1981 said `form=` is inert under `scheme="upwind"`, so the public `conservative=` flag "silently
does nothing". Measured, that is the symptom rather than the defect, and it is narrower than the
defect in one direction and wider in another:

- The OPERATOR honours `form=` under upwind. `__call__` / `@` separate `divergence` from `gradient`
  by 48.97 on a linear velocity, and the public `conservative=` flag reaches it: 9.13 through
  `grid.get_advection_operator(...) @ m`.
- The MATRIX does not represent the operator at all under upwind -- not merely form-insensitively.
  Against `__call__` on a smooth field it is off by **27.31 even for a CONSTANT velocity**, where
  the two forms coincide and there is no form to lose.

The cause is that upwinding chooses its difference direction from the local sign of the field, so
the operator is **nonlinear** and has no matrix; `as_scipy_sparse` probes it with unit vectors and
linearises it around impulses. `centered` is exact (0.000000), so the extraction machinery is sound.

The method's docstring already said "❌ Do NOT use for implicit solver Jacobians". A recommendation
the code does not enforce is what this repo keeps finding; it raises now.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.operators.differential.advection import AdvectionOperator

_N = 9
_H = 1.0 / (_N - 1)
_X = np.linspace(0.0, 1.0, _N)
_M = np.sin(2 * np.pi * _X) + 2.0


def _op(velocity, scheme, form):
    return AdvectionOperator(
        velocity_field=np.asarray(velocity)[None, :], spacings=[_H], field_shape=(_N,), scheme=scheme, form=form
    )


@pytest.mark.parametrize("form", ["divergence", "gradient"])
@pytest.mark.parametrize("velocity", [np.ones(_N), 1 + 2 * _X], ids=["constant", "linear"])
def test_the_upwind_matrix_is_refused(form, velocity):
    with pytest.raises(NotImplementedError) as exc:
        _op(velocity, "upwind", form).as_scipy_sparse()
    text = str(exc.value)
    assert "NONLINEAR" in text, "the message must say WHY there is no matrix, not just that there isn't"
    assert "centered" in text, "it must name the scheme that does work"
    assert "1981" in text


@pytest.mark.parametrize("form", ["divergence", "gradient"])
@pytest.mark.parametrize("velocity", [np.ones(_N), 1 + 2 * _X], ids=["constant", "linear"])
def test_the_centered_matrix_is_exact_against_the_operator(form, velocity):
    """Control, and the reason the refusal is scoped to upwind rather than to the method."""
    op = _op(velocity, "centered", form)
    a = op.as_scipy_sparse().toarray()
    assert np.abs(a @ _M - np.asarray(op(_M.ravel()))).max() == pytest.approx(0.0, abs=1e-12)


def test_the_operator_honours_the_form_under_upwind():
    """The correction to #1981's headline. `form=` is not inert -- the extraction lost it."""
    div = np.asarray(_op(1 + 2 * _X, "upwind", "divergence")(_M.ravel()))
    grad = np.asarray(_op(1 + 2 * _X, "upwind", "gradient")(_M.ravel()))
    assert np.abs(div - grad).max() > 1.0, (
        "the two forms must differ under upwind; if they no longer do, the operator has acquired "
        "the defect the matrix had"
    )


def test_the_public_conservative_flag_reaches_the_operator():
    """`conservative=` maps to `form=` and is honoured through `@`, which uses `_matvec`, not the
    matrix. #1981 reported it as silently doing nothing; measured, it does 9.13 worth."""
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[_N], boundary_conditions=no_flux_bc(dimension=1))
    v = (1 + 2 * _X)[None, :]
    a = np.asarray(grid.get_advection_operator(v, scheme="upwind", conservative=True) @ _M.ravel())
    b = np.asarray(grid.get_advection_operator(v, scheme="upwind", conservative=False) @ _M.ravel())
    assert np.abs(a - b).max() > 1.0


def test_no_production_caller_takes_the_refused_path():
    """The refusal is safe because nothing in the package extracts an advection matrix: every
    `as_scipy_sparse` call site in `mfgarchon/` is on `LaplacianOperator`. Pinned, so adding an
    advection one is a decision rather than an accident."""
    import pathlib

    import mfgarchon

    root = pathlib.Path(mfgarchon.__file__).parent
    # The defining module is excluded, and that exclusion is the instrument being fixed rather than
    # widened: a first version looked back six lines for "AdvectionOperator" and flagged the
    # refusal's own error message, inside advection.py -- matching the DESCRIPTION of the thing
    # instead of a call to it. Only a `.as_scipy_sparse()` whose receiver was constructed nearby as
    # an AdvectionOperator counts.
    defining = root / "operators" / "differential" / "advection.py"
    offenders = []
    for path in root.rglob("*.py"):
        if path == defining:
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            code = line.split("#", 1)[0]
            if "as_scipy_sparse()" not in code or ">>>" in line:
                continue
            window = "\n".join(lines[max(0, i - 6) : i + 1])
            if "AdvectionOperator(" in window:
                offenders.append(f"{path.relative_to(root)}:{i + 1}")
    assert offenders == [], (
        f"a production caller now extracts an advection matrix: {offenders}. Under scheme='upwind' "
        f"that path raises (#1981); under 'centered' it is exact. Confirm which scheme it uses."
    )
