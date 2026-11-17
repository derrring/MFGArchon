"""The one owner of a non-finite read during the solve (#2975).

A value a solver reads while it solves -- an array handed to it at its door, the drift and the source a time step
advances, a term of the residual a Newton step assembles, a coefficient or a boundary value -- is refused where it is
read when it holds a NaN or an infinity, naming what was read and where. Without this, such a value reached the solution
and came back as a finite wrong answer, a non-finite one returned without an error, or an error that named the time
step, the CFL condition or a linear solve.

`refuse_a_non_finite_read` is the one function that decides it, and `NonFiniteReadError` the one error it raises. Every
reader calls it where the value enters the solution, not around every return of the callable it came from: a value a
step computes and then discards, such as a Dirichlet row its boundary condition overwrites, is not judged
(docs/user/CONVENTIONS.md, "What the solve refuses as non-finite").

Construction's own reads are #2974's owner, `utils.validation.functions._refuse_a_non_finite_source`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping


class NonFiniteReadError(ValueError):
    """A value a solver reads during the solve holds a NaN or an infinity (#2975).

    A `ValueError`, since the input is at fault and not the solver, so a caller that catches `ValueError` still catches
    it; and its own type, so that a caller, and #2644's policy for inner solves, can tell it from non-convergence
    (`ConvergenceError`). Raised only by `refuse_a_non_finite_read`.
    """


def refuse_a_non_finite_read(solver: str, where: str, values: Mapping[str, Any]) -> None:
    """Refuse the first of ``values`` that holds a NaN or an infinity, naming it, the solver and where it was read.

    ``values`` maps each name to what the solver reads under it, in the order they are judged; a ``None`` is skipped,
    since an argument not handed is not read. Each is read as numpy reads it as floats, so a one-element array is
    judged as its number and an object array's ``None`` entry is NaN. ``where`` says where in the solve the read
    happens, for example "its door" or "time step 3 of 5, t=0.3".
    """
    for name, value in values.items():
        if value is None:
            continue
        array = np.asarray(value, dtype=float)
        finite = np.isfinite(array)
        if bool(np.all(finite)):
            continue
        bad = int(array.size - np.count_nonzero(finite))
        raise NonFiniteReadError(
            f"{solver}: {name} is non-finite where it reads it: {bad} of {array.size} values at {where} (#2975)."
        )
