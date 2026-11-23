"""The one owner of a non-finite read during the solve (#2975).

A non-finite value a solver reads makes the solve raise at the step whose result it reaches, or at the decision it
enters, naming the first non-finite value that step read and where. A value the step reads and discards is not judged:
a Dirichlet row its boundary condition overwrites, a repeated periodic endpoint the step drops. An entry is discarded
when the step's result does not depend on it for any value, NaN included: a zero weight discards a finite value but not
a NaN, so a NaN multiplied by a boundary value of 0 reaches the result and is named. Without this, such a value reached
the solution and came back as a finite wrong answer, a non-finite one returned without an error, or an error that named
the time step, the CFL condition or a linear solve (docs/user/CONVENTIONS.md, "What the solve refuses as non-finite").

`refuse_a_non_finite_read` is the one function that decides it, and `NonFiniteReadError` the one error it raises. A
reader calls it on its failure path, naming the read behind a non-finite result, or at a decision a read value enters;
`refuse_the_first_read_that_entered` is the failure path of a step that can re-run on what it read, and
`run_a_naming_read` runs each read an HJB-FDM failure-path naming makes that the step did not make on those inputs. A
few reads are refused where they are read, because what they decide consumes every entry or because they enter by definition: the
arrays a door is handed as the solution's own row, read as the solver will hold it, a boundary value, a volatility
override, the iterator's initial mass.

Construction's own reads are #2974's owner, `utils.validation.functions._refuse_a_non_finite_source`.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


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
    happens, for example "its door" or "time step 3 of 5, t=0.3". The message gives how many entries are non-finite
    and the first of them, ``nan``, ``inf`` or ``-inf``.
    """
    for name, value in values.items():
        if value is None:
            continue
        array = np.asarray(value, dtype=float)
        finite = np.isfinite(array)
        if bool(np.all(finite)):
            continue
        bad = int(array.size - np.count_nonzero(finite))
        first = float(array[~finite].flat[0])
        raise NonFiniteReadError(
            f"{solver}: {name} is non-finite where it reads it: {bad} of {array.size} values, the first {first}, at "
            f"{where} (#2975)."
        )


def run_a_naming_read(read: Callable[[], Any]) -> Any | None:
    """Run one read a failure-path naming makes, and return what it read, or ``None`` when the read decides nothing.

    A naming re-reads, after its step failed, values the step may not have read on those inputs: a term only the
    naming evaluates (the state penalty, a potential or a density term H holds, a composed source's parts), H at a
    substitute density, or the step's own evaluation at a state whose failure the step's handler took. A computation's
    failure there (`ValueError`, `RuntimeError`, `ArithmeticError`) says nothing about which value entered: the read is
    not named, the naming goes on to its next read, and with none named the step's own failure stands. A
    `NonFiniteReadError` is a naming and leaves; anything else is a programming error and leaves too (#2981's review 4).
    """
    try:
        return read()
    except NonFiniteReadError:
        raise
    except (ValueError, RuntimeError, ArithmeticError):
        return None


def refuse_the_first_read_that_entered(
    solver: str,
    reads: Mapping[str, Any],
    step: Callable[[dict[str, Any]], Any],
    where: Callable[[str, int], str],
) -> None:
    """A step's result is non-finite: refuse the first value it read, in reading order, whose NaN or infinity entered it.

    ``reads`` maps each name to what the step read under it, in the order the step reads them; a ``None`` is skipped.
    ``step`` re-runs the step on such a mapping and returns its result. ``where(name, i)`` says where entry ``i`` of
    ``name`` (its flat index) was read.

    The non-finite entries are listed in reading order, each read's in its flat order, and set to 0.0. A re-run asks
    only whether a non-finite read reaches the result, so ``step`` must not run the step's checks of whether its
    solution is admissible: 0.0 can make it inadmissible where the read's value would not (a volatility tensor whose
    diagonal 0.0 is not PSD), and such a check's refusal would leave in place of the naming (#2981's review 3). The
    smallest
    prefix of that list whose restoration makes the result non-finite again ends at the first entry that entered, and
    is found by bisection, in O(log n) re-runs. That needs the result to be monotone in the restored set, so that
    restoring more entries cannot make it finite again. It is: a NaN or an infinity does not cancel through the step's
    sums, products and linear solves, and #2975's census removed or reads at every decision on the step that dropped
    one. The entry is refused through `refuse_a_non_finite_read`, with that read's earlier non-finite entries, which the
    bisection found discarded, given as 0.0, so the first it reports is the one that entered, and its count of
    non-finite values runs from that entry on. The re-runs' warnings are
    not reported: they repeat the step's own.

    Returns without raising when no read is non-finite, or when the result stays non-finite with every non-finite entry
    set to 0.0. That is a computed failure, and the caller reports it; a NaN that entered beside it is then not named
    (a stated limit, docs/user/CONVENTIONS.md).
    """
    arrays = {name: np.array(value, dtype=float) for name, value in reads.items() if value is not None}
    entries = [(name, int(i)) for name, array in arrays.items() for i in np.flatnonzero(~np.isfinite(array))]
    if not entries:
        return

    def with_restored(count: int) -> dict[str, Any]:
        trial = {name: np.where(np.isfinite(array), array, 0.0) for name, array in arrays.items()}
        for name, i in entries[:count]:
            trial[name].flat[i] = arrays[name].flat[i]
        mapping = dict(reads)
        for name, array in trial.items():
            mapping[name] = array if np.ndim(reads[name]) else float(array)
        return mapping

    def non_finite(count: int) -> bool:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = step(with_restored(count))
        return not bool(np.all(np.isfinite(np.asarray(result, dtype=float))))

    if non_finite(0):
        return
    low, high = 1, len(entries)
    while low < high:
        middle = (low + high) // 2
        if non_finite(middle):
            high = middle
        else:
            low = middle + 1
    name, i = entries[low - 1]
    values = arrays[name].copy()
    for earlier, j in entries[: low - 1]:
        if earlier == name:
            values.flat[j] = 0.0
    refuse_a_non_finite_read(solver, f"{where(name, i)}, where it entered the step", {name: values})
