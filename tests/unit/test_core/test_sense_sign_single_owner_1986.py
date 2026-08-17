"""`sense_sign` has one owner. #1986

The pin is deliberately NOT `NetworkHamiltonian.sense_sign == MFGOperatorBase.sense_sign`. Once the
rival implementation is deleted both readings route through the same expression, so that comparison
is tautological and would pass over a broken owner. These are the values captured by running the
PRE-consolidation code, at `cdab5349`, before `NetworkHamiltonian`'s copy was removed:

    sense=MINIMIZE   sense_sign = 1.0   (float)
    sense=MAXIMIZE   sense_sign = -1.0  (float)

`float` is part of the capture: the deleted copy returned `1.0 / -1.0` while `_sign` holds `1 / -1`,
and consumers multiply by it, so the owner returns `float(self._sign)` rather than `self._sign`.
"""

from __future__ import annotations

import pytest

from mfgarchon.core.hamiltonian import OptimizationSense
from mfgarchon.extensions.topology import NetworkHamiltonian

_CAPTURED_BEFORE_CONSOLIDATION = {
    OptimizationSense.MINIMIZE: 1.0,
    OptimizationSense.MAXIMIZE: -1.0,
}


class _StubNetwork:
    num_nodes = 3
    adjacency_matrix = None

    def __getattr__(self, _name: str):
        return None


@pytest.mark.parametrize("sense", sorted(_CAPTURED_BEFORE_CONSOLIDATION, key=lambda s: s.value))
def test_sense_sign_matches_the_pre_consolidation_capture(sense):
    got = NetworkHamiltonian(_StubNetwork(), sense=sense).sense_sign
    want = _CAPTURED_BEFORE_CONSOLIDATION[sense]
    assert got == want, f"{sense.name}: {got} != {want} captured before the merge"
    assert isinstance(got, float), f"{sense.name}: returned {type(got).__name__}, capture was float"


def test_the_sense_to_sign_expression_has_one_owner_on_the_operator_side():
    """The consolidation's own gate: the count of places computing it must not climb back.

    Two remain, and they are on different objects rather than duplicates of each other:
    `MFGOperatorBase` (Hamiltonians) and the control-cost base. Whether those two must agree is a
    design question, open in #1986; this asserts only that no third appears and that the
    Hamiltonian side has exactly one.
    """
    import inspect

    from mfgarchon.core import hamiltonian
    from mfgarchon.extensions import topology

    src = inspect.getsource(hamiltonian)
    n = src.count("if sense == OptimizationSense.MINIMIZE else -1")
    assert n == 2, f"{n} sense->sign expressions in core.hamiltonian; 2 expected (control cost, operator base)"
    assert "OptimizationSense.MINIMIZE else -1" not in inspect.getsource(topology), (
        "NetworkHamiltonian recomputes the sign it inherits from MFGOperatorBase"
    )
