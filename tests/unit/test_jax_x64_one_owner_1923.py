"""Issue #1923: `jax_enable_x64` is process-global and had three disagreeing writers.

    backends/jax_backend.py               False when precision='float32', True otherwise  (a POLICY)
    utils/acceleration/jax_utils.py       unconditional True, at MODULE IMPORT
    alg/.../meshless_galerkin/mls_basis.py   unconditional True, PER CALL

Last writer won, and which was last depended on import and call order. Nothing failed when they
disagreed — the result is a precision, not an exception. Measured on
`jax.experimental.sparse.linalg.spsolve` against scipy on a tridiagonal system with a known answer:
error 2.4e-07 (float32) where scipy gave 4.4e-16.

THE ASYMMETRY. Two writes are *requirements* ("this needs float64") and one is a *policy* ("the user
asked for float32"). A requirement silently overwriting a policy is the defect; a policy silently
starving a requirement is equally bad. Neither can be settled by ordering, so the owner makes the
conflict visible rather than picking a winner.

THESE TESTS MUTATE PROCESS-GLOBAL STATE, which is the thing under test, so each restores what it
found. A test that leaked its own precision setting would be an instance of the bug.
"""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax", reason="jax not installed")

from mfgarchon.utils.acceleration import jax_precision  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_global_precision():
    """Restore both the flag and the recorded policy. Without this the suite's precision would
    depend on test order -- the defect, reproduced inside its own regression test."""
    before = bool(jax.config.jax_enable_x64)
    policy, source = jax_precision._POLICY, jax_precision._POLICY_SOURCE
    yield
    jax_precision._POLICY, jax_precision._POLICY_SOURCE = policy, source
    jax.config.update("jax_enable_x64", before)


def test_a_requirement_is_honoured_when_no_policy_forbids_it():
    jax_precision._POLICY, jax_precision._POLICY_SOURCE = None, None
    jax.config.update("jax_enable_x64", False)
    jax_precision.require_x64("a test")
    assert jax.config.jax_enable_x64 is True


def test_a_requirement_fails_loud_against_an_explicit_float32_policy():
    """The behavioural point of #1923. Before, `mls_basis` flipped the switch and the user's float32
    process silently became float64 for everything that followed."""
    jax_precision.set_x64_policy(False, source="JAXBackend(precision='float32')")
    with pytest.raises(RuntimeError) as exc:
        jax_precision.require_x64("the meshless MLS basis")
    text = str(exc.value)
    assert "PROCESS-GLOBAL" in text, "the message must say why this cannot be resolved locally"
    assert "float32" in text, "it must name the policy that forbids it"
    assert "separate process" in text or "precision='float64'" in text, "it must offer a way out"
    # And it must NOT have flipped the switch on its way to raising.
    assert jax.config.jax_enable_x64 is False, "the refusal must leave the policy in force"


def test_a_float64_policy_does_not_block_a_requirement():
    """Control. Without it, a blanket refusal would pass the test above and break every float64 run."""
    jax_precision.set_x64_policy(True, source="JAXBackend(precision='float64')")
    jax_precision.require_x64("a test")
    assert jax.config.jax_enable_x64 is True


def test_the_state_accessor_reports_the_policy_and_its_source():
    jax_precision.set_x64_policy(False, source="unit-test")
    effective, policy, source = jax_precision.x64_state()
    assert (effective, policy, source) == (False, False, "unit-test")


def test_there_is_exactly_one_writer_in_the_package():
    """The SSOT gate: the count of places writing a process-global switch, and nothing else.

    Scanned over the package rather than a fixed list, so a fourth writer added later is caught
    rather than joining the three silently.
    """
    import pathlib

    import mfgarchon

    root = pathlib.Path(mfgarchon.__file__).parent
    owner = root / "utils" / "acceleration" / "jax_precision.py"
    writers = sorted(
        p.relative_to(root).as_posix()
        for p in root.rglob("*.py")
        if p != owner and 'config.update("jax_enable_x64"' in p.read_text(encoding="utf-8")
    )
    assert writers == [], (
        f"jax_enable_x64 is PROCESS-GLOBAL and must have one owner "
        f"(utils/acceleration/jax_precision.py). New writers: {writers}. Use `require_x64(reason)` "
        f"to declare a need, or `set_x64_policy(...)` from a backend the user configured."
    )
