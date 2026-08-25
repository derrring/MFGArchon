"""PyYAML types scientific-notation scalars differently from the OmegaConf layer #1687 removed.

The migration table in `docs/user/configuration_system.md` states this, and a doc that states a
behaviour is a claim that rots on the next PyYAML release. These are its pin.

PyYAML implements YAML 1.1, whose float rule requires **both a decimal point and a signed
exponent**. OmegaConf resolved the scalar itself and handed Pydantic a `float`. So a spelling that
worked before #1687 can now arrive as `str`, and whether that matters depends on the FIELD TYPE,
which is the half the first version of the doc got wrong:

- `float` field -- `load_solver_config` validates non-strictly and coerces the string back, so
  nothing breaks.
- `int` field -- a string does not coerce to `int` even non-strictly, so it raises out of
  `load_solver_config` with no `strict=True` anywhere in sight.
"""

from __future__ import annotations

import pytest
import yaml

from mfgarchon.config import MFGSolverConfig, load_solver_config

#: Exactly the table in `docs/user/configuration_system.md`. If PyYAML changes, this fails and the
#: doc is what needs editing.
SPELLINGS = [
    ("1e-8", str),
    ("1E-8", str),
    ("1e8", str),
    ("-1e-8", str),
    ("1.0e8", str),  # decimal point, UNSIGNED exponent -- the case that is easy to get wrong
    ("1.5e3", str),
    ("1.0e-8", float),
    ("1.0E+8", float),
    ("1.e-8", float),
    (".5e+3", float),
]


@pytest.mark.parametrize(("spelling", "expected"), SPELLINGS)
def test_the_documented_scalar_table_is_what_pyyaml_does(spelling, expected):
    assert type(yaml.safe_load(f"x: {spelling}")["x"]) is expected


def test_the_table_is_not_vacuous():
    """Both outcomes must be present, or the parametrize above could pass by naming one class."""
    kinds = {e for _, e in SPELLINGS}
    assert kinds == {str, float}, f"the table must exercise both outcomes; got {kinds}"


def test_a_bare_exponent_still_loads_for_a_FLOAT_field(tmp_path):
    """`tolerance: 1e-8` survives, because non-strict validation coerces the string back."""
    path = tmp_path / "cfg.yaml"
    path.write_text("picard:\n  tolerance: 1e-8\n")
    assert type(yaml.safe_load(path.read_text())["picard"]["tolerance"]) is str, (
        "this test is only meaningful while PyYAML hands us a string here"
    )
    assert load_solver_config(path).picard.tolerance == pytest.approx(1e-8)


def test_a_bare_exponent_RAISES_for_an_INT_field(tmp_path):
    """`max_iterations: 1e3` does not, and no `strict=True` is involved.

    This is the case the migration doc originally got wrong: it said `load_solver_config` was
    unaffected, full stop. That holds only for float-typed fields.
    """
    path = tmp_path / "cfg.yaml"
    path.write_text("picard:\n  max_iterations: 1e3\n")
    with pytest.raises(Exception, match="valid integer"):
        load_solver_config(path)

    # The signed-exponent spelling the doc recommends does work.
    path.write_text("picard:\n  max_iterations: 1.0e+3\n")
    assert load_solver_config(path).picard.max_iterations == 1000


def test_the_int_fields_the_doc_names_are_the_int_fields_that_exist():
    """The doc lists four exposed `int` fields by name. Pin the list, not the count."""

    def walk(model, prefix=""):
        for name, field in model.model_fields.items():
            annotation = field.annotation
            if hasattr(annotation, "model_fields"):
                yield from walk(annotation, f"{prefix}{name}.")
            elif annotation is int:
                yield f"{prefix}{name}"

    assert set(walk(MFGSolverConfig)) == {
        "hjb.accuracy_order",
        "hjb.newton.max_iterations",
        "picard.anderson_memory",
        "picard.max_iterations",
    }, "docs/user/configuration_system.md names these four; update both together"
