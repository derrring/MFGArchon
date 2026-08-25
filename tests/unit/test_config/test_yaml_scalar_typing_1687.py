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


def _int_valued_fields() -> list[str]:
    """Dotted paths of every int-valued field, from pydantic's OWN JSON schema.

    Deliberately not a hand-rolled annotation walker. Two were written for this test and both were
    wrong: the first treated `Model | None` as a leaf and pruned three whole subtrees, missing 11
    of 15 fields; the second deduped by model class and lost a model reachable by two paths. The
    schema is the model's account of itself and agrees with a behavioural probe.
    """
    schema = MFGSolverConfig.model_json_schema()
    defs = schema.get("$defs", {})

    def walk(node, prefix="", seen=()):
        for name, spec in node.get("properties", {}).items():
            for option in [spec, *spec.get("anyOf", [])]:
                ref = option.get("$ref", "").rsplit("/", 1)[-1]
                if ref and ref in defs and ref not in seen:
                    yield from walk(defs[ref], f"{prefix}{name}.", (*seen, ref))
                elif option.get("type") == "integer":
                    yield f"{prefix}{name}"
                elif "enum" in option and all(isinstance(v, int) and not isinstance(v, bool) for v in option["enum"]):
                    yield f"{prefix}{name}"  # Literal[int]: same failure, different message

    return sorted(set(walk(schema)))


def _nested(path: str, raw: str) -> str:
    """A YAML document setting one dotted path to an unquoted scalar."""
    parts = path.split(".")
    return "".join(f"{'  ' * i}{p}:\n" for i, p in enumerate(parts[:-1])) + (
        f"{'  ' * (len(parts) - 1)}{parts[-1]}: {raw}\n"
    )


def test_every_int_field_rejects_a_bare_exponent_through_the_loader(tmp_path):
    """The PROPERTY, not a list of names. Naming them failed twice; this cannot go stale.

    An earlier version of this test asserted that the four fields the doc named were the four that
    exist. It passed while being wrong about 11 of 15, because the walker it compared against
    carried the same blind spot as the list -- both sides of the `==` shared the bug, so it
    certified the wrong answer instead of catching it.

    Every int-valued field rejects `5e3`: 11 with `int_parsing` and 4 -- the `Literal[int]` ones --
    with `literal_error`. Same cause, different message, so the assertion is that it raises.
    """
    fields = _int_valued_fields()
    assert "picard.max_iterations" in fields, f"the derivation lost a known field; got {fields}"
    assert len(fields) >= 10, (
        f"the derivation is what makes this test non-vacuous; it found only {len(fields)}: {fields}"
    )

    accepted = []
    for path in fields:
        cfg = tmp_path / "c.yaml"
        cfg.write_text(_nested(path, "5e3"))
        assert isinstance(yaml.safe_load(cfg.read_text()), dict), f"malformed fixture for {path}"
        try:
            load_solver_config(cfg)
            accepted.append(path)
        except Exception:  # the point is that SOMETHING refuses it, not which type
            pass
    assert not accepted, (
        f"these int fields accepted the bare-exponent spelling `5e3`, so the migration note in "
        f"docs/user/configuration_system.md is wrong about them: {accepted}"
    )

    # Negative control: the same spelling on a float field must still load, or the test above is
    # passing because `load_solver_config` rejects everything.
    cfg = tmp_path / "float.yaml"
    cfg.write_text("picard:\n  tolerance: 1e-8\n")
    assert load_solver_config(cfg).picard.tolerance == pytest.approx(1e-8)
