"""`model_fields_set` must mean "supplied by the caller" (Issue #2266).

`config/translator.py` reads `model_fields_set` to decide which fields to thread to a
solver. Two things silently broke that meaning, and both were found by adversarial review
of the rule change rather than by the suite — nothing here tested either path:

1. **A validator that assigns is recorded as a caller's value.** Pydantic records a
   `model_validator(mode="after")` assignment in `__pydantic_fields_set__` exactly as it
   records a constructor argument. So a *fresh* `FEMConfig()` reported
   `{'quadrature_order'}`, a fresh `HJBConfig()` reported `{'fdm'}` and a fresh `FPConfig()`
   reported `{'particle'}` — and the translator refused configurations nobody configured.
   FEM became unreachable through Safe and Auto mode.
2. **`to_yaml` dumped with `exclude_none`, not `exclude_unset`.** Every field was written
   out, so `from_yaml` returned a config in which everything read as explicitly set, and a
   round trip raised `NotImplementedError` on all four translator entry points.

Retirement condition: these trip if a new validator assigns without calling
`BaseConfig._forget_derived`, or if a dump stops using `exclude_unset`.
"""

from __future__ import annotations

import inspect
import tempfile
from pathlib import Path

import pydantic
import pytest

import mfgarchon.config.core as core_mod
import mfgarchon.config.mfg_methods as methods_mod
from mfgarchon.config import MFGSolverConfig
from mfgarchon.config.mfg_methods import FEMConfig, HJBConfig
from mfgarchon.config.translator import (
    backend_config_to_kwargs,
    check_logging_config,
    fp_config_to_kwargs,
    hjb_config_to_kwargs,
)
from mfgarchon.types import NumericalScheme


def _config_classes():
    """Every config model in the package, so this is a census and not a sample."""
    for mod in (core_mod, methods_mod):
        for name, cls in vars(mod).items():
            if inspect.isclass(cls) and issubclass(cls, pydantic.BaseModel) and cls.__module__ == mod.__name__:
                yield name, cls


class TestAFreshConfigHasSuppliedNothing:
    def test_no_config_class_reports_a_field_nobody_set(self):
        """A census, not a sample: the defect was found in one class and was in three."""
        lying = {name: cls().model_fields_set for name, cls in _config_classes() if _fresh_fields(cls)}
        assert not lying, f"these report fields nobody supplied: {lying}"

    def test_a_supplied_field_is_still_recorded(self):
        """The control. If `_forget_derived` over-reached, this is what would catch it."""
        assert HJBConfig(method="fem").model_fields_set == {"method"}
        assert FEMConfig(quadrature_order=7).model_fields_set == {"quadrature_order"}

    def test_the_derived_value_still_resolves(self):
        """Un-recording provenance must not stop the validator doing its job: 2p+1 at p=1."""
        assert FEMConfig().quadrature_order == 3


def _fresh_fields(cls):
    try:
        return cls().model_fields_set
    except Exception:
        return set()


class TestAYamlRoundTripPreservesProvenance:
    def test_a_round_tripped_default_config_threads_nothing(self):
        """The whole class in one assertion: an untouched config saved and reloaded must
        still look untouched, or every field arrives at the translator as a request."""
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "c.yaml"
            MFGSolverConfig().to_yaml(path)
            reloaded = MFGSolverConfig.from_yaml(path)
        assert reloaded.hjb.model_fields_set == set()
        assert reloaded.fp.model_fields_set == set()

    @pytest.mark.parametrize(
        "entry",
        [
            lambda c: hjb_config_to_kwargs(c.hjb, NumericalScheme.FDM_UPWIND),
            lambda c: fp_config_to_kwargs(c.fp, NumericalScheme.FDM_UPWIND),
            lambda c: backend_config_to_kwargs(c.backend),
            lambda c: check_logging_config(c.logging),
        ],
        ids=["hjb", "fp", "backend", "logging"],
    )
    def test_every_translator_entry_point_survives_a_round_trip(self, entry):
        """All four raised NotImplementedError before `exclude_unset`."""
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "c.yaml"
            MFGSolverConfig().to_yaml(path)
            reloaded = MFGSolverConfig.from_yaml(path)
        assert entry(reloaded) == entry(MFGSolverConfig())  # same as the in-memory control

    def test_a_supplied_field_survives_the_round_trip(self):
        """The control: `exclude_unset` must not drop what the caller DID set."""
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "c.yaml"
            MFGSolverConfig(hjb=HJBConfig(method="gfdm")).to_yaml(path)
            reloaded = MFGSolverConfig.from_yaml(path)
        assert reloaded.hjb.method == "gfdm"
        assert "method" in reloaded.hjb.model_fields_set
