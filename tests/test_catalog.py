"""Tests for SeriesDefinition field semantics.

Existing catalog tests live inline in test_clean.py because they were
about round-tripping through clean_all. This file focuses on the new
optional fields added in v0.2b: exogenous_regressors and forecastable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from usda_sandbox.catalog import (
    SeriesDefinition,
    load_catalog,
    save_catalog,
)


def _base_def(**overrides: object) -> dict[str, object]:
    """A minimal SeriesDefinition dict; override fields per test."""
    payload: dict[str, object] = {
        "series_id": "x",
        "series_name": "X",
        "commodity": "cattle",
        "metric": "price",
        "unit": "USD/cwt",
        "frequency": "monthly",
        "source_file": "x.xlsx",
        "source_sheet": "Sheet1",
        "header_rows_to_skip": 0,
        "value_columns": ["B"],
        "date_column": "A",
        "notes": "",
    }
    payload.update(overrides)
    return payload


def test_series_definition_exogenous_regressors_defaults_empty() -> None:
    sd = SeriesDefinition.model_validate(_base_def())
    assert sd.exogenous_regressors == []


def test_series_definition_forecastable_defaults_true() -> None:
    sd = SeriesDefinition.model_validate(_base_def())
    assert sd.forecastable is True


def test_series_definition_accepts_explicit_regressors() -> None:
    sd = SeriesDefinition.model_validate(
        _base_def(exogenous_regressors=["fut_1", "fut_2"])
    )
    assert sd.exogenous_regressors == ["fut_1", "fut_2"]


def test_series_definition_accepts_forecastable_false() -> None:
    sd = SeriesDefinition.model_validate(_base_def(forecastable=False))
    assert sd.forecastable is False


def test_series_definition_round_trips_with_new_fields(tmp_path: Path) -> None:
    cat = [
        SeriesDefinition.model_validate(
            _base_def(
                series_id="cash",
                exogenous_regressors=["fut_1mo", "fut_2mo"],
                forecastable=True,
            )
        ),
        SeriesDefinition.model_validate(
            _base_def(
                series_id="fut_1mo",
                series_name="Futures, 1 month",
                forecastable=False,
            )
        ),
    ]
    path = tmp_path / "catalog.json"
    save_catalog(path, cat)

    raw = json.loads(path.read_text())
    assert raw[0]["exogenous_regressors"] == ["fut_1mo", "fut_2mo"]
    assert raw[0]["forecastable"] is True
    assert raw[1]["exogenous_regressors"] == []
    assert raw[1]["forecastable"] is False

    reloaded = load_catalog(path)
    assert reloaded == cat


def test_series_definition_rejects_extra_unknown_field() -> None:
    """ConfigDict(extra='forbid') must still block typos like 'forcastable'."""
    with pytest.raises(ValidationError):
        SeriesDefinition.model_validate(_base_def(forcastable=True))
