"""Tests for the continuous front-month futures ingest module."""

from __future__ import annotations

from dataclasses import asdict
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from usda_sandbox.futures_continuous import (
    ContinuousManifestEntry,
    _default_stooq_fetcher,
    append_continuous_to_observations,
    sync_continuous_futures,
)


def test_manifest_entry_round_trips_through_dict() -> None:
    entry = ContinuousManifestEntry(
        symbol="le.c",
        sha256="abc",
        downloaded_at="2026-05-13T10:00:00+00:00",
        missing=False,
    )
    assert ContinuousManifestEntry(**asdict(entry)) == entry


def test_parse_stooq_csv_returns_period_start_and_close() -> None:
    from usda_sandbox.futures_continuous import _parse_stooq_csv
    csv_text = (
        "Date,Open,High,Low,Close,Volume\n"
        "2024-01-31,170.0,175.0,168.0,172.5,1000\n"
        "2024-02-29,172.0,178.0,170.0,176.0,1200\n"
    )
    df = _parse_stooq_csv(csv_text)
    assert df.columns == ["period_start", "close"]
    assert df["period_start"].to_list() == [date(2024, 1, 31), date(2024, 2, 29)]
    assert df["close"].to_list() == [172.5, 176.0]
    assert df["period_start"].dtype == pl.Date
    assert df["close"].dtype == pl.Float64


def test_parse_stooq_csv_returns_empty_on_no_data_marker() -> None:
    from usda_sandbox.futures_continuous import _parse_stooq_csv
    df = _parse_stooq_csv("No data")
    assert df.is_empty()
    assert df.columns == ["period_start", "close"]
    assert df["period_start"].dtype == pl.Date
    assert df["close"].dtype == pl.Float64


def test_parse_stooq_csv_returns_empty_on_empty_input() -> None:
    from usda_sandbox.futures_continuous import _parse_stooq_csv
    df = _parse_stooq_csv("")
    assert df.is_empty()
    assert df.columns == ["period_start", "close"]
