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
