"""Tests for usda_sandbox.basis — cash-to-futures basis computation."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from usda_sandbox.basis import (
    BasisStats,
    basis_stats,
    compute_basis,
    default_futures_peer,
    latest_basis,
)


def _build_obs(tmp_path: Path) -> Path:
    """Build a tiny observations.parquet with one cash + one monthly futures
    series sharing every month for 24 months."""
    months = [date(2024, m, 1) for m in range(1, 13)] + [
        date(2025, m, 1) for m in range(1, 13)
    ]
    cash_values = [200.0 + i * 1.0 for i in range(24)]
    fut_values = [195.0 + i * 1.0 for i in range(24)]  # basis = +5 consistently

    rows = []
    for d, v in zip(months, cash_values, strict=True):
        rows.append(
            {
                "series_id": "cattle_steer_choice_nebraska",
                "series_name": "Test cash",
                "commodity": "cattle",
                "metric": "price",
                "unit": "USD/cwt",
                "frequency": "monthly",
                "period_start": d,
                "period_end": d,
                "value": v,
                "source_file": "test",
                "source_sheet": "test",
                "ingested_at": "2026-05-15T00:00:00Z",
            }
        )
    for d, v in zip(months, fut_values, strict=True):
        rows.append(
            {
                "series_id": "cattle_lc_front",
                "series_name": "Test futures",
                "commodity": "cattle",
                "metric": "futures_price",
                "unit": "USD/cwt",
                "frequency": "monthly",
                "period_start": d,
                "period_end": d,
                "value": v,
                "source_file": "test",
                "source_sheet": "test",
                "ingested_at": "2026-05-15T00:00:00Z",
            }
        )
    obs = pl.DataFrame(rows)
    out = tmp_path / "observations.parquet"
    obs.write_parquet(out)
    return out


def test_default_futures_peer_known_pairs() -> None:
    assert default_futures_peer("cattle_steer_choice_nebraska") == "cattle_lc_front"
    assert default_futures_peer("hog_barrow_gilt_natbase_51_52") == "hogs_he_front"
    assert default_futures_peer("cattle_feeder_steer_750_800") == "cattle_feeder_front"
    assert default_futures_peer("unknown_id") is None


def test_compute_basis_joins_and_subtracts(tmp_path: Path) -> None:
    obs_path = _build_obs(tmp_path)
    df = compute_basis(
        "cattle_steer_choice_nebraska",
        "cattle_lc_front",
        obs_path=obs_path,
        prefer_daily_futures=False,
    )
    assert df.height == 24
    assert set(df.columns) >= {"period_start", "cash", "futures", "basis"}
    # Constructed cash − futures = +5 everywhere.
    assert (df["basis"] == 5.0).all()


def test_latest_basis_returns_most_recent(tmp_path: Path) -> None:
    obs_path = _build_obs(tmp_path)
    result = latest_basis(
        "cattle_steer_choice_nebraska", "cattle_lc_front", obs_path=obs_path
    )
    assert result is not None
    val, d = result
    assert val == 5.0
    assert d == date(2025, 12, 1)


def test_basis_stats_summary(tmp_path: Path) -> None:
    obs_path = _build_obs(tmp_path)
    stats = basis_stats(
        "cattle_steer_choice_nebraska", "cattle_lc_front", obs_path=obs_path
    )
    assert isinstance(stats, BasisStats)
    assert stats.n_obs == 24
    assert stats.latest_basis == 5.0
    assert stats.mean_basis == 5.0
    assert stats.median_basis == 5.0
    assert stats.p10_basis == 5.0
    assert stats.p90_basis == 5.0


def test_compute_basis_empty_when_no_overlap(tmp_path: Path) -> None:
    """If no period_start overlaps, the result should be an empty frame."""
    obs = pl.DataFrame(
        [
            {
                "series_id": "cattle_steer_choice_nebraska",
                "series_name": "Test cash",
                "commodity": "cattle",
                "metric": "price",
                "unit": "USD/cwt",
                "frequency": "monthly",
                "period_start": date(2024, 1, 1),
                "period_end": date(2024, 1, 1),
                "value": 200.0,
                "source_file": "test",
                "source_sheet": "test",
                "ingested_at": "2026-05-15T00:00:00Z",
            },
            {
                "series_id": "cattle_lc_front",
                "series_name": "Test futures",
                "commodity": "cattle",
                "metric": "futures_price",
                "unit": "USD/cwt",
                "frequency": "monthly",
                "period_start": date(2024, 2, 1),  # one month off
                "period_end": date(2024, 2, 1),
                "value": 195.0,
                "source_file": "test",
                "source_sheet": "test",
                "ingested_at": "2026-05-15T00:00:00Z",
            },
        ]
    )
    obs_path = tmp_path / "observations.parquet"
    obs.write_parquet(obs_path)
    df = compute_basis(
        "cattle_steer_choice_nebraska",
        "cattle_lc_front",
        obs_path=obs_path,
        prefer_daily_futures=False,
    )
    assert df.is_empty()
