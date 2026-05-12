"""Tests for the CME futures ingest and deferred-series construction.

Tests are split into sections matching futures.py structure:
- Contract calendar (this task)
- Deferred-h interpolation (Task 4)
- Sync & manifest (Task 5)
- Append to observations (Task 5)
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from usda_sandbox.futures import (
    build_deferred_series,
    contract_delivery_date,
    contract_months,
    contract_ticker,
    parse_contract_ticker,
)

# --------------------------------------------------------------------------- #
# Contract calendar
# --------------------------------------------------------------------------- #


def test_contract_months_live_cattle() -> None:
    """Live cattle trades Feb/Apr/Jun/Aug/Oct/Dec → G/J/M/Q/V/Z."""
    assert contract_months("LE") == ("G", "J", "M", "Q", "V", "Z")


def test_contract_months_lean_hogs() -> None:
    """Lean hogs trades Feb/Apr/May/Jun/Jul/Aug/Oct/Dec → G/J/K/M/N/Q/V/Z."""
    assert contract_months("HE") == ("G", "J", "K", "M", "N", "Q", "V", "Z")


def test_contract_months_unknown_commodity_raises() -> None:
    with pytest.raises(ValueError, match="unknown commodity"):
        contract_months("XX")


def test_contract_delivery_date_returns_last_day_of_contract_month() -> None:
    # G = Feb, Z = Dec
    assert contract_delivery_date("LE", "Z", 2024) == date(2024, 12, 31)
    assert contract_delivery_date("LE", "G", 2024) == date(2024, 2, 29)  # leap year
    assert contract_delivery_date("LE", "G", 2023) == date(2023, 2, 28)
    assert contract_delivery_date("HE", "N", 2025) == date(2025, 7, 31)  # July


def test_contract_delivery_date_rejects_invalid_month() -> None:
    # Live cattle doesn't have a May contract (K)
    with pytest.raises(ValueError, match="not a valid LE contract month"):
        contract_delivery_date("LE", "K", 2024)


def test_contract_ticker_builds_yahoo_symbol() -> None:
    assert contract_ticker("LE", "Z", 2024) == "LEZ24.CME"
    assert contract_ticker("HE", "G", 2009) == "HEG09.CME"


def test_parse_contract_ticker_round_trips() -> None:
    assert parse_contract_ticker("LEZ24.CME") == ("LE", "Z", 2024)
    assert parse_contract_ticker("HEG09.CME") == ("HE", "G", 2009)


def test_parse_contract_ticker_rejects_malformed() -> None:
    with pytest.raises(ValueError, match="malformed ticker"):
        parse_contract_ticker("not-a-ticker")


# --------------------------------------------------------------------------- #
# Deferred-h interpolation
# --------------------------------------------------------------------------- #


def _synth_contract_prices(
    commodity: str,
    code: str,
    year: int,
    *,
    month_end_prices: dict[date, float],
) -> pl.DataFrame:
    """Build a synthetic per-contract month-end price DataFrame.

    Columns: contract_ticker, period_start (month-end date), close (float).
    """
    ticker = contract_ticker(commodity, code, year)
    return pl.DataFrame(
        {
            "contract_ticker": [ticker] * len(month_end_prices),
            "period_start": sorted(month_end_prices.keys()),
            "close": [month_end_prices[d] for d in sorted(month_end_prices)],
        }
    ).with_columns(
        pl.col("period_start").cast(pl.Date),
        pl.col("close").cast(pl.Float64),
    )


def test_deferred_series_uses_exact_contract_when_target_matches() -> None:
    """If the target delivery month IS a contract month, use that contract."""
    # On 2024-04-30 (April end), horizon=2 → target = June 30. LE has a June (M) contract.
    apr_z = _synth_contract_prices("LE", "M", 2024, month_end_prices={date(2024, 4, 30): 175.0})
    other_irrelevant = _synth_contract_prices(
        "LE", "Q", 2024, month_end_prices={date(2024, 4, 30): 178.0}
    )
    all_contracts = pl.concat([apr_z, other_irrelevant])
    result = build_deferred_series(
        commodity="LE",
        horizon_months=2,
        per_contract=all_contracts,
        months=[date(2024, 4, 30)],
    )
    assert result.height == 1
    # period_start follows the observations.parquet convention: first of month
    assert result["period_start"][0] == date(2024, 4, 1)
    assert result["period_end"][0] == date(2024, 4, 30)
    # Pure exact match → uses LEM24's price
    assert result["value"][0] == pytest.approx(175.0)


def test_deferred_series_linear_interpolates_between_contracts() -> None:
    """If target falls between two adjacent contracts, linearly interpolate."""
    # On 2024-04-30, horizon=3 → target = July 31. LE has no July; M (Jun) and Q (Aug)
    # are adjacent. Interpolate halfway between Jun price 170 and Aug price 180.
    contracts = pl.concat(
        [
            _synth_contract_prices(
                "LE", "M", 2024, month_end_prices={date(2024, 4, 30): 170.0}
            ),
            _synth_contract_prices(
                "LE", "Q", 2024, month_end_prices={date(2024, 4, 30): 180.0}
            ),
        ]
    )
    result = build_deferred_series(
        commodity="LE",
        horizon_months=3,
        per_contract=contracts,
        months=[date(2024, 4, 30)],
    )
    # Jun 30 -> Aug 31 is 62 days; Jun 30 -> Jul 31 is 31 days; halfway -> 175
    assert result["value"][0] == pytest.approx(175.0, abs=0.1)


def test_deferred_series_uses_earliest_when_target_before_curve() -> None:
    """If target is earlier than any active contract, use the earliest contract."""
    # On 2024-04-30, horizon=1 -> target = May 31. LE has no May; Jun is earliest.
    contracts = _synth_contract_prices(
        "LE", "M", 2024, month_end_prices={date(2024, 4, 30): 170.0}
    )
    result = build_deferred_series(
        commodity="LE",
        horizon_months=1,
        per_contract=contracts,
        months=[date(2024, 4, 30)],
    )
    # Target (May 31) is before Jun 30; use Jun price
    assert result["value"][0] == pytest.approx(170.0)


def test_deferred_series_monthly_cadence() -> None:
    """build_deferred_series emits one row per requested month."""
    contracts = _synth_contract_prices(
        "LE",
        "Z",
        2024,
        month_end_prices={
            date(2024, 1, 31): 150.0,
            date(2024, 2, 29): 152.0,
            date(2024, 3, 31): 154.0,
        },
    )
    months = [date(2024, 1, 31), date(2024, 2, 29), date(2024, 3, 31)]
    result = build_deferred_series(
        commodity="LE",
        horizon_months=12,
        per_contract=contracts,
        months=months,
    )
    assert result.height == 3
    # period_start is month-START (matches cash observations.parquet convention)
    assert result["period_start"].to_list() == [
        date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1),
    ]
    # period_end is month-end (as passed in)
    assert result["period_end"].to_list() == months


def test_deferred_series_returns_null_when_no_contract_data_on_date() -> None:
    """If no active contracts exist on date t, the deferred value is null."""
    contracts = _synth_contract_prices(
        "LE", "Z", 2024, month_end_prices={date(2024, 1, 31): 150.0}
    )
    result = build_deferred_series(
        commodity="LE",
        horizon_months=6,
        per_contract=contracts,
        months=[date(2023, 1, 31)],  # before contract data exists
    )
    assert result.height == 1
    assert result["value"][0] is None


def test_deferred_series_series_id_format() -> None:
    """series_id is f'{commodity_name}_{commodity_code}_deferred_{h}mo' where
    commodity_name maps LE->cattle, HE->hogs."""
    contracts = _synth_contract_prices(
        "LE", "Z", 2024, month_end_prices={date(2024, 1, 31): 150.0}
    )
    result = build_deferred_series(
        commodity="LE",
        horizon_months=6,
        per_contract=contracts,
        months=[date(2024, 1, 31)],
    )
    assert result["series_id"][0] == "cattle_lc_deferred_6mo"

    contracts_he = _synth_contract_prices(
        "HE", "Z", 2024, month_end_prices={date(2024, 1, 31): 80.0}
    )
    result_he = build_deferred_series(
        commodity="HE",
        horizon_months=3,
        per_contract=contracts_he,
        months=[date(2024, 1, 31)],
    )
    assert result_he["series_id"][0] == "hogs_he_deferred_3mo"


# --------------------------------------------------------------------------- #
# Sync & manifest
# --------------------------------------------------------------------------- #


def _fake_fetcher_factory(
    *,
    available: dict[str, list[tuple[date, float]]],
):
    """Build a fake yfinance fetcher: takes a ticker, returns OHLCV-like data
    for the requested ticker, or empty if not in ``available``.

    Each value in ``available`` is a list of (date, close) pairs.
    """

    def fetch(ticker: str) -> pl.DataFrame:
        if ticker not in available:
            return pl.DataFrame(
                schema={"period_start": pl.Date, "close": pl.Float64}
            )
        rows = available[ticker]
        return pl.DataFrame(
            {
                "period_start": [d for d, _ in rows],
                "close": [p for _, p in rows],
            }
        ).with_columns(
            pl.col("period_start").cast(pl.Date),
            pl.col("close").cast(pl.Float64),
        )

    return fetch


def test_sync_futures_writes_manifest_and_per_contract_files(
    tmp_path,
) -> None:
    from usda_sandbox.futures import sync_futures

    raw_dir = tmp_path / "futures"
    fetcher = _fake_fetcher_factory(
        available={
            "LEZ24.CME": [
                (date(2024, 1, 31), 175.0),
                (date(2024, 2, 29), 178.0),
            ],
            "LEG24.CME": [
                (date(2024, 1, 31), 173.0),
            ],
        }
    )
    manifest = sync_futures(
        commodities=("LE",),
        start_year=2024,
        end_year=2024,
        raw_dir=raw_dir,
        fetcher=fetcher,
    )
    # Two contracts in the available dict were fetched
    assert "LEZ24.CME" in manifest
    assert "LEG24.CME" in manifest
    # Files written to disk
    assert (raw_dir / "LE_Z_2024.parquet").exists()
    assert (raw_dir / "LE_G_2024.parquet").exists()
    # Manifest file written
    assert (raw_dir / "manifest.json").exists()
    # Each entry has a non-empty sha256
    for entry in manifest.values():
        assert len(entry.sha256) == 64


def test_sync_futures_idempotent_on_unchanged_data(tmp_path) -> None:
    from usda_sandbox.futures import sync_futures

    raw_dir = tmp_path / "futures"
    fetcher = _fake_fetcher_factory(
        available={"LEZ24.CME": [(date(2024, 1, 31), 175.0)]}
    )
    first = sync_futures(
        commodities=("LE",),
        start_year=2024,
        end_year=2024,
        raw_dir=raw_dir,
        fetcher=fetcher,
    )
    sha_before = first["LEZ24.CME"].sha256
    mtime_before = (raw_dir / "LE_Z_2024.parquet").stat().st_mtime_ns

    second = sync_futures(
        commodities=("LE",),
        start_year=2024,
        end_year=2024,
        raw_dir=raw_dir,
        fetcher=fetcher,
    )
    # SHA unchanged, file not rewritten
    assert second["LEZ24.CME"].sha256 == sha_before
    assert (raw_dir / "LE_Z_2024.parquet").stat().st_mtime_ns == mtime_before


def test_sync_futures_skips_unavailable_contracts(tmp_path) -> None:
    """yfinance returns empty for an ancient or delisted contract — record and skip."""
    from usda_sandbox.futures import sync_futures

    raw_dir = tmp_path / "futures"
    # No contracts available
    fetcher = _fake_fetcher_factory(available={})
    manifest = sync_futures(
        commodities=("LE",),
        start_year=2024,
        end_year=2024,
        raw_dir=raw_dir,
        fetcher=fetcher,
    )
    # Manifest is empty (nothing fetched), no parquet files written, no exception
    assert manifest == {}
    assert not list(raw_dir.glob("LE_*.parquet"))


# --------------------------------------------------------------------------- #
# Append to observations
# --------------------------------------------------------------------------- #


def test_append_futures_to_observations_writes_24_series(tmp_path) -> None:
    from usda_sandbox.futures import (
        append_futures_to_observations,
        sync_futures,
    )

    raw_dir = tmp_path / "futures"
    # Build a tiny fetcher with one LE and one HE contract, two months each
    fetcher = _fake_fetcher_factory(
        available={
            "LEZ24.CME": [
                (date(2024, 1, 31), 175.0),
                (date(2024, 2, 29), 178.0),
            ],
            "HEZ24.CME": [
                (date(2024, 1, 31), 90.0),
                (date(2024, 2, 29), 91.0),
            ],
        }
    )
    sync_futures(
        commodities=("LE", "HE"),
        start_year=2024,
        end_year=2024,
        raw_dir=raw_dir,
        fetcher=fetcher,
    )
    obs_path = tmp_path / "observations.parquet"
    # Seed observations.parquet with a placeholder cash row so the
    # function exercises its merge path. Use the canonical schema.
    seed = pl.DataFrame(
        {
            "series_id": ["existing_cash"],
            "series_name": ["Existing cash series"],
            "commodity": ["cattle"],
            "metric": ["price"],
            "unit": ["USD/cwt"],
            "frequency": ["monthly"],
            "period_start": [date(2024, 1, 1)],
            "period_end": [date(2024, 1, 31)],
            "value": [200.0],
            "source_file": ["x.xlsx"],
            "source_sheet": ["Sheet1"],
            "ingested_at": [None],
        }
    ).with_columns(
        pl.col("period_start").cast(pl.Date),
        pl.col("period_end").cast(pl.Date),
        pl.col("value").cast(pl.Float64),
        pl.col("ingested_at").cast(pl.Datetime("us", "UTC")),
    )
    seed.write_parquet(obs_path)

    append_futures_to_observations(
        obs_path=obs_path,
        raw_dir=raw_dir,
        commodities=("LE", "HE"),
        horizons=range(1, 13),
    )

    final = pl.read_parquet(obs_path)
    series_ids = set(final["series_id"].to_list())
    # 12 LE deferred + 12 HE deferred + 1 existing cash row preserved
    expected = {f"cattle_lc_deferred_{h}mo" for h in range(1, 13)}
    expected |= {f"hogs_he_deferred_{h}mo" for h in range(1, 13)}
    expected |= {"existing_cash"}
    assert series_ids == expected


def test_append_futures_to_observations_is_idempotent(tmp_path) -> None:
    """Running it twice produces the same parquet content."""
    from usda_sandbox.futures import (
        append_futures_to_observations,
        sync_futures,
    )

    raw_dir = tmp_path / "futures"
    fetcher = _fake_fetcher_factory(
        available={
            "LEZ24.CME": [(date(2024, 1, 31), 175.0)],
        }
    )
    sync_futures(
        commodities=("LE",),
        start_year=2024,
        end_year=2024,
        raw_dir=raw_dir,
        fetcher=fetcher,
    )
    obs_path = tmp_path / "observations.parquet"
    # Empty seed
    pl.DataFrame(
        schema={
            "series_id": pl.Utf8,
            "series_name": pl.Utf8,
            "commodity": pl.Utf8,
            "metric": pl.Utf8,
            "unit": pl.Utf8,
            "frequency": pl.Utf8,
            "period_start": pl.Date,
            "period_end": pl.Date,
            "value": pl.Float64,
            "source_file": pl.Utf8,
            "source_sheet": pl.Utf8,
            "ingested_at": pl.Datetime("us", "UTC"),
        }
    ).write_parquet(obs_path)

    append_futures_to_observations(
        obs_path=obs_path,
        raw_dir=raw_dir,
        commodities=("LE",),
        horizons=range(1, 4),  # just 3 horizons for speed
    )
    height_after_first = pl.read_parquet(obs_path).height

    append_futures_to_observations(
        obs_path=obs_path,
        raw_dir=raw_dir,
        commodities=("LE",),
        horizons=range(1, 4),
    )
    height_after_second = pl.read_parquet(obs_path).height

    assert height_after_second == height_after_first
