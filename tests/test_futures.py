"""Tests for the CME futures ingest and deferred-series construction.

Tests are split into sections matching futures.py structure:
- Contract calendar (this task)
- Deferred-h interpolation (Task 4)
- Sync & manifest (Task 5)
- Append to observations (Task 5)
"""

from __future__ import annotations

from datetime import date

import pytest

from usda_sandbox.futures import (
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
