"""Tests for dashboard.components.brief — plain-English brief composition."""

from __future__ import annotations

from components.brief import compose_brief


def _entry_with_forecast() -> dict:
    """Minimal cache entry shaped like ``forecasts.json::by_series[sid]``."""
    return {
        "series_name": "Steers, choice, Nebraska direct, 65-80% Choice",
        "commodity": "cattle",
        "unit": "USD/cwt",
        "winner_model": "AutoARIMA",
        "latest_actual": {"period_start": "2026-03-01", "value": 221.0},
        "prior_month_actual": {"period_start": "2026-02-01", "value": 216.0},
        "prior_year_actual": {"period_start": "2025-03-01", "value": 187.0},
        "forward": [
            {"period_start": "2026-04-01", "point": 225.0, "lower_80": 215.0, "upper_80": 235.0},
            {"period_start": "2026-09-01", "point": 245.0, "lower_80": 220.0, "upper_80": 270.0},
        ],
    }


def test_compose_brief_renders_all_components() -> None:
    entry = _entry_with_forecast()
    brief = compose_brief(entry)
    assert "Steers, choice, Nebraska" in brief
    assert "March 2026" in brief
    assert "$221" in brief
    assert "USD/cwt" in brief
    # MoM: 221 vs 216 = +2.3% — should be "up"
    assert "up 2.3%" in brief
    # YoY: 221 vs 187 = +18.2% — should be "up"
    assert "up 18.2%" in brief
    # Forecast: 245 by Sep '26 with 80% PI of $220-$270
    assert "$245" in brief
    assert "$220" in brief
    assert "$270" in brief
    assert "AutoARIMA" in brief


def test_compose_brief_handles_missing_priors() -> None:
    entry = _entry_with_forecast()
    entry["prior_year_actual"] = None
    brief = compose_brief(entry)
    # MoM still computed, YoY falls back to "unavailable"
    assert "up 2.3%" in brief
    assert "YoY change unavailable" in brief


def test_compose_brief_handles_missing_forecast() -> None:
    entry = _entry_with_forecast()
    entry["forward"] = []
    brief = compose_brief(entry)
    # No forecast block; brief still mentions the headline + deltas
    assert "Steers, choice, Nebraska" in brief
    assert "regenerating" in brief


def test_compose_brief_handles_down_move() -> None:
    entry = _entry_with_forecast()
    entry["latest_actual"]["value"] = 200.0  # below both priors
    brief = compose_brief(entry)
    assert "down" in brief.lower()


def test_compose_brief_returns_empty_for_no_latest() -> None:
    entry = _entry_with_forecast()
    entry["latest_actual"] = None
    assert compose_brief(entry) == ""
