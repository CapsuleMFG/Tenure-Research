"""Smoke tests for the dashboard plot builders.

These verify the figure builders return valid Plotly figures with the
expected number of traces. They don't render the figures.
"""

from __future__ import annotations

from datetime import date, timedelta

import plotly.graph_objects as go
import polars as pl

from dashboard.components.plots import (
    EVENT_MARKERS,
    build_cv_overlay,
    build_forward_forecast,
    build_residual_diagnostics,
    build_series_chart,
    build_yoy_chart,
)


def _monthly_history(n: int = 60) -> pl.DataFrame:
    start = date(2018, 1, 1)
    dates = []
    cur = start
    for _ in range(n):
        dates.append(cur)
        cur = (
            date(cur.year + 1, 1, 1)
            if cur.month == 12
            else date(cur.year, cur.month + 1, 1)
        )
    values = [100.0 + i * 0.5 for i in range(n)]
    return pl.DataFrame({"period_start": dates, "value": values}).with_columns(
        pl.col("period_start").cast(pl.Date),
        pl.col("value").cast(pl.Float64),
    )


def test_build_series_chart_returns_figure_with_one_trace() -> None:
    fig = build_series_chart(_monthly_history(), label="Test series", events=[])
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1


def test_build_series_chart_renders_event_markers_inside_range() -> None:
    fig = build_series_chart(
        _monthly_history(), label="Test series", events=EVENT_MARKERS
    )
    # Event markers add layout shapes/annotations, not traces
    assert len(fig.data) == 1
    assert fig.layout.annotations is not None


def test_build_yoy_chart_returns_figure() -> None:
    history = _monthly_history()
    yoy = history.with_columns(
        yoy_pct=(
            (pl.col("value") / pl.col("value").shift(12) - 1) * 100
        ).round(2)
    ).filter(pl.col("yoy_pct").is_not_null())
    fig = build_yoy_chart(yoy, label="Test series", events=[])
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1


def test_build_cv_overlay_renders_actuals_and_per_model_segments() -> None:
    history = _monthly_history(60)
    # Synthesize a tiny cv_details: 2 models x 2 windows x 3 horizon
    rows = []
    for model in ("AutoARIMA", "Prophet"):
        for w in range(2):
            for step in range(3):
                rows.append(
                    {
                        "model": model,
                        "window": w,
                        "period_start": date(2022, 1 + step, 1),
                        "point": 110.0,
                        "lower_80": 105.0,
                        "upper_80": 115.0,
                        "actual": 112.0,
                    }
                )
    cv = pl.DataFrame(rows)
    fig = build_cv_overlay(history, cv, label="Test series", horizon=3, n_windows=2)
    assert isinstance(fig, go.Figure)
    # 1 actuals trace + 2 models * 2 windows = 5 traces
    assert len(fig.data) == 5


def test_build_residual_diagnostics_three_panels() -> None:
    rows = []
    for w in range(3):
        for step in range(4):
            rows.append(
                {
                    "model": "AutoARIMA",
                    "window": w,
                    "period_start": date(2022, 1, 1) + timedelta(days=30 * step),
                    "point": 100.0 + step,
                    "lower_80": 95.0,
                    "upper_80": 105.0,
                    "actual": 102.0 + step,
                }
            )
    cv = pl.DataFrame(rows)
    fig = build_residual_diagnostics(cv, model_name="AutoARIMA", label="Test")
    assert isinstance(fig, go.Figure)
    # Histogram + bar + Q-Q dots + Q-Q reference line = 4 traces
    assert len(fig.data) == 4


def test_build_forward_forecast_renders_history_pi_band_and_forecast() -> None:
    history = _monthly_history(60)
    forward = pl.DataFrame(
        {
            "period_start": [date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1)],
            "point": [125.0, 126.0, 127.0],
            "lower_80": [120.0, 121.0, 122.0],
            "upper_80": [130.0, 131.0, 132.0],
        }
    ).with_columns(pl.col("period_start").cast(pl.Date))
    fig = build_forward_forecast(
        history, forward, model_name="AutoARIMA", label="Test"
    )
    assert isinstance(fig, go.Figure)
    # history + PI band + forecast line = 3 traces
    assert len(fig.data) == 3
