# Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a laptop-local Streamlit dashboard that wraps the existing `usda_sandbox` package — three pages (Explore / Visualize / Forecast), persistent sidebar, and live per-window backtest progress on the Forecast page.

**Architecture:** New sibling `dashboard/` directory imports from the existing `src/usda_sandbox/` package. One additive change to `forecast.py` (a `cross_validate_iter` helper + an `iter_run_backtest` generator), plus extracted plot-builder functions in `dashboard/components/plots.py` so notebook code and dashboard code share figure construction. No changes to the existing public APIs of the package — all 56 existing tests stay green.

**Tech Stack:** Streamlit ≥1.32, polars, plotly, matplotlib (for statsmodels decomposition), statsmodels. Existing forecasters (AutoARIMA / Prophet / LightGBM) reused as-is.

**Spec:** [`docs/superpowers/specs/2026-05-08-dashboard-design.md`](../specs/2026-05-08-dashboard-design.md)

---

## File map

| Path | Action | Responsibility |
|---|---|---|
| `pyproject.toml` | modify | Add `streamlit>=1.32` to `[project.dependencies]` |
| `src/usda_sandbox/forecast.py` | modify | Add `BacktestProgress` dataclass, `cross_validate_iter` helper on `BaseForecaster`, `iter_run_backtest` generator; refactor existing `cross_validate` to delegate to the helper |
| `tests/test_forecast.py` | modify | Add 3 tests for `iter_run_backtest` and the new helper |
| `dashboard/__init__.py` | create | Empty marker |
| `dashboard/app.py` | create | Streamlit entrypoint — landing page + sidebar wiring |
| `dashboard/components/__init__.py` | create | Empty marker |
| `dashboard/components/sidebar.py` | create | Series picker, data status, refresh button |
| `dashboard/components/plots.py` | create | Pure plotly figure builders shared across pages |
| `dashboard/pages/1_Explore.py` | create | Catalog + KPIs + null-span summary |
| `dashboard/pages/2_Visualize.py` | create | Single-series charts (time series + decomposition + YoY) |
| `dashboard/pages/3_Forecast.py` | create | Live backtest UI + results panels |
| `tests/test_dashboard_plots.py` | create | Smoke tests for plot builders |
| `dashboard/README.md` | create | How to run the dashboard |
| `README.md` | modify | Add a "Dashboard" section |

---

## Task 1: Add Streamlit dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the dependency**

Edit `pyproject.toml` and add `"streamlit>=1.32",` to the `[project] dependencies` array, immediately after the `"statsmodels>=0.14",` line.

The dependencies array should now end with:
```toml
    "statsmodels>=0.14",
    "streamlit>=1.32",
]
```

- [ ] **Step 2: Sync**

Run: `uv sync`
Expected: Streamlit and its transitive deps install. No errors.

- [ ] **Step 3: Verify import**

Run: `uv run python -c "import streamlit; print(streamlit.__version__)"`
Expected: Prints a version string ≥ 1.32.

- [ ] **Step 4: Verify existing tests still pass**

Run: `uv run pytest`
Expected: 56 passed.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add streamlit for dashboard"
```

---

## Task 2: Add `cross_validate_iter` helper and refactor `cross_validate`

This is a pure refactor — no behavior change. It extracts the per-window loop in `BaseForecaster.cross_validate` into an iterator helper so Task 3 can wrap it with progress events.

**Files:**
- Modify: `src/usda_sandbox/forecast.py`
- Modify: `tests/test_forecast.py`

- [ ] **Step 1: Write a failing test asserting the iterator shape**

Append to `tests/test_forecast.py` (at the end of the "Cross-validate" section):

```python
def test_cross_validate_iter_yields_per_window(synthetic_series: pl.DataFrame) -> None:
    m = StatsForecastAutoARIMA(seed=42)
    items = list(m.cross_validate_iter(synthetic_series, horizon=3, n_windows=2))
    assert len(items) == 2
    for idx, (w, window_df) in enumerate(items):
        assert w == idx
        assert window_df.columns == _CV_COLS
        assert window_df.height == 3


def test_cross_validate_uses_iter_under_the_hood(
    synthetic_series: pl.DataFrame,
) -> None:
    m = StatsForecastAutoARIMA(seed=42)
    full = m.cross_validate(synthetic_series, horizon=3, n_windows=2)
    iterated = pl.concat(
        [df for _, df in m.cross_validate_iter(synthetic_series, horizon=3, n_windows=2)]
    ).sort(["window", "period_start"])
    assert full.equals(iterated)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_forecast.py::test_cross_validate_iter_yields_per_window tests/test_forecast.py::test_cross_validate_uses_iter_under_the_hood -v`
Expected: 2 failed with `AttributeError: ... has no attribute 'cross_validate_iter'`.

- [ ] **Step 3: Add `cross_validate_iter` and refactor `cross_validate`**

In `src/usda_sandbox/forecast.py`, locate the `BaseForecaster` class. Replace the existing `cross_validate` method with this pair of methods:

```python
    def cross_validate_iter(
        self, df: pl.DataFrame, horizon: int, n_windows: int
    ) -> Iterator[tuple[int, pl.DataFrame]]:
        """Yield ``(window_index, window_results_df)`` as each window completes.

        Same validation and windowing semantics as :meth:`cross_validate`,
        but lets callers observe progress between windows.
        """
        if horizon <= 0 or n_windows <= 0:
            raise ValueError("horizon and n_windows must both be positive")
        df = _validate_input(df)
        n = df.height
        needed = horizon * (n_windows + 1)
        if n < needed:
            raise ValueError(
                f"Need at least {needed} observations for {n_windows} windows of "
                f"horizon {horizon}; got {n}"
            )

        for w in range(n_windows):
            cutoff_idx = n - (n_windows - w) * horizon
            train = df.slice(0, cutoff_idx)
            target = df.slice(cutoff_idx, horizon)
            self.fit(train)
            pred = self.predict(horizon)
            merged = (
                pred.join(
                    target.select(
                        ["period_start", pl.col("value").alias("actual")]
                    ),
                    on="period_start",
                    how="inner",
                )
                .with_columns(window=pl.lit(w, dtype=pl.Int32))
                .select(
                    ["window", "period_start", "point", "lower_80", "upper_80", "actual"]
                )
            )
            yield w, merged

    def cross_validate(
        self, df: pl.DataFrame, horizon: int, n_windows: int
    ) -> pl.DataFrame:
        """Rolling-origin CV with ``n_windows`` non-overlapping forecast blocks.

        Window 0 is the oldest cutoff; window ``n_windows - 1`` is the most
        recent. Each window holds out ``horizon`` observations after fitting
        on everything before them. Implemented on top of
        :meth:`cross_validate_iter`.
        """
        frames = [
            merged for _, merged in self.cross_validate_iter(df, horizon, n_windows)
        ]
        return pl.concat(frames).sort(["window", "period_start"])
```

`Iterator` is already imported via `collections.abc.Sequence` — verify the imports at the top of `forecast.py` include `Iterator` from `collections.abc`. If not, add it:

```python
from collections.abc import Iterator, Sequence
```

(replacing the existing `from collections.abc import Sequence` line).

- [ ] **Step 4: Run new tests**

Run: `uv run pytest tests/test_forecast.py::test_cross_validate_iter_yields_per_window tests/test_forecast.py::test_cross_validate_uses_iter_under_the_hood -v`
Expected: 2 passed.

- [ ] **Step 5: Run the full forecast test suite to confirm no regression**

Run: `uv run pytest tests/test_forecast.py -v`
Expected: 17 passed (15 existing + 2 new).

- [ ] **Step 6: Commit**

```bash
git add src/usda_sandbox/forecast.py tests/test_forecast.py
git commit -m "refactor(forecast): extract cross_validate_iter helper"
```

---

## Task 3: Add `BacktestProgress` and `iter_run_backtest`

Generator variant of `run_backtest` that yields per-window progress events. The dashboard's Forecast page consumes this; existing `run_backtest()` keeps its single-shot signature.

**Files:**
- Modify: `src/usda_sandbox/forecast.py`
- Modify: `tests/test_forecast.py`

- [ ] **Step 1: Write a failing test for the generator's shape**

Append to `tests/test_forecast.py` (after the existing `run_backtest` tests):

```python
def test_iter_run_backtest_yields_progress_then_result(
    synthetic_obs_parquet: Path,
) -> None:
    from usda_sandbox.forecast import (
        BacktestProgress,
        BacktestResult,
        iter_run_backtest,
    )

    items = list(
        iter_run_backtest(
            "synthetic_test",
            horizon=3,
            n_windows=2,
            obs_path=synthetic_obs_parquet,
        )
    )
    # 3 models * 2 windows = 6 progress events, then 1 final result
    assert len(items) == 7
    progress_events = items[:-1]
    final = items[-1]

    assert all(isinstance(e, BacktestProgress) for e in progress_events)
    assert isinstance(final, BacktestResult)

    # First event is for the first model, first window
    assert progress_events[0].model == "AutoARIMA"
    assert progress_events[0].window == 0
    assert progress_events[0].n_windows == 2
    assert progress_events[0].running_mape is not None
    assert progress_events[0].elapsed_s >= 0


def test_iter_run_backtest_final_matches_run_backtest(
    synthetic_obs_parquet: Path,
) -> None:
    from usda_sandbox.forecast import iter_run_backtest, run_backtest

    direct = run_backtest(
        "synthetic_test", horizon=3, n_windows=2, obs_path=synthetic_obs_parquet
    )
    items = list(
        iter_run_backtest(
            "synthetic_test",
            horizon=3,
            n_windows=2,
            obs_path=synthetic_obs_parquet,
        )
    )
    final = items[-1]
    assert final.series_id == direct.series_id
    assert final.horizon == direct.horizon
    assert final.n_windows == direct.n_windows
    assert final.cv_details.equals(direct.cv_details)
    assert final.metrics.equals(direct.metrics)


def test_iter_run_backtest_unknown_series_raises(
    synthetic_obs_parquet: Path,
) -> None:
    from usda_sandbox.forecast import iter_run_backtest

    with pytest.raises(ValueError, match="No observations"):
        list(iter_run_backtest("does_not_exist", obs_path=synthetic_obs_parquet))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_forecast.py::test_iter_run_backtest_yields_progress_then_result tests/test_forecast.py::test_iter_run_backtest_final_matches_run_backtest tests/test_forecast.py::test_iter_run_backtest_unknown_series_raises -v`
Expected: 3 failed with `ImportError: cannot import name 'BacktestProgress'` (or `iter_run_backtest`).

- [ ] **Step 3: Add `time` import and the new public surface**

In `src/usda_sandbox/forecast.py`, add `import time` to the top imports (alongside the existing `import logging`, `import os`, etc.).

Then locate the existing `BacktestResult` dataclass (near the bottom of the file, just before `run_backtest`). Insert this `BacktestProgress` dataclass *immediately above* `BacktestResult`:

```python
@dataclass(frozen=True)
class BacktestProgress:
    """Yielded by :func:`iter_run_backtest` after each (model, window) completes.

    ``running_mape`` is the MAPE accumulated across the windows completed so
    far for the current model — ``None`` only if every actual is zero (which
    the metric helpers can't divide by).
    """

    model: str
    window: int
    n_windows: int
    elapsed_s: float
    running_mape: float | None
```

Then, *immediately after* the existing `run_backtest` function, append:

```python
def iter_run_backtest(
    series_id: str,
    horizon: int = 6,
    n_windows: int = 8,
    *,
    obs_path: Path | str | None = None,
    seed: int = DEFAULT_SEED,
) -> Iterator[BacktestProgress | BacktestResult]:
    """Generator variant of :func:`run_backtest`.

    Yields one :class:`BacktestProgress` event after each (model, window)
    completes (``3 * n_windows`` events total), then a single final
    :class:`BacktestResult` with the same content :func:`run_backtest` would
    return for the same inputs.
    """
    series = read_series(series_id, obs_path)
    if series.is_empty():
        raise ValueError(f"No observations found for series_id={series_id!r}")
    series = series.filter(pl.col("value").is_not_null()).select(
        ["period_start", "value"]
    )

    forecasters: list[tuple[str, BaseForecaster]] = [
        ("AutoARIMA", StatsForecastAutoARIMA(seed=seed)),
        ("Prophet", ProphetForecaster(seed=seed)),
        ("LightGBM", LightGBMForecaster(seed=seed)),
    ]

    detail_frames: list[pl.DataFrame] = []
    started = time.time()

    for name, fcst in forecasters:
        per_model_frames: list[pl.DataFrame] = []
        for w, window_df in fcst.cross_validate_iter(series, horizon, n_windows):
            per_model_frames.append(window_df)
            so_far = pl.concat(per_model_frames)
            actual = so_far["actual"].to_numpy().astype(float)
            point = so_far["point"].to_numpy().astype(float)
            running = _mape(actual, point)
            yield BacktestProgress(
                model=name,
                window=w,
                n_windows=n_windows,
                elapsed_s=time.time() - started,
                running_mape=None if running != running else running,  # NaN guard
            )
        cv = (
            pl.concat(per_model_frames)
            .sort(["window", "period_start"])
            .with_columns(model=pl.lit(name))
        )
        detail_frames.append(cv)

    cv_details = pl.concat(detail_frames).sort(["model", "window", "period_start"])
    train_values = series["value"].to_numpy().astype(float)
    metrics = _per_model_metrics(cv_details, train_values)

    yield BacktestResult(
        series_id=series_id,
        horizon=horizon,
        n_windows=n_windows,
        cv_details=cv_details,
        metrics=metrics,
    )
```

- [ ] **Step 4: Run new tests**

Run: `uv run pytest tests/test_forecast.py::test_iter_run_backtest_yields_progress_then_result tests/test_forecast.py::test_iter_run_backtest_final_matches_run_backtest tests/test_forecast.py::test_iter_run_backtest_unknown_series_raises -v`
Expected: 3 passed.

- [ ] **Step 5: Run the full test suite**

Run: `uv run pytest`
Expected: 61 passed (56 existing + 2 from Task 2 + 3 from this task).

- [ ] **Step 6: Verify lint and types**

Run: `uv run ruff check . && uv run mypy`
Expected: Both clean.

- [ ] **Step 7: Commit**

```bash
git add src/usda_sandbox/forecast.py tests/test_forecast.py
git commit -m "feat(forecast): add iter_run_backtest with per-window progress events"
```

---

## Task 4: Plot builder functions in `dashboard/components/plots.py`

Pure-function plotly figure builders. Both the dashboard pages and (optionally, in the future) notebooks can import these — no copy-pasted plot code anywhere.

**Files:**
- Create: `dashboard/__init__.py`
- Create: `dashboard/components/__init__.py`
- Create: `dashboard/components/plots.py`
- Create: `tests/test_dashboard_plots.py`

- [ ] **Step 1: Create the empty package markers**

Create `dashboard/__init__.py` with one line:
```python
"""Streamlit dashboard for the USDA livestock sandbox."""
```

Create `dashboard/components/__init__.py` with one line:
```python
"""Shared dashboard components (sidebar, plot builders)."""
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_dashboard_plots.py`:

```python
"""Smoke tests for the dashboard plot builders.

These verify the figure builders return valid Plotly figures with the
expected number of traces. They don't render the figures.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import plotly.graph_objects as go
import polars as pl
import pytest

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
    # Synthesize a tiny cv_details: 2 models × 2 windows × 3 horizon
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
```

- [ ] **Step 3: Run the test to verify failure**

Run: `uv run pytest tests/test_dashboard_plots.py -v`
Expected: All fail with `ModuleNotFoundError: No module named 'dashboard.components.plots'`.

- [ ] **Step 4: Implement the builders**

Create `dashboard/components/plots.py`:

```python
"""Plotly figure builders shared between dashboard pages.

Pure functions: input is data, output is a ``plotly.graph_objects.Figure``.
No Streamlit imports — these are testable without a running app.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
import plotly.graph_objects as go
import polars as pl
import scipy.stats as stats
from plotly.subplots import make_subplots


@dataclass(frozen=True)
class EventMarker:
    iso_date: str  # YYYY-MM-DD
    label: str


EVENT_MARKERS: tuple[EventMarker, ...] = (
    EventMarker("2008-09-01", "Financial crisis"),
    EventMarker("2014-11-01", "2014-15 cattle peak"),
    EventMarker("2020-04-01", "COVID slaughter shock"),
    EventMarker("2022-03-01", "Russia/Ukraine spike"),
    EventMarker("2025-06-01", "2024-25 cycle high"),
)

_MODEL_PALETTE: dict[str, str] = {
    "AutoARIMA": "#1f77b4",
    "Prophet": "#2ca02c",
    "LightGBM": "#d62728",
}


def _add_event_annotations(
    fig: go.Figure, events: Sequence[EventMarker]
) -> None:
    for ev in events:
        fig.add_vline(
            x=ev.iso_date,
            line_width=1,
            line_dash="dot",
            line_color="rgba(120,120,120,0.6)",
        )
        fig.add_annotation(
            x=ev.iso_date,
            yref="paper",
            y=1.02,
            text=ev.label,
            showarrow=False,
            font=dict(size=10, color="gray"),
            xanchor="left",
        )


def build_series_chart(
    history: pl.DataFrame,
    label: str,
    events: Sequence[EventMarker] = EVENT_MARKERS,
) -> go.Figure:
    """Single-series time-series chart with event markers.

    ``history`` must have ``period_start`` (Date) and ``value`` columns.
    """
    fig = go.Figure(
        go.Scatter(
            x=history["period_start"].to_list(),
            y=history["value"].to_list(),
            mode="lines",
            name=label,
            line=dict(color="rgba(50,50,50,0.85)", width=1.6),
        )
    )
    fig.update_layout(
        title=f"{label} — history",
        xaxis_title="Period",
        yaxis_title="Value",
        height=420,
        margin=dict(t=60, b=40),
        hovermode="x unified",
    )
    _add_event_annotations(fig, events)
    return fig


def build_yoy_chart(
    yoy: pl.DataFrame,
    label: str,
    events: Sequence[EventMarker] = EVENT_MARKERS,
) -> go.Figure:
    """Year-over-year percent change line chart with zero reference line.

    ``yoy`` must have ``period_start`` and ``yoy_pct`` columns.
    """
    fig = go.Figure(
        go.Scatter(
            x=yoy["period_start"].to_list(),
            y=yoy["yoy_pct"].to_list(),
            mode="lines",
            name=f"{label} YoY %",
            line=dict(color="#2563eb", width=1.4),
        )
    )
    fig.add_hline(y=0, line_width=1, line_color="rgba(0,0,0,0.4)")
    fig.update_layout(
        title=f"{label} — year-over-year change (%)",
        xaxis_title="Period",
        yaxis_title="YoY change, %",
        height=380,
        hovermode="x unified",
    )
    _add_event_annotations(fig, events)
    return fig


def build_cv_overlay(
    history: pl.DataFrame,
    cv_details: pl.DataFrame,
    label: str,
    horizon: int,
    n_windows: int,
) -> go.Figure:
    """Actuals + per-model CV-window forecast segments on shared axes.

    ``cv_details`` must have ``model``, ``window``, ``period_start``, ``point``
    columns (plus ``lower_80``/``upper_80``/``actual`` — unused here).
    """
    cv_min = cv_details["period_start"].min()
    history_recent = history.filter(
        pl.col("period_start") >= cv_min - _years(2)
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history_recent["period_start"].to_list(),
            y=history_recent["value"].to_list(),
            mode="lines",
            name="Actual",
            line=dict(color="rgba(50,50,50,0.85)", width=1.4),
        )
    )

    for model_name in sorted(cv_details["model"].unique().to_list()):
        color = _MODEL_PALETTE.get(model_name, "#888")
        sub = cv_details.filter(pl.col("model") == model_name).sort(
            ["window", "period_start"]
        )
        for w in sorted(sub["window"].unique().to_list()):
            wdf = sub.filter(pl.col("window") == w)
            fig.add_trace(
                go.Scatter(
                    x=wdf["period_start"].to_list(),
                    y=wdf["point"].to_list(),
                    mode="lines",
                    line=dict(color=color, width=1.4),
                    name=model_name,
                    legendgroup=model_name,
                    showlegend=(w == sub["window"].min()),
                )
            )

    fig.update_layout(
        title=(
            f"{label} — actuals vs. CV forecasts "
            f"(h={horizon}, {n_windows} windows)"
        ),
        xaxis_title="Period",
        yaxis_title="Value",
        height=480,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
    )
    return fig


def build_residual_diagnostics(
    cv_details: pl.DataFrame,
    model_name: str,
    label: str,
) -> go.Figure:
    """Three-panel diagnostics for one model's CV residuals.

    Panels: (1) residual histogram, (2) MAE by forecast horizon step,
    (3) Q-Q vs. standard normal.
    """
    cv = (
        cv_details.filter(pl.col("model") == model_name)
        .sort(["window", "period_start"])
        .with_columns(
            residual=pl.col("actual") - pl.col("point"),
            step=pl.col("period_start").rank(method="ordinal").over("window"),
        )
    )
    residuals = np.asarray(cv["residual"].to_list(), dtype=float)

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            f"Residual distribution (mean={residuals.mean():.2f}, "
            f"sd={residuals.std():.2f})",
            "MAE by forecast horizon",
            "Q-Q vs. standard normal",
        ),
    )

    fig.add_trace(
        go.Histogram(x=residuals, nbinsx=20, marker_color="rgba(31,119,180,0.7)"),
        row=1,
        col=1,
    )

    by_step = (
        cv.group_by("step")
        .agg(pl.col("residual").abs().mean().alias("mae"))
        .sort("step")
    )
    fig.add_trace(
        go.Bar(
            x=by_step["step"].to_list(),
            y=by_step["mae"].to_list(),
            marker_color="rgba(214,39,40,0.75)",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="Months ahead", row=1, col=2)
    fig.update_yaxes(title_text="Mean abs error", row=1, col=2)

    osm, osr = stats.probplot(residuals, dist="norm", fit=False)
    fig.add_trace(
        go.Scatter(
            x=osm,
            y=osr,
            mode="markers",
            marker=dict(color="rgba(44,160,44,0.8)", size=5),
            showlegend=False,
        ),
        row=1,
        col=3,
    )
    line_x = np.array([osm.min(), osm.max()])
    line_y = line_x * residuals.std() + residuals.mean()
    fig.add_trace(
        go.Scatter(
            x=line_x,
            y=line_y,
            mode="lines",
            line=dict(color="black", dash="dot", width=1),
            showlegend=False,
        ),
        row=1,
        col=3,
    )
    fig.update_xaxes(title_text="Theoretical quantile", row=1, col=3)
    fig.update_yaxes(title_text="Sample quantile", row=1, col=3)

    fig.update_layout(
        title_text=f"{label} — residual diagnostics ({model_name})",
        height=400,
        showlegend=False,
    )
    return fig


def build_forward_forecast(
    history: pl.DataFrame,
    forward: pl.DataFrame,
    model_name: str,
    label: str,
    history_tail_months: int = 60,
) -> go.Figure:
    """Headline forward-forecast chart: history tail + 80% PI band + point line."""
    tail = history.tail(history_tail_months)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=tail["period_start"].to_list(),
            y=tail["value"].to_list(),
            mode="lines",
            name="Historical",
            line=dict(color="rgba(50,50,50,0.9)", width=1.6),
        )
    )

    upper_x = forward["period_start"].to_list()
    lower_x = list(reversed(upper_x))
    pi_y = forward["upper_80"].to_list() + list(
        reversed(forward["lower_80"].to_list())
    )
    fig.add_trace(
        go.Scatter(
            x=upper_x + lower_x,
            y=pi_y,
            fill="toself",
            fillcolor="rgba(31,119,180,0.18)",
            line=dict(color="rgba(0,0,0,0)"),
            name="80% PI",
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forward["period_start"].to_list(),
            y=forward["point"].to_list(),
            mode="lines+markers",
            name=f"{model_name} forecast",
            line=dict(color="rgb(31,119,180)", width=2),
            marker=dict(size=6),
        )
    )

    last_actual = history["period_start"].max()
    fig.add_vline(
        x=str(last_actual), line_color="rgba(120,120,120,0.6)", line_dash="dot"
    )

    fig.update_layout(
        title=f"{label} — {model_name} forward forecast",
        xaxis_title="Period",
        yaxis_title="Value",
        height=460,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
    )
    return fig


def _years(n: int) -> "object":
    """Returns a polars-compatible duration of N years for date arithmetic."""
    import datetime as _dt

    return _dt.timedelta(days=365 * n)


__all__ = [
    "EVENT_MARKERS",
    "EventMarker",
    "build_cv_overlay",
    "build_forward_forecast",
    "build_residual_diagnostics",
    "build_series_chart",
    "build_yoy_chart",
]
```

- [ ] **Step 5: Run the smoke tests**

Run: `uv run pytest tests/test_dashboard_plots.py -v`
Expected: 6 passed.

- [ ] **Step 6: Run all tests**

Run: `uv run pytest`
Expected: 67 passed.

- [ ] **Step 7: Lint + types**

Run: `uv run ruff check . && uv run mypy`
Expected: Both clean. (Note: `mypy` config currently checks `src/usda_sandbox` only — `dashboard/` isn't type-checked, which is intentional. If ruff complains about anything cosmetic, run `uv run ruff check --fix .`.)

- [ ] **Step 8: Commit**

```bash
git add dashboard/__init__.py dashboard/components/__init__.py dashboard/components/plots.py tests/test_dashboard_plots.py
git commit -m "feat(dashboard): add plot builder functions"
```

---

## Task 5: Sidebar component

**Files:**
- Create: `dashboard/components/sidebar.py`

- [ ] **Step 1: Create the sidebar module**

Create `dashboard/components/sidebar.py`:

```python
"""Persistent sidebar — series picker, data status, refresh button.

Each page calls :func:`render_sidebar` once at the top. The chosen series
lives in ``st.session_state["series_id"]`` so it survives navigation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import streamlit as st

from usda_sandbox.clean import clean_all
from usda_sandbox.ingest import sync_downloads
from usda_sandbox.store import list_series, read_observations

DEFAULT_OBS_PATH = Path("data/clean/observations.parquet")
DEFAULT_CATALOG_PATH = Path("data/catalog.json")
DEFAULT_RAW_DIR = Path("data/raw")


@st.cache_data(ttl=300)
def cached_list_series(obs_path_str: str) -> pl.DataFrame:
    return list_series(Path(obs_path_str))


@st.cache_data(ttl=300)
def cached_dataset_overview(obs_path_str: str) -> dict[str, object]:
    obs = read_observations(Path(obs_path_str)).collect()
    return {
        "n_series": int(obs["series_id"].n_unique()),
        "n_rows": int(obs.height),
        "earliest": obs["period_start"].min(),
        "latest": obs["period_start"].max(),
    }


def _format_age(dt: datetime) -> str:
    delta = datetime.now(timezone.utc) - dt
    secs = int(delta.total_seconds())
    if secs < 60:
        return f"{secs}s ago"
    if secs < 3600:
        return f"{secs // 60}m ago"
    if secs < 86400:
        return f"{secs // 3600}h ago"
    return f"{secs // 86400}d ago"


def render_sidebar(*, obs_path: Path = DEFAULT_OBS_PATH) -> str | None:
    """Render the sidebar and return the currently-selected series_id (or None).

    If ``observations.parquet`` does not exist, returns ``None`` and shows a
    bootstrap message + refresh button.
    """
    st.sidebar.title("USDA Livestock")

    if not obs_path.exists():
        st.sidebar.warning(
            "No cleaned data yet. Click **Refresh data** below to download "
            "and clean the ERS files."
        )
        _render_refresh_button(obs_path)
        return None

    series_df = cached_list_series(str(obs_path))
    overview = cached_dataset_overview(str(obs_path))

    st.sidebar.subheader("Series")
    series_ids = series_df["series_id"].to_list()
    name_lookup = {
        sid: name
        for sid, name in zip(
            series_df["series_id"].to_list(),
            series_df["series_name"].to_list(),
            strict=True,
        )
    }

    default_index = 0
    if "series_id" in st.session_state and st.session_state["series_id"] in series_ids:
        default_index = series_ids.index(st.session_state["series_id"])

    chosen = st.sidebar.selectbox(
        label="Active series",
        options=series_ids,
        index=default_index,
        format_func=lambda sid: name_lookup.get(sid, sid),
        key="series_id",
    )

    st.sidebar.subheader("Data status")
    mtime = datetime.fromtimestamp(obs_path.stat().st_mtime, tz=timezone.utc)
    st.sidebar.markdown(
        f"**{overview['n_series']}** series · **{overview['n_rows']:,}** rows  \n"
        f"Range: `{overview['earliest']}` .. `{overview['latest']}`  \n"
        f"Last cleaned: `{_format_age(mtime)}`"
    )

    _render_refresh_button(obs_path)
    return chosen


def _render_refresh_button(obs_path: Path) -> None:
    if not st.sidebar.button("🔄 Refresh data", use_container_width=True):
        return
    with st.sidebar.status("Refreshing data...", expanded=True) as status:
        status.write("Discovering ERS download URLs...")
        try:
            sync_downloads(raw_dir=DEFAULT_RAW_DIR)
        except Exception as exc:
            status.update(label=f"Download failed: {exc}", state="error")
            return
        status.write("Cleaning all catalog series...")
        try:
            clean_all(DEFAULT_CATALOG_PATH, DEFAULT_RAW_DIR, obs_path)
        except Exception as exc:
            status.update(label=f"Clean failed: {exc}", state="error")
            return
        status.update(label="Refresh complete", state="complete")
    cached_list_series.clear()
    cached_dataset_overview.clear()
    st.rerun()
```

- [ ] **Step 2: Smoke-test the import**

Run: `uv run python -c "from dashboard.components.sidebar import render_sidebar; print('ok')"`
Expected: Prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add dashboard/components/sidebar.py
git commit -m "feat(dashboard): add persistent sidebar component"
```

---

## Task 6: App entrypoint

**Files:**
- Create: `dashboard/app.py`

- [ ] **Step 1: Create the entrypoint**

Create `dashboard/app.py`:

```python
"""USDA Livestock Sandbox — Streamlit entrypoint.

Launch with::

    uv run streamlit run dashboard/app.py

Streamlit auto-discovers pages from ``dashboard/pages/``.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from dashboard.components.sidebar import DEFAULT_OBS_PATH, render_sidebar

st.set_page_config(
    page_title="USDA Livestock Sandbox",
    page_icon="🐂",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    obs_path = Path(DEFAULT_OBS_PATH)
    render_sidebar(obs_path=obs_path)

    st.title("USDA Livestock Sandbox")
    st.markdown(
        """
A laptop-local view layer over the cleaned USDA ERS livestock and meat
data. Use the sidebar to pick a series, then visit one of the pages:

* **Explore** — what's in the cleaned store: catalog, coverage, data quality.
* **Visualize** — single-series time-series + seasonal decomposition + YoY change.
* **Forecast** — run a backtest (3 models × 12 windows) live and see the scoreboard,
  CV overlay, residuals, and a 12-month forward forecast with 80% prediction
  intervals.

The dashboard is read-only against `data/clean/observations.parquet`.
Click **Refresh data** in the sidebar to re-run ingest and cleaning from
the latest ERS release.
"""
    )

    if not obs_path.exists():
        st.info(
            "**No data yet.** Click *Refresh data* in the sidebar to "
            "download the ERS files and run the cleaner."
        )
    else:
        st.success(f"Cleaned store loaded from `{obs_path}`. Pick a page from the sidebar.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the import**

Run: `uv run python -c "import dashboard.app; print('ok')"`
Expected: Prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add dashboard/app.py
git commit -m "feat(dashboard): add app entrypoint and home page"
```

---

## Task 7: Explore page

**Files:**
- Create: `dashboard/pages/1_Explore.py`

- [ ] **Step 1: Create the page**

Create `dashboard/pages/1_Explore.py`:

```python
"""Page 1 — what's in the cleaned store."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import streamlit as st

from dashboard.components.sidebar import (
    DEFAULT_OBS_PATH,
    cached_dataset_overview,
    cached_list_series,
    render_sidebar,
)
from usda_sandbox.store import duckdb_connection

st.set_page_config(page_title="Explore — USDA Livestock", page_icon="🐂", layout="wide")
render_sidebar()

st.title("Explore the cleaned store")

obs_path = Path(DEFAULT_OBS_PATH)
if not obs_path.exists():
    st.warning("No data yet — click **Refresh data** in the sidebar.")
    st.stop()

overview = cached_dataset_overview(str(obs_path))
series_df = cached_list_series(str(obs_path))

col_a, col_b, col_c = st.columns(3)
col_a.metric("Total series", overview["n_series"])
col_b.metric("Total rows", f"{overview['n_rows']:,}")
col_c.metric(
    "Date range",
    f"{overview['earliest']} → {overview['latest']}",
)

st.subheader("Catalog")
commodities = sorted(series_df["commodity"].unique().to_list())
selected = st.multiselect(
    "Filter by commodity",
    options=commodities,
    default=commodities,
)
filtered = series_df.filter(pl.col("commodity").is_in(selected)).sort("series_id")
st.dataframe(
    filtered.select(
        [
            "series_id",
            "series_name",
            "commodity",
            "metric",
            "unit",
            "frequency",
            "n_obs",
            "n_nulls",
            "first_period",
            "last_period",
        ]
    ),
    hide_index=True,
    use_container_width=True,
)

st.subheader("By commodity")
con = duckdb_connection(obs_path)
try:
    by_commodity = con.execute(
        """
        SELECT commodity,
               COUNT(DISTINCT series_id)                          AS n_series,
               COUNT(*)                                            AS n_rows,
               SUM(CASE WHEN value IS NULL THEN 1 ELSE 0 END)      AS n_nulls,
               MIN(period_start)                                   AS earliest,
               MAX(period_start)                                   AS latest
        FROM obs
        GROUP BY commodity
        ORDER BY commodity
        """
    ).pl()
finally:
    con.close()
st.dataframe(by_commodity, hide_index=True, use_container_width=True)

st.subheader("Null spans (where source data was missing)")
null_summary = (
    series_df.filter(pl.col("n_nulls") > 0)
    .select(
        ["series_id", "series_name", "n_obs", "n_nulls", "first_period", "last_period"]
    )
    .sort("n_nulls", descending=True)
)
if null_summary.is_empty():
    st.success("Every series has full coverage — no nulls.")
else:
    st.dataframe(null_summary, hide_index=True, use_container_width=True)
```

- [ ] **Step 2: Smoke-test the import**

Streamlit pages are scripts, not modules — `import dashboard.pages.1_Explore` won't work because of the leading digit. Test by running Streamlit headless:

Run:
```bash
uv run python -c "
import importlib.util
spec = importlib.util.spec_from_file_location('explore', 'dashboard/pages/1_Explore.py')
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
print('module spec loads ok')
"
```
Expected: Prints `module spec loads ok`. (We're not executing the page; just verifying the file is syntactically valid Python.)

- [ ] **Step 3: Commit**

```bash
git add dashboard/pages/1_Explore.py
git commit -m "feat(dashboard): add Explore page"
```

---

## Task 8: Visualize page

**Files:**
- Create: `dashboard/pages/2_Visualize.py`

- [ ] **Step 1: Create the page**

Create `dashboard/pages/2_Visualize.py`:

```python
"""Page 2 — single-series deep dive."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import streamlit as st
from statsmodels.tsa.seasonal import seasonal_decompose

from dashboard.components.plots import (
    EVENT_MARKERS,
    build_series_chart,
    build_yoy_chart,
)
from dashboard.components.sidebar import DEFAULT_OBS_PATH, render_sidebar
from usda_sandbox.store import read_series

st.set_page_config(
    page_title="Visualize — USDA Livestock", page_icon="🐂", layout="wide"
)
series_id = render_sidebar()

st.title("Visualize a series")

if series_id is None:
    st.warning("No data yet — click **Refresh data** in the sidebar.")
    st.stop()

obs_path = Path(DEFAULT_OBS_PATH)

@st.cache_data(ttl=300)
def _cached_series(series_id: str, obs_path_str: str) -> pl.DataFrame:
    df = read_series(series_id, Path(obs_path_str))
    return df.filter(pl.col("value").is_not_null()).sort("period_start")


series = _cached_series(series_id, str(obs_path))
if series.is_empty():
    st.error(f"No non-null observations for series_id={series_id!r}")
    st.stop()

meta_row = series.row(0, named=True)
st.subheader(meta_row["series_name"])
m1, m2, m3, m4 = st.columns(4)
m1.metric("Commodity", meta_row["commodity"])
m2.metric("Unit", meta_row["unit"])
m3.metric("Frequency", meta_row["frequency"])
m4.metric("Observations", series.height)

# Date-range slider
min_d = series["period_start"].min()
max_d = series["period_start"].max()
date_range = st.slider(
    "Date range",
    min_value=min_d,
    max_value=max_d,
    value=(min_d, max_d),
    format="YYYY-MM",
)
filtered = series.filter(
    (pl.col("period_start") >= date_range[0])
    & (pl.col("period_start") <= date_range[1])
)
visible_events = [
    ev for ev in EVENT_MARKERS
    if pd.Timestamp(ev.iso_date).date() >= date_range[0]
    and pd.Timestamp(ev.iso_date).date() <= date_range[1]
]

st.plotly_chart(
    build_series_chart(filtered, label=meta_row["series_name"], events=visible_events),
    use_container_width=True,
)

st.subheader("Year-over-year change (%)")
yoy = (
    series.sort("period_start")
    .with_columns(
        yoy_pct=((pl.col("value") / pl.col("value").shift(12) - 1) * 100).round(2)
    )
    .filter(pl.col("yoy_pct").is_not_null())
    .filter(
        (pl.col("period_start") >= date_range[0])
        & (pl.col("period_start") <= date_range[1])
    )
)
if yoy.is_empty():
    st.info("Not enough history in this range to compute YoY change.")
else:
    st.plotly_chart(
        build_yoy_chart(yoy, label=meta_row["series_name"], events=visible_events),
        use_container_width=True,
    )

st.subheader("Seasonal decomposition")
with st.expander("Decomposition options"):
    period = st.number_input(
        "Period (months)",
        min_value=2,
        max_value=24,
        value=12,
        step=1,
    )
    model_kind = st.radio(
        "Model",
        options=("multiplicative", "additive"),
        index=0,
        horizontal=True,
    )

if filtered.height < period * 2:
    st.info(
        f"Need at least {period * 2} observations in the selected range for "
        f"period={period} decomposition; have {filtered.height}."
    )
else:
    pdf = filtered.to_pandas()
    ts = pd.Series(
        pdf["value"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(pdf["period_start"]),
    ).asfreq("MS")
    if (ts <= 0).any() and model_kind == "multiplicative":
        st.warning(
            "Series contains non-positive values — falling back to additive."
        )
        model_kind = "additive"
    ts = ts.ffill()
    result = seasonal_decompose(ts, model=model_kind, period=int(period))
    fig = result.plot()
    fig.set_size_inches(11, 7)
    fig.suptitle(
        f"{meta_row['series_name']} — {model_kind} decomposition (period={period})",
        y=1.02,
    )
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
```

- [ ] **Step 2: Smoke-test the import**

Run:
```bash
uv run python -c "
import importlib.util
spec = importlib.util.spec_from_file_location('viz', 'dashboard/pages/2_Visualize.py')
assert spec and spec.loader
print('module spec loads ok')
"
```
Expected: Prints `module spec loads ok`.

- [ ] **Step 3: Commit**

```bash
git add dashboard/pages/2_Visualize.py
git commit -m "feat(dashboard): add Visualize page"
```

---

## Task 9: Forecast page

This is the most complex page — live backtest with progress streaming, plus all results panels.

**Files:**
- Create: `dashboard/pages/3_Forecast.py`

- [ ] **Step 1: Create the page**

Create `dashboard/pages/3_Forecast.py`:

```python
"""Page 3 — live backtest + forecast results."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import streamlit as st

from dashboard.components.plots import (
    build_cv_overlay,
    build_forward_forecast,
    build_residual_diagnostics,
)
from dashboard.components.sidebar import DEFAULT_OBS_PATH, render_sidebar
from usda_sandbox.forecast import (
    BacktestProgress,
    BacktestResult,
    LightGBMForecaster,
    ProphetForecaster,
    StatsForecastAutoARIMA,
    iter_run_backtest,
)
from usda_sandbox.store import read_series

st.set_page_config(
    page_title="Forecast — USDA Livestock", page_icon="🐂", layout="wide"
)
series_id = render_sidebar()

st.title("Forecast")

if series_id is None:
    st.warning("No data yet — click **Refresh data** in the sidebar.")
    st.stop()

obs_path = Path(DEFAULT_OBS_PATH)

# ---------------- Inputs -----------------------------------------------------

cfg_a, cfg_b, cfg_c, cfg_d = st.columns([1, 1, 2, 1])
horizon = cfg_a.slider("Horizon (months)", min_value=1, max_value=12, value=6)
n_windows = cfg_b.slider("CV windows", min_value=2, max_value=24, value=12)
all_models = ("AutoARIMA", "Prophet", "LightGBM")
selected_models = cfg_c.multiselect(
    "Models", options=list(all_models), default=list(all_models)
)
run_clicked = cfg_d.button("▶ Run backtest", type="primary", use_container_width=True)

cache_key = f"backtest:{series_id}:{horizon}:{n_windows}:{','.join(sorted(selected_models))}"

# ---------------- Run --------------------------------------------------------

if run_clicked:
    if not selected_models:
        st.error("Pick at least one model.")
        st.stop()

    series = (
        read_series(series_id, obs_path)
        .filter(pl.col("value").is_not_null())
        .select(["period_start", "value"])
    )
    if series.height < horizon * (n_windows + 1):
        st.error(
            f"Series has {series.height} non-null observations; need at least "
            f"{horizon * (n_windows + 1)} for {n_windows} windows of horizon "
            f"{horizon}."
        )
        st.stop()

    with st.status(
        f"Running backtest for {series_id}...", expanded=True
    ) as status:
        progress_bar = st.progress(0.0)
        total_steps = len(all_models) * n_windows
        completed = 0
        try:
            for event in iter_run_backtest(
                series_id,
                horizon=horizon,
                n_windows=n_windows,
                obs_path=obs_path,
            ):
                if isinstance(event, BacktestProgress):
                    completed += 1
                    pct = completed / total_steps
                    progress_bar.progress(pct)
                    mape_str = (
                        f"{event.running_mape:.2f}%"
                        if event.running_mape is not None
                        else "n/a"
                    )
                    status.write(
                        f"{event.model} window {event.window + 1}/{event.n_windows} "
                        f"done · running MAPE: {mape_str} · "
                        f"elapsed: {event.elapsed_s:.1f}s"
                    )
                elif isinstance(event, BacktestResult):
                    st.session_state[cache_key] = event
            status.update(label="Backtest complete", state="complete")
        except Exception as exc:
            status.update(label=f"Backtest failed: {exc}", state="error")
            st.exception(exc)
            st.stop()

# ---------------- Results ----------------------------------------------------

result: BacktestResult | None = st.session_state.get(cache_key)
if result is None:
    st.info(
        "Configure the inputs above and click **Run backtest** to see "
        "scoreboard, CV overlay, residual diagnostics, and a 12-month "
        "forward forecast."
    )
    st.stop()

# Filter to selected models only
filtered_cv = result.cv_details.filter(pl.col("model").is_in(selected_models))
filtered_metrics = result.metrics.filter(pl.col("model").is_in(selected_models))

st.subheader("Scoreboard")
st.dataframe(
    filtered_metrics.with_columns(
        pl.col("mape").round(2),
        pl.col("smape").round(2),
        pl.col("mase").round(2),
    ).sort("mape"),
    hide_index=True,
    use_container_width=True,
)

# Pick the winner from the filtered set
winner = filtered_metrics.sort("mape")["model"][0]
st.success(
    f"**Winner:** {winner} — MAPE "
    f"{filtered_metrics.filter(pl.col('model') == winner)['mape'][0]:.2f}%"
)

st.subheader("Actuals vs. CV forecasts")
history = (
    read_series(series_id, obs_path)
    .filter(pl.col("value").is_not_null())
    .sort("period_start")
)
label = history.select(pl.col("period_start").min()).item()
series_label = series_id  # raw id is fine here; sidebar already shows pretty name
st.plotly_chart(
    build_cv_overlay(
        history,
        filtered_cv,
        label=series_label,
        horizon=result.horizon,
        n_windows=result.n_windows,
    ),
    use_container_width=True,
)

st.subheader(f"Residual diagnostics — {winner}")
st.plotly_chart(
    build_residual_diagnostics(filtered_cv, model_name=winner, label=series_label),
    use_container_width=True,
)

st.subheader("Forward 12-month forecast")
forecaster_registry = {
    "AutoARIMA": StatsForecastAutoARIMA,
    "Prophet": ProphetForecaster,
    "LightGBM": LightGBMForecaster,
}
with st.spinner(f"Refitting {winner} on full history..."):
    fcst = forecaster_registry[winner](seed=42)
    fcst.fit(history.select(["period_start", "value"]))
    forward = fcst.predict(horizon=12)

st.plotly_chart(
    build_forward_forecast(history, forward, model_name=winner, label=series_label),
    use_container_width=True,
)

with st.expander("Numeric forecast table"):
    st.dataframe(
        forward.with_columns(
            pl.col("point").round(2),
            pl.col("lower_80").round(2),
            pl.col("upper_80").round(2),
        ),
        hide_index=True,
        use_container_width=True,
    )
```

- [ ] **Step 2: Smoke-test the import**

Run:
```bash
uv run python -c "
import importlib.util
spec = importlib.util.spec_from_file_location('forecast_page', 'dashboard/pages/3_Forecast.py')
assert spec and spec.loader
print('module spec loads ok')
"
```
Expected: Prints `module spec loads ok`.

- [ ] **Step 3: End-to-end smoke test by launching streamlit**

Run (in background — Streamlit blocks the terminal):
```bash
uv run streamlit run dashboard/app.py --server.headless true --server.port 8501 &
sleep 5
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8501
kill %1 2>/dev/null || true
```

Expected: Prints `200` (the Streamlit server is responding). If it doesn't, check the Streamlit output for import or page-load errors and fix them before committing.

- [ ] **Step 4: Run the full test suite**

Run: `uv run pytest`
Expected: 67 passed.

- [ ] **Step 5: Commit**

```bash
git add dashboard/pages/3_Forecast.py
git commit -m "feat(dashboard): add Forecast page with live backtest"
```

---

## Task 10: README updates

**Files:**
- Modify: `README.md`
- Create: `dashboard/README.md`

- [ ] **Step 1: Add a Dashboard section to the top-level README**

In `README.md`, find the **"How to run"** section. After the existing code block (which ends with `uv run mypy`), add a new code-fenced block and a new section:

Insert this directly after the existing "How to run" code block:

````markdown
### Launch the dashboard

```bash
uv run streamlit run dashboard/app.py
```

Opens at http://localhost:8501. Three pages: **Explore** (catalog and
coverage), **Visualize** (single-series time-series + decomposition + YoY),
and **Forecast** (live backtest with per-window progress + scoreboard +
12-month forward forecast). See [`dashboard/README.md`](dashboard/README.md)
for details.
````

- [ ] **Step 2: Create the dashboard README**

Create `dashboard/README.md`:

```markdown
# Dashboard

Local Streamlit app over the cleaned `data/clean/observations.parquet` store.

## Launch

```bash
uv sync
uv run streamlit run dashboard/app.py
```

Opens at http://localhost:8501.

## Pages

* **Explore** — catalog table (filterable by commodity), per-commodity
  rollup via DuckDB, and a null-span summary. Read-only.
* **Visualize** — single-series view chosen via the sidebar series picker.
  Date-range slider, time-series with event markers, year-over-year
  percent change, and statsmodels seasonal decomposition (multiplicative
  by default; falls back to additive for series with non-positive values).
* **Forecast** — interactive backtest. Configure horizon (1–12 months),
  number of CV windows (2–24), and which models to include
  (AutoARIMA / Prophet / LightGBM). Click **Run backtest** to watch
  per-model / per-window progress stream in. Results: scoreboard with
  MAPE / sMAPE / MASE, CV-overlay chart, residual diagnostics for the
  winner, and a 12-month forward forecast with shaded 80% prediction
  interval.

## Sidebar

* **Series picker** — populated from `list_series()`; selection persists
  across pages via `st.session_state`.
* **Data status** — series count, total rows, date range, and how
  recently `observations.parquet` was rebuilt.
* **Refresh data** — runs `sync_downloads()` and `clean_all()` in
  sequence with progress streamed into a status panel; clears the
  cached series list on success.

## Architecture

```
dashboard/
├── app.py                # entrypoint + landing page
├── components/
│   ├── plots.py          # plotly figure builders (pure functions, smoke-tested)
│   └── sidebar.py        # series picker / data status / refresh button
└── pages/
    ├── 1_Explore.py
    ├── 2_Visualize.py
    └── 3_Forecast.py
```

The dashboard imports from `src/usda_sandbox/` but never modifies it. The
package is the contract; the dashboard is a replaceable view layer.

## Caching

* `cached_list_series` and `cached_dataset_overview` use
  `@st.cache_data(ttl=300)`; both are cleared on Refresh.
* Backtest results live in `st.session_state` keyed on
  `(series_id, horizon, n_windows, models)` — flipping pages doesn't
  re-run the computation.

## Known limitations (v1)

* Single-user (`st.session_state` is per-browser-session).
* Backtest cancellation isn't supported — flipping pages mid-run interrupts
  the iterator (Streamlit reruns the script). Acceptable: just rerun.
* No persisted backtest history; nothing is written to disk by the dashboard.
* No exogenous regressors (CME futures, weather) — those are out of scope
  for v0.1; see top-level README "What's not in v0.1".
```

- [ ] **Step 3: Verify the README links work**

Run:
```bash
uv run python -c "
from pathlib import Path
top = Path('README.md').read_text(encoding='utf-8')
assert 'streamlit run dashboard/app.py' in top, 'top README missing launch command'
assert Path('dashboard/README.md').exists(), 'dashboard README missing'
print('readmes look good')
"
```
Expected: Prints `readmes look good`.

- [ ] **Step 4: Final gate — full lint, type, test**

Run: `uv run ruff check . && uv run mypy && uv run pytest`
Expected: All clean. 67 tests pass.

- [ ] **Step 5: Commit**

```bash
git add README.md dashboard/README.md
git commit -m "docs: README + dashboard README"
```

- [ ] **Step 6: Push the branch**

If working on `main`:
```bash
git push origin main
```

If working on a feature branch:
```bash
git push -u origin <branch-name>
```

---

## Self-review notes

**Spec coverage:**
- Architecture (`dashboard/` sibling, `src/usda_sandbox/` unchanged) → Tasks 4-9 ✓
- Three pages (Explore / Visualize / Forecast) → Tasks 7, 8, 9 ✓
- Sidebar (picker, status, refresh) → Task 5 ✓
- Live computation: `iter_run_backtest` generator → Task 3 ✓
- Live computation: refresh button streaming → Task 5 (`_render_refresh_button`) ✓
- Caching: `@st.cache_data(ttl=300)` on read paths → Task 5 ✓
- Backtest results in `st.session_state` → Task 9 ✓
- Existing `run_backtest()` unchanged → Task 3 (additive only) ✓
- Tests on the new computation surface (`iter_run_backtest`, plot builders) → Tasks 3, 4 ✓
- Existing 56 tests stay green → checked at end of every task ✓
- Streamlit dependency added → Task 1 ✓
- Error handling for missing parquet, missing series, bad backtest config → Tasks 5, 8, 9 ✓
- README updates → Task 10 ✓

**Type / signature consistency:** `BacktestProgress` (model, window, n_windows, elapsed_s, running_mape) used identically in Task 3 (definition) and Task 9 (consumption). `iter_run_backtest` signature matches `run_backtest` plus generator return type. `cross_validate_iter` yields `tuple[int, pl.DataFrame]` consistent across Tasks 2 and 3. Plot builder signatures in Task 4 match exactly what's called in Tasks 8 and 9.

**Placeholder scan:** No TBDs, no "implement later", every step has actual code or an exact command. The one judgment call is in Task 9 Step 3 (the curl smoke test) — if the user's machine doesn't have curl available (rare on modern Windows but possible), they can replace it with a Python `urllib.request.urlopen('http://localhost:8501')` check, which is also a one-liner.
