"""Three forecasters with a common interface, plus a backtest harness.

All three implementations expose the same surface:

    fit(df)           # df is polars with columns ['period_start', 'value']
    predict(h)        # returns ['period_start', 'point', 'lower_80', 'upper_80']
    cross_validate(df, horizon, n_windows)
                      # returns ['window', 'period_start', 'point',
                      #          'lower_80', 'upper_80', 'actual']

``run_backtest`` runs all three against a single series and returns
``(detailed_results, per_model_metrics)``. Every random source is seeded so
that the same inputs produce the same outputs run-to-run.
"""

from __future__ import annotations

import logging
import os
import random
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from prophet import Prophet
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

from .store import read_series

DEFAULT_SEED = 42

# Quiet the chatty libraries down to WARNING — they print INFO on every fit.
for _logger in ("prophet", "cmdstanpy", "lightgbm", "statsforecast"):
    logging.getLogger(_logger).setLevel(logging.WARNING)


@contextmanager
def _seeded(seed: int) -> Any:
    """Set seeds for python ``random`` and numpy for the duration of the block.

    Restores the previous states on exit so a forecaster fit in CV doesn't
    leak its RNG state into the caller.
    """
    py_state = random.getstate()
    np_state = np.random.get_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def _validate_input(df: pl.DataFrame) -> pl.DataFrame:
    if "period_start" not in df.columns or "value" not in df.columns:
        raise ValueError(
            "Input DataFrame must have 'period_start' and 'value' columns; "
            f"got {df.columns}"
        )
    out = df.select(["period_start", "value"]).sort("period_start")
    if out["value"].null_count() > 0:
        raise ValueError(
            "Input contains null values. Drop or impute them before fitting."
        )
    return out


def _next_n_months(start: pd.Timestamp, n: int) -> list[pd.Timestamp]:
    return list(pd.date_range(start=start, periods=n, freq="MS"))


# --------------------------------------------------------------------------- #
# Base class
# --------------------------------------------------------------------------- #


class BaseForecaster(ABC):
    """Common surface and a default time-series CV implementation."""

    seed: int

    @abstractmethod
    def fit(self, df: pl.DataFrame) -> None: ...

    @abstractmethod
    def predict(self, horizon: int) -> pl.DataFrame: ...

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


# --------------------------------------------------------------------------- #
# 1. StatsForecast AutoARIMA
# --------------------------------------------------------------------------- #


class StatsForecastAutoARIMA(BaseForecaster):
    """Auto-tuned seasonal ARIMA via Nixtla's ``statsforecast``."""

    def __init__(self, seed: int = DEFAULT_SEED, season_length: int = 12) -> None:
        self.seed = seed
        self.season_length = season_length
        self._sf: StatsForecast | None = None

    def fit(self, df: pl.DataFrame) -> None:
        clean = _validate_input(df)
        pdf = clean.to_pandas()
        train = pd.DataFrame(
            {
                "unique_id": "series_0",
                "ds": pd.to_datetime(pdf["period_start"]),
                "y": pdf["value"].astype(float),
            }
        )
        with _seeded(self.seed), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sf = StatsForecast(
                models=[AutoARIMA(season_length=self.season_length)],
                freq="MS",
                n_jobs=1,
            )
            sf.fit(train)
        self._sf = sf

    def predict(self, horizon: int) -> pl.DataFrame:
        if self._sf is None:
            raise RuntimeError("Call fit() before predict().")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = self._sf.predict(h=horizon, level=[80])
        if hasattr(forecast, "to_pandas"):
            forecast = forecast.to_pandas()
        if not isinstance(forecast, pd.DataFrame):
            forecast = pd.DataFrame(forecast)
        cols = list(forecast.columns)
        model_col = next(
            c for c in cols if c not in {"unique_id", "ds"} and "-" not in c
        )
        lo_col = next(c for c in cols if c.endswith("-lo-80"))
        hi_col = next(c for c in cols if c.endswith("-hi-80"))
        out = pl.from_pandas(forecast[["ds", model_col, lo_col, hi_col]])
        return out.rename(
            {
                "ds": "period_start",
                model_col: "point",
                lo_col: "lower_80",
                hi_col: "upper_80",
            }
        ).with_columns(pl.col("period_start").cast(pl.Date))


# --------------------------------------------------------------------------- #
# 2. Prophet
# --------------------------------------------------------------------------- #


class ProphetForecaster(BaseForecaster):
    """Facebook Prophet at MAP (default) — deterministic given the same data."""

    def __init__(self, seed: int = DEFAULT_SEED) -> None:
        self.seed = seed
        self._model: Prophet | None = None
        self._last_period: pd.Timestamp | None = None

    def fit(self, df: pl.DataFrame) -> None:
        clean = _validate_input(df)
        pdf = clean.to_pandas()
        train = pd.DataFrame(
            {
                "ds": pd.to_datetime(pdf["period_start"]),
                "y": pdf["value"].astype(float),
            }
        )
        with _seeded(self.seed), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Suppress cmdstanpy stdout chatter
            prev = os.environ.get("CMDSTANPY_LOG", "")
            os.environ["CMDSTANPY_LOG"] = "WARNING"
            try:
                model = Prophet(
                    interval_width=0.80,
                    yearly_seasonality=True,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                )
                model.fit(train)
            finally:
                if prev:
                    os.environ["CMDSTANPY_LOG"] = prev
                else:
                    os.environ.pop("CMDSTANPY_LOG", None)
        self._model = model
        self._last_period = train["ds"].iloc[-1]

    def predict(self, horizon: int) -> pl.DataFrame:
        if self._model is None or self._last_period is None:
            raise RuntimeError("Call fit() before predict().")
        future = pd.DataFrame(
            {
                "ds": _next_n_months(
                    self._last_period + pd.offsets.MonthBegin(1), horizon
                )
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = self._model.predict(future)
        out = pl.from_pandas(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]])
        return out.rename(
            {
                "ds": "period_start",
                "yhat": "point",
                "yhat_lower": "lower_80",
                "yhat_upper": "upper_80",
            }
        ).with_columns(pl.col("period_start").cast(pl.Date))


# --------------------------------------------------------------------------- #
# 3. LightGBM with lag + rolling + calendar features
# --------------------------------------------------------------------------- #


class LightGBMForecaster(BaseForecaster):
    """Gradient-boosted recursive forecaster.

    Features:
      * Lags 1, 2, 3, 6, 12, 24
      * Rolling means over windows 3, 6, 12 (computed from lag-1 backwards
        so they don't leak the current observation)
      * Calendar: month-of-year (one-hot via integer), quarter, year-trend
    Prediction intervals come from the 80th percentile of training-data
    naive-1 absolute differences (the same statistic that anchors MASE).
    In-sample model residuals are too narrow because the GBM nearly
    memorizes the training set; LightGBM's quantile regression on this data
    is unreliable. Naive-1 residuals capture typical month-to-month
    volatility and produce well-ordered intervals run-to-run.
    """

    LAGS: tuple[int, ...] = (1, 2, 3, 6, 12, 24)
    ROLLING_WINDOWS: tuple[int, ...] = (3, 6, 12)

    def __init__(
        self,
        seed: int = DEFAULT_SEED,
        n_estimators: int = 400,
        learning_rate: float = 0.05,
    ) -> None:
        self.seed = seed
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self._model_point: lgb.LGBMRegressor | None = None
        self._residual_halfwidth_80: float = 0.0
        self._feature_columns: list[str] | None = None
        self._history: pd.DataFrame | None = None  # ds, value (most recent last)

    @classmethod
    def _build_features(cls, history: pd.DataFrame) -> pd.DataFrame:
        """Given a (ds, value) history, return a feature frame keyed by ds."""
        df = history.copy().sort_values("ds").reset_index(drop=True)
        df["value"] = df["value"].astype(float)
        for lag in cls.LAGS:
            df[f"lag_{lag}"] = df["value"].shift(lag)
        for w in cls.ROLLING_WINDOWS:
            df[f"rollmean_{w}"] = df["value"].shift(1).rolling(w).mean()
        df["month"] = df["ds"].dt.month
        df["quarter"] = df["ds"].dt.quarter
        df["year_trend"] = df["ds"].dt.year - df["ds"].dt.year.min()
        return df

    def _feature_matrix(
        self, df: pd.DataFrame, feature_cols: Sequence[str]
    ) -> pd.DataFrame:
        return df[list(feature_cols)]

    def fit(self, df: pl.DataFrame) -> None:
        clean = _validate_input(df)
        pdf = clean.to_pandas().rename(columns={"period_start": "ds"})
        pdf["ds"] = pd.to_datetime(pdf["ds"])
        feats = self._build_features(pdf).dropna().reset_index(drop=True)
        if feats.empty:
            raise ValueError(
                f"Not enough history to build features (need at least "
                f"{max(self.LAGS) + max(self.ROLLING_WINDOWS) + 1} observations)"
            )
        feature_cols = [
            *[f"lag_{lag}" for lag in self.LAGS],
            *[f"rollmean_{w}" for w in self.ROLLING_WINDOWS],
            "month",
            "quarter",
            "year_trend",
        ]
        X = feats[feature_cols]  # noqa: N806 — sklearn convention
        y = feats["value"]

        with _seeded(self.seed), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model_point = lgb.LGBMRegressor(
                objective="regression",
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=31,
                min_child_samples=5,
                random_state=self.seed,
                seed=self.seed,
                deterministic=True,
                force_row_wise=True,
                verbose=-1,
            )
            self._model_point.fit(X, y)
            naive_diffs = np.abs(np.diff(pdf["value"].to_numpy(dtype=float)))
            self._residual_halfwidth_80 = (
                float(np.quantile(naive_diffs, 0.80)) if naive_diffs.size else 0.0
            )

        self._feature_columns = feature_cols
        self._history = pdf[["ds", "value"]].copy()

    def predict(self, horizon: int) -> pl.DataFrame:
        if (
            self._model_point is None
            or self._history is None
            or self._feature_columns is None
        ):
            raise RuntimeError("Call fit() before predict().")

        history = self._history.copy()
        last_ds = history["ds"].iloc[-1]
        future_dates = _next_n_months(last_ds + pd.offsets.MonthBegin(1), horizon)
        rows: list[dict[str, Any]] = []
        halfwidth = self._residual_halfwidth_80

        for ds in future_dates:
            extended = pd.concat(
                [history, pd.DataFrame({"ds": [ds], "value": [np.nan]})],
                ignore_index=True,
            )
            feats = self._build_features(extended).iloc[[-1]]
            X = feats[self._feature_columns]  # noqa: N806 — sklearn convention
            point = float(self._model_point.predict(X)[0])
            rows.append(
                {
                    "period_start": ds.date(),
                    "point": point,
                    "lower_80": point - halfwidth,
                    "upper_80": point + halfwidth,
                }
            )
            # Recursive: feed the point forecast back as the next period's value
            history = pd.concat(
                [history, pd.DataFrame({"ds": [ds], "value": [point]})],
                ignore_index=True,
            )

        return pl.DataFrame(rows).with_columns(
            pl.col("period_start").cast(pl.Date),
            pl.col("point").cast(pl.Float64),
            pl.col("lower_80").cast(pl.Float64),
            pl.col("upper_80").cast(pl.Float64),
        )


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def _mape(actual: np.ndarray[Any, Any], point: np.ndarray[Any, Any]) -> float:
    nz = actual != 0
    if not nz.any():
        return float("nan")
    return float(np.mean(np.abs((actual[nz] - point[nz]) / actual[nz])) * 100)


def _smape(actual: np.ndarray[Any, Any], point: np.ndarray[Any, Any]) -> float:
    denom = np.abs(actual) + np.abs(point)
    nz = denom > 0
    if not nz.any():
        return float("nan")
    return float(np.mean(2 * np.abs(actual[nz] - point[nz]) / denom[nz]) * 100)


def _mase(
    actual: np.ndarray[Any, Any],
    point: np.ndarray[Any, Any],
    train_values: np.ndarray[Any, Any],
) -> float:
    if len(train_values) < 2:
        return float("nan")
    naive_mae = float(np.mean(np.abs(np.diff(train_values))))
    if naive_mae == 0:
        return float("nan")
    return float(np.mean(np.abs(actual - point)) / naive_mae)


def _per_model_metrics(
    cv_results: pl.DataFrame, train_values: np.ndarray[Any, Any]
) -> pl.DataFrame:
    rows = []
    for model_name, sub in cv_results.group_by("model"):
        actual = sub["actual"].to_numpy().astype(float)
        point = sub["point"].to_numpy().astype(float)
        rows.append(
            {
                "model": model_name[0] if isinstance(model_name, tuple) else model_name,
                "n_obs": int(sub.height),
                "mape": _mape(actual, point),
                "smape": _smape(actual, point),
                "mase": _mase(actual, point, train_values),
            }
        )
    return pl.DataFrame(rows).sort("model")


# --------------------------------------------------------------------------- #
# Top-level backtest
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BacktestResult:
    series_id: str
    horizon: int
    n_windows: int
    cv_details: pl.DataFrame  # one row per (model, window, period_start)
    metrics: pl.DataFrame  # one row per model with mape/smape/mase


def run_backtest(
    series_id: str,
    horizon: int = 6,
    n_windows: int = 8,
    *,
    obs_path: Path | str | None = None,
    seed: int = DEFAULT_SEED,
) -> BacktestResult:
    """Run all three forecasters with rolling-origin CV on one series.

    Returns a :class:`BacktestResult` containing both the per-fold predictions
    (for plotting) and per-model accuracy metrics. Deterministic given the
    same series data and seed.
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
    for name, fcst in forecasters:
        cv = fcst.cross_validate(series, horizon=horizon, n_windows=n_windows)
        cv = cv.with_columns(model=pl.lit(name))
        detail_frames.append(cv)

    cv_details = pl.concat(detail_frames).sort(["model", "window", "period_start"])

    train_values = series["value"].to_numpy().astype(float)
    metrics = _per_model_metrics(cv_details, train_values)

    return BacktestResult(
        series_id=series_id,
        horizon=horizon,
        n_windows=n_windows,
        cv_details=cv_details,
        metrics=metrics,
    )
