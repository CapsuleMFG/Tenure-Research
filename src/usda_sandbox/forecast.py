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
import time
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


def _align_exog(
    df: pl.DataFrame, exog: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Align df and exog on period_start, dropping rows where any exog column
    is null."""
    if "period_start" not in exog.columns:
        raise ValueError("exog must have a 'period_start' column")
    reg_cols = [c for c in exog.columns if c != "period_start"]
    if not reg_cols:
        raise ValueError("exog must have at least one regressor column")
    merged = df.join(exog, on="period_start", how="inner")
    merged = merged.drop_nulls(subset=reg_cols)
    aligned_df = merged.select(df.columns).sort("period_start")
    aligned_exog = merged.select(["period_start", *reg_cols]).sort("period_start")
    return aligned_df, aligned_exog


def _check_exog_future(exog_future: pl.DataFrame, horizon: int) -> None:
    if exog_future.height != horizon:
        raise ValueError(
            f"exog_future must have exactly horizon ({horizon}) rows; "
            f"got {exog_future.height}"
        )


def _next_n_months(start: pd.Timestamp, n: int) -> list[pd.Timestamp]:
    return list(pd.date_range(start=start, periods=n, freq="MS"))


# --------------------------------------------------------------------------- #
# Base class
# --------------------------------------------------------------------------- #


class BaseForecaster(ABC):
    """Common surface and a default time-series CV implementation."""

    seed: int

    @abstractmethod
    def fit(
        self, df: pl.DataFrame, exog: pl.DataFrame | None = None
    ) -> None: ...

    @abstractmethod
    def predict(
        self, horizon: int, exog_future: pl.DataFrame | None = None
    ) -> pl.DataFrame: ...

    def cross_validate_iter(
        self,
        df: pl.DataFrame,
        horizon: int,
        n_windows: int,
        exog: pl.DataFrame | None = None,
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
            if exog is not None:
                train_dates = train["period_start"].to_list()
                target_dates = target["period_start"].to_list()
                exog_train = exog.filter(
                    pl.col("period_start").is_in(train_dates)
                )
                exog_future = (
                    exog.filter(
                        pl.col("period_start").is_in(target_dates)
                    )
                    .drop("period_start")
                )
                self.fit(train, exog=exog_train)
                pred = self.predict(horizon, exog_future=exog_future)
            else:
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
        self,
        df: pl.DataFrame,
        horizon: int,
        n_windows: int,
        exog: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Rolling-origin CV with ``n_windows`` non-overlapping forecast blocks.

        Window 0 is the oldest cutoff; window ``n_windows - 1`` is the most
        recent. Each window holds out ``horizon`` observations after fitting
        on everything before them. Implemented on top of
        :meth:`cross_validate_iter`.
        """
        frames = [
            merged
            for _, merged in self.cross_validate_iter(df, horizon, n_windows, exog=exog)
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
        self._exog_cols: list[str] = []
        self._last_train_date: pd.Timestamp | None = None

    def fit(self, df: pl.DataFrame, exog: pl.DataFrame | None = None) -> None:
        clean = _validate_input(df)
        if exog is not None:
            clean, aligned_exog = _align_exog(clean, exog)
            self._exog_cols = [c for c in aligned_exog.columns if c != "period_start"]
        else:
            self._exog_cols = []
        pdf = clean.to_pandas()
        train = pd.DataFrame(
            {
                "unique_id": "series_0",
                "ds": pd.to_datetime(pdf["period_start"]),
                "y": pdf["value"].astype(float),
            }
        )
        if self._exog_cols:
            exog_pdf = aligned_exog.to_pandas()
            for col in self._exog_cols:
                train[col] = exog_pdf[col].astype(float).values
        self._last_train_date = train["ds"].max()
        with _seeded(self.seed), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sf = StatsForecast(
                models=[AutoARIMA(season_length=self.season_length)],
                freq="MS",
                n_jobs=1,
            )
            sf.fit(train)
        self._sf = sf

    def predict(self, horizon: int, exog_future: pl.DataFrame | None = None) -> pl.DataFrame:
        if self._sf is None or self._last_train_date is None:
            raise RuntimeError("Call fit() before predict().")
        x_df: pd.DataFrame | None = None
        if self._exog_cols:
            if exog_future is None:
                raise ValueError(
                    "model was fit with exog; exog_future must be provided"
                )
            _check_exog_future(exog_future, horizon)
            future_dates = pd.date_range(
                start=self._last_train_date + pd.offsets.MonthBegin(1),
                periods=horizon,
                freq="MS",
            )
            x_df = pd.DataFrame({"unique_id": "series_0", "ds": future_dates})
            for col in self._exog_cols:
                x_df[col] = exog_future[col].to_pandas().astype(float).values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if x_df is not None:
                forecast = self._sf.predict(h=horizon, level=[80], X_df=x_df)
            else:
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
        self._exog_cols: list[str] = []

    def fit(self, df: pl.DataFrame, exog: pl.DataFrame | None = None) -> None:
        clean = _validate_input(df)
        if exog is not None:
            clean, aligned_exog = _align_exog(clean, exog)
            self._exog_cols = [c for c in aligned_exog.columns if c != "period_start"]
        else:
            self._exog_cols = []
        pdf = clean.to_pandas()
        train = pd.DataFrame(
            {
                "ds": pd.to_datetime(pdf["period_start"]),
                "y": pdf["value"].astype(float),
            }
        )
        if self._exog_cols:
            exog_pdf = aligned_exog.to_pandas()
            for col in self._exog_cols:
                train[col] = exog_pdf[col].astype(float).values
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
                for col in self._exog_cols:
                    model.add_regressor(col)
                model.fit(train)
            finally:
                if prev:
                    os.environ["CMDSTANPY_LOG"] = prev
                else:
                    os.environ.pop("CMDSTANPY_LOG", None)
        self._model = model
        self._last_period = train["ds"].iloc[-1]

    def predict(self, horizon: int, exog_future: pl.DataFrame | None = None) -> pl.DataFrame:
        if self._model is None or self._last_period is None:
            raise RuntimeError("Call fit() before predict().")
        if self._exog_cols:
            if exog_future is None:
                raise ValueError(
                    "model was fit with exog; exog_future must be provided"
                )
            _check_exog_future(exog_future, horizon)
        future = pd.DataFrame(
            {
                "ds": _next_n_months(
                    self._last_period + pd.offsets.MonthBegin(1), horizon
                )
            }
        )
        if self._exog_cols and exog_future is not None:
            for col in self._exog_cols:
                future[col] = exog_future[col].to_pandas().astype(float).values
        with _seeded(self.seed), warnings.catch_warnings():
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
        self._exog_cols: list[str] = []

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

    def fit(self, df: pl.DataFrame, exog: pl.DataFrame | None = None) -> None:
        clean = _validate_input(df)
        if exog is not None:
            clean, aligned_exog = _align_exog(clean, exog)
            self._exog_cols = [c for c in aligned_exog.columns if c != "period_start"]
        else:
            self._exog_cols = []
        pdf = clean.to_pandas().rename(columns={"period_start": "ds"})
        pdf["ds"] = pd.to_datetime(pdf["ds"])
        feats = self._build_features(pdf).dropna().reset_index(drop=True)
        if self._exog_cols:
            exog_pdf = aligned_exog.to_pandas()
            exog_pdf = exog_pdf.rename(columns={"period_start": "ds"})
            exog_pdf["ds"] = pd.to_datetime(exog_pdf["ds"])
            # Merge exog onto the feature frame (inner join on ds, after dropna)
            feats = feats.merge(
                exog_pdf[["ds", *self._exog_cols]], on="ds", how="inner"
            ).reset_index(drop=True)
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
            *self._exog_cols,
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

    def predict(self, horizon: int, exog_future: pl.DataFrame | None = None) -> pl.DataFrame:
        if (
            self._model_point is None
            or self._history is None
            or self._feature_columns is None
        ):
            raise RuntimeError("Call fit() before predict().")
        if self._exog_cols:
            if exog_future is None:
                raise ValueError(
                    "model was fit with exog; exog_future must be provided"
                )
            _check_exog_future(exog_future, horizon)
            # Pre-extract exog values as dict of lists for fast step-indexed access
            exog_arrays: dict[str, list[float]] = {
                col: exog_future[col].to_list() for col in self._exog_cols
            }
        else:
            exog_arrays = {}

        history = self._history.copy()
        last_ds = history["ds"].iloc[-1]
        future_dates = _next_n_months(last_ds + pd.offsets.MonthBegin(1), horizon)
        rows: list[dict[str, Any]] = []
        halfwidth = self._residual_halfwidth_80

        for step, ds in enumerate(future_dates):
            extended = pd.concat(
                [history, pd.DataFrame({"ds": [ds], "value": [np.nan]})],
                ignore_index=True,
            )
            feats = self._build_features(extended).iloc[[-1]].copy()
            # Inject exog columns for this step
            for col, vals in exog_arrays.items():
                feats[col] = vals[step]
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


def iter_run_backtest(
    series_id: str,
    horizon: int = 6,
    n_windows: int = 8,
    *,
    obs_path: Path | str | None = None,
    seed: int = DEFAULT_SEED,
    models: Sequence[str] | None = None,
    catalog_path: Path | str | None = "data/catalog.json",
) -> Iterator[BacktestProgress | BacktestResult]:
    """Generator variant of :func:`run_backtest`.

    Yields one :class:`BacktestProgress` event after each (model, window)
    completes (``len(models) * n_windows`` events total), then a single final
    :class:`BacktestResult` with the same content :func:`run_backtest` would
    return for the same inputs.

    ``models`` optionally restricts which forecasters are run.

    ``catalog_path`` (default ``data/catalog.json``) is used to look up the
    target series's ``exogenous_regressors`` field. If the catalog file
    exists and the target's regressors are non-empty, those series are
    fetched from ``obs_path`` and passed through to each forecaster's
    ``cross_validate_iter`` as ``exog``. If the catalog file does not exist
    or ``catalog_path=None``, the v0.2a path (no exog) runs unchanged.
    """
    series = read_series(series_id, obs_path)
    if series.is_empty():
        raise ValueError(f"No observations found for series_id={series_id!r}")
    series = series.filter(pl.col("value").is_not_null()).select(
        ["period_start", "value"]
    )

    exog = _load_exog_for_target(
        series_id=series_id,
        obs_path=obs_path,
        catalog_path=catalog_path,
    )

    all_forecasters: list[tuple[str, BaseForecaster]] = [
        ("AutoARIMA", StatsForecastAutoARIMA(seed=seed)),
        ("Prophet", ProphetForecaster(seed=seed)),
        ("LightGBM", LightGBMForecaster(seed=seed)),
    ]
    if models is None:
        forecasters = all_forecasters
    else:
        wanted = set(models)
        forecasters = [(n, f) for n, f in all_forecasters if n in wanted]
        if not forecasters:
            raise ValueError(
                f"No forecasters match the requested models: {sorted(wanted)}. "
                f"Available: {[n for n, _ in all_forecasters]}"
            )

    detail_frames: list[pl.DataFrame] = []
    started = time.time()

    for name, fcst in forecasters:
        per_model_frames: list[pl.DataFrame] = []
        for w, window_df in fcst.cross_validate_iter(
            series, horizon, n_windows, exog=exog
        ):
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
                running_mape=None if running != running else running,
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


def _load_exog_for_target(
    *,
    series_id: str,
    obs_path: Path | str | None,
    catalog_path: Path | str | None,
) -> pl.DataFrame | None:
    """If the catalog says the target has exogenous_regressors, load and
    pivot them. Returns None if no exog is configured or the catalog is
    unavailable."""
    if catalog_path is None:
        return None
    catalog_path = Path(catalog_path)
    if not catalog_path.exists():
        return None
    from .catalog import load_catalog
    from .store import read_observations

    catalog = load_catalog(catalog_path)
    by_id = {sd.series_id: sd for sd in catalog}
    target = by_id.get(series_id)
    if target is None or not target.exogenous_regressors:
        return None

    obs = read_observations(obs_path).collect()
    long = (
        obs.filter(pl.col("series_id").is_in(target.exogenous_regressors))
        .select(["series_id", "period_start", "value"])
    )
    if long.is_empty():
        return None
    wide = long.pivot(values="value", index="period_start", on="series_id").sort(
        "period_start"
    )
    return wide
