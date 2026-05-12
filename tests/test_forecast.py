"""Tests for the forecasting module.

These exercise the common interface (fit / predict / cross_validate),
input validation, and determinism. They use small synthetic series so the
suite stays fast — Prophet is the slowest but still finishes in a couple of
seconds at this scale.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from usda_sandbox.forecast import (
    BaseForecaster,
    LightGBMForecaster,
    ProphetForecaster,
    StatsForecastAutoARIMA,
    run_backtest,
)


def _synthetic_monthly(n: int = 60, seed: int = 0) -> pl.DataFrame:
    """Trend + seasonal monthly series with a fixed seed."""
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

    rng = np.random.default_rng(seed)
    t = np.arange(n)
    values = (
        100.0
        + 0.5 * t
        + 10.0 * np.sin(2 * np.pi * t / 12)
        + rng.normal(0, 2, size=n)
    )
    return pl.DataFrame({"period_start": dates, "value": values}).with_columns(
        pl.col("period_start").cast(pl.Date),
        pl.col("value").cast(pl.Float64),
    )


@pytest.fixture(scope="module")
def synthetic_series() -> pl.DataFrame:
    return _synthetic_monthly(60, seed=0)


@pytest.fixture(scope="module")
def synthetic_obs_parquet(
    synthetic_series: pl.DataFrame, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """Write the synthetic series to a parquet shaped like ``observations.parquet``."""
    path = tmp_path_factory.mktemp("forecast_obs") / "observations.parquet"
    obs = synthetic_series.with_columns(
        series_id=pl.lit("synthetic_test"),
        series_name=pl.lit("Synthetic test series"),
        commodity=pl.lit("test"),
        metric=pl.lit("price"),
        unit=pl.lit("USD/cwt"),
        frequency=pl.lit("monthly"),
        period_end=pl.col("period_start"),
        source_file=pl.lit("synthetic.xlsx"),
        source_sheet=pl.lit("Sheet1"),
        ingested_at=pl.lit(None).cast(pl.Datetime("us", "UTC")),
    )
    obs.write_parquet(path)
    return path


# --------------------------------------------------------------------------- #
# Per-forecaster API surface
# --------------------------------------------------------------------------- #


_PREDICT_COLS = ["period_start", "point", "lower_80", "upper_80"]
_CV_COLS = ["window", "period_start", "point", "lower_80", "upper_80", "actual"]


@pytest.mark.parametrize(
    "forecaster_cls",
    [StatsForecastAutoARIMA, ProphetForecaster, LightGBMForecaster],
)
def test_predict_shape_and_ordering(
    forecaster_cls: type[BaseForecaster], synthetic_series: pl.DataFrame
) -> None:
    m = forecaster_cls(seed=42)
    m.fit(synthetic_series)
    pred = m.predict(horizon=4)

    assert pred.columns == _PREDICT_COLS
    assert pred.height == 4
    # period_start strictly after the training data, and contiguous monthly
    last_train = synthetic_series["period_start"].max()
    assert pred["period_start"][0] > last_train
    # 80% PI is always well-ordered: lower <= point <= upper
    assert (pred["lower_80"] <= pred["point"]).all()
    assert (pred["point"] <= pred["upper_80"]).all()


@pytest.mark.parametrize(
    "forecaster_cls",
    [StatsForecastAutoARIMA, ProphetForecaster, LightGBMForecaster],
)
def test_predict_without_fit_raises(
    forecaster_cls: type[BaseForecaster],
) -> None:
    m = forecaster_cls(seed=42)
    with pytest.raises(RuntimeError, match="fit"):
        m.predict(horizon=3)


# --------------------------------------------------------------------------- #
# Cross-validate
# --------------------------------------------------------------------------- #


def test_cross_validate_window_structure(synthetic_series: pl.DataFrame) -> None:
    # Use AutoARIMA (fastest) for the structural check
    m = StatsForecastAutoARIMA(seed=42)
    cv = m.cross_validate(synthetic_series, horizon=3, n_windows=2)

    assert cv.columns == _CV_COLS
    assert cv.height == 2 * 3
    assert sorted(cv["window"].unique().to_list()) == [0, 1]
    # Each window's period_starts come from the actual series, in order
    n = synthetic_series.height
    expected_window_0_start = synthetic_series["period_start"][n - 6]
    assert cv.filter(pl.col("window") == 0)["period_start"][0] == expected_window_0_start
    # Actuals join correctly
    actuals = cv["actual"].to_list()
    assert len(actuals) == 6 and all(a is not None for a in actuals)


def test_cross_validate_rejects_too_few_obs(synthetic_series: pl.DataFrame) -> None:
    short = synthetic_series.head(10)
    with pytest.raises(ValueError, match="at least"):
        StatsForecastAutoARIMA().cross_validate(short, horizon=3, n_windows=4)


def test_cross_validate_rejects_bad_args(synthetic_series: pl.DataFrame) -> None:
    m = StatsForecastAutoARIMA()
    with pytest.raises(ValueError, match="positive"):
        m.cross_validate(synthetic_series, horizon=0, n_windows=2)
    with pytest.raises(ValueError, match="positive"):
        m.cross_validate(synthetic_series, horizon=3, n_windows=0)


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


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #


def test_fit_rejects_missing_columns() -> None:
    df = pl.DataFrame({"date": [date(2020, 1, 1)], "y": [1.0]})
    with pytest.raises(ValueError, match="columns"):
        StatsForecastAutoARIMA().fit(df)


def test_fit_rejects_nulls(synthetic_series: pl.DataFrame) -> None:
    bad = synthetic_series.with_columns(
        value=pl.when(pl.int_range(synthetic_series.height) == 5)
        .then(None)
        .otherwise(pl.col("value"))
    )
    with pytest.raises(ValueError, match="null"):
        StatsForecastAutoARIMA().fit(bad)


# --------------------------------------------------------------------------- #
# Determinism — same seed, same data, same outputs
# --------------------------------------------------------------------------- #


def test_autoarima_is_deterministic(synthetic_series: pl.DataFrame) -> None:
    a = StatsForecastAutoARIMA(seed=42)
    b = StatsForecastAutoARIMA(seed=42)
    a.fit(synthetic_series)
    b.fit(synthetic_series)
    pa = a.predict(horizon=4)
    pb = b.predict(horizon=4)
    assert pa.equals(pb)


def test_lightgbm_is_deterministic(synthetic_series: pl.DataFrame) -> None:
    a = LightGBMForecaster(seed=42)
    b = LightGBMForecaster(seed=42)
    a.fit(synthetic_series)
    b.fit(synthetic_series)
    pa = a.predict(horizon=4)
    pb = b.predict(horizon=4)
    assert pa.equals(pb)


# --------------------------------------------------------------------------- #
# run_backtest — end-to-end
# --------------------------------------------------------------------------- #


def test_run_backtest_end_to_end(synthetic_obs_parquet: Path) -> None:
    result = run_backtest(
        "synthetic_test",
        horizon=3,
        n_windows=2,
        obs_path=synthetic_obs_parquet,
    )
    assert result.series_id == "synthetic_test"
    assert result.horizon == 3
    assert result.n_windows == 2

    # 3 models * 2 windows * 3 horizon = 18 rows
    assert result.cv_details.height == 18
    assert set(result.cv_details["model"].unique().to_list()) == {
        "AutoARIMA",
        "Prophet",
        "LightGBM",
    }

    # Metrics: one row per model with finite mape/smape/mase
    assert result.metrics.height == 3
    for col in ("mape", "smape", "mase"):
        assert result.metrics[col].is_finite().all()


def test_run_backtest_unknown_series_raises(synthetic_obs_parquet: Path) -> None:
    with pytest.raises(ValueError, match="No observations"):
        run_backtest("does_not_exist", obs_path=synthetic_obs_parquet)


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


def test_iter_run_backtest_respects_models_filter(
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
            models=["AutoARIMA", "LightGBM"],  # exclude Prophet
        )
    )
    progress_events = [e for e in items if isinstance(e, BacktestProgress)]
    final = items[-1]
    assert isinstance(final, BacktestResult)

    # Only the two requested models, 2 windows each = 4 progress events
    assert len(progress_events) == 4
    seen_models = {e.model for e in progress_events}
    assert seen_models == {"AutoARIMA", "LightGBM"}

    # Final result also only contains the requested models
    assert set(final.metrics["model"].to_list()) == {"AutoARIMA", "LightGBM"}


def test_iter_run_backtest_unknown_model_raises(
    synthetic_obs_parquet: Path,
) -> None:
    from usda_sandbox.forecast import iter_run_backtest

    with pytest.raises(ValueError, match="No forecasters match"):
        list(
            iter_run_backtest(
                "synthetic_test",
                horizon=3,
                n_windows=2,
                obs_path=synthetic_obs_parquet,
                models=["NotARealModel"],
            )
        )


# --------------------------------------------------------------------------- #
# Exog regressor support (v0.2b)
# --------------------------------------------------------------------------- #


def _exog_series(monthly: pl.DataFrame, *, jitter: float = 0.5) -> pl.DataFrame:
    """Build a synthetic exog DataFrame aligned with the monthly series."""
    rng = np.random.default_rng(1)
    n = monthly.height
    return pl.DataFrame(
        {
            "period_start": monthly["period_start"].to_list(),
            "reg_a": (monthly["value"].to_numpy() * 0.9 + rng.normal(0, jitter, n)).tolist(),
            "reg_b": (monthly["value"].to_numpy() * 0.5 + 20 + rng.normal(0, jitter, n)).tolist(),
        }
    ).with_columns(
        pl.col("period_start").cast(pl.Date),
        pl.col("reg_a").cast(pl.Float64),
        pl.col("reg_b").cast(pl.Float64),
    )


@pytest.mark.parametrize(
    "forecaster_cls",
    [StatsForecastAutoARIMA, ProphetForecaster, LightGBMForecaster],
)
def test_fit_predict_with_exog_returns_correct_shape(
    forecaster_cls: type[BaseForecaster], synthetic_series: pl.DataFrame
) -> None:
    exog = _exog_series(synthetic_series)
    exog_train = exog.head(synthetic_series.height - 4)
    exog_future = exog.tail(4).drop("period_start").select(["reg_a", "reg_b"])

    m = forecaster_cls(seed=42)
    m.fit(synthetic_series.head(synthetic_series.height - 4), exog=exog_train)
    pred = m.predict(horizon=4, exog_future=exog_future)

    assert pred.columns == _PREDICT_COLS
    assert pred.height == 4
    assert (pred["lower_80"] <= pred["point"]).all()
    assert (pred["point"] <= pred["upper_80"]).all()


@pytest.mark.parametrize(
    "forecaster_cls",
    [StatsForecastAutoARIMA, ProphetForecaster, LightGBMForecaster],
)
def test_fit_without_exog_unchanged_from_v02a(
    forecaster_cls: type[BaseForecaster], synthetic_series: pl.DataFrame
) -> None:
    """Calling fit(df) with no exog must produce the same result as before."""
    m = forecaster_cls(seed=42)
    m.fit(synthetic_series)
    pred = m.predict(horizon=4)
    assert pred.height == 4
    assert pred.columns == _PREDICT_COLS


def test_cross_validate_with_exog_threads_through(synthetic_series: pl.DataFrame) -> None:
    exog = _exog_series(synthetic_series)
    m = StatsForecastAutoARIMA(seed=42)
    cv = m.cross_validate(synthetic_series, horizon=3, n_windows=2, exog=exog)
    assert cv.height == 6
    assert cv.columns == _CV_COLS


def test_predict_raises_when_exog_future_length_mismatches_horizon(
    synthetic_series: pl.DataFrame,
) -> None:
    exog = _exog_series(synthetic_series)
    m = StatsForecastAutoARIMA(seed=42)
    m.fit(synthetic_series.head(50), exog=exog.head(50))
    bad_future = exog.tail(3).drop("period_start")
    with pytest.raises(ValueError, match="exog_future"):
        m.predict(horizon=5, exog_future=bad_future)


def test_fit_drops_rows_where_any_exog_column_is_null(
    synthetic_series: pl.DataFrame,
) -> None:
    exog = _exog_series(synthetic_series).with_columns(
        reg_a=pl.when(pl.int_range(synthetic_series.height) == 5)
        .then(None)
        .otherwise(pl.col("reg_a"))
    )
    m = StatsForecastAutoARIMA(seed=42)
    m.fit(synthetic_series, exog=exog)
    pred = m.predict(
        horizon=2,
        exog_future=exog.tail(2).drop("period_start").select(["reg_a", "reg_b"]),
    )
    assert pred.height == 2


# --------------------------------------------------------------------------- #
# Catalog-driven exog lookup (Task 7)
# --------------------------------------------------------------------------- #


def test_iter_run_backtest_loads_catalog_exog(
    synthetic_obs_parquet: Path, tmp_path: Path
) -> None:
    """When the target's SeriesDefinition has exogenous_regressors,
    iter_run_backtest fetches them and passes them through."""
    from usda_sandbox.catalog import SeriesDefinition, save_catalog
    from usda_sandbox.forecast import (
        BacktestResult,
        iter_run_backtest,
    )

    # Use the existing synthetic obs parquet's series_id and add a synthetic
    # exog series to the same parquet
    obs = pl.read_parquet(synthetic_obs_parquet)
    exog_rows = obs.with_columns(
        series_id=pl.lit("test_exog_1"),
        series_name=pl.lit("Test exog series"),
        value=pl.col("value") * 0.9,
    )
    obs_with_exog = pl.concat([obs, exog_rows], how="vertical_relaxed")
    obs_path = tmp_path / "obs_with_exog.parquet"
    obs_with_exog.write_parquet(obs_path)

    # Build a catalog with the target pointing at the exog series
    catalog = [
        SeriesDefinition(
            series_id="synthetic_test",
            series_name="Synthetic test series",
            commodity="test",
            metric="price",
            unit="USD/cwt",
            frequency="monthly",
            source_file="synthetic.xlsx",
            source_sheet="Sheet1",
            header_rows_to_skip=0,
            value_columns=["B"],
            date_column="A",
            notes="",
            exogenous_regressors=["test_exog_1"],
        ),
        SeriesDefinition(
            series_id="test_exog_1",
            series_name="Test exog 1",
            commodity="test",
            metric="exog",
            unit="USD/cwt",
            frequency="monthly",
            source_file="x.xlsx",
            source_sheet="Sheet1",
            header_rows_to_skip=0,
            value_columns=["B"],
            date_column="A",
            notes="",
            forecastable=False,
        ),
    ]
    catalog_path = tmp_path / "catalog.json"
    save_catalog(catalog_path, catalog)

    items = list(
        iter_run_backtest(
            "synthetic_test",
            horizon=3,
            n_windows=2,
            obs_path=obs_path,
            catalog_path=catalog_path,
            models=["AutoARIMA"],  # one model for speed
        )
    )
    final = items[-1]
    assert isinstance(final, BacktestResult)
    # The CV still produced the expected number of rows (1 model x 2 windows x 3 horizon)
    assert final.cv_details.height == 6


def test_iter_run_backtest_no_catalog_path_uses_v02a_behavior(
    synthetic_obs_parquet: Path,
) -> None:
    """If catalog_path is None and the catalog file doesn't exist at the
    default path, the backtest runs without exog (v0.2a behavior)."""
    from usda_sandbox.forecast import BacktestResult, iter_run_backtest

    items = list(
        iter_run_backtest(
            "synthetic_test",
            horizon=3,
            n_windows=2,
            obs_path=synthetic_obs_parquet,
            catalog_path=None,
            models=["AutoARIMA"],
        )
    )
    final = items[-1]
    assert isinstance(final, BacktestResult)
    assert final.cv_details.height == 6
