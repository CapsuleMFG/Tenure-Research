"""Precompute the headline forecast cache for the LivestockBrief dashboard.

The Streamlit app's home page must be instant — we can't fit AutoARIMA /
Prophet / LightGBM live on every visit. This module bakes a single JSON
file, ``data/clean/forecasts.json``, containing the winning model and
forward forecast for every forecastable series. The web app reads from
that cache; live fitting only happens behind the Forecast page's
*Advanced* expander when an analyst wants to re-run.

Schema
------
::

    {
      "generated_at": "2026-05-12T21:03:11Z",
      "horizon": 6,
      "n_windows": 12,
      "forward_horizon": 12,
      "by_series": {
        "<series_id>": {
          "series_name": "...",
          "commodity": "...",
          "unit": "USD/cwt",
          "winner_model": "AutoARIMA",
          "winner_metrics": {"mape": 6.29, "smape": 6.46, "mase": 2.94},
          "scoreboard": [
            {"model": "AutoARIMA", "mape": ..., "smape": ..., "mase": ...},
            ...
          ],
          "latest_actual": {"period_start": "2026-03-01", "value": 221.0},
          "prior_month_actual": {"period_start": "2026-02-01", "value": 216.0},
          "prior_year_actual": {"period_start": "2025-03-01", "value": 187.5},
          "forward": [
            {"period_start": "2026-04-01", "point": 233.57,
             "lower_80": 215.4, "upper_80": 251.7},
            ...
          ],
          "conformal_scale_h1": 1.32,
          "conformal_scale_h_last": 1.85
        },
        ...
      },
      "by_series_errors": {
        "<series_id>": "human-readable error message"
      }
    }

The cache is built by :func:`build_forecast_cache`. The CLI entrypoint is
``python -m usda_sandbox.precompute`` (registered below).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from .calibration import apply_conformal_scaling, conformal_scale_factors_per_horizon
from .catalog import load_catalog
from .forecast import (
    LightGBMForecaster,
    ProphetForecaster,
    StatsForecastAutoARIMA,
    run_backtest,
)
from .store import read_observations, read_series

logger = logging.getLogger(__name__)

DEFAULT_CACHE_PATH = Path("data/clean/forecasts.json")
DEFAULT_CV_HORIZON = 6
DEFAULT_N_WINDOWS = 12
DEFAULT_FORWARD_HORIZON = 12

_FORECASTER_REGISTRY = {
    "AutoARIMA": StatsForecastAutoARIMA,
    "Prophet": ProphetForecaster,
    "LightGBM": LightGBMForecaster,
}


def _next_n_months(start: date, n: int) -> list[date]:
    """Return ``n`` month-start dates after ``start`` (month-start convention)."""
    out: list[date] = []
    yr, mo = start.year, start.month
    for _ in range(n):
        mo += 1
        if mo == 13:
            mo = 1
            yr += 1
        out.append(date(yr, mo, 1))
    return out


def _series_forward_forecast(
    *,
    series_id: str,
    winner: str,
    cv_details: pl.DataFrame,
    cv_horizon: int,
    forward_horizon: int,
    obs_path: Path,
    catalog_path: Path,
    seed: int = 42,
) -> tuple[pl.DataFrame, list[float]]:
    """Fit the winning model on full history and forecast ``forward_horizon`` ahead.

    Returns the calibrated forecast DataFrame plus the per-horizon
    conformal scale factors actually applied (so they can be surfaced in
    the cache for diagnostics).
    """
    history = (
        read_series(series_id, obs_path)
        .filter(pl.col("value").is_not_null())
        .sort("period_start")
    )

    catalog = load_catalog(catalog_path)
    target_def = next((sd for sd in catalog if sd.series_id == series_id), None)
    regressor_ids = list(target_def.exogenous_regressors) if target_def else []

    fcst = _FORECASTER_REGISTRY[winner](seed=seed)

    if regressor_ids:
        exog_long = (
            read_observations(obs_path)
            .filter(pl.col("series_id").is_in(regressor_ids))
            .select(["series_id", "period_start", "value"])
            .collect()
        )
        exog_wide = exog_long.pivot(
            values="value", index="period_start", on="series_id"
        ).sort("period_start")
        aligned = history.select(["period_start", "value"]).join(
            exog_wide, on="period_start", how="inner"
        )
        history_for_fit = aligned.select(["period_start", "value"])
        exog_for_fit = aligned.select(["period_start", *regressor_ids])

        last_exog_row = exog_wide.tail(1)
        last_history_date = history_for_fit["period_start"].max()
        if last_history_date is None:
            raise ValueError(f"Series {series_id!r} has no history after exog alignment")
        future_dates = _next_n_months(last_history_date, forward_horizon)
        exog_future = pl.DataFrame(
            {
                "period_start": future_dates,
                **{col: [last_exog_row[col][0]] * forward_horizon for col in regressor_ids},
            }
        ).with_columns(pl.col("period_start").cast(pl.Date))

        fcst.fit(history_for_fit, exog=exog_for_fit)
        forward = fcst.predict(horizon=forward_horizon, exog_future=exog_future)
    else:
        fcst.fit(history.select(["period_start", "value"]))
        forward = fcst.predict(horizon=forward_horizon)

    per_h_scales = conformal_scale_factors_per_horizon(
        cv_details, model_name=winner, horizon=cv_horizon
    )
    forward_scales = per_h_scales + [per_h_scales[-1]] * max(
        0, forward.height - len(per_h_scales)
    )
    forward_scales = forward_scales[: forward.height]
    calibrated = apply_conformal_scaling(forward, scale=forward_scales)
    return calibrated, forward_scales


def _series_priors(history: pl.DataFrame) -> tuple[dict | None, dict | None, dict | None]:
    """Return (latest, prior_month, prior_year) actual records or ``None``."""
    non_null = history.filter(pl.col("value").is_not_null()).sort("period_start")
    if non_null.is_empty():
        return None, None, None

    latest = non_null.tail(1).row(0, named=True)
    latest_record = {
        "period_start": latest["period_start"].isoformat(),
        "value": float(latest["value"]),
    }

    prior_month_record: dict | None = None
    if non_null.height >= 2:
        prior_month = non_null.slice(non_null.height - 2, 1).row(0, named=True)
        prior_month_record = {
            "period_start": prior_month["period_start"].isoformat(),
            "value": float(prior_month["value"]),
        }

    prior_year_record: dict | None = None
    if non_null.height >= 13:
        prior_year = non_null.slice(non_null.height - 13, 1).row(0, named=True)
        prior_year_record = {
            "period_start": prior_year["period_start"].isoformat(),
            "value": float(prior_year["value"]),
        }

    return latest_record, prior_month_record, prior_year_record


def _series_sparkline(history: pl.DataFrame, months: int = 24) -> list[dict]:
    tail = history.filter(pl.col("value").is_not_null()).sort("period_start").tail(months)
    return [
        {
            "period_start": row["period_start"].isoformat(),
            "value": float(row["value"]),
        }
        for row in tail.iter_rows(named=True)
    ]


def _forward_to_records(forward: pl.DataFrame) -> list[dict]:
    return [
        {
            "period_start": row["period_start"].isoformat(),
            "point": float(row["point"]),
            "lower_80": float(row["lower_80"]),
            "upper_80": float(row["upper_80"]),
        }
        for row in forward.iter_rows(named=True)
    ]


def build_forecast_cache(
    *,
    obs_path: Path | str = "data/clean/observations.parquet",
    catalog_path: Path | str = "data/catalog.json",
    out_path: Path | str = DEFAULT_CACHE_PATH,
    cv_horizon: int = DEFAULT_CV_HORIZON,
    n_windows: int = DEFAULT_N_WINDOWS,
    forward_horizon: int = DEFAULT_FORWARD_HORIZON,
    only_series: Sequence[str] | None = None,
    sparkline_months: int = 24,
) -> Path:
    """Rebuild ``forecasts.json`` for every forecastable monthly series.

    For each catalog entry with ``forecastable=True`` and monthly frequency:

    1. Run :func:`run_backtest` (3 models × ``n_windows`` CV windows).
    2. Pick the winner by lowest MAPE.
    3. Refit the winner on full history and forecast ``forward_horizon`` months.
    4. Conformally calibrate the PI per horizon step from CV residuals.
    5. Record the latest, prior-month, and prior-year actuals; tail sparkline.

    Failures for individual series are recorded under ``by_series_errors``
    so one bad series doesn't sink the cache. The function returns the
    output path.
    """
    obs_path = Path(obs_path)
    catalog_path = Path(catalog_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    catalog = load_catalog(catalog_path)
    targets = [
        sd
        for sd in catalog
        if sd.forecastable and sd.frequency == "monthly"
    ]
    if only_series is not None:
        keep = set(only_series)
        targets = [sd for sd in targets if sd.series_id in keep]

    by_series: dict[str, dict] = {}
    errors: dict[str, str] = {}

    for target in targets:
        sid = target.series_id
        logger.info("Precomputing forecast for %s", sid)
        try:
            history = read_series(sid, obs_path).sort("period_start")
            non_null = history.filter(pl.col("value").is_not_null())

            min_obs_needed = cv_horizon * (n_windows + 1)
            if non_null.height < min_obs_needed:
                raise RuntimeError(
                    f"Only {non_null.height} non-null observations; need "
                    f"{min_obs_needed} for {n_windows} CV windows of horizon "
                    f"{cv_horizon}."
                )

            result = run_backtest(
                series_id=sid,
                horizon=cv_horizon,
                n_windows=n_windows,
                obs_path=obs_path,
                catalog_path=catalog_path,
            )

            metrics_sorted = result.metrics.sort("mape")
            winner = str(metrics_sorted["model"][0])
            winner_row = metrics_sorted.filter(pl.col("model") == winner).row(
                0, named=True
            )
            winner_metrics = {
                "mape": round(float(winner_row["mape"]), 4),
                "smape": round(float(winner_row["smape"]), 4),
                "mase": round(float(winner_row["mase"]), 4),
            }
            scoreboard = [
                {
                    "model": str(r["model"]),
                    "mape": round(float(r["mape"]), 4),
                    "smape": round(float(r["smape"]), 4),
                    "mase": round(float(r["mase"]), 4),
                }
                for r in metrics_sorted.iter_rows(named=True)
            ]

            forward, scales = _series_forward_forecast(
                series_id=sid,
                winner=winner,
                cv_details=result.cv_details,
                cv_horizon=cv_horizon,
                forward_horizon=forward_horizon,
                obs_path=obs_path,
                catalog_path=catalog_path,
            )

            latest, prior_month, prior_year = _series_priors(history)

            by_series[sid] = {
                "series_name": target.series_name,
                "commodity": target.commodity,
                "metric": target.metric,
                "unit": target.unit,
                "frequency": target.frequency,
                "notes": target.notes,
                "winner_model": winner,
                "winner_metrics": winner_metrics,
                "scoreboard": scoreboard,
                "horizon": cv_horizon,
                "n_windows": n_windows,
                "latest_actual": latest,
                "prior_month_actual": prior_month,
                "prior_year_actual": prior_year,
                "sparkline": _series_sparkline(history, months=sparkline_months),
                "forward": _forward_to_records(forward),
                "conformal_scale_h1": round(scales[0], 4) if scales else None,
                "conformal_scale_h_last": round(scales[-1], 4) if scales else None,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Precompute failed for %s: %s", sid, exc)
            errors[sid] = f"{type(exc).__name__}: {exc}\n" + traceback.format_exc(limit=2)

    cache = {
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "horizon": cv_horizon,
        "n_windows": n_windows,
        "forward_horizon": forward_horizon,
        "by_series": by_series,
        "by_series_errors": errors,
    }

    out_path.write_text(json.dumps(cache, indent=2, default=str), encoding="utf-8")
    logger.info(
        "Forecast cache written to %s (%d series, %d errors)",
        out_path,
        len(by_series),
        len(errors),
    )
    return out_path


def load_forecast_cache(path: Path | str = DEFAULT_CACHE_PATH) -> dict:
    """Read the forecast cache JSON. Returns an empty stub if it doesn't exist."""
    path = Path(path)
    if not path.exists():
        return {
            "generated_at": None,
            "by_series": {},
            "by_series_errors": {},
        }
    return json.loads(path.read_text(encoding="utf-8"))


def _cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Precompute the LivestockBrief forecast cache."
    )
    parser.add_argument("--obs-path", default="data/clean/observations.parquet")
    parser.add_argument("--catalog-path", default="data/catalog.json")
    parser.add_argument("--out-path", default=str(DEFAULT_CACHE_PATH))
    parser.add_argument("--horizon", type=int, default=DEFAULT_CV_HORIZON)
    parser.add_argument("--n-windows", type=int, default=DEFAULT_N_WINDOWS)
    parser.add_argument(
        "--forward-horizon", type=int, default=DEFAULT_FORWARD_HORIZON
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Restrict to specific series ids (for debugging).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    out = build_forecast_cache(
        obs_path=args.obs_path,
        catalog_path=args.catalog_path,
        out_path=args.out_path,
        cv_horizon=args.horizon,
        n_windows=args.n_windows,
        forward_horizon=args.forward_horizon,
        only_series=args.only,
    )
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())


__all__ = [
    "DEFAULT_CACHE_PATH",
    "DEFAULT_CV_HORIZON",
    "DEFAULT_FORWARD_HORIZON",
    "DEFAULT_N_WINDOWS",
    "build_forecast_cache",
    "load_forecast_cache",
]
