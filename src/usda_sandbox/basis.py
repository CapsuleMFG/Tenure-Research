"""Cash-to-futures basis computation.

For a producer, the price they actually receive is the regional cash price,
not the futures price. The difference is the *basis*:

    basis = cash − nearby_futures

A positive basis means cash trades above the futures contract; a negative
basis means cash trades at a discount. The basis is what determines the
gap between a hedged price and the realized one.

This module exposes pure functions that read the tidy ``observations.parquet``
and return polars DataFrames or scalar summaries — no Streamlit, no I/O
beyond the parquet read.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl

from .store import DEFAULT_OBS_PATH, read_series

__all__ = [
    "BasisStats",
    "basis_stats",
    "compute_basis",
    "latest_basis",
]


# Catalog convention: a cash series's "natural" futures peer is encoded by
# its catalog ``exogenous_regressors[0]``. We mirror that here as a
# fallback map for callers that don't load the catalog. Anything not in
# the map either has no good futures peer (lamb) or should be looked up
# from the catalog by the caller.
_DEFAULT_FUTURES_PEER: dict[str, str] = {
    "cattle_steer_choice_tx_ok_nm":   "cattle_lc_front",
    "cattle_steer_choice_nebraska":   "cattle_lc_front",
    "cattle_feeder_steer_500_550":    "cattle_feeder_front",
    "cattle_feeder_steer_750_800":    "cattle_feeder_front",
    "hog_barrow_gilt_natbase_51_52":  "hogs_he_front",
    "boxed_beef_cutout_choice":       "cattle_lc_front",
    "boxed_beef_cutout_select":       "cattle_lc_front",
    "pork_cutout_composite":          "hogs_he_front",
}


def default_futures_peer(cash_series_id: str) -> str | None:
    """Return the conventional front-month futures series id for ``cash_series_id``."""
    return _DEFAULT_FUTURES_PEER.get(cash_series_id)


@dataclass(frozen=True)
class BasisStats:
    """Summary stats over a basis time series."""

    cash_series_id: str
    futures_series_id: str
    latest_basis: float | None
    latest_date: date | None
    mean_basis: float | None
    median_basis: float | None
    p10_basis: float | None
    p90_basis: float | None
    n_obs: int


def compute_basis(
    cash_series_id: str,
    futures_series_id: str,
    *,
    obs_path: Path | str = DEFAULT_OBS_PATH,
    prefer_daily_futures: bool = True,
) -> pl.DataFrame:
    """Return a tidy basis time series joined on ``period_start``.

    Columns: ``period_start``, ``cash``, ``futures``, ``basis``.

    If ``prefer_daily_futures`` is True and a ``<futures_series_id>_daily``
    series exists in the store, we'll use that instead — daily futures
    align to the cash dates via monthly resampling (last close in the
    month). Otherwise we use the monthly futures series as-is.

    Rows with null cash *or* futures are dropped.
    """
    obs_path = Path(obs_path)
    cash = (
        read_series(cash_series_id, obs_path)
        .filter(pl.col("value").is_not_null())
        .select(pl.col("period_start"), cash=pl.col("value"))
    )
    if cash.is_empty():
        return pl.DataFrame(
            schema={
                "period_start": pl.Date,
                "cash": pl.Float64,
                "futures": pl.Float64,
                "basis": pl.Float64,
            }
        )

    daily_id = f"{futures_series_id}_daily"
    futures: pl.DataFrame
    if prefer_daily_futures:
        try:
            daily = (
                read_series(daily_id, obs_path)
                .filter(pl.col("value").is_not_null())
                .sort("period_start")
            )
        except Exception:
            daily = pl.DataFrame()
        if not daily.is_empty():
            # Resample daily to month-start by taking the last close in
            # each month, then align that to cash period_starts.
            futures = (
                daily.with_columns(period_start=pl.col("period_start").dt.month_start())
                .group_by("period_start")
                .agg(pl.col("value").last().alias("futures"))
                .sort("period_start")
            )
        else:
            monthly = read_series(futures_series_id, obs_path).filter(
                pl.col("value").is_not_null()
            )
            futures = monthly.select(
                pl.col("period_start"), futures=pl.col("value")
            )
    else:
        monthly = read_series(futures_series_id, obs_path).filter(
            pl.col("value").is_not_null()
        )
        futures = monthly.select(
            pl.col("period_start"), futures=pl.col("value")
        )

    joined = cash.join(futures, on="period_start", how="inner").with_columns(
        basis=pl.col("cash") - pl.col("futures")
    )
    return joined.sort("period_start")


def latest_basis(
    cash_series_id: str,
    futures_series_id: str | None = None,
    *,
    obs_path: Path | str = DEFAULT_OBS_PATH,
) -> tuple[float, date] | None:
    """Return ``(basis, period_start)`` of the most recent overlapping observation.

    Returns ``None`` if either side is empty or they don't overlap.
    """
    peer = futures_series_id or default_futures_peer(cash_series_id)
    if peer is None:
        return None
    df = compute_basis(cash_series_id, peer, obs_path=obs_path)
    if df.is_empty():
        return None
    row = df.tail(1).row(0, named=True)
    return float(row["basis"]), row["period_start"]


def basis_stats(
    cash_series_id: str,
    futures_series_id: str | None = None,
    *,
    obs_path: Path | str = DEFAULT_OBS_PATH,
    lookback_months: int = 60,
) -> BasisStats:
    """Compute mean/median/percentiles of basis over the last ``lookback_months``."""
    peer = futures_series_id or default_futures_peer(cash_series_id)
    if peer is None:
        return BasisStats(
            cash_series_id=cash_series_id,
            futures_series_id="",
            latest_basis=None,
            latest_date=None,
            mean_basis=None,
            median_basis=None,
            p10_basis=None,
            p90_basis=None,
            n_obs=0,
        )

    df = compute_basis(cash_series_id, peer, obs_path=obs_path).tail(lookback_months)
    if df.is_empty():
        return BasisStats(
            cash_series_id=cash_series_id,
            futures_series_id=peer,
            latest_basis=None,
            latest_date=None,
            mean_basis=None,
            median_basis=None,
            p10_basis=None,
            p90_basis=None,
            n_obs=0,
        )

    last = df.tail(1).row(0, named=True)
    return BasisStats(
        cash_series_id=cash_series_id,
        futures_series_id=peer,
        latest_basis=float(last["basis"]),
        latest_date=last["period_start"],
        mean_basis=float(df["basis"].mean() or 0.0),
        median_basis=float(df["basis"].median() or 0.0),
        p10_basis=float(df["basis"].quantile(0.10) or 0.0),
        p90_basis=float(df["basis"].quantile(0.90) or 0.0),
        n_obs=df.height,
    )
