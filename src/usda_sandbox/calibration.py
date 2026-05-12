"""Conformal calibration for prediction intervals.

Post-processing functions that take an already-produced forecast
DataFrame (from a fitted forecaster's :meth:`predict`) and the cross-
validation residuals (from :func:`run_backtest` / :func:`iter_run_backtest`)
and produce a calibrated PI that achieves a target empirical coverage rate.

The design and rationale live in
``docs/superpowers/specs/2026-05-08-conformal-pi-design.md``.
"""

from __future__ import annotations

import polars as pl

__all__ = ["apply_conformal_scaling"]

# Avoid divide-by-zero on degenerate flat PIs.
_HALF_WIDTH_EPS = 1e-9


def apply_conformal_scaling(
    forecast: pl.DataFrame,
    scale: float,
) -> pl.DataFrame:
    """Scale a forecast's PI half-widths by a multiplicative factor.

    ``forecast`` must have columns ``period_start``, ``point``, ``lower_80``,
    ``upper_80``. The output preserves the schema and column order. The
    ``point`` column is untouched; ``lower_80`` and ``upper_80`` are scaled
    independently around ``point`` so the original asymmetry is preserved.

    Examples
    --------
    >>> fc.row(0, named=True)
    {'period_start': ..., 'point': 100.0, 'lower_80': 90.0, 'upper_80': 115.0}
    >>> apply_conformal_scaling(fc, scale=2.0).row(0, named=True)
    {'period_start': ..., 'point': 100.0, 'lower_80': 80.0, 'upper_80': 130.0}
    """
    return forecast.with_columns(
        lower_80=pl.col("point") - scale * (pl.col("point") - pl.col("lower_80")),
        upper_80=pl.col("point") + scale * (pl.col("upper_80") - pl.col("point")),
    ).select(forecast.columns)
