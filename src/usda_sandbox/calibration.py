"""Conformal calibration for prediction intervals.

Post-processing functions that take an already-produced forecast
DataFrame (from a fitted forecaster's :meth:`predict`) and the cross-
validation residuals (from :func:`run_backtest` / :func:`iter_run_backtest`)
and produce a calibrated PI that achieves a target empirical coverage rate.

The design and rationale live in
``docs/superpowers/specs/2026-05-08-conformal-pi-design.md``.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

__all__ = [
    "apply_conformal_scaling",
    "conformal_scale_factor",
    "conformal_scale_factors_per_horizon",
]

# Avoid divide-by-zero on degenerate flat PIs.
_HALF_WIDTH_EPS = 1e-9


def apply_conformal_scaling(
    forecast: pl.DataFrame,
    scale: float | Sequence[float],
) -> pl.DataFrame:
    """Scale a forecast's PI half-widths by a multiplicative factor.

    ``forecast`` must have columns ``period_start``, ``point``, ``lower_80``,
    ``upper_80``. The output preserves the schema and column order. The
    ``point`` column is untouched; ``lower_80`` and ``upper_80`` are scaled
    independently around ``point`` so the original asymmetry is preserved.

    ``scale`` may be either a scalar (applied to every row) or a sequence of
    floats whose length matches ``forecast.height`` (one scale per row, used
    for per-horizon calibration where step-1 and step-12 get different
    factors).

    Examples
    --------
    >>> fc.row(0, named=True)
    {'period_start': ..., 'point': 100.0, 'lower_80': 90.0, 'upper_80': 115.0}
    >>> apply_conformal_scaling(fc, scale=2.0).row(0, named=True)
    {'period_start': ..., 'point': 100.0, 'lower_80': 80.0, 'upper_80': 130.0}
    """
    if isinstance(scale, (int, float)):
        scale_expr: pl.Expr = pl.lit(float(scale))
    else:
        scales = list(scale)
        if len(scales) != forecast.height:
            raise ValueError(
                f"scale sequence length ({len(scales)}) must match forecast "
                f"height ({forecast.height})"
            )
        scale_expr = pl.Series("__scale", scales, dtype=pl.Float64)

    return forecast.with_columns(
        lower_80=pl.col("point") - scale_expr * (pl.col("point") - pl.col("lower_80")),
        upper_80=pl.col("point") + scale_expr * (pl.col("upper_80") - pl.col("point")),
    ).select(forecast.columns)


def conformal_scale_factor(
    cv_details: pl.DataFrame,
    *,
    model_name: str,
    target_coverage: float = 0.80,
) -> float:
    """Compute the scaling factor that calibrates a model's CV PI to a target.

    For each calibration row, the "stretch ratio" is the multiplicative
    factor the band would have needed to expand (on the relevant side) to
    just contain the actual:

    - If ``actual <= point``: ``(point - actual) / (point - lower_80)``
    - If ``actual >  point``: ``(actual - point) / (upper_80 - point)``

    The returned scale is the ``target_coverage``-th quantile of those
    ratios across all calibration rows for ``model_name``. Multiplying the
    forward forecast's half-widths by this scale yields a PI that achieves
    ``target_coverage`` empirical coverage on the calibration set.

    Parameters
    ----------
    cv_details
        Output of ``BacktestResult.cv_details`` — must have columns
        ``model``, ``point``, ``lower_80``, ``upper_80``, ``actual``.
    model_name
        Which model's residuals to use. Each model has its own
        characteristic miscoverage and gets its own scale.
    target_coverage
        Desired empirical coverage rate. Must be strictly in (0, 1).

    Returns
    -------
    float
        Scale factor. > 1 inflates the band; < 1 deflates it; 1.0 only
        if perfectly calibrated.

    Raises
    ------
    ValueError
        If ``target_coverage`` is outside (0, 1), or if no usable rows
        exist for the requested model.
    """
    if not 0.0 < target_coverage < 1.0:
        raise ValueError(
            f"target_coverage must be in (0, 1); got {target_coverage}"
        )

    rows = cv_details.filter(
        (pl.col("model") == model_name) & pl.col("actual").is_not_null()
    )
    if rows.is_empty():
        raise ValueError(
            f"no rows for model {model_name!r} in cv_details "
            f"(or all rows have null actual)"
        )

    # Stretch ratio per row, asymmetric — pick the relevant side of the band.
    annotated = rows.with_columns(
        stretch=pl.when(pl.col("actual") <= pl.col("point"))
        .then(
            (pl.col("point") - pl.col("actual"))
            / pl.max_horizontal(pl.col("point") - pl.col("lower_80"), _HALF_WIDTH_EPS)
        )
        .otherwise(
            (pl.col("actual") - pl.col("point"))
            / pl.max_horizontal(pl.col("upper_80") - pl.col("point"), _HALF_WIDTH_EPS)
        )
    )

    result = annotated["stretch"].quantile(target_coverage, interpolation="linear")
    if result is None:
        raise ValueError(
            f"no rows for model {model_name!r} in cv_details "
            f"(or all rows have null actual)"
        )
    return float(result)


def conformal_scale_factors_per_horizon(
    cv_details: pl.DataFrame,
    *,
    model_name: str,
    horizon: int,
    target_coverage: float = 0.80,
) -> list[float]:
    """Compute one calibration scale per horizon step.

    Unlike :func:`conformal_scale_factor`, which collapses all CV rows into a
    single scalar, this returns a list of length ``horizon`` — ``scales[0]``
    calibrates the first step of the forward forecast, ``scales[-1]`` the
    last. The step assignment is derived from within-window ordering of
    ``period_start`` (step 1 = earliest, step ``horizon`` = latest).

    This produces tighter intervals than the scalar approach when a model's
    miscoverage shape is not flat across horizons (e.g., AutoARIMA is usually
    well-calibrated at h=1 but overconfident at h=12).

    Raises
    ------
    ValueError
        If ``target_coverage`` is outside (0, 1), ``horizon < 1``, the
        requested model has no rows, or any window has fewer than ``horizon``
        rows.
    """
    if not 0.0 < target_coverage < 1.0:
        raise ValueError(
            f"target_coverage must be in (0, 1); got {target_coverage}"
        )
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1; got {horizon}")

    rows = cv_details.filter(pl.col("model") == model_name)
    if rows.is_empty():
        raise ValueError(
            f"no rows for model {model_name!r} in cv_details"
        )

    # Rank rows within each window by period_start to derive step (1-indexed).
    ranked = rows.sort(["window", "period_start"]).with_columns(
        step=pl.col("period_start").rank("ordinal").over("window").cast(pl.Int64),
    )
    max_step = int(ranked["step"].max() or 0)
    if max_step < horizon:
        raise ValueError(
            f"requested horizon={horizon} but cv_details has only {max_step} "
            f"steps per window for model {model_name!r}"
        )

    ranked = ranked.filter(pl.col("actual").is_not_null())

    scales: list[float] = []
    for step in range(1, horizon + 1):
        step_rows = ranked.filter(pl.col("step") == step)
        if step_rows.is_empty():
            raise ValueError(
                f"no usable rows for model {model_name!r} at step {step} "
                f"(all actuals null?)"
            )
        annotated = step_rows.with_columns(
            stretch=pl.when(pl.col("actual") <= pl.col("point"))
            .then(
                (pl.col("point") - pl.col("actual"))
                / pl.max_horizontal(
                    pl.col("point") - pl.col("lower_80"), _HALF_WIDTH_EPS
                )
            )
            .otherwise(
                (pl.col("actual") - pl.col("point"))
                / pl.max_horizontal(
                    pl.col("upper_80") - pl.col("point"), _HALF_WIDTH_EPS
                )
            )
        )
        q = annotated["stretch"].quantile(target_coverage, interpolation="linear")
        if q is None:
            raise ValueError(
                f"no usable rows for model {model_name!r} at step {step}"
            )
        scales.append(float(q))
    return scales
