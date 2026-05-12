"""Tests for the conformal calibration module.

All tests use small synthetic DataFrames so they run in milliseconds.
The 69 existing tests stay green; this file adds the new coverage.
"""

from __future__ import annotations

import math
from datetime import date

import polars as pl
import pytest

from usda_sandbox.calibration import (
    apply_conformal_scaling,
    conformal_scale_factor,
)


def _sample_forecast() -> pl.DataFrame:
    """Forecast DataFrame matching the schema produced by forecaster.predict()."""
    return pl.DataFrame(
        {
            "period_start": [date(2026, 4, 1), date(2026, 5, 1), date(2026, 6, 1)],
            "point": [100.0, 110.0, 120.0],
            "lower_80": [90.0, 95.0, 100.0],
            "upper_80": [115.0, 130.0, 145.0],
        }
    ).with_columns(
        pl.col("period_start").cast(pl.Date),
        pl.col("point").cast(pl.Float64),
        pl.col("lower_80").cast(pl.Float64),
        pl.col("upper_80").cast(pl.Float64),
    )


def test_apply_scaling_identity_at_one() -> None:
    fc = _sample_forecast()
    out = apply_conformal_scaling(fc, scale=1.0)
    assert out.equals(fc)


def test_apply_scaling_doubles_half_widths_preserving_asymmetry() -> None:
    fc = pl.DataFrame(
        {
            "period_start": [date(2026, 4, 1)],
            "point": [100.0],
            "lower_80": [90.0],   # 10 below point
            "upper_80": [115.0],  # 15 above point
        }
    ).with_columns(pl.col("period_start").cast(pl.Date))
    out = apply_conformal_scaling(fc, scale=2.0)
    # Lower side: 10 → 20 below point → 80
    assert out["lower_80"][0] == pytest.approx(80.0)
    # Upper side: 15 → 30 above point → 130
    assert out["upper_80"][0] == pytest.approx(130.0)
    # Point unchanged
    assert out["point"][0] == 100.0
    # Date unchanged
    assert out["period_start"][0] == date(2026, 4, 1)


def test_apply_scaling_zero_scale_collapses_to_point() -> None:
    fc = _sample_forecast()
    out = apply_conformal_scaling(fc, scale=0.0)
    for row in out.iter_rows(named=True):
        assert row["lower_80"] == pytest.approx(row["point"])
        assert row["upper_80"] == pytest.approx(row["point"])


def test_apply_scaling_preserves_schema_and_column_order() -> None:
    fc = _sample_forecast()
    out = apply_conformal_scaling(fc, scale=1.5)
    assert out.columns == fc.columns
    assert dict(out.schema) == dict(fc.schema)
    assert out.height == fc.height


def test_apply_scaling_empty_forecast_returns_empty() -> None:
    fc = pl.DataFrame(
        schema={
            "period_start": pl.Date,
            "point": pl.Float64,
            "lower_80": pl.Float64,
            "upper_80": pl.Float64,
        }
    )
    out = apply_conformal_scaling(fc, scale=1.5)
    assert out.height == 0
    assert dict(out.schema) == dict(fc.schema)


# --------------------------------------------------------------------------- #
# Helpers for conformal_scale_factor tests
# --------------------------------------------------------------------------- #


def _synth_cv(
    *,
    model: str,
    stretches: list[float] | None = None,
    actual_above_point: bool = True,
    n_rows: int | None = None,
) -> pl.DataFrame:
    """Build a synthetic cv_details DataFrame with a known stretch ratio per row.

    Each row's lower_80 is fixed at point - 10, upper_80 at point + 10
    (symmetric band of half-width 10). The actual is placed so its
    stretch ratio is exactly the requested value.

    A "stretch" of 0.5 places the actual halfway between point and the
    band edge; a stretch of 2.0 places it twice as far as the band edge.
    """
    if stretches is None:
        assert n_rows is not None, "either stretches or n_rows must be given"
        stretches = [0.5] * n_rows

    rows = []
    for idx, s in enumerate(stretches):
        point = 100.0
        half_width = 10.0
        actual = point + s * half_width if actual_above_point else point - s * half_width
        rows.append(
            {
                "window": idx // 6,
                "period_start": date(2025, 1 + (idx % 12), 1) if idx < 12 else date(2026, 1, 1),
                "point": point,
                "lower_80": point - half_width,
                "upper_80": point + half_width,
                "actual": actual,
                "model": model,
            }
        )
    return pl.DataFrame(rows).with_columns(
        pl.col("window").cast(pl.Int32),
        pl.col("period_start").cast(pl.Date),
        pl.col("point").cast(pl.Float64),
        pl.col("lower_80").cast(pl.Float64),
        pl.col("upper_80").cast(pl.Float64),
        pl.col("actual").cast(pl.Float64),
    )


# --------------------------------------------------------------------------- #
# conformal_scale_factor — behavior
# --------------------------------------------------------------------------- #


def test_scale_factor_inflates_overconfident_model() -> None:
    """Every actual is 2x the band's edge → scale should be >= 2."""
    cv = _synth_cv(model="M1", stretches=[2.0] * 12)
    assert conformal_scale_factor(cv, model_name="M1") == pytest.approx(2.0)


def test_scale_factor_deflates_when_actuals_near_point() -> None:
    """Actuals at exactly the point (zero residual) → scale near 0."""
    cv = _synth_cv(model="M1", stretches=[0.0] * 12)
    assert conformal_scale_factor(cv, model_name="M1") == pytest.approx(0.0)


def test_scale_factor_returns_target_quantile_of_stretches() -> None:
    """Stretch ratios 0.1, 0.2, ..., 1.0; expect scale ≈ 0.82 at q=0.80.

    polars uses linear interpolation for quantiles by default, so the
    80th percentile of arange(0.1, 1.1, 0.1) is 0.82.
    """
    stretches = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    cv = _synth_cv(model="M1", stretches=stretches)
    assert conformal_scale_factor(cv, model_name="M1") == pytest.approx(0.82, abs=0.05)


def test_scale_factor_target_coverage_respected() -> None:
    """Higher target_coverage → larger scale (we need to widen more)."""
    stretches = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    cv = _synth_cv(model="M1", stretches=stretches)
    s_50 = conformal_scale_factor(cv, model_name="M1", target_coverage=0.50)
    s_95 = conformal_scale_factor(cv, model_name="M1", target_coverage=0.95)
    assert s_95 > s_50


def test_scale_factor_works_when_actuals_below_point() -> None:
    """Lower-side stretches should be picked up symmetrically."""
    cv = _synth_cv(model="M1", stretches=[1.5] * 12, actual_above_point=False)
    assert conformal_scale_factor(cv, model_name="M1") == pytest.approx(1.5)


def test_scale_factor_filters_by_model_name() -> None:
    """Only the requested model's rows contribute."""
    cv_a = _synth_cv(model="ModelA", stretches=[3.0] * 12)
    cv_b = _synth_cv(model="ModelB", stretches=[0.1] * 12)
    combined = pl.concat([cv_a, cv_b])
    assert conformal_scale_factor(combined, model_name="ModelA") == pytest.approx(3.0)
    assert conformal_scale_factor(combined, model_name="ModelB") == pytest.approx(0.1)


def test_scale_factor_unknown_model_raises() -> None:
    cv = _synth_cv(model="M1", stretches=[1.0] * 12)
    with pytest.raises(ValueError, match="no rows for model"):
        conformal_scale_factor(cv, model_name="DoesNotExist")


def test_scale_factor_empty_cv_raises() -> None:
    empty = pl.DataFrame(
        schema={
            "window": pl.Int32,
            "period_start": pl.Date,
            "point": pl.Float64,
            "lower_80": pl.Float64,
            "upper_80": pl.Float64,
            "actual": pl.Float64,
            "model": pl.Utf8,
        }
    )
    with pytest.raises(ValueError, match="no rows for model"):
        conformal_scale_factor(empty, model_name="M1")


def test_scale_factor_invalid_target_coverage_raises() -> None:
    cv = _synth_cv(model="M1", stretches=[1.0] * 12)
    with pytest.raises(ValueError, match="target_coverage"):
        conformal_scale_factor(cv, model_name="M1", target_coverage=0.0)
    with pytest.raises(ValueError, match="target_coverage"):
        conformal_scale_factor(cv, model_name="M1", target_coverage=1.0)
    with pytest.raises(ValueError, match="target_coverage"):
        conformal_scale_factor(cv, model_name="M1", target_coverage=-0.5)


def test_scale_factor_drops_null_actuals() -> None:
    """Rows with null `actual` are filtered out; finite scale is still produced."""
    cv = _synth_cv(model="M1", stretches=[1.0] * 5)
    # Inject a null-actual row at the end
    null_row = pl.DataFrame(
        {
            "window": [99],
            "period_start": [date(2027, 1, 1)],
            "point": [100.0],
            "lower_80": [90.0],
            "upper_80": [110.0],
            "actual": [None],
            "model": ["M1"],
        }
    ).with_columns(
        pl.col("window").cast(pl.Int32),
        pl.col("period_start").cast(pl.Date),
        pl.col("point").cast(pl.Float64),
        pl.col("lower_80").cast(pl.Float64),
        pl.col("upper_80").cast(pl.Float64),
        pl.col("actual").cast(pl.Float64),
    )
    combined = pl.concat([cv, null_row])
    s = conformal_scale_factor(combined, model_name="M1")
    assert math.isfinite(s)
    assert s == pytest.approx(1.0)


def test_scale_factor_handles_degenerate_flat_pi() -> None:
    """A row where lower_80 == point doesn't blow up; clamped by 1e-9."""
    cv = pl.DataFrame(
        {
            "window": [0, 1, 2],
            "period_start": [date(2026, 1, 1), date(2026, 2, 1), date(2026, 3, 1)],
            "point": [100.0, 100.0, 100.0],
            "lower_80": [100.0, 90.0, 90.0],  # row 0 has zero lower half-width
            "upper_80": [100.0, 110.0, 110.0],  # row 0 has zero upper half-width
            "actual": [99.0, 100.0, 100.0],  # row 0 will produce a huge stretch ratio
            "model": ["M1", "M1", "M1"],
        }
    ).with_columns(
        pl.col("window").cast(pl.Int32),
        pl.col("period_start").cast(pl.Date),
        pl.col("point").cast(pl.Float64),
        pl.col("lower_80").cast(pl.Float64),
        pl.col("upper_80").cast(pl.Float64),
        pl.col("actual").cast(pl.Float64),
    )
    s = conformal_scale_factor(cv, model_name="M1")
    assert math.isfinite(s)
