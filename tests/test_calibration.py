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
    conformal_scale_factors_per_horizon,
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


# --------------------------------------------------------------------------- #
# Helpers + tests for per-horizon scale factors
# --------------------------------------------------------------------------- #


def _synth_cv_per_horizon(
    *,
    model: str,
    per_step_stretches: list[list[float]],
    actual_above_point: bool = True,
) -> pl.DataFrame:
    """Build a cv_details DataFrame with explicit (window, step) control.

    ``per_step_stretches[step_idx]`` is a list of stretch ratios across
    windows for that step. All inner lists must have the same length
    (= number of windows). Within each window, ``period_start`` is monotonically
    increasing so the per-horizon helper can rank rows by step.
    """
    n_steps = len(per_step_stretches)
    n_windows = len(per_step_stretches[0])
    for s in per_step_stretches:
        assert len(s) == n_windows, "all per-step lists must have the same length"

    rows = []
    point = 100.0
    half_width = 10.0
    for w in range(n_windows):
        for step in range(n_steps):
            stretch = per_step_stretches[step][w]
            actual = (
                point + stretch * half_width
                if actual_above_point
                else point - stretch * half_width
            )
            # Distinct period_starts per (window, step) so within-window ordering is well-defined.
            base_month = 1 + step
            year = 2024 + w
            rows.append(
                {
                    "window": w,
                    "period_start": date(year, base_month, 1),
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


def test_per_horizon_returns_one_scale_per_step() -> None:
    cv = _synth_cv_per_horizon(
        model="M1",
        per_step_stretches=[
            [0.5, 0.5, 0.5, 0.5],  # step 1
            [1.0, 1.0, 1.0, 1.0],  # step 2
            [2.0, 2.0, 2.0, 2.0],  # step 3
        ],
    )
    scales = conformal_scale_factors_per_horizon(
        cv, model_name="M1", horizon=3
    )
    assert len(scales) == 3
    assert scales[0] == pytest.approx(0.5)
    assert scales[1] == pytest.approx(1.0)
    assert scales[2] == pytest.approx(2.0)


def test_per_horizon_scales_grow_with_step_when_far_horizons_miss_more() -> None:
    """Realistic shape: near-term tight, far-term wide. Scales should grow."""
    cv = _synth_cv_per_horizon(
        model="M1",
        per_step_stretches=[
            [0.3, 0.4, 0.5, 0.6],
            [0.8, 0.9, 1.0, 1.1],
            [1.5, 1.6, 1.7, 1.8],
        ],
    )
    scales = conformal_scale_factors_per_horizon(
        cv, model_name="M1", horizon=3
    )
    assert scales[0] < scales[1] < scales[2]


def test_per_horizon_horizon_1_matches_scalar() -> None:
    """horizon=1 should produce a single scale that matches the scalar version."""
    cv = _synth_cv_per_horizon(
        model="M1",
        per_step_stretches=[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]],
    )
    scales = conformal_scale_factors_per_horizon(
        cv, model_name="M1", horizon=1
    )
    scalar = conformal_scale_factor(cv, model_name="M1")
    assert len(scales) == 1
    assert scales[0] == pytest.approx(scalar)


def test_per_horizon_target_coverage_respected() -> None:
    cv = _synth_cv_per_horizon(
        model="M1",
        per_step_stretches=[
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        ],
    )
    s50 = conformal_scale_factors_per_horizon(
        cv, model_name="M1", horizon=2, target_coverage=0.50
    )
    s95 = conformal_scale_factors_per_horizon(
        cv, model_name="M1", horizon=2, target_coverage=0.95
    )
    for a, b in zip(s50, s95, strict=True):
        assert b > a


def test_per_horizon_filters_by_model_name() -> None:
    cv_a = _synth_cv_per_horizon(
        model="ModelA",
        per_step_stretches=[[3.0] * 4, [3.0] * 4],
    )
    cv_b = _synth_cv_per_horizon(
        model="ModelB",
        per_step_stretches=[[0.1] * 4, [0.1] * 4],
    )
    combined = pl.concat([cv_a, cv_b])
    a = conformal_scale_factors_per_horizon(combined, model_name="ModelA", horizon=2)
    b = conformal_scale_factors_per_horizon(combined, model_name="ModelB", horizon=2)
    assert a == pytest.approx([3.0, 3.0])
    assert b == pytest.approx([0.1, 0.1])


def test_per_horizon_unknown_model_raises() -> None:
    cv = _synth_cv_per_horizon(model="M1", per_step_stretches=[[1.0] * 4])
    with pytest.raises(ValueError, match="no rows for model"):
        conformal_scale_factors_per_horizon(
            cv, model_name="DoesNotExist", horizon=1
        )


def test_per_horizon_invalid_target_coverage_raises() -> None:
    cv = _synth_cv_per_horizon(model="M1", per_step_stretches=[[1.0] * 4])
    with pytest.raises(ValueError, match="target_coverage"):
        conformal_scale_factors_per_horizon(
            cv, model_name="M1", horizon=1, target_coverage=0.0
        )


def test_per_horizon_invalid_horizon_raises() -> None:
    cv = _synth_cv_per_horizon(model="M1", per_step_stretches=[[1.0] * 4])
    with pytest.raises(ValueError, match="horizon"):
        conformal_scale_factors_per_horizon(
            cv, model_name="M1", horizon=0
        )


def test_per_horizon_horizon_mismatch_raises() -> None:
    """Asking for horizon=5 when CV only has 3 steps/window should raise."""
    cv = _synth_cv_per_horizon(
        model="M1",
        per_step_stretches=[[1.0] * 4, [1.0] * 4, [1.0] * 4],
    )
    with pytest.raises(ValueError, match="horizon"):
        conformal_scale_factors_per_horizon(
            cv, model_name="M1", horizon=5
        )


def test_per_horizon_drops_null_actuals() -> None:
    """A null actual at step 2 in one window doesn't poison other steps or windows."""
    cv = _synth_cv_per_horizon(
        model="M1",
        per_step_stretches=[
            [0.5, 0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0, 1.0],
        ],
    )
    # Null out one (window=2, step=1) actual
    cv = cv.with_columns(
        pl.when(
            (pl.col("window") == 2) & (pl.col("period_start") == date(2026, 2, 1))
        )
        .then(None)
        .otherwise(pl.col("actual"))
        .alias("actual")
    )
    scales = conformal_scale_factors_per_horizon(
        cv, model_name="M1", horizon=2
    )
    assert scales[0] == pytest.approx(0.5)
    assert scales[1] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# apply_conformal_scaling — Sequence[float] path
# --------------------------------------------------------------------------- #


def test_apply_scaling_accepts_sequence_one_scale_per_row() -> None:
    fc = _sample_forecast()
    out = apply_conformal_scaling(fc, scale=[1.0, 2.0, 0.5])
    # Row 0: scale=1 → unchanged
    assert out["lower_80"][0] == pytest.approx(90.0)
    assert out["upper_80"][0] == pytest.approx(115.0)
    # Row 1: point=110, lower=95 (15 below), upper=130 (20 above), scale=2 →
    # lower=110-30=80, upper=110+40=150
    assert out["lower_80"][1] == pytest.approx(80.0)
    assert out["upper_80"][1] == pytest.approx(150.0)
    # Row 2: point=120, lower=100 (20 below), upper=145 (25 above), scale=0.5 →
    # lower=120-10=110, upper=120+12.5=132.5
    assert out["lower_80"][2] == pytest.approx(110.0)
    assert out["upper_80"][2] == pytest.approx(132.5)


def test_apply_scaling_sequence_length_mismatch_raises() -> None:
    fc = _sample_forecast()  # 3 rows
    with pytest.raises(ValueError, match="length"):
        apply_conformal_scaling(fc, scale=[1.0, 2.0])  # 2 != 3


def test_apply_scaling_sequence_preserves_point_column() -> None:
    fc = _sample_forecast()
    out = apply_conformal_scaling(fc, scale=[1.5, 2.5, 0.75])
    assert out["point"].to_list() == fc["point"].to_list()


def test_apply_scaling_sequence_preserves_schema_and_order() -> None:
    fc = _sample_forecast()
    out = apply_conformal_scaling(fc, scale=[1.0, 1.0, 1.0])
    assert out.columns == fc.columns
    assert dict(out.schema) == dict(fc.schema)


def test_apply_scaling_sequence_empty_forecast_accepts_empty_scales() -> None:
    fc = pl.DataFrame(
        schema={
            "period_start": pl.Date,
            "point": pl.Float64,
            "lower_80": pl.Float64,
            "upper_80": pl.Float64,
        }
    )
    out = apply_conformal_scaling(fc, scale=[])
    assert out.height == 0


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
