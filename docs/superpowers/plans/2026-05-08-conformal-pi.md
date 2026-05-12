# Conformal-Calibrated Prediction Intervals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the dashboard's bogus 80% PI with a conformally-calibrated band that achieves actual 80% empirical coverage on each series' CV residuals.

**Architecture:** New sibling module `src/usda_sandbox/calibration.py` with two pure functions (`conformal_scale_factor`, `apply_conformal_scaling`). The dashboard's Forecast page wires them into the forward forecast as three new lines plus a caption. `forecast.py` itself doesn't change, so all 69 existing tests stay green.

**Tech Stack:** polars (DataFrame operations), pytest (testing). No new dependencies.

**Spec:** [`docs/superpowers/specs/2026-05-08-conformal-pi-design.md`](../specs/2026-05-08-conformal-pi-design.md)

---

## File map

| Path | Action | Responsibility |
|---|---|---|
| `src/usda_sandbox/calibration.py` | create | `apply_conformal_scaling`, `conformal_scale_factor` — pure post-processing functions |
| `tests/test_calibration.py` | create | ~14 tests on synthetic data |
| `dashboard/pages/3_Forecast.py` | modify | Three new lines after the forward-forecast refit + one new caption explaining the calibration scale factor |

---

## Task 1: `apply_conformal_scaling` — the simpler function, TDD'd first

This is the band-width scaling function: takes a forecast DataFrame and a scalar multiplier, returns a new forecast with the half-widths around `point` scaled by that factor. Independent of any conformal math — just arithmetic on three columns.

**Files:**
- Create: `src/usda_sandbox/calibration.py`
- Create: `tests/test_calibration.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_calibration.py`:

```python
"""Tests for the conformal calibration module.

All tests use small synthetic DataFrames so they run in milliseconds.
The 69 existing tests stay green; this file adds the new coverage.
"""

from __future__ import annotations

import math
from datetime import date

import polars as pl
import pytest

from usda_sandbox.calibration import apply_conformal_scaling


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_calibration.py -v`
Expected: 5 failed with `ModuleNotFoundError: No module named 'usda_sandbox.calibration'`.

- [ ] **Step 3: Implement `apply_conformal_scaling`**

Create `src/usda_sandbox/calibration.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_calibration.py -v`
Expected: 5 passed.

- [ ] **Step 5: Run full suite + lint + types to confirm no regression**

Run: `uv run pytest && uv run ruff check . && uv run mypy`
Expected: 74 tests passed (69 existing + 5 new), ruff clean, mypy clean.

- [ ] **Step 6: Commit**

```bash
git add src/usda_sandbox/calibration.py tests/test_calibration.py
git commit -m "feat(calibration): add apply_conformal_scaling"
```

---

## Task 2: `conformal_scale_factor` — the calibration math

This is the function that computes the multiplicative scale factor from CV residuals. Per the spec, it uses the locally-weighted split-CP recipe: stretch ratio per row, target-th quantile.

**Files:**
- Modify: `src/usda_sandbox/calibration.py`
- Modify: `tests/test_calibration.py`

- [ ] **Step 1: Add failing tests for `conformal_scale_factor`**

Append to `tests/test_calibration.py`, just below the existing imports and helper (you'll need to extend the import block):

Replace the top of the file (the existing import block) with:

```python
"""Tests for the conformal calibration module.

All tests use small synthetic DataFrames so they run in milliseconds.
The 69 existing tests stay green; this file adds the new coverage.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import polars as pl
import pytest

from usda_sandbox.calibration import (
    apply_conformal_scaling,
    conformal_scale_factor,
)
```

Then append at the end of the file:

```python
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
        if actual_above_point:
            actual = point + s * half_width
        else:
            actual = point - s * half_width
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_calibration.py -v -k "scale_factor"`
Expected: 11 failed with `ImportError: cannot import name 'conformal_scale_factor'`.

- [ ] **Step 3: Implement `conformal_scale_factor`**

Edit `src/usda_sandbox/calibration.py` — update the `__all__` and add the new function.

Replace the existing `__all__` line and `_HALF_WIDTH_EPS` block (which currently reads `__all__ = ["apply_conformal_scaling"]`) with:

```python
__all__ = ["apply_conformal_scaling", "conformal_scale_factor"]

# Avoid divide-by-zero on degenerate flat PIs.
_HALF_WIDTH_EPS = 1e-9


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

    return float(annotated["stretch"].quantile(target_coverage, interpolation="linear"))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_calibration.py -v`
Expected: 16 passed (5 from Task 1 + 11 new in Task 2).

- [ ] **Step 5: Run full suite + lint + types**

Run: `uv run pytest && uv run ruff check . && uv run mypy`
Expected: 85 tests pass (69 existing + 5 from Task 1 + 11 from Task 2), ruff clean, mypy clean.

- [ ] **Step 6: Commit**

```bash
git add src/usda_sandbox/calibration.py tests/test_calibration.py
git commit -m "feat(calibration): add conformal_scale_factor"
```

---

## Task 3: Dashboard integration

Wire both functions into the Forecast page's forward-forecast step. Three new lines + one new caption.

**Files:**
- Modify: `dashboard/pages/3_Forecast.py`

- [ ] **Step 1: Add the imports**

Open `dashboard/pages/3_Forecast.py`. The existing imports near the top include:

```python
from usda_sandbox.forecast import (
    BacktestProgress,
    BacktestResult,
    LightGBMForecaster,
    ProphetForecaster,
    StatsForecastAutoARIMA,
    iter_run_backtest,
)
from usda_sandbox.store import read_series
```

Add a new import line below those, in alphabetical block order:

```python
from usda_sandbox.calibration import (
    apply_conformal_scaling,
    conformal_scale_factor,
)
from usda_sandbox.forecast import (
    BacktestProgress,
    BacktestResult,
    LightGBMForecaster,
    ProphetForecaster,
    StatsForecastAutoARIMA,
    iter_run_backtest,
)
from usda_sandbox.store import read_series
```

- [ ] **Step 2: Apply calibration to the forward forecast**

Find this block near the bottom of the file (the "Forward 12-month forecast" section):

```python
with st.spinner(f"Refitting {winner} on full history..."):
    fcst = forecaster_registry[winner](seed=42)
    fcst.fit(history.select(["period_start", "value"]))
    forward = fcst.predict(horizon=12)

st.plotly_chart(
    build_forward_forecast(history, forward, model_name=winner, label=series_label),
    use_container_width=True,
)
```

Insert the calibration step between the spinner block and the `st.plotly_chart` call:

```python
with st.spinner(f"Refitting {winner} on full history..."):
    fcst = forecaster_registry[winner](seed=42)
    fcst.fit(history.select(["period_start", "value"]))
    forward = fcst.predict(horizon=12)

# Calibrate the forward forecast's PI against CV residuals so the
# "80% PI" actually means 80% empirical coverage on this series.
_scale = conformal_scale_factor(result.cv_details, model_name=winner)
forward = apply_conformal_scaling(forward, scale=_scale)

st.plotly_chart(
    build_forward_forecast(history, forward, model_name=winner, label=series_label),
    use_container_width=True,
)
```

- [ ] **Step 3: Add a caption explaining the calibration scale factor**

Immediately after the `st.plotly_chart(build_forward_forecast(...))` call you just modified (and before the `with st.expander("Numeric forecast table"):` block), insert:

```python
st.caption(
    f"Prediction interval has been **conformally calibrated** against the "
    f"{result.n_windows} CV windows above. The model's native band was scaled "
    f"by **{_scale:.2f}x** to land at 80% empirical coverage on the calibration "
    f"set. A factor above 1.0 means the raw model was overconfident; below 1.0 "
    f"means it was overconservative."
)
```

- [ ] **Step 4: Smoke-test that the page still imports cleanly**

Streamlit pages can't be imported as regular modules (leading-digit filename), so test via `spec_from_file_location`:

Run:
```bash
uv run python -c "
import importlib.util
spec = importlib.util.spec_from_file_location('forecast_page', 'dashboard/pages/3_Forecast.py')
assert spec and spec.loader
print('module spec loads ok')
"
```

Expected: prints `module spec loads ok`.

- [ ] **Step 5: Run full suite + lint to confirm no regression**

Run: `uv run pytest && uv run ruff check .`
Expected: 85 tests pass, ruff clean.

- [ ] **Step 6: End-to-end smoke test by launching Streamlit headless**

Run:
```bash
uv run streamlit run dashboard/app.py --server.headless true --server.port 8513 > /tmp/sl.log 2>&1 &
SLPID=$!
sleep 10
curl -s -o /dev/null -w "HTTP: %{http_code}\n" http://localhost:8513
kill $SLPID 2>/dev/null
sleep 2
```

Expected: prints `HTTP: 200`. If it doesn't, check `/tmp/sl.log` for import errors and fix before committing.

- [ ] **Step 7: Commit**

```bash
git add dashboard/pages/3_Forecast.py
git commit -m "feat(dashboard): apply conformal calibration to forward forecast PI"
```

- [ ] **Step 8: Push to remote**

```bash
git push origin main
```

Expected: branch updated on `origin/main`.

---

## Self-review notes

**Spec coverage:**

- New module `src/usda_sandbox/calibration.py` → Task 1 creates it, Task 2 extends it ✓
- `apply_conformal_scaling` signature matches spec → Task 1 ✓
- `conformal_scale_factor` signature matches spec (named-only kwargs, default target_coverage=0.80) → Task 2 ✓
- Algorithm: stretch ratio per row, target-th quantile → Task 2 implementation matches §2 of spec ✓
- 1e-9 clamp on half-width → Task 2 (`_HALF_WIDTH_EPS`) ✓
- Symmetric scaling preserving asymmetry → Task 1 implementation expression ✓
- Dashboard integration: three new lines + caption → Task 3 ✓
- Tests for both functions on synthetic data → Tasks 1-2 (5+11 = 16 tests) ✓
- Edge cases (empty input, unknown model, null actuals, degenerate flat PI, invalid target) → Task 2 tests ✓
- No changes to `forecast.py` → all three tasks leave it alone ✓
- No new dependencies → calibration.py uses only polars (already a dep) ✓

**Type / signature consistency:**

- `apply_conformal_scaling(forecast, scale) -> pl.DataFrame` matches in Task 1 implementation, Task 1 tests, Task 3 dashboard usage ✓
- `conformal_scale_factor(cv_details, *, model_name, target_coverage=0.80) -> float` matches across Task 2 implementation, Task 2 tests, Task 3 dashboard usage ✓
- `_HALF_WIDTH_EPS = 1e-9` defined once in Task 1 (in calibration.py), referenced by Task 2's implementation ✓

**Placeholder scan:** None. Every step has either explicit code, an exact command with expected output, or both. The Task 3 import-block instruction shows the *exact* surrounding context (existing imports + new import) so the engineer doesn't have to guess where to insert.
