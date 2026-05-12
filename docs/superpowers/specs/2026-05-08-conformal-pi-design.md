# Conformal-Calibrated Prediction Intervals (v0.2a)

**Status:** Draft, pending user review
**Date:** 2026-05-08
**Author:** Tyler + Claude (brainstormed)

## Goal

Replace the dashboard's bogus model-native 80% PI with a conformally
calibrated band that achieves actual 80% empirical coverage on the
series' recent history. Pure post-processing of existing CV
residuals — no new data, no model retraining, no API breakage.

The dashboard already documents that the current PI "lies during shock
periods" (the residual Q-Q plot shows this). This makes the band stop
lying — at least under conditions resembling the calibration set.

After this lands, the user can read the shaded band on the Forecast
page's forward forecast as *"prices land inside this band ~80% of the
time on data that looks like recent history"* — and that claim will be
defensible.

## Architecture

New sibling module to `forecast.py`:

```
src/usda_sandbox/
├── ingest.py
├── catalog.py
├── clean.py
├── store.py
├── forecast.py
└── calibration.py        # NEW — this spec
```

Two public functions:

```python
def conformal_scale_factor(
    cv_details: pl.DataFrame,
    *,
    model_name: str,
    target_coverage: float = 0.80,
) -> float: ...

def apply_conformal_scaling(
    forecast: pl.DataFrame,
    scale: float,
) -> pl.DataFrame: ...
```

`forecast.py` itself does not change. The dashboard's Forecast page
imports the two functions and applies them as three new lines after
the existing forward-forecast refit step.

`calibration.py` lives separately from `forecast.py` because:

- `forecast.py` owns "fit and predict" (already ~450 lines, full).
- Calibration is post-hoc — operates on already-produced forecasts and
  separate calibration data. Different responsibility, different
  testable unit.
- Future calibration variants (conformal quantile regression, ACI,
  EnbPI) drop into the same module without bloating `forecast.py`.

## Algorithm

This is a "locally weighted split-CP" recipe (Lei et al. 2018, §5.2).
The model's native half-widths act as local-volatility estimates — the
model already knows month 6 is more uncertain than month 1 — and we use
those local estimates as weights so calibration preserves the
horizon-growing shape of the band and only fixes the magnitude.

### Step 1: Nonconformity score per calibration row

For each row `i` in `cv_details` belonging to the chosen model, the
**stretch ratio** is the multiplicative factor the band would have
needed to expand on the relevant side in order to just contain the
actual:

```python
if actual_i <= point_i:
    stretch_i = (point_i - actual_i) / max(point_i - lower_80_i, 1e-9)
else:
    stretch_i = (actual_i - point_i) / max(upper_80_i - point_i, 1e-9)
```

`stretch_i < 1` means the actual was inside the band on that side.
`stretch_i > 1` means the band missed by a factor of `stretch_i`.

The `1e-9` clamp guards against degenerate flat PIs (model
overconfident enough to report zero half-width). Without the clamp,
such rows would produce `inf`. With it, they produce very large
finite values that correctly pull the resulting scale factor up.

### Step 2: Take the target-th quantile

```python
scale = quantile(stretch_array, target_coverage)
```

For `target_coverage = 0.80`, this is the 80th-percentile stretch
ratio. By construction, scaling the model's bands by `scale` makes the
scaled bands cover at least 80% of calibration rows on the relevant
side.

### Step 3: Apply to the forward forecast

```python
new_lower = point - scale * (point - lower_80)
new_upper = point + scale * (upper_80 - point)
```

Asymmetry preserved (lower-side and upper-side deviations scale
independently from their original distances to the point). Shape
preserved (horizon-growing widths stay horizon-growing; only the
overall magnitude is recalibrated).

### Expected effect on this project's series

The dashboard's residual Q-Q diagnostic already shows that the model's
native PI undercovers — empirical coverage of CV residuals is typically
in the 60-70% range when the nominal target is 80%. We expect `scale`
to land in the **1.2-1.7×** range for cash and wholesale prices, with
larger inflation on series that include shock periods (COVID, 2014-15
cattle peak) in their CV windows.

For series that happen to have been calm during the calibration period,
`scale` may dip below 1.0 — the band correctly *tightens* in that
case. This is the right behavior (we're not just inflating; we're
calibrating).

## API surface

### `conformal_scale_factor`

```python
def conformal_scale_factor(
    cv_details: pl.DataFrame,
    *,
    model_name: str,
    target_coverage: float = 0.80,
) -> float:
    """Compute the scale factor that calibrates a model's CV PI to a target.

    Parameters
    ----------
    cv_details
        Output of ``BacktestResult.cv_details`` — must have columns
        ``model``, ``point``, ``lower_80``, ``upper_80``, ``actual``.
    model_name
        Which model's residuals to use for calibration (each model has
        its own characteristic miscoverage).
    target_coverage
        Desired empirical coverage rate. Must be in (0, 1). Default 0.80.

    Returns
    -------
    float
        Scalar multiplier. Values > 1 inflate the band; < 1 deflate it.
        Equals 1.0 only if the model's native PI is already perfectly
        calibrated.

    Raises
    ------
    ValueError
        If ``cv_details`` has no rows for ``model_name``, if
        ``target_coverage`` is outside (0, 1), or if all rows for the
        model have null actuals.
    """
```

### `apply_conformal_scaling`

```python
def apply_conformal_scaling(
    forecast: pl.DataFrame,
    scale: float,
) -> pl.DataFrame:
    """Scale a forecast DataFrame's PI by a multiplicative factor.

    Parameters
    ----------
    forecast
        DataFrame with columns ``period_start``, ``point``, ``lower_80``,
        ``upper_80``. Schema and column order are preserved in the output.
    scale
        Multiplier for the half-widths around ``point``. Asymmetry on
        either side of ``point`` is preserved (each side scales
        independently from its original distance).

    Returns
    -------
    pl.DataFrame
        Same shape and dtypes as input. ``point`` and ``period_start``
        are unchanged; ``lower_80`` and ``upper_80`` are updated.
    """
```

Neither function touches `forecast.py`, `clean.py`, `store.py`, or any
existing public API. Both are pure functions (no I/O, no global state).

## Dashboard integration

Three new lines in `dashboard/pages/3_Forecast.py`, immediately after
the existing winner-refit block:

```python
from usda_sandbox.calibration import (
    apply_conformal_scaling,
    conformal_scale_factor,
)

# Existing:
with st.spinner(f"Refitting {winner} on full history..."):
    fcst = forecaster_registry[winner](seed=42)
    fcst.fit(history.select(["period_start", "value"]))
    forward = fcst.predict(horizon=12)

# New:
_scale = conformal_scale_factor(result.cv_details, model_name=winner)
forward = apply_conformal_scaling(forward, scale=_scale)
```

A new caption appears under the forward-forecast chart explaining what
happened:

```python
st.caption(
    f"Prediction interval has been **conformally calibrated** against "
    f"the {result.n_windows} CV windows above. The model's native band "
    f"was scaled by **{_scale:.2f}x** to land at 80% empirical coverage "
    f"on the calibration set. A factor above 1.0 means the raw model "
    f"was overconfident; below 1.0 means it was overconservative."
)
```

The existing forward-forecast caption ("Gray line = history. Blue line
= ... Shaded band = 80% prediction interval... read it as 'under
business-as-usual conditions, prices land in this range about 80% of
the time.'") stays — but the claim is now defensible. The "business as
usual" hedge becomes more accurate: the calibrated band achieves
exactly 80% coverage *on data resembling the calibration set*, which
is what BAU means in practice.

## Tests

`tests/test_calibration.py` — new file. All tests use synthetic data so
they run in milliseconds. The 69 existing tests stay green; this adds
~11 new tests bringing the total to 80.

### `conformal_scale_factor`

```python
def test_scale_factor_inflates_overconfident_model():
    """Actuals systematically 2x further than the band's edge → scale > 1.5."""

def test_scale_factor_deflates_overcoverage():
    """Actuals always at the point (zero residual) → scale near 0."""

def test_scale_factor_hits_target_quantile():
    """Stretch ratios 0.1..1.0; expect scale ≈ 0.82 (80th percentile of 10)."""

def test_scale_factor_handles_asymmetric_band():
    """Actual always above point → only upper side stretches; finite scale."""

def test_scale_factor_target_coverage_respected():
    """Same data, higher target → larger scale."""

def test_scale_factor_unknown_model_raises():
    """ValueError when model_name not in cv_details."""

def test_scale_factor_empty_cv_raises():
    """ValueError on empty input."""

def test_scale_factor_drops_null_residuals():
    """Null actuals are filtered out; finite scale is still produced."""

def test_scale_factor_invalid_target_coverage_raises():
    """target_coverage outside (0, 1) raises ValueError."""

def test_scale_factor_handles_degenerate_flat_pi():
    """lower_80 == point or upper_80 == point doesn't blow up
    (guarded by 1e-9 clamp). Resulting scale is finite."""
```

### `apply_conformal_scaling`

```python
def test_apply_scaling_identity_at_one():
    """scale=1.0 returns input bitwise equal."""

def test_apply_scaling_doubles_half_widths():
    """Concrete numerical check: scale=2.0 on asymmetric band [90, 100, 115]
    produces [80, 100, 130]. Asymmetry preserved."""

def test_apply_scaling_preserves_schema():
    """Output schema, column order, and dtypes match input."""

def test_apply_scaling_zero_scale_collapses_to_point():
    """scale=0.0 produces lower_80 == upper_80 == point."""
```

## Edge cases

| Case | Behavior |
|---|---|
| Empty `cv_details` | `ValueError("no rows ...")` |
| `model_name` not present in `cv_details` | `ValueError("no rows for model ...")` |
| Null actual or null PI bound on a CV row | Row filtered before scoring |
| Zero half-width on a CV row (degenerate model PI) | Clamped to `1e-9`; row still contributes a large finite stretch ratio |
| `scale = 0` (calibration says all-inside) | Forward bands collapse to the point. Unusual; technically correct |
| `target_coverage` outside `(0, 1)` | `ValueError` |
| All actuals null for the chosen model | `ValueError("no usable rows ...")` |
| Forward forecast has zero rows | Returns empty DataFrame unchanged |

## Out of scope for v0.2a

The following are real improvements but live in later releases:

- **Per-step quantile calibration** — separate scale per horizon step.
  Better calibration with more data; underpowered at 12 CV windows.
  Worth revisiting once v0.2b adds futures (which usually require more
  CV history anyway).
- **Adaptive conformal inference (ACI)** — online recalibration that
  adjusts coverage as new actuals come in. Useful for streaming
  forecasts, overkill for a sandbox.
- **Conformal quantile regression** — fitting a quantile model
  directly rather than scaling existing intervals. More principled
  but more code and compute.
- **User-configurable target coverage** in the dashboard (80%/90%/95%
  toggle). Easy to add later; not needed for v1.
- **Per-anchor empirical coverage report** in the dashboard — show
  the user the achieved coverage on each series. Nice diagnostic but
  separate feature.

## What this does not fix

- **Regime changes still break the band.** Calibration assumes the
  forward forecast period resembles the calibration period statistically.
  If a 2020-scale shock arrives during the forecast window, the band
  will still under-cover. The Q-Q plot in the residual diagnostics
  remains the right tool to anticipate this.
- **The point forecast is unchanged.** Calibration adjusts uncertainty,
  not central tendency. If the model is biased on level, this won't fix
  it.
- **MASE > 1 stays.** Calibration doesn't help the model beat a naive
  one-step forecast; it just makes the uncertainty estimate honest.

## Open questions for the user

None — every decision has a default. If something feels wrong on
review, flag it inline and we'll iterate.
