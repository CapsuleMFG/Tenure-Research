"""Page 3 -- live backtest + forecast results."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import streamlit as st
from components.plots import (
    build_cv_overlay,
    build_forward_forecast,
    build_residual_diagnostics,
)
from components.sidebar import (
    DEFAULT_OBS_PATH,
    cached_list_series,
    cached_series_notes,
    render_sidebar,
)

from usda_sandbox.calibration import (
    apply_conformal_scaling,
    conformal_scale_factors_per_horizon,
)
from usda_sandbox.catalog import load_catalog as _load_catalog
from usda_sandbox.forecast import (
    BacktestProgress,
    BacktestResult,
    LightGBMForecaster,
    ProphetForecaster,
    StatsForecastAutoARIMA,
    iter_run_backtest,
)
from usda_sandbox.store import read_series

st.set_page_config(
    page_title="Forecast -- USDA Livestock", page_icon="🐂", layout="wide"
)
series_id = render_sidebar(frequencies=["monthly"], forecastable_only=True)

st.title("Forecast")

if series_id is None:
    st.warning(
        "No monthly series available — click **Refresh data** in the sidebar "
        "(or note that the Forecast page only supports monthly series; "
        "quarterly WASDE series live on Explore and Visualize)."
    )
    st.stop()

obs_path = Path(DEFAULT_OBS_PATH)

# Pull the chosen series' metadata for human-readable labels and the
# explanatory panel below.
_meta_df = cached_list_series(str(obs_path)).filter(pl.col("series_id") == series_id)
if _meta_df.is_empty():
    st.error(f"Series {series_id!r} not found in catalog.")
    st.stop()
_meta = _meta_df.row(0, named=True)
series_name = _meta["series_name"]

# Series snapshot — what is this thing, in plain terms.
_notes = cached_series_notes().get(series_id, "")
with st.container(border=True):
    a, b, c, d = st.columns([3, 1, 1, 1])
    a.markdown(f"**{series_name}**  \n`{series_id}`")
    b.markdown(f"**Commodity**  \n{_meta['commodity']}")
    c.markdown(f"**Unit**  \n{_meta['unit']}")
    d.markdown(f"**Frequency**  \n{_meta['frequency']}")
    if _notes:
        st.markdown(f"_{_notes}_")

# Inspect the chosen series so slider bounds match what's actually possible.
_series_full = (
    read_series(series_id, obs_path)
    .filter(pl.col("value").is_not_null())
    .select(["period_start", "value"])
)
n_obs = _series_full.height

# Adaptive slider bounds: the cross-validator needs horizon * (n_windows + 1) <= n_obs.
# Cap horizon so at least 2 CV windows are achievable; cap n_windows from the chosen
# horizon. If a series is too short for any meaningful backtest, surface that early.
max_horizon = max(1, min(12, n_obs // 3 - 1))
default_horizon = min(6, max_horizon)

# ---------------- Inputs -----------------------------------------------------

cfg_a, cfg_b, cfg_c, cfg_d = st.columns([1, 1, 2, 1])
horizon = cfg_a.slider(
    "Horizon (months)", min_value=1, max_value=max_horizon, value=default_horizon
)
max_n_windows = max(2, min(24, n_obs // horizon - 1))
default_n_windows = min(12, max_n_windows)
n_windows = cfg_b.slider(
    "CV windows",
    min_value=2,
    max_value=max_n_windows,
    value=default_n_windows,
)
all_models = ("AutoARIMA", "Prophet", "LightGBM")
selected_models = cfg_c.multiselect(
    "Models", options=list(all_models), default=list(all_models)
)
run_clicked = cfg_d.button("> Run backtest", type="primary", use_container_width=True)

st.caption(
    f"Series has **{n_obs}** non-null observations. Sliders are bounded so "
    f"any combination produces a valid backtest."
)

cache_key = f"backtest:{series_id}:{horizon}:{n_windows}:{','.join(sorted(selected_models))}"

# ---------------- Run --------------------------------------------------------

if run_clicked:
    if not selected_models:
        st.error("Pick at least one model.")
        st.stop()

    series = (
        read_series(series_id, obs_path)
        .filter(pl.col("value").is_not_null())
        .select(["period_start", "value"])
    )
    if series.height < horizon * (n_windows + 1):
        st.error(
            f"Series has {series.height} non-null observations; need at least "
            f"{horizon * (n_windows + 1)} for {n_windows} windows of horizon "
            f"{horizon}."
        )
        st.stop()

    with st.status(
        f"Running backtest for {series_name}...", expanded=True
    ) as status:
        progress_bar = st.progress(0.0)
        total_steps = len(selected_models) * n_windows
        completed = 0
        try:
            for event in iter_run_backtest(
                series_id,
                horizon=horizon,
                n_windows=n_windows,
                obs_path=obs_path,
                models=selected_models,
            ):
                if isinstance(event, BacktestProgress):
                    completed += 1
                    pct = completed / total_steps
                    progress_bar.progress(pct)
                    mape_str = (
                        f"{event.running_mape:.2f}%"
                        if event.running_mape is not None
                        else "n/a"
                    )
                    status.write(
                        f"{event.model} window {event.window + 1}/{event.n_windows} "
                        f"done - running MAPE: {mape_str} - "
                        f"elapsed: {event.elapsed_s:.1f}s"
                    )
                elif isinstance(event, BacktestResult):
                    st.session_state[cache_key] = event
            status.update(label="Backtest complete", state="complete")
        except Exception as exc:
            status.update(label=f"Backtest failed: {exc}", state="error")
            st.exception(exc)
            st.stop()

# ---------------- Results ----------------------------------------------------

result: BacktestResult | None = st.session_state.get(cache_key)
if result is None:
    st.info(
        "Configure the inputs above and click **Run backtest** to see "
        "scoreboard, CV overlay, residual diagnostics, and a 12-month "
        "forward forecast."
    )
    st.stop()

# Filter to selected models only
filtered_cv = result.cv_details.filter(pl.col("model").is_in(selected_models))
filtered_metrics = result.metrics.filter(pl.col("model").is_in(selected_models))

with st.expander("How to read these results", expanded=False):
    st.markdown(
        """
**What we did.** For each model, we ran the same exercise N times: hide
the most recent stretch of history, ask the model to predict it, then
score the prediction against what actually happened. Each "stretch" is
one CV window. The scoreboard averages across all windows so the
numbers describe how the model **usually** does, not how it did on one
lucky/unlucky stretch.

**The metrics, in plain terms:**
- **MAPE** — the average miss as a *percent* of the actual value.
  Lower is better. *5%* means the forecast is typically off by 5% of
  the price.
- **sMAPE** — like MAPE but symmetric, so it doesn't blow up when
  actuals are tiny.
- **MASE** — model error vs. a naive *"tomorrow looks like today"*
  baseline. **Below 1.0** means the model beats naive. **Above 1.0**
  means naive wins. On these series MASE typically lands around 2-7
  for 6-month forecasts — model errors are 2-7x the typical
  month-to-month price change. Useful for **level/range, not for
  tactical timing.**
        """.strip()
    )

st.subheader("Scoreboard")
st.dataframe(
    filtered_metrics.with_columns(
        pl.col("mape").round(2),
        pl.col("smape").round(2),
        pl.col("mase").round(2),
    ).sort("mape"),
    hide_index=True,
    use_container_width=True,
)

# Surface exogenous regressors used (if any) — catalog-driven, set in data/catalog.json
_target_def = next(
    (sd for sd in _load_catalog("data/catalog.json") if sd.series_id == series_id),
    None,
)
if _target_def is not None and _target_def.exogenous_regressors:
    _commodity_label = "Live Cattle" if _target_def.commodity == "cattle" else "Lean Hogs"
    st.caption(
        f"This series was forecast with **{len(_target_def.exogenous_regressors)} "
        f"exogenous regressors**: deferred {_commodity_label} futures (1-12 months "
        f"ahead). Each forecaster (AutoARIMA, Prophet, LightGBM) sees these alongside "
        f"the cash history."
    )

# Pick the winner from the filtered set
winner = filtered_metrics.sort("mape")["model"][0]
winner_metrics = filtered_metrics.filter(pl.col("model") == winner).row(0, named=True)

# History first so we can quote a recent price level in the contextual blurb
history = (
    read_series(series_id, obs_path)
    .filter(pl.col("value").is_not_null())
    .sort("period_start")
)
last_actual = float(history["value"].tail(1).item())
typical_err = winner_metrics["mape"] / 100.0 * last_actual

st.success(
    f"**Winner:** {winner} — MAPE {winner_metrics['mape']:.2f}%  ·  "
    f"sMAPE {winner_metrics['smape']:.2f}%  ·  MASE {winner_metrics['mase']:.2f}"
)
st.markdown(
    f"At the recent level of **${last_actual:,.2f} {_meta['unit']}**, a "
    f"**{winner_metrics['mape']:.1f}% MAPE** means the typical "
    f"**{result.horizon}-month forecast is off by about "
    f"±${typical_err:,.2f} {_meta['unit']}**. Some windows do better, some "
    f"worse — that's the average across all {result.n_windows} CV windows. "
    f"MASE of {winner_metrics['mase']:.2f} says the model's error is roughly "
    f"**{winner_metrics['mase']:.1f}x the typical month-to-month price change**, "
    f"which is normal for a 6-month horizon."
)

st.subheader("Actuals vs. CV forecasts")
st.caption(
    "Gray line = the actual price. Each colored segment = one model's "
    "forecast for one CV window (so 12 segments per model). Bunching "
    "tightly around the gray line means consistent skill; segments that "
    "wander above or below the actual reveal where the model systematically "
    "missed."
)
series_label = series_name  # human-readable for chart titles
st.plotly_chart(
    build_cv_overlay(
        history,
        filtered_cv,
        label=series_label,
        horizon=result.horizon,
        n_windows=result.n_windows,
    ),
    use_container_width=True,
)

st.subheader(f"Residual diagnostics — {winner}")
st.caption(
    "**Residual** = actual minus forecast. Three views of how the winner missed "
    "across all CV windows: (1) the **distribution** should be centered near "
    "zero — a shifted center means the model is systematically biased; "
    "(2) **MAE by horizon** shows how error grows the further out you forecast; "
    "(3) **Q-Q vs. normal** — if residuals are normally distributed, the dots "
    "track the dotted line. Tails curving away from the line mean the 80% "
    "prediction interval is too narrow during shock periods (2020 COVID, etc.)."
)
st.plotly_chart(
    build_residual_diagnostics(filtered_cv, model_name=winner, label=series_label),
    use_container_width=True,
)

st.subheader("Forward 12-month forecast")
st.caption(
    "Gray line = history. Blue line = the winning model's point forecast for "
    "the next 12 months. Shaded blue band = **80% prediction interval** — "
    "read it as *\"under business-as-usual conditions, prices land in this "
    "range about 80% of the time.\"* The band widens further out because "
    "uncertainty compounds. Don't trust the band during regime changes "
    "(slaughter shocks, sudden inventory swings) — the residual diagnostics "
    "above show where the band tends to under-cover."
)
forecaster_registry = {
    "AutoARIMA": StatsForecastAutoARIMA,
    "Prophet": ProphetForecaster,
    "LightGBM": LightGBMForecaster,
}
with st.spinner(f"Refitting {winner} on full history..."):
    fcst = forecaster_registry[winner](seed=42)
    fcst.fit(history.select(["period_start", "value"]))
    forward = fcst.predict(horizon=12)

# Calibrate the forward forecast's PI against CV residuals so the
# "80% PI" actually means 80% empirical coverage on this series. A
# per-horizon scale (one factor per step) tracks how miscoverage grows
# with the forecast horizon; if CV used a shorter horizon than the
# forward forecast, the last calibrated step's scale carries forward.
_per_h_scales = conformal_scale_factors_per_horizon(
    result.cv_details, model_name=winner, horizon=result.horizon
)
_forward_scales = _per_h_scales + [_per_h_scales[-1]] * max(
    0, forward.height - len(_per_h_scales)
)
_forward_scales = _forward_scales[: forward.height]
forward = apply_conformal_scaling(forward, scale=_forward_scales)

st.plotly_chart(
    build_forward_forecast(history, forward, model_name=winner, label=series_label),
    use_container_width=True,
)
st.caption(
    f"Prediction interval has been **conformally calibrated per horizon** "
    f"against the {result.n_windows} CV windows above. The model's native "
    f"band was scaled by **{_forward_scales[0]:.2f}x at h=1** and "
    f"**{_forward_scales[-1]:.2f}x at h={forward.height}** to land at 80% "
    f"empirical coverage on the calibration set. Factors above 1.0 mean the "
    f"raw model was overconfident at that horizon; below 1.0 means "
    f"overconservative."
)

with st.expander("Numeric forecast table"):
    st.dataframe(
        forward.with_columns(
            pl.col("point").round(2),
            pl.col("lower_80").round(2),
            pl.col("upper_80").round(2),
        ),
        hide_index=True,
        use_container_width=True,
    )
