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
from components.sidebar import DEFAULT_OBS_PATH, render_sidebar
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
series_id = render_sidebar()

st.title("Forecast")

if series_id is None:
    st.warning("No data yet -- click **Refresh data** in the sidebar.")
    st.stop()

obs_path = Path(DEFAULT_OBS_PATH)

# ---------------- Inputs -----------------------------------------------------

cfg_a, cfg_b, cfg_c, cfg_d = st.columns([1, 1, 2, 1])
horizon = cfg_a.slider("Horizon (months)", min_value=1, max_value=12, value=6)
n_windows = cfg_b.slider("CV windows", min_value=2, max_value=24, value=12)
all_models = ("AutoARIMA", "Prophet", "LightGBM")
selected_models = cfg_c.multiselect(
    "Models", options=list(all_models), default=list(all_models)
)
run_clicked = cfg_d.button("> Run backtest", type="primary", use_container_width=True)

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
        f"Running backtest for {series_id}...", expanded=True
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

# Pick the winner from the filtered set
winner = filtered_metrics.sort("mape")["model"][0]
st.success(
    f"**Winner:** {winner} -- MAPE "
    f"{filtered_metrics.filter(pl.col('model') == winner)['mape'][0]:.2f}%"
)

st.subheader("Actuals vs. CV forecasts")
history = (
    read_series(series_id, obs_path)
    .filter(pl.col("value").is_not_null())
    .sort("period_start")
)
series_label = series_id  # raw id is fine here; sidebar already shows pretty name
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

st.subheader(f"Residual diagnostics -- {winner}")
st.plotly_chart(
    build_residual_diagnostics(filtered_cv, model_name=winner, label=series_label),
    use_container_width=True,
)

st.subheader("Forward 12-month forecast")
forecaster_registry = {
    "AutoARIMA": StatsForecastAutoARIMA,
    "Prophet": ProphetForecaster,
    "LightGBM": LightGBMForecaster,
}
with st.spinner(f"Refitting {winner} on full history..."):
    fcst = forecaster_registry[winner](seed=42)
    fcst.fit(history.select(["period_start", "value"]))
    forward = fcst.predict(horizon=12)

st.plotly_chart(
    build_forward_forecast(history, forward, model_name=winner, label=series_label),
    use_container_width=True,
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
