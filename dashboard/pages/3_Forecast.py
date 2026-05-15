"""Forecast — simplified default view + Advanced backtest expander.

Default state (95% of visitors):
    * Series picker (sidebar).
    * Plain-English brief.
    * 12-month forward forecast chart from the precomputed cache.
    * Scoreboard, in a small collapsed expander.

Advanced expander (analysts):
    * Live backtest config (horizon, CV windows, model selection).
    * "Run backtest" with progress streaming.
    * Scoreboard / CV overlay / residual diagnostics / live forward forecast.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import streamlit as st
from components.brief import compose_brief
from components.cache import cache_horizon, get_series_entry
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
from components.theme import INK_SOFT, inject_global_css, set_page_chrome

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
from usda_sandbox.store import read_observations, read_series

set_page_chrome(page_title="Forecast")
inject_global_css()
series_id = render_sidebar(frequencies=["monthly"], forecastable_only=True)

if series_id is None:
    st.warning(
        "No monthly series available — click **Refresh data** in the sidebar "
        "(the Forecast page is monthly-only; quarterly series live on Series)."
    )
    st.stop()

obs_path = Path(DEFAULT_OBS_PATH)

# Series metadata (always needed)
_meta_df = cached_list_series(str(obs_path)).filter(pl.col("series_id") == series_id)
if _meta_df.is_empty():
    st.error(f"Series {series_id!r} not found in catalog.")
    st.stop()
_meta = _meta_df.row(0, named=True)
series_name = _meta["series_name"]

st.markdown(f"# Forecast: {series_name}")
notes = cached_series_notes().get(series_id, "")
if notes:
    st.markdown(
        f"<p style='color:{INK_SOFT};font-size:0.95rem;margin-top:-0.5rem;'>"
        f"{notes}</p>",
        unsafe_allow_html=True,
    )

# ---- Default view: read from precomputed cache --------------------------

cache_entry = get_series_entry(series_id)
horizon_info = cache_horizon()

if cache_entry is None:
    st.info(
        "**No precomputed forecast** for this series yet. Open the "
        "**Advanced** expander below to run a live backtest."
    )
else:
    brief_html = compose_brief(cache_entry)
    if brief_html:
        st.markdown(
            f"<div class='lb-brief-headline'>{brief_html}</div>",
            unsafe_allow_html=True,
        )

    forward_records = cache_entry.get("forward") or []
    if forward_records:
        forward = pl.DataFrame(
            {
                "period_start": [date.fromisoformat(r["period_start"]) for r in forward_records],
                "point": [r["point"] for r in forward_records],
                "lower_80": [r["lower_80"] for r in forward_records],
                "upper_80": [r["upper_80"] for r in forward_records],
            }
        )
        history = (
            read_series(series_id, obs_path)
            .filter(pl.col("value").is_not_null())
            .sort("period_start")
        )
        winner = str(cache_entry.get("winner_model", "model"))
        st.plotly_chart(
            build_forward_forecast(
                history, forward, model_name=winner, label=series_name
            ),
            use_container_width=True,
        )
        st.caption(
            f"Forecast: **{winner}**, selected from a {cache_entry.get('n_windows', '?')}-"
            f"window backtest. 80% prediction interval is conformally calibrated "
            f"per horizon step from the CV residuals."
        )

    with st.expander("Model comparison (scoreboard)"):
        scoreboard = pl.DataFrame(cache_entry.get("scoreboard") or [])
        if scoreboard.is_empty():
            st.info("No scoreboard available in the cache.")
        else:
            st.dataframe(
                scoreboard.with_columns(
                    pl.col("mape").round(2),
                    pl.col("smape").round(2),
                    pl.col("mase").round(2),
                ),
                hide_index=True,
                use_container_width=True,
            )
            st.caption(
                "MAPE = average % miss. sMAPE = symmetric MAPE. MASE = error "
                "relative to a naive 'last value' baseline (>1 means worse than "
                "naive; <1 means better)."
            )

# ---- Advanced expander: the original live-backtest UI -------------------

with st.expander("Advanced — run a live backtest", expanded=False):
    st.caption(
        "For analysts. Re-runs the full bake-off live (3 models × your chosen "
        "CV windows) and shows the scoreboard, CV overlay, residual "
        "diagnostics, and a freshly fit 12-month forward forecast. Slower than "
        "the cached default above."
    )

    _series_full = (
        read_series(series_id, obs_path)
        .filter(pl.col("value").is_not_null())
        .select(["period_start", "value"])
    )
    n_obs = _series_full.height

    _catalog_full = _load_catalog("data/catalog.json")
    _target_def_full = next((sd for sd in _catalog_full if sd.series_id == series_id), None)
    _regressor_ids = list(_target_def_full.exogenous_regressors) if _target_def_full else []
    effective_n_obs = n_obs
    overlap_n_obs = n_obs
    if _regressor_ids:
        _exog_long = (
            read_observations(obs_path)
            .filter(pl.col("series_id").is_in(_regressor_ids))
            .select(["series_id", "period_start", "value"])
            .collect()
        )
        _exog_wide = _exog_long.pivot(values="value", index="period_start", on="series_id")
        _reg_cols = [c for c in _exog_wide.columns if c != "period_start"]
        _exog_full = _exog_wide.filter(
            pl.all_horizontal([pl.col(c).is_not_null() for c in _reg_cols])
        )
        _cash_dates = set(_series_full["period_start"].to_list())
        _exog_dates = set(_exog_full["period_start"].to_list())
        overlap_n_obs = len(_cash_dates & _exog_dates)
        effective_n_obs = min(n_obs, overlap_n_obs)

    max_horizon = max(1, min(12, effective_n_obs // 3 - 1))
    default_horizon = min(6, max_horizon)

    cfg_a, cfg_b, cfg_c, cfg_d = st.columns([1, 1, 2, 1])
    horizon = cfg_a.slider(
        "Horizon (months)", min_value=1, max_value=max_horizon, value=default_horizon
    )
    max_n_windows = max(2, min(24, effective_n_obs // horizon - 1))
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
    run_clicked = cfg_d.button(
        "> Run backtest", type="primary", use_container_width=True
    )

    if _regressor_ids and overlap_n_obs < n_obs:
        st.caption(
            f"Series has **{n_obs}** non-null cash observations, but only "
            f"**{overlap_n_obs}** align with all {len(_regressor_ids)} futures "
            f"regressors (yfinance's per-contract history is short). Sliders "
            f"are bounded by the smaller number so CV always has enough rows."
        )

    cache_key = (
        f"backtest:{series_id}:{horizon}:{n_windows}:{','.join(sorted(selected_models))}"
    )

    if run_clicked:
        if not selected_models:
            st.error("Pick at least one model.")
            st.stop()

        series = (
            read_series(series_id, obs_path)
            .filter(pl.col("value").is_not_null())
            .select(["period_start", "value"])
        )
        _required_rows = horizon * (n_windows + 1)
        if effective_n_obs < _required_rows:
            _suffix = (
                f" (after aligning with {len(_regressor_ids)} futures regressors)"
                if _regressor_ids and overlap_n_obs < n_obs
                else ""
            )
            st.error(
                f"Series has {effective_n_obs} effective observations{_suffix}; "
                f"need at least {_required_rows} for {n_windows} windows of "
                f"horizon {horizon}."
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
                            f"{event.model} window {event.window + 1}/"
                            f"{event.n_windows} done — running MAPE: "
                            f"{mape_str} — elapsed: {event.elapsed_s:.1f}s"
                        )
                    elif isinstance(event, BacktestResult):
                        st.session_state[cache_key] = event
                status.update(label="Backtest complete", state="complete")
            except Exception as exc:
                status.update(label=f"Backtest failed: {exc}", state="error")
                st.exception(exc)
                st.stop()

    result: BacktestResult | None = st.session_state.get(cache_key)
    if result is None:
        st.info(
            "Configure the inputs above and click **Run backtest** to see the "
            "live scoreboard, CV overlay, residual diagnostics, and a "
            "freshly-fit 12-month forward forecast."
        )
    else:
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

        live_winner = filtered_metrics.sort("mape")["model"][0]
        live_winner_metrics = filtered_metrics.filter(
            pl.col("model") == live_winner
        ).row(0, named=True)
        history = (
            read_series(series_id, obs_path)
            .filter(pl.col("value").is_not_null())
            .sort("period_start")
        )
        last_actual = float(history["value"].tail(1).item())
        typical_err = live_winner_metrics["mape"] / 100.0 * last_actual

        st.success(
            f"**Live winner:** {live_winner} — MAPE "
            f"{live_winner_metrics['mape']:.2f}% &middot; sMAPE "
            f"{live_winner_metrics['smape']:.2f}% &middot; MASE "
            f"{live_winner_metrics['mase']:.2f}"
        )
        st.markdown(
            f"At the recent level of **${last_actual:,.2f} {_meta['unit']}**, a "
            f"**{live_winner_metrics['mape']:.1f}% MAPE** means the typical "
            f"{result.horizon}-month forecast is off by about "
            f"**±${typical_err:,.2f} {_meta['unit']}**."
        )

        st.subheader("Actuals vs. CV forecasts")
        st.plotly_chart(
            build_cv_overlay(
                history,
                filtered_cv,
                label=series_name,
                horizon=result.horizon,
                n_windows=result.n_windows,
            ),
            use_container_width=True,
        )

        st.subheader(f"Residual diagnostics — {live_winner}")
        st.plotly_chart(
            build_residual_diagnostics(
                filtered_cv, model_name=live_winner, label=series_name
            ),
            use_container_width=True,
        )

        st.subheader("Live forward 12-month forecast")
        forecaster_registry = {
            "AutoARIMA": StatsForecastAutoARIMA,
            "Prophet": ProphetForecaster,
            "LightGBM": LightGBMForecaster,
        }
        with st.spinner(f"Refitting {live_winner} on full history..."):
            fcst = forecaster_registry[live_winner](seed=42)
            if _regressor_ids:
                _exog_history_long = (
                    read_observations(obs_path)
                    .filter(pl.col("series_id").is_in(_regressor_ids))
                    .select(["series_id", "period_start", "value"])
                    .collect()
                )
                _exog_history_wide = (
                    _exog_history_long.pivot(
                        values="value", index="period_start", on="series_id"
                    )
                    .sort("period_start")
                )
                _aligned = history.select(["period_start", "value"]).join(
                    _exog_history_wide, on="period_start", how="inner"
                )
                _history_for_fit = _aligned.select(["period_start", "value"])
                _exog_for_fit = _aligned.select(["period_start", *_regressor_ids])
                _last_exog_row = _exog_history_wide.tail(1)
                _last_history_date = _history_for_fit["period_start"].max()
                _future_dates: list[date] = []
                _yr, _mo = _last_history_date.year, _last_history_date.month
                for _ in range(12):
                    _mo += 1
                    if _mo == 13:
                        _mo = 1
                        _yr += 1
                    _future_dates.append(date(_yr, _mo, 1))
                _exog_future = pl.DataFrame(
                    {
                        "period_start": _future_dates,
                        **{
                            col: [_last_exog_row[col][0]] * 12
                            for col in _regressor_ids
                        },
                    }
                ).with_columns(pl.col("period_start").cast(pl.Date))
                fcst.fit(_history_for_fit, exog=_exog_for_fit)
                forward = fcst.predict(horizon=12, exog_future=_exog_future)
            else:
                fcst.fit(history.select(["period_start", "value"]))
                forward = fcst.predict(horizon=12)

        _per_h_scales = conformal_scale_factors_per_horizon(
            result.cv_details, model_name=live_winner, horizon=result.horizon
        )
        _forward_scales = _per_h_scales + [_per_h_scales[-1]] * max(
            0, forward.height - len(_per_h_scales)
        )
        _forward_scales = _forward_scales[: forward.height]
        forward = apply_conformal_scaling(forward, scale=_forward_scales)

        st.plotly_chart(
            build_forward_forecast(
                history, forward, model_name=live_winner, label=series_name
            ),
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

# ---- Footer note ---------------------------------------------------------

st.markdown("---")
gen_str = ""
if horizon_info is not None:
    gen_str = (
        f"Default forecast uses {horizon_info[0]}-month horizon, "
        f"{horizon_info[1]} CV windows, and {horizon_info[2]}-month forward "
        "forecast. "
    )
st.markdown(
    f"<p style='color:{INK_SOFT};font-size:0.85rem;'>"
    f"{gen_str}See <a href='/Methodology' target='_self'>Methodology</a> for "
    "how the models, prediction intervals, and conformal calibration work. "
    "This page is informational and not financial advice."
    "</p>",
    unsafe_allow_html=True,
)
