"""Page 2 — single-series deep dive."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import streamlit as st
from components.plots import (
    EVENT_MARKERS,
    build_series_chart,
    build_yoy_chart,
)
from components.sidebar import (
    DEFAULT_OBS_PATH,
    cached_series_notes,
    render_sidebar,
)
from statsmodels.tsa.seasonal import seasonal_decompose

from usda_sandbox.store import read_series

st.set_page_config(
    page_title="Visualize — USDA Livestock", page_icon="🐂", layout="wide"
)
series_id = render_sidebar()

st.title("Visualize a series")

if series_id is None:
    st.warning("No data yet — click **Refresh data** in the sidebar.")
    st.stop()

obs_path = Path(DEFAULT_OBS_PATH)

@st.cache_data(ttl=300)
def _cached_series(series_id: str, obs_path_str: str) -> pl.DataFrame:
    df = read_series(series_id, Path(obs_path_str))
    return df.filter(pl.col("value").is_not_null()).sort("period_start")


series = _cached_series(series_id, str(obs_path))
if series.is_empty():
    st.error(f"No non-null observations for series_id={series_id!r}")
    st.stop()

meta_row = series.row(0, named=True)
st.subheader(meta_row["series_name"])
_notes = cached_series_notes().get(series_id, "")
if _notes:
    st.markdown(f"_{_notes}_")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Commodity", meta_row["commodity"])
m2.metric("Unit", meta_row["unit"])
m3.metric("Frequency", meta_row["frequency"])
m4.metric("Observations", series.height)

# Date-range slider
min_d = series["period_start"].min()
max_d = series["period_start"].max()
date_range = st.slider(
    "Date range",
    min_value=min_d,
    max_value=max_d,
    value=(min_d, max_d),
    format="YYYY-MM",
)
filtered = series.filter(
    (pl.col("period_start") >= date_range[0])
    & (pl.col("period_start") <= date_range[1])
)
visible_events = [
    ev for ev in EVENT_MARKERS
    if pd.Timestamp(ev.iso_date).date() >= date_range[0]
    and pd.Timestamp(ev.iso_date).date() <= date_range[1]
]

st.plotly_chart(
    build_series_chart(filtered, label=meta_row["series_name"], events=visible_events),
    use_container_width=True,
)

st.subheader("Year-over-year change (%)")
yoy = (
    series.sort("period_start")
    .with_columns(
        yoy_pct=((pl.col("value") / pl.col("value").shift(12) - 1) * 100).round(2)
    )
    .filter(pl.col("yoy_pct").is_not_null())
    .filter(
        (pl.col("period_start") >= date_range[0])
        & (pl.col("period_start") <= date_range[1])
    )
)
if yoy.is_empty():
    st.info("Not enough history in this range to compute YoY change.")
else:
    st.plotly_chart(
        build_yoy_chart(yoy, label=meta_row["series_name"], events=visible_events),
        use_container_width=True,
    )

st.subheader("Seasonal decomposition")
with st.expander("Decomposition options"):
    period = st.number_input(
        "Period (months)",
        min_value=2,
        max_value=24,
        value=12,
        step=1,
    )
    model_kind = st.radio(
        "Model",
        options=("multiplicative", "additive"),
        index=0,
        horizontal=True,
    )

if filtered.height < period * 2:
    st.info(
        f"Need at least {period * 2} observations in the selected range for "
        f"period={period} decomposition; have {filtered.height}."
    )
else:
    pdf = filtered.to_pandas()
    ts = pd.Series(
        pdf["value"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(pdf["period_start"]),
    ).asfreq("MS")
    if (ts <= 0).any() and model_kind == "multiplicative":
        st.warning(
            "Series contains non-positive values — falling back to additive."
        )
        model_kind = "additive"
    ts = ts.ffill()
    result = seasonal_decompose(ts, model=model_kind, period=int(period))
    fig = result.plot()
    fig.set_size_inches(11, 7)
    fig.suptitle(
        f"{meta_row['series_name']} — {model_kind} decomposition (period={period})",
        y=1.02,
    )
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
