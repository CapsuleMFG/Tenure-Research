"""Catalog & coverage — what's in the cleaned store."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import streamlit as st
from components.sidebar import (
    DEFAULT_OBS_PATH,
    cached_dataset_overview,
    cached_list_series,
    render_sidebar,
)
from components.theme import INK_SOFT, inject_global_css

from usda_sandbox.store import duckdb_connection

inject_global_css()
render_sidebar(persistent_picker=False)

st.markdown("# Catalog & coverage")
st.markdown(
    f"<p style='color:{INK_SOFT};font-size:1.0rem;'>"
    "Every series in the cleaned store, with row counts, date ranges, and "
    "data-quality gaps. Use this page to see exactly what's behind the "
    "forecasts on the Brief and Forecast pages."
    "</p>",
    unsafe_allow_html=True,
)

obs_path = Path(DEFAULT_OBS_PATH)
if not obs_path.exists():
    st.warning("No data yet — click **Refresh data** in the sidebar.")
    st.stop()

overview = cached_dataset_overview(str(obs_path))
series_df = cached_list_series(str(obs_path))

col_a, col_b, col_c = st.columns(3)
col_a.metric("Total series", overview["n_series"])
col_b.metric("Total rows", f"{overview['n_rows']:,}")
col_c.metric(
    "Date range",
    f"{overview['earliest']} -> {overview['latest']}",
)

st.subheader("Catalog")
commodities = sorted(series_df["commodity"].unique().to_list())
selected = st.multiselect(
    "Filter by commodity",
    options=commodities,
    default=commodities,
)
filtered = series_df.filter(pl.col("commodity").is_in(selected)).sort("series_id")
st.dataframe(
    filtered.select(
        [
            "series_id",
            "series_name",
            "commodity",
            "metric",
            "unit",
            "frequency",
            "n_obs",
            "n_nulls",
            "first_period",
            "last_period",
        ]
    ),
    hide_index=True,
    use_container_width=True,
)

st.subheader("By commodity")
con = duckdb_connection(obs_path)
try:
    by_commodity = con.execute(
        """
        SELECT commodity,
               COUNT(DISTINCT series_id)                          AS n_series,
               COUNT(*)                                            AS n_rows,
               SUM(CASE WHEN value IS NULL THEN 1 ELSE 0 END)      AS n_nulls,
               MIN(period_start)                                   AS earliest,
               MAX(period_start)                                   AS latest
        FROM obs
        GROUP BY commodity
        ORDER BY commodity
        """
    ).pl()
finally:
    con.close()
st.dataframe(by_commodity, hide_index=True, use_container_width=True)

st.subheader("Null spans (where source data was missing)")
null_summary = (
    series_df.filter(pl.col("n_nulls") > 0)
    .select(
        ["series_id", "series_name", "n_obs", "n_nulls", "first_period", "last_period"]
    )
    .sort("n_nulls", descending=True)
)
if null_summary.is_empty():
    st.success("Every series has full coverage — no nulls.")
else:
    st.dataframe(null_summary, hide_index=True, use_container_width=True)
