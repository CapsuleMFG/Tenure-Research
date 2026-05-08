"""USDA Livestock Sandbox — Streamlit entrypoint.

Launch with::

    uv run streamlit run dashboard/app.py

Streamlit auto-discovers pages from ``dashboard/pages/``.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from dashboard.components.sidebar import DEFAULT_OBS_PATH, render_sidebar

st.set_page_config(
    page_title="USDA Livestock Sandbox",
    page_icon="🐂",
    layout="wide",
    initial_sidebar_state="expanded",
)


obs_path = Path(DEFAULT_OBS_PATH)
render_sidebar(obs_path=obs_path)

st.title("USDA Livestock Sandbox")
st.markdown(
    """
A laptop-local view layer over the cleaned USDA ERS livestock and meat
data. Use the sidebar to pick a series, then visit one of the pages:

* **Explore** — what's in the cleaned store: catalog, coverage, data quality.
* **Visualize** — single-series time-series + seasonal decomposition + YoY change.
* **Forecast** — run a backtest (3 models x 12 windows) live and see the scoreboard,
  CV overlay, residuals, and a 12-month forward forecast with 80% prediction
  intervals.

The dashboard is read-only against `data/clean/observations.parquet`.
Click **Refresh data** in the sidebar to re-run ingest and cleaning from
the latest ERS release.
"""
)

if not obs_path.exists():
    st.info(
        "**No data yet.** Click *Refresh data* in the sidebar to "
        "download the ERS files and run the cleaner."
    )
else:
    st.success(f"Cleaned store loaded from `{obs_path}`. Pick a page from the sidebar.")
