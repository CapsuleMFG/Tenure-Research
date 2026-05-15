"""LivestockBrief — Streamlit Cloud entrypoint.

Uses ``st.navigation`` (Streamlit 1.36+) to explicitly register all pages
so they work whether the app runs from the repo root (Streamlit Cloud
deployment) or from ``dashboard/`` (legacy local invocation). The path
strings are relative to *this* file.

Per Streamlit's multipage contract, ``st.set_page_config`` is called once
here in the entrypoint; individual pages do not call it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make ``dashboard/components`` and ``src/usda_sandbox`` importable from pages.
ROOT = Path(__file__).resolve().parent
for sub in (ROOT / "dashboard", ROOT / "src"):
    p = str(sub)
    if p not in sys.path:
        sys.path.insert(0, p)

st.set_page_config(
    page_title="LivestockBrief",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Each page is one file. Titles drive sidebar labels.
brief = st.Page(
    "dashboard/app.py",
    title="Brief",
    icon=":material/dashboard:",
    default=True,
)
catalog = st.Page(
    "dashboard/pages/1_Explore.py",
    title="Catalog",
    icon=":material/list_alt:",
)
series = st.Page(
    "dashboard/pages/2_Series.py",
    title="Series",
    icon=":material/show_chart:",
)
forecast = st.Page(
    "dashboard/pages/3_Forecast.py",
    title="Forecast",
    icon=":material/insights:",
)
methodology = st.Page(
    "dashboard/pages/4_Methodology.py",
    title="Methodology",
    icon=":material/menu_book:",
)
about = st.Page(
    "dashboard/pages/5_About.py",
    title="About",
    icon=":material/info:",
)

navigation = st.navigation(
    {
        "": [brief],
        "Explore": [catalog, series, forecast],
        "Reference": [methodology, about],
    }
)
navigation.run()
