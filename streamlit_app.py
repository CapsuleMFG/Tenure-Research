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

FAVICON_PATH = ROOT / "dashboard" / "static" / "favicon.png"
page_icon: str | Path = FAVICON_PATH if FAVICON_PATH.exists() else "🌾"

st.set_page_config(
    page_title="LivestockBrief — daily livestock prices, forecasts, decisions",
    page_icon=page_icon,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject Open Graph + Twitter card meta tags so shared links get a real
# preview instead of generic Streamlit chrome. Runs once per page load via
# an iframe shim that mutates the parent document's <head>.
from components.theme import inject_og_tags  # noqa: E402

inject_og_tags(
    title="LivestockBrief",
    description=(
        "Daily livestock prices, USDA-grounded 6-month forecasts with "
        "honest uncertainty, and a transparent sell-now / hold decision "
        "tool for cattle and hog producers. Free, public."
    ),
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
# v3.0 direct-market tools (primary)
plan = st.Page(
    "dashboard/pages/6_Plan.py",
    title="Plan",
    icon=":material/route:",
)
costs = st.Page(
    "dashboard/pages/7_Costs.py",
    title="Costs",
    icon=":material/payments:",
)
pricing = st.Page(
    "dashboard/pages/8_Pricing.py",
    title="Pricing",
    icon=":material/sell:",
)
# v2.0 commodity tools (kept for users who want them)
decide = st.Page(
    "dashboard/pages/6_Decide.py",
    title="Decide (commodity)",
    icon=":material/balance:",
)
breakeven = st.Page(
    "dashboard/pages/7_Breakeven.py",
    title="Feedlot breakeven",
    icon=":material/calculate:",
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
        "Plan your operation": [plan, costs, pricing],
        "Explore data": [catalog, series, forecast],
        "Commodity tools (v2.0)": [decide, breakeven],
        "Reference": [methodology, about],
    }
)
navigation.run()
