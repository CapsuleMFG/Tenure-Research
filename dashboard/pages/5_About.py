"""About — credits, version, refresh status, GitHub link, disclaimer."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import streamlit as st
from components.cache import cache_generated_at, cache_horizon
from components.sidebar import DEFAULT_OBS_PATH, render_sidebar
from components.theme import BRAND_NAME, BRAND_TAGLINE, INK_SOFT, inject_global_css, set_page_chrome

from usda_sandbox import __version__

set_page_chrome(page_title="About")
inject_global_css()
render_sidebar(persistent_picker=False)

st.markdown(f"# About {BRAND_NAME}")
st.markdown(
    f"<p style='color:{INK_SOFT};font-size:1.0rem;'>{BRAND_TAGLINE}</p>",
    unsafe_allow_html=True,
)

st.markdown("## What this is")
st.markdown(
    """
LivestockBrief is a small public dashboard that takes USDA Economic Research
Service livestock and meat data, runs three classical forecasting models
against every series, and surfaces the winning model's forecast in plain
English alongside an honest, calibrated prediction interval.

It's free, it's read-only, and the source is on GitHub.
"""
)

st.markdown("## Status")

obs_path = Path(DEFAULT_OBS_PATH)
if obs_path.exists():
    obs_mtime = datetime.fromtimestamp(obs_path.stat().st_mtime, tz=UTC)
    obs_label = obs_mtime.strftime("%Y-%m-%d %H:%M UTC")
else:
    obs_label = "no data yet"

gen_at = cache_generated_at() or "not yet generated"
hi = cache_horizon()
horizon_label = (
    f"{hi[0]}-month horizon, {hi[1]} CV windows, {hi[2]}-month forward"
    if hi is not None
    else "n/a"
)

c1, c2 = st.columns(2)
c1.metric("Code version", __version__)
c2.metric("Cleaned data refreshed", obs_label)
st.markdown(
    f"**Latest forecast snapshot:** `{gen_at}`  \n"
    f"**Forecast configuration:** {horizon_label}"
)

st.markdown("## Source")
st.markdown(
    """
- Project home: **[github.com/tylermark/usda-livestock-sandbox](https://github.com/tylermark/usda-livestock-sandbox)**
- License: see repository
- Issues / feedback: open a GitHub issue
"""
)

st.markdown("## Disclaimer")
st.markdown(
    """
LivestockBrief is **educational and informational only**. The forecasts
shown here are model outputs from classical statistical methods applied to
public USDA data, with calibrated prediction intervals. They are not
investment, trading, hedging, or financial advice of any kind. The authors
make no representations about the accuracy or completeness of the data and
accept no liability for decisions made based on this site.

Read the [Methodology](/Methodology) page for the full picture of what these
numbers are and what they aren't.
"""
)
