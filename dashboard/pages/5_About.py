"""About — credits, version, refresh status, GitHub link, disclaimer."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import streamlit as st
from components.cache import cache_generated_at, cache_horizon
from components.sidebar import DEFAULT_OBS_PATH, render_sidebar
from components.theme import BRAND_NAME, BRAND_TAGLINE, INK_SOFT, inject_global_css

from usda_sandbox import __version__

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
Tenure Brief is a free public dashboard built for **direct-market
cattle ranchers** — farms that raise, finish, and (often) slaughter
their own cattle, selling freezer beef directly to consumers as
quarters, halves, wholes, or retail cuts.

The headline tools (v3.0):

- **Plan** — model your cow-calf, stocker, or finish-and-direct
  operation. Pure cost-stack math; transparent line items; outputs
  per-head margins and annual operation P&L.
- **Costs** — today's feed grain (corn, soybean meal, oats), feeder
  cattle, and a hay reference, so you know what's moving on the input side.
- **Pricing** — research-derived ranges for share / hanging-weight /
  retail-cut pricing, with a calculator that converts your hanging
  weight + $/lb into share-size revenue.

Underneath the tools sits a USDA-grounded forecasting layer (AutoARIMA,
Prophet, LightGBM with calibrated 80% PIs) and daily front-month CME
futures via yfinance — the same data used for the macro-context cards
on the home page and the commodity-decide tool kept from v2.0.

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
Tenure Brief is **educational and informational only**. The forecasts
shown here are model outputs from classical statistical methods applied to
public USDA data, with calibrated prediction intervals. They are not
investment, trading, hedging, or financial advice of any kind. The authors
make no representations about the accuracy or completeness of the data and
accept no liability for decisions made based on this site.

Read the [Methodology](/Methodology) page for the full picture of what these
numbers are and what they aren't.
"""
)
