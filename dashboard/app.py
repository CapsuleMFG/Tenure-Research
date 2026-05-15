"""LivestockBrief — Streamlit entrypoint and landing page (the Brief).

Launch with::

    uv run streamlit run dashboard/app.py

Streamlit treats this file as the home page. Additional pages live under
``dashboard/pages/`` and are auto-discovered into the sidebar.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from components.brief import compose_brief, render_commodity_card
from components.cache import cache_generated_at, cached_forecast_cache
from components.sidebar import DEFAULT_OBS_PATH, render_sidebar
from components.theme import (
    BRAND_NAME,
    BRAND_TAGLINE,
    inject_global_css,
    set_page_chrome,
)

set_page_chrome(page_title="Brief")
inject_global_css()

render_sidebar(persistent_picker=False)

# Headline series featured on the home page, in display order.
HEADLINE_SERIES: list[tuple[str, str]] = [
    ("cattle_steer_choice_nebraska", "Fed Cattle"),
    ("cattle_feeder_steer_750_800", "Feeder Cattle"),
    ("hog_barrow_gilt_natbase_51_52", "Hogs"),
    ("boxed_beef_cutout_choice", "Beef Wholesale"),
    ("pork_cutout_composite", "Pork Wholesale"),
    ("lamb_slaughter_choice_san_angelo", "Lamb"),
]
LEAD_SERIES = "cattle_steer_choice_nebraska"

st.markdown(
    f"<h1 style='margin-bottom:0.1rem'>{BRAND_NAME}</h1>"
    f"<p style='color:#5A5550;font-size:1.02rem;margin-top:0;'>{BRAND_TAGLINE}</p>",
    unsafe_allow_html=True,
)

obs_path = Path(DEFAULT_OBS_PATH)
if not obs_path.exists():
    st.warning(
        "**Initial data sync needed.** Open the **Admin** expander in the "
        "sidebar and click *Refresh data*, then *Rebuild forecast cache*. "
        "On a deployed instance the scheduled refresh handles this; on first "
        "launch you do it once manually."
    )
    st.stop()

cache = cached_forecast_cache()
by_series = cache.get("by_series", {})

if not by_series:
    st.info(
        "**Forecast snapshot generating.** The cleaned data is loaded but "
        "the forecast cache hasn't been built yet. Open the **Admin** "
        "expander in the sidebar and click *Rebuild forecast cache*."
    )
    st.stop()

# ---- Headline brief on the lead series ------------------------------------

lead = by_series.get(LEAD_SERIES)
if lead is not None:
    brief_html = compose_brief(lead)
    if brief_html:
        st.markdown(
            f"<div class='lb-brief-headline'>{brief_html}</div>",
            unsafe_allow_html=True,
        )

# ---- Commodity-card grid --------------------------------------------------

st.markdown("### What the market looks like right now")

cols_per_row = 3
rows = [
    HEADLINE_SERIES[i : i + cols_per_row]
    for i in range(0, len(HEADLINE_SERIES), cols_per_row)
]
for row_idx, row in enumerate(rows):
    cols = st.columns(cols_per_row)
    for col_idx, (sid, group_label) in enumerate(row):
        with cols[col_idx]:
            entry = by_series.get(sid)
            if entry is None:
                st.markdown(
                    "<div class='lb-card'>"
                    f"<div class='lb-card-eyebrow'>{group_label.upper()}</div>"
                    "<div class='lb-card-title'>Series unavailable</div>"
                    "<div style='color:#5A5550;font-size:0.85rem;margin-top:0.6rem;'>"
                    "Not in the latest snapshot. See Methodology for "
                    "refresh status.</div></div>",
                    unsafe_allow_html=True,
                )
                continue
            render_commodity_card(entry, key_prefix=f"home_r{row_idx}c{col_idx}")
            if st.button(
                "View detail →",
                key=f"home_detail_{sid}",
                use_container_width=True,
            ):
                st.session_state["series_id"] = sid
                st.switch_page("pages/2_Series.py")

# ---- Footer hint ----------------------------------------------------------

st.markdown("---")
gen_at = cache_generated_at()
gen_label = gen_at[:10] if gen_at else "unavailable"
st.markdown(
    "<p style='color:#5A5550;font-size:0.85rem;'>"
    f"Latest forecast snapshot: <strong>{gen_label}</strong>. "
    "See <a href='/Methodology' target='_self'>Methodology</a> for how the "
    "forecasts are produced and how to read the prediction intervals. "
    "This page is informational and not financial advice."
    "</p>",
    unsafe_allow_html=True,
)
