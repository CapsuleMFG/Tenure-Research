"""LivestockBrief — Streamlit entrypoint and landing page (the Brief).

Launch with::

    uv run streamlit run dashboard/app.py

Streamlit treats this file as the home page. Additional pages live under
``dashboard/pages/`` and are auto-discovered into the sidebar.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import streamlit as st
from components.brief import compose_brief, display_unit, render_commodity_card
from components.cache import cache_generated_at, cached_forecast_cache
from components.sidebar import DEFAULT_OBS_PATH, render_sidebar
from components.theme import (
    BRAND_NAME,
    DOWN,
    INK_SOFT,
    UP,
    inject_global_css,
)

from usda_sandbox.store import read_series

inject_global_css()

render_sidebar(persistent_picker=False)

# Daily front-month strip on home, reframed for direct-market: feeder
# (replacement cost), live cattle (commodity floor), corn (supplement feed).
DAILY_FUT_SERIES: list[tuple[str, str]] = [
    ("cattle_feeder_front_daily", "Feeder Cattle (GF) — buy side"),
    ("cattle_lc_front_daily", "Live Cattle (LE) — commodity floor"),
    ("corn_front_daily", "Corn (ZC) — finishing feed"),
]

# Headline series featured on the home page, in display order. For the
# v3.0 direct-market audience, lead with feeder + buying-relevant series;
# keep boxed beef as a downstream/demand signal but de-emphasize the
# 5-area fed steer (which was the v2.0 commodity lead).
HEADLINE_SERIES: list[tuple[str, str]] = [
    ("cattle_feeder_steer_500_550", "Light Feeders (Buy)"),
    ("cattle_feeder_steer_750_800", "Placement Feeders (Sell)"),
    ("cattle_steer_choice_nebraska", "Fed Cattle (Floor)"),
    ("boxed_beef_cutout_choice", "Beef Wholesale (Demand)"),
    ("hog_barrow_gilt_natbase_51_52", "Hogs"),
    ("pork_cutout_composite", "Pork Wholesale"),
]
LEAD_SERIES = "cattle_feeder_steer_750_800"

st.markdown(
    f"<h1 style='margin-bottom:0.1rem'>{BRAND_NAME}</h1>"
    f"<p style='color:#5A5550;font-size:1.05rem;margin-top:0.2rem;"
    f"max-width:780px;line-height:1.45;'>"
    "<strong>For direct-market cattle ranchers</strong> — farms that "
    "raise, finish, and (often) slaughter their own cattle, selling "
    "freezer beef directly to consumers. Free. Public. No login."
    "</p>",
    unsafe_allow_html=True,
)

# --- Quick-start onboarding panel -----------------------------------------

with st.container(border=True):
    st.markdown("#### What you'll get in 5 minutes")
    a, b, c = st.columns(3)
    with a:
        st.markdown(
            "<div style='font-size:0.92rem;line-height:1.5;'>"
            "<div style='font-size:1.5rem;'>📋</div>"
            "<strong>Your operation, modeled</strong><br>"
            f"<span style='color:{INK_SOFT}'>Pick cow-calf, stocker, or "
            "finish-and-direct. Pick your region (pre-fills typical "
            "costs). Plug in your numbers. Get per-head margin and "
            "annual P&L.</span></div>",
            unsafe_allow_html=True,
        )
    with b:
        st.markdown(
            "<div style='font-size:0.92rem;line-height:1.5;'>"
            "<div style='font-size:1.5rem;'>📈</div>"
            "<strong>Today's input costs</strong><br>"
            f"<span style='color:{INK_SOFT}'>Daily corn, soybean meal, "
            "and feeder cattle prices — the inputs that drive your "
            "real costs. From CME futures and USDA ERS.</span></div>",
            unsafe_allow_html=True,
        )
    with c:
        st.markdown(
            "<div style='font-size:0.92rem;line-height:1.5;'>"
            "<div style='font-size:1.5rem;'>🥩</div>"
            "<strong>Freezer-beef pricing</strong><br>"
            f"<span style='color:{INK_SOFT}'>Research-derived $/lb "
            "hanging ranges for grain-finished, grass-finished, and "
            "premium-branded beef. Plus a share-size calculator.</span>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.markdown("**How to use it**")
    st.markdown(
        "<ol style='color:#1F1F1F;font-size:0.92rem;line-height:1.55;"
        "margin-top:0.2rem;padding-left:1.2rem;'>"
        "<li>Go to <strong>Plan</strong> (sidebar). Pick the tab that "
        "matches your operation: cow-calf, stocker, or finish-and-direct.</li>"
        "<li>Pick your <strong>region</strong> from the dropdown. The cost "
        "stack pre-fills with typical numbers for your geography.</li>"
        "<li>Adjust any input to match your reality. The math updates live; "
        "every cost line is shown so nothing is hidden.</li>"
        "<li>Click <strong>🔗 Copy shareable link</strong> at the bottom "
        "to save your inputs in the URL. Or use <strong>📋 Save / load "
        "scenarios</strong> at the top to keep multiple named versions "
        "in your browser.</li>"
        "</ol>",
        unsafe_allow_html=True,
    )

    cta_a, cta_b, cta_c = st.columns([1, 1, 1])
    with cta_a:
        if st.button(
            "📋 Start with Plan", type="primary", use_container_width=True,
            key="cta_plan",
        ):
            st.switch_page("dashboard/pages/6_Plan.py")
    with cta_b:
        if st.button(
            "📈 See today's costs", use_container_width=True,
            key="cta_costs",
        ):
            st.switch_page("dashboard/pages/7_Costs.py")
    with cta_c:
        if st.button(
            "🥩 Pricing reference", use_container_width=True,
            key="cta_pricing",
        ):
            st.switch_page("dashboard/pages/8_Pricing.py")

obs_path = Path(DEFAULT_OBS_PATH)
if not obs_path.exists():
    st.info(
        "**Loading market data…** The dataset refresh is queued; this "
        "page will populate as soon as the next scheduled sync completes "
        "(typically within a few hours)."
    )
    st.stop()

cache = cached_forecast_cache()
by_series = cache.get("by_series", {})

if not by_series:
    st.info(
        "**Forecast snapshot generating…** The cleaned data is in place "
        "and the next forecast bake is queued. Check back shortly."
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

# ---- Today's futures (daily refresh) --------------------------------------


def _render_futures_strip() -> None:
    """One-line strip of today's three front-month futures with day-over-day %."""
    cols = st.columns(len(DAILY_FUT_SERIES))
    for col, (sid, label) in zip(cols, DAILY_FUT_SERIES, strict=True):
        try:
            df = (
                read_series(sid, obs_path)
                .filter(pl.col("value").is_not_null())
                .sort("period_start")
            )
        except Exception:
            df = pl.DataFrame()
        if df.height < 2:
            with col:
                st.markdown(
                    f"<div class='lb-card'>"
                    f"<div class='lb-card-eyebrow'>{label}</div>"
                    f"<div class='lb-card-title'>Daily futures unavailable</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            continue
        last = float(df["value"].tail(1).item())
        prev = float(df["value"].tail(2).head(1).item())
        last_date = df["period_start"].tail(1).item()
        pct = (last / prev - 1.0) * 100.0 if prev else 0.0
        arrow = "▲" if pct > 0.05 else ("▼" if pct < -0.05 else "▬")
        color = UP if pct > 0.05 else (DOWN if pct < -0.05 else INK_SOFT)
        # Corn is reported in cents/bushel; convert to $/bu for display
        if sid == "corn_front_daily":
            display_value = (
                f"${last/100:.2f}<span class='lb-card-unit'>/bu</span>"
            )
        else:
            short = display_unit("USD/cwt")
            display_value = (
                f"${last:,.2f}<span class='lb-card-unit'>/{short}</span>"
            )
        with col:
            st.markdown(
                "<div class='lb-card'>"
                f"<div class='lb-card-eyebrow'>{label.upper()}</div>"
                f"<div class='lb-card-title'>Front-month, {last_date}</div>"
                f"<div class='lb-card-price'>{display_value}</div>"
                f"<div class='lb-card-deltas'>"
                f"<span style='color:{color};font-weight:600;'>"
                f"{arrow} {abs(pct):.2f}%</span> "
                f"<span style='color:{INK_SOFT}'>day-over-day</span></div>"
                "</div>",
                unsafe_allow_html=True,
            )


st.markdown("---")
st.markdown("### Today's front-month futures")
st.caption(
    "CME daily closes via yfinance. These are the prices the industry "
    "discovers daily — your local cash settles toward them through basis."
)
_render_futures_strip()

# ---- Commodity-card grid --------------------------------------------------

st.markdown("### Reference series — what the market looks like right now")
st.caption(
    "Monthly USDA ERS prices with 6-month forecasts and recent trend. "
    "Use these as macro context — the feeder cards are most relevant to "
    "buy-side decisions; the wholesale cards signal downstream demand."
)

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
                "Open series →",
                key=f"home_detail_{sid}",
                use_container_width=True,
                type="tertiary",
            ):
                st.session_state["series_id"] = sid
                st.switch_page("dashboard/pages/2_Series.py")

# ---- Footer hint ----------------------------------------------------------

st.markdown("---")
gen_at = cache_generated_at()
gen_label = gen_at[:10] if gen_at else "unavailable"
st.markdown(
    "<p style='color:#5A5550;font-size:0.85rem;'>"
    f"Latest forecast snapshot: <strong>{gen_label}</strong>. "
    "See <a href='/Methodology' target='_self'>Methodology</a> for how "
    "the data is sourced and the math works, or <a href='/About' "
    "target='_self'>About</a> for the project background. "
    "This site is informational and not financial advice."
    "</p>",
    unsafe_allow_html=True,
)
