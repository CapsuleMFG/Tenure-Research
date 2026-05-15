"""Tenure Brief — Streamlit entrypoint and landing page (the Brief).

Launch with::

    uv run streamlit run dashboard/app.py

Streamlit treats this file as the home page. Additional pages live under
``dashboard/pages/`` and are auto-discovered into the sidebar.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import streamlit as st
from components.brief import display_unit, render_commodity_card
from components.cache import cache_generated_at, cached_forecast_cache
from components.sidebar import DEFAULT_OBS_PATH, render_sidebar
from components.theme import (
    ACCENT,
    BRAND_NAME,
    BRAND_TAGLINE,
    DOWN,
    INK_SOFT,
    PARCHMENT_DEEP,
    UP,
    inject_global_css,
)

from usda_sandbox.direct_market import (
    compute_cow_calf_economics,
    cow_calf_inputs_for_region,
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

# ---- Hero -----------------------------------------------------------------

st.markdown(
    f"""
    <div style='padding:2.2rem 0 0.6rem 0;'>
      <div style='font-size:0.78rem;letter-spacing:0.12em;
                  text-transform:uppercase;color:{ACCENT};font-weight:600;'>
        A Tenure product
      </div>
      <h1 style='margin-top:0.3rem;margin-bottom:0.4rem;
                 font-size:2.6rem;line-height:1.1;letter-spacing:-0.02em;'>
        {BRAND_NAME}
      </h1>
      <p style='color:{INK_SOFT};font-size:1.15rem;max-width:720px;
                line-height:1.45;margin-top:0.4rem;margin-bottom:0;'>
        {BRAND_TAGLINE}<br>
        <span style='color:#1F1F1F;'>Plan your operation. Track your inputs.
        Set your freezer-beef pricing — with the math shown and the sources
        cited.</span>
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- "See it in action" sample-scenario hook -----------------------------
# A single live computation: what a default Southeast cow-calf operation
# looks like. Renders the result card from direct_market so a first-time
# visitor sees the Plan tool's output before they have to enter anything.

SAMPLE_REGION = "Southeast (TN/GA/KY/AR/FL)"
try:
    sample_inputs = cow_calf_inputs_for_region(SAMPLE_REGION)
    sample_econ = compute_cow_calf_economics(sample_inputs)
    sample_margin = sample_econ.margin_per_calf
    sample_total = sample_econ.total_margin
    sample_n = sample_econ.n_calves_weaned
    sample_be = sample_econ.breakeven_weaned_price_per_cwt
    sample_cost = sample_econ.annual_cost_per_cow
except Exception:
    sample_econ = None

if sample_econ is not None:
    margin_color = "#3E7D5A" if sample_margin >= 0 else ACCENT
    st.markdown(
        f"""
        <div style='border:1px solid rgba(180,82,30,0.25);border-radius:14px;
                    padding:1.4rem 1.6rem;margin:1.4rem 0 1rem 0;
                    background:linear-gradient(180deg, #FFFBF5 0%, {PARCHMENT_DEEP} 100%);'>
          <div style='font-size:0.78rem;letter-spacing:0.1em;
                      text-transform:uppercase;color:{INK_SOFT};
                      font-weight:600;margin-bottom:0.5rem;'>
            See it in action — sample run
          </div>
          <div style='font-size:1.0rem;color:#1F1F1F;margin-bottom:0.9rem;
                      line-height:1.5;max-width:720px;'>
            A typical <strong>Southeast cow-calf operation</strong> (60 cows,
            year-round fescue/bermuda pasture, ~1 ton hay/cow). Numbers below
            come straight from the <strong>Plan</strong> tool with the
            <em>Southeast</em> regional preset. The buttons below this
            box are the click targets — this card is illustrative only.
          </div>
          <div style='display:flex;flex-wrap:wrap;gap:1.6rem;
                      margin-top:0.4rem;'>
            <div>
              <div style='font-size:0.72rem;color:{INK_SOFT};
                          text-transform:uppercase;letter-spacing:0.08em;'>
                Annual cost per cow
              </div>
              <div style='font-size:1.6rem;font-weight:600;
                          font-family:Iowan Old Style,Source Serif Pro,Georgia,serif;'>
                ${sample_cost:,.0f}
              </div>
            </div>
            <div>
              <div style='font-size:0.72rem;color:{INK_SOFT};
                          text-transform:uppercase;letter-spacing:0.08em;'>
                Margin per calf
              </div>
              <div style='font-size:1.6rem;font-weight:600;color:{margin_color};
                          font-family:Iowan Old Style,Source Serif Pro,Georgia,serif;'>
                ${sample_margin:+,.0f}
              </div>
            </div>
            <div>
              <div style='font-size:0.72rem;color:{INK_SOFT};
                          text-transform:uppercase;letter-spacing:0.08em;'>
                Annual operation P&amp;L
              </div>
              <div style='font-size:1.6rem;font-weight:600;color:{margin_color};
                          font-family:Iowan Old Style,Source Serif Pro,Georgia,serif;'>
                ${sample_total:+,.0f}
              </div>
            </div>
            <div>
              <div style='font-size:0.72rem;color:{INK_SOFT};
                          text-transform:uppercase;letter-spacing:0.08em;'>
                Breakeven weaned price
              </div>
              <div style='font-size:1.6rem;font-weight:600;
                          font-family:Iowan Old Style,Source Serif Pro,Georgia,serif;'>
                ${sample_be:.0f}/cwt
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Primary CTA — single, clear, accent-colored.
cta_main_a, cta_main_b, cta_main_c = st.columns([2, 2, 2])
with cta_main_a:
    if st.button(
        "📋 Try it with your numbers →",
        type="primary", use_container_width=True, key="cta_plan_main",
    ):
        st.switch_page("dashboard/pages/6_Plan.py")
with cta_main_b:
    if st.button(
        "📈 See today's input costs",
        use_container_width=True, key="cta_costs_main",
    ):
        st.switch_page("dashboard/pages/7_Costs.py")
with cta_main_c:
    if st.button(
        "🥩 Pricing reference",
        use_container_width=True, key="cta_pricing_main",
    ):
        st.switch_page("dashboard/pages/8_Pricing.py")

# ---- What you'll get + how to use ----------------------------------------

st.markdown("&nbsp;", unsafe_allow_html=True)
st.markdown("### What you'll get")
st.caption(
    "The three tools that make up Tenure Brief. The buttons under each "
    "card are the click targets — the cards themselves are descriptions."
)

a, b, c = st.columns(3)
with a:
    st.markdown(
        "<div class='lb-card' style='height:100%;'>"
        "<div style='font-size:1.8rem;margin-bottom:0.3rem;'>📋</div>"
        "<div class='lb-card-title'>Your operation, modeled</div>"
        f"<div style='color:{INK_SOFT};font-size:0.9rem;line-height:1.5;"
        f"margin-top:0.4rem;'>"
        "Three modes — cow-calf, stocker, finish-and-direct. Six regional "
        "cost presets. Per-head margin and annual P&L with every cost "
        "line shown."
        "</div></div>",
        unsafe_allow_html=True,
    )
    if st.button("Open Plan →", key="wyg_plan",
                 use_container_width=True, type="tertiary"):
        st.switch_page("dashboard/pages/6_Plan.py")
with b:
    st.markdown(
        "<div class='lb-card' style='height:100%;'>"
        "<div style='font-size:1.8rem;margin-bottom:0.3rem;'>📈</div>"
        "<div class='lb-card-title'>Today's input costs</div>"
        f"<div style='color:{INK_SOFT};font-size:0.9rem;line-height:1.5;"
        f"margin-top:0.4rem;'>"
        "Daily corn, soybean meal, and feeder cattle prices from CME via "
        "yfinance + USDA ERS monthly cash. Take any number straight back "
        "to Plan."
        "</div></div>",
        unsafe_allow_html=True,
    )
    if st.button("Open Costs →", key="wyg_costs",
                 use_container_width=True, type="tertiary"):
        st.switch_page("dashboard/pages/7_Costs.py")
with c:
    st.markdown(
        "<div class='lb-card' style='height:100%;'>"
        "<div style='font-size:1.8rem;margin-bottom:0.3rem;'>🥩</div>"
        "<div class='lb-card-title'>Freezer-beef pricing</div>"
        f"<div style='color:{INK_SOFT};font-size:0.9rem;line-height:1.5;"
        f"margin-top:0.4rem;'>"
        "Hanging-weight ranges by finishing style, share-size calculator, "
        "and per-cut retail benchmarks. Anchored to USDA AMS + extension "
        "data + a real producer survey."
        "</div></div>",
        unsafe_allow_html=True,
    )
    if st.button("Open Pricing →", key="wyg_pricing",
                 use_container_width=True, type="tertiary"):
        st.switch_page("dashboard/pages/8_Pricing.py")

st.markdown("&nbsp;", unsafe_allow_html=True)
st.markdown("### How to use it")
st.markdown(
    f"""
    <div style='display:flex;flex-wrap:wrap;gap:1rem;margin-top:0.4rem;'>
      <div style='flex:1 1 220px;padding:0.85rem 1.1rem;border-radius:10px;
                  background:{PARCHMENT_DEEP};border-left:3px solid {ACCENT};'>
        <div style='font-size:0.72rem;color:{ACCENT};font-weight:700;
                    letter-spacing:0.1em;'>STEP 1</div>
        <div style='font-size:0.95rem;line-height:1.45;margin-top:0.25rem;'>
          Open <strong>Plan</strong>. Pick the tab matching your operation —
          cow-calf, stocker, or finish-and-direct.
        </div>
      </div>
      <div style='flex:1 1 220px;padding:0.85rem 1.1rem;border-radius:10px;
                  background:{PARCHMENT_DEEP};border-left:3px solid {ACCENT};'>
        <div style='font-size:0.72rem;color:{ACCENT};font-weight:700;
                    letter-spacing:0.1em;'>STEP 2</div>
        <div style='font-size:0.95rem;line-height:1.45;margin-top:0.25rem;'>
          Pick your <strong>region</strong>. Pasture, hay, vet, and other
          costs pre-fill with typical numbers for your geography.
        </div>
      </div>
      <div style='flex:1 1 220px;padding:0.85rem 1.1rem;border-radius:10px;
                  background:{PARCHMENT_DEEP};border-left:3px solid {ACCENT};'>
        <div style='font-size:0.72rem;color:{ACCENT};font-weight:700;
                    letter-spacing:0.1em;'>STEP 3</div>
        <div style='font-size:0.95rem;line-height:1.45;margin-top:0.25rem;'>
          Adjust any input to match your real operation. Math updates
          live — the margin card at the bottom is your answer.
        </div>
      </div>
      <div style='flex:1 1 220px;padding:0.85rem 1.1rem;border-radius:10px;
                  background:{PARCHMENT_DEEP};border-left:3px solid {ACCENT};'>
        <div style='font-size:0.72rem;color:{ACCENT};font-weight:700;
                    letter-spacing:0.1em;'>STEP 4</div>
        <div style='font-size:0.95rem;line-height:1.45;margin-top:0.25rem;'>
          <strong>Save scenarios</strong> by name (browser localStorage) or
          <strong>copy a shareable link</strong> that bundles your inputs.
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

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


st.markdown("&nbsp;", unsafe_allow_html=True)
st.markdown("---")

with st.expander("📊 Today's market context", expanded=False):
    st.caption(
        "Optional context — feeder cattle, live cattle, and corn front-"
        "month futures from CME via yfinance. Cash prices settle toward "
        "these through basis. Use them as macro signals, not as the price "
        "you'll get."
    )
    _render_futures_strip()
    st.markdown("&nbsp;", unsafe_allow_html=True)
    st.markdown("**Reference series — monthly USDA ERS**")
    st.caption(
        "Six-month forecasts with calibrated 80% prediction intervals. "
        "Feeder cards inform buy-side decisions; wholesale cards signal "
        "downstream demand."
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
    f"""
    <div style='display:flex;justify-content:space-between;flex-wrap:wrap;
                gap:0.6rem;margin-top:1rem;'>
      <p style='color:{INK_SOFT};font-size:0.82rem;margin:0;
                max-width:560px;'>
        Built by <strong>Tenure</strong>. The math is honest and the sources
        are cited — see <a href='/Methodology' target='_self'>Methodology</a>
        for how the data flows and <a href='/About' target='_self'>About</a>
        for the project background. <strong>Not financial advice.</strong>
      </p>
      <p style='color:{INK_SOFT};font-size:0.82rem;margin:0;
                text-align:right;'>
        Forecast snapshot: <code>{gen_label}</code>
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)
