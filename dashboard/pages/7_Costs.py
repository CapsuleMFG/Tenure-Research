"""Costs — today's input prices for the direct-market rancher.

Single read-only page: snapshot of the prices that drive a rancher's
operating costs and revenue. No editing, no decisions — this is the
"what does my market look like this morning" page.

Includes:
* Feeder cattle (buy side): GF daily front + Oklahoma 500 lb auction.
* Feed grains: corn, soybean meal, oats — latest closes + 1-month spark.
* Hay reference: a static national-average band with a "enter your local"
  field that the user can persist for use in the Plan page.
* Cull-cow proxy: boxed-beef cutout trend (since cull cow prices track it).
"""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import polars as pl
import streamlit as st
from components.sidebar import DEFAULT_OBS_PATH, render_sidebar
from components.theme import (
    ACCENT,
    DOWN,
    INK_SOFT,
    PARCHMENT_DEEP,
    UP,
    inject_global_css,
)

from usda_sandbox.store import read_series

inject_global_css()
render_sidebar(persistent_picker=False)

st.markdown("# Today's input costs")
st.markdown(
    f"<p style='color:{INK_SOFT};font-size:1.0rem;line-height:1.5;"
    f"max-width:780px;'>"
    "What your operation pays for, today. Use this page as a "
    "<strong>directional read</strong> on input costs — if corn is up "
    "10% over the last month, your finishing feed budget needs the same "
    "bump. Take any price you see here back to <strong>Plan</strong> as "
    "an updated cost input."
    "</p>",
    unsafe_allow_html=True,
)

obs_path = Path(DEFAULT_OBS_PATH)


def _series_history(series_id: str, months: int | None = None) -> pl.DataFrame:
    try:
        df = (
            read_series(series_id, obs_path)
            .filter(pl.col("value").is_not_null())
            .sort("period_start")
        )
    except Exception:
        return pl.DataFrame()
    if months is not None and df.height > months * 22:
        df = df.tail(months * 22)
    return df


def _spark(df: pl.DataFrame, color: str = ACCENT) -> go.Figure:
    if df.is_empty():
        return go.Figure()
    fig = go.Figure(
        go.Scatter(
            x=df["period_start"].to_list(),
            y=df["value"].to_list(),
            mode="lines",
            line=dict(color=color, width=1.5),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        height=60, margin=dict(l=0, r=0, t=4, b=4),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig


def _pct_change(df: pl.DataFrame, lookback_rows: int) -> float | None:
    if df.height < lookback_rows + 1:
        return None
    last = float(df["value"].tail(1).item())
    earlier = float(df["value"].tail(lookback_rows + 1).head(1).item())
    if earlier == 0:
        return None
    return (last / earlier - 1.0) * 100.0


# --- Feeder cattle ---------------------------------------------------------

st.markdown("### Feeder cattle (replacement / buy side)")
feeder_pairs = [
    ("cattle_feeder_front_daily", "Feeder front-month (GF)", "daily"),
    ("cattle_feeder_steer_500_550", "500-550 lb, OK auction", "monthly"),
    ("cattle_feeder_steer_750_800", "750-800 lb, OK auction", "monthly"),
]
cols = st.columns(len(feeder_pairs))
for col, (sid, label, freq) in zip(cols, feeder_pairs, strict=True):
    df = _series_history(sid, months=6)
    if df.is_empty():
        col.markdown(f"_{label}: data unavailable_")
        continue
    last = float(df["value"].tail(1).item())
    last_date = df["period_start"].tail(1).item()
    pct = _pct_change(df, 21 if freq == "daily" else 1)
    arrow = "—"
    color = INK_SOFT
    if pct is not None:
        if pct > 0.5:
            arrow = "▲"
            color = UP
        elif pct < -0.5:
            arrow = "▼"
            color = DOWN
    with col:
        st.markdown(
            "<div class='lb-card'>"
            f"<div class='lb-card-eyebrow'>{label.upper()}</div>"
            f"<div class='lb-card-title'>{freq.capitalize()}, {last_date}</div>"
            f"<div class='lb-card-price'>${last:,.2f}"
            "<span class='lb-card-unit'>/cwt</span></div>"
            + (
                f"<div class='lb-card-deltas'>"
                f"<span style='color:{color};font-weight:600;'>"
                f"{arrow} {abs(pct):.2f}%</span> "
                f"<span style='color:{INK_SOFT}'>"
                f"{'1-mo' if freq == 'daily' else 'last vs prior'}</span></div>"
                if pct is not None else ""
            )
            + "</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(_spark(df), use_container_width=True,
                        config={"displayModeBar": False}, key=f"sp_{sid}")

# --- Feed grains -----------------------------------------------------------

st.markdown("### Feed grains (corn, soybean meal, oats)")
grain_pairs = [
    ("corn_front_daily", "Corn (ZC)", "cents/bushel"),
    ("soybean_meal_front_daily", "Soybean meal (ZM)", "USD/short_ton"),
    ("oats_front_daily", "Oats (ZO)", "cents/bushel"),
]
cols = st.columns(len(grain_pairs))
for col, (sid, label, unit) in zip(cols, grain_pairs, strict=True):
    df = _series_history(sid, months=6)
    if df.is_empty():
        col.markdown(f"_{label}: data unavailable_")
        continue
    last = float(df["value"].tail(1).item())
    last_date = df["period_start"].tail(1).item()
    pct = _pct_change(df, 21)
    arrow = "—"
    color = INK_SOFT
    if pct is not None:
        if pct > 0.5:
            arrow = "▲"
            color = UP
        elif pct < -0.5:
            arrow = "▼"
            color = DOWN
    display_value = (
        f"${last/100:.2f}/bu"
        if unit == "cents/bushel"
        else f"${last:,.0f}/short ton"
    )
    with col:
        st.markdown(
            "<div class='lb-card'>"
            f"<div class='lb-card-eyebrow'>{label.upper()}</div>"
            f"<div class='lb-card-title'>Daily, {last_date}</div>"
            f"<div class='lb-card-price'>{display_value}</div>"
            + (
                f"<div class='lb-card-deltas'>"
                f"<span style='color:{color};font-weight:600;'>"
                f"{arrow} {abs(pct):.2f}%</span> "
                f"<span style='color:{INK_SOFT}'>1-mo change</span></div>"
                if pct is not None else ""
            )
            + "</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(_spark(df), use_container_width=True,
                        config={"displayModeBar": False}, key=f"sp_{sid}")

# --- Hay reference ---------------------------------------------------------

st.markdown("### Hay reference")
st.markdown(
    f"<div style='background:{PARCHMENT_DEEP};border-radius:8px;"
    f"padding:0.9rem 1.1rem;font-size:0.95rem;line-height:1.55;'>"
    "Hay is hyperlocal — drought zones, transport distance, and bale "
    "format (small square vs round vs large square) move prices "
    "significantly. As a starting reference, USDA NASS late-2020s "
    "national averages have run roughly:"
    "<ul style='margin-top:0.5rem;margin-bottom:0.5rem;'>"
    "<li><strong>Alfalfa hay:</strong> $200-$280/short ton</li>"
    "<li><strong>Mixed grass hay:</strong> $160-$220/short ton</li>"
    "<li><strong>Premium dairy-quality alfalfa:</strong> $260-$340/short ton</li>"
    "</ul>"
    "Drought years (e.g., 2022-23 across the Plains) push these up "
    "30-60%. Your local price is the only one that matters for your "
    "operation — enter it below to flow through to the Plan page."
    "</div>",
    unsafe_allow_html=True,
)
local_hay = st.number_input(
    "Your local hay price ($/short ton)",
    min_value=0.0, max_value=600.0,
    value=float(st.session_state.get("local_hay_per_ton", 220.0)),
    step=5.0,
    help="Saved across pages — used as the default on the Plan / Cow-calf tab.",
)
st.session_state["local_hay_per_ton"] = local_hay

# --- Known gap: cull cow data ----------------------------------------------

st.markdown("### Known gap — cull cow market")
st.markdown(
    f"<div style='background:{PARCHMENT_DEEP};border-radius:8px;"
    f"padding:0.9rem 1.1rem;font-size:0.92rem;line-height:1.55;'>"
    "<strong>We don't track cull cow auction prices.</strong> A previous "
    "build of this page tried to proxy them via the boxed-beef-cutout "
    "trend, but that signal is several steps removed from your local "
    "sale barn and we'd rather show no data than misleading data. "
    "<br><br>"
    "Cull cow values matter for replacement-buying decisions and for "
    "deciding when to move open or old cows. A future version could "
    "ingest <a href='https://www.ams.usda.gov/market-news/livestock-poultry-and-grain-market-news' "
    "target='_blank' rel='noopener'>AMS LMR weekly cow auction</a> "
    "summary reports (LM_CT170 series); the obstacle is that USDA's "
    "MARS API requires eAuth Level 2 identity proofing, which we've "
    "kept out of scope for an open-access dashboard."
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("---")
st.caption(
    "All daily values are from CME via yfinance (front-month continuous, "
    "not contract-specific). Monthly cash values are from USDA ERS. "
    "Your local prices will always be the truth — use these as "
    "directional signals."
)
