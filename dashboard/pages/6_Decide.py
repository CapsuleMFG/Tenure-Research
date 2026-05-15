"""Decide — sell-now / hold recommendation.

Synthesizes today's cash + today's futures + basis + the user's breakeven
+ the cached 6-month forecast into a deterministic recommendation.
All math is in :mod:`usda_sandbox.decision`; this page is just the form
and the rendering.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import streamlit as st
from components.cache import get_series_entry
from components.sidebar import DEFAULT_OBS_PATH, render_sidebar
from components.theme import (
    ACCENT,
    INK_SOFT,
    PARCHMENT_DEEP,
    inject_global_css,
)

from usda_sandbox.basis import default_futures_peer, latest_basis
from usda_sandbox.decision import DecisionInputs, recommend
from usda_sandbox.store import read_series

inject_global_css()
render_sidebar(persistent_picker=False)

st.markdown("# Decide: sell now, hold, or hedge?")
st.markdown(
    f"<p style='color:{INK_SOFT};font-size:1.0rem;'>"
    "Combine today's cash, the nearby futures, your breakeven, and the "
    "6-month forecast into a transparent recommendation. The rules are "
    "deterministic and the reasoning shows every number it used."
    "</p>",
    unsafe_allow_html=True,
)

# --- Inputs -----------------------------------------------------------------

obs_path = Path(DEFAULT_OBS_PATH)

CASH_OPTIONS: dict[str, str] = {
    "cattle_steer_choice_nebraska": "Fed steers — Nebraska Choice (65–80%)",
    "cattle_steer_choice_tx_ok_nm": "Fed steers — TX/OK/NM Choice (35–65%)",
    "cattle_feeder_steer_750_800": "Feeder steers — 750–800 lb, Oklahoma",
    "cattle_feeder_steer_500_550": "Feeder steers — 500–550 lb, Oklahoma",
    "hog_barrow_gilt_natbase_51_52": "Hogs — National Base 51–52% lean",
}

col_a, col_b = st.columns([2, 1])
with col_a:
    cash_series_id = st.selectbox(
        "Which commodity / region?",
        options=list(CASH_OPTIONS.keys()),
        format_func=lambda k: CASH_OPTIONS[k],
        key="decide_cash_series",
    )
with col_b:
    breakeven = st.number_input(
        "Your breakeven ($/cwt)",
        min_value=50.0, max_value=600.0,
        value=float(st.session_state.get("breakeven_per_cwt", 214.0)),
        step=1.0,
        help="From the Breakeven page or wherever you compute it.",
    )

futures_series_id = default_futures_peer(cash_series_id)
if futures_series_id is None:
    st.error(f"No conventional futures peer for {cash_series_id}.")
    st.stop()
daily_futures_series_id = f"{futures_series_id}_daily"

# --- Load latest prices -----------------------------------------------------

cash_df = (
    read_series(cash_series_id, obs_path)
    .filter(pl.col("value").is_not_null())
    .sort("period_start")
)
if cash_df.is_empty():
    st.error("No cash observations available for this series yet.")
    st.stop()
latest_cash_row = cash_df.tail(1).row(0, named=True)
cash_now = float(latest_cash_row["value"])
cash_date: date = latest_cash_row["period_start"]
unit = latest_cash_row["unit"]

# Prefer daily futures for "today's futures"; fall back to monthly.
try:
    daily_fut = (
        read_series(daily_futures_series_id, obs_path)
        .filter(pl.col("value").is_not_null())
        .sort("period_start")
    )
except Exception:
    daily_fut = pl.DataFrame()

if not daily_fut.is_empty():
    fr = daily_fut.tail(1).row(0, named=True)
    futures_now = float(fr["value"])
    futures_date: date = fr["period_start"]
    futures_source = "daily front-month (yfinance)"
else:
    monthly_fut = (
        read_series(futures_series_id, obs_path)
        .filter(pl.col("value").is_not_null())
        .sort("period_start")
    )
    if monthly_fut.is_empty():
        st.error("No futures observations available; cannot compute basis.")
        st.stop()
    mr = monthly_fut.tail(1).row(0, named=True)
    futures_now = float(mr["value"])
    futures_date = mr["period_start"]
    futures_source = "monthly front-month (yfinance)"

# Basis: latest overlapping cash/futures (monthly resolution for basis since
# cash is monthly). Fall back to today's cash − today's futures if no overlap.
b = latest_basis(cash_series_id, futures_series_id, obs_path=obs_path)
if b is not None:
    basis_now, basis_date = b
else:
    basis_now = cash_now - futures_now
    basis_date = cash_date

# 6-month forecast from cache
fc_entry = get_series_entry(cash_series_id)
if fc_entry is None or not fc_entry.get("forward"):
    st.warning(
        "No cached forecast yet for this series. Run "
        "`python -m usda_sandbox.precompute` (or use **Admin → Rebuild "
        "forecast cache** in the sidebar) to populate it."
    )
    st.stop()
fwd = fc_entry["forward"]
target_index = min(5, len(fwd) - 1)  # 6-month forward (0-indexed)
fwd_target = fwd[target_index]

inputs = DecisionInputs(
    cash_now=cash_now,
    futures_now=futures_now,
    basis_now=basis_now,
    breakeven_per_cwt=breakeven,
    forecast_point=float(fwd_target["point"]),
    forecast_pi_lo=float(fwd_target["lower_80"]),
    forecast_pi_hi=float(fwd_target["upper_80"]),
    unit=unit,
)
rec = recommend(inputs)

# --- Render -----------------------------------------------------------------

# Headline
action_label = {
    "sell_now": "SELL NOW",
    "sell_with_downside": "SELL NOW (downside risk)",
    "hold": "HOLD",
    "hedge_or_hold": "HEDGE OR HOLD",
    "neutral": "WAIT",
}[rec.action]
action_color = {
    "sell_now": "#3E7D5A",
    "sell_with_downside": "#3E7D5A",
    "hold": ACCENT,
    "hedge_or_hold": "#A36829",
    "neutral": INK_SOFT,
}[rec.action]

st.markdown(
    f"<div style='border:2px solid {action_color};border-radius:10px;"
    f"padding:1.2rem 1.4rem;margin:1rem 0;'>"
    f"<div style='font-size:0.78rem;color:{INK_SOFT};letter-spacing:0.1em;"
    f"text-transform:uppercase;'>Recommendation</div>"
    f"<div style='font-size:1.8rem;font-weight:600;color:{action_color};"
    f"font-family:Iowan Old Style, Source Serif Pro, Georgia, serif;'>"
    f"{action_label}</div>"
    f"<div style='font-size:1.05rem;color:#1F1F1F;margin-top:0.3rem;'>"
    f"{rec.headline}</div>"
    f"</div>",
    unsafe_allow_html=True,
)

st.markdown("### The numbers")
m1, m2, m3, m4 = st.columns(4)
m1.metric(
    "Cash today",
    f"${cash_now:,.2f}",
    delta=f"{cash_date}",
    delta_color="off",
)
m2.metric(
    "Futures",
    f"${futures_now:,.2f}",
    delta=f"{futures_date}",
    delta_color="off",
)
m3.metric(
    "Basis",
    f"${basis_now:+,.2f}",
    delta=f"vs {basis_date}",
    delta_color="off",
)
m4.metric(
    "Breakeven",
    f"${breakeven:,.2f}",
    delta=f"margin ${rec.margin_today:+,.2f}",
    delta_color="normal" if rec.margin_today >= 0 else "inverse",
)

st.markdown("### 6-month forecast")
n1, n2, n3 = st.columns(3)
n1.metric(
    "Point forecast",
    f"${inputs.forecast_point:,.2f}",
    delta=f"margin ${rec.margin_6m:+,.2f}",
    delta_color="normal" if rec.margin_6m >= 0 else "inverse",
)
n2.metric(
    "PI lower (80%)",
    f"${inputs.forecast_pi_lo:,.2f}",
    delta=f"margin ${rec.margin_6m_lo:+,.2f}",
    delta_color="normal" if rec.margin_6m_lo >= 0 else "inverse",
)
n3.metric(
    "PI upper (80%)",
    f"${inputs.forecast_pi_hi:,.2f}",
    delta=f"margin ${rec.margin_6m_hi:+,.2f}",
    delta_color="normal" if rec.margin_6m_hi >= 0 else "inverse",
)

st.markdown(
    f"<div style='background:{PARCHMENT_DEEP};border-radius:8px;"
    f"padding:1rem 1.2rem;margin:0.6rem 0 1rem 0;font-size:0.95rem;"
    f"line-height:1.55;color:#1F1F1F;'>"
    f"<strong>Reasoning.</strong> {rec.reasoning}"
    f"</div>",
    unsafe_allow_html=True,
)

with st.expander("How this recommendation is computed"):
    st.markdown(
        """
The decision tool applies a small set of deterministic rules to today's
inputs (rules are matched top-down; first match wins):

1. **Sell now — capture today's margin.** Today's margin is positive and
   the 6-month forecast doesn't beat it by more than 10%. Holding adds
   real feeding/yardage/price risk for limited expected gain.
2. **Sell now — downside risk dominates.** Today's margin is positive but
   the 80% PI lower bound at 6 months dips below breakeven. The bad
   scenario in the PI is meaningfully bad; the good cash now is real.
3. **Hold.** Today's a loss, but the 6-month PI is entirely above
   breakeven. Wait if working capital and pen space allow.
4. **Hedge or hold and reassess.** Both today's cash and 6-month point
   are below breakeven. Consider locking in via futures hedge, or wait
   for the picture to clarify.
5. **Wait / neutral.** Margins positive both today and forward, neither
   dominates. A short 30–60 day wait is reasonable.

The 10% buffer in rule 1 is intentionally generous: over a 6-month feed
cycle, the forecast has to *clearly* beat today's price to justify the
holding risk.

> **Not financial advice.** This is a reasoning aid. The recommendation
> doesn't account for your cash flow, working capital, pen space, hedge
> position, or local basis dynamics beyond what shows up in the cash
> series. Use it to structure the question, then make your own call.
        """
    )

# --- Footer source / freshness --------------------------------------------

st.markdown("---")
st.markdown(
    f"<p style='color:{INK_SOFT};font-size:0.85rem;'>"
    f"Cash source: <code>{cash_series_id}</code> (monthly USDA ERS, "
    f"latest {cash_date}). Futures: {futures_source}, latest {futures_date}. "
    f"Forecast: {fc_entry.get('winner_model', 'cached')} at horizon "
    f"{target_index + 1}, calibrated 80% PI."
    "</p>",
    unsafe_allow_html=True,
)
