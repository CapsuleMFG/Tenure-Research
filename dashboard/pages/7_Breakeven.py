"""Breakeven — feedlot cost-of-production calculator.

A producer enters their placement assumptions; the page returns the
$/cwt sale price they must clear on the finished animal to break even.
The result is stashed in ``st.session_state["breakeven_per_cwt"]`` so
the Decide page can use it without forcing the user to re-enter.
"""

from __future__ import annotations

import streamlit as st
from components.sidebar import render_sidebar
from components.theme import (
    INK_SOFT,
    PARCHMENT_DEEP,
    inject_global_css,
)

from usda_sandbox.breakeven import (
    FeedlotInputs,
    compute_feedlot_economics,
    default_inputs,
)

inject_global_css()
render_sidebar(persistent_picker=False)

st.markdown("# Breakeven calculator")
st.markdown(
    f"<p style='color:{INK_SOFT};font-size:1.0rem;'>"
    "What price per cwt does the finished animal need to clear to recoup "
    "feeder cost, gain, yardage, interest, and death loss? Use this to "
    "set the breakeven that feeds the <strong>Decide</strong> tool."
    "</p>",
    unsafe_allow_html=True,
)

defaults = default_inputs()
session_inputs = st.session_state.get("breakeven_inputs")
if session_inputs is None:
    session_inputs = defaults

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("### Placement")
    feeder_cost = st.number_input(
        "Feeder cost ($/cwt)",
        min_value=50.0, max_value=500.0,
        value=float(session_inputs.feeder_cost_per_cwt), step=1.0,
        help="Price you paid for the feeder calf, per hundredweight.",
    )
    feeder_weight = st.number_input(
        "Feeder placement weight (lbs)",
        min_value=300.0, max_value=1000.0,
        value=float(session_inputs.feeder_weight_lbs), step=10.0,
    )
    finished_weight = st.number_input(
        "Target finished weight (lbs)",
        min_value=900.0, max_value=1700.0,
        value=float(session_inputs.finished_weight_lbs), step=10.0,
    )
    days_on_feed = st.number_input(
        "Days on feed",
        min_value=60, max_value=300,
        value=int(session_inputs.days_on_feed), step=5,
    )

with col_b:
    st.markdown("### Costs")
    cost_of_gain = st.number_input(
        "Cost of gain ($/lb)",
        min_value=0.50, max_value=3.00,
        value=float(session_inputs.cost_of_gain_per_lb), step=0.05,
        help="Feed + medicine + variable per pound of gain. KSU range: ~$1.00–$1.40.",
    )
    yardage = st.number_input(
        "Yardage ($/head/day)",
        min_value=0.10, max_value=2.00,
        value=float(session_inputs.yardage_per_day), step=0.05,
        help="Pen rental, labor, utilities, manure handling. Typical: $0.50–$0.70.",
    )
    interest_rate = st.number_input(
        "Annual interest rate (%)",
        min_value=0.0, max_value=20.0,
        value=float(session_inputs.interest_rate_annual * 100), step=0.25,
        help="Applied to the feeder cost over days on feed.",
    ) / 100.0
    death_loss = st.number_input(
        "Death loss (%)",
        min_value=0.0, max_value=5.0,
        value=float(session_inputs.death_loss_pct * 100), step=0.1,
        help="Typical: 0.5–1.5%.",
    ) / 100.0

inputs = FeedlotInputs(
    feeder_cost_per_cwt=feeder_cost,
    feeder_weight_lbs=feeder_weight,
    finished_weight_lbs=finished_weight,
    days_on_feed=days_on_feed,
    cost_of_gain_per_lb=cost_of_gain,
    yardage_per_day=yardage,
    interest_rate_annual=interest_rate,
    death_loss_pct=death_loss,
)

try:
    econ = compute_feedlot_economics(inputs)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

st.session_state["breakeven_inputs"] = inputs
st.session_state["breakeven_per_cwt"] = econ.breakeven_per_cwt

# --- Headline result ---------------------------------------------------------

st.markdown(
    f"<div class='lb-brief-headline'>"
    f"Your breakeven is <em>${econ.breakeven_per_cwt:,.2f}/cwt</em> on the "
    f"finished animal. Total cost per surviving head: "
    f"<strong>${econ.total_cost:,.2f}</strong>."
    f"</div>",
    unsafe_allow_html=True,
)

st.markdown("### Cost breakdown per head")
m1, m2, m3 = st.columns(3)
m1.metric("Feeder cost", f"${econ.feeder_cost_total:,.0f}")
m2.metric("Cost of gain", f"${econ.cost_of_gain_total:,.0f}")
m3.metric("Yardage", f"${econ.yardage_total:,.0f}")
m4, m5, m6 = st.columns(3)
m4.metric("Interest", f"${econ.interest_total:,.0f}")
m5.metric("Death-loss spread", f"${econ.death_loss_addon:,.0f}")
m6.metric("Total", f"${econ.total_cost:,.0f}")

# --- Sensitivity hint --------------------------------------------------------

st.markdown(
    f"<div style='background:{PARCHMENT_DEEP};border-radius:6px;"
    f"padding:0.7rem 1rem;margin-top:1rem;font-size:0.9rem;color:#1F1F1F;'>"
    f"<strong>Sensitivity at-a-glance.</strong> A $0.10/lb change in cost "
    f"of gain shifts breakeven by about <strong>"
    f"${(finished_weight - feeder_weight) * 0.10 / finished_weight * 100:.2f}/cwt"
    f"</strong>. A $10/cwt change in feeder cost shifts breakeven by about "
    f"<strong>${10 * feeder_weight / finished_weight:.2f}/cwt</strong>."
    f"</div>",
    unsafe_allow_html=True,
)

# --- Decide jump -------------------------------------------------------------

st.markdown("---")
st.markdown(
    f"<p style='color:{INK_SOFT};font-size:0.9rem;'>"
    f"Your breakeven of <strong>${econ.breakeven_per_cwt:,.2f}/cwt</strong> "
    f"is saved for the Decide tool.</p>",
    unsafe_allow_html=True,
)
if st.button(
    "Take this breakeven to the Decide tool →", type="primary", key="goto_decide"
):
    st.switch_page("dashboard/pages/6_Decide.py")

st.markdown(
    f"<p style='color:{INK_SOFT};font-size:0.8rem;margin-top:1.5rem;'>"
    "Defaults are KSU/ISU late-2020s feedlot averages. This is a "
    "single-placement calc; for breakeven on a pen full of mixed-weight "
    "cattle, run the calc per group. Not financial advice."
    "</p>",
    unsafe_allow_html=True,
)
