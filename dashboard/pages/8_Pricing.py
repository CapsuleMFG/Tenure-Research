"""Pricing — freezer-beef share + retail pricing reference.

Static reference data (research-derived ranges from extension surveys)
with a small calculator that converts the rancher's hanging weight +
$/lb into total share-size revenue. The numbers are intentionally ranges
rather than single values — pricing varies considerably by region,
grass-fed vs grain-fed, breed, and brand maturity.

Pair this page with Plan to bridge from "what does my operation cost?"
to "what should I charge?"
"""

from __future__ import annotations

import streamlit as st
from components.sidebar import render_sidebar
from components.theme import (
    INK_SOFT,
    PARCHMENT_DEEP,
    inject_global_css,
)

from usda_sandbox.direct_pricing import (
    REFERENCE_HANGING_PRICING,
    REFERENCE_RETAIL_BUNDLE,
    REFERENCE_SHARE_PRICING,
    expected_retail_yield_lbs,
    value_share,
)
from usda_sandbox.sources import PRICING_SOURCES

inject_global_css()
render_sidebar(persistent_picker=False)

st.markdown("# Pricing reference")
st.markdown(
    f"<p style='color:{INK_SOFT};font-size:1.0rem;'>"
    "Research-derived pricing ranges for direct-market beef, late 2020s. "
    "Use them as starting points and sanity checks against your local "
    "market — actual price depends on your region, breed, finishing "
    "style, brand maturity, and customer base."
    "</p>",
    unsafe_allow_html=True,
)

# --- Hanging-weight pricing -----------------------------------------------

st.markdown("### Hanging-weight pricing (by finishing style)")
st.caption(
    "Customer pays *your* price per pound of hanging (carcass) weight, "
    "plus the cut-and-wrap fees billed by the processor (~$0.75–$1.10/lb)."
)
cols = st.columns(len(REFERENCE_HANGING_PRICING))
for col, (key, rng) in zip(cols, REFERENCE_HANGING_PRICING.items(), strict=True):
    label = key.replace("_", " ").title()
    with col:
        st.markdown(
            "<div class='lb-card'>"
            f"<div class='lb-card-eyebrow'>{label.upper()}</div>"
            f"<div class='lb-card-title'>{rng.unit}</div>"
            f"<div class='lb-card-price'>${rng.mid:,.2f}</div>"
            f"<div class='lb-card-deltas'>"
            f"<span style='color:{INK_SOFT}'>range:</span> "
            f"<strong>${rng.low:,.2f} — ${rng.high:,.2f}</strong></div>"
            f"<div class='lb-card-fcst'>"
            f"<div style='font-size:0.82rem;color:{INK_SOFT};line-height:1.5;'>"
            f"{rng.note}</div></div>"
            "</div>",
            unsafe_allow_html=True,
        )

# --- Share-size calculator -------------------------------------------------

st.markdown("### What does a share cost the customer?")
st.caption(
    "Enter your hanging weight per animal and your $/lb hanging-weight "
    "price. We'll show what each share size yields you in gross revenue, "
    "and what your customer pays before they add the processor's "
    "cut-and-wrap fees."
)

# Default hanging weight either from a recent Plan-page calc or sensible default
default_hang = float(st.session_state.get("fd_hanging_lbs", 800.0))
c1, c2 = st.columns(2)
hang = c1.number_input(
    "Hanging weight per animal (lbs)",
    min_value=300.0, max_value=1200.0,
    value=default_hang, step=10.0,
)
price = c2.number_input(
    "Your hanging-weight price ($/lb)",
    min_value=2.0, max_value=20.0,
    value=6.50, step=0.10,
)

st.markdown("")
share_cols = st.columns(len(REFERENCE_SHARE_PRICING))
for col, share in zip(share_cols, REFERENCE_SHARE_PRICING, strict=True):
    s_hang, s_total = value_share(hang, share.fraction, price)
    s_retail = expected_retail_yield_lbs(s_hang)
    with col:
        st.markdown(
            "<div class='lb-card'>"
            f"<div class='lb-card-eyebrow'>{share.name.upper()} SHARE</div>"
            f"<div class='lb-card-title'>{int(share.fraction * 100)}% of a carcass</div>"
            f"<div class='lb-card-price'>${s_total:,.0f}"
            "<span class='lb-card-unit'>/share</span></div>"
            f"<div class='lb-card-deltas'>"
            f"<span style='color:{INK_SOFT}'>hanging:</span> "
            f"<strong>{s_hang:.0f} lbs</strong> &middot; "
            f"<span style='color:{INK_SOFT}'>retail cuts:</span> "
            f"<strong>~{s_retail:.0f} lbs</strong></div>"
            f"<div class='lb-card-fcst'>"
            f"<div style='font-size:0.78rem;color:{INK_SOFT};line-height:1.45;'>"
            f"Typical retail range for the share: "
            f"<strong>{share.typical_retail_lbs[0]:.0f}-"
            f"{share.typical_retail_lbs[1]:.0f} lbs</strong>"
            f"</div></div>"
            "</div>",
            unsafe_allow_html=True,
        )

# --- Retail cuts -----------------------------------------------------------

st.markdown("### Retail per-cut pricing (farmers market / CSA)")
st.caption(
    "If you sell individual cuts at a farmers market or via a monthly "
    "subscription box, here are typical late-2020s ranges. Ground beef "
    "is the majority of your inventory by weight; steaks and roasts "
    "carry the premium."
)
retail_cols = st.columns(len(REFERENCE_RETAIL_BUNDLE))
for col, (key, rng) in zip(retail_cols, REFERENCE_RETAIL_BUNDLE.items(), strict=True):
    label = key.replace("_", " ").title()
    with col:
        st.markdown(
            "<div class='lb-card'>"
            f"<div class='lb-card-eyebrow'>{label.upper()}</div>"
            f"<div class='lb-card-title'>{rng.unit}</div>"
            f"<div class='lb-card-price'>${rng.mid:.2f}</div>"
            f"<div class='lb-card-deltas'>"
            f"<span style='color:{INK_SOFT}'>range:</span> "
            f"<strong>${rng.low:.2f} — ${rng.high:.2f}</strong></div>"
            f"<div class='lb-card-fcst'>"
            f"<div style='font-size:0.78rem;color:{INK_SOFT};line-height:1.4;'>"
            f"{rng.note}</div></div>"
            "</div>",
            unsafe_allow_html=True,
        )

# --- Notes ----------------------------------------------------------------

st.markdown("---")
st.markdown(
    f"<div style='background:{PARCHMENT_DEEP};border-radius:8px;"
    f"padding:0.9rem 1.1rem;margin:0.6rem 0;font-size:0.92rem;"
    f"line-height:1.55;color:#1F1F1F;'>"
    f"<strong>How to use these.</strong> Start with the hanging-weight "
    f"range that matches your finishing style. Build your share pricing "
    f"first (it's how most direct-market customers buy). The retail-cut "
    f"prices are useful as a sanity check: "
    f"<em>does your share pricing imply per-pound retail values that "
    f"line up with what people pay at the farmers market?</em> If your "
    f"implied retail is way below the ground-beef range, you're "
    f"underpricing your shares; if way above, customers will balk."
    f"</div>",
    unsafe_allow_html=True,
)

st.caption(
    "Numbers are anchored to USDA AMS quarterly reports, extension-service "
    "publications, and a survey of real producer websites — see "
    "**Where these ranges come from** below. Always validate against your "
    "local market — there's no central index for direct-market beef "
    "pricing, and regional variation is substantial. Not financial advice."
)

with st.expander("Where these ranges come from"):
    lines = [
        f"<div style='font-size:0.8rem;color:{INK_SOFT};margin-bottom:0.4rem;"
        f"text-transform:uppercase;letter-spacing:0.08em;'>Pricing sources</div>"
    ]
    for s in PRICING_SOURCES:
        lines.append(
            f"<div style='font-size:0.85rem;line-height:1.5;"
            f"margin-bottom:0.55rem;'>"
            f"<a href='{s.url}' target='_blank' rel='noopener'>"
            f"<strong>{s.title}</strong></a> "
            f"<span style='color:{INK_SOFT}'>— {s.publisher} ({s.year}).</span><br>"
            f"<span style='color:{INK_SOFT}'>{s.relevance}</span>"
            f"</div>"
        )
    st.markdown(
        f"<div style='background:{PARCHMENT_DEEP};border-radius:8px;"
        f"padding:0.9rem 1.1rem;margin-top:0.4rem;'>"
        + "".join(lines) + "</div>",
        unsafe_allow_html=True,
    )
