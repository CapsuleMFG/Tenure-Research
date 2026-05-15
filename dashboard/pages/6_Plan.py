"""Plan — direct-market rancher's central decision tool.

Three modes (tabs):

* **Cow-calf**: maintain a breeding herd, sell weaned calves.
* **Stocker**: buy weaned calves, graze, sell to a feedlot.
* **Finish & direct**: raise/buy, finish on the farm, sell freezer beef
  direct to consumers.

Each tab shows the current relevant market context (feeder cattle for
buy/sell sides, Live Cattle for the commodity-floor sanity check on the
freezer-beef tab), takes the producer's operation parameters, and
returns a clear margin picture with annual operation P&L.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import streamlit as st
from components.scenarios import render_scenarios_panel
from components.sidebar import DEFAULT_OBS_PATH, render_sidebar
from components.theme import (
    ACCENT,
    INK_SOFT,
    PARCHMENT_DEEP,
    inject_global_css,
)

from usda_sandbox.direct_market import (
    COW_CALF_REGIONS,
    FINISH_DIRECT_REGIONS,
    STOCKER_REGIONS,
    CowCalfInputs,
    FinishDirectInputs,
    StockerInputs,
    compute_cow_calf_economics,
    compute_finish_direct_economics,
    compute_stocker_economics,
    cow_calf_inputs_for_region,
    default_cow_calf_inputs,
    default_finish_direct_inputs,
    default_stocker_inputs,
    finish_direct_inputs_for_region,
    stocker_inputs_for_region,
)
from usda_sandbox.sources import (
    COW_CALF_SOURCES,
    FINISH_DIRECT_SOURCES,
    STOCKER_SOURCES,
    Source,
)
from usda_sandbox.store import read_series


def _render_sources_panel(sources: tuple[Source, ...], header: str) -> None:
    """Render a 'Where these defaults come from' citations block."""
    lines = [
        f"<div style='font-size:0.8rem;color:{INK_SOFT};margin-bottom:0.4rem;"
        f"text-transform:uppercase;letter-spacing:0.08em;'>{header}</div>"
    ]
    for s in sources:
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
        f"padding:0.9rem 1.1rem;margin:0.8rem 0;'>"
        + "".join(lines) + "</div>",
        unsafe_allow_html=True,
    )

inject_global_css()
render_sidebar(persistent_picker=False)

st.markdown("# Plan your operation")
st.markdown(
    f"<p style='color:{INK_SOFT};font-size:1.0rem;line-height:1.5;"
    f"max-width:780px;'>"
    "Three tabs for three operation modes. Pick yours, pick your "
    "<strong>region</strong> (pre-fills typical costs), then adjust "
    "every input to match your real operation. The big margin card at "
    "the bottom of each tab is the answer — everything above it is the "
    "math that produced it."
    "</p>",
    unsafe_allow_html=True,
)

# --- Scenarios -------------------------------------------------------------
# Named save/load via browser localStorage. Each scenario stores the
# entire query string of the current Plan inputs; clicking a saved
# scenario navigates the page to that querystring and the existing
# _qp_get_* helpers repopulate every input.
with st.expander("📋 Save / load scenarios", expanded=False):
    st.caption(
        "Save the current inputs under a name. Stored in this browser's "
        "localStorage — survives page reloads and tab closes; not synced "
        "across devices. To capture the current state, click the "
        "**Copy shareable link** button at the bottom of any tab first "
        "(it writes your inputs into the URL), then come back here and "
        "name + save the scenario."
    )
    render_scenarios_panel(
        accent=ACCENT,
        parchment_deep=PARCHMENT_DEEP,
    )

obs_path = Path(DEFAULT_OBS_PATH)


# --- URL-param helpers -----------------------------------------------------
# Plan inputs survive page navigation via session_state, and survive
# bookmarks via st.query_params. On page load, query_params override
# session_state. The "Copy shareable link" buttons below each tab write
# the current input set back to st.query_params so the URL becomes
# bookmarkable.


def _qp_get_float(key: str, default: float) -> float:
    try:
        v = st.query_params.get(key)
        if v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _qp_get_int(key: str, default: int) -> int:
    try:
        v = st.query_params.get(key)
        if v is None:
            return default
        return int(v)
    except (TypeError, ValueError):
        return default


def _qp_get_str(key: str, default: str) -> str:
    v = st.query_params.get(key)
    if v is None:
        return default
    return str(v)


def _share_link_button(prefix: str, params: dict[str, object]) -> None:
    """Write a 'Copy shareable link' button that, on click, stuffs the
    current input set into ``st.query_params`` so the page URL becomes
    a bookmarkable snapshot of the form state."""
    if st.button("🔗 Copy shareable link (save my inputs in the URL)",
                 key=f"share_{prefix}",
                 use_container_width=True,
                 type="tertiary"):
        for k, v in params.items():
            if v is None:
                continue
            st.query_params[k] = str(v)
        st.toast(
            "URL updated — bookmark or share this page to keep these "
            "inputs.", icon="🔗"
        )


def _latest_value(series_id: str) -> tuple[float | None, str | None]:
    """Return (latest non-null value, ISO date) for a series."""
    try:
        df = (
            read_series(series_id, obs_path)
            .filter(pl.col("value").is_not_null())
            .sort("period_start")
        )
    except Exception:
        return None, None
    if df.is_empty():
        return None, None
    row = df.tail(1).row(0, named=True)
    return float(row["value"]), str(row["period_start"])


# Pull current market signals once for use in all three tabs.
le_today, le_date = _latest_value("cattle_lc_front_daily")
gf_today, gf_date = _latest_value("cattle_feeder_front_daily")
corn_today, corn_date = _latest_value("corn_front_daily")
sbm_today, _ = _latest_value("soybean_meal_front_daily")
fed_750, _ = _latest_value("cattle_feeder_steer_750_800")
fed_500, _ = _latest_value("cattle_feeder_steer_500_550")


def _market_chip(label: str, value: str, sub: str = "") -> str:
    return (
        f"<div style='display:inline-block;background:{PARCHMENT_DEEP};"
        f"border-radius:6px;padding:0.4rem 0.7rem;margin-right:0.5rem;"
        f"margin-bottom:0.3rem;'>"
        f"<div style='font-size:0.7rem;color:{INK_SOFT};letter-spacing:0.06em;"
        f"text-transform:uppercase;'>{label}</div>"
        f"<div style='font-size:1.0rem;font-weight:600;'>{value}</div>"
        + (f"<div style='font-size:0.72rem;color:{INK_SOFT};'>{sub}</div>" if sub else "")
        + "</div>"
    )


tab_cc, tab_st, tab_fd = st.tabs([
    "Cow-calf",
    "Stocker",
    "Finish & direct (freezer beef)",
])

# --- Cow-calf ---------------------------------------------------------------

with tab_cc:
    st.markdown(
        "<div style='background:#F0EBE1;border-radius:8px;"
        "padding:0.85rem 1.1rem;margin-bottom:1rem;font-size:0.95rem;"
        "line-height:1.5;'>"
        "<strong>Cow-calf operation.</strong> You maintain a breeding "
        "herd and sell weaned calves (~500–600 lb) into the feeder "
        "market each fall. This tab computes your annual per-cow cost, "
        "your per-calf cost, your margin at today's feeder prices, and "
        "the breakeven $/cwt you need to clear at weaning."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("### Today's relevant market")
    chips = []
    if fed_500 is not None:
        chips.append(_market_chip("500-550 lb feeder", f"${fed_500:,.2f}/cwt",
                                   "OK auction, monthly"))
    if gf_today is not None:
        chips.append(_market_chip("Feeder front-month (GF)", f"${gf_today:,.2f}/cwt",
                                   f"daily, {gf_date}"))
    if chips:
        st.markdown("".join(chips), unsafe_allow_html=True)

    st.caption(
        "Weaned calf price drives your cow-calf revenue. If today's feeder "
        "market is strong, fall calf sale revenue should benefit; if weak, "
        "consider retained ownership or stocker through-put."
    )

    # Region preset picker — pre-fills the cost inputs for the chosen
    # geography. Defaults to the baseline if URL param ?cc_region= is unset.
    region_options = list(COW_CALF_REGIONS.keys())
    default_region = _qp_get_str("cc_region", region_options[0])
    if default_region not in region_options:
        default_region = region_options[0]
    region = st.selectbox(
        "Pick your region (pre-fills typical costs)",
        options=region_options,
        index=region_options.index(default_region),
        key="cc_region",
        help=(
            "Each preset bundles representative pasture/hay/vet costs "
            "for that geography. You can tweak any field after picking."
        ),
    )
    st.caption(COW_CALF_REGIONS[region]["description"])

    # If the user just switched regions, OR if there's no cached session,
    # rebuild the inputs from the region preset.
    prior_region = st.session_state.get("cc_region_prior")
    region_changed = prior_region != region
    st.session_state["cc_region_prior"] = region

    if region_changed or "cc_inputs" not in st.session_state:
        cc_session = cow_calf_inputs_for_region(region)
    else:
        cc_session = st.session_state["cc_inputs"]

    # URL-param overrides take precedence over the regional default.
    cc_session = CowCalfInputs(
        n_cows=_qp_get_int("cc_n_cows", cc_session.n_cows),
        calving_rate=_qp_get_float("cc_calving", cc_session.calving_rate),
        weaning_rate=_qp_get_float("cc_weaning", cc_session.weaning_rate),
        weaned_weight_lbs=_qp_get_float("cc_ww", cc_session.weaned_weight_lbs),
        weaned_price_per_cwt=_qp_get_float("cc_wp", cc_session.weaned_price_per_cwt),
        pasture_acres_per_cow=_qp_get_float("cc_acres", cc_session.pasture_acres_per_cow),
        pasture_cost_per_acre=_qp_get_float("cc_acre_cost", cc_session.pasture_cost_per_acre),
        hay_tons_per_cow=_qp_get_float("cc_hay_t", cc_session.hay_tons_per_cow),
        hay_cost_per_ton=_qp_get_float("cc_hay_c", cc_session.hay_cost_per_ton),
        supplement_cost_per_cow=_qp_get_float("cc_supp", cc_session.supplement_cost_per_cow),
        vet_breeding_per_cow=_qp_get_float("cc_vet", cc_session.vet_breeding_per_cow),
        fixed_per_cow=_qp_get_float("cc_fixed", cc_session.fixed_per_cow),
        bull_pct=_qp_get_float("cc_bullp", cc_session.bull_pct),
        bull_annual_cost=_qp_get_float("cc_bullc", cc_session.bull_annual_cost),
    )

    _ = default_cow_calf_inputs()  # referenced for legacy default; unused

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Herd**")
        n_cows = st.number_input("Number of cows", 1, 2000,
                                 value=cc_session.n_cows, step=5, key="cc_n")
        calving = st.number_input("Calving rate (%)", 50.0, 100.0,
                                  value=cc_session.calving_rate * 100, step=1.0,
                                  key="cc_calving") / 100.0
        weaning = st.number_input("Weaning rate (%)", 50.0, 100.0,
                                  value=cc_session.weaning_rate * 100, step=1.0,
                                  key="cc_weaning") / 100.0
        ww = st.number_input("Weaning weight (lbs)", 300.0, 800.0,
                             value=cc_session.weaned_weight_lbs, step=10.0,
                             key="cc_ww")
        wp = st.number_input(
            "Weaned calf price ($/cwt)", 100.0, 600.0,
            value=cc_session.weaned_price_per_cwt, step=5.0,
            help="Auction price for your typical weight class.", key="cc_wp",
        )
    with c2:
        st.markdown("**Pasture & feed**")
        acres = st.number_input("Pasture acres / cow", 0.5, 50.0,
                                value=cc_session.pasture_acres_per_cow,
                                step=0.5, key="cc_acres")
        acre_cost = st.number_input("Pasture cost ($/acre/year)", 0.0, 500.0,
                                    value=cc_session.pasture_cost_per_acre,
                                    step=5.0, key="cc_acre_cost")
        hay_tons = st.number_input("Hay tons / cow / year", 0.0, 8.0,
                                   value=cc_session.hay_tons_per_cow,
                                   step=0.1, key="cc_hay_t")
        hay_cost = st.number_input("Hay $/ton", 0.0, 600.0,
                                   value=cc_session.hay_cost_per_ton,
                                   step=5.0, key="cc_hay_c")
        supp = st.number_input("Supplement $/cow/year", 0.0, 500.0,
                               value=cc_session.supplement_cost_per_cow,
                               step=5.0, key="cc_supp")
    with c3:
        st.markdown("**Other costs**")
        vet = st.number_input("Vet/breeding $/cow", 0.0, 400.0,
                              value=cc_session.vet_breeding_per_cow,
                              step=5.0, key="cc_vet")
        fixed = st.number_input("Fence/labor/fuel $/cow", 0.0, 600.0,
                                value=cc_session.fixed_per_cow,
                                step=5.0, key="cc_fixed")
        bull_pct = st.number_input("Bulls per cow", 0.0, 0.20,
                                   value=cc_session.bull_pct, step=0.01,
                                   key="cc_bullp")
        bull_cost = st.number_input("Bull annual cost ($/head)", 0.0, 8000.0,
                                    value=cc_session.bull_annual_cost,
                                    step=50.0, key="cc_bullc")

    inputs = CowCalfInputs(
        n_cows=n_cows, calving_rate=calving, weaning_rate=weaning,
        weaned_weight_lbs=ww, weaned_price_per_cwt=wp,
        pasture_acres_per_cow=acres, pasture_cost_per_acre=acre_cost,
        hay_tons_per_cow=hay_tons, hay_cost_per_ton=hay_cost,
        supplement_cost_per_cow=supp, vet_breeding_per_cow=vet,
        fixed_per_cow=fixed, bull_pct=bull_pct, bull_annual_cost=bull_cost,
    )
    try:
        econ = compute_cow_calf_economics(inputs)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    st.session_state["cc_inputs"] = inputs

    margin_color = "#3E7D5A" if econ.margin_per_calf >= 0 else "#B4521E"
    st.markdown(
        f"<div style='border:2px solid {margin_color};border-radius:10px;"
        f"padding:1rem 1.3rem;margin:1rem 0;'>"
        f"<div style='font-size:0.78rem;color:{INK_SOFT};letter-spacing:0.1em;"
        f"text-transform:uppercase;'>Per-calf margin</div>"
        f"<div style='font-size:2.0rem;font-weight:600;color:{margin_color};"
        f"font-family:Iowan Old Style, Source Serif Pro, Georgia, serif;'>"
        f"${econ.margin_per_calf:+,.0f}/calf</div>"
        f"<div style='font-size:0.95rem;'>Total operation: "
        f"<strong>${econ.total_margin:+,.0f}/year</strong> "
        f"on {econ.n_calves_weaned} weaned calves.</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("Cost per cow / year", f"${econ.annual_cost_per_cow:,.0f}")
    m2.metric("Cost per calf weaned", f"${econ.cost_per_calf:,.0f}")
    m3.metric("Breakeven weaned price",
              f"${econ.breakeven_weaned_price_per_cwt:,.2f}/cwt")
    st.caption(
        f"Breakeven on weaned calves: **${econ.breakeven_weaned_price_per_cwt:,.2f}/cwt**. "
        f"You're pricing the market at ${inputs.weaned_price_per_cwt:,.2f}/cwt, "
        f"so margin per cwt of weaned calf is "
        f"**${inputs.weaned_price_per_cwt - econ.breakeven_weaned_price_per_cwt:+,.2f}/cwt**."
    )

    _share_link_button("cc", {
        "cc_region": region,
        "cc_n_cows": n_cows,
        "cc_calving": calving,
        "cc_weaning": weaning,
        "cc_ww": ww,
        "cc_wp": wp,
        "cc_acres": acres,
        "cc_acre_cost": acre_cost,
        "cc_hay_t": hay_tons,
        "cc_hay_c": hay_cost,
        "cc_supp": supp,
        "cc_vet": vet,
        "cc_fixed": fixed,
        "cc_bullp": bull_pct,
        "cc_bullc": bull_cost,
    })

    with st.expander("Where these defaults come from"):
        _render_sources_panel(COW_CALF_SOURCES, "Cow-calf sources")

# --- Stocker ----------------------------------------------------------------

with tab_st:
    st.markdown(
        "<div style='background:#F0EBE1;border-radius:8px;"
        "padding:0.85rem 1.1rem;margin-bottom:1rem;font-size:0.95rem;"
        "line-height:1.5;'>"
        "<strong>Stocker operation.</strong> You buy weaned calves "
        "(~450–550 lb), graze them up to feedlot placement weight "
        "(~700–800 lb), and sell to a feedlot. The margin lives in the "
        "gap between buy price and sell price, net of grass, hay, "
        "supplement, vet, death loss, and interest on the placement."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("### Today's relevant market")
    chips = []
    if fed_500 is not None:
        chips.append(_market_chip("500-550 lb (buy)", f"${fed_500:,.2f}/cwt"))
    if fed_750 is not None:
        chips.append(_market_chip("750-800 lb (sell)", f"${fed_750:,.2f}/cwt"))
    if gf_today is not None:
        chips.append(_market_chip("GF front-month", f"${gf_today:,.2f}/cwt",
                                   f"daily, {gf_date}"))
    if chips:
        st.markdown("".join(chips), unsafe_allow_html=True)

    st.caption(
        "Stocker margin lives in the gap between the buy price (light "
        "calves) and the sell price (placement-weight feeders) net of "
        "grass + supplement + interest + death loss. Watch the futures "
        "(GF) for forward signals on where placement prices are headed."
    )

    # Region picker first — pre-fills the cost stack.
    st_region_options = list(STOCKER_REGIONS.keys())
    default_st_region = _qp_get_str("st_region", st_region_options[0])
    if default_st_region not in st_region_options:
        default_st_region = st_region_options[0]
    st_region = st.selectbox(
        "Pick your region (pre-fills typical costs)",
        options=st_region_options,
        index=st_region_options.index(default_st_region),
        key="st_region",
    )
    st.caption(STOCKER_REGIONS[st_region]["description"])

    prior_st_region = st.session_state.get("st_region_prior")
    st_region_changed = prior_st_region != st_region
    st.session_state["st_region_prior"] = st_region

    if st_region_changed or "st_inputs" not in st.session_state:
        sd_session = stocker_inputs_for_region(st_region)
    else:
        sd_session = st.session_state["st_inputs"]

    sd_session = StockerInputs(
        n_head=_qp_get_int("st_n", sd_session.n_head),
        purchase_weight_lbs=_qp_get_float("st_pw", sd_session.purchase_weight_lbs),
        purchase_price_per_cwt=_qp_get_float("st_pp", sd_session.purchase_price_per_cwt),
        sale_weight_lbs=_qp_get_float("st_sw", sd_session.sale_weight_lbs),
        sale_price_per_cwt=_qp_get_float("st_sp", sd_session.sale_price_per_cwt),
        days_on_grass=_qp_get_int("st_days", sd_session.days_on_grass),
        pasture_cost_per_head_per_day=_qp_get_float(
            "st_pday", sd_session.pasture_cost_per_head_per_day),
        hay_supplement_cost_per_head=_qp_get_float(
            "st_hay", sd_session.hay_supplement_cost_per_head),
        feed_supplement_cost_per_head=_qp_get_float(
            "st_feed", sd_session.feed_supplement_cost_per_head),
        vet_per_head=_qp_get_float("st_vet", sd_session.vet_per_head),
        death_loss_pct=_qp_get_float("st_dl", sd_session.death_loss_pct),
        interest_rate_annual=_qp_get_float("st_ir", sd_session.interest_rate_annual),
    )
    _ = default_stocker_inputs()  # referenced for legacy; unused
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Animals**")
        n_head = st.number_input("Head placed", 1, 5000,
                                 value=sd_session.n_head, step=10, key="st_n")
        pw = st.number_input("Purchase weight (lbs)", 300.0, 800.0,
                             value=sd_session.purchase_weight_lbs, step=10.0,
                             key="st_pw")
        pp = st.number_input("Purchase price ($/cwt)", 100.0, 600.0,
                             value=sd_session.purchase_price_per_cwt, step=5.0,
                             key="st_pp")
        sw = st.number_input("Sale weight (lbs)", 500.0, 1100.0,
                             value=sd_session.sale_weight_lbs, step=10.0,
                             key="st_sw")
        sp = st.number_input("Sale price ($/cwt)", 100.0, 600.0,
                             value=sd_session.sale_price_per_cwt, step=5.0,
                             key="st_sp")
    with c2:
        st.markdown("**Grass + supplement**")
        days = st.number_input("Days on grass", 30, 365,
                               value=sd_session.days_on_grass, step=10,
                               key="st_days")
        pday = st.number_input("Pasture $/head/day", 0.0, 3.0,
                               value=sd_session.pasture_cost_per_head_per_day,
                               step=0.05, key="st_pday")
        hay = st.number_input("Hay supplement $/head", 0.0, 300.0,
                              value=sd_session.hay_supplement_cost_per_head,
                              step=5.0, key="st_hay")
        feed = st.number_input("Feed supplement $/head", 0.0, 300.0,
                               value=sd_session.feed_supplement_cost_per_head,
                               step=5.0, key="st_feed")
    with c3:
        st.markdown("**Other**")
        vet = st.number_input("Vet/processing $/head", 0.0, 200.0,
                              value=sd_session.vet_per_head, step=5.0,
                              key="st_vet")
        dl = st.number_input("Death loss (%)", 0.0, 10.0,
                             value=sd_session.death_loss_pct * 100,
                             step=0.1, key="st_dl") / 100.0
        ir = st.number_input("Interest rate (%/yr)", 0.0, 20.0,
                             value=sd_session.interest_rate_annual * 100,
                             step=0.25, key="st_ir") / 100.0

    inputs = StockerInputs(
        n_head=n_head, purchase_weight_lbs=pw, purchase_price_per_cwt=pp,
        sale_weight_lbs=sw, sale_price_per_cwt=sp, days_on_grass=days,
        pasture_cost_per_head_per_day=pday, hay_supplement_cost_per_head=hay,
        feed_supplement_cost_per_head=feed, vet_per_head=vet,
        death_loss_pct=dl, interest_rate_annual=ir,
    )
    try:
        econ = compute_stocker_economics(inputs)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    st.session_state["st_inputs"] = inputs

    margin_color = "#3E7D5A" if econ.margin_per_head >= 0 else "#B4521E"
    st.markdown(
        f"<div style='border:2px solid {margin_color};border-radius:10px;"
        f"padding:1rem 1.3rem;margin:1rem 0;'>"
        f"<div style='font-size:0.78rem;color:{INK_SOFT};letter-spacing:0.1em;"
        f"text-transform:uppercase;'>Per-head margin</div>"
        f"<div style='font-size:2.0rem;font-weight:600;color:{margin_color};"
        f"font-family:Iowan Old Style, Source Serif Pro, Georgia, serif;'>"
        f"${econ.margin_per_head:+,.0f}/head</div>"
        f"<div style='font-size:0.95rem;'>Total cycle: "
        f"<strong>${econ.total_margin:+,.0f}</strong> on {inputs.n_head} head.</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("Total cost/head", f"${econ.total_cost_per_head:,.0f}")
    m2.metric("Sale revenue/head", f"${econ.sale_revenue_per_head:,.0f}")
    m3.metric("Breakeven sale price",
              f"${econ.breakeven_sale_price_per_cwt:,.2f}/cwt")

    _share_link_button("st", {
        "st_region": st_region,
        "st_n": n_head,
        "st_pw": pw,
        "st_pp": pp,
        "st_sw": sw,
        "st_sp": sp,
        "st_days": days,
        "st_pday": pday,
        "st_hay": hay,
        "st_feed": feed,
        "st_vet": vet,
        "st_dl": dl,
        "st_ir": ir,
    })

    with st.expander("Where these defaults come from"):
        _render_sources_panel(STOCKER_SOURCES, "Stocker sources")

# --- Finish & direct (freezer beef) ----------------------------------------

with tab_fd:
    st.markdown(
        "<div style='background:#F0EBE1;border-radius:8px;"
        "padding:0.85rem 1.1rem;margin-bottom:1rem;font-size:0.95rem;"
        "line-height:1.5;'>"
        "<strong>Finish-and-direct (freezer beef).</strong> You raise "
        "or buy feeders, finish them on the farm, and sell freezer beef "
        "directly to consumers as quarters / halves / wholes. This tab "
        "shows your cost per pound of hanging weight, your retail "
        "margin, and a <em>commodity-floor sanity check</em>: what the "
        "same animal would have fetched at today's Live Cattle futures, "
        "so you can see your direct-market premium explicitly."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("### Today's relevant market")
    chips = []
    if le_today is not None:
        chips.append(_market_chip("Live Cattle (LE)", f"${le_today:,.2f}/cwt",
                                   f"daily, {le_date}"))
    if corn_today is not None:
        # ZC is reported in cents/bushel; convert to $/bu for display
        chips.append(_market_chip("Corn (ZC)", f"${corn_today/100:.2f}/bu",
                                   f"daily, {corn_date}"))
    if sbm_today is not None:
        chips.append(_market_chip("Soybean meal (ZM)", f"${sbm_today:,.0f}/short ton"))
    if chips:
        st.markdown("".join(chips), unsafe_allow_html=True)

    st.caption(
        "Live Cattle futures set the *commodity floor* — what the same "
        "animal would fetch sold into the commodity market. Your direct-"
        "to-consumer retail price needs to clear that floor with margin "
        "to spare (because direct selling costs time, marketing, and "
        "abattoir slots that commodity selling doesn't). Corn and SBM "
        "drive your supplemental finishing feed cost."
    )

    # Region picker first — pre-fills retail + abattoir + cut-and-wrap.
    fd_region_options = list(FINISH_DIRECT_REGIONS.keys())
    default_fd_region = _qp_get_str("fd_region", fd_region_options[0])
    if default_fd_region not in fd_region_options:
        default_fd_region = fd_region_options[0]
    fd_region = st.selectbox(
        "Pick your region (pre-fills typical retail + processing costs)",
        options=fd_region_options,
        index=fd_region_options.index(default_fd_region),
        key="fd_region",
    )
    st.caption(FINISH_DIRECT_REGIONS[fd_region]["description"])

    prior_fd_region = st.session_state.get("fd_region_prior")
    fd_region_changed = prior_fd_region != fd_region
    st.session_state["fd_region_prior"] = fd_region

    if fd_region_changed or "fd_inputs" not in st.session_state:
        fd_session = finish_direct_inputs_for_region(fd_region)
    else:
        fd_session = st.session_state["fd_inputs"]

    fd_session = FinishDirectInputs(
        n_head=_qp_get_int("fd_n", fd_session.n_head),
        feeder_cost_per_head=_qp_get_float("fd_fc", fd_session.feeder_cost_per_head),
        days_on_farm=_qp_get_int("fd_days", fd_session.days_on_farm),
        finished_live_weight_lbs=_qp_get_float(
            "fd_flw", fd_session.finished_live_weight_lbs),
        dressing_pct=_qp_get_float("fd_dp", fd_session.dressing_pct),
        pasture_cost_per_head_per_day=_qp_get_float(
            "fd_pday", fd_session.pasture_cost_per_head_per_day),
        hay_supplement_cost_per_head=_qp_get_float(
            "fd_hay", fd_session.hay_supplement_cost_per_head),
        grain_supplement_cost_per_head=_qp_get_float(
            "fd_grain", fd_session.grain_supplement_cost_per_head),
        vet_per_head=_qp_get_float("fd_vet", fd_session.vet_per_head),
        death_loss_pct=_qp_get_float("fd_dl", fd_session.death_loss_pct),
        abattoir_slaughter_fee_per_head=_qp_get_float(
            "fd_slt", fd_session.abattoir_slaughter_fee_per_head),
        cut_and_wrap_per_lb_hanging=_qp_get_float(
            "fd_cw", fd_session.cut_and_wrap_per_lb_hanging),
        other_per_head=_qp_get_float("fd_other", fd_session.other_per_head),
        direct_retail_per_lb_hanging=_qp_get_float(
            "fd_retail", fd_session.direct_retail_per_lb_hanging),
    )
    _ = default_finish_direct_inputs()  # referenced for legacy; unused

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Animal**")
        n_head = st.number_input("Head finished", 1, 500,
                                 value=fd_session.n_head, step=1, key="fd_n")
        fcost = st.number_input("Feeder cost ($/head)", 200.0, 4000.0,
                                value=fd_session.feeder_cost_per_head,
                                step=25.0, key="fd_fc",
                                help="What you paid (or what the on-farm calf cost to raise).")
        days = st.number_input("Days on farm (finish)", 60, 500,
                               value=fd_session.days_on_farm, step=10, key="fd_days")
        flw = st.number_input("Finished live weight (lbs)", 900.0, 1700.0,
                              value=fd_session.finished_live_weight_lbs,
                              step=10.0, key="fd_flw")
        dp = st.number_input("Dressing % (hanging / live)", 0.55, 0.68,
                             value=fd_session.dressing_pct, step=0.005,
                             key="fd_dp", format="%.3f")
    with c2:
        st.markdown("**Feed + care**")
        pday = st.number_input("Pasture $/head/day", 0.0, 3.0,
                               value=fd_session.pasture_cost_per_head_per_day,
                               step=0.05, key="fd_pday")
        hay = st.number_input("Hay supplement $/head", 0.0, 600.0,
                              value=fd_session.hay_supplement_cost_per_head,
                              step=10.0, key="fd_hay")
        grain = st.number_input("Grain supplement $/head", 0.0, 1500.0,
                                value=fd_session.grain_supplement_cost_per_head,
                                step=10.0, key="fd_grain")
        vet = st.number_input("Vet/health $/head", 0.0, 300.0,
                              value=fd_session.vet_per_head, step=5.0,
                              key="fd_vet")
        dl = st.number_input("Death loss (%)", 0.0, 5.0,
                             value=fd_session.death_loss_pct * 100, step=0.1,
                             key="fd_dl") / 100.0
    with c3:
        st.markdown("**Slaughter + sale**")
        slt = st.number_input("Abattoir slaughter fee ($/head)", 0.0, 500.0,
                              value=fd_session.abattoir_slaughter_fee_per_head,
                              step=5.0, key="fd_slt")
        cw = st.number_input("Cut-and-wrap ($/lb hanging)", 0.0, 3.0,
                             value=fd_session.cut_and_wrap_per_lb_hanging,
                             step=0.05, key="fd_cw")
        other = st.number_input("Other $/head (marketing, storage, etc.)",
                                0.0, 500.0, value=fd_session.other_per_head,
                                step=5.0, key="fd_other")
        retail = st.number_input("Your direct retail ($/lb hanging)",
                                 0.0, 25.0,
                                 value=fd_session.direct_retail_per_lb_hanging,
                                 step=0.10, key="fd_retail",
                                 help=("What you charge per pound of hanging "
                                       "weight (typical: $5.50-$10.00)."))

    inputs = FinishDirectInputs(
        n_head=n_head, feeder_cost_per_head=fcost, days_on_farm=days,
        finished_live_weight_lbs=flw, dressing_pct=dp,
        pasture_cost_per_head_per_day=pday, hay_supplement_cost_per_head=hay,
        grain_supplement_cost_per_head=grain, vet_per_head=vet,
        death_loss_pct=dl, abattoir_slaughter_fee_per_head=slt,
        cut_and_wrap_per_lb_hanging=cw, other_per_head=other,
        direct_retail_per_lb_hanging=retail,
    )
    try:
        econ = compute_finish_direct_economics(
            inputs, live_cattle_futures_per_cwt=le_today
        )
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    st.session_state["fd_inputs"] = inputs
    st.session_state["fd_hanging_lbs"] = econ.hanging_weight_lbs

    margin_color = "#3E7D5A" if econ.margin_per_head >= 0 else "#B4521E"
    st.markdown(
        f"<div style='border:2px solid {margin_color};border-radius:10px;"
        f"padding:1rem 1.3rem;margin:1rem 0;'>"
        f"<div style='font-size:0.78rem;color:{INK_SOFT};letter-spacing:0.1em;"
        f"text-transform:uppercase;'>Per-head margin</div>"
        f"<div style='font-size:2.0rem;font-weight:600;color:{margin_color};"
        f"font-family:Iowan Old Style, Source Serif Pro, Georgia, serif;'>"
        f"${econ.margin_per_head:+,.0f}/head</div>"
        f"<div style='font-size:0.95rem;'>Total cycle: "
        f"<strong>${econ.total_margin:+,.0f}</strong> on {inputs.n_head} head."
        f"</div></div>",
        unsafe_allow_html=True,
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Hanging weight/head", f"{econ.hanging_weight_lbs:.0f} lbs")
    m2.metric("Cost/head", f"${econ.total_cost_per_head:,.0f}")
    m3.metric("Breakeven hanging price",
              f"${econ.breakeven_per_lb_hanging:,.2f}/lb")
    m4.metric("Your retail hanging price",
              f"${inputs.direct_retail_per_lb_hanging:,.2f}/lb")

    if econ.commodity_floor_per_lb_hanging is not None:
        premium = inputs.direct_retail_per_lb_hanging - econ.commodity_floor_per_lb_hanging
        premium_color = ACCENT if premium > 0 else "#B4521E"
        st.markdown(
            f"<div style='background:{PARCHMENT_DEEP};border-radius:8px;"
            f"padding:0.9rem 1.1rem;margin:0.6rem 0;font-size:0.95rem;'>"
            f"<strong>Direct-market premium check.</strong> Live Cattle "
            f"futures at ${le_today:,.2f}/cwt translate to a commodity floor "
            f"of <strong>${econ.commodity_floor_per_lb_hanging:,.2f}/lb hanging</strong>. "
            f"You're charging <strong>${inputs.direct_retail_per_lb_hanging:,.2f}/lb</strong>, "
            f"<span style='color:{premium_color};font-weight:600;'>"
            f"${premium:+,.2f}/lb above the commodity floor</span>. "
            f"That premium has to cover your marketing, customer service, "
            f"and the time direct-selling takes — make sure it does."
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        f"<p style='color:{INK_SOFT};font-size:0.85rem;margin-top:1rem;'>"
        f"Head to <strong>Pricing</strong> to see how share-size offerings "
        f"price out (quarter / half / whole), or to <strong>Costs</strong> "
        f"to track your input costs over time."
        f"</p>",
        unsafe_allow_html=True,
    )

    _share_link_button("fd", {
        "fd_region": fd_region,
        "fd_n": n_head,
        "fd_fc": fcost,
        "fd_days": days,
        "fd_flw": flw,
        "fd_dp": dp,
        "fd_pday": pday,
        "fd_hay": hay,
        "fd_grain": grain,
        "fd_vet": vet,
        "fd_dl": dl,
        "fd_slt": slt,
        "fd_cw": cw,
        "fd_other": other,
        "fd_retail": retail,
    })

    with st.expander("Where these defaults come from"):
        _render_sources_panel(FINISH_DIRECT_SOURCES, "Finish & direct sources")

st.markdown("---")
st.caption(
    "Not financial advice. Inputs are KSU/ISU/UTK extension-survey "
    "midpoints for the late 2020s; replace them with your real numbers. "
    "Pasture and hay costs vary wildly by region — most operations are "
    "off the default by 20-40% in at least one line."
)
