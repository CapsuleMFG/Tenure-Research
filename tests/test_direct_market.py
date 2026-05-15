"""Tests for usda_sandbox.direct_market — direct-market rancher economics."""

from __future__ import annotations

import pytest

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

# --- Cow-calf -------------------------------------------------------------


def test_cow_calf_defaults_produce_plausible_output() -> None:
    """Mid-size operation, mid-2020s inputs → realistic per-cow economics."""
    econ = compute_cow_calf_economics(default_cow_calf_inputs())
    # Cost-per-cow lives roughly $900-$1300 in extension publications.
    assert 800 < econ.annual_cost_per_cow < 1400
    # Default inputs are weakly profitable.
    assert econ.margin_per_calf > 0
    # 60 cows × .88 × .94 = ~50 calves weaned
    assert 45 < econ.n_calves_weaned < 55
    # Total revenue = calves × calf_revenue
    expected_total_rev = econ.n_calves_weaned * econ.weaned_calf_revenue
    # Allow some rounding slack from int casting of calves_weaned
    assert abs(econ.total_revenue - expected_total_rev) < econ.weaned_calf_revenue


def test_cow_calf_breakeven_matches_cost_per_calf() -> None:
    inp = default_cow_calf_inputs()
    econ = compute_cow_calf_economics(inp)
    # Breakeven $/cwt × weight ÷ 100 should equal cost-per-calf.
    implied = econ.breakeven_weaned_price_per_cwt * inp.weaned_weight_lbs / 100.0
    assert abs(implied - econ.cost_per_calf) < 0.01


def test_cow_calf_invalid_rates_raise() -> None:
    base = default_cow_calf_inputs()
    with pytest.raises(ValueError, match="calving_rate"):
        compute_cow_calf_economics(
            CowCalfInputs(**{**base.__dict__, "calving_rate": 0.0})
        )
    with pytest.raises(ValueError, match="weaning_rate"):
        compute_cow_calf_economics(
            CowCalfInputs(**{**base.__dict__, "weaning_rate": 1.5})
        )
    with pytest.raises(ValueError, match="n_cows"):
        compute_cow_calf_economics(
            CowCalfInputs(**{**base.__dict__, "n_cows": 0})
        )


# --- Stocker --------------------------------------------------------------


def test_stocker_defaults_produce_plausible_output() -> None:
    econ = compute_stocker_economics(default_stocker_inputs())
    # Stocker breakeven on a 775-lb sale is in the mid-$200s/cwt range.
    assert 220 < econ.breakeven_sale_price_per_cwt < 320
    # Component sanity
    assert econ.purchase_cost_per_head > 0
    assert econ.grass_cost_per_head > 0
    assert econ.interest_cost_per_head > 0
    assert econ.death_loss_addon_per_head > 0


def test_stocker_invalid_weights_raise() -> None:
    base = default_stocker_inputs()
    with pytest.raises(ValueError, match="sale_weight"):
        compute_stocker_economics(
            StockerInputs(**{**base.__dict__, "sale_weight_lbs": base.purchase_weight_lbs})
        )
    with pytest.raises(ValueError, match="death_loss_pct"):
        compute_stocker_economics(
            StockerInputs(**{**base.__dict__, "death_loss_pct": 1.0})
        )


# --- Finish & direct ------------------------------------------------------


def test_finish_direct_defaults_show_direct_premium_over_commodity() -> None:
    econ = compute_finish_direct_economics(
        default_finish_direct_inputs(),
        live_cattle_futures_per_cwt=248.0,
    )
    # 1350 lb × 0.61 dressing ≈ 824 lb hanging
    assert 800 < econ.hanging_weight_lbs < 850
    # Default retail price ($6.50/lb hanging) yields positive margin
    assert econ.margin_per_head > 0
    # Commodity floor at LE=$248: 248/100 / 0.61 = ~$4.07/lb hanging
    assert econ.commodity_floor_per_lb_hanging is not None
    assert 3.5 < econ.commodity_floor_per_lb_hanging < 4.5
    # Direct premium = retail - commodity floor — should be meaningfully positive
    premium = 6.50 - econ.commodity_floor_per_lb_hanging
    assert premium > 1.5


def test_finish_direct_without_futures_floor_omits_commodity_field() -> None:
    econ = compute_finish_direct_economics(default_finish_direct_inputs())
    assert econ.commodity_floor_per_lb_hanging is None


def test_finish_direct_breakeven_equals_cost_per_lb_hanging() -> None:
    econ = compute_finish_direct_economics(default_finish_direct_inputs())
    assert econ.breakeven_per_lb_hanging == econ.cost_per_lb_hanging


def test_finish_direct_invalid_dressing_raises() -> None:
    base = default_finish_direct_inputs()
    with pytest.raises(ValueError, match="dressing_pct"):
        compute_finish_direct_economics(
            FinishDirectInputs(**{**base.__dict__, "dressing_pct": 0.40})
        )
    with pytest.raises(ValueError, match="dressing_pct"):
        compute_finish_direct_economics(
            FinishDirectInputs(**{**base.__dict__, "dressing_pct": 0.80})
        )


# --- Regional presets ------------------------------------------------------


def test_cow_calf_regions_cover_six_geographies() -> None:
    assert len(COW_CALF_REGIONS) == 6
    for cfg in COW_CALF_REGIONS.values():
        # Every region has the keys the builder consumes
        assert {
            "pasture_acres_per_cow", "pasture_cost_per_acre",
            "hay_tons_per_cow", "hay_cost_per_ton",
            "supplement_cost_per_cow", "vet_breeding_per_cow",
            "fixed_per_cow", "weaning_weight_lbs", "description",
        } <= cfg.keys()


def test_cow_calf_regional_costs_differ_meaningfully() -> None:
    """Southeast (year-round forage) should be cheaper than Northeast
    (long winter, expensive land). If they aren't, the presets aren't
    differentiated enough to be useful."""
    se = compute_cow_calf_economics(
        cow_calf_inputs_for_region("Southeast (TN/GA/KY/AR/FL)")
    )
    ne = compute_cow_calf_economics(
        cow_calf_inputs_for_region("Northeast (PA/NY/VT/MD/VA)")
    )
    assert se.annual_cost_per_cow < ne.annual_cost_per_cow
    # At least $400/cow gap between SE and NE — they're fundamentally
    # different cost structures.
    assert (ne.annual_cost_per_cow - se.annual_cost_per_cow) > 400


def test_cow_calf_region_builder_with_unknown_region_raises() -> None:
    with pytest.raises(ValueError, match="Unknown region"):
        cow_calf_inputs_for_region("Mars")


def test_stocker_regions_cover_six_geographies() -> None:
    assert len(STOCKER_REGIONS) == 6
    for cfg in STOCKER_REGIONS.values():
        assert {
            "days_on_grass", "pasture_cost_per_head_per_day",
            "hay_supplement_cost_per_head", "feed_supplement_cost_per_head",
            "vet_per_head", "death_loss_pct", "description",
        } <= cfg.keys()


def test_stocker_regional_pasture_cost_differs() -> None:
    """SE pasture should be cheapest per day; NE most expensive."""
    se = stocker_inputs_for_region("Southeast (TN/GA/KY fescue/bermuda)")
    ne = stocker_inputs_for_region("Northeast (PA/NY/VT/MD/VA)")
    assert se.pasture_cost_per_head_per_day < ne.pasture_cost_per_head_per_day


def test_finish_direct_regions_cover_six_geographies() -> None:
    assert len(FINISH_DIRECT_REGIONS) == 6
    for cfg in FINISH_DIRECT_REGIONS.values():
        assert {
            "grain_supplement_cost_per_head",
            "hay_supplement_cost_per_head",
            "abattoir_slaughter_fee_per_head",
            "cut_and_wrap_per_lb_hanging",
            "direct_retail_per_lb_hanging", "description",
        } <= cfg.keys()


def test_finish_direct_northeast_premium_over_midwest() -> None:
    """NE direct-retail should exceed Midwest by a meaningful amount —
    that's the NYC/Boston premium relative to flyover-metro markets."""
    mw = finish_direct_inputs_for_region("Midwest / Corn Belt (IA/MO/IL/IN)")
    ne = finish_direct_inputs_for_region("Northeast (PA/NY/VT/MA/MD)")
    assert ne.direct_retail_per_lb_hanging > mw.direct_retail_per_lb_hanging
    assert (ne.direct_retail_per_lb_hanging - mw.direct_retail_per_lb_hanging) >= 0.50


def test_finish_direct_midwest_grain_cheapest() -> None:
    """Corn Belt should be the cheapest grain region — that's its raison d'être."""
    mw = finish_direct_inputs_for_region("Midwest / Corn Belt (IA/MO/IL/IN)")
    for other_region in FINISH_DIRECT_REGIONS:
        if other_region.startswith("Midwest"):
            continue
        other = finish_direct_inputs_for_region(other_region)
        assert mw.grain_supplement_cost_per_head <= other.grain_supplement_cost_per_head, (
            f"Midwest grain ({mw.grain_supplement_cost_per_head}) not <= "
            f"{other_region} ({other.grain_supplement_cost_per_head})"
        )


def test_region_builders_propagate_user_overrides() -> None:
    """Builder kwargs (n_cows, prices) should win over preset defaults."""
    cc = cow_calf_inputs_for_region(
        "Plains (KS/NE/OK/TX panhandle)",
        n_cows=200, weaned_price_per_cwt=350.0,
    )
    assert cc.n_cows == 200
    assert cc.weaned_price_per_cwt == 350.0

    st_inputs = stocker_inputs_for_region(
        "Plains (KS/OK wheat pasture)",
        n_head=500, sale_price_per_cwt=300.0,
    )
    assert st_inputs.n_head == 500
    assert st_inputs.sale_price_per_cwt == 300.0

    fd = finish_direct_inputs_for_region(
        "Midwest / Corn Belt (IA/MO/IL/IN)",
        n_head=50, finished_live_weight_lbs=1450.0,
    )
    assert fd.n_head == 50
    assert fd.finished_live_weight_lbs == 1450.0
