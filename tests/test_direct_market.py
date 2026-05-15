"""Tests for usda_sandbox.direct_market — direct-market rancher economics."""

from __future__ import annotations

import pytest

from usda_sandbox.direct_market import (
    CowCalfInputs,
    FinishDirectInputs,
    StockerInputs,
    compute_cow_calf_economics,
    compute_finish_direct_economics,
    compute_stocker_economics,
    default_cow_calf_inputs,
    default_finish_direct_inputs,
    default_stocker_inputs,
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
