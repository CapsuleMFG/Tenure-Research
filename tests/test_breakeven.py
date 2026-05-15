"""Tests for usda_sandbox.breakeven — feedlot breakeven calc."""

from __future__ import annotations

import pytest

from usda_sandbox.breakeven import (
    FeedlotInputs,
    compute_feedlot_economics,
    default_inputs,
)


def test_default_inputs_produce_plausible_breakeven() -> None:
    """Default KSU-style inputs should yield a 2020s-realistic breakeven."""
    econ = compute_feedlot_economics(default_inputs())
    # Late-2020s fed cattle breakevens land roughly $200-$240/cwt.
    assert 195 < econ.breakeven_per_cwt < 260
    assert econ.total_cost > 2000
    # Sanity: components add up to total cost (within float tolerance).
    parts_sum = (
        econ.feeder_cost_total
        + econ.cost_of_gain_total
        + econ.yardage_total
        + econ.interest_total
        + econ.death_loss_addon
    )
    assert abs(parts_sum - econ.total_cost) < 1e-6


def test_breakeven_increases_with_feeder_cost() -> None:
    base = default_inputs()
    higher = FeedlotInputs(
        feeder_cost_per_cwt=base.feeder_cost_per_cwt + 20.0,
        feeder_weight_lbs=base.feeder_weight_lbs,
        finished_weight_lbs=base.finished_weight_lbs,
        days_on_feed=base.days_on_feed,
        cost_of_gain_per_lb=base.cost_of_gain_per_lb,
        yardage_per_day=base.yardage_per_day,
        interest_rate_annual=base.interest_rate_annual,
        death_loss_pct=base.death_loss_pct,
    )
    e1 = compute_feedlot_economics(base)
    e2 = compute_feedlot_economics(higher)
    assert e2.breakeven_per_cwt > e1.breakeven_per_cwt


def test_breakeven_decreases_with_higher_finished_weight() -> None:
    base = default_inputs()
    heavier = FeedlotInputs(
        feeder_cost_per_cwt=base.feeder_cost_per_cwt,
        feeder_weight_lbs=base.feeder_weight_lbs,
        finished_weight_lbs=base.finished_weight_lbs + 100.0,
        days_on_feed=base.days_on_feed + 15,  # more days for more weight
        cost_of_gain_per_lb=base.cost_of_gain_per_lb,
        yardage_per_day=base.yardage_per_day,
        interest_rate_annual=base.interest_rate_annual,
        death_loss_pct=base.death_loss_pct,
    )
    e1 = compute_feedlot_economics(base)
    e2 = compute_feedlot_economics(heavier)
    # More weight to spread fixed costs over → lower breakeven per cwt.
    assert e2.breakeven_per_cwt < e1.breakeven_per_cwt


def test_invalid_inputs_raise() -> None:
    base = default_inputs()
    with pytest.raises(ValueError, match="finished_weight"):
        compute_feedlot_economics(
            FeedlotInputs(
                feeder_cost_per_cwt=base.feeder_cost_per_cwt,
                feeder_weight_lbs=800.0,
                finished_weight_lbs=800.0,  # not larger
                days_on_feed=base.days_on_feed,
                cost_of_gain_per_lb=base.cost_of_gain_per_lb,
                yardage_per_day=base.yardage_per_day,
                interest_rate_annual=base.interest_rate_annual,
                death_loss_pct=base.death_loss_pct,
            )
        )
    with pytest.raises(ValueError, match="death_loss_pct"):
        compute_feedlot_economics(
            FeedlotInputs(
                feeder_cost_per_cwt=base.feeder_cost_per_cwt,
                feeder_weight_lbs=base.feeder_weight_lbs,
                finished_weight_lbs=base.finished_weight_lbs,
                days_on_feed=base.days_on_feed,
                cost_of_gain_per_lb=base.cost_of_gain_per_lb,
                yardage_per_day=base.yardage_per_day,
                interest_rate_annual=base.interest_rate_annual,
                death_loss_pct=1.0,  # invalid
            )
        )
