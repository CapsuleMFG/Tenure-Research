"""Cattle-feeding breakeven economics.

Closed-form math for the breakeven $/cwt a fed-cattle operator must clear
on finished animals to recoup feeder purchase, cost of gain, yardage,
interest, and death loss. Numbers come from Kansas State / Iowa State
extension feedlot cost-of-production averages.

Pure functions; no Streamlit, no I/O. The dashboard's Breakeven page
imports these and feeds the Decide page.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "FeedlotEconomics",
    "FeedlotInputs",
    "compute_feedlot_economics",
    "default_inputs",
]


@dataclass(frozen=True)
class FeedlotInputs:
    """One feedlot placement's economic inputs.

    All monetary inputs are USD, all weights in pounds.

    Attributes
    ----------
    feeder_cost_per_cwt : float
        Price paid for the feeder calf (USD per hundredweight).
    feeder_weight_lbs : float
        Placement weight (typical: 750 lbs).
    finished_weight_lbs : float
        Target slaughter weight (typical: 1400 lbs).
    days_on_feed : int
        Time in the lot (typical: 165).
    cost_of_gain_per_lb : float
        Feed + medicine + other variable per lb of gain (typical: $1.20).
    yardage_per_day : float
        Daily fixed cost (pen rental, labor, utilities) per head (typical: $0.55).
    interest_rate_annual : float
        Annual interest rate on the placement (typical: 0.08 = 8%).
    death_loss_pct : float
        Fraction of placements that die (typical: 0.01 = 1%).
    """

    feeder_cost_per_cwt: float
    feeder_weight_lbs: float
    finished_weight_lbs: float
    days_on_feed: int
    cost_of_gain_per_lb: float
    yardage_per_day: float
    interest_rate_annual: float
    death_loss_pct: float


@dataclass(frozen=True)
class FeedlotEconomics:
    """Closed-form output for one placement."""

    feeder_cost_total: float
    cost_of_gain_total: float
    yardage_total: float
    interest_total: float
    death_loss_addon: float
    total_cost: float
    breakeven_per_cwt: float


def default_inputs() -> FeedlotInputs:
    """KSU-ish averages for late-2020s fed cattle."""
    return FeedlotInputs(
        feeder_cost_per_cwt=270.0,
        feeder_weight_lbs=750.0,
        finished_weight_lbs=1400.0,
        days_on_feed=165,
        cost_of_gain_per_lb=1.20,
        yardage_per_day=0.55,
        interest_rate_annual=0.08,
        death_loss_pct=0.01,
    )


def compute_feedlot_economics(inputs: FeedlotInputs) -> FeedlotEconomics:
    """Compute the breakeven $/cwt from one set of placement assumptions.

    Cost components (per head, before death-loss spread):

    * **Feeder cost** = feeder_cost_per_cwt × feeder_weight_lbs / 100
    * **Cost of gain** = (finished − placement weight) × cost_of_gain_per_lb
    * **Yardage** = yardage_per_day × days_on_feed
    * **Interest** = feeder_cost × interest_rate_annual × days_on_feed / 365
      (the simplest standard: interest accrues only on the feeder cost,
      not on accumulating gain; OK for a rough breakeven.)

    Death loss is spread over the survivors by multiplying the per-head
    cost by 1 / (1 − death_loss_pct). The breakeven sale price per cwt is
    then total_cost / finished_weight_lbs × 100.
    """
    if inputs.finished_weight_lbs <= inputs.feeder_weight_lbs:
        raise ValueError("finished_weight_lbs must exceed feeder_weight_lbs")
    if not 0.0 <= inputs.death_loss_pct < 1.0:
        raise ValueError("death_loss_pct must be in [0, 1)")

    feeder_cost = inputs.feeder_cost_per_cwt * inputs.feeder_weight_lbs / 100.0
    weight_gain = inputs.finished_weight_lbs - inputs.feeder_weight_lbs
    cog_total = weight_gain * inputs.cost_of_gain_per_lb
    yardage_total = inputs.yardage_per_day * inputs.days_on_feed
    interest_total = (
        feeder_cost * inputs.interest_rate_annual * inputs.days_on_feed / 365.0
    )

    pre_loss_cost = feeder_cost + cog_total + yardage_total + interest_total
    # Spread surviving-head cost over the (1 - death_loss) survivors.
    death_loss_addon = pre_loss_cost * (
        1.0 / (1.0 - inputs.death_loss_pct) - 1.0
    )
    total_cost = pre_loss_cost + death_loss_addon
    breakeven = total_cost / inputs.finished_weight_lbs * 100.0

    return FeedlotEconomics(
        feeder_cost_total=feeder_cost,
        cost_of_gain_total=cog_total,
        yardage_total=yardage_total,
        interest_total=interest_total,
        death_loss_addon=death_loss_addon,
        total_cost=total_cost,
        breakeven_per_cwt=breakeven,
    )
