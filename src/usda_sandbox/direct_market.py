"""Direct-market rancher economics.

Three modes, three cost structures, three pure-function calculators.
All math here; the Streamlit Plan page just supplies the inputs and
formats the outputs.

* **Cow-calf**: maintain a breeding herd; sell weaned calves into the
  feeder market (or retain for finishing). Costs are per cow per year;
  output is per-calf breakeven plus annual operation P&L.
* **Stocker**: buy weaned calves, graze them up to feedlot placement
  weight (650-850 lbs), sell to a feedlot. Margin between buy and sell.
* **Finish-and-direct**: raise / buy feeders, finish them on the farm,
  sell freezer beef directly to consumers. The signature direct-market
  mode for a farm with on-site or custom-exempt slaughter access.

Unlike the v2.0 feedlot calculator (commodity feed × days on feed ×
yardage), this module's cost stack reflects what a pasture-based or
small-scale finishing operation actually pays for: acres of pasture,
tons of hay, vet/breeding, fence, labor, fuel, cut-and-wrap fees.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "CowCalfEconomics",
    "CowCalfInputs",
    "FinishDirectEconomics",
    "FinishDirectInputs",
    "StockerEconomics",
    "StockerInputs",
    "compute_cow_calf_economics",
    "compute_finish_direct_economics",
    "compute_stocker_economics",
    "default_cow_calf_inputs",
    "default_finish_direct_inputs",
    "default_stocker_inputs",
]


# --------------------------------------------------------------------------- #
# Cow-calf
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CowCalfInputs:
    """Annual cost + revenue inputs for one cow-calf operation."""

    n_cows: int
    calving_rate: float           # fraction of cows that calve (0.85 ≈ typical)
    weaning_rate: float           # fraction of calved that wean (0.95 ≈ typical)
    weaned_weight_lbs: float      # typical: 500-575 lbs
    weaned_price_per_cwt: float   # current feeder market for that weight
    # Costs per cow per year:
    pasture_acres_per_cow: float
    pasture_cost_per_acre: float  # rent or amortized owned cost
    hay_tons_per_cow: float       # typical: 1.5-2.5
    hay_cost_per_ton: float       # local; user-entered
    supplement_cost_per_cow: float  # mineral, protein tubs, etc.
    vet_breeding_per_cow: float   # vet + AI/bull + ear tags
    fixed_per_cow: float          # fence/labor/fuel/depreciation amortized per cow
    bull_pct: float = 0.04        # 1 bull per ~25 cows
    bull_annual_cost: float = 0.0  # depreciation + maintenance per bull/year


@dataclass(frozen=True)
class CowCalfEconomics:
    n_calves_weaned: int
    weaned_calf_revenue: float        # $ per calf
    annual_cost_per_cow: float
    cost_per_calf: float
    margin_per_calf: float
    total_revenue: float
    total_cost: float
    total_margin: float
    breakeven_weaned_price_per_cwt: float


def default_cow_calf_inputs() -> CowCalfInputs:
    """Mid-size operation, mixed-region averages."""
    return CowCalfInputs(
        n_cows=60,
        calving_rate=0.88,
        weaning_rate=0.94,
        weaned_weight_lbs=525.0,
        weaned_price_per_cwt=320.0,
        pasture_acres_per_cow=4.0,
        pasture_cost_per_acre=55.0,
        hay_tons_per_cow=2.0,
        hay_cost_per_ton=220.0,
        supplement_cost_per_cow=85.0,
        vet_breeding_per_cow=90.0,
        fixed_per_cow=170.0,
        bull_pct=0.04,
        bull_annual_cost=2400.0,
    )


def compute_cow_calf_economics(inputs: CowCalfInputs) -> CowCalfEconomics:
    if inputs.n_cows <= 0:
        raise ValueError("n_cows must be positive")
    if not 0.0 < inputs.calving_rate <= 1.0:
        raise ValueError("calving_rate must be in (0, 1]")
    if not 0.0 < inputs.weaning_rate <= 1.0:
        raise ValueError("weaning_rate must be in (0, 1]")

    pasture = inputs.pasture_acres_per_cow * inputs.pasture_cost_per_acre
    hay = inputs.hay_tons_per_cow * inputs.hay_cost_per_ton
    bull_cost_per_cow = inputs.bull_pct * inputs.bull_annual_cost
    annual_cost_per_cow = (
        pasture + hay + inputs.supplement_cost_per_cow
        + inputs.vet_breeding_per_cow + inputs.fixed_per_cow + bull_cost_per_cow
    )

    calves_weaned = inputs.n_cows * inputs.calving_rate * inputs.weaning_rate
    if calves_weaned <= 0:
        raise ValueError("calculation produced zero weaned calves; check rates")

    calf_revenue = inputs.weaned_weight_lbs / 100.0 * inputs.weaned_price_per_cwt
    total_cost = inputs.n_cows * annual_cost_per_cow
    cost_per_calf = total_cost / calves_weaned
    margin_per_calf = calf_revenue - cost_per_calf
    total_revenue = calves_weaned * calf_revenue
    total_margin = total_revenue - total_cost

    breakeven_cwt = (cost_per_calf / inputs.weaned_weight_lbs) * 100.0

    return CowCalfEconomics(
        n_calves_weaned=round(calves_weaned),
        weaned_calf_revenue=calf_revenue,
        annual_cost_per_cow=annual_cost_per_cow,
        cost_per_calf=cost_per_calf,
        margin_per_calf=margin_per_calf,
        total_revenue=total_revenue,
        total_cost=total_cost,
        total_margin=total_margin,
        breakeven_weaned_price_per_cwt=breakeven_cwt,
    )


# --------------------------------------------------------------------------- #
# Stocker
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StockerInputs:
    """One stocker cycle: buy weaned calf, graze, sell at feedlot weight."""

    n_head: int
    purchase_weight_lbs: float
    purchase_price_per_cwt: float
    sale_weight_lbs: float
    sale_price_per_cwt: float
    days_on_grass: int
    pasture_cost_per_head_per_day: float  # daily grass cost
    hay_supplement_cost_per_head: float    # winter / drought supplement
    feed_supplement_cost_per_head: float   # mineral + supplemental grain
    vet_per_head: float                    # health + dewormer + processing
    death_loss_pct: float                  # typical: 1-3%
    interest_rate_annual: float            # on purchase cost over days on grass


@dataclass(frozen=True)
class StockerEconomics:
    purchase_cost_per_head: float
    grass_cost_per_head: float
    interest_cost_per_head: float
    death_loss_addon_per_head: float
    total_cost_per_head: float
    sale_revenue_per_head: float
    margin_per_head: float
    breakeven_sale_price_per_cwt: float
    total_revenue: float
    total_cost: float
    total_margin: float


def default_stocker_inputs() -> StockerInputs:
    return StockerInputs(
        n_head=120,
        purchase_weight_lbs=525.0,
        purchase_price_per_cwt=320.0,
        sale_weight_lbs=775.0,
        sale_price_per_cwt=290.0,
        days_on_grass=180,
        pasture_cost_per_head_per_day=0.45,
        hay_supplement_cost_per_head=45.0,
        feed_supplement_cost_per_head=25.0,
        vet_per_head=35.0,
        death_loss_pct=0.015,
        interest_rate_annual=0.08,
    )


def compute_stocker_economics(inputs: StockerInputs) -> StockerEconomics:
    if inputs.sale_weight_lbs <= inputs.purchase_weight_lbs:
        raise ValueError("sale_weight_lbs must exceed purchase_weight_lbs")
    if not 0.0 <= inputs.death_loss_pct < 1.0:
        raise ValueError("death_loss_pct must be in [0, 1)")

    purchase_cost = inputs.purchase_weight_lbs / 100.0 * inputs.purchase_price_per_cwt
    grass_cost = inputs.pasture_cost_per_head_per_day * inputs.days_on_grass
    interest_cost = (
        purchase_cost * inputs.interest_rate_annual * inputs.days_on_grass / 365.0
    )

    pre_loss = (
        purchase_cost + grass_cost + inputs.hay_supplement_cost_per_head
        + inputs.feed_supplement_cost_per_head + inputs.vet_per_head + interest_cost
    )
    death_loss_addon = pre_loss * (1.0 / (1.0 - inputs.death_loss_pct) - 1.0)
    total_per_head = pre_loss + death_loss_addon

    sale_revenue = inputs.sale_weight_lbs / 100.0 * inputs.sale_price_per_cwt
    margin_per_head = sale_revenue - total_per_head
    breakeven_cwt = total_per_head / inputs.sale_weight_lbs * 100.0

    return StockerEconomics(
        purchase_cost_per_head=purchase_cost,
        grass_cost_per_head=grass_cost,
        interest_cost_per_head=interest_cost,
        death_loss_addon_per_head=death_loss_addon,
        total_cost_per_head=total_per_head,
        sale_revenue_per_head=sale_revenue,
        margin_per_head=margin_per_head,
        breakeven_sale_price_per_cwt=breakeven_cwt,
        total_revenue=sale_revenue * inputs.n_head,
        total_cost=total_per_head * inputs.n_head,
        total_margin=margin_per_head * inputs.n_head,
    )


# --------------------------------------------------------------------------- #
# Finish & direct-market
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FinishDirectInputs:
    """Raise or buy a feeder; finish on the farm; sell freezer beef.

    Revenue model is per-pound hanging weight (the carcass weight after
    slaughter, before cut-and-wrap losses). A 1400-lb live steer typically
    yields ~830 lbs hanging (60% dressing %) and ~580 lbs retail cuts
    (70% of hanging). Producers price either by hanging-weight or by
    quarter-share total.
    """

    n_head: int
    feeder_cost_per_head: float            # if raised on-farm, use opportunity cost
    days_on_farm: int                       # placement to slaughter
    finished_live_weight_lbs: float        # typical: 1200-1400 grass; 1300-1500 grain
    dressing_pct: float                     # hanging / live; 0.58-0.62 typical
    pasture_cost_per_head_per_day: float
    hay_supplement_cost_per_head: float
    grain_supplement_cost_per_head: float  # 0 for purely grass-fed
    vet_per_head: float
    death_loss_pct: float
    abattoir_slaughter_fee_per_head: float  # 75-200 typical
    cut_and_wrap_per_lb_hanging: float      # ~$0.75-$1.10
    other_per_head: float                   # marketing, transport, storage
    direct_retail_per_lb_hanging: float     # producer's set price


@dataclass(frozen=True)
class FinishDirectEconomics:
    hanging_weight_lbs: float
    total_cost_per_head: float
    cost_per_lb_hanging: float
    sale_revenue_per_head: float
    margin_per_head: float
    breakeven_per_lb_hanging: float
    commodity_floor_per_lb_hanging: float | None  # $/lb hanging implied by LE futures
    total_revenue: float
    total_cost: float
    total_margin: float


def default_finish_direct_inputs() -> FinishDirectInputs:
    """A small-scale grass-plus-grain finishing operation."""
    return FinishDirectInputs(
        n_head=10,
        feeder_cost_per_head=1700.0,         # ~775 lb feeder at $220/cwt
        days_on_farm=210,
        finished_live_weight_lbs=1350.0,
        dressing_pct=0.61,
        pasture_cost_per_head_per_day=0.55,
        hay_supplement_cost_per_head=180.0,
        grain_supplement_cost_per_head=320.0,
        vet_per_head=55.0,
        death_loss_pct=0.01,
        abattoir_slaughter_fee_per_head=125.0,
        cut_and_wrap_per_lb_hanging=0.95,
        other_per_head=75.0,
        direct_retail_per_lb_hanging=6.50,
    )


def compute_finish_direct_economics(
    inputs: FinishDirectInputs,
    *,
    live_cattle_futures_per_cwt: float | None = None,
) -> FinishDirectEconomics:
    """Compute finish-and-direct economics.

    ``live_cattle_futures_per_cwt`` is optional context: when supplied,
    we translate it to an equivalent $/lb hanging *commodity floor*
    (futures × 100 ÷ dressing_pct ÷ 100 = futures-live × dressing-implied)
    so the producer can see how their direct retail price compares to
    selling the same animal at commodity prices. This is the "is your
    direct price actually worth the work?" sanity check.
    """
    if not 0.45 <= inputs.dressing_pct <= 0.70:
        raise ValueError(
            "dressing_pct must be in [0.45, 0.70]; "
            f"got {inputs.dressing_pct}"
        )
    if not 0.0 <= inputs.death_loss_pct < 1.0:
        raise ValueError("death_loss_pct must be in [0, 1)")

    hanging_lbs = inputs.finished_live_weight_lbs * inputs.dressing_pct

    pasture = inputs.pasture_cost_per_head_per_day * inputs.days_on_farm
    cut_wrap = inputs.cut_and_wrap_per_lb_hanging * hanging_lbs

    pre_loss = (
        inputs.feeder_cost_per_head
        + pasture
        + inputs.hay_supplement_cost_per_head
        + inputs.grain_supplement_cost_per_head
        + inputs.vet_per_head
        + inputs.abattoir_slaughter_fee_per_head
        + cut_wrap
        + inputs.other_per_head
    )
    death_loss_addon = pre_loss * (1.0 / (1.0 - inputs.death_loss_pct) - 1.0)
    total_per_head = pre_loss + death_loss_addon

    sale_revenue = inputs.direct_retail_per_lb_hanging * hanging_lbs
    margin_per_head = sale_revenue - total_per_head
    cost_per_lb = total_per_head / hanging_lbs
    breakeven_per_lb = cost_per_lb

    commodity_floor: float | None = None
    if live_cattle_futures_per_cwt is not None:
        # Live cattle futures $/cwt → $/lb live → $/lb hanging via dressing %
        per_lb_live = live_cattle_futures_per_cwt / 100.0
        commodity_floor = per_lb_live / inputs.dressing_pct

    return FinishDirectEconomics(
        hanging_weight_lbs=hanging_lbs,
        total_cost_per_head=total_per_head,
        cost_per_lb_hanging=cost_per_lb,
        sale_revenue_per_head=sale_revenue,
        margin_per_head=margin_per_head,
        breakeven_per_lb_hanging=breakeven_per_lb,
        commodity_floor_per_lb_hanging=commodity_floor,
        total_revenue=sale_revenue * inputs.n_head,
        total_cost=total_per_head * inputs.n_head,
        total_margin=margin_per_head * inputs.n_head,
    )
