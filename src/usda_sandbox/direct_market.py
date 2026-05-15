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
    "COW_CALF_REGIONS",
    "FINISH_DIRECT_REGIONS",
    "STOCKER_REGIONS",
    "CowCalfEconomics",
    "CowCalfInputs",
    "FinishDirectEconomics",
    "FinishDirectInputs",
    "StockerEconomics",
    "StockerInputs",
    "compute_cow_calf_economics",
    "compute_finish_direct_economics",
    "compute_stocker_economics",
    "cow_calf_inputs_for_region",
    "default_cow_calf_inputs",
    "default_finish_direct_inputs",
    "default_stocker_inputs",
    "finish_direct_inputs_for_region",
    "stocker_inputs_for_region",
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


# Regional cow-calf cost stacks. Sourced from:
# - NASS pasture rental rates 2024 (national avg $15.50/ac; Iowa $63.50;
#   Maryland $50.50; Texas/Oklahoma $8-15/ac; rangeland states lower).
# - ISU Iowa Beef Center 2025 (NGP detail).
# - TAMU 2024 native + improved pasture budgets (Southern Plains).
# - UKY 2024 spring-calving estimates (Mid-South / SE proxy).
# - USDA ERS Amber Waves Dec 2024 on cost-by-region patterns.
COW_CALF_REGIONS: dict[str, dict[str, object]] = {
    "Plains (KS/NE/OK/TX panhandle)": {
        "description": (
            "Native and improved pasture; moderate hay feeding (~2 tons/cow); "
            "low-to-moderate pasture rent; the v3.0 baseline."
        ),
        "pasture_acres_per_cow": 4.0,
        "pasture_cost_per_acre": 25.0,
        "hay_tons_per_cow": 2.2,
        "hay_cost_per_ton": 220.0,
        "supplement_cost_per_cow": 95.0,
        "vet_breeding_per_cow": 75.0,
        "fixed_per_cow": 250.0,
        "weaning_weight_lbs": 550.0,
    },
    "Midwest (IA/MO/IL/IN)": {
        "description": (
            "High-quality cool-season pasture; competing with corn for "
            "land; pasture rent $40-65/ac (NASS Iowa $63.50/ac 2024); "
            "2-2.5 tons hay/cow; smaller herd sizes."
        ),
        "pasture_acres_per_cow": 2.5,
        "pasture_cost_per_acre": 60.0,
        "hay_tons_per_cow": 2.5,
        "hay_cost_per_ton": 220.0,
        "supplement_cost_per_cow": 90.0,
        "vet_breeding_per_cow": 80.0,
        "fixed_per_cow": 280.0,
        "weaning_weight_lbs": 575.0,
    },
    "Southeast (TN/GA/KY/AR/FL)": {
        "description": (
            "Year-round forage; shorter hay-feeding window (0.5-1.5 tons/cow); "
            "higher parasite/fly vet costs; humid heat = lower weaning "
            "weights; smaller herds dominate."
        ),
        "pasture_acres_per_cow": 2.0,
        "pasture_cost_per_acre": 45.0,
        "hay_tons_per_cow": 1.0,
        "hay_cost_per_ton": 180.0,
        "supplement_cost_per_cow": 80.0,
        "vet_breeding_per_cow": 110.0,
        "fixed_per_cow": 240.0,
        "weaning_weight_lbs": 520.0,
    },
    "Pacific NW / Mountain (MT/ID/OR/WA)": {
        "description": (
            "Mix of irrigated meadow and rangeland; ~1.5-2 tons hay/cow; "
            "longer winter than Plains but shorter than Northeast; "
            "moderate pasture rent."
        ),
        "pasture_acres_per_cow": 5.0,
        "pasture_cost_per_acre": 30.0,
        "hay_tons_per_cow": 2.0,
        "hay_cost_per_ton": 250.0,
        "supplement_cost_per_cow": 90.0,
        "vet_breeding_per_cow": 75.0,
        "fixed_per_cow": 260.0,
        "weaning_weight_lbs": 560.0,
    },
    "Northeast (PA/NY/VT/MD/VA)": {
        "description": (
            "Long winter = 3+ tons hay/cow; highest land costs (NASS "
            "Maryland $50.50/ac 2024 with higher rates further north); "
            "smaller herds; lower stocking rates per acre."
        ),
        "pasture_acres_per_cow": 3.0,
        "pasture_cost_per_acre": 85.0,
        "hay_tons_per_cow": 3.0,
        "hay_cost_per_ton": 260.0,
        "supplement_cost_per_cow": 100.0,
        "vet_breeding_per_cow": 95.0,
        "fixed_per_cow": 320.0,
        "weaning_weight_lbs": 545.0,
    },
    "Southwest rangeland (AZ/NM/NV/W-TX)": {
        "description": (
            "Arid rangeland; 20-80 acres/cow depending on rainfall; "
            "very low $/acre but high supplementation needs; large pastures, "
            "low hay use, high vet/transport costs."
        ),
        "pasture_acres_per_cow": 40.0,
        "pasture_cost_per_acre": 4.0,
        "hay_tons_per_cow": 0.8,
        "hay_cost_per_ton": 270.0,
        "supplement_cost_per_cow": 150.0,
        "vet_breeding_per_cow": 90.0,
        "fixed_per_cow": 270.0,
        "weaning_weight_lbs": 500.0,
    },
}


def cow_calf_inputs_for_region(
    region: str,
    *,
    n_cows: int = 60,
    calving_rate: float = 0.90,
    weaning_rate: float = 0.94,
    weaned_price_per_cwt: float = 315.0,
) -> CowCalfInputs:
    """Build a :class:`CowCalfInputs` from a regional preset.

    ``region`` must be a key in :data:`COW_CALF_REGIONS`. Non-cost inputs
    (n_cows, rates, prices) take user-supplied values so the caller can
    plug in their actual herd.
    """
    if region not in COW_CALF_REGIONS:
        raise ValueError(
            f"Unknown region {region!r}; valid: {sorted(COW_CALF_REGIONS)}"
        )
    cfg = COW_CALF_REGIONS[region]
    return CowCalfInputs(
        n_cows=n_cows,
        calving_rate=calving_rate,
        weaning_rate=weaning_rate,
        weaned_weight_lbs=float(cfg["weaning_weight_lbs"]),
        weaned_price_per_cwt=weaned_price_per_cwt,
        pasture_acres_per_cow=float(cfg["pasture_acres_per_cow"]),
        pasture_cost_per_acre=float(cfg["pasture_cost_per_acre"]),
        hay_tons_per_cow=float(cfg["hay_tons_per_cow"]),
        hay_cost_per_ton=float(cfg["hay_cost_per_ton"]),
        supplement_cost_per_cow=float(cfg["supplement_cost_per_cow"]),
        vet_breeding_per_cow=float(cfg["vet_breeding_per_cow"]),
        fixed_per_cow=float(cfg["fixed_per_cow"]),
        bull_pct=0.04,
        bull_annual_cost=2800.0,
    )


def default_cow_calf_inputs() -> CowCalfInputs:
    """Mid-size operation, mixed-region 2024 averages.

    Source triangulation:
    - **ISU Iowa Beef Center 2024** Northern Great Plains: $950/cow
      operating + $900/cow fixed; feed $610/cow.
    - **OSU Cow-Calf Corner 2024**: ~2.4 tons hay/cow/year average,
      ~30 lbs/cow/day for ~160 days of hay feeding; vet $25, breeding $40.
    - **TAMU 2024 native-pasture budget**: Southern Plains hay needs lower
      (~1.5 tons/cow), pasture rents higher per acre but lower acres/cow.

    We use the midpoint that lands roughly $1,250-$1,400/cow total —
    representative of a 60-head spring-calving operation in transitional
    geography (Midwest or transitional Plains). See
    :data:`usda_sandbox.sources.COW_CALF_SOURCES`.
    """
    return CowCalfInputs(
        n_cows=60,
        calving_rate=0.90,        # ISU/OSU "good management" baseline
        weaning_rate=0.94,        # ~95% of calved on average
        weaned_weight_lbs=550.0,  # 525-575 typical range
        weaned_price_per_cwt=315.0,  # 2024-2025 hot feeder market
        pasture_acres_per_cow=4.0,
        pasture_cost_per_acre=65.0,    # mid of $50-150 range; rented
        hay_tons_per_cow=2.2,           # ISU NGP avg ~2.4; we go slightly lower
        hay_cost_per_ton=230.0,         # NASS 2024 all-hay national average ~$220-240
        supplement_cost_per_cow=95.0,   # mineral + protein tubs + creep
        vet_breeding_per_cow=75.0,      # OSU vet $25 + breeding $50 (incl bull share)
        fixed_per_cow=250.0,            # fence/labor/fuel/depreciation, mid-size operation
        bull_pct=0.04,                  # 1 bull / ~25 cows
        bull_annual_cost=2800.0,        # depreciation + maintenance per bull/year
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


# Regional stocker presets. The dominant variation across geography:
# - **Plains**: wheat-pasture stocker, Nov-March, ~$0.55/head/day grazing
#   on small-grain pasture, low hay needs, low death loss.
# - **Midwest**: summer cool-season grass, longer cycle, slightly higher
#   pasture cost, low hay needs.
# - **Southeast**: tall fescue / bermudagrass on year-round pasture, low
#   pasture cost per day, longer cycle.
# - **PNW/Mountain**: irrigated meadow + dry-land mix, moderate cost.
# - **Northeast**: small-scale on improved pasture; highest pasture cost.
# - **Southwest rangeland**: many acres, low /day cost, higher death loss.
STOCKER_REGIONS: dict[str, dict[str, object]] = {
    "Plains (KS/OK wheat pasture)": {
        "description": (
            "Classic Nov-March wheat-pasture stocker (OSU CR-212). Higher "
            "$/head/day on rented small-grain pasture; low hay supplement; "
            "low death loss because of managed grazing."
        ),
        "days_on_grass": 150,
        "pasture_cost_per_head_per_day": 0.55,
        "hay_supplement_cost_per_head": 25.0,
        "feed_supplement_cost_per_head": 20.0,
        "vet_per_head": 30.0,
        "death_loss_pct": 0.012,
    },
    "Midwest (IA/MO/IL summer grass)": {
        "description": (
            "Cool-season grass stocker May–October; moderate pasture cost; "
            "minimal hay; competing with row-crop ground for acres."
        ),
        "days_on_grass": 165,
        "pasture_cost_per_head_per_day": 0.50,
        "hay_supplement_cost_per_head": 20.0,
        "feed_supplement_cost_per_head": 25.0,
        "vet_per_head": 30.0,
        "death_loss_pct": 0.015,
    },
    "Southeast (TN/GA/KY fescue/bermuda)": {
        "description": (
            "Year-round forage on fescue/bermudagrass; lower $/day "
            "pasture cost; longer cycles possible; higher vet (parasites)."
        ),
        "days_on_grass": 210,
        "pasture_cost_per_head_per_day": 0.35,
        "hay_supplement_cost_per_head": 30.0,
        "feed_supplement_cost_per_head": 25.0,
        "vet_per_head": 50.0,
        "death_loss_pct": 0.020,
    },
    "Pacific NW / Mountain (MT/ID/OR/WA)": {
        "description": (
            "Mix of irrigated meadow and dry-land grass; moderate per-day "
            "cost; longer pre-conditioning at higher elevations."
        ),
        "days_on_grass": 180,
        "pasture_cost_per_head_per_day": 0.45,
        "hay_supplement_cost_per_head": 55.0,
        "feed_supplement_cost_per_head": 25.0,
        "vet_per_head": 35.0,
        "death_loss_pct": 0.015,
    },
    "Northeast (PA/NY/VT/MD/VA)": {
        "description": (
            "Small-scale stocker on improved pasture; highest $/day cost "
            "but shorter cycles; less common as a standalone enterprise."
        ),
        "days_on_grass": 150,
        "pasture_cost_per_head_per_day": 0.75,
        "hay_supplement_cost_per_head": 70.0,
        "feed_supplement_cost_per_head": 35.0,
        "vet_per_head": 45.0,
        "death_loss_pct": 0.015,
    },
    "Southwest rangeland (AZ/NM/NV/W-TX)": {
        "description": (
            "Dry-land rangeland; many acres needed; very low $/head/day "
            "but higher death loss from heat, water-source distance, and "
            "predator pressure."
        ),
        "days_on_grass": 200,
        "pasture_cost_per_head_per_day": 0.30,
        "hay_supplement_cost_per_head": 60.0,
        "feed_supplement_cost_per_head": 45.0,
        "vet_per_head": 45.0,
        "death_loss_pct": 0.025,
    },
}


def stocker_inputs_for_region(
    region: str,
    *,
    n_head: int = 120,
    purchase_weight_lbs: float = 525.0,
    purchase_price_per_cwt: float = 315.0,
    sale_weight_lbs: float = 775.0,
    sale_price_per_cwt: float = 285.0,
    interest_rate_annual: float = 0.085,
) -> StockerInputs:
    """Build a :class:`StockerInputs` from a regional preset."""
    if region not in STOCKER_REGIONS:
        raise ValueError(
            f"Unknown region {region!r}; valid: {sorted(STOCKER_REGIONS)}"
        )
    cfg = STOCKER_REGIONS[region]
    return StockerInputs(
        n_head=n_head,
        purchase_weight_lbs=purchase_weight_lbs,
        purchase_price_per_cwt=purchase_price_per_cwt,
        sale_weight_lbs=sale_weight_lbs,
        sale_price_per_cwt=sale_price_per_cwt,
        days_on_grass=int(cfg["days_on_grass"]),
        pasture_cost_per_head_per_day=float(cfg["pasture_cost_per_head_per_day"]),
        hay_supplement_cost_per_head=float(cfg["hay_supplement_cost_per_head"]),
        feed_supplement_cost_per_head=float(cfg["feed_supplement_cost_per_head"]),
        vet_per_head=float(cfg["vet_per_head"]),
        death_loss_pct=float(cfg["death_loss_pct"]),
        interest_rate_annual=interest_rate_annual,
    )


def default_stocker_inputs() -> StockerInputs:
    """OSU CR-212 (Sept 2024) wheat-pasture stocker baseline + Plains avgs.

    OSU's 150-head November-to-March example purchased at 450 lb and sold
    at 669 lb on small-grain pasture at $0.40/lb gain. We use a slightly
    longer cycle (180 days, 525→775 lb gain) for a more typical native-
    rangeland stocker run. See :data:`usda_sandbox.sources.STOCKER_SOURCES`.
    """
    return StockerInputs(
        n_head=120,
        purchase_weight_lbs=525.0,
        purchase_price_per_cwt=315.0,  # aligned with cow-calf weaned-price baseline
        sale_weight_lbs=775.0,
        sale_price_per_cwt=285.0,      # heavier feeders run slightly below light
        days_on_grass=180,
        pasture_cost_per_head_per_day=0.45,  # rangeland; wheat pasture runs ~$0.55
        hay_supplement_cost_per_head=45.0,
        feed_supplement_cost_per_head=25.0,
        vet_per_head=35.0,
        death_loss_pct=0.015,           # OSU CR-212 baseline 1-2% for managed grazing
        interest_rate_annual=0.085,     # mid-2024 ag loan rate
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


# Regional finish-and-direct presets. Three things vary by region:
# - **Retail $/lb hanging**: NE / PNW commands the premium (NYC, Boston,
#   Portland, Seattle markets); Plains and Midwest mid; SE / SW lower.
# - **Grain supplement cost**: cheap in Corn Belt, expensive far from it.
# - **Abattoir + cut-and-wrap fees**: highest in NE (small-plant
#   capacity, premium labor), lowest in Midwest (high small-plant
#   density), moderate elsewhere.
FINISH_DIRECT_REGIONS: dict[str, dict[str, object]] = {
    "Plains (KS/OK/TX panhandle)": {
        "description": (
            "Cheap grain access; lower abattoir density; mid-tier retail. "
            "Strong for grain-finished freezer beef; grass-finish viable."
        ),
        "grain_supplement_cost_per_head": 290.0,
        "hay_supplement_cost_per_head": 150.0,
        "abattoir_slaughter_fee_per_head": 115.0,
        "cut_and_wrap_per_lb_hanging": 0.85,
        "direct_retail_per_lb_hanging": 6.00,
    },
    "Midwest / Corn Belt (IA/MO/IL/IN)": {
        "description": (
            "Cheapest grain access; densest small-plant abattoir network; "
            "strong direct-market demand in metro Chicago / Indy / St. Louis."
        ),
        "grain_supplement_cost_per_head": 260.0,
        "hay_supplement_cost_per_head": 160.0,
        "abattoir_slaughter_fee_per_head": 110.0,
        "cut_and_wrap_per_lb_hanging": 0.80,
        "direct_retail_per_lb_hanging": 6.50,
    },
    "Southeast (TN/GA/KY/NC/VA)": {
        "description": (
            "Year-round forage favors grass-finish; moderate grain access; "
            "moderate retail (Atlanta, Nashville, Charlotte metros)."
        ),
        "grain_supplement_cost_per_head": 350.0,
        "hay_supplement_cost_per_head": 130.0,
        "abattoir_slaughter_fee_per_head": 140.0,
        "cut_and_wrap_per_lb_hanging": 0.95,
        "direct_retail_per_lb_hanging": 6.25,
    },
    "Pacific NW / Mountain (MT/ID/OR/WA)": {
        "description": (
            "Grass-finish strong (Mountain Beef territory); high retail in "
            "Portland / Seattle metros; longer hay needs in winter."
        ),
        "grain_supplement_cost_per_head": 360.0,
        "hay_supplement_cost_per_head": 220.0,
        "abattoir_slaughter_fee_per_head": 145.0,
        "cut_and_wrap_per_lb_hanging": 1.05,
        "direct_retail_per_lb_hanging": 7.00,
    },
    "Northeast (PA/NY/VT/MA/MD)": {
        "description": (
            "Highest retail (NYC / Boston / Philly metros); long winter "
            "drives hay; small-plant capacity tight; abattoir + C&W fees "
            "are the highest in the country."
        ),
        "grain_supplement_cost_per_head": 410.0,
        "hay_supplement_cost_per_head": 260.0,
        "abattoir_slaughter_fee_per_head": 175.0,
        "cut_and_wrap_per_lb_hanging": 1.15,
        "direct_retail_per_lb_hanging": 7.50,
    },
    "Southwest (AZ/NM/NV/W-TX)": {
        "description": (
            "Arid finishing is hard; grain heavy if you finish; lower "
            "retail outside Phoenix / Denver; abattoir distances long."
        ),
        "grain_supplement_cost_per_head": 380.0,
        "hay_supplement_cost_per_head": 200.0,
        "abattoir_slaughter_fee_per_head": 150.0,
        "cut_and_wrap_per_lb_hanging": 1.00,
        "direct_retail_per_lb_hanging": 5.75,
    },
}


def finish_direct_inputs_for_region(
    region: str,
    *,
    n_head: int = 10,
    feeder_cost_per_head: float = 2000.0,
    days_on_farm: int = 210,
    finished_live_weight_lbs: float = 1350.0,
    dressing_pct: float = 0.61,
    pasture_cost_per_head_per_day: float = 0.55,
    vet_per_head: float = 55.0,
    death_loss_pct: float = 0.01,
    other_per_head: float = 75.0,
) -> FinishDirectInputs:
    """Build a :class:`FinishDirectInputs` from a regional preset."""
    if region not in FINISH_DIRECT_REGIONS:
        raise ValueError(
            f"Unknown region {region!r}; valid: {sorted(FINISH_DIRECT_REGIONS)}"
        )
    cfg = FINISH_DIRECT_REGIONS[region]
    return FinishDirectInputs(
        n_head=n_head,
        feeder_cost_per_head=feeder_cost_per_head,
        days_on_farm=days_on_farm,
        finished_live_weight_lbs=finished_live_weight_lbs,
        dressing_pct=dressing_pct,
        pasture_cost_per_head_per_day=pasture_cost_per_head_per_day,
        hay_supplement_cost_per_head=float(cfg["hay_supplement_cost_per_head"]),
        grain_supplement_cost_per_head=float(cfg["grain_supplement_cost_per_head"]),
        vet_per_head=vet_per_head,
        death_loss_pct=death_loss_pct,
        abattoir_slaughter_fee_per_head=float(cfg["abattoir_slaughter_fee_per_head"]),
        cut_and_wrap_per_lb_hanging=float(cfg["cut_and_wrap_per_lb_hanging"]),
        other_per_head=other_per_head,
        direct_retail_per_lb_hanging=float(cfg["direct_retail_per_lb_hanging"]),
    )


def default_finish_direct_inputs() -> FinishDirectInputs:
    """Small-scale grass-plus-grain finishing operation, 2024 baselines.

    Cross-validated against:
    - **MSU 2024 worksheet**: $125/head slaughter + ~$1/lb hanging
      cut-and-wrap; $3.80/lb hanging carcass + processing → $7.95/lb
      retail-equivalent.
    - **OSU Meat Sci Extension**: cut-and-wrap $0.55-$0.80/lb baseline,
      higher for specialty packaging.
    - **USDA AMS Grass-Fed Beef Report (April 2024)**: avg hanging-weight
      $4.31/lb across small producers; range $3.15-$5.45. We default the
      *direct retail* price to $6.50 as the median across observed real
      producer websites (Deer Run $6.50, Mountain $6.70, Blessing Falls $6.95).

    Feeder cost reflects late-2024 / 2025 strong feeder market
    (~$260-$280/cwt for 750-800 lb steers).
    See :data:`usda_sandbox.sources.FINISH_DIRECT_SOURCES`.
    """
    return FinishDirectInputs(
        n_head=10,
        feeder_cost_per_head=2000.0,        # ~775 lb @ $260/cwt — 2024-25 market
        days_on_farm=210,
        finished_live_weight_lbs=1350.0,
        dressing_pct=0.61,
        pasture_cost_per_head_per_day=0.55,
        hay_supplement_cost_per_head=180.0,
        grain_supplement_cost_per_head=320.0,
        vet_per_head=55.0,
        death_loss_pct=0.01,
        abattoir_slaughter_fee_per_head=125.0,   # MSU 2024 worksheet midpoint
        cut_and_wrap_per_lb_hanging=0.95,        # mid of $0.75-$1.10 range
        other_per_head=75.0,
        direct_retail_per_lb_hanging=6.50,       # USDA AMS + producer median
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
