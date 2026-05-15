"""Freezer-beef pricing reference + yield math.

Direct-market beef pricing has no central index. These are research-derived
ranges from late-2020s extension surveys (KSU, PSU, UTK, U of Idaho); the
numbers reflect typical asking prices for direct-to-consumer beef sales
in the US, with regional variation noted.

Always quote the ranges as ranges to users — pricing varies considerably
by region, grain-fed vs grass-fed, breed, and the producer's brand. The
purpose here is "give the rancher a starting point and a sanity-check
upper/lower bound," not a single number.

Yield math: a typical fed beef carcass yields roughly
    finished_live_weight × 0.60 ≈ hanging weight
    hanging_weight × 0.65 to 0.72 ≈ retail-cut yield
with bones, fat trim, and shrink accounting for the rest.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "REFERENCE_HANGING_PRICING",
    "REFERENCE_RETAIL_BUNDLE",
    "REFERENCE_SHARE_PRICING",
    "PriceRange",
    "ShareSize",
    "expected_retail_yield_lbs",
    "value_share",
]


@dataclass(frozen=True)
class PriceRange:
    low: float
    mid: float
    high: float
    unit: str
    note: str


@dataclass(frozen=True)
class ShareSize:
    name: str            # "Quarter", "Half", "Whole"
    fraction: float      # of a whole carcass
    typical_hanging_lbs: tuple[float, float]  # (low, high)
    typical_retail_lbs: tuple[float, float]   # post cut-and-wrap


# Hanging-weight pricing (what the producer charges per lb of hanging /
# carcass weight, plus separate cut-and-wrap fees billed by the abattoir).
#
# Anchored to: USDA AMS National Grass-Fed Beef Report (April 2024),
# MSU 2024 Pricing Freezer Beef worksheet, and a 2024-2026 survey of
# 5 real producer websites (Climbing Stump $4, Deer Run $6.50, Mountain
# Beef $6.70, Blessing Falls $6.95, Grand View $9). See
# :data:`usda_sandbox.sources.PRICING_SOURCES`.
REFERENCE_HANGING_PRICING: dict[str, PriceRange] = {
    "grain_finished": PriceRange(
        low=4.50, mid=5.75, high=7.50,
        unit="$/lb hanging",
        note=(
            "Grain-finished (typical Angus / commercial cross). MSU 2024 "
            "reference example is $3.80/lb carcass before processing, "
            "$7.95/lb retail-equivalent after $125 slaughter + $1/lb cut-"
            "and-wrap. Producer hanging asking prices commonly cluster "
            "$5.00-$6.50."
        ),
    ),
    "grass_finished": PriceRange(
        low=4.50, mid=6.50, high=9.00,
        unit="$/lb hanging",
        note=(
            "Grass-finished. USDA AMS April 2024 average $4.31/lb hanging "
            "across small producers; published range $3.15-$5.45. Real "
            "producer sites in 2024-26 cluster around $6.00-$7.00 (Deer "
            "Run $6.50, Mountain Beef $6.70, Blessing Falls $6.95). High "
            "end is established direct-market brands."
        ),
    ),
    "premium_branded": PriceRange(
        low=7.00, mid=9.00, high=12.00,
        unit="$/lb hanging",
        note=(
            "Premium / branded / heritage-breed / regenerative-certified. "
            "Grand View Beef (2026) at $9.00/lb hanging; established "
            "brands in high-income metros can sustain $10-$12. Requires "
            "a real brand and customer pipeline — not a starting point."
        ),
    ),
}


# Whole / half / quarter share sizes (typical carcass ~800-900 lbs hanging
# from a 1300-1500 lb finished steer at 60% dressing).
REFERENCE_SHARE_PRICING: tuple[ShareSize, ...] = (
    ShareSize(
        name="Quarter",
        fraction=0.25,
        typical_hanging_lbs=(180.0, 225.0),
        typical_retail_lbs=(115.0, 160.0),
    ),
    ShareSize(
        name="Half",
        fraction=0.50,
        typical_hanging_lbs=(360.0, 450.0),
        typical_retail_lbs=(230.0, 320.0),
    ),
    ShareSize(
        name="Whole",
        fraction=1.00,
        typical_hanging_lbs=(720.0, 900.0),
        typical_retail_lbs=(460.0, 640.0),
    ),
)


# Per-cut retail pricing — for producers selling individual cuts at
# farmers markets or via a CSA-style monthly box. Per-pound, late-2020s.
REFERENCE_RETAIL_BUNDLE: dict[str, PriceRange] = {
    "ground_beef": PriceRange(
        low=7.00, mid=9.00, high=12.00,
        unit="$/lb",
        note="Most of the carcass by weight; the producer's volume product.",
    ),
    "steaks": PriceRange(
        low=18.00, mid=24.00, high=32.00,
        unit="$/lb",
        note="Ribeye, NY strip, sirloin. Premium retail at the upper end.",
    ),
    "roasts": PriceRange(
        low=10.00, mid=13.00, high=16.00,
        unit="$/lb",
        note="Chuck roast, rump, sirloin tip. Holiday and slow-cook market.",
    ),
    "specialty": PriceRange(
        low=8.00, mid=14.00, high=22.00,
        unit="$/lb",
        note="Stew, brisket, short ribs, organ meats. Wide variance.",
    ),
}


def expected_retail_yield_lbs(
    hanging_weight_lbs: float, retail_yield_pct: float = 0.68
) -> float:
    """Convert hanging weight to expected retail-cut weight.

    Typical retail yield is 65-72% of hanging; default 68% is a midpoint
    that matches grain-finished steers with normal trim. Grass-finished
    typically yields slightly lower (66-68%); heavier-conformation beef
    can run higher.
    """
    if not 0.55 <= retail_yield_pct <= 0.78:
        raise ValueError(
            "retail_yield_pct must be in [0.55, 0.78]; "
            f"got {retail_yield_pct}"
        )
    return hanging_weight_lbs * retail_yield_pct


def value_share(
    hanging_weight_lbs: float,
    fraction: float,
    price_per_lb_hanging: float,
) -> tuple[float, float]:
    """Return (share_hanging_lbs, share_total_$) for a share-size sale.

    Used by the Plan and Pricing pages to compute "what does a half share
    cost the customer if I price hanging at $X/lb."
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    share_hanging = hanging_weight_lbs * fraction
    return share_hanging, share_hanging * price_per_lb_hanging
