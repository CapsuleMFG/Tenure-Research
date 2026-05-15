"""Source citations for direct-market defaults and pricing references.

Single registry that the Plan and Pricing pages render as "Where these
numbers come from" sections. Keeps the URLs in one place so when the
defaults are refreshed annually we update both the data and the citations
together.

Each ``Source`` records:
- ``title``: human-readable label
- ``publisher``: which extension service / agency
- ``year``: when the data we extracted was current
- ``url``: where to verify
- ``relevance``: what we actually use from it (one line)
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "COW_CALF_SOURCES",
    "FINISH_DIRECT_SOURCES",
    "PRICING_SOURCES",
    "STOCKER_SOURCES",
    "Source",
]


@dataclass(frozen=True)
class Source:
    title: str
    publisher: str
    year: str
    url: str
    relevance: str


COW_CALF_SOURCES: tuple[Source, ...] = (
    Source(
        title="Iowa Cow-Calf Production Costs — November 2025",
        publisher="Iowa State University, Iowa Beef Center",
        year="2025",
        url="https://www.iowabeefcenter.org/gb/2025/November2025CowCalfCosts.html",
        relevance=(
            "2024 cow-calf operating costs in the Northern Great Plains: "
            "$950/cow operating + $900/cow fixed; feed is the dominant "
            "share at $610/cow."
        ),
    ),
    Source(
        title="2024 Cow-Calf Budgets",
        publisher="Texas A&M AgriLife Extension Agricultural Economics",
        year="2024",
        url="https://agecoext.tamu.edu/resources/crop-livestock-budgets/by-commodity/cow-calf/2024-cow-calf-budgets/",
        relevance=(
            "Regional Texas budgets with native pasture, improved pasture, "
            "and Calf-1 variants; used to triangulate Southern Plains "
            "pasture and hay costs."
        ),
    ),
    Source(
        title="Cow-Calf Profitability Estimates for 2023 and 2024",
        publisher="University of Kentucky Department of Agricultural Economics",
        year="2024",
        url="https://agecon.mgcafe.uky.edu/cow-calf-profitability-estimates-2023-and-2024-spring-calving-herd",
        relevance=(
            "Spring-calving herd cost-and-returns for the Mid-South; "
            "used for vet/breeding line averages and weaning-weight assumptions."
        ),
    ),
    Source(
        title="Livestock Enterprise Budgets for Iowa — 2025",
        publisher="Iowa State University Extension, Ag Decision Maker B1-21",
        year="2025",
        url="https://www.extension.iastate.edu/agdm/livestock/pdf/b1-21.pdf",
        relevance=(
            "Detailed cost-per-cow breakdown by line item (feed, hay, "
            "vet, breeding, fixed/labor)."
        ),
    ),
)


STOCKER_SOURCES: tuple[Source, ...] = (
    Source(
        title="Should I Buy (or Retain) Stockers to Graze Wheat Pasture? (CR-212)",
        publisher="Oklahoma State University Extension",
        year="2024",
        url="https://extension.okstate.edu/fact-sheets/should-i-buy-or-retain-stockers-to-graze-wheat-pasture.html",
        relevance=(
            "Stocker enterprise budget: 150 steers, Nov purchase at 450 lb, "
            "March sale at 669 lb. Pasture rental at $0.40/lb gain; "
            "sensitivity is ~$131/head variation per $20/cwt buy-price move."
        ),
    ),
    Source(
        title="OSU Sample Budgets — Livestock",
        publisher="Oklahoma State University Extension",
        year="2024",
        url="https://extension.okstate.edu/programs/farm-management-and-finance/budgets/sample-budgets",
        relevance=(
            "Reference grass-stocker and dual-purpose budgets used for "
            "death-loss, vet-per-head, and interest assumptions."
        ),
    ),
)


FINISH_DIRECT_SOURCES: tuple[Source, ...] = (
    Source(
        title="How Much Should You Charge? Pricing Your Meat Cuts",
        publisher="Ohio State University Extension (Small Ruminants)",
        year="2023",
        url="https://u.osu.edu/sheep/2023/01/17/how-much-should-you-charge-pricing-your-meat-cuts/",
        relevance=(
            "Methodology for converting live weight → hanging cost → final "
            "retail price with 25% markup; cites 58% dressing for grass-fed Angus."
        ),
    ),
    Source(
        title="Selling and Pricing Freezer Beef",
        publisher="Ohio State University Meat Science Extension",
        year="2024",
        url="https://meatsci.osu.edu/sites/meatsci/files/imce/Selling%20and%20Pricing%20Freezer%20Beef%20FINAL.pdf",
        relevance=(
            "Slaughter fee, cut-and-wrap fee, and retail markup ranges "
            "for direct-market freezer beef operations."
        ),
    ),
    Source(
        title="Pricing Freezer Beef Worksheet (grain-finished)",
        publisher="Michigan State University Extension",
        year="2024",
        url="https://www.canr.msu.edu/resources/grain-fed-freezer-beef-pricing-worksheet-1",
        relevance=(
            "Worksheet template used to validate our cost-of-gain, "
            "abattoir-fee, and cut-and-wrap default ranges for grain-finished."
        ),
    ),
)


PRICING_SOURCES: tuple[Source, ...] = (
    Source(
        title="National Grass-Fed Beef Report (Quarterly)",
        publisher="USDA Agricultural Marketing Service",
        year="2024-Q2",
        url="https://www.ams.usda.gov/mnreports/lsmngfbeef.pdf",
        relevance=(
            "April 2024 average grass-fed hanging-weight price $4.31/lb; "
            "range $3.15–$5.45 across small producers. Source for the "
            "grass-finished mid and the low end of the published range."
        ),
    ),
    Source(
        title="How Much Should You Charge? Pricing Your Meat Cuts",
        publisher="Penn State / Ohio State Extension",
        year="2023-2024",
        url="https://extension.psu.edu/how-much-should-you-charge-pricing-your-meat-cuts",
        relevance=(
            "Per-cut retail pricing methodology; calibration for "
            "ground-beef vs steaks vs roasts spread."
        ),
    ),
    Source(
        title="How to Price Freezer Beef",
        publisher="Michigan State University Extension",
        year="2024",
        url="https://www.canr.msu.edu/news/how-to-price-freezer-beef",
        relevance=(
            "Reference $3.80/lb carcass + $125/head slaughter + $1/lb "
            "cut-and-wrap → $7.95/lb final retail equivalent — anchor "
            "for the grain-finished pricing band."
        ),
    ),
    Source(
        title="Direct-Market Producer Pricing Survey (informal)",
        publisher="Aggregated from current 2024-2026 producer websites",
        year="2024-2026",
        url="https://www.kdfarms.site/farm-blog/2024/beef-hanging-weight-prices-2024",
        relevance=(
            "Real producer hanging-weight prices observed: $4.00 "
            "(Climbing Stump), $6.50 (Deer Run), $6.70 (Mountain Beef), "
            "$6.95 (Blessing Falls), $9.00 (Grand View). Drove the "
            "premium-branded upper range."
        ),
    ),
)
