"""Methodology — plain-English explainer for data, models, and uncertainty."""

from __future__ import annotations

import streamlit as st
from components.cache import cache_generated_at, cache_horizon
from components.sidebar import render_sidebar
from components.theme import INK_SOFT, inject_global_css

inject_global_css()
render_sidebar(persistent_picker=False)

st.markdown("# How Tenure Brief works")
st.markdown(
    f"<p style='color:{INK_SOFT};font-size:1.0rem;'>"
    "Plain-English explanation of where the data comes from, what the models "
    "do, and how to read the prediction intervals."
    "</p>",
    unsafe_allow_html=True,
)

st.markdown("## Where the data comes from")
st.markdown(
    """
**USDA Economic Research Service.** All cash prices, wholesale prices, and
quarterly supply/disappearance numbers come from the
[Livestock and Meat Domestic Data](https://www.ers.usda.gov/data-products/livestock-and-meat-domestic-data)
product page. We mirror the XLSX downloads, parse them into a tidy parquet
store, and refresh on a schedule.

**CME futures via yfinance.** For series where futures pricing helps forecast
the cash market (cattle, hogs, feeder cattle), we pull the continuous
front-month contract for Live Cattle (LE), Lean Hogs (HE), and Feeder Cattle
(GF) from Yahoo Finance and feed them as a single exogenous regressor.

**What's *not* in here.** AMS LMR (the daily mandatory-reporting feed),
weather, feed cost indices, packer margins, anything sentiment-derived — all
explicitly out of scope for v1.
"""
)

st.markdown("## How the data refreshes")
gen_at = cache_generated_at()
gen_label = gen_at or "not yet generated"
st.markdown(
    f"""
**Two GitHub Actions crons keep the app live:**

*Weekly (Sunday night UTC)* — full refresh + forecast rebake:

1. Discovers and downloads any updated XLSX/ZIP files from the ERS product page.
2. Pulls monthly continuous front-month futures for LE / HE / GF via yfinance.
3. Cleans every catalog series into the tidy `observations.parquet` store.
4. Re-runs the per-series bake-off and re-bakes `forecasts.json`.
5. Commits both to the `data` branch — Streamlit Cloud auto-redeploys.

*Daily (22:00 UTC)* — daily price refresh, no forecast rebake:

1. Pulls **daily** front-month closes for LE / HE / GF via yfinance.
2. Commits the updated `observations.parquet` to the `data` branch.

Daily AMS LMR cash was considered and rejected: USDA's MARS API requires
eAuth Level 2 (in-person identity proofing), which is unrealistic for an
open dashboard. Public-PDF scraping is fragile and fails silently when
USDA tweaks report layouts. Daily front-month futures cover the
"what is the market doing today" need defensibly: cash settles to futures
over the contract month, and basis is what bridges the two.

**Latest forecast snapshot:** `{gen_label}`.
"""
)

st.markdown("## The three forecasting models")
st.markdown(
    """
We score three classical forecasters against each other for every series:

- **AutoARIMA** ([Nixtla statsforecast](https://github.com/Nixtla/statsforecast)).
  Auto-tuned seasonal ARIMA. Strong when the series has stable autocorrelation
  structure and modest seasonality (which describes most cattle and hog series).
- **Prophet** ([Meta Prophet](https://facebook.github.io/prophet/)).
  Decomposes the series into trend + seasonality + holidays. Best on series
  with strong seasonal patterns and tolerable trend regime shifts.
- **LightGBM** with engineered lag features
  ([microsoft/LightGBM](https://github.com/microsoft/LightGBM)). Gradient-boosted
  trees over lags (1, 2, 3, 6, 12, 24 months), rolling means, and calendar
  features. Catches nonlinearities the other two miss.

For each series, we rolling-origin cross-validate every model and pick the
**winner by lowest MAPE** (mean absolute percent error). The default
forecast shown on each Series page is the winner's prediction; you can re-run
the bake-off live from the **Advanced** expander on the Forecast page.

No deep learning, no LLMs, no fine-tuning. Classical methods only — by design,
so the model is auditable and runs on a laptop.
"""
)

st.markdown("## How to read the prediction interval")
st.markdown(
    """
Every forecast comes with an **80% prediction interval** — the shaded band
on the chart. Read it as:

> *"Under business-as-usual conditions, the price is expected to land inside
> this range about 80% of the time."*

The band widens further out because uncertainty compounds. A 1-month-ahead
forecast is much tighter than a 12-month-ahead forecast.

**Conformal calibration.** The models' own native PIs are usually too narrow.
We **calibrate them** by looking at how often the actual price landed inside
the band during cross-validation, then scaling the band up or down to hit
80% empirical coverage. We do this *per horizon step* — the calibration
factor at h=1 is usually different from the factor at h=12.

**When to *not* trust the band.** During regime changes (slaughter shocks,
inventory cliffs, macro disruptions) the residuals get heavy-tailed and the
80% band under-covers. The Q-Q plot in the Forecast page's Advanced expander
shows you where this happens for the current winner.
"""
)

hi = cache_horizon()
if hi is not None:
    cv_h, n_w, fwd_h = hi
    st.caption(
        f"Configured defaults: bake-off uses **{cv_h}-month CV horizon × "
        f"{n_w} windows**; the displayed forward forecast goes **{fwd_h} "
        f"months** ahead."
    )

st.markdown("## The v3.0 direct-market tools")
st.markdown(
    """
v3.0 reorients the app around **direct-market cattle ranchers** — farms
that raise, finish, and sell freezer beef directly to consumers. The
v2.0 commodity tools (Decide / Feedlot breakeven) remain available
under "Commodity tools (v2.0)" in the sidebar, but the headline tools
are now:

**Plan** — three tabs for three operation modes:

- *Cow-calf* — maintain a breeding herd, sell weaned calves into the
  feeder market. Pasture acres × cost/acre, hay tons/cow, vet/breeding,
  fence/labor amortized per cow; revenue is calf weight × $/cwt at
  weaning. Output: per-calf cost, per-calf margin, annual operation P&L,
  breakeven weaned-price-per-cwt.
- *Stocker* — buy weaned calves, graze, sell to a feedlot. Margin lives
  in the gap between buy and sell prices net of grass + supplement +
  interest + death loss.
- *Finish and direct (freezer beef)* — raise or buy feeders, finish on
  the farm, sell direct as freezer beef. Cost stack: feeder + pasture +
  hay + grain + vet + abattoir + cut-and-wrap. Output: cost per pound
  hanging, margin per head, and a *commodity floor sanity check* (what
  the same animal would have sold for at LE futures prices, so you can
  see your direct premium explicitly).

**Costs** — daily snapshot of feed grains (CME corn / soybean meal /
oats), feeder cattle (GF futures + ERS Oklahoma auctions), and a hay
reference band with a "enter your local $/ton" input that flows back
into the Plan page.

**Pricing** — research-derived ranges for hanging-weight pricing
(grain-finished vs grass-finished vs premium branded), share sizing
(quarter / half / whole), and individual retail cuts (ground beef,
steaks, roasts, specialty). Plus a calculator that converts your
hanging weight + $/lb into share-size revenue.

### v2.0 commodity tools (still available)

**Basis.** The Series page shows the latest cash-to-nearby-futures
basis, the 5-year mean, and a p10–p90 band. Useful if you hedge.

**Decide (commodity).** The v2.0 sell-now/hold tool for fed-cattle
operators selling into commodity markets. Combines cash + futures +
breakeven + 6-month forecast into a deterministic recommendation.

**Feedlot breakeven.** The v2.0 closed-form feedlot cost-of-production
calculator (yardage + cost of gain + interest + death loss). Different
math from the Plan page's pasture-shaped cost stack — use this one if
your operation actually looks like a feedlot.
"""
)

st.markdown("## Limits we know about")
st.markdown(
    """
- **MASE is typically 2-7 across the board.** The model errors are 2-7× the
  typical month-to-month price change. That makes the forecasts useful for
  *level and range* (and timing of broad cycles), **not** for tactical
  week-to-week selling decisions.
- **The cattle cycle is long.** Most series shift through 5-10 year regimes;
  the forecasts assume the current regime continues. Cycle peaks and troughs
  are not directly modeled.
- **Lamb data effectively stops in 2018.** The San Angelo series went dark
  in source after AMS changed reporting; the chart shows what we have.
- **Quarterly series (supply/disappearance) don't get forecasts.** They live
  on Series and Explore for context only.
- **Not financial advice.** This is a public educational dashboard. Do not
  trade on these numbers.
"""
)
