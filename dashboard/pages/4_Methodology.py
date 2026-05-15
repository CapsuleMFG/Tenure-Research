"""Methodology — plain-English explainer for data, models, and uncertainty."""

from __future__ import annotations

import streamlit as st
from components.cache import cache_generated_at, cache_horizon
from components.sidebar import render_sidebar
from components.theme import INK_SOFT, inject_global_css

inject_global_css()
render_sidebar(persistent_picker=False)

st.markdown("# How LivestockBrief works")
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
A GitHub Actions cron runs once a week (Sunday night UTC):

1. Discovers and downloads any updated XLSX/ZIP files from the ERS product page.
2. Pulls continuous front-month futures for LE / HE / GF via yfinance.
3. Cleans every catalog series into the tidy `observations.parquet` store.
4. Re-runs the per-series bake-off and re-bakes `forecasts.json`.
5. Commits both to the `data` branch — Streamlit Cloud auto-redeploys.

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
