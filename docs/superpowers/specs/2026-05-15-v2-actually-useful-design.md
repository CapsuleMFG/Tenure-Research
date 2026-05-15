# v2.0 "Actually Useful" — Design

**Date:** 2026-05-15
**Status:** Approved (autonomous under `/goal`)
**Predecessor:** [2026-05-15-shippable-app-design.md](2026-05-15-shippable-app-design.md)

> v1.0 ships a presentable forecasting dashboard, but its end-user value
> is thin — a monthly ERS series with a 6-month forecast at MASE 2-7 is
> educational, not actionable. v2.0 closes the gap by adding the layers
> a producer actually uses: daily price refresh, basis, breakeven, and a
> "sell now / hold" decision tool.

## Scope

Four new capabilities, in priority order:

1. **Daily futures refresh** — Live Cattle (LE=F), Feeder Cattle (GF=F),
   Lean Hogs (HE=F) front-month *daily* closes via yfinance. Real producers
   watch these daily; they incorporate AMS daily cash + the market's
   expectation of next reports.

2. **Basis-to-local-market layer** — basis = local cash − nearby futures.
   Computed at whatever cadence both sides exist (daily where possible,
   monthly fallback). Shown as a card on the Series page and incorporated
   into the decision tool.

3. **Breakeven calculator** — cattle-feeder economics: feeder cost ($/cwt)
   × purchase weight + cost-of-gain × weight gained + yardage × days +
   interest, all expressed as a $/cwt breakeven on the finished weight.
   Sensible KSU defaults; inputs are sliders.

4. **Sell-now decision tool** — synthesizes the above into a recommendation
   for one user-described holding ("I have N head, X weight, region Y, my
   breakeven is Z"). Outputs: today's expected price, the basis-adjusted
   margin, the 6-month forecast margin range, and a recommended action
   (sell / hold / hedge), with the reasoning shown.

### Considered and explicitly cut

5. ~~**AMS LMR direct integration**~~ — initially scaffolded as an
   `AMS_API_KEY` opt-in. Removed mid-implementation when we confirmed
   USDA MARS API requires **eAuth Level 2** (in-person identity proofing
   at a USDA Service Center), not the casual email registration we'd
   assumed. The public-PDF scraping alternative was rejected as too
   fragile (USDA tweaks report layouts every few quarters; scrapes break
   silently). Daily front-month futures cover the daily-price layer
   defensibly — see README "Why daily futures and not daily AMS cash?"
   A future contributor with a producer/extension role and the time to
   maintain a PDF pipeline could revisit this in v3.0.

## Architecture

### Data layer

```
existing:
  data/raw/*.xlsx                  USDA ERS monthly cash + WASDE
  data/raw/futures_continuous/     monthly yfinance front-month
  data/clean/observations.parquet  tidy long-format store

added:
  data/raw/futures_daily/          NEW: daily yfinance front-month
```

Schema unchanged — daily futures appear in `observations.parquet` as new
`series_id`s with `frequency="daily"`. The store's existing schema already
supports this (the v0.2c cleaner just happens to only emit monthly).

### Compute layer

New modules in `src/usda_sandbox/`:

| Module | Purpose |
|---|---|
| `futures_daily.py` | yfinance daily fetch for LE=F, GF=F, HE=F; same idempotent manifest pattern as `futures_continuous.py` |
| `basis.py` | `compute_basis(cash_series_id, futures_series_id, obs_path)` → polars DF of basis values; `latest_basis()`, `basis_stats()` helpers |
| `breakeven.py` | Pure-functions calc: `breakeven_per_cwt(...)`, `placement_economics(...)` |
| `decision.py` | Synthesizes price + basis + breakeven + cached forecast → `Recommendation` (action + reasoning) |

All new modules expose pure functions with explicit inputs; the dashboard
imports them but they don't depend on Streamlit.

### UI layer

New / changed pages:

| Page | Change |
|---|---|
| **Brief** (home) | Show today's futures front (daily refresh) alongside latest monthly cash; brief includes "future contract X is up Y% week-over-week" |
| **Series** | Add a "Basis" card showing current basis + historical band |
| **Forecast** | Add a one-line "vs today's futures" caption next to the forecast |
| **Decide** *(new)* | The sell-now tool; main v2.0 deliverable |
| **Breakeven** *(new)* | Breakeven calculator; feeds Decide |
| **Methodology** | New section on basis, breakeven, and decision logic |
| **About** | Bump version to 2.0, add disclaimer about decision tool |

### Deploy layer

| Change | Why |
|---|---|
| New workflow `.github/workflows/refresh-daily.yml` (daily cron, 22:00 UTC) | Pulls daily futures via yfinance; pushes to `data` branch |
| Existing weekly workflow keeps rebaking monthly forecasts | Forecasts stay monthly; only the daily layer changes |

## Decision-tool logic (the v2.0 product)

Inputs (user supplies on the Decide page):
- **Commodity:** Fed Cattle / Feeder Cattle / Hogs
- **Holding size:** head count (optional, just for $/total)
- **Carcass weight class** (Fed Cattle): light/medium/heavy
- **Region:** TX/OK/NM, Nebraska, National, etc. (drives which cash series is used)
- **Breakeven $/cwt:** user-entered (or pulled from Breakeven page state)

Computed values:
- `latest_cash` = today's regional cash (latest non-null value of the ERS monthly series)
- `latest_futures` = today's nearby futures close
- `basis` = `latest_cash` − `latest_futures`
- `forecast_6m_point`, `forecast_6m_pi_lo`, `forecast_6m_pi_hi` (from precompute cache)
- `margin_today` = `latest_cash` − `breakeven`
- `margin_6m_point` = `forecast_6m_point` − `breakeven`
- `margin_6m_lo` = `forecast_6m_pi_lo` − `breakeven`

Recommendation logic (deterministic, transparent):
1. If `margin_today` ≥ 0 AND `margin_6m_point` < `margin_today` − 10% buffer
   → **Sell now.** "Margin already strong; forecast suggests it's likely to weaken."
2. If `margin_today` < 0 AND `margin_6m_lo` > 0
   → **Hold.** "Loss today but 6-month range is above breakeven across most of the PI."
3. If `margin_today` ≥ 0 AND `margin_6m_lo` < 0
   → **Sell now (downside risk).** "Today's margin is real cash; 6-month PI dips below breakeven."
4. If `margin_today` < 0 AND `margin_6m_point` < 0
   → **Hedge or hold and reassess.** "No good window in the forecast horizon."
5. Otherwise → **Hold.** "Margin neutral; forecast suggests modest improvement."

Each recommendation includes the inputs that drove it ("$238 cash − $192 breakeven
= $46/cwt margin today; 6-month forecast $245 means $53/cwt margin then; sell now
captures most of the upside without taking 6 more months of yardage/feed risk").

This logic is **deterministic and auditable** — no LLM, no opaque scoring.

## What's out of scope for v2.0

- Mobile-native UI (still responsive Streamlit)
- Email/SMS alerts
- User accounts / saved holdings
- Multi-tenant: still single-tenant public dashboard
- Real-time intraday data (we only refresh once daily)
- Position sizing / hedging math beyond "consider hedging"
- Custom local cash markets beyond what ERS publishes

## Migration / non-breaking guarantees

- All v1.0 series IDs unchanged.
- `observations.parquet` schema unchanged.
- v1.0 pages (Brief, Catalog, Series, Forecast, Methodology, About) remain.
- Tests stay green throughout.

## Implementation order

1. `futures_daily.py` + ingest into `observations.parquet` (`frequency="daily"`)
2. `basis.py` + tests
3. `breakeven.py` + tests
4. `decision.py` + tests
5. Decide page (`dashboard/pages/6_Decide.py`)
6. Breakeven page (`dashboard/pages/7_Breakeven.py`)
7. Brief page update (today's futures)
8. Series page basis card
9. `.github/workflows/refresh-daily.yml`
10. Methodology + About + README updates
11. Tag v2.0

## Caveats

This is still an educational app. The Decide tool's recommendation is a
**reasoning aid**, not a trade signal. The Methodology page makes this
explicit; the Decide page repeats it inline below every recommendation.
