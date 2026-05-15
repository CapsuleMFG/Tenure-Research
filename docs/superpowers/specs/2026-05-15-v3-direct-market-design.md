# v3.0 "Direct-Market Ranchers" — Design

**Date:** 2026-05-15
**Status:** Approved (autonomous under `/goal`)
**Predecessors:** v1.0 (shippable), v2.0 (actually-useful for commodity producers)

> v1.0 / v2.0 was built for fed-cattle and hog operations selling into
> commodity markets. The user's actual target audience is direct-market
> ranchers — farms that raise, finish, and (often) slaughter their own
> cattle, selling freezer beef as quarters/halves/whole or as retail
> cuts. v3.0 reorients the app to that audience.

## What changes

A direct-market rancher's economics are fundamentally different from a
feedlot's:

| | Feedlot (v2.0 target) | Direct-market rancher (v3.0 target) |
|---|---|---|
| Sale | Cash market or formula contract | Direct to consumer or local butcher / co-op |
| Sale price | 5-Area / regional cash $/cwt | Their own retail price; quarter share $/lb hanging |
| Cost driver | Cost of gain (commodity feed) | Pasture (acres × stocking rate), hay, cut-and-wrap fees |
| Decision | Sell now vs hold at commodity price | What to charge, when to buy replacement stock, when to cull |
| Useful price data | 5-Area, boxed beef, LE futures | Feeder cattle (buy side), cull cow (replace), corn/SBM (supplement) |
| Risk lens | Basis to nearby futures | Direct-customer demand softness; input-cost spikes |

v3.0 keeps the v2.0 commodity tools (some users want them) but
reorients the home page, navigation, and central "decision" tool around
the direct-market workflow.

## Scope

Four new capabilities, in priority order:

1. **Direct-market economics module** (`direct_market.py`):
   pure-function breakeven math for the three modes the user runs:
   - **Cow-calf**: maintain a breeding herd; sell weaned calves
   - **Finish-and-direct**: buy or raise feeders, finish on the farm,
     sell as freezer beef
   - **Stocker** (transitional): graze weaned calves up to feedlot
     placement weight; sell to a feedlot
   Each mode has its own cost stack (pasture, hay, supplement, vet,
   fencing, fuel, abattoir fees) and sale structure.

2. **Costs / inputs page**: daily feed prices (CME corn, soybean meal,
   oats from yfinance), feeder cattle price ribbon, hay reference
   (user-entered + state extension averages), cull-cow market proxy.

3. **Freezer-beef pricing reference**: research-derived ranges for
   quarter / half / whole shares; hanging-weight pricing; retail-cut
   bundle pricing; with a calculator that converts hanging weight to
   retail yield for a typical breakdown.

4. **Plan page**: the new central tool. User picks one of three modes,
   enters their operation parameters, and gets:
   - Per-head and per-cow breakeven
   - Sale-price scenarios (commodity floor vs direct-market retail)
   - Annual operation P&L given current inputs
   - "What if" sensitivity (hay +10%, feeder calves +$20/cwt, etc.)

### Deferred to v3.1

- **FSIS plant directory** (was #5 on the original v3.0 list). The FSIS
  website is Akamai-protected and blocks our sandbox; would work in CI
  but needs verification in production. Code is structured to add this
  in v3.1 without touching the rest of v3.0.

## What doesn't change

- All of `src/usda_sandbox/` v1.0/v2.0 modules. Forecasting, basis,
  feedlot breakeven, decision tool — all kept.
- All v1.0/v2.0 data series in `observations.parquet`.
- All existing pages remain (renamed / reorganized — see nav below).
- Tests stay green.

## Information architecture (v3.0)

Sidebar reorganization:

```
PRIMARY (direct-market):
  Brief              home — direct-market headline
  Plan               the central decision tool (replaces Decide for this audience)
  Costs              daily feed + hay + feeder + cull prices
  Pricing            freezer-beef share + retail pricing reference

EXPLORE:
  Catalog            unchanged
  Series             unchanged
  Forecast           unchanged

COMMODITY (v2.0, kept):
  Decide (v2.0)      original sell-now tool for commodity producers
  Feedlot Breakeven  original feedlot calc

REFERENCE:
  Methodology
  About
```

The v2.0 Decide and Breakeven pages get parenthetical "(v2.0 / feedlot)"
labels so a direct-market user knows they're not the headline tools.

## Data layer

```
existing:
  ERS monthly cash         (cattle, hogs, lamb, boxed beef, pork cutout)
  yfinance monthly futures (LE / GF / HE continuous front-month)
  yfinance daily futures   (LE / GF / HE daily, v2.0)
  forecasts.json           (per-series 6m forecast + scoreboard)

added in v3.0 (yfinance daily, same idempotent pattern):
  ZC=F   CME Corn daily front-month
  ZM=F   CME Soybean Meal daily front-month
  ZO=F   CME Oats daily front-month
```

All new daily grain series go into the same `observations.parquet` schema
(`series_id`, `frequency="daily"`, `commodity="grain"`,
`metric="futures_price"`). No new tables, no new schemas.

## Compute layer

| Module | Purpose |
|---|---|
| `direct_market.py` | Three dataclasses for cow-calf / stocker / finish-direct inputs; `compute_*_economics()` pure functions; total operation P&L; sensitivity helpers |
| `direct_pricing.py` | Static reference: quarter/half/whole share pricing ranges from KSU/PSU/UTK extension surveys; hanging-weight → retail-yield conversion |
| `feed_costs.py` | Thin helpers: latest corn/SBM/oats from observations; simple "cost of supplement per head per day" calc |

## UI layer

### Brief (reframed)

Top strip now shows the prices that matter to *this* audience:

- **Feeder Cattle (GF)** — daily front-month + change. *"What you'd pay for replacements right now."*
- **Live Cattle (LE)** — daily front-month + change. *"Macro floor; the commodity benchmark your freezer-beef must beat."*
- **Corn (ZC)** — daily front-month + change. *"Supplement cost driver if you finish on grain."*

Below the strip: a 1-paragraph "what direct-market ranchers should be
watching" written brief, replacing the boxed-beef-and-Nebraska-steer one.

Below that: keep the existing commodity cards (still useful as context),
but **add a prominent CTA card** that says "Plan your operation →" linking
to the new Plan page.

### Plan page (new — the central tool)

Tabbed UI with three modes:

**Tab 1 — Cow-calf**
- Inputs: # cows, calving %, weaning weight, weaned calf $/cwt, pasture
  acres, $/acre/year, hay tons/cow/year, hay $/ton, vet/breeding $/cow,
  fence + labor + fuel $/cow.
- Outputs: cost per cow, cost per calf, sale revenue per calf,
  margin per calf, annual operation P&L.

**Tab 2 — Stocker** (buy weaned calf, graze to feedlot weight, sell to feedlot)
- Inputs: # placed, purchase weight + $/cwt, sale weight + $/cwt,
  days on grass, ADG, hay supplement, vet + death loss.
- Outputs: per-head breakeven, margin at current feeder $/cwt.

**Tab 3 — Finish & direct-market** (raise/buy feeders, finish on farm, sell freezer beef)
- Inputs: # head, finished hanging weight, abattoir + cut-and-wrap fees,
  feed costs, pasture/hay, your retail price per lb hanging.
- Outputs: per-head cost, hanging-weight breakeven $/lb, your direct
  retail margin, equivalent commodity floor for sanity check.

Each tab also surfaces the relevant current market signals (feeder
prices today for cow-calf and stocker tabs; LE + corn for finish tab).

### Costs page (new)

A read-only "what are my inputs doing this week" snapshot:

- **Feed grains**: latest daily close + 1-month change for corn,
  soybean meal, oats. Sparklines.
- **Feeder cattle**: today's GF front-month + the latest monthly ERS
  Oklahoma 500-550 and 750-800 feeder steer prices.
- **Hay reference**: a static table from NASS state averages with a
  user-entry override ("enter your local $/ton").
- **Cull cow proxy**: since we don't have AMS cull cow auction data,
  show boxed beef cutout trend as a proxy (cull cow prices correlate
  with cutout strength) + a sentence explaining the proxy.

### Pricing page (new)

Reference data, formatted as cards:

- **Hanging-weight share pricing**: typical $/lb hanging weight by region
  (research-derived; intentionally a range, not a single number).
- **Quarter / half / whole share**: typical total-share pricing on a
  500-700 lb hanging carcass.
- **Retail cut bundles**: ~ranges per pound for ground beef, steaks,
  roasts.
- **Pricing calculator**: enter your hanging weight + your $/lb hanging,
  see what each share size yields gross.

All numbers are explicitly labeled "research-derived ranges, late
2020s." A clear caveat: these are starting points; local market checks
required.

## Implementation order

1. Spec (this document)
2. Daily grain futures (`futures_daily.py` extended for ZC, ZM, ZO)
3. `direct_market.py` + tests
4. `direct_pricing.py` + tests
5. Plan page (Streamlit)
6. Costs page (Streamlit)
7. Pricing page (Streamlit)
8. Brief reframe
9. Navigation reorg in `streamlit_app.py`
10. Methodology + About + README updates
11. Smoke test full suite
12. Tag v3.0 + push

## Out of scope for v3.0

- FSIS plant directory (deferred to v3.1)
- User accounts / saved plans
- Email alerts on feeder price moves
- Mobile-native UI (still responsive Streamlit)
- Local hay price feeds beyond user-entered (NASS Quick Stats requires
  an API key and the value is hyperlocal anyway)
- Direct-customer CRM / order tracking (way out of scope)

## Migration / non-breaking guarantees

- All v1.0/v2.0 modules stay importable; no breaking changes.
- The v2.0 Decide page and Feedlot Breakeven page remain accessible
  with parenthetical labels so commodity users still have them.
- 183 tests still pass; new tests added for the new modules.
