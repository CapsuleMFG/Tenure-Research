"""Sell-now / hold decision synthesis.

Combines: today's cash price, futures price, basis, breakeven, and the
cached 6-month forecast — into a deterministic recommendation.

Logic intentionally simple and transparent: every recommendation comes
with the inputs that drove it and the rule that fired. No LLM, no opaque
score. The Methodology page documents the full rule set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = [
    "DecisionInputs",
    "Recommendation",
    "recommend",
]


Action = Literal["sell_now", "hold", "sell_with_downside", "hedge_or_hold", "neutral"]


@dataclass(frozen=True)
class DecisionInputs:
    """All numbers needed to make one sell-now/hold decision.

    Attributes
    ----------
    cash_now : float
        Today's regional cash price per cwt.
    futures_now : float
        Today's nearby futures price per cwt.
    basis_now : float
        ``cash_now − futures_now``. Stored explicitly so callers don't
        have to subtract twice.
    breakeven_per_cwt : float
        Producer's per-cwt breakeven (from the Breakeven calc or user input).
    forecast_point : float
        6-month forecast point estimate per cwt.
    forecast_pi_lo : float
        80% PI lower bound per cwt.
    forecast_pi_hi : float
        80% PI upper bound per cwt.
    unit : str
        Display unit, defaults to "USD/cwt".
    """

    cash_now: float
    futures_now: float
    basis_now: float
    breakeven_per_cwt: float
    forecast_point: float
    forecast_pi_lo: float
    forecast_pi_hi: float
    unit: str = "USD/cwt"


@dataclass(frozen=True)
class Recommendation:
    action: Action
    headline: str        # one-line plain English
    reasoning: str       # 2-4 sentence rationale citing the actual numbers
    margin_today: float  # cash_now - breakeven
    margin_6m: float     # forecast_point - breakeven
    margin_6m_lo: float  # forecast_pi_lo - breakeven
    margin_6m_hi: float  # forecast_pi_hi - breakeven


# The buffer below which a future margin is "not meaningfully better than today".
# 10% is intentionally generous; over 6 months you take on real feeding/yardage
# risk for that nominal upside, so the future has to clearly beat today.
_MEANINGFUL_BUFFER_PCT = 0.10


def _fmt(v: float, unit: str = "USD/cwt") -> str:
    return f"${v:,.2f}/{unit}" if unit else f"${v:,.2f}"


def recommend(inputs: DecisionInputs) -> Recommendation:
    """Return a deterministic recommendation from price + breakeven + forecast.

    Rules (first match wins):

    1. **Sell now, capture upside.** ``margin_today ≥ 0`` and the forecast
       point margin is not meaningfully higher than today's
       (``margin_6m < margin_today × (1 + buffer)``).
       Translation: take the cash now; the forecast doesn't see clearly better.

    2. **Sell now, downside risk.** ``margin_today ≥ 0`` but
       ``margin_6m_lo < 0``. The bottom of the PI dips below breakeven, so
       holding bets on the central case while taking real loss risk.

    3. **Hold.** ``margin_today < 0`` and ``margin_6m_lo ≥ 0``. Today's a
       loss, but the 6-month PI is entirely above breakeven.

    4. **Hedge or hold and reassess.** ``margin_today < 0`` and
       ``margin_6m_point < 0``. No good window — locking in via futures
       hedge is worth considering, or hold and reassess.

    5. **Neutral / wait.** Anything else. Margins are positive both today
       and forward, neither dominates; a small wait is reasonable.
    """
    margin_today = inputs.cash_now - inputs.breakeven_per_cwt
    margin_6m = inputs.forecast_point - inputs.breakeven_per_cwt
    margin_6m_lo = inputs.forecast_pi_lo - inputs.breakeven_per_cwt
    margin_6m_hi = inputs.forecast_pi_hi - inputs.breakeven_per_cwt
    unit = inputs.unit

    common_inputs = (
        f"Cash today: {_fmt(inputs.cash_now, unit)}. "
        f"Breakeven: {_fmt(inputs.breakeven_per_cwt, unit)}. "
        f"6-month forecast: {_fmt(inputs.forecast_point, unit)} "
        f"(80% PI {_fmt(inputs.forecast_pi_lo, unit)} to "
        f"{_fmt(inputs.forecast_pi_hi, unit)})."
    )

    # Rule 1: today margin already strong, forecast doesn't beat it
    if margin_today >= 0 and margin_6m < margin_today * (1.0 + _MEANINGFUL_BUFFER_PCT):
        return Recommendation(
            action="sell_now",
            headline="Sell now — capture today's margin.",
            reasoning=(
                f"{common_inputs} Margin today is "
                f"{_fmt(margin_today, unit)}; the 6-month point forecast "
                f"projects {_fmt(margin_6m, unit)}, which is not meaningfully "
                f"better. Holding adds 6 months of yardage, feed, and price "
                f"risk for limited expected gain."
            ),
            margin_today=margin_today,
            margin_6m=margin_6m,
            margin_6m_lo=margin_6m_lo,
            margin_6m_hi=margin_6m_hi,
        )

    # Rule 2: margin today is positive but downside risk dips below breakeven
    if margin_today >= 0 and margin_6m_lo < 0:
        return Recommendation(
            action="sell_with_downside",
            headline="Sell now — downside risk dominates.",
            reasoning=(
                f"{common_inputs} Today's margin is {_fmt(margin_today, unit)} "
                f"(real cash). The 80% PI lower bound at 6 months is "
                f"{_fmt(inputs.forecast_pi_lo, unit)} — below breakeven "
                f"({_fmt(margin_6m_lo, unit)} margin in the bad scenario). "
                f"Locking in today removes the meaningful loss risk."
            ),
            margin_today=margin_today,
            margin_6m=margin_6m,
            margin_6m_lo=margin_6m_lo,
            margin_6m_hi=margin_6m_hi,
        )

    # Rule 3: today is a loss but full PI is above breakeven
    if margin_today < 0 and margin_6m_lo >= 0:
        return Recommendation(
            action="hold",
            headline="Hold — the 6-month range clears breakeven.",
            reasoning=(
                f"{common_inputs} Selling today would lose "
                f"{_fmt(abs(margin_today), unit)}. The 6-month PI is entirely "
                f"above breakeven (lower bound margin "
                f"{_fmt(margin_6m_lo, unit)}). Waiting is favored if your "
                f"working capital and pen space allow."
            ),
            margin_today=margin_today,
            margin_6m=margin_6m,
            margin_6m_lo=margin_6m_lo,
            margin_6m_hi=margin_6m_hi,
        )

    # Rule 4: today and 6m point both below breakeven
    if margin_today < 0 and margin_6m < 0:
        return Recommendation(
            action="hedge_or_hold",
            headline="Hedge or hold and reassess — no clear window.",
            reasoning=(
                f"{common_inputs} Both today's cash and the 6-month point "
                f"forecast are below breakeven. Consider hedging at "
                f"{_fmt(inputs.futures_now, unit)} (basis "
                f"{_fmt(inputs.basis_now, unit)}) to cap downside, or hold "
                f"and reassess in 30 days when more data points in."
            ),
            margin_today=margin_today,
            margin_6m=margin_6m,
            margin_6m_lo=margin_6m_lo,
            margin_6m_hi=margin_6m_hi,
        )

    # Rule 5: neutral fallback
    return Recommendation(
        action="neutral",
        headline="Hold or wait — both margins positive, neither dominates.",
        reasoning=(
            f"{common_inputs} Margin today is {_fmt(margin_today, unit)}; "
            f"6-month margin is {_fmt(margin_6m, unit)}. The forecast offers "
            f"some upside but doesn't dominate today's price, and downside "
            f"risk doesn't dip below breakeven. A short wait of 30-60 days "
            f"is reasonable to let the market clarify."
        ),
        margin_today=margin_today,
        margin_6m=margin_6m,
        margin_6m_lo=margin_6m_lo,
        margin_6m_hi=margin_6m_hi,
    )
