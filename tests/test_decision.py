"""Tests for usda_sandbox.decision — sell-now / hold rules."""

from __future__ import annotations

from usda_sandbox.decision import DecisionInputs, recommend


def _base(**kwargs) -> DecisionInputs:
    defaults = dict(
        cash_now=220.0,
        futures_now=215.0,
        basis_now=5.0,
        breakeven_per_cwt=200.0,
        forecast_point=225.0,
        forecast_pi_lo=205.0,
        forecast_pi_hi=250.0,
        unit="USD/cwt",
    )
    defaults.update(kwargs)
    return DecisionInputs(**defaults)


def test_rule1_sell_now_when_margin_today_strong() -> None:
    # Today margin = 40, 6m margin = 42 → 42 < 40 * 1.10 = 44 → sell now
    rec = recommend(_base(cash_now=240.0, forecast_point=242.0,
                          forecast_pi_lo=220.0, forecast_pi_hi=265.0))
    assert rec.action == "sell_now"
    assert rec.margin_today == 40.0
    assert "capture" in rec.headline.lower()


def test_rule2_sell_with_downside_when_pi_lo_below_breakeven() -> None:
    # Today margin = 10, forecast_point=220 → 6m margin 20 > 10*1.1
    # but PI lo 180 → margin_lo = -20 < 0
    rec = recommend(_base(cash_now=210.0, forecast_point=220.0,
                          forecast_pi_lo=180.0, forecast_pi_hi=260.0))
    assert rec.action == "sell_with_downside"
    assert rec.margin_today == 10.0
    assert rec.margin_6m_lo == -20.0


def test_rule3_hold_when_loss_today_but_full_pi_clears() -> None:
    rec = recommend(_base(cash_now=190.0, forecast_point=240.0,
                          forecast_pi_lo=210.0, forecast_pi_hi=275.0))
    assert rec.action == "hold"
    assert rec.margin_today == -10.0
    assert rec.margin_6m_lo == 10.0


def test_rule4_hedge_or_hold_when_both_today_and_6m_loss() -> None:
    rec = recommend(_base(cash_now=180.0, forecast_point=190.0,
                          forecast_pi_lo=170.0, forecast_pi_hi=210.0))
    assert rec.action == "hedge_or_hold"
    assert rec.margin_today < 0
    assert rec.margin_6m < 0


def test_rule5_neutral_fallback() -> None:
    # Today margin = 20, 6m margin = 50 (clearly > 20*1.1), PI lo just above BE
    rec = recommend(_base(cash_now=220.0, forecast_point=250.0,
                          forecast_pi_lo=205.0, forecast_pi_hi=290.0))
    assert rec.action == "neutral"
    assert rec.margin_today == 20.0
    assert rec.margin_6m == 50.0


def test_reasoning_cites_actual_numbers() -> None:
    rec = recommend(_base(cash_now=240.0, breakeven_per_cwt=200.0,
                          forecast_point=242.0,
                          forecast_pi_lo=220.0, forecast_pi_hi=265.0))
    # Reasoning should mention cash, breakeven, forecast values
    assert "$240" in rec.reasoning
    assert "$200" in rec.reasoning
    assert "$242" in rec.reasoning


def test_all_actions_produce_non_empty_strings() -> None:
    """Headline and reasoning must never be empty for any of the five rules."""
    scenarios = [
        _base(cash_now=240.0, forecast_point=242.0,
              forecast_pi_lo=220.0, forecast_pi_hi=265.0),  # rule 1
        _base(cash_now=210.0, forecast_point=220.0,
              forecast_pi_lo=180.0, forecast_pi_hi=260.0),  # rule 2
        _base(cash_now=190.0, forecast_point=240.0,
              forecast_pi_lo=210.0, forecast_pi_hi=275.0),  # rule 3
        _base(cash_now=180.0, forecast_point=190.0,
              forecast_pi_lo=170.0, forecast_pi_hi=210.0),  # rule 4
        _base(cash_now=220.0, forecast_point=250.0,
              forecast_pi_lo=205.0, forecast_pi_hi=290.0),  # rule 5
    ]
    seen_actions = set()
    for d in scenarios:
        rec = recommend(d)
        assert rec.headline
        assert rec.reasoning
        seen_actions.add(rec.action)
    # We should have hit every distinct action.
    assert seen_actions == {
        "sell_now", "sell_with_downside", "hold", "hedge_or_hold", "neutral"
    }
