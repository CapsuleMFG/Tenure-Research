"""Tests for usda_sandbox.direct_pricing — freezer-beef pricing reference + yield."""

from __future__ import annotations

import pytest

from usda_sandbox.direct_pricing import (
    REFERENCE_HANGING_PRICING,
    REFERENCE_RETAIL_BUNDLE,
    REFERENCE_SHARE_PRICING,
    expected_retail_yield_lbs,
    value_share,
)


def test_hanging_pricing_categories_present() -> None:
    assert {"grain_finished", "grass_finished", "premium_branded"} <= set(
        REFERENCE_HANGING_PRICING
    )
    for cat, rng in REFERENCE_HANGING_PRICING.items():
        assert rng.low < rng.mid < rng.high, f"{cat}: range not ordered"
        assert rng.unit == "$/lb hanging"


def test_share_sizes_cover_quarter_half_whole() -> None:
    names = {s.name for s in REFERENCE_SHARE_PRICING}
    assert names == {"Quarter", "Half", "Whole"}
    fractions = sorted(s.fraction for s in REFERENCE_SHARE_PRICING)
    assert fractions == [0.25, 0.50, 1.00]
    for s in REFERENCE_SHARE_PRICING:
        assert s.typical_hanging_lbs[0] < s.typical_hanging_lbs[1]
        assert s.typical_retail_lbs[0] < s.typical_retail_lbs[1]
        # Retail < hanging within each share (cut-and-wrap losses)
        assert s.typical_retail_lbs[0] < s.typical_hanging_lbs[0]


def test_retail_bundle_categories_present() -> None:
    assert {"ground_beef", "steaks", "roasts", "specialty"} <= set(
        REFERENCE_RETAIL_BUNDLE
    )


def test_expected_retail_yield_default() -> None:
    # 830 lb hanging × 0.68 default ≈ 564 lb retail
    out = expected_retail_yield_lbs(830.0)
    assert 560 < out < 568


def test_expected_retail_yield_custom_pct() -> None:
    out = expected_retail_yield_lbs(800.0, retail_yield_pct=0.70)
    assert out == pytest.approx(560.0)


def test_expected_retail_yield_invalid_pct_raises() -> None:
    with pytest.raises(ValueError, match="retail_yield_pct"):
        expected_retail_yield_lbs(800.0, retail_yield_pct=0.40)


def test_value_share_quarter() -> None:
    share_hang, total = value_share(800.0, 0.25, 6.50)
    assert share_hang == pytest.approx(200.0)
    assert total == pytest.approx(1300.0)


def test_value_share_invalid_fraction_raises() -> None:
    with pytest.raises(ValueError, match="fraction"):
        value_share(800.0, 0.0, 6.50)
    with pytest.raises(ValueError, match="fraction"):
        value_share(800.0, 1.5, 6.50)
