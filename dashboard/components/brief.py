"""Plain-English market brief composition + commodity card rendering.

The brief is *deterministic templated text*, not LLM-generated — predictable,
fast, and faithful to the underlying numbers. It reads from
``forecasts.json`` produced by :mod:`usda_sandbox.precompute`.

Two public surfaces:

* :func:`compose_brief` — returns a one-paragraph HTML string per series.
* :func:`render_commodity_card` — emits Streamlit markup for one card.

A small inline sparkline is rendered via Plotly; everything else is plain
HTML/CSS so the cards remain fast and styleable.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass

import plotly.graph_objects as go
import streamlit as st

from .theme import ACCENT, INK_SOFT

_MONTH_LABEL = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}
_MONTH_SHORT = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}


@dataclass(frozen=True)
class _DeltaParts:
    direction: str         # "up", "down", "flat"
    pct_abs: float | None  # absolute %, e.g. 2.4
    badge_html: str        # ready-to-paste HTML
    sentence: str          # ready-to-paste English


def _safe_pct(curr: float | None, prior: float | None) -> float | None:
    if curr is None or prior is None or prior == 0:
        return None
    return (curr / prior - 1.0) * 100.0


def _delta(curr: float | None, prior: float | None, *, label: str) -> _DeltaParts:
    pct = _safe_pct(curr, prior)
    if pct is None:
        return _DeltaParts(
            direction="flat",
            pct_abs=None,
            badge_html=f"<span style='color:{INK_SOFT}'>{label}: n/a</span>",
            sentence=f"{label} change unavailable",
        )
    if pct >= 0.5:
        direction = "up"
        color_cls = "lb-up"
        word = "up"
    elif pct <= -0.5:
        direction = "down"
        color_cls = "lb-down"
        word = "down"
    else:
        direction = "flat"
        color_cls = ""
        word = "essentially flat"
    abs_pct = abs(pct)
    arrow = "▲" if direction == "up" else ("▼" if direction == "down" else "▬")
    badge = (
        f"<span class='{color_cls}'>{arrow} {abs_pct:.1f}%</span>"
        f" <span style='color:{INK_SOFT}'>{label}</span>"
    )
    sentence = (
        f"{word} {abs_pct:.1f}%" if direction != "flat" else "essentially flat"
    )
    return _DeltaParts(direction, abs_pct, badge, sentence)


def _format_money(value: float, unit: str) -> str:
    """Format a price magnitude; unit is rendered separately by callers."""
    _ = unit  # accepted for API symmetry; unit comes in via display_unit
    if abs(value) >= 1000:
        return f"${value:,.0f}"
    if abs(value) >= 100:
        return f"${value:,.1f}"
    return f"${value:,.2f}"


def display_unit(unit: str) -> str:
    """Short-form unit string suitable for appending to a dollar price.

    The catalog stores ``USD/cwt``, ``USD/lb``, etc. — but in UI the
    dollar sign already implies USD, so the display form drops the leading
    ``USD/``. Non-USD units pass through unchanged.
    """
    if not unit:
        return ""
    if unit.startswith("USD/"):
        return unit[len("USD/"):]
    return unit


def _parse_iso_date(s: str) -> _dt.date:
    return _dt.date.fromisoformat(s)


def _month_label(d: _dt.date) -> str:
    return f"{_MONTH_LABEL[d.month]} {d.year}"


def _forecast_summary(entry: dict) -> dict | None:
    """Derive end-of-forward forecast values + horizon length."""
    forward = entry.get("forward") or []
    if not forward:
        return None
    last = forward[-1]
    end_date = _parse_iso_date(last["period_start"])
    return {
        "horizon_months": len(forward),
        "end_point": float(last["point"]),
        "end_lower": float(last["lower_80"]),
        "end_upper": float(last["upper_80"]),
        "end_date": end_date,
    }


def compose_brief(entry: dict) -> str:
    """Return a 2-3 sentence HTML paragraph summarizing one series.

    The function expects a single entry from ``forecasts.json::by_series``.
    Returns an empty string if the entry is too incomplete to summarize
    (callers should fall back to a placeholder card).
    """
    latest = entry.get("latest_actual")
    if latest is None:
        return ""
    prior_month = entry.get("prior_month_actual")
    prior_year = entry.get("prior_year_actual")

    latest_date = _parse_iso_date(latest["period_start"])
    latest_value = float(latest["value"])
    unit = entry["unit"]
    short_unit = display_unit(unit)

    mom = _delta(latest_value, prior_month["value"] if prior_month else None, label="MoM")
    yoy = _delta(latest_value, prior_year["value"] if prior_year else None, label="YoY")

    series_name = entry["series_name"]
    money = _format_money(latest_value, unit)

    fc = _forecast_summary(entry)
    if fc is None:
        return (
            f"<span><em>{series_name}</em> closed "
            f"{_month_label(latest_date)} at <strong>{money}/{short_unit}</strong>, "
            f"{mom.sentence} MoM and {yoy.sentence} YoY. Forecast snapshot is "
            f"regenerating.</span>"
        )

    winner = entry.get("winner_model", "the winning model")
    end_money = _format_money(fc["end_point"], unit)
    pi_lo = _format_money(fc["end_lower"], unit)
    pi_hi = _format_money(fc["end_upper"], unit)
    end_label = f"{_MONTH_SHORT[fc['end_date'].month]} '{str(fc['end_date'].year)[-2:]}"

    return (
        f"<em>{series_name}</em> closed {_month_label(latest_date)} at "
        f"<strong>{money}/{short_unit}</strong>, {mom.sentence} month-over-month and "
        f"{yoy.sentence} year-over-year. Our {fc['horizon_months']}-month "
        f"{winner} forecast expects <strong>{end_money}/{short_unit}</strong> by "
        f"{end_label}, with 80% confidence the price lands between "
        f"{pi_lo} and {pi_hi}."
    )


def _sparkline_figure(points: list[dict], unit: str) -> go.Figure:
    if not points:
        return go.Figure()
    xs = [p["period_start"] for p in points]
    ys = [p["value"] for p in points]
    last_y = ys[-1]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color=ACCENT, width=1.6),
            hovertemplate="%{x|%b %Y}<br>$%{y:,.2f} " + unit + "<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[xs[-1]],
            y=[last_y],
            mode="markers",
            marker=dict(size=5, color=ACCENT),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.update_layout(
        height=70,
        margin=dict(l=0, r=0, t=4, b=4),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        hovermode="x",
    )
    return fig


def render_commodity_card(entry: dict, *, key_prefix: str = "card") -> None:
    """Render a single commodity-card block in Streamlit's current column.

    The card shows: eyebrow (commodity), title (series name), latest price,
    MoM/YoY badges, a 24-month sparkline, and a forecast hint. The whole
    card is wrapped in a button-styled link to the Series page via
    ``st.session_state["series_id"]``.
    """
    latest = entry.get("latest_actual")
    if latest is None:
        st.markdown(
            "<div class='lb-card'>"
            f"<div class='lb-card-eyebrow'>{entry.get('commodity', '').upper()}</div>"
            f"<div class='lb-card-title'>{entry['series_name']}</div>"
            f"<div style='color:{INK_SOFT};font-size:0.85rem;margin-top:0.6rem;'>"
            "Data temporarily unavailable.</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    latest_value = float(latest["value"])
    unit = entry["unit"]
    short_unit = display_unit(unit)
    prior_month = entry.get("prior_month_actual")
    prior_year = entry.get("prior_year_actual")
    mom = _delta(latest_value, prior_month["value"] if prior_month else None, label="MoM")
    yoy = _delta(latest_value, prior_year["value"] if prior_year else None, label="YoY")

    fc = _forecast_summary(entry)
    fcst_html = ""
    if fc is not None:
        end_money = _format_money(fc["end_point"], unit)
        end_label = f"{_MONTH_SHORT[fc['end_date'].month]} '{str(fc['end_date'].year)[-2:]}"
        pi_half = (fc["end_upper"] - fc["end_lower"]) / 2.0
        fcst_html = (
            "<div class='lb-card-fcst'>"
            "<div class='lb-card-fcst-label'>"
            f"{fc['horizon_months']}-month forecast"
            "</div>"
            f"→ <strong>{end_money}/{short_unit}</strong> by {end_label}"
            f" <span style='color:{INK_SOFT}'>(±{_format_money(pi_half, unit)})</span>"
            "</div>"
        )

    price_html = (
        f"<div class='lb-card-price'>{_format_money(latest_value, unit)}"
        f"<span class='lb-card-unit'>/{short_unit}</span></div>"
    )

    st.markdown(
        "<div class='lb-card'>"
        f"<div class='lb-card-eyebrow'>{entry.get('commodity', '').upper()}</div>"
        f"<div class='lb-card-title'>{entry['series_name']}</div>"
        f"{price_html}"
        f"<div class='lb-card-deltas'>{mom.badge_html} &nbsp; {yoy.badge_html}</div>",
        unsafe_allow_html=True,
    )
    fig = _sparkline_figure(entry.get("sparkline", []), unit)
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False},
        key=f"{key_prefix}_spark",
    )
    if fcst_html:
        st.markdown(fcst_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


__all__ = [
    "compose_brief",
    "display_unit",
    "render_commodity_card",
]
