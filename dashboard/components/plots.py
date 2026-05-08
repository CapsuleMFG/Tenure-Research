"""Plotly figure builders shared between dashboard pages.

Pure functions: input is data, output is a ``plotly.graph_objects.Figure``.
No Streamlit imports — these are testable without a running app.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
import polars as pl
import scipy.stats as stats
from plotly.subplots import make_subplots


@dataclass(frozen=True)
class EventMarker:
    iso_date: str  # YYYY-MM-DD
    label: str


EVENT_MARKERS: tuple[EventMarker, ...] = (
    EventMarker("2008-09-01", "Financial crisis"),
    EventMarker("2014-11-01", "2014-15 cattle peak"),
    EventMarker("2020-04-01", "COVID slaughter shock"),
    EventMarker("2022-03-01", "Russia/Ukraine spike"),
    EventMarker("2025-06-01", "2024-25 cycle high"),
)

_MODEL_PALETTE: dict[str, str] = {
    "AutoARIMA": "#1f77b4",
    "Prophet": "#2ca02c",
    "LightGBM": "#d62728",
}


def _add_event_annotations(
    fig: go.Figure, events: Sequence[EventMarker]
) -> None:
    for ev in events:
        fig.add_vline(
            x=ev.iso_date,
            line_width=1,
            line_dash="dot",
            line_color="rgba(120,120,120,0.6)",
        )
        fig.add_annotation(
            x=ev.iso_date,
            yref="paper",
            y=1.02,
            text=ev.label,
            showarrow=False,
            font=dict(size=10, color="gray"),
            xanchor="left",
        )


def build_series_chart(
    history: pl.DataFrame,
    label: str,
    events: Sequence[EventMarker] = EVENT_MARKERS,
) -> go.Figure:
    """Single-series time-series chart with event markers.

    ``history`` must have ``period_start`` (Date) and ``value`` columns.
    """
    fig = go.Figure(
        go.Scatter(
            x=history["period_start"].to_list(),
            y=history["value"].to_list(),
            mode="lines",
            name=label,
            line=dict(color="rgba(50,50,50,0.85)", width=1.6),
        )
    )
    fig.update_layout(
        title=f"{label} — history",
        xaxis_title="Period",
        yaxis_title="Value",
        height=420,
        margin=dict(t=60, b=40),
        hovermode="x unified",
    )
    _add_event_annotations(fig, events)
    return fig


def build_yoy_chart(
    yoy: pl.DataFrame,
    label: str,
    events: Sequence[EventMarker] = EVENT_MARKERS,
) -> go.Figure:
    """Year-over-year percent change line chart with zero reference line.

    ``yoy`` must have ``period_start`` and ``yoy_pct`` columns.
    """
    fig = go.Figure(
        go.Scatter(
            x=yoy["period_start"].to_list(),
            y=yoy["yoy_pct"].to_list(),
            mode="lines",
            name=f"{label} YoY %",
            line=dict(color="#2563eb", width=1.4),
        )
    )
    fig.add_hline(y=0, line_width=1, line_color="rgba(0,0,0,0.4)")
    fig.update_layout(
        title=f"{label} — year-over-year change (%)",
        xaxis_title="Period",
        yaxis_title="YoY change, %",
        height=380,
        hovermode="x unified",
    )
    _add_event_annotations(fig, events)
    return fig


def build_cv_overlay(
    history: pl.DataFrame,
    cv_details: pl.DataFrame,
    label: str,
    horizon: int,
    n_windows: int,
) -> go.Figure:
    """Actuals + per-model CV-window forecast segments on shared axes.

    ``cv_details`` must have ``model``, ``window``, ``period_start``, ``point``
    columns (plus ``lower_80``/``upper_80``/``actual`` — unused here).
    """
    cv_min = cv_details["period_start"].min()
    history_recent = history.filter(
        pl.col("period_start") >= cv_min - _dt.timedelta(days=365 * 2)
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history_recent["period_start"].to_list(),
            y=history_recent["value"].to_list(),
            mode="lines",
            name="Actual",
            line=dict(color="rgba(50,50,50,0.85)", width=1.4),
        )
    )

    for model_name in sorted(cv_details["model"].unique().to_list()):
        color = _MODEL_PALETTE.get(model_name, "#888")
        sub = cv_details.filter(pl.col("model") == model_name).sort(
            ["window", "period_start"]
        )
        for w in sorted(sub["window"].unique().to_list()):
            wdf = sub.filter(pl.col("window") == w)
            fig.add_trace(
                go.Scatter(
                    x=wdf["period_start"].to_list(),
                    y=wdf["point"].to_list(),
                    mode="lines",
                    line=dict(color=color, width=1.4),
                    name=model_name,
                    legendgroup=model_name,
                    showlegend=(w == sub["window"].min()),
                )
            )

    fig.update_layout(
        title=(
            f"{label} — actuals vs. CV forecasts "
            f"(h={horizon}, {n_windows} windows)"
        ),
        xaxis_title="Period",
        yaxis_title="Value",
        height=480,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
    )
    return fig


def build_residual_diagnostics(
    cv_details: pl.DataFrame,
    model_name: str,
    label: str,
) -> go.Figure:
    """Three-panel diagnostics for one model's CV residuals.

    Panels: (1) residual histogram, (2) MAE by forecast horizon step,
    (3) Q-Q vs. standard normal.
    """
    cv = (
        cv_details.filter(pl.col("model") == model_name)
        .sort(["window", "period_start"])
        .with_columns(
            residual=pl.col("actual") - pl.col("point"),
            step=pl.col("period_start").rank(method="ordinal").over("window"),
        )
    )
    residuals = np.asarray(cv["residual"].to_list(), dtype=float)

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            f"Residual distribution (mean={residuals.mean():.2f}, "
            f"sd={residuals.std():.2f})",
            "MAE by forecast horizon",
            "Q-Q vs. standard normal",
        ),
    )

    fig.add_trace(
        go.Histogram(x=residuals, nbinsx=20, marker_color="rgba(31,119,180,0.7)"),
        row=1,
        col=1,
    )

    by_step = (
        cv.group_by("step")
        .agg(pl.col("residual").abs().mean().alias("mae"))
        .sort("step")
    )
    fig.add_trace(
        go.Bar(
            x=by_step["step"].to_list(),
            y=by_step["mae"].to_list(),
            marker_color="rgba(214,39,40,0.75)",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="Months ahead", row=1, col=2)
    fig.update_yaxes(title_text="Mean abs error", row=1, col=2)

    osm, osr = stats.probplot(residuals, dist="norm", fit=False)
    fig.add_trace(
        go.Scatter(
            x=osm,
            y=osr,
            mode="markers",
            marker=dict(color="rgba(44,160,44,0.8)", size=5),
            showlegend=False,
        ),
        row=1,
        col=3,
    )
    line_x = np.array([osm.min(), osm.max()])
    line_y = line_x * residuals.std() + residuals.mean()
    fig.add_trace(
        go.Scatter(
            x=line_x,
            y=line_y,
            mode="lines",
            line=dict(color="black", dash="dot", width=1),
            showlegend=False,
        ),
        row=1,
        col=3,
    )
    fig.update_xaxes(title_text="Theoretical quantile", row=1, col=3)
    fig.update_yaxes(title_text="Sample quantile", row=1, col=3)

    fig.update_layout(
        title_text=f"{label} — residual diagnostics ({model_name})",
        height=400,
        showlegend=False,
    )
    return fig


def build_forward_forecast(
    history: pl.DataFrame,
    forward: pl.DataFrame,
    model_name: str,
    label: str,
    history_tail_months: int = 60,
) -> go.Figure:
    """Headline forward-forecast chart: history tail + 80% PI band + point line."""
    tail = history.tail(history_tail_months)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=tail["period_start"].to_list(),
            y=tail["value"].to_list(),
            mode="lines",
            name="Historical",
            line=dict(color="rgba(50,50,50,0.9)", width=1.6),
        )
    )

    upper_x = forward["period_start"].to_list()
    lower_x = list(reversed(upper_x))
    pi_y = forward["upper_80"].to_list() + list(
        reversed(forward["lower_80"].to_list())
    )
    fig.add_trace(
        go.Scatter(
            x=upper_x + lower_x,
            y=pi_y,
            fill="toself",
            fillcolor="rgba(31,119,180,0.18)",
            line=dict(color="rgba(0,0,0,0)"),
            name="80% PI",
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forward["period_start"].to_list(),
            y=forward["point"].to_list(),
            mode="lines+markers",
            name=f"{model_name} forecast",
            line=dict(color="rgb(31,119,180)", width=2),
            marker=dict(size=6),
        )
    )

    last_actual = history["period_start"].max()
    fig.add_vline(
        x=str(last_actual), line_color="rgba(120,120,120,0.6)", line_dash="dot"
    )

    fig.update_layout(
        title=f"{label} — {model_name} forward forecast",
        xaxis_title="Period",
        yaxis_title="Value",
        height=460,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
    )
    return fig


__all__ = [
    "EVENT_MARKERS",
    "EventMarker",
    "build_cv_overlay",
    "build_forward_forecast",
    "build_residual_diagnostics",
    "build_series_chart",
    "build_yoy_chart",
]
