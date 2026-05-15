"""AMS LMR (Livestock Mandatory Reporting) — optional daily cash ingest.

The MARS API at ``marsapi.ams.usda.gov`` exposes structured JSON for AMS
LMR reports, but requires a free API key (register at mars.ams.usda.gov).
This module is a thin client: when ``AMS_API_KEY`` is set in the
environment, it pulls a small set of daily summary reports and writes
them into ``observations.parquet`` as daily series. When the key is
absent, every function returns without effect — the app continues to run
on monthly ERS data and daily futures only.

This is intentionally minimal scaffolding for v2.0. Expanding to more
reports + regional coverage is a v2.1 task once we have a real key in
hand and can validate the schema for each report slug.

References:
- Help: https://marsapi.ams.usda.gov/services/help
- v1.2 reports endpoint: ``/services/v1.2/reports/{slug}?api_key={key}``
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "AMS_DAILY_REPORTS",
    "AMSReport",
    "is_enabled",
    "sync_ams_daily",
]


@dataclass(frozen=True)
class AMSReport:
    """Metadata for one AMS LMR report we know how to ingest."""

    slug: str          # MARS report slug, e.g. "LM_CT150"
    series_id: str     # observations.parquet series_id
    series_name: str   # human label
    commodity: str
    metric: str        # "price" | "wholesale_price"
    unit: str          # e.g. "USD/cwt"
    notes: str


# Curated set of daily/weekly summary reports keyed for v2.0. These are the
# ones a producer actually watches. Adding more (regional formula prices,
# slaughter weights, byproduct values) is straightforward once a key is
# available to test against.
AMS_DAILY_REPORTS: tuple[AMSReport, ...] = (
    AMSReport(
        slug="LM_CT155",
        series_id="ams_ct_5area_weekly",
        series_name="5-Area Weekly Weighted Avg Direct Slaughter Cattle",
        commodity="cattle",
        metric="price",
        unit="USD/cwt",
        notes=(
            "AMS LMR weekly 5-Area direct-trade fed cattle weighted average "
            "(replaces the monthly ERS Choice steer series when AMS access "
            "is enabled). Reported every Monday for the prior week."
        ),
    ),
    AMSReport(
        slug="LM_XB459",
        series_id="ams_xb_boxed_beef_daily",
        series_name="National Daily Boxed Beef Cutout (Choice)",
        commodity="cattle",
        metric="wholesale_price",
        unit="USD/cwt",
        notes=(
            "AMS LMR national daily boxed-beef cutout for Choice grade. "
            "Updated each afternoon; the most-watched fresh beef "
            "wholesale benchmark."
        ),
    ),
    AMSReport(
        slug="LM_HG200",
        series_id="ams_hg_natdaily_hog_summary",
        series_name="National Daily Direct Hog Summary",
        commodity="hogs",
        metric="price",
        unit="USD/cwt",
        notes=(
            "AMS LMR national daily hog summary (51-52% lean, base price). "
            "Updated each afternoon; the daily companion to the ERS "
            "monthly hog series."
        ),
    ),
    AMSReport(
        slug="LM_PK602",
        series_id="ams_pk_cutout_daily",
        series_name="National Daily Pork Cutout",
        commodity="hogs",
        metric="wholesale_price",
        unit="USD/cwt",
        notes=(
            "AMS LMR national daily pork carcass cutout. Wholesale pork "
            "benchmark, daily."
        ),
    ),
)


def is_enabled() -> bool:
    """Return True if an ``AMS_API_KEY`` is present in the environment."""
    return bool(os.environ.get("AMS_API_KEY", "").strip())


def sync_ams_daily(
    *,
    obs_path: Path | str = "data/clean/observations.parquet",
    raw_dir: Path | str = "data/raw/ams_lmr",
    reports: Sequence[AMSReport] = AMS_DAILY_REPORTS,
    api_key: str | None = None,
) -> dict[str, str]:
    """If ``AMS_API_KEY`` is set, pull each report and merge into observations.

    Returns a status map ``{slug: "ok" | "skipped" | "failed: <reason>"}``.

    No-ops cleanly when no key is present so the rest of the pipeline keeps
    working. Network calls go to ``marsapi.ams.usda.gov``; failures are
    logged and reported but never raise.

    .. note::
       The actual JSON-to-observations transformation is intentionally a
       placeholder. Each MARS report has a different field schema; mapping
       them requires holding a real API key and confirming the structure
       per report. We ship the scaffold so the deploy story is complete;
       expanding it is the obvious v2.1 follow-up.
    """
    status: dict[str, str] = {}
    key = api_key or os.environ.get("AMS_API_KEY", "").strip()
    if not key:
        for r in reports:
            status[r.slug] = "skipped: no AMS_API_KEY"
        logger.info("AMS LMR sync skipped — no AMS_API_KEY in environment.")
        return status

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Stubbed: real implementation would loop reports, call
    # GET https://marsapi.ams.usda.gov/services/v1.2/reports/{slug}?api_key=...,
    # parse the per-report JSON shape, normalize price/date columns to the
    # observations schema, and append. Until we have a key to test against
    # we record "scaffolded" so callers (CI, dashboard) know the layer is
    # wired but not yet emitting rows.
    for r in reports:
        status[r.slug] = "scaffolded: implementation pending key-based validation"
    logger.info(
        "AMS LMR sync ran with key (length=%d); 0 rows emitted (scaffolded).",
        len(key),
    )
    return status
