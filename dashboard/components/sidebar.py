"""Persistent sidebar — series picker, data status, refresh button.

Each page calls :func:`render_sidebar` once at the top. The chosen series
lives in ``st.session_state["series_id"]`` so it survives navigation.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import streamlit as st

from usda_sandbox.clean import clean_all
from usda_sandbox.ingest import sync_downloads
from usda_sandbox.store import list_series, read_observations

DEFAULT_OBS_PATH = Path("data/clean/observations.parquet")
DEFAULT_CATALOG_PATH = Path("data/catalog.json")
DEFAULT_RAW_DIR = Path("data/raw")


@st.cache_data(ttl=300)
def cached_list_series(obs_path_str: str) -> pl.DataFrame:
    return list_series(Path(obs_path_str))


@st.cache_data(ttl=300)
def cached_dataset_overview(obs_path_str: str) -> dict[str, object]:
    obs = read_observations(Path(obs_path_str)).collect()
    return {
        "n_series": int(obs["series_id"].n_unique()),
        "n_rows": int(obs.height),
        "earliest": obs["period_start"].min(),
        "latest": obs["period_start"].max(),
    }


def _format_age(dt: datetime) -> str:
    delta = datetime.now(UTC) - dt
    secs = int(delta.total_seconds())
    if secs < 60:
        return f"{secs}s ago"
    if secs < 3600:
        return f"{secs // 60}m ago"
    if secs < 86400:
        return f"{secs // 3600}h ago"
    return f"{secs // 86400}d ago"


def render_sidebar(*, obs_path: Path = DEFAULT_OBS_PATH) -> str | None:
    """Render the sidebar and return the currently-selected series_id (or None).

    If ``observations.parquet`` does not exist, returns ``None`` and shows a
    bootstrap message + refresh button.
    """
    st.sidebar.title("USDA Livestock")

    if not obs_path.exists():
        st.sidebar.warning(
            "No cleaned data yet. Click **Refresh data** below to download "
            "and clean the ERS files."
        )
        _render_refresh_button(obs_path)
        return None

    series_df = cached_list_series(str(obs_path))
    overview = cached_dataset_overview(str(obs_path))

    st.sidebar.subheader("Series")
    series_ids = series_df["series_id"].to_list()
    name_lookup = {
        sid: name
        for sid, name in zip(
            series_df["series_id"].to_list(),
            series_df["series_name"].to_list(),
            strict=True,
        )
    }

    default_index = 0
    if "series_id" in st.session_state and st.session_state["series_id"] in series_ids:
        default_index = series_ids.index(st.session_state["series_id"])

    chosen = st.sidebar.selectbox(
        label="Active series",
        options=series_ids,
        index=default_index,
        format_func=lambda sid: name_lookup.get(sid, sid),
        key="series_id",
    )

    st.sidebar.subheader("Data status")
    mtime = datetime.fromtimestamp(obs_path.stat().st_mtime, tz=UTC)
    st.sidebar.markdown(
        f"**{overview['n_series']}** series · **{overview['n_rows']:,}** rows  \n"
        f"Range: `{overview['earliest']}` .. `{overview['latest']}`  \n"
        f"Last cleaned: `{_format_age(mtime)}`"
    )

    _render_refresh_button(obs_path)
    return chosen


def _render_refresh_button(obs_path: Path) -> None:
    if not st.sidebar.button("🔄 Refresh data", use_container_width=True):
        return
    with st.sidebar.status("Refreshing data...", expanded=True) as status:
        status.write("Discovering ERS download URLs...")
        try:
            sync_downloads(raw_dir=DEFAULT_RAW_DIR)
        except Exception as exc:
            status.update(label=f"Download failed: {exc}", state="error")
            return
        status.write("Cleaning all catalog series...")
        try:
            clean_all(DEFAULT_CATALOG_PATH, DEFAULT_RAW_DIR, obs_path)
        except Exception as exc:
            status.update(label=f"Clean failed: {exc}", state="error")
            return
        status.update(label="Refresh complete", state="complete")
    cached_list_series.clear()
    cached_dataset_overview.clear()
    st.rerun()
