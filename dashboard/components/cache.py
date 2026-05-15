"""Streamlit-cached helpers for reading the forecast cache JSON.

The cache is rebuilt weekly by GitHub Actions (or manually via
``python -m usda_sandbox.precompute``); the dashboard only ever reads from
it. Keeping these accessors in one place lets every page hit the same
cached load instead of re-parsing the JSON.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from usda_sandbox.precompute import DEFAULT_CACHE_PATH, load_forecast_cache


@st.cache_data(ttl=300)
def cached_forecast_cache(path_str: str = str(DEFAULT_CACHE_PATH)) -> dict:
    """Return the parsed forecasts.json (or a stub if it doesn't exist)."""
    return load_forecast_cache(Path(path_str))


def get_series_entry(series_id: str, *, path_str: str = str(DEFAULT_CACHE_PATH)) -> dict | None:
    """Return the cache entry for ``series_id``, or ``None`` if absent."""
    cache = cached_forecast_cache(path_str)
    return cache.get("by_series", {}).get(series_id)


def cache_generated_at(*, path_str: str = str(DEFAULT_CACHE_PATH)) -> str | None:
    """ISO timestamp of when the cache was last regenerated, or ``None``."""
    cache = cached_forecast_cache(path_str)
    return cache.get("generated_at")


def cache_horizon(*, path_str: str = str(DEFAULT_CACHE_PATH)) -> tuple[int, int, int] | None:
    """Return ``(cv_horizon, n_windows, forward_horizon)`` from the cache."""
    cache = cached_forecast_cache(path_str)
    if cache.get("horizon") is None:
        return None
    return (
        int(cache["horizon"]),
        int(cache["n_windows"]),
        int(cache.get("forward_horizon", 12)),
    )


__all__ = [
    "cache_generated_at",
    "cache_horizon",
    "cached_forecast_cache",
    "get_series_entry",
]
