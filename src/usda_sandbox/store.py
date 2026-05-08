"""Convenience accessors for the cleaned observations parquet.

Notebooks and ad-hoc analysis should reach for these helpers rather than
opening ``data/clean/observations.parquet`` directly. Two readers are exposed:
polars (for tidy-data work) and DuckDB (for SQL-shaped analytical queries).
The DuckDB connection comes pre-loaded with a view named ``obs`` over the
parquet file so callers can ``SELECT * FROM obs WHERE ...`` immediately.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl

DEFAULT_OBS_PATH = Path("data/clean/observations.parquet")
OBS_VIEW_NAME = "obs"


def _resolve_path(path: Path | str | None) -> Path:
    """Coerce ``path`` to a :class:`Path`, defaulting to the canonical location."""
    return Path(path) if path is not None else DEFAULT_OBS_PATH


def read_observations(path: Path | str | None = None) -> pl.LazyFrame:
    """Return a lazy frame over the full observations table.

    Use this when you intend to filter, aggregate, or join — polars will only
    materialize the columns and rows you need.
    """
    return pl.scan_parquet(_resolve_path(path))


def read_series(
    series_id: str,
    path: Path | str | None = None,
) -> pl.DataFrame:
    """Return one tidy series sorted by ``period_start``."""
    return (
        read_observations(path)
        .filter(pl.col("series_id") == series_id)
        .sort("period_start")
        .collect()
    )


def list_series(path: Path | str | None = None) -> pl.DataFrame:
    """One row per ``series_id`` summarizing what's in the store.

    Columns: ``series_id``, ``series_name``, ``commodity``, ``metric``,
    ``unit``, ``frequency``, ``n_obs``, ``n_nulls``, ``first_period``,
    ``last_period``. Sorted by ``series_id``.
    """
    return (
        read_observations(path)
        .group_by("series_id")
        .agg(
            pl.col("series_name").first(),
            pl.col("commodity").first(),
            pl.col("metric").first(),
            pl.col("unit").first(),
            pl.col("frequency").first(),
            pl.len().alias("n_obs"),
            pl.col("value").null_count().alias("n_nulls"),
            pl.col("period_start").min().alias("first_period"),
            pl.col("period_start").max().alias("last_period"),
        )
        .sort("series_id")
        .collect()
    )


def duckdb_connection(
    path: Path | str | None = None,
    *,
    view_name: str = OBS_VIEW_NAME,
) -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with a view named ``view_name`` over the parquet file.

    The parquet location is resolved to an absolute path so the view continues
    to work after callers change their working directory. The connection is
    the caller's to close.
    """
    abs_path = _resolve_path(path).resolve()
    if not abs_path.exists():
        raise FileNotFoundError(f"Observations parquet not found at {abs_path}")
    # Identifier is escaped, path is single-quote-escaped for DuckDB's literal syntax.
    # Prepared parameters aren't permitted in DDL, so we inline both safely.
    quoted_view = '"' + view_name.replace('"', '""') + '"'
    quoted_path = "'" + str(abs_path).replace("'", "''") + "'"
    con = duckdb.connect(":memory:")
    con.execute(
        f"CREATE OR REPLACE VIEW {quoted_view} AS SELECT * FROM read_parquet({quoted_path})"
    )
    return con
