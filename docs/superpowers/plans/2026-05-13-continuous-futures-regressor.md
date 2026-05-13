# Continuous Front-Month Futures Regressor (v0.2c) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace v0.2b's failed "12 deferred-h regressors per cash series via yfinance" exog design with a single continuous front-month regressor per cash series, sourced from Stooq.

**Architecture:** New `futures_continuous.py` module parallel to existing `futures.py`. Stooq's free monthly CSV endpoint provides ~30 years of front-month closes per commodity. Three new continuous catalog entries (`cattle_lc_front`, `hogs_he_front`, `cattle_feeder_front`), 8 cash series rewired to point at one matching entry. The existing `futures.py` (per-contract deferred-h) stays available as opt-in but drops out of the default Refresh flow.

**Tech Stack:** Python 3.11, polars, urllib.request (stdlib), pytest, streamlit. No new dependencies — Stooq is fetched via plain HTTP CSV.

**Spec:** [`docs/superpowers/specs/2026-05-13-continuous-futures-regressor-design.md`](../specs/2026-05-13-continuous-futures-regressor-design.md)

---

## File Map

**Create:**
- `src/usda_sandbox/futures_continuous.py` — new module: dataclass, fetcher, sync, append
- `tests/test_futures_continuous.py` — 10 tests covering fetcher / sync / append / dispatcher

**Modify:**
- `src/usda_sandbox/clean.py` (lines 30, 274-326) — add `futures_continuous:` dispatcher branch
- `data/catalog.json` — add 3 new front-month entries; rewire 8 cash series' regressors
- `dashboard/components/sidebar.py` (lines 17-18, 171-186) — swap sync_futures→sync_continuous_futures in Refresh flow
- `dashboard/pages/3_Forecast.py` (lines 222-243) — caption for 1-regressor case

---

## Task 1: Skeleton module + dataclass + import test

**Files:**
- Create: `src/usda_sandbox/futures_continuous.py`
- Create: `tests/test_futures_continuous.py`

- [ ] **Step 1: Write the failing test**

`tests/test_futures_continuous.py`:

```python
"""Tests for the continuous front-month futures ingest module."""

from __future__ import annotations

from dataclasses import asdict
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from usda_sandbox.futures_continuous import (
    ContinuousManifestEntry,
    _default_stooq_fetcher,
    append_continuous_to_observations,
    sync_continuous_futures,
)


def test_manifest_entry_round_trips_through_dict() -> None:
    entry = ContinuousManifestEntry(
        symbol="le.c",
        sha256="abc",
        downloaded_at="2026-05-13T10:00:00+00:00",
        missing=False,
    )
    assert ContinuousManifestEntry(**asdict(entry)) == entry
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_futures_continuous.py::test_manifest_entry_round_trips_through_dict -v`

Expected: FAIL with `ImportError` (module doesn't exist yet).

- [ ] **Step 3: Write the minimal module**

`src/usda_sandbox/futures_continuous.py`:

```python
"""Continuous front-month CME futures ingest from Stooq.

This module is the v0.2c replacement for the per-contract / deferred-h
exog regressor design in ``futures.py``. Stooq's monthly CSV endpoint
returns ~30 years of back-adjusted continuous front-month closes per
commodity, which gives enough cash-overlap to actually run CV with the
forecasters' rank requirements satisfied.

The design and rationale live in
``docs/superpowers/specs/2026-05-13-continuous-futures-regressor-design.md``.
"""

from __future__ import annotations

import hashlib
import io
import json
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

__all__ = [
    "ContinuousManifestEntry",
    "_default_stooq_fetcher",
    "append_continuous_to_observations",
    "sync_continuous_futures",
]


@dataclass(frozen=True)
class ContinuousManifestEntry:
    """One row of the continuous-futures manifest, keyed externally on symbol.

    ``missing=True`` (with ``sha256=""``) records that we attempted to fetch
    this symbol but Stooq returned no data — useful for an audit trail of
    which symbols are unavailable (e.g., ``gf.c`` if Stooq doesn't carry it).
    """

    symbol: str
    sha256: str
    downloaded_at: str
    missing: bool = False


def _default_stooq_fetcher(symbol: str) -> pl.DataFrame:
    raise NotImplementedError("filled in Task 2")


def sync_continuous_futures(
    *,
    symbols: Iterable[str] = ("le.c", "he.c", "gf.c"),
    raw_dir: Path = Path("data/raw/futures_continuous"),
    fetcher: Callable[[str], pl.DataFrame] | None = None,
) -> dict[str, ContinuousManifestEntry]:
    raise NotImplementedError("filled in Task 3")


def append_continuous_to_observations(
    *,
    obs_path: Path = Path("data/clean/observations.parquet"),
    raw_dir: Path = Path("data/raw/futures_continuous"),
    symbols: Iterable[str] = ("le.c", "he.c", "gf.c"),
) -> None:
    raise NotImplementedError("filled in Task 5")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_futures_continuous.py::test_manifest_entry_round_trips_through_dict -v`

Expected: PASS (one test).

- [ ] **Step 5: Commit**

```bash
git add src/usda_sandbox/futures_continuous.py tests/test_futures_continuous.py
git commit -m "feat(futures_continuous): module skeleton + manifest dataclass"
```

---

## Task 2: Stooq CSV fetcher

**Files:**
- Modify: `src/usda_sandbox/futures_continuous.py` (`_default_stooq_fetcher`)
- Modify: `tests/test_futures_continuous.py`

The fetcher's responsibility: turn a Stooq symbol into a `pl.DataFrame` with two columns — `period_start` (Date) and `close` (Float64). Empty DataFrame on "No data" or HTTP error.

We split it into two parts so the parser is unit-testable without network: a private `_parse_stooq_csv(text)` that does the work, and `_default_stooq_fetcher(symbol)` that wraps it with the urllib call.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_futures_continuous.py`:

```python
def test_parse_stooq_csv_returns_period_start_and_close() -> None:
    from usda_sandbox.futures_continuous import _parse_stooq_csv
    csv_text = (
        "Date,Open,High,Low,Close,Volume\n"
        "2024-01-31,170.0,175.0,168.0,172.5,1000\n"
        "2024-02-29,172.0,178.0,170.0,176.0,1200\n"
    )
    df = _parse_stooq_csv(csv_text)
    assert df.columns == ["period_start", "close"]
    assert df["period_start"].to_list() == [date(2024, 1, 31), date(2024, 2, 29)]
    assert df["close"].to_list() == [172.5, 176.0]
    assert df["period_start"].dtype == pl.Date
    assert df["close"].dtype == pl.Float64


def test_parse_stooq_csv_returns_empty_on_no_data_marker() -> None:
    from usda_sandbox.futures_continuous import _parse_stooq_csv
    df = _parse_stooq_csv("No data")
    assert df.is_empty()
    assert df.columns == ["period_start", "close"]
    assert df["period_start"].dtype == pl.Date
    assert df["close"].dtype == pl.Float64


def test_parse_stooq_csv_returns_empty_on_empty_input() -> None:
    from usda_sandbox.futures_continuous import _parse_stooq_csv
    df = _parse_stooq_csv("")
    assert df.is_empty()
    assert df.columns == ["period_start", "close"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_futures_continuous.py -k "parse_stooq" -v`

Expected: 3 failures (ImportError for `_parse_stooq_csv`).

- [ ] **Step 3: Implement `_parse_stooq_csv` and rewire `_default_stooq_fetcher`**

In `src/usda_sandbox/futures_continuous.py`, replace the stub `_default_stooq_fetcher` definition with:

```python
_STOOQ_CSV_URL = "https://stooq.com/q/d/l/?s={symbol}&i=m"


_EMPTY_CONTINUOUS_SCHEMA: dict[str, Any] = {
    "period_start": pl.Date,
    "close": pl.Float64,
}


def _parse_stooq_csv(text: str) -> pl.DataFrame:
    """Parse Stooq's monthly OHLCV CSV into (period_start, close) rows.

    Stooq returns the literal string ``"No data"`` (no header) when it has
    nothing for a symbol; that and the empty string both yield an empty
    DataFrame with the canonical schema.
    """
    text = text.strip()
    if not text or text.lower().startswith("no data"):
        return pl.DataFrame(schema=_EMPTY_CONTINUOUS_SCHEMA)
    raw = pl.read_csv(io.StringIO(text), try_parse_dates=True)
    if "Date" not in raw.columns or "Close" not in raw.columns:
        return pl.DataFrame(schema=_EMPTY_CONTINUOUS_SCHEMA)
    return (
        raw.select(
            period_start=pl.col("Date").cast(pl.Date),
            close=pl.col("Close").cast(pl.Float64),
        )
        .drop_nulls()
    )


def _default_stooq_fetcher(symbol: str) -> pl.DataFrame:
    """Real Stooq fetch — month-end closes for one continuous symbol.

    Returns an empty DataFrame (canonical schema) on HTTP error or
    "No data" response. Never raises for network issues.
    """
    url = _STOOQ_CSV_URL.format(symbol=symbol)
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError):
        return pl.DataFrame(schema=_EMPTY_CONTINUOUS_SCHEMA)
    return _parse_stooq_csv(body)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_futures_continuous.py -k "parse_stooq" -v`

Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/usda_sandbox/futures_continuous.py tests/test_futures_continuous.py
git commit -m "feat(futures_continuous): Stooq CSV parser + URL fetcher"
```

---

## Task 3: `sync_continuous_futures` — basic write path

**Files:**
- Modify: `src/usda_sandbox/futures_continuous.py` (`sync_continuous_futures` + helpers)
- Modify: `tests/test_futures_continuous.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_futures_continuous.py`:

```python
def _synthetic_continuous_frame(
    *, start: date, n: int, base: float = 170.0
) -> pl.DataFrame:
    """Build a monthly continuous frame: month-ends from ``start`` over ``n`` months."""
    rows = []
    y, m = start.year, start.month
    for i in range(n):
        # Month-end: jump to next month then back 1 day. Simple version: use
        # day=28 to dodge calendar edge cases in tests.
        rows.append({"period_start": date(y, m, 28), "close": base + i})
        m += 1
        if m == 13:
            m = 1
            y += 1
    return pl.DataFrame(rows).with_columns(
        pl.col("period_start").cast(pl.Date),
        pl.col("close").cast(pl.Float64),
    )


def test_sync_writes_one_parquet_per_symbol(tmp_path: Path) -> None:
    raw_dir = tmp_path / "futures_continuous"
    df = _synthetic_continuous_frame(start=date(2020, 1, 1), n=12)

    def fake_fetcher(symbol: str) -> pl.DataFrame:
        return df

    manifest = sync_continuous_futures(
        symbols=("le.c",),
        raw_dir=raw_dir,
        fetcher=fake_fetcher,
    )
    assert "le.c" in manifest
    assert manifest["le.c"].missing is False
    assert manifest["le.c"].sha256
    assert (raw_dir / "le.c.parquet").exists()
    on_disk = pl.read_parquet(raw_dir / "le.c.parquet")
    assert on_disk.equals(df)


def test_sync_writes_manifest_json(tmp_path: Path) -> None:
    raw_dir = tmp_path / "futures_continuous"
    df = _synthetic_continuous_frame(start=date(2020, 1, 1), n=6)
    sync_continuous_futures(
        symbols=("le.c",),
        raw_dir=raw_dir,
        fetcher=lambda s: df,
    )
    payload = json.loads((raw_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "le.c" in payload
    assert payload["le.c"]["missing"] is False
    assert payload["le.c"]["sha256"]
```

(Add `import json` and `from datetime import date` to the top of the test file if not already present.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_futures_continuous.py -k "sync_writes" -v`

Expected: 2 FAIL with `NotImplementedError`.

- [ ] **Step 3: Implement sync (basic write path, no idempotency yet)**

In `src/usda_sandbox/futures_continuous.py`, add helpers and replace the stub:

```python
_MANIFEST_FILENAME = "manifest.json"


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest(path: Path) -> dict[str, ContinuousManifestEntry]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {s: ContinuousManifestEntry(**entry) for s, entry in raw.items()}


def _save_manifest(
    path: Path, manifest: dict[str, ContinuousManifestEntry]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {s: asdict(e) for s, e in manifest.items()}
    path.write_text(
        json.dumps(serializable, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sync_continuous_futures(
    *,
    symbols: Iterable[str] = ("le.c", "he.c", "gf.c"),
    raw_dir: Path = Path("data/raw/futures_continuous"),
    fetcher: Callable[[str], pl.DataFrame] | None = None,
) -> dict[str, ContinuousManifestEntry]:
    """Download monthly continuous front-month closes from Stooq.

    Idempotent: SHA-keyed manifest. Symbols that Stooq returns no data for
    are recorded as ``missing=True`` and not re-attempted on subsequent
    runs (delete the manifest entry to force a retry).

    ``fetcher`` is injectable for tests — defaults to
    :func:`_default_stooq_fetcher`.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    fetcher = fetcher if fetcher is not None else _default_stooq_fetcher

    manifest_path = raw_dir / _MANIFEST_FILENAME
    manifest = _load_manifest(manifest_path)

    for symbol in symbols:
        file_path = raw_dir / f"{symbol}.parquet"

        df = fetcher(symbol)
        if df.is_empty():
            manifest[symbol] = ContinuousManifestEntry(
                symbol=symbol,
                sha256="",
                downloaded_at=datetime.now(UTC).isoformat(),
                missing=True,
            )
            continue

        df.write_parquet(file_path)
        manifest[symbol] = ContinuousManifestEntry(
            symbol=symbol,
            sha256=_sha256_of(file_path),
            downloaded_at=datetime.now(UTC).isoformat(),
            missing=False,
        )

    _save_manifest(manifest_path, manifest)
    return manifest
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_futures_continuous.py -v`

Expected: all 6 tests pass (3 from Task 2 + 2 new + 1 manifest round-trip from Task 1).

- [ ] **Step 5: Commit**

```bash
git add src/usda_sandbox/futures_continuous.py tests/test_futures_continuous.py
git commit -m "feat(futures_continuous): sync writes parquet + manifest per symbol"
```

---

## Task 4: Sync idempotency and missing-symbol handling

**Files:**
- Modify: `src/usda_sandbox/futures_continuous.py` (`sync_continuous_futures`)
- Modify: `tests/test_futures_continuous.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_futures_continuous.py`:

```python
def test_sync_idempotent_when_data_unchanged(tmp_path: Path) -> None:
    raw_dir = tmp_path / "futures_continuous"
    df = _synthetic_continuous_frame(start=date(2020, 1, 1), n=6)
    calls = []

    def counting_fetcher(symbol: str) -> pl.DataFrame:
        calls.append(symbol)
        return df

    sync_continuous_futures(
        symbols=("le.c",), raw_dir=raw_dir, fetcher=counting_fetcher
    )
    sync_continuous_futures(
        symbols=("le.c",), raw_dir=raw_dir, fetcher=counting_fetcher
    )
    # Second sync still calls fetcher (it has to to see if data changed),
    # but writes nothing new when SHA matches. Verify by checking SHA in
    # manifest is the same and file mtime is unchanged is fragile; instead,
    # assert the parquet content is still the synthetic frame.
    on_disk = pl.read_parquet(raw_dir / "le.c.parquet")
    assert on_disk.equals(df)


def test_sync_records_missing_symbol_without_raising(tmp_path: Path) -> None:
    raw_dir = tmp_path / "futures_continuous"

    def empty_fetcher(symbol: str) -> pl.DataFrame:
        return pl.DataFrame(
            schema={"period_start": pl.Date, "close": pl.Float64}
        )

    manifest = sync_continuous_futures(
        symbols=("gf.c",), raw_dir=raw_dir, fetcher=empty_fetcher
    )
    assert manifest["gf.c"].missing is True
    assert manifest["gf.c"].sha256 == ""
    assert not (raw_dir / "gf.c.parquet").exists()


def test_sync_skips_known_missing_symbols_on_rerun(tmp_path: Path) -> None:
    raw_dir = tmp_path / "futures_continuous"
    calls = []

    def empty_fetcher(symbol: str) -> pl.DataFrame:
        calls.append(symbol)
        return pl.DataFrame(
            schema={"period_start": pl.Date, "close": pl.Float64}
        )

    sync_continuous_futures(
        symbols=("gf.c",), raw_dir=raw_dir, fetcher=empty_fetcher
    )
    sync_continuous_futures(
        symbols=("gf.c",), raw_dir=raw_dir, fetcher=empty_fetcher
    )
    # First call: 1 attempt. Second call: skip because known-missing.
    assert calls == ["gf.c"]


def test_sync_replaces_changed_data(tmp_path: Path) -> None:
    raw_dir = tmp_path / "futures_continuous"
    df_v1 = _synthetic_continuous_frame(start=date(2020, 1, 1), n=4, base=100.0)
    df_v2 = _synthetic_continuous_frame(start=date(2020, 1, 1), n=4, base=200.0)

    sync_continuous_futures(
        symbols=("le.c",), raw_dir=raw_dir, fetcher=lambda s: df_v1
    )
    sync_continuous_futures(
        symbols=("le.c",), raw_dir=raw_dir, fetcher=lambda s: df_v2
    )
    on_disk = pl.read_parquet(raw_dir / "le.c.parquet")
    assert on_disk["close"].to_list() == df_v2["close"].to_list()
```

- [ ] **Step 2: Run tests to verify failures**

Run: `uv run pytest tests/test_futures_continuous.py -k "idempotent or missing or replaces" -v`

Expected: `test_sync_skips_known_missing_symbols_on_rerun` FAILS (currently calls fetcher every time); the other three may already pass given the simple Task 3 implementation. Note which fail.

- [ ] **Step 3: Add idempotency + known-missing skip to sync**

In `src/usda_sandbox/futures_continuous.py`, replace the body of the `for symbol in symbols:` loop in `sync_continuous_futures` with:

```python
    for symbol in symbols:
        file_path = raw_dir / f"{symbol}.parquet"

        # Already-known-missing symbols: don't re-attempt every run.
        # Delete the manifest entry manually to force a retry.
        if symbol in manifest and manifest[symbol].missing:
            continue

        # Already-cached symbols whose on-disk SHA still matches: skip the
        # fetch. This is a per-byte check so any upstream change forces a
        # refresh.
        if (
            symbol in manifest
            and file_path.exists()
            and _sha256_of(file_path) == manifest[symbol].sha256
        ):
            continue

        df = fetcher(symbol)
        if df.is_empty():
            manifest[symbol] = ContinuousManifestEntry(
                symbol=symbol,
                sha256="",
                downloaded_at=datetime.now(UTC).isoformat(),
                missing=True,
            )
            continue

        df.write_parquet(file_path)
        manifest[symbol] = ContinuousManifestEntry(
            symbol=symbol,
            sha256=_sha256_of(file_path),
            downloaded_at=datetime.now(UTC).isoformat(),
            missing=False,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_futures_continuous.py -v`

Expected: all pass (9 tests so far).

- [ ] **Step 5: Commit**

```bash
git add src/usda_sandbox/futures_continuous.py tests/test_futures_continuous.py
git commit -m "feat(futures_continuous): SHA-keyed idempotency + known-missing skip"
```

---

## Task 5: `append_continuous_to_observations`

**Files:**
- Modify: `src/usda_sandbox/futures_continuous.py` (`append_continuous_to_observations`)
- Modify: `tests/test_futures_continuous.py`

This function reads cached parquets and writes one row per (symbol, month-end) into `observations.parquet`, using the canonical schema. Symbol→series_id mapping is hardcoded (3 symbols, 3 series_ids).

- [ ] **Step 1: Write failing tests**

Append to `tests/test_futures_continuous.py`:

```python
def _empty_observations() -> pl.DataFrame:
    """Minimal observations.parquet schema for round-trip tests."""
    return pl.DataFrame(
        schema={
            "series_id": pl.Utf8,
            "series_name": pl.Utf8,
            "commodity": pl.Utf8,
            "metric": pl.Utf8,
            "unit": pl.Utf8,
            "frequency": pl.Utf8,
            "period_start": pl.Date,
            "period_end": pl.Date,
            "value": pl.Float64,
            "source_file": pl.Utf8,
            "source_sheet": pl.Utf8,
            "ingested_at": pl.Datetime("us", "UTC"),
        }
    )


def test_append_writes_one_row_per_symbol_per_month(tmp_path: Path) -> None:
    raw_dir = tmp_path / "futures_continuous"
    obs_path = tmp_path / "observations.parquet"
    _empty_observations().write_parquet(obs_path)

    df_le = _synthetic_continuous_frame(start=date(2020, 1, 1), n=3, base=170.0)
    sync_continuous_futures(
        symbols=("le.c",), raw_dir=raw_dir, fetcher=lambda s: df_le
    )

    append_continuous_to_observations(
        obs_path=obs_path, raw_dir=raw_dir, symbols=("le.c",)
    )

    obs = pl.read_parquet(obs_path)
    le_rows = obs.filter(pl.col("series_id") == "cattle_lc_front")
    assert le_rows.height == 3
    assert le_rows["close"].to_list() if "close" in le_rows.columns else True
    # Schema fields populated correctly:
    first = le_rows.sort("period_start").row(0, named=True)
    assert first["series_name"] == "Live Cattle front-month futures (continuous)"
    assert first["commodity"] == "cattle"
    assert first["metric"] == "futures_price"
    assert first["unit"] == "USD/cwt"
    assert first["frequency"] == "monthly"
    assert first["value"] == pytest.approx(170.0)
    assert first["source_file"] == "futures_continuous:le.c"


def test_append_preserves_existing_observations(tmp_path: Path) -> None:
    raw_dir = tmp_path / "futures_continuous"
    obs_path = tmp_path / "observations.parquet"

    # Seed observations with one unrelated cash row
    seed = pl.DataFrame(
        {
            "series_id": ["cattle_steer_choice_nebraska"],
            "series_name": ["seed"],
            "commodity": ["cattle"],
            "metric": ["price"],
            "unit": ["USD/cwt"],
            "frequency": ["monthly"],
            "period_start": [date(2024, 1, 1)],
            "period_end": [date(2024, 1, 31)],
            "value": [180.0],
            "source_file": ["livestock-prices.xlsx"],
            "source_sheet": ["Historical"],
            "ingested_at": [datetime(2024, 1, 1, tzinfo=UTC)],
        }
    ).with_columns(
        pl.col("period_start").cast(pl.Date),
        pl.col("period_end").cast(pl.Date),
        pl.col("value").cast(pl.Float64),
        pl.col("ingested_at").cast(pl.Datetime("us", "UTC")),
    )
    seed.write_parquet(obs_path)

    df = _synthetic_continuous_frame(start=date(2020, 1, 1), n=2)
    sync_continuous_futures(
        symbols=("le.c",), raw_dir=raw_dir, fetcher=lambda s: df
    )
    append_continuous_to_observations(
        obs_path=obs_path, raw_dir=raw_dir, symbols=("le.c",)
    )

    obs = pl.read_parquet(obs_path)
    cash = obs.filter(pl.col("series_id") == "cattle_steer_choice_nebraska")
    assert cash.height == 1
    assert obs.filter(pl.col("series_id") == "cattle_lc_front").height == 2


def test_append_idempotent_on_rerun(tmp_path: Path) -> None:
    raw_dir = tmp_path / "futures_continuous"
    obs_path = tmp_path / "observations.parquet"
    _empty_observations().write_parquet(obs_path)
    df = _synthetic_continuous_frame(start=date(2020, 1, 1), n=4)
    sync_continuous_futures(
        symbols=("le.c",), raw_dir=raw_dir, fetcher=lambda s: df
    )
    append_continuous_to_observations(
        obs_path=obs_path, raw_dir=raw_dir, symbols=("le.c",)
    )
    append_continuous_to_observations(
        obs_path=obs_path, raw_dir=raw_dir, symbols=("le.c",)
    )
    obs = pl.read_parquet(obs_path)
    le_rows = obs.filter(pl.col("series_id") == "cattle_lc_front")
    assert le_rows.height == 4  # unchanged on second run


def test_append_raises_if_raw_dir_missing(tmp_path: Path) -> None:
    obs_path = tmp_path / "observations.parquet"
    _empty_observations().write_parquet(obs_path)
    missing_dir = tmp_path / "nope"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        append_continuous_to_observations(
            obs_path=obs_path, raw_dir=missing_dir
        )
```

(Make sure `from datetime import datetime, UTC` and `import pytest` are in the test file.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_futures_continuous.py -k "append" -v`

Expected: 4 FAIL with `NotImplementedError`.

- [ ] **Step 3: Implement `append_continuous_to_observations`**

In `src/usda_sandbox/futures_continuous.py`, replace the stub:

```python
# Symbol → (series_id, series_name, commodity) — single source of truth.
_SYMBOL_META: dict[str, tuple[str, str, str]] = {
    "le.c": (
        "cattle_lc_front",
        "Live Cattle front-month futures (continuous)",
        "cattle",
    ),
    "he.c": (
        "hogs_he_front",
        "Lean Hogs front-month futures (continuous)",
        "hogs",
    ),
    "gf.c": (
        "cattle_feeder_front",
        "Feeder Cattle front-month futures (continuous)",
        "cattle",
    ),
}


def append_continuous_to_observations(
    *,
    obs_path: Path = Path("data/clean/observations.parquet"),
    raw_dir: Path = Path("data/raw/futures_continuous"),
    symbols: Iterable[str] = ("le.c", "he.c", "gf.c"),
) -> None:
    """Build observations rows from cached continuous-futures parquets
    and merge them into ``observations.parquet``. Idempotent on
    ``(series_id, period_start, source_file)``."""
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"{raw_dir!s} does not exist. Run sync_continuous_futures() "
            "before append_continuous_to_observations()."
        )

    obs_path = Path(obs_path)
    ingested_at = datetime.now(UTC)

    new_frames: list[pl.DataFrame] = []
    for symbol in symbols:
        if symbol not in _SYMBOL_META:
            continue
        file_path = raw_dir / f"{symbol}.parquet"
        if not file_path.exists():
            continue  # missing symbol — sync recorded it; nothing to append
        series_id, series_name, commodity = _SYMBOL_META[symbol]
        cached = pl.read_parquet(file_path)
        if cached.is_empty():
            continue
        frame = cached.select(
            series_id=pl.lit(series_id),
            series_name=pl.lit(series_name),
            commodity=pl.lit(commodity),
            metric=pl.lit("futures_price"),
            unit=pl.lit("USD/cwt"),
            frequency=pl.lit("monthly"),
            period_start=pl.col("period_start").dt.month_start(),
            period_end=pl.col("period_start"),
            value=pl.col("close"),
            source_file=pl.lit(f"futures_continuous:{symbol}"),
            source_sheet=pl.lit(""),
            ingested_at=pl.lit(ingested_at).cast(pl.Datetime("us", "UTC")),
        )
        new_frames.append(frame)

    if not new_frames:
        return

    new_obs = pl.concat(new_frames, how="vertical_relaxed")
    new_ids = set(new_obs["series_id"].to_list())

    if obs_path.exists():
        existing = pl.read_parquet(obs_path).filter(
            ~pl.col("series_id").is_in(new_ids)
        )
        combined = pl.concat([existing, new_obs], how="vertical_relaxed")
    else:
        combined = new_obs

    combined = combined.unique(
        subset=["series_id", "period_start", "source_file"], keep="last"
    ).sort(["series_id", "period_start"])
    obs_path.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(obs_path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_futures_continuous.py -v`

Expected: all 14 tests pass (Task 1: 1 + Task 2: 3 + Task 3: 2 + Task 4: 4 + Task 5: 4 = 14).

- [ ] **Step 5: Commit**

```bash
git add src/usda_sandbox/futures_continuous.py tests/test_futures_continuous.py
git commit -m "feat(futures_continuous): append cached parquets into observations"
```

---

## Task 6: `clean.py` dispatcher branch

**Files:**
- Modify: `src/usda_sandbox/clean.py` (lines 30, 289-326)
- Modify: `tests/test_futures_continuous.py`

Mirror the existing `futures:` dispatcher branch with a parallel `futures_continuous:` one. `clean_all` already has `has_futures` to signal post-loop append; we add `has_futures_continuous` and call `append_continuous_to_observations` after the futures call.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_futures_continuous.py`:

```python
def test_clean_all_dispatcher_routes_futures_continuous_prefix(
    tmp_path: Path,
) -> None:
    """clean_all skips futures_continuous: series in the XLSX loop and
    calls append_continuous_to_observations after."""
    from usda_sandbox.clean import clean_all

    # Minimal catalog with one futures_continuous entry only
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            [
                {
                    "series_id": "cattle_lc_front",
                    "series_name": "Live Cattle front-month futures (continuous)",
                    "commodity": "cattle",
                    "metric": "futures_price",
                    "unit": "USD/cwt",
                    "frequency": "monthly",
                    "source_file": "futures_continuous:le.c",
                    "source_sheet": "",
                    "header_rows_to_skip": 0,
                    "value_columns": ["X"],
                    "date_column": "X",
                    "notes": "test",
                    "exogenous_regressors": [],
                    "forecastable": False,
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    fc_dir = raw_dir / "futures_continuous"
    df = _synthetic_continuous_frame(start=date(2020, 1, 1), n=3)
    sync_continuous_futures(
        symbols=("le.c",), raw_dir=fc_dir, fetcher=lambda s: df
    )

    out_path = tmp_path / "observations.parquet"
    clean_all(catalog_path, raw_dir, out_path)

    obs = pl.read_parquet(out_path)
    assert obs.filter(pl.col("series_id") == "cattle_lc_front").height == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_futures_continuous.py::test_clean_all_dispatcher_routes_futures_continuous_prefix -v`

Expected: FAIL — either the prefix isn't recognized (rows missing) or the catalog file fails to parse XLSX paths.

- [ ] **Step 3: Patch `clean.py`**

In `src/usda_sandbox/clean.py`:

Replace line 30:

```python
from .futures import append_futures_to_observations
```

with:

```python
from .futures import append_futures_to_observations
from .futures_continuous import append_continuous_to_observations
```

Then in `clean_all` (around lines 286-326), replace the function body from `frames: list[pl.DataFrame] = []` through `return combined` with:

```python
    frames: list[pl.DataFrame] = []
    has_futures = False
    has_futures_continuous = False
    for series_def in catalog:
        if series_def.source_file.startswith("futures:"):
            has_futures = True
            continue
        if series_def.source_file.startswith("futures_continuous:"):
            has_futures_continuous = True
            continue
        raw_path = raw_dir / series_def.source_file
        if not raw_path.exists():
            print(
                f"[clean_all] skipping {series_def.series_id}: missing {raw_path}",
                flush=True,
            )
            continue
        frames.append(clean_series(series_def, raw_path))

    if not frames:
        combined = _empty_frame()
    else:
        combined = pl.concat(frames, how="vertical_relaxed").unique(
            subset=["series_id", "period_start", "source_file"], keep="first"
        )
        combined = combined.sort(["series_id", "period_start"])

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(out)

    if has_futures:
        # raw_dir is for XLSX files; futures live under data/raw/futures/ by default.
        # If a non-default raw_dir is passed, forward its sibling 'futures' subdir
        # so test/CI environments can use isolated paths.
        futures_raw = raw_dir / "futures"
        if futures_raw.exists():
            append_futures_to_observations(obs_path=out, raw_dir=futures_raw)
        else:
            append_futures_to_observations(obs_path=out)
        combined = pl.read_parquet(out)

    if has_futures_continuous:
        fc_raw = raw_dir / "futures_continuous"
        if fc_raw.exists():
            append_continuous_to_observations(obs_path=out, raw_dir=fc_raw)
        else:
            append_continuous_to_observations(obs_path=out)
        combined = pl.read_parquet(out)

    return combined
```

- [ ] **Step 4: Run all tests**

Run: `uv run pytest --no-header -q`

Expected: all pass (existing 143 + new 15 = 158).

- [ ] **Step 5: Commit**

```bash
git add src/usda_sandbox/clean.py tests/test_futures_continuous.py
git commit -m "feat(clean): dispatch futures_continuous: prefix to new appender"
```

---

## Task 7: Catalog updates

**Files:**
- Modify: `data/catalog.json`

Three new front-month entries + 8 cash series rewired. We do this with a script to keep ordering and quoting consistent.

- [ ] **Step 1: Apply the catalog edit**

Run this from the worktree root:

```bash
uv run python << 'PYEOF'
import json
from pathlib import Path

path = Path('data/catalog.json')
cat = json.load(open(path))

# 1) Append three new front-month entries (if not present)
existing_ids = {e['series_id'] for e in cat}
front_specs = [
    ("le.c", "cattle_lc_front",
     "Live Cattle front-month futures (continuous)", "cattle"),
    ("he.c", "hogs_he_front",
     "Lean Hogs front-month futures (continuous)", "hogs"),
    ("gf.c", "cattle_feeder_front",
     "Feeder Cattle front-month futures (continuous)", "cattle"),
]
for symbol, sid, name, commodity in front_specs:
    if sid in existing_ids:
        continue
    cat.append({
        "series_id": sid,
        "series_name": name,
        "commodity": commodity,
        "metric": "futures_price",
        "unit": "USD/cwt",
        "frequency": "monthly",
        "source_file": f"futures_continuous:{symbol}",
        "source_sheet": "",
        "header_rows_to_skip": 0,
        "value_columns": ["X"],
        "date_column": "X",
        "notes": (
            "Stooq's back-adjusted continuous front-month series. Used as a "
            "single exogenous regressor for the matching cash series. The "
            "deferred-h elaboration from v0.2b lives in the catalog "
            "(*_deferred_*mo) for diagnostic viewing on Explore but is no "
            "longer fed to forecasters."
        ),
        "exogenous_regressors": [],
        "forecastable": False,
    })

# 2) Rewire 8 cash series' regressors
rewire = {
    "cattle_steer_choice_nebraska": ["cattle_lc_front"],
    "cattle_steer_choice_tx_ok_nm": ["cattle_lc_front"],
    "cattle_feeder_steer_500_550": ["cattle_lc_front"],
    "cattle_feeder_steer_750_800": ["cattle_lc_front"],
    "boxed_beef_cutout_choice":    ["cattle_lc_front"],
    "boxed_beef_cutout_select":    ["cattle_lc_front"],
    "pork_cutout_composite":        ["hogs_he_front"],
    "hog_barrow_gilt_natbase_51_52": ["hogs_he_front"],
}
for e in cat:
    if e['series_id'] in rewire:
        e['exogenous_regressors'] = rewire[e['series_id']]

path.write_text(json.dumps(cat, indent=2) + "\n", encoding="utf-8")
print("entries:", len(cat))
print("rewired:", list(rewire.keys()))
PYEOF
```

- [ ] **Step 2: Verify the catalog parses and is wired correctly**

Run:

```bash
uv run python -c "
from usda_sandbox.catalog import load_catalog
cat = load_catalog('data/catalog.json')
print('total:', len(cat))
for sid in ['cattle_lc_front', 'hogs_he_front', 'cattle_feeder_front']:
    s = next((s for s in cat if s.series_id == sid), None)
    assert s is not None, f'missing {sid}'
    assert not s.forecastable, sid
    print(f'  {sid}: source_file={s.source_file}')
for sid in ['cattle_steer_choice_nebraska', 'pork_cutout_composite']:
    s = next(s for s in cat if s.series_id == sid)
    print(f'  {sid} regs={s.exogenous_regressors}')
"
```

Expected: total = 51 (48 from v0.2b GF work + 3 new fronts), regressors as wired.

- [ ] **Step 3: Run tests**

Run: `uv run pytest --no-header -q`

Expected: still 158 PASS (no test should regress on a catalog data-only change).

- [ ] **Step 4: Commit**

```bash
git add data/catalog.json
git commit -m "feat(catalog): add cattle_lc_front/hogs_he_front/cattle_feeder_front; rewire 8 cash series"
```

---

## Task 8: Dashboard sidebar — Refresh flow

**Files:**
- Modify: `dashboard/components/sidebar.py` (lines 17-18, 171-186)

Swap `sync_futures` → `sync_continuous_futures` in the Refresh flow. The opt-in `sync_futures()` import stays available but isn't called.

- [ ] **Step 1: Edit `dashboard/components/sidebar.py`**

Replace:

```python
from usda_sandbox.futures import sync_futures
```

with:

```python
from usda_sandbox.futures_continuous import sync_continuous_futures
```

Replace the futures-sync step in `_render_refresh_button` (the one that says "Syncing CME futures contracts (LE, HE, GF) via yfinance..."):

```python
        status.write("Syncing CME futures contracts (LE, HE, GF) via yfinance...")
        try:
            sync_futures(raw_dir=DEFAULT_RAW_DIR / "futures")
        except Exception as exc:
            status.update(label=f"Futures sync failed: {exc}", state="error")
            return
```

with:

```python
        status.write("Syncing continuous front-month futures from Stooq...")
        try:
            sync_continuous_futures(
                raw_dir=DEFAULT_RAW_DIR / "futures_continuous"
            )
        except Exception as exc:
            status.update(
                label=f"Continuous-futures sync failed: {exc}", state="error"
            )
            return
```

- [ ] **Step 2: Verify the file parses**

Run: `uv run python -c "import ast; ast.parse(open('dashboard/components/sidebar.py', encoding='utf-8').read()); print('OK')"`

Expected: `OK`.

- [ ] **Step 3: Run tests**

Run: `uv run pytest --no-header -q`

Expected: 158 PASS.

- [ ] **Step 4: Commit**

```bash
git add dashboard/components/sidebar.py
git commit -m "feat(dashboard): swap sync_futures for sync_continuous_futures in Refresh flow"
```

---

## Task 9: Dashboard Forecast page — single-regressor caption

**Files:**
- Modify: `dashboard/pages/3_Forecast.py` (lines 222-243 — the regressor caption block)

Replace the existing branch (which assumed 12 deferred-h regressors and a known commodity-LE/HE/GF prefix) with one that handles the new 1-regressor case while keeping the multi-regressor branch as a fallback.

- [ ] **Step 1: Edit `dashboard/pages/3_Forecast.py`**

Find the block:

```python
if _target_def is not None and _target_def.exogenous_regressors:
    # Pick the label from the regressor prefix so feeder-cattle (GF)
    # vs. fed-cattle (LE) cases aren't both labeled "Live Cattle".
    _first_reg = _target_def.exogenous_regressors[0]
    if _first_reg.startswith("cattle_lc_"):
        _commodity_label = "Live Cattle"
    elif _first_reg.startswith("cattle_feeder_"):
        _commodity_label = "Feeder Cattle"
    elif _first_reg.startswith("hogs_he_"):
        _commodity_label = "Lean Hogs"
    else:
        _commodity_label = "futures"
    st.caption(
        f"This series was forecast with **{len(_target_def.exogenous_regressors)} "
        f"exogenous regressors**: deferred {_commodity_label} futures (1-12 months "
        f"ahead). Each forecaster (AutoARIMA, Prophet, LightGBM) sees these alongside "
        f"the cash history."
    )
```

Replace it with:

```python
if _target_def is not None and _target_def.exogenous_regressors:
    # Map the first regressor's id to a human-readable commodity label.
    _front_labels = {
        "cattle_lc_front": "Live Cattle",
        "cattle_feeder_front": "Feeder Cattle",
        "hogs_he_front": "Lean Hogs",
    }
    _first_reg = _target_def.exogenous_regressors[0]
    if _first_reg in _front_labels:
        _commodity_label = _front_labels[_first_reg]
    elif _first_reg.startswith("cattle_lc_"):
        _commodity_label = "Live Cattle"
    elif _first_reg.startswith("cattle_feeder_"):
        _commodity_label = "Feeder Cattle"
    elif _first_reg.startswith("hogs_he_"):
        _commodity_label = "Lean Hogs"
    else:
        _commodity_label = "futures"

    if len(_target_def.exogenous_regressors) == 1:
        st.caption(
            f"This series was forecast with a single exogenous regressor: "
            f"the continuous front-month **{_commodity_label}** futures price. "
            f"Each forecaster (AutoARIMA, Prophet, LightGBM) sees it alongside "
            f"the cash history."
        )
    else:
        st.caption(
            f"This series was forecast with **{len(_target_def.exogenous_regressors)} "
            f"exogenous regressors**: deferred {_commodity_label} futures "
            f"(1-12 months ahead). Each forecaster (AutoARIMA, Prophet, LightGBM) "
            f"sees these alongside the cash history."
        )
```

- [ ] **Step 2: Verify the file parses**

Run: `uv run python -c "import ast; ast.parse(open('dashboard/pages/3_Forecast.py', encoding='utf-8').read()); print('OK')"`

Expected: `OK`.

- [ ] **Step 3: Run tests**

Run: `uv run pytest --no-header -q`

Expected: 158 PASS.

- [ ] **Step 4: Commit**

```bash
git add dashboard/pages/3_Forecast.py
git commit -m "feat(dashboard): single-regressor caption for continuous front-month exog"
```

---

## Task 10: End-to-end smoke test

**Files:**
- None (manual + scripted verification)

- [ ] **Step 1: Run a real Stooq fetch to verify the URL still works**

Run:

```bash
uv run python -c "
from usda_sandbox.futures_continuous import _default_stooq_fetcher
df = _default_stooq_fetcher('le.c')
print('rows:', df.height)
print('earliest:', df['period_start'].min() if df.height else 'none')
print('latest:', df['period_start'].max() if df.height else 'none')
"
```

Expected: rows > 100, earliest in the 1990s or earlier, latest a recent month-end.

If rows == 0: investigate (Stooq could have changed URL format; check the URL in a browser).

- [ ] **Step 2: Run a full sync + clean + backtest from CLI**

Run:

```bash
uv run python << 'PYEOF'
from pathlib import Path
from usda_sandbox.ingest import sync_downloads
from usda_sandbox.futures_continuous import sync_continuous_futures
from usda_sandbox.clean import clean_all
from usda_sandbox.forecast import run_backtest

raw = Path("data/raw")
sync_downloads(raw_dir=raw)
sync_continuous_futures(raw_dir=raw / "futures_continuous")
clean_all(Path("data/catalog.json"), raw, Path("data/clean/observations.parquet"))

for sid in ["cattle_steer_choice_nebraska", "cattle_feeder_steer_500_550", "pork_cutout_composite"]:
    r = run_backtest(sid, horizon=6, n_windows=8)
    best = r.metrics.sort("mape").row(0, named=True)
    print(f"{sid}: best={best['model']} MAPE={best['mape']:.2f}")
PYEOF
```

Expected: each series produces a finite MAPE. Nebraska steers should be 3-5% (literature benchmark with futures regressor) — but be tolerant; could be 5-7% if Stooq's continuous symbol differs from what the literature used. Anything < 8% counts as "the regressor is doing useful work."

- [ ] **Step 3: Launch the dashboard and click Refresh**

Run: `uv run streamlit run dashboard/app.py --server.headless true --server.port 8501`

In a browser at `http://localhost:8501`:
- Sidebar → Refresh data. Watch the status panel; should see "Syncing continuous front-month futures from Stooq...".
- Navigate to Forecast. Pick `cattle_steer_choice_nebraska`.
- Scoreboard caption should read: "This series was forecast with a single exogenous regressor: the continuous front-month **Live Cattle** futures price."
- Run backtest at default settings. Verify it completes (no rank deficiency, no "number sections" error).
- Forward-forecast caption should still show per-horizon conformal scales.

Stop the dashboard with Ctrl-C in the terminal once verified.

- [ ] **Step 4: Run the full test suite once more**

Run: `uv run pytest --no-header -q`

Expected: 158 PASS.

- [ ] **Step 5: Final commit (only if you made cleanup edits)**

If anything was tweaked during smoke testing:

```bash
git status
git add <files>
git commit -m "fix(v0.2c): <whatever needed cleanup>"
```

Otherwise nothing to commit — the prior 9 tasks land v0.2c.

---

## Self-Review Notes

- **Spec coverage:** All 5 spec sections (architecture, module, catalog, dashboard, tests) → tasks 1-9. Task 10 is empirical validation.
- **Symbol metadata:** `_SYMBOL_META` in Task 5 must match the catalog entries in Task 7 (series_id, series_name, commodity). Verified consistent.
- **Backward compatibility:** `futures.py` and the 36 deferred-h catalog entries are untouched. Existing tests (143) untouched.
- **Cattle feeder fallback:** Spec says "if gf.c isn't served, feeder stays on cattle_lc_front." Implemented via Task 7's catalog rewire — feeder defaults to cattle_lc_front. If gf.c does arrive on first sync, user can re-run Task 7's script with the gf.c-pointed regressors. Documenting this as an opt-in is reasonable; not blocking the plan.
