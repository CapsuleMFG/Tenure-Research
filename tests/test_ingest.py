"""Tests for the ingest layer.

The HTTP layer is mocked via ``httpx.MockTransport`` so these tests are
hermetic — they neither hit the network nor depend on USDA's actual page
markup. They cover URL discovery and the three idempotency cases:
first-run download, no-op re-sync, and replacement on remote change.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import httpx
import pytest

from usda_sandbox.ingest import (
    ERS_PRODUCT_URL,
    discover_download_urls,
    filename_from_url,
    load_manifest,
    sync_downloads,
)

PRODUCT_URL = "https://example.test/data-products/livestock-and-meat-domestic-data"

SAMPLE_PAGE_HTML = """
<html><body>
  <a href="/sites/files/LDPM-Cattle.xlsx">Cattle prices</a>
  <a href="https://example.test/files/Hogs.xlsx">Hogs</a>
  <a href="/files/MeatStocks.zip">Meat stocks</a>
  <a href="/files/notes.pdf">Methodology PDF</a>
  <a href="/about">About</a>
  <a href="/sites/files/LDPM-Cattle.xlsx">Duplicate cattle link</a>
</body></html>
"""


def _make_client(
    page_html: str,
    files: Mapping[str, bytes],
    *,
    page_url: str = PRODUCT_URL,
) -> httpx.Client:
    """Build an httpx.Client whose transport serves a fake product page and files."""

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == page_url:
            return httpx.Response(200, text=page_html)
        for name, body in files.items():
            if url.endswith("/" + name):
                return httpx.Response(200, content=body)
        return httpx.Response(404, text=f"not mocked: {url}")

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_discover_download_urls_finds_xlsx_and_zip_only() -> None:
    urls = discover_download_urls(SAMPLE_PAGE_HTML, base_url=PRODUCT_URL)

    assert urls == [
        "https://example.test/sites/files/LDPM-Cattle.xlsx",
        "https://example.test/files/Hogs.xlsx",
        "https://example.test/files/MeatStocks.zip",
    ]
    assert all(u.startswith("http") for u in urls)


def test_discover_handles_default_base_url_constant() -> None:
    # Smoke-test that the default base URL still resolves relative hrefs.
    urls = discover_download_urls('<a href="/files/X.xlsx">x</a>')
    assert urls == [
        "https://www.ers.usda.gov/files/X.xlsx",
    ]
    assert ERS_PRODUCT_URL.endswith("livestock-and-meat-domestic-data")


def test_filename_from_url_decodes_and_strips_query() -> None:
    assert filename_from_url("https://x/path/Hogs%20Prices.xlsx?v=1") == "Hogs Prices.xlsx"


def test_filename_from_url_rejects_empty() -> None:
    with pytest.raises(ValueError):
        filename_from_url("https://x/")


def test_sync_downloads_first_run(tmp_path: Path) -> None:
    files = {
        "LDPM-Cattle.xlsx": b"cattle-bytes",
        "Hogs.xlsx": b"hogs-bytes",
        "MeatStocks.zip": b"PK\x03\x04stub",
    }
    client = _make_client(SAMPLE_PAGE_HTML, files)

    raw_dir = tmp_path / "raw"
    manifest = sync_downloads(raw_dir=raw_dir, product_url=PRODUCT_URL, client=client)

    assert (raw_dir / "LDPM-Cattle.xlsx").read_bytes() == b"cattle-bytes"
    assert (raw_dir / "Hogs.xlsx").read_bytes() == b"hogs-bytes"
    assert (raw_dir / "MeatStocks.zip").read_bytes() == b"PK\x03\x04stub"

    # No leftover .part files
    assert not list(raw_dir.glob("*.part"))

    # Manifest written and loadable
    on_disk = json.loads((raw_dir / "manifest.json").read_text())
    assert set(on_disk.keys()) == set(manifest.keys()) == {
        "https://example.test/sites/files/LDPM-Cattle.xlsx",
        "https://example.test/files/Hogs.xlsx",
        "https://example.test/files/MeatStocks.zip",
    }
    for entry in manifest.values():
        assert len(entry.sha256) == 64
        assert entry.downloaded_at  # ISO-8601 string


def test_sync_downloads_is_idempotent_for_unchanged_files(tmp_path: Path) -> None:
    files = {"Hogs.xlsx": b"hogs-v1"}
    page = '<a href="/files/Hogs.xlsx">Hogs</a>'
    raw_dir = tmp_path / "raw"

    first = sync_downloads(
        raw_dir=raw_dir,
        product_url=PRODUCT_URL,
        client=_make_client(page, files, page_url=PRODUCT_URL),
    )
    first_entry = next(iter(first.values()))
    first_mtime = (raw_dir / "Hogs.xlsx").stat().st_mtime_ns

    # Second sync with identical content — manifest entry must be preserved verbatim
    second = sync_downloads(
        raw_dir=raw_dir,
        product_url=PRODUCT_URL,
        client=_make_client(page, files, page_url=PRODUCT_URL),
    )
    second_entry = next(iter(second.values()))

    assert second_entry == first_entry  # same sha + same downloaded_at
    # The on-disk file should not have been rewritten
    assert (raw_dir / "Hogs.xlsx").stat().st_mtime_ns == first_mtime
    assert not list(raw_dir.glob("*.part"))


def test_sync_downloads_replaces_changed_file(tmp_path: Path) -> None:
    page = '<a href="/files/Hogs.xlsx">Hogs</a>'
    raw_dir = tmp_path / "raw"

    sync_downloads(
        raw_dir=raw_dir,
        product_url=PRODUCT_URL,
        client=_make_client(page, {"Hogs.xlsx": b"v1"}, page_url=PRODUCT_URL),
    )
    sha_v1 = next(iter(load_manifest(raw_dir / "manifest.json").values())).sha256

    sync_downloads(
        raw_dir=raw_dir,
        product_url=PRODUCT_URL,
        client=_make_client(page, {"Hogs.xlsx": b"v2-different-bytes"}, page_url=PRODUCT_URL),
    )
    new_entry = next(iter(load_manifest(raw_dir / "manifest.json").values()))

    assert new_entry.sha256 != sha_v1
    assert (raw_dir / "Hogs.xlsx").read_bytes() == b"v2-different-bytes"
    assert not list(raw_dir.glob("*.part"))


def test_sync_downloads_persists_manifest_across_runs(tmp_path: Path) -> None:
    files = {"Hogs.xlsx": b"hogs"}
    page = '<a href="/files/Hogs.xlsx">Hogs</a>'
    raw_dir = tmp_path / "raw"

    sync_downloads(
        raw_dir=raw_dir,
        product_url=PRODUCT_URL,
        client=_make_client(page, files, page_url=PRODUCT_URL),
    )
    # Reload from disk through a fresh function call
    reloaded = load_manifest(raw_dir / "manifest.json")
    assert "https://example.test/files/Hogs.xlsx" in reloaded
