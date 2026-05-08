"""Discover and download USDA ERS livestock data files.

The ERS "Livestock and Meat Domestic Data" product page links to a handful of
XLSX (and occasionally ZIP) downloads. This module fetches that page, finds
those links, and mirrors them into ``data/raw/`` while maintaining a manifest
keyed on URL so subsequent runs are idempotent and replace files only when
their content has actually changed.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import httpx

ERS_PRODUCT_URL = (
    "https://www.ers.usda.gov/data-products/livestock-and-meat-domestic-data"
)
DEFAULT_RAW_DIR = Path("data/raw")
MANIFEST_FILENAME = "manifest.json"
DOWNLOADABLE_EXTENSIONS = (".xlsx", ".zip")
_HREF_RE = re.compile(r"""href\s*=\s*["']([^"']+)["']""", re.IGNORECASE)


@dataclass(frozen=True)
class ManifestEntry:
    """One row of the download manifest, keyed externally on ``url``."""

    url: str
    filename: str
    sha256: str
    downloaded_at: str


def discover_download_urls(html: str, base_url: str = ERS_PRODUCT_URL) -> list[str]:
    """Extract absolute XLSX/ZIP URLs from a product-page HTML body.

    Order is preserved; duplicates are dropped. Relative hrefs are resolved
    against ``base_url`` so callers always receive fully-qualified URLs.
    """
    seen: set[str] = set()
    out: list[str] = []
    for match in _HREF_RE.finditer(html):
        href = match.group(1).strip()
        # Strip fragment and query for extension testing
        path_only = urlparse(href).path.lower()
        if not path_only.endswith(DOWNLOADABLE_EXTENSIONS):
            continue
        absolute = urljoin(base_url, href)
        if absolute in seen:
            continue
        seen.add(absolute)
        out.append(absolute)
    return out


def filename_from_url(url: str) -> str:
    """Recover the on-disk filename from a download URL."""
    path = urlparse(url).path
    name = unquote(Path(path).name)
    if not name:
        raise ValueError(f"Cannot derive filename from URL: {url!r}")
    return name


def sha256_file(path: Path) -> str:
    """Stream-hash a file with SHA-256."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path: Path) -> dict[str, ManifestEntry]:
    """Read the manifest at ``path`` into a ``{url: ManifestEntry}`` dict."""
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {url: ManifestEntry(**entry) for url, entry in raw.items()}


def save_manifest(path: Path, manifest: dict[str, ManifestEntry]) -> None:
    """Write ``manifest`` to ``path`` as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {url: asdict(entry) for url, entry in manifest.items()}
    path.write_text(
        json.dumps(serializable, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@contextmanager
def _client_or_default(client: httpx.Client | None) -> Iterator[httpx.Client]:
    if client is not None:
        yield client
        return
    with httpx.Client(follow_redirects=True, timeout=120.0) as owned:
        yield owned


def _download_to(url: str, dest: Path, client: httpx.Client) -> str:
    """Stream-download ``url`` to ``dest``, returning the file's SHA-256."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    with client.stream("GET", url) as resp:
        resp.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in resp.iter_bytes():
                if not chunk:
                    continue
                h.update(chunk)
                fh.write(chunk)
    return h.hexdigest()


def fetch_product_page(
    url: str = ERS_PRODUCT_URL,
    *,
    client: httpx.Client | None = None,
) -> str:
    """Fetch the ERS product page HTML."""
    with _client_or_default(client) as c:
        resp = c.get(url)
        resp.raise_for_status()
        return resp.text


def sync_downloads(
    raw_dir: Path | str = DEFAULT_RAW_DIR,
    *,
    product_url: str = ERS_PRODUCT_URL,
    client: httpx.Client | None = None,
    now: datetime | None = None,
) -> dict[str, ManifestEntry]:
    """Mirror every XLSX/ZIP linked from the product page into ``raw_dir``.

    Idempotency contract: each URL is downloaded into a temporary ``.part``
    file and hashed. If the resulting SHA-256 matches the existing manifest
    entry, the temporary file is discarded and the manifest is left untouched.
    Otherwise the destination file is atomically replaced and the manifest
    entry (with a fresh ``downloaded_at`` timestamp) is updated.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_dir / MANIFEST_FILENAME
    manifest = load_manifest(manifest_path)

    with _client_or_default(client) as c:
        page_resp = c.get(product_url)
        page_resp.raise_for_status()
        urls = discover_download_urls(page_resp.text, base_url=product_url)

        for url in urls:
            filename = filename_from_url(url)
            dest = raw_dir / filename
            tmp = raw_dir / (filename + ".part")
            try:
                new_sha = _download_to(url, tmp, c)
            except httpx.HTTPError:
                if tmp.exists():
                    tmp.unlink()
                raise

            existing = manifest.get(url)
            unchanged = (
                existing is not None
                and dest.exists()
                and existing.sha256 == new_sha
            )
            if unchanged:
                tmp.unlink()
                continue

            if dest.exists():
                dest.unlink()
            tmp.replace(dest)
            manifest[url] = ManifestEntry(
                url=url,
                filename=filename,
                sha256=new_sha,
                downloaded_at=(now or datetime.now(UTC)).isoformat(),
            )

    save_manifest(manifest_path, manifest)
    return manifest
