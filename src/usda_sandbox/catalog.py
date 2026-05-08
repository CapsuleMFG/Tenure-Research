"""Catalog of USDA series definitions.

Each :class:`SeriesDefinition` describes one logical time series and the
spreadsheet coordinates needed to extract it. The catalog is persisted as a
JSON list at ``data/catalog.json``; entries are added by hand after inspecting
the actual XLSX files in ``data/raw/``.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

VALID_FREQUENCIES = frozenset({"monthly", "quarterly", "annual"})


class SeriesDefinition(BaseModel):
    """One row of the catalog.

    ``date_column`` is either a single column letter (e.g. ``"A"``) for
    spreadsheets that put a real date in one cell, or two letters joined by
    ``+`` (e.g. ``"A+B"``) for WASDE-style sheets that split year and quarter
    label across two columns. ``value_columns`` is a list to accommodate
    future series that aggregate across columns; today every entry uses a
    single column letter.
    """

    model_config = ConfigDict(extra="forbid")

    series_id: str = Field(..., min_length=1)
    series_name: str
    commodity: str
    metric: str
    unit: str
    frequency: str
    source_file: str
    source_sheet: str
    header_rows_to_skip: int = Field(..., ge=0)
    value_columns: list[str] = Field(..., min_length=1)
    date_column: str = Field(..., min_length=1)
    notes: str = ""

    @field_validator("frequency")
    @classmethod
    def _check_frequency(cls, v: str) -> str:
        if v not in VALID_FREQUENCIES:
            raise ValueError(
                f"frequency must be one of {sorted(VALID_FREQUENCIES)}, got {v!r}"
            )
        return v

    @field_validator("value_columns", "date_column")
    @classmethod
    def _check_column_letters(cls, v: list[str] | str) -> list[str] | str:
        cols = v if isinstance(v, list) else [v]
        for raw in cols:
            for piece in raw.split("+"):
                if not piece.strip().isalpha():
                    raise ValueError(
                        f"Column reference must be alphabetic letters (got {piece!r})"
                    )
        return v


def load_catalog(path: str | Path) -> list[SeriesDefinition]:
    """Read the on-disk catalog into a list of :class:`SeriesDefinition`."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Catalog file {path} must contain a JSON array")
    return [SeriesDefinition.model_validate(entry) for entry in raw]


def save_catalog(path: str | Path, catalog: list[SeriesDefinition]) -> None:
    """Write a catalog list back to disk as pretty-printed JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps([s.model_dump() for s in catalog], indent=2) + "\n",
        encoding="utf-8",
    )
