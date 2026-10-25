"""Source-grounded audit ledger for FAULTLINE.

Source of truth: FAULTLINE_Development_Plan.md Section 3 (Hard Rules) & Section 4.
Every number used in analysis must map to a verified LedgerEntry with a source SHA-256.
Missing inputs result in explicit abstain records, never zero-filling or guessing.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from faultline.schemas import LedgerEntry


def compute_file_sha256(file_path: Path | str) -> str:
    """Calculate SHA-256 hash of a file on disk."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Source file not found at {path}")

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_bytes_sha256(content: bytes) -> str:
    """Calculate SHA-256 hash of bytes in memory."""
    return hashlib.sha256(content).hexdigest()


class FinancialLedger:
    """An immutable-by-default collection of verified financial records and abstain logs."""

    def __init__(self) -> None:
        self._entries: dict[str, LedgerEntry] = {}
        self._next_id: int = 1
        self._abstain_log: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterable[LedgerEntry]:
        return iter(self._entries.values())

    def add_entry(self, entry: LedgerEntry) -> str:
        """Add a validated LedgerEntry to the ledger.

        Raises ValueError if an entry with the same ID already exists.
        """
        if entry.id in self._entries:
            raise ValueError(f"LedgerEntry with id '{entry.id}' already exists")
        self._entries[entry.id] = entry
        return entry.id

    def create_and_add_entry(
        self,
        metric: str,
        value: float,
        unit: str,
        period: str,
        period_kind: str,
        source_url: str,
        source_type: str,
        snippet: str,
        source_sha256: str,
        doc_page: int | None = None,
        retrieved_at: datetime | None = None,
        entry_id: str | None = None,
    ) -> LedgerEntry:
        """Factory method to construct, validate, and register a new LedgerEntry."""
        if entry_id is None:
            while f"L{self._next_id:04d}" in self._entries:
                self._next_id += 1
            entry_id = f"L{self._next_id:04d}"
            self._next_id += 1

        if retrieved_at is None:
            retrieved_at = datetime.now(timezone.utc)

        entry = LedgerEntry(
            id=entry_id,
            metric=metric,
            value=value,
            unit=unit,
            period=period,
            period_kind=period_kind,  # type: ignore[arg-type]
            source_url=source_url,
            source_type=source_type,  # type: ignore[arg-type]
            doc_page=doc_page,
            snippet=snippet,
            source_sha256=source_sha256,
            retrieved_at=retrieved_at,
        )
        self.add_entry(entry)
        return entry

    def get(self, entry_id: str) -> LedgerEntry | None:
        """Look up an entry by unique ledger ID."""
        return self._entries.get(entry_id)

    def get_by_metric_and_period(self, metric: str, period: str) -> LedgerEntry | None:
        """Look up a specific metric in a given period.

        Returns None if not disclosed (never invents a zero).
        """
        for entry in self._entries.values():
            if entry.metric == metric and entry.period == period:
                return entry
        return None

    def get_series(self, metric: str) -> dict[str, LedgerEntry]:
        """Return a mapping of period -> LedgerEntry for a specific metric."""
        return {
            entry.period: entry
            for entry in self._entries.values()
            if entry.metric == metric
        }

    def all_entries(self) -> list[LedgerEntry]:
        """Return all ledger entries in insertion order."""
        return list(self._entries.values())

    def check_availability(
        self, required_metrics: list[str], period: str
    ) -> tuple[bool, list[str]]:
        """Check if all required metrics are present for a given period.

        Returns:
            (is_complete, missing_metrics)
        """
        missing: list[str] = []
        for metric in required_metrics:
            if self.get_by_metric_and_period(metric, period) is None:
                missing.append(metric)
        return len(missing) == 0, missing

    def log_abstain(
        self,
        engine: str,
        check_name: str,
        reason: str,
        period: str | None = None,
        missing_fields: list[str] | None = None,
    ) -> dict[str, Any]:
        """Record an explicit abstain event when data is missing (Rule R4).

        Never zero-fills or guesses missing data.
        """
        record: dict[str, Any] = {
            "engine": engine,
            "check": check_name,
            "period": period,
            "missing_fields": missing_fields or [],
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._abstain_log.append(record)
        return record

    @property
    def abstain_log(self) -> list[dict[str, Any]]:
        """Return copy of logged abstain events."""
        return list(self._abstain_log)

    def verify_entry_source(self, entry_id: str, file_path: Path | str) -> bool:
        """Validate whether a cached local source file matches the SHA-256 recorded in the entry."""
        entry = self.get(entry_id)
        if entry is None:
            raise KeyError(f"Entry ID '{entry_id}' not found in ledger")

        actual_sha256 = compute_file_sha256(file_path)
        return actual_sha256 == entry.source_sha256

    def save_to_json(self, file_path: Path | str) -> None:
        """Save entire ledger and abstain log to a JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "entries": [entry.model_dump(mode="json") for entry in self._entries.values()],
            "abstain_log": self._abstain_log,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_json(cls, file_path: Path | str) -> FinancialLedger:
        """Load ledger and abstain log from a JSON file."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Ledger file not found at {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        ledger = cls()
        for entry_dict in data.get("entries", []):
            entry = LedgerEntry.model_validate(entry_dict)
            ledger.add_entry(entry)

        ledger._abstain_log = data.get("abstain_log", [])
        return ledger
