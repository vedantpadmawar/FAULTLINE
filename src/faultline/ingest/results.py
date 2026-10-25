"""Financial results document ingestion pipeline.

Source of truth: FAULTLINE_Development_Plan.md Section 4 & Task T07.
Coordinates:
- Source PDF caching with SHA-256 calculation (Rule R8)
- Page-by-page table extraction
- Period normalization (e.g., "Q2 FY26" -> "FY26-Q2")
- Validation and creation of LedgerEntry records
- Flagging of ambiguous/low-confidence figures for manual review
- Preserves absent metrics as missing (Rule R4: never zero-fills or guesses)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import io
from pathlib import Path
import re
from typing import Any

from faultline.ingest.pdf_tables import (
    ExtractedCandidate,
    RawPageContent,
    extract_candidates_from_page,
    extract_pdf_pages,
)
from faultline.ledger import FinancialLedger
from faultline.schemas import LedgerEntry, SUPPORTED_METRICS

# Period regex matching
QUARTER_REGEX = re.compile(r"\b(q[1-4])\s*(fy\s*20?\d\d|20?\d\d)\b|\b(fy\s*20?\d\d|20?\d\d)\s*(q[1-4])\b", re.IGNORECASE)
HALF_YEAR_REGEX = re.compile(r"\b(h[1-2])\s*(fy\s*20?\d\d|20?\d\d)\b", re.IGNORECASE)
DATE_END_REGEX = re.compile(
    r"\b(quarter|three\s+months?|period)\s+ended\s+([a-z]+|\d{1,2})[\s\.\-]+(\d{1,2}|[a-z]+)[\s\.\-]+(20\d\d)\b",
    re.IGNORECASE,
)


@dataclass
class IngestionReport:
    """Detailed summary of the financial report ingestion process."""

    source_path: str
    source_sha256: str
    cached_path: str
    total_pages: int
    entries_created: list[LedgerEntry]
    review_candidates: list[ExtractedCandidate]
    disclosed_metrics: list[str]
    missing_metrics: list[str]


def compute_sha256(content: bytes) -> str:
    """Compute hex SHA-256 hash of byte content."""
    return hashlib.sha256(content).hexdigest()


def normalize_period_label(raw_str: str) -> tuple[str, str]:
    """Normalize raw period strings to canonical format (e.g., 'FY26-Q2', 'Q').

    Returns:
        (canonical_period, period_kind) where period_kind is 'Q', 'H', or 'Y'.
    """
    clean = raw_str.strip().upper()

    # Match Q patterns: e.g. "Q2 FY26", "FY26-Q2", "Q2FY26"
    q_match = re.search(r"Q([1-4])[\s\-_/]*FY?[\s\-_/]*(20)?(\d\d)", clean)
    if q_match:
        q_num, _, yr = q_match.groups()
        return f"FY{yr}-Q{q_num}", "Q"

    fy_q_match = re.search(r"FY?[\s\-_/]*(20)?(\d\d)[\s\-_/]*Q([1-4])", clean)
    if fy_q_match:
        _, yr, q_num = fy_q_match.groups()
        return f"FY{yr}-Q{q_num}", "Q"

    # Match H patterns: e.g. "H1 FY26"
    h_match = re.search(r"H([1-2])[\s\-_/]*FY?[\s\-_/]*(20)?(\d\d)", clean)
    if h_match:
        h_num, _, yr = h_match.groups()
        return f"FY{yr}-H{h_num}", "H"

    # Match Full Year: e.g. "FY25", "FY 2025"
    y_match = re.search(r"FY?[\s\-_/]*(20)?(\d\d)$", clean)
    if y_match:
        _, yr = y_match.groups()
        return f"FY{yr}-Y", "Y"

    # Fallback default
    return clean, "Q"


def cache_source_document(
    file_bytes: bytes,
    original_name: str,
    cache_dir: Path | str = "data/cache",
) -> tuple[str, Path]:
    """Save raw document bytes to cache directory named by its SHA-256 hash (Rule R8)."""
    dir_path = Path(cache_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    sha = compute_sha256(file_bytes)
    extension = Path(original_name).suffix or ".pdf"
    cached_file = dir_path / f"{sha}{extension}"

    if not cached_file.exists():
        cached_file.write_bytes(file_bytes)

    return sha, cached_file


def ingest_financial_pdf(
    file_source: Path | str | bytes | io.BytesIO,
    source_url: str = "file://local/filing.pdf",
    default_period: str = "FY26-Q2",
    ledger: FinancialLedger | None = None,
    cache_dir: Path | str = "data/cache",
    confidence_threshold: float = 0.80,
    periods_header_map: dict[int, str] | None = None,
) -> IngestionReport:
    """Ingest a financial statement PDF, extract proof-carrying numbers, and populate ledger.

    Args:
        file_source: Path to PDF or raw bytes.
        source_url: Public source URL for proof trail.
        default_period: Canonical period name if not auto-detected from headers.
        ledger: Optional FinancialLedger instance to populate directly.
        cache_dir: Directory to cache immutable PDF copy with SHA-256.
        confidence_threshold: Cutoff below which extractions are flagged for review.
        periods_header_map: Optional column index to period mapping for multi-period tables.

    Returns:
        IngestionReport summarizing created ledger entries and review candidates.
    """
    if ledger is None:
        ledger = FinancialLedger()

    # 1. Read bytes and cache source (Rule R8)
    if isinstance(file_source, (str, Path)):
        p = Path(file_source)
        if not p.is_file():
            raise FileNotFoundError(f"Financial PDF not found at {p}")
        file_bytes = p.read_bytes()
        original_name = p.name
    elif isinstance(file_source, bytes):
        file_bytes = file_source
        original_name = "upload.pdf"
    else:
        file_bytes = file_source.read()
        original_name = "stream.pdf"

    source_sha256, cached_path = cache_source_document(
        file_bytes=file_bytes,
        original_name=original_name,
        cache_dir=cache_dir,
    )

    # 2. Extract text pages
    pages = extract_pdf_pages(io.BytesIO(file_bytes))
    total_pages = len(pages)

    canonical_period, period_kind = normalize_period_label(default_period)

    # 3. Extract candidate rows across all pages
    all_candidates: list[ExtractedCandidate] = []
    for page in pages:
        cands = extract_candidates_from_page(
            page=page,
            periods_header_map=periods_header_map,
            default_period=canonical_period,
        )
        all_candidates.extend(cands)

    # 4. Filter and build LedgerEntry objects
    entries_created: list[LedgerEntry] = []
    review_candidates: list[ExtractedCandidate] = []
    disclosed_metrics_set: set[str] = set()

    for cand in all_candidates:
        if cand.needs_review or cand.confidence < confidence_threshold:
            review_candidates.append(cand)
            continue

        # Prevent duplicate entries for same metric and period from same document
        existing = ledger.get_by_metric_and_period(cand.metric, cand.period)
        if existing is not None:
            continue

        entry = ledger.create_and_add_entry(
            metric=cand.metric,
            value=cand.value,
            unit=cand.unit,
            period=cand.period,
            period_kind=period_kind,
            source_url=source_url,
            source_type="filing",
            snippet=cand.snippet,
            source_sha256=source_sha256,
            doc_page=cand.doc_page,
            retrieved_at=datetime.now(timezone.utc),
        )
        entries_created.append(entry)
        disclosed_metrics_set.add(cand.metric)

    # 5. Record missing metrics (Rule R4: keep unavailable values missing, never fabricate)
    missing_metrics = sorted(list(SUPPORTED_METRICS - disclosed_metrics_set))

    return IngestionReport(
        source_path=str(original_name),
        source_sha256=source_sha256,
        cached_path=str(cached_path),
        total_pages=total_pages,
        entries_created=entries_created,
        review_candidates=review_candidates,
        disclosed_metrics=sorted(list(disclosed_metrics_set)),
        missing_metrics=missing_metrics,
    )
