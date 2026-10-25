"""PDF table and financial statement extraction utilities.

Source of truth: FAULTLINE_Development_Plan.md Section 4 & Task T07.
Generic financial table and line extraction:
- Page-by-page text parsing with exact page numbers
- Financial row and column extraction
- Canonical metric pattern matching
- Preservation of exact source snippets and confidence scoring
- Zero company-specific hardcoding (Rule R1)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import io
from pathlib import Path
import re
from typing import Any

import pypdf

# Canonical patterns for financial metrics in Indian banking & NBFC reporting
METRIC_ROW_PATTERNS: dict[str, list[re.Pattern]] = {
    "gross_advances": [
        re.compile(r"\b(gross\s+advances|gross\s+loans?|total\s+gross\s+advances|gross\s+loan\s+book|loans?\s+(&|and)\s+advances\s*\(gross\))\b", re.IGNORECASE),
        re.compile(r"\b(total\s+advances|total\s+loans?)\b", re.IGNORECASE),
    ],
    "gnpa_amount": [
        re.compile(r"\b(gross\s+npa\s*(amount)?|gross\s+non[-\s]performing\s*(assets?|advances?)?|stage\s*3\s+gross)\b", re.IGNORECASE),
        re.compile(r"\b(gnpa\s*(amount)?)\b", re.IGNORECASE),
    ],
    "nnpa_amount": [
        re.compile(r"\b(net\s+npa\s*(amount)?|net\s+non[-\s]performing\s*(assets?|advances?)?|nnpa\s*(amount)?)\b", re.IGNORECASE),
    ],
    "writeoffs_period": [
        re.compile(r"\b(write[-\s]offs?(\s+during\s+the\s+period)?|written\s+off|technical\s+write[-\s]offs?)\b", re.IGNORECASE),
        re.compile(r"\b(reductions?\s+due\s+to\s+write[-\s]offs?)\b", re.IGNORECASE),
    ],
    "provisions_total": [
        re.compile(r"\b(total\s+provisions?|provisions?\s+held|provision\s+for\s+(npas?|bad\s+debts?|loan\s+losses?)|ecl\s+provisions?)\b", re.IGNORECASE),
    ],
    "stage1_assets": [
        re.compile(r"\b(stage\s*(1|i)\s*(loans?|assets?|advances?))\b", re.IGNORECASE),
    ],
    "stage2_assets": [
        re.compile(r"\b(stage\s*(2|ii)\s*(loans?|assets?|advances?))\b", re.IGNORECASE),
    ],
    "stage3_assets": [
        re.compile(r"\b(stage\s*(3|iii)\s*(loans?|assets?|advances?))\b", re.IGNORECASE),
    ],
    "borrowings_bank": [
        re.compile(r"\b(borrowings?\s+from\s+banks?|term\s+loans?\s+from\s+banks?|bank\s+borrowings?|refinance\s+from\s+banks?)\b", re.IGNORECASE),
    ],
    "borrowings_bonds": [
        re.compile(r"\b(debt\s+securities|non[-\s]convertible\s+debentures|ncds?|subordinated\s+debt|bonds)\b", re.IGNORECASE),
    ],
    "borrowings_deposits": [
        re.compile(r"\b(deposits?|public\s+deposits?|term\s+deposits?|fixed\s+deposits?)\b", re.IGNORECASE),
    ],
    "borrowings_securitisation": [
        re.compile(r"\b(securitisation|ptc|pass\s+through\s+certificates?|asset[-\s]backed\s+securities?)\b", re.IGNORECASE),
    ],
    "borrowings_foreign": [
        re.compile(r"\b(foreign\s+currency\s+borrowings?|ecb|external\s+commercial\s+borrowings?)\b", re.IGNORECASE),
    ],
    "borrowings_short_term": [
        re.compile(r"\b(commercial\s+papers?|short[-\s]term\s+borrowings?|borrowings?\s+repayable\s+on\s+demand|short[-\s]term\s+debt)\b", re.IGNORECASE),
    ],
    "total_borrowings": [
        re.compile(r"\b(total\s+borrowings?|total\s+debt(\s+securities)?|aggregate\s+borrowings?)\b", re.IGNORECASE),
    ],
    "assigned_loans_outstanding": [
        re.compile(r"\b(direct\s+assignments?|assigned\s+(loans?|portfolio)|off[-\s]balance\s+sheet\s+assigned)\b", re.IGNORECASE),
    ],
    "co_lending_aum": [
        re.compile(r"\b(co[-\s]lending(\s+aum)?|co[-\s]origination)\b", re.IGNORECASE),
    ],
    "credit_cost": [
        re.compile(r"\b(credit\s+costs?|annualised\s+credit\s+cost)\b", re.IGNORECASE),
    ],
}

# Regex to find numbers in financial tables (e.g. 15,200.50, 1,25,000.50, 51000.0, (450.00), -120.0, 3.5%)
NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])\(?(-?\d{1,2}(?:,\d{2})*(?:,\d{3})(?:\.\d+)?|-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?)\)?%?"
)


@dataclass
class RawPageContent:
    """Raw text and metadata extracted from a single PDF page."""

    page_number: int  # 1-indexed
    text: str


@dataclass
class ExtractedCandidate:
    """An extracted financial figure candidate with source evidence and confidence."""

    metric: str
    value: float
    unit: str  # "INR_cr", "pct", etc.
    period: str
    doc_page: int
    snippet: str
    confidence: float  # 0.0 to 1.0
    needs_review: bool
    review_reason: str | None = None


def extract_pdf_pages(file_source: Path | str | bytes | io.BytesIO) -> list[RawPageContent]:
    """Extract page text from a PDF document preserving 1-indexed page numbers."""
    if isinstance(file_source, (str, Path)):
        reader = pypdf.PdfReader(str(file_source))
    elif isinstance(file_source, bytes):
        reader = pypdf.PdfReader(io.BytesIO(file_source))
    else:
        reader = pypdf.PdfReader(file_source)

    pages: list[RawPageContent] = []
    for idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(RawPageContent(page_number=idx, text=text))
    return pages


def parse_numeric_cell(cell_str: str) -> float | None:
    """Parse a financial number string, handling parentheses as negative numbers and commas."""
    s = cell_str.strip()
    if not s or s in ["-", "—", "NA", "N.A.", "nil", "Nil"]:
        return None

    is_negative = False
    if s.startswith("(") and s.endswith(")"):
        is_negative = True
        s = s[1:-1].strip()
    elif s.startswith("-"):
        is_negative = True
        s = s[1:].strip()

    s = s.replace(",", "").rstrip("%")
    try:
        val = float(s)
        return -val if is_negative else val
    except ValueError:
        return None


def detect_unit_scale(text: str) -> str:
    """Detect reporting unit scale from page header or table text (default: INR_cr)."""
    lower = text.lower()
    if "lakh" in lower or "inr in lakh" in lower or "rs. in lakh" in lower:
        return "INR_lakh"
    elif "thousand" in lower:
        return "INR_thousand"
    elif "million" in lower or "mn" in lower:
        return "INR_million"
    elif "percent" in lower or "%" in lower:
        return "pct"
    return "INR_cr"


def match_metric_row(line: str) -> tuple[str | None, float]:
    """Match a table row label against standard metrics.

    Returns:
        (metric_name, confidence)
    """
    for metric, patterns in METRIC_ROW_PATTERNS.items():
        for pat in patterns:
            if pat.search(line):
                # Strong exact regex match
                return metric, 0.95
    return None, 0.0


def extract_candidates_from_page(
    page: RawPageContent,
    periods_header_map: dict[int, str] | None = None,
    default_period: str = "FY26-Q2",
) -> list[ExtractedCandidate]:
    """Extract candidate financial figures from a single page's text lines."""
    candidates: list[ExtractedCandidate] = []
    lines = page.text.splitlines()
    unit = detect_unit_scale(page.text[:500])  # check first few lines for header unit

    for line in lines:
        stripped = line.strip()
        if len(stripped) < 5:
            continue

        metric, base_conf = match_metric_row(stripped)
        if not metric:
            continue

        # Extract numeric tokens from the line
        numbers = NUMBER_PATTERN.findall(stripped)
        parsed_numbers = []
        for n_str in numbers:
            val = parse_numeric_cell(n_str)
            if val is not None:
                parsed_numbers.append((val, n_str))

        if not parsed_numbers:
            continue

        # If a single number is found on the row
        if len(parsed_numbers) == 1:
            val, raw_num = parsed_numbers[0]
            cand = ExtractedCandidate(
                metric=metric,
                value=val,
                unit="pct" if metric == "credit_cost" else unit,
                period=default_period,
                doc_page=page.page_number,
                snippet=stripped,
                confidence=base_conf,
                needs_review=False,
            )
            candidates.append(cand)
        else:
            # Multiple columns on the same line
            # If column period mapping is supplied, map by index
            if periods_header_map:
                for col_idx, (val, raw_num) in enumerate(parsed_numbers):
                    if col_idx in periods_header_map:
                        period_name = periods_header_map[col_idx]
                        candidates.append(
                            ExtractedCandidate(
                                metric=metric,
                                value=val,
                                unit="pct" if metric == "credit_cost" else unit,
                                period=period_name,
                                doc_page=page.page_number,
                                snippet=stripped,
                                confidence=base_conf * 0.9,
                                needs_review=False,
                            )
                        )
            else:
                # Ambiguous column assignment: pick primary column (usually latest period), flag for review
                val, raw_num = parsed_numbers[0]
                candidates.append(
                    ExtractedCandidate(
                        metric=metric,
                        value=val,
                        unit="pct" if metric == "credit_cost" else unit,
                        period=default_period,
                        doc_page=page.page_number,
                        snippet=stripped,
                        confidence=base_conf * 0.6,
                        needs_review=True,
                        review_reason=f"Multiple numbers {numbers} on line without column header mapping",
                    )
                )

    return candidates
