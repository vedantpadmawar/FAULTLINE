"""Drift Engine: Note normalisation and disclosure drift detection.

Source of truth: FAULTLINE_Development_Plan.md Section 5.3 & Hard Rules R3, R9.
Computes:
- Normalisation: lowercase, collapse whitespace, strip numbering, currency, digits, drop headers/footers
- headings_removed: normalised headings in last year not in this year (fuzzy match ratio < 0.9)
- headings_added: the reverse
- redefined: matched paragraphs (heading match) whose difflib ratio < 0.85 after normalisation
- cosmetic: changes that vanish after normalisation (ignored)
- Output: list of DriftItem schemas with page references and similarity
- Hard Rule R9: Disclosure drift alone can never publish a finding.
"""

from __future__ import annotations

from dataclasses import dataclass
import difflib
import re
from typing import Callable

from faultline.schemas import DriftItem

# Regex patterns for normalization
HEADER_FOOTER_PATTERNS = [
    re.compile(r"^page\s+\d+(\s+of\s+\d+)?$", re.IGNORECASE),
    re.compile(r"^annual\s+report\s+20\d\d(-\d\d)?$", re.IGNORECASE),
    re.compile(r"^notes\s+to\s+(the\s+)?financial\s+statements$", re.IGNORECASE),
]

CURRENCY_PATTERN = re.compile(
    r"\b(inr|rs\.?|rupees?|usd|crores?|cr\.?|lakhs?|mn|bn)\b|[₹$€£]",
    re.IGNORECASE,
)
DIGITS_PATTERN = re.compile(r"\b\d+([,\.]\d+)*%?\b|\d+")
PREFIX_NUMBERING_PATTERN = re.compile(
    r"^(note\s+)?(\d+(\.\d+)*|[a-z]\)|\([a-z0-9]+\))\s*[:\.\-]?\s*",
    re.IGNORECASE,
)
WHITESPACE_PATTERN = re.compile(r"\s+")
PUNCTUATION_PATTERN = re.compile(r"[^\w\s]")


@dataclass
class NoteSection:
    """A distinct disclosure note extracted from an annual report."""

    heading: str
    text: str
    page: int | None = None


def normalize_text(text: str) -> str:
    """Normalize note text for drift detection.

    Steps (Section 5.3):
    1. Lowercase
    2. Strip header/footer lines
    3. Strip numbering prefixes and all digits/currency
    4. Strip non-word punctuation
    5. Collapse whitespace
    """
    if not text:
        return ""

    lines = text.splitlines()
    filtered_lines = []
    for line in lines:
        stripped = line.strip()
        # Drop standalone header/footer lines
        if any(pat.match(stripped) for pat in HEADER_FOOTER_PATTERNS):
            continue
        filtered_lines.append(line)

    combined = " ".join(filtered_lines).lower()
    # Strip currency and digits
    combined = CURRENCY_PATTERN.sub(" ", combined)
    combined = DIGITS_PATTERN.sub(" ", combined)
    # Strip punctuation
    combined = PUNCTUATION_PATTERN.sub(" ", combined)
    # Collapse whitespace
    normalized = WHITESPACE_PATTERN.sub(" ", combined).strip()
    return normalized


def normalize_heading(heading: str) -> str:
    """Normalize note heading: remove note prefixes and numbering, lowercase, collapse whitespace."""
    if not heading:
        return ""
    h = heading.strip().lower()
    # Remove prefix like "Note 14.2: " or "18. "
    h = PREFIX_NUMBERING_PATTERN.sub("", h)
    # Remove remaining numbers and currency
    h = CURRENCY_PATTERN.sub(" ", h)
    h = DIGITS_PATTERN.sub(" ", h)
    h = PUNCTUATION_PATTERN.sub(" ", h)
    return WHITESPACE_PATTERN.sub(" ", h).strip()


def fuzzy_similarity(s1: str, s2: str) -> float:
    """Compute SequenceMatcher ratio between two normalized strings."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return float(difflib.SequenceMatcher(None, s1, s2).ratio())


def detect_disclosure_drift(
    last_year_notes: list[NoteSection],
    this_year_notes: list[NoteSection],
    heading_threshold: float = 0.90,
    redefinition_threshold: float = 0.85,
) -> list[DriftItem]:
    """Detect removed, added, and redefined disclosure notes between consecutive annual reports.

    Args:
        last_year_notes: List of NoteSection objects from previous annual report.
        this_year_notes: List of NoteSection objects from current annual report.
        heading_threshold: Cutoff for heading match (default 0.90).
        redefinition_threshold: Cutoff below which matched paragraphs are redefined (default 0.85).

    Returns:
        List of DriftItem objects identifying material changes.
    """
    drift_items: list[DriftItem] = []

    # Map normalized headings
    ly_norm = [(sec, normalize_heading(sec.heading), normalize_text(sec.text)) for sec in last_year_notes]
    ty_norm = [(sec, normalize_heading(sec.heading), normalize_text(sec.text)) for sec in this_year_notes]

    matched_ty_indices: set[int] = set()
    matched_ly_indices: set[int] = set()

    # 1. Match headings from last year to this year
    for ly_idx, (ly_sec, ly_h_norm, ly_t_norm) in enumerate(ly_norm):
        best_sim = 0.0
        best_ty_idx = -1

        for ty_idx, (ty_sec, ty_h_norm, ty_t_norm) in enumerate(ty_norm):
            sim = fuzzy_similarity(ly_h_norm, ty_h_norm)
            if sim > best_sim:
                best_sim = sim
                best_ty_idx = ty_idx

        if best_sim < heading_threshold or best_ty_idx == -1:
            # Heading removed in this year's annual report
            drift_items.append(
                DriftItem(
                    kind="removed",
                    heading=ly_sec.heading,
                    text_before=ly_sec.text,
                    text_after=None,
                    similarity=float(round(best_sim, 4)),
                    page_before=ly_sec.page,
                    page_after=None,
                )
            )
        else:
            matched_ly_indices.add(ly_idx)
            matched_ty_indices.add(best_ty_idx)

            # Compare body text for redefinition
            ty_sec, ty_h_norm, ty_t_norm = ty_norm[best_ty_idx]
            text_sim = fuzzy_similarity(ly_t_norm, ty_t_norm)

            if text_sim < redefinition_threshold:
                drift_items.append(
                    DriftItem(
                        kind="redefined",
                        heading=ty_sec.heading,
                        text_before=ly_sec.text,
                        text_after=ty_sec.text,
                        similarity=float(round(text_sim, 4)),
                        page_before=ly_sec.page,
                        page_after=ty_sec.page,
                    )
                )

    # 2. Check for added headings in this year not in last year
    for ty_idx, (ty_sec, ty_h_norm, ty_t_norm) in enumerate(ty_norm):
        if ty_idx in matched_ty_indices:
            continue

        best_sim = 0.0
        for ly_idx, (ly_sec, ly_h_norm, ly_t_norm) in enumerate(ly_norm):
            sim = fuzzy_similarity(ty_h_norm, ly_h_norm)
            if sim > best_sim:
                best_sim = sim

        if best_sim < heading_threshold:
            drift_items.append(
                DriftItem(
                    kind="added",
                    heading=ty_sec.heading,
                    text_before=None,
                    text_after=ty_sec.text,
                    similarity=float(round(best_sim, 4)),
                    page_before=None,
                    page_after=ty_sec.page,
                )
            )

    return drift_items


def summarize_drift_item(
    item: DriftItem,
    llm_summarizer: Callable[[str], str] | None = None,
) -> str:
    """Summarize a DriftItem in qualitative prose without adding numbers (Rule R3)."""
    if llm_summarizer is not None:
        prompt = (
            f"Summarize the following disclosure change in concise qualitative language.\n"
            f"CRITICAL RULE: DO NOT include any numbers, figures, or quantities.\n"
            f"Type: {item.kind}\nHeading: {item.heading}\n"
            f"Text before: {item.text_before or 'None'}\n"
            f"Text after: {item.text_after or 'None'}"
        )
        summary = llm_summarizer(prompt)
        # Strip any accidental numbers generated by LLM (Rule R3 / M04)
        summary_clean = DIGITS_PATTERN.sub("", summary)
        return summary_clean.strip()

    # Deterministic template summary (no computed numbers)
    if item.kind == "removed":
        pg = f" (page {item.page_before})" if item.page_before else ""
        return f"Disclosure note '{item.heading}'{pg} was dropped from annual report notes."
    elif item.kind == "added":
        pg = f" (page {item.page_after})" if item.page_after else ""
        return f"New disclosure note '{item.heading}'{pg} was added to annual report notes."
    else:  # redefined
        return f"Accounting definitions under note '{item.heading}' were materially altered."


def can_drift_publish_alone(drift_items: list[DriftItem], numeric_fault_flagged: bool) -> bool:
    """Hard Rule R9: Disclosure drift alone can never publish a finding.

    It can only corroborate a numeric fault or be reported as a supplementary note.
    """
    if not numeric_fault_flagged:
        return False
    return len(drift_items) > 0
