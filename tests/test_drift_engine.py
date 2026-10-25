"""Unit tests for the Disclosure Drift Engine (Section 5.3 of FAULTLINE_Development_Plan.md).

Tests:
1. Text and heading normalization (numbers, currency, header/footer lines stripped).
2. Detection of removed headings (ratio < 0.90).
3. Detection of added headings (ratio < 0.90).
4. Detection of materially redefined paragraphs (ratio < 0.85).
5. Immunity to cosmetic formatting, simple renumbering, and routine number updates.
6. Canary note pairs benchmark: evaluates recall (M12) and false alarm rate (M13).
7. Qualitative summarization without numbers (Rule R3).
8. Hard Rule R9 enforcement: drift cannot publish alone without numeric fault corroboration.
"""

import sys
from pathlib import Path
import pytest

# Ensure src/ is importable
src_path = str(Path(__file__).resolve().parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from faultline.engines.drift import (
    DriftItem,
    NoteSection,
    can_drift_publish_alone,
    detect_disclosure_drift,
    normalize_heading,
    normalize_text,
    summarize_drift_item,
)
from eval.canary.drift import generate_synthetic_note_pairs


def test_text_normalization():
    """Verify normalization strips headers, numbers, currency, and collapses whitespace."""
    raw = (
        "Annual Report 2025\n"
        "Page 12 of 90\n"
        "Note 14.1: Gross advances stood at INR 15,200.50 cr (Rs. 15,200.50 crore).\n"
        "Provisions were $450 mn."
    )
    norm = normalize_text(raw)

    # All numbers, currency, and headers must be stripped
    assert "annual report" not in norm
    assert "page 12" not in norm
    assert "15200" not in norm
    assert "450" not in norm
    assert "inr" not in norm
    assert "rs" not in norm
    assert "$" not in norm
    assert "gross advances stood at" in norm
    assert "provisions were" in norm


def test_heading_normalization_renumbering():
    """Verify heading normalization removes note numbering prefixes."""
    h1 = "Note 14.2: Movement of Non-Performing Assets"
    h2 = "15.3 Movement of Non-Performing Assets"

    norm1 = normalize_heading(h1)
    norm2 = normalize_heading(h2)

    assert norm1 == "movement of non performing assets"
    assert norm2 == "movement of non performing assets"
    assert norm1 == norm2


def test_cosmetic_edits_ignored():
    """Cosmetic changes (renumbering, formatting, number updates) are ignored."""
    ly = [
        NoteSection(
            heading="Note 12.1: Employee Benefits and Remuneration",
            text="Employee benefit expenses during the period were INR 450.0 cr.",
            page=50,
        ),
    ]
    # This year: renumbered to 13.2, updated from 450 to 520, extra spacing and casing
    ty = [
        NoteSection(
            heading="13.2 EMPLOYEE BENEFITS AND REMUNERATION",
            text="Employee benefit expenses during the period were Rs. 520.0 crore.",
            page=54,
        ),
    ]

    items = detect_disclosure_drift(ly, ty)
    # Must NOT be detected as drift (redefined, removed, or added)
    assert len(items) == 0


def test_removed_heading_detected():
    """A heading present in last year and missing in this year is flagged as 'removed'."""
    ly = [
        NoteSection(
            heading="Note 1: Corporate Information",
            text="General information about corporate structure.",
            page=10,
        ),
        NoteSection(
            heading="Note 34: Movements of Restructured Standard Assets",
            text="Detailed breakdown of restructured assets under RBI resolution framework.",
            page=88,
        ),
    ]
    ty = [
        NoteSection(
            heading="Note 1: Corporate Information",
            text="General information about corporate structure.",
            page=10,
        ),
    ]

    items = detect_disclosure_drift(ly, ty)
    assert len(items) == 1
    assert items[0].kind == "removed"
    assert items[0].heading == "Note 34: Movements of Restructured Standard Assets"
    assert items[0].page_before == 88
    assert items[0].page_after is None


def test_added_heading_detected():
    """A new heading present in this year and absent in last year is flagged as 'added'."""
    ly = [
        NoteSection(
            heading="Note 1: Corporate Information",
            text="General information.",
            page=10,
        ),
    ]
    ty = [
        NoteSection(
            heading="Note 1: Corporate Information",
            text="General information.",
            page=10,
        ),
        NoteSection(
            heading="Note 40: Digital Lending Partner Disclosures",
            text="Details of digital lending applications and default loss guarantee contracts.",
            page=115,
        ),
    ]

    items = detect_disclosure_drift(ly, ty)
    assert len(items) == 1
    assert items[0].kind == "added"
    assert "Digital Lending Partner" in items[0].heading
    assert items[0].page_after == 115


def test_material_redefinition_detected():
    """A note whose body text is materially altered (similarity < 0.85) is flagged as 'redefined'."""
    ly = [
        NoteSection(
            heading="Note 18: Stage 2 Staging Criteria",
            text="Accounts past due by 30 days are mandatorily transferred to Stage 2 with individual cash flow assessment.",
            page=70,
        ),
    ]
    ty = [
        NoteSection(
            heading="Note 18: Stage 2 Staging Criteria",
            text="Accounts are classified based on collective management judgment using forward-looking macroeconomic models without rigid 30-day triggers.",
            page=72,
        ),
    ]

    items = detect_disclosure_drift(ly, ty)
    assert len(items) == 1
    assert items[0].kind == "redefined"
    assert items[0].similarity is not None and items[0].similarity < 0.85
    assert items[0].page_before == 70
    assert items[0].page_after == 72


def test_canary_drift_recall_and_false_alarm():
    """Benchmark against synthetic note pairs (evaluating M12 recall and M13 false alarm)."""
    ly_notes, ty_notes, ground_truth = generate_synthetic_note_pairs()
    detected_items = detect_disclosure_drift(ly_notes, ty_notes)

    # True positives: GT_01 (removed), GT_02 (added), GT_03 (redefined)
    detected_kinds = {item.kind for item in detected_items}
    assert "removed" in detected_kinds
    assert "added" in detected_kinds
    assert "redefined" in detected_kinds

    # Expected count of genuine drift items is exactly 3
    assert len(detected_items) == 3

    # Negatives: GT_04 (renumbering) and GT_05 (number updates) must NOT be detected
    flagged_headings = [item.heading for item in detected_items]
    assert not any("Statutory Reserve" in h for h in flagged_headings)
    assert not any("Related Party" in h for h in flagged_headings)


def test_drift_item_summarizer_without_numbers():
    """Verify that summarization strictly adheres to Rule R3 (no numbers)."""
    item = DriftItem(
        kind="removed",
        heading="Note 24: Movements of Stage 2 Assets",
        text_before="Stage 2 loans were 1,500 cr.",
        text_after=None,
        similarity=0.0,
        page_before=84,
        page_after=None,
    )

    summary = summarize_drift_item(item)
    # Default template summary
    assert "dropped from annual report notes" in summary

    # Custom LLM summarizer mock that accidentally produced numbers
    mock_llm = lambda prompt: "Management omitted Note 24 on page 84 which previously disclosed 1500 cr."
    sanitized_summary = summarize_drift_item(item, llm_summarizer=mock_llm)

    # Accidental numbers must be sanitized out
    assert "1500" not in sanitized_summary
    assert "84" not in sanitized_summary


def test_rule_r9_drift_cannot_publish_alone():
    """Rule R9: Disclosure drift alone can NEVER publish a finding without numeric fault."""
    drift_items = [
        DriftItem(kind="removed", heading="Dropped Note", page_before=50),
        DriftItem(kind="redefined", heading="Altered Note", page_before=60, page_after=62),
    ]

    # Numeric fault NOT flagged -> CANNOT publish
    assert can_drift_publish_alone(drift_items, numeric_fault_flagged=False) is False

    # Numeric fault IS flagged -> CAN corroborate
    assert can_drift_publish_alone(drift_items, numeric_fault_flagged=True) is True
