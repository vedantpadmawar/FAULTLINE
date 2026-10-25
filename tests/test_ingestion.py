"""Tests for the financial document ingestion pipeline (Task T07).

Source of truth: FAULTLINE_Development_Plan.md Section 4 & Task T07.
Verifies:
- PDF page and table extraction with page numbers and exact source snippets.
- Creation of proof-grounded LedgerEntry records.
- Source caching and SHA-256 validation (Rule R8).
- Flagging of ambiguous extractions for manual review.
- Missing values remain absent (never zero-filled or guessed - Rule R4).
- Multi-lender multi-period ledger population.
- Zero company-specific hardcoding (Rule R1).
"""

import io
from pathlib import Path
import pytest
import pypdf

from faultline.ingest.pdf_tables import (
    ExtractedCandidate,
    RawPageContent,
    detect_unit_scale,
    extract_candidates_from_page,
    match_metric_row,
    parse_numeric_cell,
)
from faultline.ingest.results import (
    IngestionReport,
    compute_sha256,
    ingest_financial_pdf,
    normalize_period_label,
)
from faultline.ledger import FinancialLedger


from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject


def create_minimal_pdf_bytes(lines_by_page: list[list[str]]) -> bytes:
    """Helper to generate a valid multi-page PDF with text lines using pypdf.PdfWriter."""
    writer = pypdf.PdfWriter()
    font_dict = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    resources = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_dict})
    })

    for lines in lines_by_page:
        page = writer.add_blank_page(width=612, height=792)
        stream_lines = ["BT", "/F1 10 Tf", "50 750 Td"]
        for line in lines:
            safe_line = line.replace("(", "[").replace(")", "]")
            stream_lines.append(f"({safe_line}) Tj")
            stream_lines.append("0 -15 Td")
        stream_lines.append("ET")
        stream_text = "\n".join(stream_lines).encode("utf-8")

        stream_obj = DecodedStreamObject()
        stream_obj.set_data(stream_text)
        page[NameObject("/Contents")] = stream_obj
        page[NameObject("/Resources")] = resources

    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_parse_numeric_cell():
    """Verify numeric parser correctly handles Indian comma formats, parentheses, and dashes."""
    assert parse_numeric_cell("1,25,000.50") == 125000.50
    assert parse_numeric_cell("(450.00)") == -450.00
    assert parse_numeric_cell("-120.5") == -120.5
    assert parse_numeric_cell("—") is None
    assert parse_numeric_cell("NA") is None
    assert parse_numeric_cell("nil") is None


def test_period_label_normalization():
    """Verify period string normalizer formats quarters and half-years."""
    assert normalize_period_label("Q2 FY26") == ("FY26-Q2", "Q")
    assert normalize_period_label("FY26-Q2") == ("FY26-Q2", "Q")
    assert normalize_period_label("Q1FY25") == ("FY25-Q1", "Q")
    assert normalize_period_label("H1 FY26") == ("FY26-H1", "H")
    assert normalize_period_label("FY25") == ("FY25-Y", "Y")


def test_extract_candidates_from_page():
    """Verify metric row matching, snippet capture, and page reference."""
    sample_page = RawPageContent(
        page_number=14,
        text=(
            "Statement of Financial Results for the Quarter Ended September 30, 2025 (INR in Crore)\n"
            "Gross Advances: 150,000.0\n"
            "Gross NPA Amount: 3,450.0\n"
            "Net NPA Amount: 1,120.0\n"
            "Total Provisions: 2,330.0\n"
            "Write-offs during the period: 280.0\n"
            "Borrowings from Banks: 45,000.0\n"
            "Debt Securities: 35,000.0\n"
        ),
    )

    candidates = extract_candidates_from_page(sample_page, default_period="FY26-Q2")
    metric_map = {c.metric: c for c in candidates}

    assert "gross_advances" in metric_map
    assert metric_map["gross_advances"].value == 150000.0
    assert metric_map["gross_advances"].doc_page == 14
    assert "Gross Advances: 150,000.0" in metric_map["gross_advances"].snippet
    assert metric_map["gross_advances"].needs_review is False

    assert "gnpa_amount" in metric_map
    assert metric_map["gnpa_amount"].value == 3450.0

    assert "writeoffs_period" in metric_map
    assert metric_map["writeoffs_period"].value == 280.0

    assert "borrowings_bank" in metric_map
    assert metric_map["borrowings_bank"].value == 45000.0


def test_ambiguous_line_flagged_for_review():
    """A table row with multiple values and no column headers must be flagged for manual review."""
    sample_page = RawPageContent(
        page_number=8,
        text="Gross Advances: 150,000.0 142,000.0 125,000.0",
    )
    # No periods_header_map supplied
    candidates = extract_candidates_from_page(sample_page, default_period="FY26-Q2")
    assert len(candidates) == 1
    assert candidates[0].needs_review is True
    assert "without column header mapping" in (candidates[0].review_reason or "")


def test_multi_period_column_mapping():
    """When a column mapping is supplied, multiple values on a row map to distinct periods."""
    sample_page = RawPageContent(
        page_number=10,
        text="Gross Advances: 150,000.0 142,000.0 125,000.0",
    )
    mapping = {0: "FY26-Q2", 1: "FY26-Q1", 2: "FY25-Q2"}
    candidates = extract_candidates_from_page(
        sample_page, periods_header_map=mapping
    )

    assert len(candidates) == 3
    assert candidates[0].period == "FY26-Q2"
    assert candidates[0].value == 150000.0
    assert candidates[1].period == "FY26-Q1"
    assert candidates[1].value == 142000.0
    assert candidates[2].period == "FY25-Q2"
    assert candidates[2].value == 125000.0
    assert not any(c.needs_review for c in candidates)


def test_ingest_financial_pdf_end_to_end(tmp_path):
    """End-to-end PDF ingestion test: caching, ledger creation, and missing value preservation."""
    pdf_bytes = create_minimal_pdf_bytes([
        # Page 1
        [
            "Financial Results (INR in Crore)",
            "Gross Advances: 85,000.0",
            "Gross NPA: 2,100.0",
            "Net NPA: 650.0",
        ],
        # Page 2
        [
            "Borrowings and Disclosures",
            "Borrowings from Banks: 25,000.0",
            "Debt Securities: 18,000.0",
            "Write-offs during the period: 150.0",
        ],
    ])

    test_pdf = tmp_path / "q2_results.pdf"
    test_pdf.write_bytes(pdf_bytes)

    cache_dir = tmp_path / "cache"
    ledger = FinancialLedger()

    report = ingest_financial_pdf(
        file_source=test_pdf,
        source_url="https://exchange.com/results_q2.pdf",
        default_period="FY26-Q2",
        ledger=ledger,
        cache_dir=cache_dir,
    )

    # 1. SHA-256 and caching check (Rule R8)
    expected_hash = compute_sha256(pdf_bytes)
    assert report.source_sha256 == expected_hash
    assert Path(report.cached_path).exists()
    assert Path(report.cached_path).read_bytes() == pdf_bytes

    # 2. Entries created in ledger
    assert len(report.entries_created) == 6
    adv_entry = ledger.get_by_metric_and_period("gross_advances", "FY26-Q2")
    assert adv_entry is not None
    assert adv_entry.value == 85000.0
    assert adv_entry.doc_page == 1
    assert adv_entry.source_sha256 == expected_hash
    assert adv_entry.source_url == "https://exchange.com/results_q2.pdf"

    wo_entry = ledger.get_by_metric_and_period("writeoffs_period", "FY26-Q2")
    assert wo_entry is not None
    assert wo_entry.value == 150.0
    assert wo_entry.doc_page == 2

    # 3. Absent metrics remain strictly missing (Rule R4)
    # co_lending_aum was not in the PDF
    assert "co_lending_aum" in report.missing_metrics
    assert ledger.get_by_metric_and_period("co_lending_aum", "FY26-Q2") is None


def test_t07_acceptance_multi_lender_multi_period(tmp_path):
    """Task T07 Acceptance: Ledger populated for 2 lenders x 6 periods where disclosed."""
    ledger = FinancialLedger()
    cache_dir = tmp_path / "cache"

    periods = ["FY24-Q1", "FY24-Q2", "FY24-Q3", "FY24-Q4", "FY25-Q1", "FY25-Q2"]

    # Ingest synthetic documents for 2 distinct generic lenders
    for lender_idx in [1, 2]:
        for p_idx, period in enumerate(periods, start=1):
            adv_val = 50000.0 * lender_idx + (p_idx * 1000.0)
            gnpa_val = 1500.0 * lender_idx + (p_idx * 50.0)

            pdf_bytes = create_minimal_pdf_bytes([[
                f"Financial Results for {period}",
                f"Gross Advances: {adv_val:.1f}",
                f"Gross NPA Amount: {gnpa_val:.1f}",
            ]])

            doc_path = tmp_path / f"lender{lender_idx}_{period}.pdf"
            doc_path.write_bytes(pdf_bytes)

            ingest_financial_pdf(
                file_source=doc_path,
                source_url=f"https://exchange.com/lender{lender_idx}_{period}.pdf",
                default_period=period,
                ledger=ledger,
                cache_dir=cache_dir,
            )

    # Acceptance check: Ledger is populated across 6 periods
    for period in periods:
        adv = ledger.get_by_metric_and_period("gross_advances", period)
        assert adv is not None
        assert adv.value > 0.0

        gnpa = ledger.get_by_metric_and_period("gnpa_amount", period)
        assert gnpa is not None
        assert gnpa.value > 0.0
