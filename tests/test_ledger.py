"""Tests for the FAULTLINE financial ledger.

Verifies Task T02 requirements and Hard Rules R2, R4, and R8:
- Complete ledger round-trip (save/load).
- Missing values remain None (never zero-filled or fabricated).
- Abstain logging with explicit reasons.
- File SHA-256 validation and tampering detection.
"""

from datetime import datetime, timezone
import pytest
from faultline.ledger import FinancialLedger, compute_bytes_sha256, compute_file_sha256
from faultline.schemas import LedgerEntry


@pytest.fixture
def sample_entry() -> LedgerEntry:
    now = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)
    return LedgerEntry(
        id="L0001",
        metric="gross_advances",
        value=125000.0,
        unit="INR_cr",
        period="FY25-Q4",
        period_kind="Q",
        source_url="https://example.com/fy25q4.pdf",
        source_type="filing",
        doc_page=12,
        snippet="Total gross advances reached INR 125,000 cr",
        source_sha256="abcd1234efgh5678abcd1234efgh5678abcd1234efgh5678abcd1234efgh5678",
        retrieved_at=now,
    )


def test_ledger_add_and_get(sample_entry):
    """Test adding an entry and retrieving it by ID."""
    ledger = FinancialLedger()
    assert len(ledger) == 0

    assigned_id = ledger.add_entry(sample_entry)
    assert assigned_id == "L0001"
    assert len(ledger) == 1

    retrieved = ledger.get("L0001")
    assert retrieved is not None
    assert retrieved.value == 125000.0
    assert retrieved.metric == "gross_advances"

    # Non-existent ID returns None
    assert ledger.get("L9999") is None


def test_ledger_prevent_duplicate_id(sample_entry):
    """Adding an entry with an existing ID must raise a ValueError."""
    ledger = FinancialLedger()
    ledger.add_entry(sample_entry)

    with pytest.raises(ValueError, match="already exists"):
        ledger.add_entry(sample_entry)


def test_ledger_factory_auto_id():
    """Factory method assigns auto-incremented IDs starting with L0001."""
    ledger = FinancialLedger()
    e1 = ledger.create_and_add_entry(
        metric="gross_advances",
        value=100.0,
        unit="INR_cr",
        period="FY25-Q1",
        period_kind="Q",
        source_url="https://example.com/p1.pdf",
        source_type="filing",
        snippet="100.0 cr",
        source_sha256="hash1",
    )
    e2 = ledger.create_and_add_entry(
        metric="gnpa_amount",
        value=5.0,
        unit="INR_cr",
        period="FY25-Q1",
        period_kind="Q",
        source_url="https://example.com/p1.pdf",
        source_type="filing",
        snippet="5.0 cr",
        source_sha256="hash1",
    )
    assert e1.id == "L0001"
    assert e2.id == "L0002"
    assert len(ledger) == 2


def test_ledger_missing_value_rule_r4():
    """Rule R4: Missing value must return None, NEVER zero and never guessed."""
    ledger = FinancialLedger()
    ledger.create_and_add_entry(
        metric="gross_advances",
        value=20000.0,
        unit="INR_cr",
        period="FY25-Q2",
        period_kind="Q",
        source_url="https://example.com/p.pdf",
        source_type="filing",
        snippet="20000",
        source_sha256="h",
    )

    # Metric present
    adv = ledger.get_by_metric_and_period("gross_advances", "FY25-Q2")
    assert adv is not None
    assert adv.value == 20000.0

    # Metric not present for this period -> None
    writeoffs = ledger.get_by_metric_and_period("writeoffs_period", "FY25-Q2")
    assert writeoffs is None  # NOT 0.0

    # Availability check
    complete, missing = ledger.check_availability(
        ["gross_advances", "writeoffs_period"], "FY25-Q2"
    )
    assert not complete
    assert missing == ["writeoffs_period"]


def test_ledger_abstain_logging():
    """Rule R4: An explicit abstain record is created with a reason when data is missing."""
    ledger = FinancialLedger()
    record = ledger.log_abstain(
        engine="quality",
        check_name="like_for_like_ratio",
        period="FY25-Q3",
        missing_fields=["writeoffs_period"],
        reason="Write-offs not disclosed in quarterly release; refusing to assume zero.",
    )

    assert record["engine"] == "quality"
    assert record["period"] == "FY25-Q3"
    assert "refusing to assume zero" in record["reason"]
    assert len(ledger.abstain_log) == 1


def test_ledger_series_lookup():
    """Verify get_series maps period -> LedgerEntry."""
    ledger = FinancialLedger()
    for period, val in [("FY25-Q1", 10.0), ("FY25-Q2", 12.0), ("FY25-Q3", 15.0)]:
        ledger.create_and_add_entry(
            metric="credit_cost",
            value=val,
            unit="pct",
            period=period,
            period_kind="Q",
            source_url="https://example.com/doc",
            source_type="filing",
            snippet=f"{val}%",
            source_sha256="sha",
        )

    series = ledger.get_series("credit_cost")
    assert len(series) == 3
    assert series["FY25-Q2"].value == 12.0


def test_ledger_save_load_roundtrip(tmp_path, sample_entry):
    """Test saving ledger to JSON and reloading preserves entries and abstain logs."""
    ledger = FinancialLedger()
    ledger.add_entry(sample_entry)
    ledger.log_abstain(
        engine="funding",
        check_name="offbook_route",
        period="FY25-Q4",
        reason="Direct assignment volume not disclosed in investor presentation",
    )

    save_file = tmp_path / "ledger.json"
    ledger.save_to_json(save_file)

    loaded = FinancialLedger.load_from_json(save_file)
    assert len(loaded) == 1
    loaded_entry = loaded.get("L0001")
    assert loaded_entry is not None
    assert loaded_entry.value == sample_entry.value
    assert loaded_entry.source_sha256 == sample_entry.source_sha256
    assert len(loaded.abstain_log) == 1
    assert loaded.abstain_log[0]["check"] == "offbook_route"


def test_source_sha256_verification_and_tampering(tmp_path):
    """T02 Acceptance Test: Validating SHA-256 passes on clean file, fails if tampered."""
    source_content = b"Official Financial Statement Q2 2025: Advances INR 50,000 Cr."
    expected_sha256 = compute_bytes_sha256(source_content)

    test_doc = tmp_path / "report_q2.txt"
    test_doc.write_bytes(source_content)

    # Verify helper matches
    assert compute_file_sha256(test_doc) == expected_sha256

    ledger = FinancialLedger()
    entry = ledger.create_and_add_entry(
        metric="gross_advances",
        value=50000.0,
        unit="INR_cr",
        period="FY26-Q2",
        period_kind="Q",
        source_url="https://exchange.com/filing.txt",
        source_type="filing",
        snippet="Advances INR 50,000 Cr.",
        source_sha256=expected_sha256,
    )

    # 1. Unmodified file matches
    assert ledger.verify_entry_source(entry.id, test_doc) is True

    # 2. Tampering with file breaks the hash check
    test_doc.write_bytes(b"Tampered Financial Statement: Advances INR 99,999 Cr.")
    assert ledger.verify_entry_source(entry.id, test_doc) is False
