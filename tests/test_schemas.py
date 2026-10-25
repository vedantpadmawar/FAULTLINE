"""Tests for FAULTLINE Pydantic v2 schemas.

Verifies data contracts defined in FAULTLINE_Development_Plan.md Section 4.
"""

from datetime import datetime, timezone
import pytest
from pydantic import ValidationError

from faultline.schemas import (
    Candidate,
    Dossier,
    DriftItem,
    EngineResult,
    LedgerEntry,
    SUPPORTED_METRICS,
)


def test_supported_metrics_coverage():
    """Verify standard supported metrics are registered."""
    expected_metrics = {
        "gross_advances",
        "gnpa_amount",
        "nnpa_amount",
        "writeoffs_period",
        "stage1_assets",
        "stage2_assets",
        "stage3_assets",
        "provisions_total",
        "credit_cost",
        "borrowings_bank",
        "borrowings_bonds",
        "borrowings_deposits",
        "borrowings_securitisation",
        "borrowings_foreign",
        "borrowings_short_term",
        "assigned_loans_outstanding",
        "co_lending_aum",
        "total_borrowings",
    }
    assert expected_metrics.issubset(SUPPORTED_METRICS)


def test_ledger_entry_valid():
    """Verify construction and validation of a valid LedgerEntry."""
    now = datetime.now(timezone.utc)
    entry = LedgerEntry(
        id="L0001",
        metric="gross_advances",
        value=150000.5,
        unit="INR_cr",
        period="FY26-Q2",
        period_kind="Q",
        source_url="https://example.com/results_q2.pdf",
        source_type="filing",
        doc_page=14,
        snippet="Gross advances stood at INR 150,000.5 cr as of September 30, 2025",
        source_sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        retrieved_at=now,
    )
    assert entry.id == "L0001"
    assert entry.metric == "gross_advances"
    assert entry.value == 150000.5
    assert entry.period_kind == "Q"
    assert entry.source_type == "filing"
    assert entry.doc_page == 14

    # Serialization roundtrip
    dumped = entry.model_dump(mode="json")
    restored = LedgerEntry.model_validate(dumped)
    assert restored.id == entry.id
    assert restored.value == entry.value


def test_ledger_entry_invalid_literals():
    """Verify validation fails for invalid period_kind or source_type."""
    now = datetime.now(timezone.utc)
    # Invalid period_kind
    with pytest.raises(ValidationError):
        LedgerEntry(
            id="L0002",
            metric="gross_advances",
            value=100.0,
            unit="INR_cr",
            period="FY26-Q2",
            period_kind="M",  # type: ignore[arg-type] # Not in ["Q", "H", "Y"]
            source_url="https://example.com/doc.pdf",
            source_type="filing",
            snippet="100",
            source_sha256="abc",
            retrieved_at=now,
        )

    # Invalid source_type
    with pytest.raises(ValidationError):
        LedgerEntry(
            id="L0003",
            metric="gross_advances",
            value=100.0,
            unit="INR_cr",
            period="FY26-Q2",
            period_kind="Q",
            source_url="https://example.com/doc.pdf",
            source_type="internal_leak",  # type: ignore[arg-type] # Not in ["filing", "management", "outside"]
            snippet="100",
            source_sha256="abc",
            retrieved_at=now,
        )


def test_engine_result_tested_and_abstained():
    """Verify EngineResult schema for both tested and abstained states."""
    tested = EngineResult(
        engine="quality",
        status="tested",
        periods=["FY25-Q1", "FY25-Q2"],
        series={"reported": [0.035, 0.034], "like_for_like": [0.048, 0.051]},
        flags=[True, True],
        persistence_periods=2,
        ledger_ids=["L0001", "L0002"],
    )
    assert tested.status == "tested"
    assert tested.persistence_periods == 2

    abstained = EngineResult(
        engine="quality",
        status="abstained",
        abstain_reason="Write-offs not disclosed in quarterly release; cannot compute like-for-like",
        periods=[],
        series={},
        flags=[],
        persistence_periods=0,
        ledger_ids=[],
    )
    assert abstained.status == "abstained"
    assert "Write-offs not disclosed" in (abstained.abstain_reason or "")


def test_drift_item_schema():
    """Verify DriftItem schema for note comparison."""
    item = DriftItem(
        kind="removed",
        heading="Note 34: Movement of Non-Performing Assets",
        text_before="Opening balance: 500 cr; Additions: 120 cr...",
        text_after=None,
        similarity=0.0,
        page_before=85,
        page_after=None,
    )
    assert item.kind == "removed"
    assert item.page_after is None


def test_candidate_schema():
    """Verify Candidate fault window schema."""
    candidate = Candidate(
        id="CAND-001",
        engines=["quality", "funding"],
        window=("FY25-Q1", "FY25-Q4"),
        fault_start_period="FY25-Q2",
        summary_numbers={"max_gap_bps": 140.0, "short_term_shift_pct": 8.5},
        drift_items=[],
        hypotheses=["Asset quality divergence corroborated by off-book funding shift."],
        red_team_log=[{"test": "seasonality", "outcome": "failed", "reason": "Gap persists across 3 quarters"}],
        gate={"peer_adjusted": 2.1, "persistence": 3},
        state="unexplained",
    )
    assert candidate.id == "CAND-001"
    assert candidate.state == "unexplained"
    assert len(candidate.hypotheses) == 1


def test_dossier_schema():
    """Verify Dossier schema containing final verified finding."""
    dossier = Dossier(
        company="LenderAlpha",
        discovery="Persistent divergence between reported GNPA and like-for-like restatement",
        why_it_matters=[
            "Reported GNPA ratio held flat at 2.1% while restated ratio climbed to 3.8%.",
            "Denominator expansion of 34% YoY masked gross NPA accumulation.",
            "Write-offs exceeded additions for 3 consecutive quarters.",
        ],
        evidence=["L0001", "L0002", "L0003"],
        timeline=[{"period": "FY25-Q2", "event": "First flag detected"}],
        numbers={"reported_gnpa_pct": 2.1, "restated_gnpa_pct": 3.8, "gap_bps": 170.0},
        counterargument="Management attributes accelerated write-offs to post-COVID standard provisioning normalization.",
        falsification_condition="Reported additions drop below 1.0% of opening advances in FY26-Q1.",
        evidence_score=78.5,
        evidence_label="Strong",
        not_tested=["co_lending_default_guarantees"],
    )
    assert dossier.evidence_label == "Strong"
    assert dossier.evidence_score >= 70
    assert len(dossier.why_it_matters) == 3
    assert len(dossier.evidence) == 3
