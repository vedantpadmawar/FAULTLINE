"""Test Rules R2 & R3 and Gate Metrics M03 & M04.

Source of truth: FAULTLINE_Development_Plan.md Section 3 (Hard Rules) & Task T03.
- R2: Every number in any output has a ledger entry (M03: unsupported_claims = 0).
- R3: LLM text must not contain computed numbers (M04: llm_authored_numbers = 0).
Acceptance: fail on a deliberate violation, pass on clean code.
"""

import pytest
from faultline.ledger import FinancialLedger
from faultline.schemas import Dossier
from faultline.verify.claim_checker import verify_dossier_grounding


@pytest.fixture
def populated_ledger() -> FinancialLedger:
    ledger = FinancialLedger()
    ledger.create_and_add_entry(
        entry_id="L0001",
        metric="gnpa_amount",
        value=2100.0,
        unit="INR_cr",
        period="FY25-Q4",
        period_kind="Q",
        source_url="https://example.com/filing.pdf",
        source_type="filing",
        doc_page=10,
        snippet="Gross NPA was INR 2100 cr",
        source_sha256="hash1",
    )
    ledger.create_and_add_entry(
        entry_id="L0002",
        metric="gross_advances",
        value=100000.0,
        unit="INR_cr",
        period="FY25-Q4",
        period_kind="Q",
        source_url="https://example.com/filing.pdf",
        source_type="filing",
        doc_page=10,
        snippet="Gross advances stood at INR 100000 cr",
        source_sha256="hash1",
    )
    ledger.create_and_add_entry(
        entry_id="L0003",
        metric="writeoffs_period",
        value=450.0,
        unit="INR_cr",
        period="FY25-Q4",
        period_kind="Q",
        source_url="https://example.com/filing.pdf",
        source_type="filing",
        doc_page=11,
        snippet="Write-offs during the quarter were INR 450 cr",
        source_sha256="hash1",
    )
    return ledger


def test_grounded_dossier_passes(populated_ledger):
    """Clean dossier with all numbers verified against the ledger passes validation."""
    dossier = Dossier(
        company="GenericLender",
        discovery="Like-for-like restatement reveals divergence",
        why_it_matters=[
            "1. Gross advances reached 100000.0 INR_cr.",
            "2. Write-offs amounted to 450.0 INR_cr.",
        ],
        evidence=["L0001", "L0002", "L0003"],
        timeline=[],
        numbers={"gross_advances": 100000.0, "writeoffs": 450.0},
        counterargument="Management cited normalization.",
        falsification_condition="Write-offs drop below 450.0 in next quarter.",
        evidence_score=85.0,
        evidence_label="Strong",
        not_tested=[],
    )

    result = verify_dossier_grounding(dossier, populated_ledger)
    assert result["is_valid"] is True
    assert result["unsupported_claims"] == []
    assert result["llm_authored_numbers"] == []


def test_unsupported_numeric_claim_fails_m03(populated_ledger):
    """Deliberate violation: Claiming a number not in the ledger fails M03."""
    dossier = Dossier(
        company="GenericLender",
        discovery="Divergence found",
        why_it_matters=["Advances grew."],
        evidence=["L0001"],  # only L0001 (value 2100.0) is in evidence
        timeline=[],
        # 99999.0 is NOT in evidence or ledger
        numbers={"invented_metric": 99999.0},
        counterargument="None",
        falsification_condition="None",
        evidence_score=50.0,
        evidence_label="Moderate",
    )

    result = verify_dossier_grounding(dossier, populated_ledger)
    assert result["is_valid"] is False
    assert len(result["unsupported_claims"]) >= 1
    assert any("invented_metric" in s for s in result["unsupported_claims"])


def test_missing_evidence_id_fails_m03(populated_ledger):
    """Deliberate violation: Referencing an evidence ID that does not exist in ledger fails M03."""
    dossier = Dossier(
        company="GenericLender",
        discovery="Divergence found",
        why_it_matters=["Advances grew."],
        evidence=["L9999"],  # Non-existent ID
        timeline=[],
        numbers={},
        counterargument="None",
        falsification_condition="None",
        evidence_score=50.0,
        evidence_label="Moderate",
    )

    result = verify_dossier_grounding(dossier, populated_ledger)
    assert result["is_valid"] is False
    assert any("L9999" in s for s in result["unsupported_claims"])


def test_llm_authored_numbers_in_prose_fails_m04(populated_ledger):
    """Deliberate violation: LLM prose hallucinating an ungrounded number fails M04."""
    dossier = Dossier(
        company="GenericLender",
        discovery="Divergence found",
        why_it_matters=[
            # 88.5% is an ungrounded hallucinated number not present in the ledger!
            "Slippages unexpectedly spiked by 88.5% in the retail segment.",
        ],
        evidence=["L0001", "L0002"],
        timeline=[],
        numbers={"gross_advances": 100000.0},
        counterargument="None",
        falsification_condition="None",
        evidence_score=50.0,
        evidence_label="Moderate",
    )

    result = verify_dossier_grounding(dossier, populated_ledger)
    assert result["is_valid"] is False
    assert len(result["llm_authored_numbers"]) >= 1
    assert any("88.5%" in s for s in result["llm_authored_numbers"])
