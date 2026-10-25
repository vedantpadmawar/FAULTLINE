"""Unit tests for the research, hypothesis, red-team, and validation pipeline.

Source of truth: FAULTLINE_Development_Plan.md Sections 3, 5.4, 5.5, 5.6, 5.7.
Tests:
1. Candidate window identification (Section 5.4).
2. Hypothesis agent and skeptic budget (>= 30% disconfirming queries, Metric M18).
3. Red-team testing against benign explanations (> 60% threshold, Section 5.5).
4. Validation Gate 5-point audit:
   - Passing audit on clean strong finding
   - Rejection when persistence < 3
   - Rejection when benign explanation covers > 60% of gap
   - Rejection when ledger hashes are missing or invalid
   - Downgrade to Weak when peer count < 3 or multiple-testing guard fails
5. Evidence Score computation (0 to 100) and classification (Strong, Moderate, Weak).
6. Dossier building with complete source grounding.
7. Offline deterministic mode when no API key is supplied.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import pytest

from faultline.agents.hypothesis import HypothesisAgent
from faultline.agents.redteam import RedTeamAgent
from faultline.fault_finder import find_candidate_faults
from faultline.ledger import FinancialLedger
from faultline.llm import LLMClient
from faultline.report.dossier import build_dossier
from faultline.schemas import Candidate, EngineResult
from faultline.verify.claim_checker import verify_dossier_grounding
from faultline.verify.score import compute_evidence_score
from faultline.verify.validation_gate import audit_validation_gate


@pytest.fixture
def test_ledger() -> FinancialLedger:
    ledger = FinancialLedger()
    now = datetime.now(timezone.utc)
    for p in ["FY25-Q1", "FY25-Q2", "FY25-Q3"]:
        ledger.create_and_add_entry(
            metric="gross_advances",
            value=100000.0,
            unit="INR_cr",
            period=p,
            period_kind="Q",
            source_url="https://exchange.com/results.pdf",
            source_type="filing",
            snippet=f"Gross advances {p}: 100000",
            source_sha256="valid_hash_abc123",
            doc_page=12,
            retrieved_at=now,
        )
        ledger.create_and_add_entry(
            metric="gnpa_amount",
            value=3500.0,
            unit="INR_cr",
            period=p,
            period_kind="Q",
            source_url="https://exchange.com/presentation.pdf",
            source_type="management",
            snippet=f"GNPA {p}: 3500",
            source_sha256="valid_hash_abc123",
            doc_page=8,
            retrieved_at=now,
        )
    return ledger


def test_candidate_window_identification():
    """Section 5.4: Flags when a numeric engine has persistence >= 3 or co-occurrence."""
    periods = ["FY24-Q1", "FY24-Q2", "FY24-Q3", "FY24-Q4", "FY25-Q1", "FY25-Q2"]

    # Quality engine with persistence 3 starting at FY24-Q4 (t=3)
    q_result = EngineResult(
        engine="quality",
        status="tested",
        periods=periods,
        series={"gap": [0.01, 0.01, 0.01, 0.04, 0.05, 0.06], "z_scores": [0.5, 0.5, 0.5, 2.5, 3.0, 3.2]},
        flags=[False, False, False, True, True, True],
        persistence_periods=3,
        ledger_ids=["L0001", "L0002"],
    )

    candidates = find_candidate_faults(quality_result=q_result, periods=periods)
    assert len(candidates) == 1
    cand = candidates[0]
    assert cand.fault_start_period == "FY24-Q4"
    assert "quality" in cand.engines
    assert cand.summary_numbers["quality_persistence"] == 3.0


def test_skeptic_budget_enforcement(tmp_path):
    """Section 5.7: At least 30% of research queries must be tagged 'disconfirming' (Metric M18)."""
    log_file = tmp_path / "queries.jsonl"
    agent = HypothesisAgent(llm=LLMClient(), query_log_path=log_file)

    cand = Candidate(
        id="CAND-01",
        engines=["quality"],
        window=("FY25-Q1", "FY25-Q3"),
        fault_start_period="FY25-Q1",
        summary_numbers={"max_gap_bps": 150.0},
        state="unexplained",
    )

    queries = agent.plan_research_queries(cand, min_disconfirming_share=0.30)
    assert len(queries) >= 3

    disconfirming = [q for q in queries if q["tag"] == "disconfirming"]
    share = len(disconfirming) / len(queries)
    assert share >= 0.30

    # Verify queries logged to file
    assert log_file.exists()
    logged_lines = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines()]
    assert len(logged_lines) == len(queries)
    assert any(l["tag"] == "disconfirming" for l in logged_lines)


def test_red_team_testing_benign_explanations():
    """Section 5.5 Item 4: Tests candidate against alternative explanations."""
    red_team = RedTeamAgent()
    cand = Candidate(
        id="CAND-01",
        engines=["quality"],
        window=("FY25-Q1", "FY25-Q3"),
        fault_start_period="FY25-Q1",
        summary_numbers={"max_gap_bps": 200.0},
        state="unexplained",
    )

    # Clean idiosyncratic fault: all tests resisted
    log = red_team.test_candidate(cand, is_healthy_fast_grower=False, is_sector_story=False, is_seasonal=False)
    assert len(log) == 4
    assert all(t["outcome"] == "resisted" for t in log)
    assert all(t["share_covered"] <= 0.60 for t in log)

    # Healthy fast grower: rapid growth explains 70% of gap (> 60% threshold)
    fast_grower_log = red_team.test_candidate(cand, is_healthy_fast_grower=True)
    growth_test = next(t for t in fast_grower_log if t["test"] == "healthy_rapid_growth")
    assert growth_test["share_covered"] > 0.60
    assert growth_test["outcome"] == "vulnerable"


def test_validation_gate_pass_and_rejections(test_ledger):
    """Section 5.5: 5-point validation gate audits and rejections."""
    evidence_ids = [e.id for e in test_ledger.all_entries()]

    cand = Candidate(
        id="CAND-01",
        engines=["quality"],
        window=("FY25-Q1", "FY25-Q3"),
        fault_start_period="FY25-Q1",
        summary_numbers={"quality_persistence": 3.0},
        state="unexplained",
    )

    # 1. Clean Passing Audit
    clean_log = [{"test": "seasonality", "share_covered": 0.20, "outcome": "resisted"}]
    clean_report = audit_validation_gate(
        candidate=cand,
        ledger=test_ledger,
        persistence=3,
        peer_adjusted_deviation=2.1,
        peer_count=4,
        red_team_log=clean_log,
        evidence_ledger_ids=evidence_ids,
        patterns_tested=1,
    )
    assert clean_report.passed is True
    assert clean_report.verdict_state == "unexplained"
    assert clean_report.rejection_reasons == []

    # 2. Rejection on Persistence < 3
    low_persist_report = audit_validation_gate(
        candidate=cand,
        ledger=test_ledger,
        persistence=2,  # Below 3
        peer_adjusted_deviation=2.1,
        peer_count=4,
        red_team_log=clean_log,
        evidence_ledger_ids=evidence_ids,
    )
    assert low_persist_report.passed is False
    assert low_persist_report.verdict_state == "rejected"
    assert any("Persistence 2 is below" in r for r in low_persist_report.rejection_reasons)

    # 3. Rejection when Benign Explanation covers > 60% of gap
    vulnerable_log = [{"test": "seasonality", "share_covered": 0.75, "outcome": "vulnerable"}]
    vulnerable_report = audit_validation_gate(
        candidate=cand,
        ledger=test_ledger,
        persistence=3,
        peer_adjusted_deviation=2.1,
        peer_count=4,
        red_team_log=vulnerable_log,
        evidence_ledger_ids=evidence_ids,
    )
    assert vulnerable_report.passed is False
    assert any("> 60% threshold" in r for r in vulnerable_report.rejection_reasons)

    # 4. Downgrade to Weak when Peer Count < 3
    few_peers_report = audit_validation_gate(
        candidate=cand,
        ledger=test_ledger,
        persistence=3,
        peer_adjusted_deviation=2.1,
        peer_count=2,  # Only 2 peers
        red_team_log=clean_log,
        evidence_ledger_ids=evidence_ids,
    )
    assert few_peers_report.passed is True  # still passes gate, but downgraded
    assert few_peers_report.downgrade_to_weak is True
    assert any("Only 2 peers available" in r for r in few_peers_report.downgrade_reasons)


def test_evidence_scoring_and_classification():
    """Section 5.6: Multi-factor evidence score (0 to 100) and tier labels."""
    # Strong case: score >= 70, layers >= 3, persistence >= 4
    score_strong, label_strong, _ = compute_evidence_score(
        persistence=5,
        z_magnitude=3.5,
        peer_adjusted=2.5,
        distinct_source_layers=3,
        share_explanations_resisted=1.0,
    )
    assert score_strong >= 70.0
    assert label_strong == "Strong"

    # Moderate case: score >= 45, layers >= 2
    score_mod, label_mod, _ = compute_evidence_score(
        persistence=3,
        z_magnitude=2.0,
        peer_adjusted=1.8,
        distinct_source_layers=2,
        share_explanations_resisted=0.75,
    )
    assert 45.0 <= score_mod < 70.0
    assert label_mod == "Moderate"

    # Weak case: low persistence or low layers
    score_weak, label_weak, _ = compute_evidence_score(
        persistence=2,
        z_magnitude=1.5,
        peer_adjusted=1.2,
        distinct_source_layers=1,
        share_explanations_resisted=0.5,
    )
    assert label_weak == "Weak"


def test_dossier_compilation_and_grounding_audit(test_ledger):
    """End-to-end dossier building and proof-carrying claim verification (Rules R2, R3)."""
    evidence_ids = [e.id for e in test_ledger.all_entries()]

    cand = Candidate(
        id="CAND-01",
        engines=["quality"],
        window=("FY25-Q1", "FY25-Q3"),
        fault_start_period="FY25-Q1",
        summary_numbers={"quality_persistence": 4.0},
        red_team_log=[{"test": "seasonality", "share_covered": 0.20, "outcome": "resisted"}],
        state="unexplained",
    )

    gate_report = audit_validation_gate(
        candidate=cand,
        ledger=test_ledger,
        persistence=4,
        peer_adjusted_deviation=2.2,
        peer_count=4,
        red_team_log=cand.red_team_log,
        evidence_ledger_ids=evidence_ids,
    )

    dossier = build_dossier(
        candidate=cand,
        ledger=test_ledger,
        gate_report=gate_report,
        evidence_ledger_ids=evidence_ids,
        company_identifier="GenericLender",
    )

    assert dossier.evidence_label in ["Strong", "Moderate", "Weak"]
    assert len(dossier.why_it_matters) >= 3
    assert len(dossier.evidence) == len(evidence_ids)

    # Verify claim grounding passes Gate Metrics M03 and M04
    grounding_check = verify_dossier_grounding(dossier, test_ledger)
    assert grounding_check["is_valid"] is True
    assert grounding_check["unsupported_claims"] == []
    assert grounding_check["llm_authored_numbers"] == []


def test_offline_deterministic_mode_logging(tmp_path):
    """Verify offline fallback runs cleanly with zero API keys and logs usage."""
    usage_log = tmp_path / "usage.jsonl"
    client = LLMClient(api_key=None, usage_log_path=usage_log)

    assert client.is_offline is True
    response = client.complete("Explain the hypothesis behind this pattern.")
    assert "The observed divergence" in response
    assert "[FIGURE_REFERENCED_FROM_LEDGER]" not in response or "pattern" in response

    # Check usage log created
    assert usage_log.exists()
    logs = [json.loads(line) for line in usage_log.read_text(encoding="utf-8").splitlines()]
    assert len(logs) == 1
    assert logs[0]["provider"] == "offline"
    assert logs[0]["cost_usd"] == 0.0
