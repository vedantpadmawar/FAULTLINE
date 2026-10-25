"""Metric M02 and M03 evaluation: Claim grounding rate and unsupported claims.

Source of truth: FAULTLINE_Development_Plan.md Section 6 & Metrics M02, M03.
M02: Dossier claims with a source URL, page, and a snippet containing the value.
M03 (Gate): Count of claims with no ledger entry (must be 0).
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

# Ensure src/ and repo root are importable
repo_root = Path(__file__).resolve().parent.parent
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from eval.canary.generate import generate_canary_dataset, lender_to_ledger_entries
from faultline.ledger import FinancialLedger
from faultline.report.dossier import build_dossier
from faultline.schemas import Candidate, EngineResult
from faultline.verify.claim_checker import verify_dossier_grounding


def evaluate_grounding(output_dir: Path | str = "eval/results") -> dict[str, Any]:
    """Audit dossiers and ledger entries for source grounding and snippet matches."""
    lenders, ground_truth = generate_canary_dataset(seed=42)

    total_claims = 0
    grounded_claims = 0
    unsupported_claims_count = 0

    # Build ledger from first 5 lenders
    ledger = FinancialLedger()
    for lender in lenders[:5]:
        entries = lender_to_ledger_entries(lender)
        for e in entries:
            ledger.add_entry(e)

    # Verify all ledger entries have URL, page, snippet, and snippet contains the value
    for entry in ledger:
        total_claims += 1
        has_url = bool(entry.source_url)
        has_page = entry.doc_page is not None and entry.doc_page > 0
        has_snippet = bool(entry.snippet)
        val_matches = (
            str(entry.value) in entry.snippet
            or str(round(entry.value, 2)) in entry.snippet
            or str(int(entry.value)) in entry.snippet
            or f"{entry.value:.2f}" in entry.snippet
            or f"{entry.value:.1f}" in entry.snippet
        ) if has_snippet else False

        if has_url and has_page and has_snippet and val_matches:
            grounded_claims += 1
        else:
            unsupported_claims_count += 1

    # Check a sample dossier
    sample_candidate = Candidate(
        id="CAND-FY24-Q3",
        engines=["quality"],
        window=("FY24-Q3", "FY25-Q4"),
        fault_start_period="FY24-Q3",
        summary_numbers={"max_gap_bps": 350.0, "quality_persistence": 4.0},
        drift_items=[],
        hypotheses=[],
        red_team_log=[],
        gate={"patterns_tested": 1, "first_engine": "quality"},
        state="unexplained",
    )
    first_entries = list(ledger)[:4]
    evidence_ids = [e.id for e in first_entries]

    from faultline.verify.validation_gate import GateAuditReport

    gate_rep = GateAuditReport(
        passed=True,
        verdict_state="unexplained",
        rejection_reasons=[],
        downgrade_to_weak=False,
        downgrade_reasons=[],
        condition_checks={"persistence_gte_3": True},
    )

    dossier = build_dossier(
        candidate=sample_candidate,
        ledger=ledger,
        gate_report=gate_rep,
        evidence_ledger_ids=evidence_ids,
        company_identifier="SYNTH_01",
    )

    dossier_check = verify_dossier_grounding(dossier, ledger)
    unsupported_in_dossier = len(dossier_check["unsupported_claims"])
    llm_numbers = len(dossier_check["llm_authored_numbers"])

    rate = (grounded_claims / total_claims * 100.0) if total_claims > 0 else 0.0

    result = {
        "M02_claim_grounding_rate": {
            "value": float(round(rate, 2)),
            "unit": "percent",
            "grounded_claims": grounded_claims,
            "total_claims": total_claims,
        },
        "M03_unsupported_claims": {
            "value": unsupported_in_dossier,
            "unit": "count",
            "gate_passed": unsupported_in_dossier == 0,
        },
        "M04_llm_authored_numbers": {
            "value": llm_numbers,
            "unit": "count",
            "gate_passed": llm_numbers == 0,
        },
    }

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "m02_m03_grounding.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    res = evaluate_grounding()
    print(f"M02 Claim Grounding Rate: {res['M02_claim_grounding_rate']['value']}% ({res['M02_claim_grounding_rate']['grounded_claims']}/{res['M02_claim_grounding_rate']['total_claims']})")
    print(f"M03 Unsupported Claims:   {res['M03_unsupported_claims']['value']} (Gate: {res['M03_unsupported_claims']['gate_passed']})")
    print(f"M04 LLM Authored Numbers: {res['M04_llm_authored_numbers']['value']} (Gate: {res['M04_llm_authored_numbers']['gate_passed']})")
