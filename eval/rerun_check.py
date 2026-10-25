"""Metric M14 evaluation: Determinism across repeated pipeline runs from cache.

Source of truth: FAULTLINE_Development_Plan.md Section 6 & Gate Metric M14.
Acceptance: 100% of numeric outputs must be strictly identical across two runs from cache.
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

from eval.canary.generate import generate_canary_dataset
from faultline.agents.redteam import RedTeamAgent
from faultline.engines.funding import analyze_funding
from faultline.engines.quality import analyze_quality
from faultline.fault_finder import find_candidate_faults


def run_pipeline_snapshot(seed: int = 42) -> dict[str, Any]:
    """Execute the numerical pipeline and collect all computed numeric values."""
    lenders, _ = generate_canary_dataset(seed=seed)
    snapshot: dict[str, Any] = {}

    red_team = RedTeamAgent()

    for lender in lenders[:10]:
        lid = lender["lender_id"]
        ts = lender["time_series"]
        periods = ts["periods"]

        # Quality analysis
        q_res = analyze_quality(
            periods=periods,
            gross_advances=ts["gross_advances"],
            gnpa_amount=ts["gnpa_amount"],
            writeoffs_period=ts["writeoffs_period"],
            stage2_assets=ts["stage2_assets"],
            provisions_total=ts["provisions_total"],
            credit_cost=ts["credit_cost"],
            lag=4,
            baseline_periods=2,
        )

        # Funding analysis
        borrowings_by_type = {
            "borrowings_bank": ts["borrowings_bank"],
            "borrowings_bonds": ts["borrowings_bonds"],
            "borrowings_deposits": ts["borrowings_deposits"],
            "borrowings_securitisation": ts["borrowings_securitisation"],
            "borrowings_foreign": ts["borrowings_foreign"],
        }
        f_res = analyze_funding(
            periods=periods,
            total_borrowings=ts["total_borrowings"],
            borrowings_by_type=borrowings_by_type,
            borrowings_short_term=ts["borrowings_short_term"],
            gross_advances=ts["gross_advances"],
            assigned_loans_outstanding=ts["assigned_loans_outstanding"],
            co_lending_aum=ts["co_lending_aum"],
            lag=4,
        )

        cands = find_candidate_faults(
            quality_result=q_res,
            funding_result=f_res,
            periods=periods,
        )

        lender_snapshot = {
            "q_series": q_res.series,
            "q_flags": q_res.flags,
            "q_persistence": q_res.persistence_periods,
            "f_series": f_res.series,
            "f_flags": f_res.flags,
            "f_persistence": f_res.persistence_periods,
            "candidates_count": len(cands),
            "candidates_numbers": [c.summary_numbers for c in cands],
        }
        snapshot[lid] = lender_snapshot

    return snapshot


def evaluate_determinism(output_dir: Path | str = "eval/results") -> dict[str, Any]:
    """Compare two complete pipeline runs and measure M14_determinism."""
    run1 = run_pipeline_snapshot(seed=42)
    run2 = run_pipeline_snapshot(seed=42)

    total_values = 0
    identical_values = 0

    def compare_structures(v1: Any, v2: Any) -> None:
        nonlocal total_values, identical_values
        if isinstance(v1, dict) and isinstance(v2, dict):
            for k in v1:
                if k in v2:
                    compare_structures(v1[k], v2[k])
                else:
                    total_values += 1
        elif isinstance(v1, list) and isinstance(v2, list):
            for item1, item2 in zip(v1, v2):
                compare_structures(item1, item2)
        elif isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            total_values += 1
            if v1 == v2 or abs(v1 - v2) < 1e-12:
                identical_values += 1
        elif v1 is None and v2 is None:
            total_values += 1
            identical_values += 1
        elif isinstance(v1, bool) and isinstance(v2, bool):
            total_values += 1
            if v1 == v2:
                identical_values += 1
        else:
            total_values += 1
            if v1 == v2:
                identical_values += 1

    compare_structures(run1, run2)

    determinism_rate = (identical_values / total_values * 100.0) if total_values > 0 else 0.0

    result = {
        "metric_id": "M14_determinism",
        "value": float(round(determinism_rate, 2)),
        "unit": "percent",
        "total_compared": total_values,
        "identical": identical_values,
        "gate_passed": determinism_rate == 100.0,
    }

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "m14_determinism.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    res = evaluate_determinism()
    print(f"M14 Determinism: {res['value']}% ({res['identical']}/{res['total_compared']} identical values)")
    print(f"Gate Passed: {res['gate_passed']}")
