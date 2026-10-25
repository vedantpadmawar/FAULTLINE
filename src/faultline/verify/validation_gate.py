"""Validation Gate: 5-point audit filter before dossier publication.

Source of truth: FAULTLINE_Development_Plan.md Section 5.5 & Hard Rules R2, R4, R5, R8.
All conditions must pass to publish:
1. Numeric persistence >= 3
2. Peer-adjusted deviation >= 1.5 (if < 3 peers have data: capped at Weak)
3. Data quality: every input has a ledger entry with a matching file hash
4. Alternative explanations: no benign explanation covers > 60% of the gap
5. Multiple-testing guard: peer_adjusted >= 1.5 + 0.25 * log2(patterns_tested) or downgrade to Weak.

Failure states: 'rejected' or 'no_reliable_finding'.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Literal

from faultline.ledger import FinancialLedger
from faultline.schemas import Candidate


@dataclass
class GateAuditReport:
    """Detailed audit report from the validation gate."""

    passed: bool
    verdict_state: Literal["explained", "unexplained", "no_reliable_finding", "rejected"]
    rejection_reasons: list[str]
    downgrade_to_weak: bool
    downgrade_reasons: list[str]
    condition_checks: dict[str, bool]


def audit_validation_gate(
    candidate: Candidate,
    ledger: FinancialLedger,
    persistence: int,
    peer_adjusted_deviation: float,
    peer_count: int,
    red_team_log: list[dict[str, Any]],
    patterns_tested: int = 1,
    evidence_ledger_ids: list[str] | None = None,
    cache_dir: str | None = None,
) -> GateAuditReport:
    """Execute the 5-point validation gate on a candidate fault window.

    Returns:
        GateAuditReport with pass/fail decision, reasons, and verdict state.
    """
    rejection_reasons: list[str] = []
    downgrade_reasons: list[str] = []
    condition_checks: dict[str, bool] = {}

    # 1. Condition 1: Numeric persistence >= 3
    cond1_passed = persistence >= 3
    condition_checks["persistence_gte_3"] = cond1_passed
    if not cond1_passed:
        rejection_reasons.append(
            f"Condition 1 failed: Persistence {persistence} is below minimum threshold of 3 periods."
        )

    # 2. Condition 2: Peer-adjusted deviation >= 1.5
    cond2_passed = peer_adjusted_deviation >= 1.5
    condition_checks["peer_adjusted_gte_1_5"] = cond2_passed
    if not cond2_passed:
        rejection_reasons.append(
            f"Condition 2 failed: Peer-adjusted deviation {peer_adjusted_deviation:.2f} is below 1.5."
        )

    if peer_count < 3:
        downgrade_reasons.append(
            f"Condition 2 notice: Only {peer_count} peers available (need at least 3); evidence capped at Weak."
        )

    # 3. Condition 3: Data quality - every input has a ledger entry with valid hash
    evidence_ids = evidence_ledger_ids or []
    missing_entries = []
    for eid in evidence_ids:
        entry = ledger.get(eid)
        if entry is None or not entry.source_sha256:
            missing_entries.append(eid)

    cond3_passed = len(missing_entries) == 0
    condition_checks["data_quality_hashes_valid"] = cond3_passed
    if not cond3_passed:
        rejection_reasons.append(
            f"Condition 3 failed: Evidence IDs {missing_entries} lack ledger entries or valid hashes."
        )

    # 4. Condition 4: No benign alternative explanation covers > 60% of gap
    vulnerable_tests = [
        t for t in red_team_log if t.get("share_covered", 0.0) > 0.60
    ]
    cond4_passed = len(vulnerable_tests) == 0
    condition_checks["alternative_explanations_resisted"] = cond4_passed
    if not cond4_passed:
        for vt in vulnerable_tests:
            rejection_reasons.append(
                f"Condition 4 failed: Benign explanation '{vt.get('test')}' covers "
                f"{vt.get('share_covered', 0):.1%} of the gap (> 60% threshold)."
            )

    # 5. Condition 5: Multiple-testing guard
    # Require peer_adjusted >= 1.5 + 0.25 * log2(patterns_tested)
    patterns = max(1, patterns_tested)
    multi_test_threshold = 1.5 + 0.25 * math.log2(patterns)
    cond5_passed = peer_adjusted_deviation >= multi_test_threshold
    condition_checks["multiple_testing_guard"] = cond5_passed
    if not cond5_passed:
        downgrade_reasons.append(
            f"Condition 5 notice: Peer deviation {peer_adjusted_deviation:.2f} did not meet multiple-testing "
            f"threshold {multi_test_threshold:.2f} for {patterns} patterns tested; downgraded to Weak."
        )

    # Determine overall gate result
    has_critical_rejection = not (cond1_passed and cond2_passed and cond3_passed and cond4_passed)
    downgrade_to_weak = (peer_count < 3) or (not cond5_passed)

    if has_critical_rejection:
        verdict_state: Literal["explained", "unexplained", "no_reliable_finding", "rejected"] = "rejected"
        passed = False
    else:
        passed = True
        verdict_state = "unexplained"

    return GateAuditReport(
        passed=passed,
        verdict_state=verdict_state,
        rejection_reasons=rejection_reasons,
        downgrade_to_weak=downgrade_to_weak,
        downgrade_reasons=downgrade_reasons,
        condition_checks=condition_checks,
    )
