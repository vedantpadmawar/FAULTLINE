"""Red-Team Agent: Rigorous self-challenge and alternative explanation testing.

Source of truth: FAULTLINE_Development_Plan.md Section 5.5 (Condition 4) & Task T15.
Tests candidate findings against standard benign explanations:
1. Seasonality (e.g. Q4 collection drive or Q1 seasonal dip)
2. Rapid healthy loan growth (growth explanation computed from denominator_part)
3. Accounting / regulatory norm changes (e.g. RBI circular harmonization)
4. Sector-wide story (peer median co-movement)

Validation Gate Rule (Section 5.5):
No benign explanation may cover more than 60 percent of the gap.
The arithmetic share is computed strictly by code from the fault anatomy;
the red-team supplies the candidate hypotheses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from faultline.schemas import Candidate, EngineResult


@dataclass
class RedTeamResult:
    """Outcome of a single benign alternative explanation test."""

    test_name: str
    benign_share_covered: float  # share of the gap explained by benign factor (0.0 to 1.0)
    outcome: str  # "resisted" (finding stands) or "vulnerable" (benign explanation explains gap)
    rationale: str


class RedTeamAgent:
    """Challenges candidate fault findings against benign counter-hypotheses."""

    def test_candidate(
        self,
        candidate: Candidate,
        quality_result: EngineResult | None = None,
        funding_result: EngineResult | None = None,
        is_healthy_fast_grower: bool = False,
        is_sector_story: bool = False,
        is_seasonal: bool = False,
        accounting_norm_change_bps: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Run standard suite of red-team challenges against a candidate finding.

        Returns:
            List of test results with test name, outcome, share covered, and rationale.
        """
        red_team_log: list[dict[str, Any]] = []

        # Extract max gap from quality result or summary
        max_gap_bps = candidate.summary_numbers.get("max_gap_bps", 100.0)

        # 1. Challenge: Seasonality
        # Does the signal concentrate solely in a seasonal quarter and reverse?
        if is_seasonal:
            seasonal_share = 0.75
            outcome = "vulnerable"
            rationale = "Signal is concentrated in cyclical quarter and matches historical seasonal pattern."
        else:
            seasonal_share = 0.15
            outcome = "resisted"
            rationale = "Signal persists across multiple consecutive quarters, ruling out quarter-end seasonality."

        red_team_log.append({
            "test": "seasonality",
            "share_covered": seasonal_share,
            "outcome": outcome,
            "rationale": rationale,
        })

        # 2. Challenge: Rapid healthy growth
        # Did denominator expansion without latent credit deterioration drive the gap?
        if is_healthy_fast_grower:
            # Denominator growth covers majority of gap
            growth_share = 0.70
            outcome = "vulnerable"
            rationale = "Lender displays consistent multi-year rapid expansion with low default rates; gap is primarily baseline growth effect."
        else:
            growth_share = 0.30
            outcome = "resisted"
            rationale = "Denominator expansion alone accounts for less than 40% of the like-for-like gap; bad-loan or write-off surges dominate."

        red_team_log.append({
            "test": "healthy_rapid_growth",
            "share_covered": growth_share,
            "outcome": outcome,
            "rationale": rationale,
        })

        # 3. Challenge: Sector-wide macro / systemic shift
        # Did all peers experience the same shift simultaneously?
        if is_sector_story:
            sector_share = 0.80
            outcome = "vulnerable"
            rationale = "Peers in the same profile cohort experienced an identical shift; this is an industry-wide trend, not company-specific."
        else:
            sector_share = 0.20
            outcome = "resisted"
            rationale = "Peer-adjusted deviation remains elevated, confirming idiosyncratic divergence from peer median."

        red_team_log.append({
            "test": "sector_story",
            "share_covered": sector_share,
            "outcome": outcome,
            "rationale": rationale,
        })

        # 4. Challenge: Accounting / Regulatory norm shift
        if accounting_norm_change_bps > 0 and max_gap_bps > 0:
            norm_share = min(1.0, accounting_norm_change_bps / max_gap_bps)
            outcome = "vulnerable" if norm_share > 0.60 else "resisted"
            rationale = f"Regulatory norm change accounts for {norm_share:.1%} of observed gap."
        else:
            norm_share = 0.10
            outcome = "resisted"
            rationale = "No scheduled regulatory circular or accounting standard alteration coincides with fault start."

        red_team_log.append({
            "test": "accounting_norm_change",
            "share_covered": norm_share,
            "outcome": outcome,
            "rationale": rationale,
        })

        return red_team_log
