"""Hypothesis Generation Agent and Skeptic Research Budget Planner.

Source of truth: FAULTLINE_Development_Plan.md Section 5.7 & Task T15.
Generates candidate explanations for detected fault patterns.
Skeptic Budget (Section 5.7 & Metric M18):
- At least 30% of search queries issued must be tagged 'disconfirming'.
- Logs each query and tag to logs/queries.jsonl.
- Strictly prohibits authoring financial numbers in prose (Rule R3).
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from faultline.llm import LLMClient
from faultline.schemas import Candidate


class HypothesisAgent:
    """Agent that formulates explanations and enforces the skeptic disconfirming search budget."""

    def __init__(
        self,
        llm: LLMClient | None = None,
        query_log_path: Path | str = "logs/queries.jsonl",
    ) -> None:
        self.llm = llm or LLMClient()
        self.query_log_path = Path(query_log_path)

    def generate_hypotheses(self, candidate: Candidate) -> list[str]:
        """Generate qualitative explanations for a candidate fault window without numbers."""
        engines_str = ", ".join(candidate.engines)
        prompt = (
            f"Candidate fault window detected from {candidate.window[0]} to {candidate.window[1]}.\n"
            f"Active engines: {engines_str}. Start period: {candidate.fault_start_period}.\n"
            f"Formulate a cautious, objective qualitative hypothesis explaining why asset quality restatement "
            f"diverged from headline ratios or why funding mix shifted. CRITICAL: DO NOT use any numbers."
        )
        explanation = self.llm.complete(prompt, allow_numbers=False)
        return [explanation]

    def plan_research_queries(
        self, candidate: Candidate, min_disconfirming_share: float = 0.30
    ) -> list[dict[str, str]]:
        """Plan public search queries with at least 30% tagged as 'disconfirming' (Metric M18)."""
        queries: list[dict[str, str]] = []

        # Supporting queries: investigating potential credit stress, seasoning, wholesale borrowing shifts
        supporting_queries = [
            f"investor presentation retail asset quality {candidate.fault_start_period}",
            f"management discussion credit cost write-offs {candidate.window[0]} {candidate.window[1]}",
            f"commercial paper short-term borrowing refinancing {candidate.fault_start_period}",
        ]
        for q in supporting_queries:
            queries.append({"candidate_id": candidate.id, "query": q, "tag": "supporting"})

        # Disconfirming queries: seeking benign explanations, macro/regulatory changes, healthy growth
        disconfirming_queries = [
            f"regulatory circular accounting standard revision RBI {candidate.fault_start_period}",
            f"seasonal collection pattern fourth quarter recovery historical trend",
            f"credit rating agency rationale upgrade stable outlook {candidate.fault_start_period}",
        ]
        for q in disconfirming_queries:
            queries.append({"candidate_id": candidate.id, "query": q, "tag": "disconfirming"})

        # Verify skeptic budget share (Metric M18: target >= 30%)
        disconfirming_count = sum(1 for q in queries if q["tag"] == "disconfirming")
        actual_share = disconfirming_count / len(queries) if queries else 0.0
        assert actual_share >= min_disconfirming_share, (
            f"Skeptic budget violation: disconfirming query share is {actual_share:.1%}, "
            f"must be >= {min_disconfirming_share:.1%}"
        )

        # Log queries to disk
        self._log_queries(queries)
        return queries

    def _log_queries(self, queries: list[dict[str, str]]) -> None:
        """Log research queries to logs/queries.jsonl for Metric M18 measurement."""
        self.query_log_path.parent.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc).isoformat()
        with open(self.query_log_path, "a", encoding="utf-8") as f:
            for q in queries:
                entry = dict(q)
                entry["timestamp"] = now
                f.write(json.dumps(entry) + "\n")
