"""Data contracts and Pydantic v2 schemas for FAULTLINE.

Source of truth: FAULTLINE_Development_Plan.md Section 4.
"""

from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field, field_validator


# Recognized metric names according to Section 4
SUPPORTED_METRICS = {
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


class LedgerEntry(BaseModel):
    """Immutable single record of an extracted financial metric with complete proof-grounding."""

    id: str  # e.g., "L0001"
    metric: str  # see SUPPORTED_METRICS
    value: float
    unit: str  # "INR_cr", "pct", "count", etc.
    period: str  # e.g., "FY26-Q2" or "FY26-H1"
    period_kind: Literal["Q", "H", "Y"]
    source_url: str
    source_type: Literal["filing", "management", "outside"]
    doc_page: int | None = None
    snippet: str  # exact text containing the value
    source_sha256: str
    retrieved_at: datetime

    @field_validator("id")
    @classmethod
    def validate_id_format(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("LedgerEntry id cannot be empty")
        return v.strip()

    @field_validator("metric")
    @classmethod
    def validate_metric(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Metric name cannot be empty")
        return v.strip()


class EngineResult(BaseModel):
    """Output of an analysis engine (Quality, Funding, Drift)."""

    engine: Literal["quality", "funding", "drift"]
    status: Literal["tested", "abstained"]
    abstain_reason: str | None = None
    periods: list[str] = Field(default_factory=list)
    series: dict[str, list[float | None]] = Field(default_factory=dict)  # reported, like_for_like, peer_median, etc.
    flags: list[bool] = Field(default_factory=list)
    persistence_periods: int = 0
    ledger_ids: list[str] = Field(default_factory=list)


class DriftItem(BaseModel):
    """A disclosure change identified between consecutive annual report notes."""

    kind: Literal["removed", "added", "redefined"]
    heading: str
    text_before: str | None = None
    text_after: str | None = None
    similarity: float | None = None
    page_before: int | None = None
    page_after: int | None = None


class Candidate(BaseModel):
    """Candidate fault window identified across engines before validation gate."""

    id: str
    engines: list[str]
    window: tuple[str, str]
    fault_start_period: str
    summary_numbers: dict[str, float] = Field(default_factory=dict)  # code-computed only
    drift_items: list[DriftItem] = Field(default_factory=list)
    hypotheses: list[str] = Field(default_factory=list)  # LLM text, strictly NO computed numbers
    red_team_log: list[dict] = Field(default_factory=list)  # {test, outcome, reason}
    gate: dict = Field(default_factory=dict)
    state: Literal["explained", "unexplained", "no_reliable_finding", "rejected"]


class Dossier(BaseModel):
    """Final published finding that has passed the validation gate."""

    company: str
    discovery: str
    why_it_matters: list[str] = Field(default_factory=list)  # 3 to 5 bullets
    evidence: list[str] = Field(default_factory=list)  # ledger ids
    timeline: list[dict] = Field(default_factory=list)
    numbers: dict[str, float] = Field(default_factory=dict)
    counterargument: str
    falsification_condition: str
    evidence_score: float  # 0 to 100
    evidence_label: Literal["Strong", "Moderate", "Weak"]
    not_tested: list[str] = Field(default_factory=list)
