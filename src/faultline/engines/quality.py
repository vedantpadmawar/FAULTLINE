"""Quality Engine: Like-for-like quality ledger and fault anatomy decomposition.

Source of truth: FAULTLINE_Development_Plan.md Section 5.1 & Hard Rules R2, R4.
Ordinary code computes every number.
Computes:
- Reported gross bad-loan ratio: G[t] / B[t]
- Like-for-like adjusted ratio on lagged book: (G[t] + W[t]) / B[t-lag]
- Reported-versus-restated gap: like_for_like[t] - reported[t]
- Exact Fault Anatomy decomposition:
    denominator_part = G[t] * (1/B[t-lag] - 1/B[t])
    writeoff_part    = W[t] / B[t-lag]
    identity check   : gap[t] == denominator_part + writeoff_part (tolerance 1e-9)
- Robust z-score against lender's own history (excluding latest 2 periods)
- Peer-adjusted z-score and flag persistence
- Additional metrics: Stage 2 share, Provision Coverage Ratio (PCR), Credit Cost
- Never zero-fills or guesses missing write-offs or data; returns explicit abstention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

from faultline.schemas import EngineResult


IDENTITY_TOLERANCE = 1e-9


@dataclass(frozen=True)
class FaultAnatomyDecomposition:
    """Exact arithmetic breakdown of reported vs restated bad-loan gap."""

    period: str
    reported_ratio: float
    like_for_like_ratio: float
    gap: float
    denominator_part: float
    writeoff_part: float
    identity_holds: bool
    identity_diff: float
    writeoffs_disclosed: bool


def compute_fault_anatomy(
    b_curr: float,
    b_lag: float,
    g_curr: float,
    w_sum: float | None = None,
    period: str = "",
) -> FaultAnatomyDecomposition:
    """Decompose the gap between reported and like-for-like bad-loan ratios.

    Formula:
        reported[t]       = G[t] / B[t]
        like_for_like[t]  = (G[t] + W[t]) / B[t-lag]
        gap[t]            = like_for_like[t] - reported[t]
        denominator_part  = G[t] * (1/B[t-lag] - 1/B[t])
        writeoff_part     = W[t] / B[t-lag] (or 0.0 if undisclosed)
        identity check    : gap[t] == denominator_part + writeoff_part

    Raises:
        ValueError if advances <= 0 or gnpa < 0.
    """
    if b_curr <= 0 or b_lag <= 0:
        raise ValueError(f"Advances must be strictly positive (got b_curr={b_curr}, b_lag={b_lag})")
    if g_curr < 0:
        raise ValueError(f"Gross bad loans cannot be negative (got g_curr={g_curr})")

    reported = g_curr / b_curr
    wo_disclosed = w_sum is not None

    effective_w = w_sum if wo_disclosed else 0.0
    like_for_like = (g_curr + effective_w) / b_lag
    gap = like_for_like - reported

    denominator_part = g_curr * ((1.0 / b_lag) - (1.0 / b_curr))
    writeoff_part = effective_w / b_lag

    identity_diff = abs(gap - (denominator_part + writeoff_part))
    identity_holds = identity_diff < IDENTITY_TOLERANCE

    return FaultAnatomyDecomposition(
        period=period,
        reported_ratio=reported,
        like_for_like_ratio=like_for_like,
        gap=gap,
        denominator_part=denominator_part,
        writeoff_part=writeoff_part,
        identity_holds=identity_holds,
        identity_diff=identity_diff,
        writeoffs_disclosed=wo_disclosed,
    )


def compute_robust_z_scores(
    gaps: list[float | None],
    exclude_recent: int = 2,
    baseline_periods: int | None = None,
    min_spread: float = 1e-6,
) -> tuple[list[float | None], float | None, float | None]:
    """Compute robust z-scores for gaps against historical center and MAD spread.

    Formula (Section 5.1):
        center = median(history)
        spread = 1.4826 * MAD(history)
        z = (gap[t] - center) / spread

    History uses either the pre-fault baseline track record (baseline_periods) or past history
    excluding recent periods, preventing an emerging or active fault from inflating its own baseline.
    """
    valid_points = [(idx, val) for idx, val in enumerate(gaps) if val is not None]
    if len(valid_points) <= exclude_recent:
        return [None] * len(gaps), None, None

    if baseline_periods is not None and len(valid_points) >= baseline_periods + 2:
        history_values = np.array([val for _, val in valid_points[:baseline_periods]], dtype=float)
    else:
        history_values = np.array([val for _, val in valid_points[:-exclude_recent]], dtype=float)

    if len(history_values) == 0:
        return [None] * len(gaps), None, None

    center = float(np.median(history_values))
    abs_deviations = np.abs(history_values - center)
    mad = float(np.median(abs_deviations))
    spread = max(1.4826 * mad, min_spread)

    z_scores: list[float | None] = []
    for g in gaps:
        if g is None:
            z_scores.append(None)
        else:
            z = (g - center) / spread
            z_scores.append(float(round(z, 4)))

    return z_scores, center, spread


def calculate_persistence(flags: list[bool]) -> int:
    """Calculate the longest run of consecutive flagged periods."""
    max_streak = 0
    current_streak = 0
    for f in flags:
        if f:
            current_streak += 1
            if current_streak > max_streak:
                max_streak = current_streak
        else:
            current_streak = 0
    return max_streak


def analyze_quality(
    periods: list[str],
    gross_advances: list[float | None],
    gnpa_amount: list[float | None],
    writeoffs_period: list[float | None] | None = None,
    stage2_assets: list[float | None] | None = None,
    provisions_total: list[float | None] | None = None,
    credit_cost: list[float | None] | None = None,
    peer_z_scores: list[list[float]] | list[float] | None = None,
    lag: int = 4,
    ledger_ids: list[str] | None = None,
    baseline_periods: int | None = 2,
    min_spread: float = 0.0035,
) -> EngineResult:
    """Run full asset quality restatement, fault anatomy decomposition, and fault detection.

    Args:
        periods: List of quarter/period names (e.g. ['FY23-Q1', ...]).
        gross_advances: Gross loan book B[t].
        gnpa_amount: Gross bad-loan amount G[t].
        writeoffs_period: Disclosed write-offs in each period.
        stage2_assets: Disclosed Stage 2 loans.
        provisions_total: Total provisions balance.
        credit_cost: Credit cost ratio (if disclosed).
        peer_z_scores: Optional peer z-scores for peer-adjustment.
        lag: Comparison lag (default 4 for quarterly data).
        ledger_ids: Tracking IDs for proof-carrying ledger records.
        baseline_periods: Pre-fault baseline periods for historical reference (default 2).
        min_spread: Minimum economically meaningful spread floor (default 35 bps).

    Returns:
        EngineResult conforming to FAULTLINE schemas.
    """
    n = len(periods)
    if n == 0:
        return EngineResult(
            engine="quality",
            status="abstained",
            abstain_reason="No periods supplied for quality analysis.",
        )

    # Validate that gross advances and gnpa amounts have enough data
    advances_count = sum(1 for v in gross_advances if v is not None and v > 0)
    gnpa_count = sum(1 for v in gnpa_amount if v is not None and v >= 0)

    if advances_count < lag + 1 or gnpa_count < lag + 1:
        return EngineResult(
            engine="quality",
            status="abstained",
            abstain_reason=(
                f"Insufficient disclosed data: required at least {lag + 1} periods of gross advances "
                f"and gross NPA amounts (found {advances_count} advances, {gnpa_count} GNPAs)."
            ),
            periods=periods,
            ledger_ids=ledger_ids or [],
        )

    # Check write-offs disclosure (Rule R4 / Section 5.1)
    has_writeoffs = (
        writeoffs_period is not None
        and any(w is not None for w in writeoffs_period)
    )

    reported_series: list[float | None] = []
    lfl_series: list[float | None] = []
    gap_series: list[float | None] = []
    denom_parts: list[float | None] = []
    wo_parts: list[float | None] = []
    identities_hold: list[bool] = []

    # Compute per-period ratios and decompositions
    for t in range(n):
        b_curr = gross_advances[t]
        g_curr = gnpa_amount[t]

        if b_curr is None or g_curr is None or b_curr <= 0 or g_curr < 0:
            reported_series.append(None)
            lfl_series.append(None)
            gap_series.append(None)
            denom_parts.append(None)
            wo_parts.append(None)
            continue

        reported_series.append(float(round(g_curr / b_curr, 6)))

        if t < lag:
            lfl_series.append(None)
            gap_series.append(None)
            denom_parts.append(None)
            wo_parts.append(None)
            continue

        b_lag = gross_advances[t - lag]
        if b_lag is None or b_lag <= 0:
            lfl_series.append(None)
            gap_series.append(None)
            denom_parts.append(None)
            wo_parts.append(None)
            continue

        # Sum write-offs over last `lag` periods if disclosed
        w_sum: float | None = None
        if has_writeoffs and writeoffs_period is not None:
            recent_w = [writeoffs_period[t - k] for k in range(lag)]
            if all(w is not None for w in recent_w):
                w_sum = float(sum(w for w in recent_w if w is not None))

        decomp = compute_fault_anatomy(
            b_curr=b_curr,
            b_lag=b_lag,
            g_curr=g_curr,
            w_sum=w_sum,
            period=periods[t],
        )

        lfl_series.append(float(decomp.like_for_like_ratio))
        gap_series.append(float(decomp.gap))
        denom_parts.append(float(decomp.denominator_part))
        wo_parts.append(float(decomp.writeoff_part))
        identities_hold.append(decomp.identity_holds)

    # Compute robust z-scores against history
    z_scores, center, spread = compute_robust_z_scores(
        gap_series,
        exclude_recent=2,
        baseline_periods=baseline_periods,
        min_spread=min_spread,
    )

    # Compute peer-adjusted z-scores and flags (flag: z >= 2.0 and peer_adjusted >= 1.5)
    peer_adjusted_series: list[float | None] = []
    flags: list[bool] = []

    for t in range(n):
        z = z_scores[t]
        if z is None:
            peer_adjusted_series.append(None)
            flags.append(False)
            continue

        peer_adj = z
        if peer_z_scores is not None and t < len(peer_z_scores):
            peers_at_t = peer_z_scores[t]
            if isinstance(peers_at_t, list) and len(peers_at_t) > 0:
                peer_median = float(np.median(peers_at_t))
                peer_adj = z - peer_median
            elif isinstance(peers_at_t, (int, float)):
                peer_adj = z - float(peers_at_t)

        peer_adjusted_series.append(float(round(peer_adj, 4)))
        is_flagged = (z >= 2.0) and (peer_adj >= 1.5)
        flags.append(is_flagged)

    persistence = calculate_persistence(flags)

    # Compute additional ratios: Stage 2 share, PCR, Credit Cost
    stage2_share_series: list[float | None] = []
    pcr_series: list[float | None] = []
    credit_cost_series: list[float | None] = []

    for t in range(n):
        b = gross_advances[t]
        g = gnpa_amount[t]

        # Stage 2 share
        if stage2_assets is not None and t < len(stage2_assets) and stage2_assets[t] is not None and b:
            stage2_share_series.append(float(round(stage2_assets[t] / b, 6)))
        else:
            stage2_share_series.append(None)

        # Provision Coverage Ratio
        if provisions_total is not None and t < len(provisions_total) and provisions_total[t] is not None and g and g > 0:
            pcr_series.append(float(round(provisions_total[t] / g, 6)))
        else:
            pcr_series.append(None)

        # Credit Cost
        if credit_cost is not None and t < len(credit_cost) and credit_cost[t] is not None:
            credit_cost_series.append(float(round(credit_cost[t], 6)))
        else:
            credit_cost_series.append(None)

    abstain_note = None
    if not has_writeoffs:
        abstain_note = (
            "Write-offs not disclosed; like-for-like restated without write-off add-backs. "
            "Write-off effect marked not tested."
        )

    return EngineResult(
        engine="quality",
        status="tested",
        abstain_reason=abstain_note,
        periods=periods,
        series={
            "reported": reported_series,
            "like_for_like": lfl_series,
            "gap": gap_series,
            "denominator_part": denom_parts,
            "writeoff_part": wo_parts,
            "z_scores": z_scores,
            "peer_adjusted": peer_adjusted_series,
            "stage2_share": stage2_share_series,
            "pcr": pcr_series,
            "credit_cost": credit_cost_series,
        },
        flags=flags,
        persistence_periods=persistence,
        ledger_ids=ledger_ids or [],
    )
