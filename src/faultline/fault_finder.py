"""Fault Finder: Cross-engine candidate window detection.

Source of truth: FAULTLINE_Development_Plan.md Section 5.4.
Candidate windows:
- Periods where at least 2 engines flag within +-1 period of each other, OR
- One numeric engine has persistence >= 3 (drift then only corroborates).
Records which engine flagged first.
A sequence is a hypothesis, not proof of cause.
"""

from __future__ import annotations

from typing import Any

from faultline.schemas import Candidate, DriftItem, EngineResult


def find_candidate_faults(
    quality_result: EngineResult | None = None,
    funding_result: EngineResult | None = None,
    drift_items: list[DriftItem] | None = None,
    periods: list[str] | None = None,
    patterns_tested_count: int = 1,
) -> list[Candidate]:
    """Identify candidate fault windows across Quality, Funding, and Drift engines.

    Rules (Section 5.4):
    1. At least 2 engines flag within +-1 period of each other, OR
    2. One numeric engine has persistence >= 3.
    """
    candidates: list[Candidate] = []
    drift_items = drift_items or []

    # Identify shared periods
    all_periods = periods
    if not all_periods:
        if quality_result and quality_result.periods:
            all_periods = quality_result.periods
        elif funding_result and funding_result.periods:
            all_periods = funding_result.periods
        else:
            all_periods = []

    if not all_periods:
        return []

    n = len(all_periods)
    q_flags = quality_result.flags if quality_result and quality_result.status == "tested" else [False] * n
    f_flags = funding_result.flags if funding_result and funding_result.status == "tested" else [False] * n

    # Check condition 2: One numeric engine has persistence >= 3
    q_persist = quality_result.persistence_periods if quality_result else 0
    f_persist = funding_result.persistence_periods if funding_result else 0

    has_persistent_numeric = (q_persist >= 3) or (f_persist >= 3)

    # Check condition 1: Co-occurrence within +-1 period
    co_occurring_indices: set[int] = set()
    for t in range(n):
        # Look in window [t-1, t, t+1]
        window_indices = [idx for idx in [t - 1, t, t + 1] if 0 <= idx < n]
        q_active = any(q_flags[idx] for idx in window_indices)
        f_active = any(f_flags[idx] for idx in window_indices)
        if q_active and f_active:
            co_occurring_indices.add(t)

    is_candidate = has_persistent_numeric or len(co_occurring_indices) > 0

    if not is_candidate:
        return []

    # Find earliest flag period
    flagged_periods_info: list[tuple[int, str]] = []
    for t in range(n):
        if q_flags[t]:
            flagged_periods_info.append((t, "quality"))
        if f_flags[t]:
            flagged_periods_info.append((t, "funding"))

    if not flagged_periods_info:
        return []

    flagged_periods_info.sort(key=lambda x: x[0])
    first_idx, first_engine = flagged_periods_info[0]
    last_idx = max(x[0] for x in flagged_periods_info)

    engines_involved = sorted(list({x[1] for x in flagged_periods_info}))
    if drift_items:
        engines_involved.append("drift")

    fault_start_period = all_periods[first_idx]
    window = (all_periods[first_idx], all_periods[last_idx])

    # Summary numbers (code-computed only, no LLM math)
    summary_numbers: dict[str, float] = {}
    if quality_result and quality_result.series:
        gaps = [g for g in quality_result.series.get("gap", []) if g is not None]
        if gaps:
            summary_numbers["max_gap_bps"] = float(round(max(gaps) * 10000, 2))
        z_scores = [z for z in quality_result.series.get("z_scores", []) if z is not None]
        if z_scores:
            summary_numbers["max_quality_z"] = float(round(max(z_scores), 2))
        summary_numbers["quality_persistence"] = float(q_persist)

    if funding_result and funding_result.series:
        summary_numbers["funding_persistence"] = float(f_persist)
        st_shifts = [s for s in funding_result.series.get("short_term_shift", []) if s is not None]
        if st_shifts:
            summary_numbers["max_short_term_shift_pct"] = float(round(max(st_shifts) * 100, 2))

    cand = Candidate(
        id=f"CAND-{fault_start_period}",
        engines=engines_involved,
        window=window,
        fault_start_period=fault_start_period,
        summary_numbers=summary_numbers,
        drift_items=drift_items,
        hypotheses=[],
        red_team_log=[],
        gate={"patterns_tested": patterns_tested_count, "first_engine": first_engine},
        state="unexplained",
    )
    candidates.append(cand)
    return candidates
