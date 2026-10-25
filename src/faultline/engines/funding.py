"""Funding Engine: Funding fault map and borrowing shift analysis.

Source of truth: FAULTLINE_Development_Plan.md Section 5.2 & Hard Rules R2, R4, R7.
Ordinary code computes every number.
Computes:
- Share of each funder type in total borrowings: share_i[t] = borrowings_i[t] / total_borrowings[t]
- Concentration index: HHI = sum(share_i^2)
- Short-term borrowing share: short_term_share = borrowings_short_term / total_borrowings
- Off-book exposure share:
    offbook_share = (assigned + securitised + co_lending_aum) / (gross_advances + that off-book amount)
- Shift across lag: shift_i[t] = share_i[t] - share_i[t-lag]
- Peer-adjusted shift flagging:
    flag = |shift_i - median(peer_shift_i)| >= max(0.05, 2 * peer_MAD_i)  # 5 percentage points minimum
- Persistence: periods the shift keeps the same sign
- Accounting check: shares sum to 100% within 0.5 percentage points (tolerance 0.005, Metric M08).
"""

from __future__ import annotations

from typing import Any
import numpy as np

from faultline.schemas import EngineResult

SHARES_SUM_TOLERANCE = 0.005  # 0.5 percentage points tolerance for Metric M08

STANDARD_FUNDER_TYPES = [
    "borrowings_bank",
    "borrowings_bonds",
    "borrowings_deposits",
    "borrowings_securitisation",
    "borrowings_foreign",
]


def compute_hhi(shares: list[float]) -> float:
    """Calculate the Herfindahl-Hirschman Index (HHI) for funding concentration.

    HHI = sum(share_i^2).
    Range: 1/N to 1.0 (higher = more concentrated).
    """
    valid_shares = [s for s in shares if s is not None and s >= 0.0]
    if not valid_shares:
        return 0.0
    return float(round(sum(s**2 for s in valid_shares), 6))


def compute_offbook_share(
    gross_advances: float | None,
    assigned_loans: float | None = None,
    securitised: float | None = None,
    co_lending_aum: float | None = None,
) -> float | None:
    """Calculate the off-book exposure share.

    Formula (Section 5.2):
        offbook_amount = assigned + securitised + co_lending_aum
        offbook_share = offbook_amount / (gross_advances + offbook_amount)

    Returns None if gross advances or all off-book components are undisclosed.
    """
    if gross_advances is None or gross_advances <= 0:
        return None

    components = [assigned_loans, securitised, co_lending_aum]
    disclosed_components = [c for c in components if c is not None]
    if not disclosed_components:
        return None

    offbook_amount = sum(disclosed_components)
    total_expanded_book = gross_advances + offbook_amount
    if total_expanded_book <= 0:
        return None

    return float(round(offbook_amount / total_expanded_book, 6))


def calculate_sign_persistence(shifts: list[float | None], flags: list[bool]) -> int:
    """Calculate the longest run of consecutive periods where flagged shift maintains the same sign."""
    max_streak = 0
    current_streak = 0
    current_sign = 0

    for shift, is_flagged in zip(shifts, flags):
        if not is_flagged or shift is None or shift == 0:
            current_streak = 0
            current_sign = 0
            continue

        sign = 1 if shift > 0 else -1
        if sign == current_sign:
            current_streak += 1
        else:
            current_sign = sign
            current_streak = 1

        if current_streak > max_streak:
            max_streak = current_streak

    return max_streak


def analyze_funding(
    periods: list[str],
    total_borrowings: list[float | None],
    borrowings_by_type: dict[str, list[float | None]],
    borrowings_short_term: list[float | None] | None = None,
    gross_advances: list[float | None] | None = None,
    assigned_loans_outstanding: list[float | None] | None = None,
    co_lending_aum: list[float | None] | None = None,
    peer_shifts: dict[str, list[list[float]]] | None = None,
    lag: int = 4,
    ledger_ids: list[str] | None = None,
) -> EngineResult:
    """Run full funding analysis, concentration metrics, and peer-adjusted shift detection.

    Args:
        periods: List of quarter/period names.
        total_borrowings: Total borrowing balance in each period.
        borrowings_by_type: Mapping of funder category name -> list of amounts.
        borrowings_short_term: Short-term borrowings in each period.
        gross_advances: Gross advances (needed for off-book ratio).
        assigned_loans_outstanding: Direct assignment balance.
        co_lending_aum: Co-lending balance.
        peer_shifts: Optional mapping of category -> per-period list of peer shifts.
        lag: Comparison lag (default 4 for quarterly data).
        ledger_ids: Tracking IDs for proof-carrying ledger records.

    Returns:
        EngineResult conforming to FAULTLINE schemas.
    """
    n = len(periods)
    if n == 0:
        return EngineResult(
            engine="funding",
            status="abstained",
            abstain_reason="No periods supplied for funding analysis.",
        )

    # Validate that total borrowings has sufficient disclosed data
    valid_tb_count = sum(1 for tb in total_borrowings if tb is not None and tb > 0)
    if valid_tb_count < lag + 1:
        return EngineResult(
            engine="funding",
            status="abstained",
            abstain_reason=(
                f"Insufficient total borrowings disclosures: required at least {lag + 1} periods, "
                f"found {valid_tb_count}."
            ),
            periods=periods,
            ledger_ids=ledger_ids or [],
        )

    funder_categories = list(borrowings_by_type.keys())
    if not funder_categories:
        return EngineResult(
            engine="funding",
            status="abstained",
            abstain_reason="No borrowing category breakdowns supplied.",
            periods=periods,
            ledger_ids=ledger_ids or [],
        )

    # 1. Compute per-period shares and HHI
    shares_by_category: dict[str, list[float | None]] = {cat: [] for cat in funder_categories}
    hhi_series: list[float | None] = []
    short_term_share_series: list[float | None] = []
    offbook_share_series: list[float | None] = []
    shares_sum_checks: list[bool] = []

    for t in range(n):
        tb = total_borrowings[t]
        if tb is None or tb <= 0:
            for cat in funder_categories:
                shares_by_category[cat].append(None)
            hhi_series.append(None)
            short_term_share_series.append(None)
            offbook_share_series.append(None)
            continue

        period_shares = []
        for cat in funder_categories:
            cat_val = borrowings_by_type[cat][t] if t < len(borrowings_by_type[cat]) else None
            if cat_val is not None and cat_val >= 0:
                s = float(round(cat_val / tb, 6))
                shares_by_category[cat].append(s)
                period_shares.append(s)
            else:
                shares_by_category[cat].append(None)

        # Sum check (Metric M08)
        if len(period_shares) == len(funder_categories):
            sum_shares = sum(period_shares)
            shares_sum_checks.append(abs(sum_shares - 1.0) <= SHARES_SUM_TOLERANCE)
        else:
            shares_sum_checks.append(False)

        # Concentration index
        hhi_series.append(compute_hhi(period_shares))

        # Short term share
        if borrowings_short_term is not None and t < len(borrowings_short_term) and borrowings_short_term[t] is not None:
            st_val = borrowings_short_term[t]
            short_term_share_series.append(float(round(st_val / tb, 6)))
        else:
            short_term_share_series.append(None)

        # Off-book exposure share
        adv = gross_advances[t] if gross_advances is not None and t < len(gross_advances) else None
        asgn = assigned_loans_outstanding[t] if assigned_loans_outstanding is not None and t < len(assigned_loans_outstanding) else None
        colend = co_lending_aum[t] if co_lending_aum is not None and t < len(co_lending_aum) else None
        sec = borrowings_by_type.get("borrowings_securitisation", [None] * n)[t]
        offbook_share_series.append(compute_offbook_share(adv, asgn, sec, colend))

    # 2. Compute shifts across lag
    shifts_by_category: dict[str, list[float | None]] = {cat: [] for cat in funder_categories}
    st_shifts: list[float | None] = []
    offbook_shifts: list[float | None] = []

    for t in range(n):
        if t < lag:
            for cat in funder_categories:
                shifts_by_category[cat].append(None)
            st_shifts.append(None)
            offbook_shifts.append(None)
            continue

        for cat in funder_categories:
            s_curr = shares_by_category[cat][t]
            s_lag = shares_by_category[cat][t - lag]
            if s_curr is not None and s_lag is not None:
                shifts_by_category[cat].append(float(round(s_curr - s_lag, 6)))
            else:
                shifts_by_category[cat].append(None)

        # Short term shift
        st_curr = short_term_share_series[t]
        st_lag = short_term_share_series[t - lag]
        if st_curr is not None and st_lag is not None:
            st_shifts.append(float(round(st_curr - st_lag, 6)))
        else:
            st_shifts.append(None)

        # Off-book shift
        ob_curr = offbook_share_series[t]
        ob_lag = offbook_share_series[t - lag]
        if ob_curr is not None and ob_lag is not None:
            offbook_shifts.append(float(round(ob_curr - ob_lag, 6)))
        else:
            offbook_shifts.append(None)

    # 3. Detect flagged shifts per period (Section 5.2)
    # flag = |shift_i - median(peer_shift_i)| >= max(0.05, 2 * peer_MAD_i)
    flags_per_category: dict[str, list[bool]] = {cat: [False] * n for cat in funder_categories}
    period_flags: list[bool] = [False] * n

    for t in range(lag, n):
        any_flag_at_t = False
        for cat in funder_categories:
            sh = shifts_by_category[cat][t]
            if sh is None:
                continue

            peer_median = 0.0
            peer_mad = 0.0
            if peer_shifts is not None and cat in peer_shifts and t < len(peer_shifts[cat]):
                peers_at_t = peer_shifts[cat][t]
                if peers_at_t:
                    peer_median = float(np.median(peers_at_t))
                    peer_mad = float(np.median(np.abs(np.array(peers_at_t) - peer_median)))

            threshold = max(0.05, 2.0 * peer_mad)
            is_cat_flagged = abs(sh - peer_median) >= threshold
            flags_per_category[cat][t] = is_cat_flagged
            if is_cat_flagged:
                any_flag_at_t = True

        period_flags[t] = any_flag_at_t

    # 4. Persistence: maximum consecutive periods the shift keeps the same sign
    cat_persistences = [
        calculate_sign_persistence(shifts_by_category[cat], flags_per_category[cat])
        for cat in funder_categories
    ]
    max_persistence = max(cat_persistences) if cat_persistences else 0

    # Package output series
    series_dict: dict[str, list[float | None]] = {
        "hhi": hhi_series,
        "short_term_share": short_term_share_series,
        "short_term_shift": st_shifts,
        "offbook_share": offbook_share_series,
        "offbook_shift": offbook_shifts,
    }
    for cat in funder_categories:
        series_dict[f"share_{cat}"] = shares_by_category[cat]
        series_dict[f"shift_{cat}"] = shifts_by_category[cat]

    return EngineResult(
        engine="funding",
        status="tested",
        periods=periods,
        series=series_dict,
        flags=period_flags,
        persistence_periods=max_persistence,
        ledger_ids=ledger_ids or [],
    )
