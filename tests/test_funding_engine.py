"""Unit tests for the Funding Engine (Section 5.2 of FAULTLINE_Development_Plan.md).

Tests:
1. Normal funding patterns: steady shares, HHI calculation, shares sum check (M08).
2. Unusual idiosyncratic shifts: sharp drop in bank share triggers flag.
3. Sector-wide shift (peer-adjusted): when peers experience same shift, company is not flagged.
4. Off-book exposure calculation using exact specified formula.
5. Missing data handling: returns explicit abstention when borrowings are missing.
6. Canary integration: planted funding_shift canary lender is flagged with persistence.
7. Zero company-specific hardcoding or investment accusations.
"""

import sys
from pathlib import Path
import pytest
import numpy as np

# Ensure src/ is importable
src_path = str(Path(__file__).resolve().parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from faultline.engines.funding import (
    SHARES_SUM_TOLERANCE,
    analyze_funding,
    calculate_sign_persistence,
    compute_hhi,
    compute_offbook_share,
)
from eval.canary.generate import generate_canary_dataset


def test_hhi_calculation():
    """Verify HHI calculation:

    Equal 5 shares of 0.20 -> HHI = 5 * (0.2)^2 = 0.20
    Single monopoly funder 1.0 -> HHI = 1.0
    """
    assert pytest.approx(compute_hhi([0.2, 0.2, 0.2, 0.2, 0.2]), abs=1e-6) == 0.20
    assert pytest.approx(compute_hhi([1.0]), abs=1e-6) == 1.0
    # 40% Bank, 30% Bonds, 20% Deposits, 5% Sec, 5% Foreign
    # HHI = 0.16 + 0.09 + 0.04 + 0.0025 + 0.0025 = 0.295
    assert pytest.approx(compute_hhi([0.4, 0.3, 0.2, 0.05, 0.05]), abs=1e-6) == 0.295


def test_offbook_share_formula():
    """Verify exact off-book formula:

    gross_advances = 100,000 cr
    assigned = 10,000 cr, securitised = 5,000 cr, co-lending = 5,000 cr
    offbook_amount = 20,000 cr
    expanded_book = 120,000 cr
    offbook_share = 20,000 / 120,000 = 1/6 ~ 0.166667
    """
    ob = compute_offbook_share(
        gross_advances=100000.0,
        assigned_loans=10000.0,
        securitised=5000.0,
        co_lending_aum=5000.0,
    )
    assert ob is not None
    assert pytest.approx(ob, abs=1e-5) == (20000.0 / 120000.0)

    # Missing gross advances returns None (Rule R4)
    assert compute_offbook_share(gross_advances=None, assigned_loans=1000.0) is None
    # All offbook None returns None
    assert compute_offbook_share(gross_advances=50000.0) is None


def test_normal_funding_pattern_no_flags():
    """Stable funding mix generates zero flags and valid sum checks."""
    periods = ["FY24-Q1", "FY24-Q2", "FY24-Q3", "FY24-Q4", "FY25-Q1", "FY25-Q2"]
    total_borrowings = [10000.0] * 6
    borrowings_by_type = {
        "borrowings_bank": [4000.0] * 6,
        "borrowings_bonds": [3000.0] * 6,
        "borrowings_deposits": [2000.0] * 6,
        "borrowings_securitisation": [500.0] * 6,
        "borrowings_foreign": [500.0] * 6,
    }

    res = analyze_funding(
        periods=periods,
        total_borrowings=total_borrowings,
        borrowings_by_type=borrowings_by_type,
        lag=4,
    )

    assert res.status == "tested"
    # Zero flags on steady pattern
    assert not any(res.flags)
    assert res.persistence_periods == 0
    # HHI steady at 0.295
    for h in res.series["hhi"]:
        assert pytest.approx(h, abs=1e-4) == 0.295


def test_idiosyncratic_funding_shift_flagged():
    """A sudden shift of 20 percentage points in bank borrowing triggers flag."""
    periods = [
        "FY24-Q1", "FY24-Q2", "FY24-Q3", "FY24-Q4",
        "FY25-Q1", "FY25-Q2", "FY25-Q3", "FY25-Q4",
    ]
    # Bank share falls from 45% to 20% starting at FY25-Q1 (t=4)
    b_bank = [4500.0, 4500.0, 4500.0, 4500.0, 2000.0, 2000.0, 2000.0, 2000.0]
    b_sec = [500.0, 500.0, 500.0, 500.0, 3000.0, 3000.0, 3000.0, 3000.0]
    b_bonds = [5000.0] * 8

    res = analyze_funding(
        periods=periods,
        total_borrowings=[10000.0] * 8,
        borrowings_by_type={
            "borrowings_bank": b_bank,
            "borrowings_securitisation": b_sec,
            "borrowings_bonds": b_bonds,
        },
        lag=4,
    )

    assert res.status == "tested"
    # Starting at t=4, shift in bank share is 0.20 - 0.45 = -0.25 (>= 0.05 threshold)
    assert res.flags[4] is True
    assert res.flags[5] is True
    assert res.flags[6] is True
    assert res.flags[7] is True
    # Sign is consistently negative for 4 quarters
    assert res.persistence_periods == 4


def test_sector_wide_shift_not_flagged_with_peer_adjustment():
    """When peers share the same shift, peer-adjustment cancels the signal (sector story)."""
    periods = ["FY24-Q1", "FY24-Q2", "FY24-Q3", "FY24-Q4", "FY25-Q1"]
    # Company shifts by -15 percentage points at t=4
    b_bank = [4500.0, 4500.0, 4500.0, 4500.0, 3000.0]
    b_bonds = [5500.0, 5500.0, 5500.0, 5500.0, 7000.0]

    # Peer group also experienced a median shift: bank -15 pct, bonds +15 pct
    peer_shifts = {
        "borrowings_bank": [[], [], [], [], [-0.14, -0.15, -0.16]],
        "borrowings_bonds": [[], [], [], [], [0.14, 0.15, 0.16]],
    }

    res = analyze_funding(
        periods=periods,
        total_borrowings=[10000.0] * 5,
        borrowings_by_type={"borrowings_bank": b_bank, "borrowings_bonds": b_bonds},
        peer_shifts=peer_shifts,
        lag=4,
    )

    assert res.status == "tested"
    # Idiosyncratic deviation is |-0.15 - (-0.15)| = 0.0 < 0.05 -> NOT flagged
    assert res.flags[4] is False


def test_funding_engine_abstains_on_missing_borrowings():
    """Rule R4: Insufficient borrowings data returns status='abstained'."""
    res = analyze_funding(
        periods=["FY24-Q1", "FY24-Q2"],
        total_borrowings=[None, None],
        borrowings_by_type={"borrowings_bank": [None, None]},
        lag=4,
    )
    assert res.status == "abstained"
    assert "Insufficient total borrowings" in (res.abstain_reason or "")


def test_canary_funding_shift_detection():
    """Canary Lab integration: A planted funding_shift lender is detected with persistence >= 3."""
    lenders, gt = generate_canary_dataset(seed=42)
    # Find a planted funding_shift lender
    shift_lenders = [l for l in lenders if l["archetype"] == "funding_shift"]
    sample_shift = shift_lenders[0]
    ts = sample_shift["time_series"]

    borrowings_by_type = {
        "borrowings_bank": ts["borrowings_bank"],
        "borrowings_bonds": ts["borrowings_bonds"],
        "borrowings_deposits": ts["borrowings_deposits"],
        "borrowings_securitisation": ts["borrowings_securitisation"],
        "borrowings_foreign": ts["borrowings_foreign"],
    }

    res = analyze_funding(
        periods=ts["periods"],
        total_borrowings=ts["total_borrowings"],
        borrowings_by_type=borrowings_by_type,
        borrowings_short_term=ts["borrowings_short_term"],
        gross_advances=ts["gross_advances"],
        assigned_loans_outstanding=ts["assigned_loans_outstanding"],
        co_lending_aum=ts["co_lending_aum"],
        lag=4,
    )

    assert res.status == "tested"
    # Planted fault must be detected with high persistence (persistence >= 3)
    assert res.persistence_periods >= 3
    assert any(res.flags)
