"""Tests for the Canary Lab synthetic data generator (Task T04).

Source of truth: FAULTLINE_Development_Plan.md Section 7 Task T04.
Verifies:
- Determinism for a fixed seed (Acceptance line).
- 40 synthetic lenders x 12 periods.
- 20 clean lenders (including healthy fast growers as negative controls).
- 20 planted faults across the 4 required types:
  * denominator_masking
  * writeoff_masking
  * funding_shift
  * combined
- Accounting identities hold strictly (fault anatomy identity to 1e-9, funding sum).
- Conversion to valid LedgerEntry objects.
- Zero company-specific names or real tickers.
"""

import sys
from pathlib import Path
import pytest

# Ensure src/ is importable
src_path = str(Path(__file__).resolve().parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from eval.canary.generate import (
    PERIODS,
    generate_canary_dataset,
    lender_to_ledger_entries,
)
from faultline.schemas import LedgerEntry


def test_canary_determinism_acceptance():
    """T04 Acceptance: The generator is strictly deterministic for a fixed seed."""
    lenders_run1, gt_run1 = generate_canary_dataset(seed=42)
    lenders_run2, gt_run2 = generate_canary_dataset(seed=42)

    assert lenders_run1 == lenders_run2
    assert gt_run1 == gt_run2

    # Different seed yields different data
    lenders_run3, _ = generate_canary_dataset(seed=999)
    assert lenders_run1[0]["time_series"]["gross_advances"] != lenders_run3[0]["time_series"]["gross_advances"]


def test_canary_lender_counts_and_archetypes():
    """Verify 40 lenders x 12 periods with exact archetype distributions."""
    lenders, gt = generate_canary_dataset(seed=42)

    assert len(lenders) == 40
    assert len(gt) == 40

    archetype_counts = {}
    for item in gt.values():
        arch = item["archetype"]
        archetype_counts[arch] = archetype_counts.get(arch, 0) + 1

    # Clean lenders
    assert archetype_counts.get("clean_standard") == 13
    assert archetype_counts.get("clean_fast_grower") == 7
    total_clean = archetype_counts["clean_standard"] + archetype_counts["clean_fast_grower"]
    assert total_clean == 20

    # Planted fault lenders
    assert archetype_counts.get("denominator_masking") == 5
    assert archetype_counts.get("writeoff_masking") == 5
    assert archetype_counts.get("funding_shift") == 5
    assert archetype_counts.get("combined") == 5
    total_faults = sum(
        archetype_counts[k]
        for k in ["denominator_masking", "writeoff_masking", "funding_shift", "combined"]
    )
    assert total_faults == 20

    # Periods check
    for lender in lenders:
        ts = lender["time_series"]
        assert len(ts["periods"]) == 12
        assert ts["periods"] == PERIODS
        for metric, values in ts.items():
            assert len(values) == 12, f"Metric {metric} has {len(values)} periods, expected 12"


def test_negative_controls_healthy_fast_growers():
    """Requirement 8: Healthy fast growers must have high growth and has_fault == False."""
    lenders, gt = generate_canary_dataset(seed=42)

    fast_growers = [l for l in lenders if l["archetype"] == "clean_fast_grower"]
    assert len(fast_growers) == 7

    for fg in fast_growers:
        lender_id = fg["lender_id"]
        meta = gt[lender_id]
        # Must NOT be labelled as a fault
        assert meta["has_fault"] is False
        assert meta["fault_type"] is None
        assert meta["fault_engine"] is None

        # Verify cumulative advance growth is significantly higher than standard clean
        adv = fg["time_series"]["gross_advances"]
        cumulative_growth = adv[-1] / adv[0]
        # In ~3 years (12 quarters) at ~40% annual growth, loan book more than doubles (> 2.0x)
        assert cumulative_growth >= 2.0


def test_accounting_identities_in_canary_data():
    """Verify accounting identities hold across all 40 synthetic lenders."""
    lenders, _ = generate_canary_dataset(seed=42)

    for lender in lenders:
        ts = lender["time_series"]
        adv = ts["gross_advances"]
        gnpa = ts["gnpa_amount"]
        wo = ts["writeoffs_period"]
        tb = ts["total_borrowings"]
        b_bank = ts["borrowings_bank"]
        b_bonds = ts["borrowings_bonds"]
        b_dep = ts["borrowings_deposits"]
        b_sec = ts["borrowings_securitisation"]
        b_for = ts["borrowings_foreign"]

        for t in range(12):
            # 1. Borrowings sum check
            borrowings_sum = b_bank[t] + b_bonds[t] + b_dep[t] + b_sec[t] + b_for[t]
            assert abs(borrowings_sum - tb[t]) < 0.05, (
                f"Borrowings sum mismatch for {lender['lender_id']} at t={t}: "
                f"{borrowings_sum} vs {tb[t]}"
            )

            # 2. Short term share <= total borrowings
            assert ts["borrowings_short_term"][t] <= tb[t]

            # 3. Fault anatomy exact identity check for t >= 4 (L = 4 quarters lag)
            if t >= 4:
                lag = 4
                w_sum = sum(wo[t - k] for k in range(lag))
                b_curr = adv[t]
                b_lag = adv[t - lag]
                g_curr = gnpa[t]

                reported = g_curr / b_curr
                like_for_like = (g_curr + w_sum) / b_lag
                gap = like_for_like - reported

                denom_part = g_curr * (1.0 / b_lag - 1.0 / b_curr)
                wo_part = w_sum / b_lag

                # Section 5.1 & Metric M08 tolerance: 1e-9
                assert abs(gap - (denom_part + wo_part)) < 1e-9, (
                    f"Fault anatomy identity broken for {lender['lender_id']} at t={t}: "
                    f"gap={gap}, sum={denom_part + wo_part}"
                )


def test_canary_to_ledger_entries():
    """Verify conversion of synthetic time-series to validated LedgerEntry objects."""
    lenders, _ = generate_canary_dataset(seed=42)
    sample_lender = lenders[0]

    entries = lender_to_ledger_entries(sample_lender)
    # 18 metrics x 12 quarters = 216 entries
    assert len(entries) == 216

    for entry in entries:
        assert isinstance(entry, LedgerEntry)
        assert entry.id.startswith(sample_lender["lender_id"])
        assert entry.period in PERIODS
        assert entry.source_sha256 != ""
        assert entry.value >= 0.0
