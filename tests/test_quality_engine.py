"""Unit tests for the Quality Engine (Section 5.1 of FAULTLINE_Development_Plan.md).

Tests:
1. Hand-calculated fault anatomy decomposition.
2. Exact accounting identity closure (|gap - (denom + wo)| < 1e-9).
3. Missing write-off behavior (Rule R4: no silent zeroing, note attached, identity holds).
4. Insufficient data triggers explicit abstention.
5. Supplementary ratios: Stage 2 share, PCR, and Credit Cost.
6. Robust z-score computation against history excluding latest 2 periods.
7. Peer-adjusted flagging and persistence tracking.
8. Healthy fast-grower canary negative control (no false alarm persistence).
"""

import sys
from pathlib import Path
import pytest
import numpy as np

# Ensure src/ is importable
src_path = str(Path(__file__).resolve().parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from faultline.engines.quality import (
    IDENTITY_TOLERANCE,
    analyze_quality,
    calculate_persistence,
    compute_fault_anatomy,
    compute_robust_z_scores,
)
from eval.canary.generate import generate_canary_dataset


def test_hand_calculated_fault_anatomy():
    """Verify exact hand-calculated figures:

    B_lag = 10,000 cr, B_curr = 13,000 cr
    G_curr = 390 cr, W_sum = 140 cr
    reported = 390 / 13000 = 0.03 (3.0%)
    like_for_like = (390 + 140) / 10000 = 0.053 (5.3%)
    gap = 0.053 - 0.03 = 0.023 (2.3%)
    denom_part = 390 * (1/10000 - 1/13000) = 0.009 (0.9%)
    wo_part = 140 / 10000 = 0.014 (1.4%)
    identity: 0.009 + 0.014 = 0.023
    """
    decomp = compute_fault_anatomy(
        b_curr=13000.0,
        b_lag=10000.0,
        g_curr=390.0,
        w_sum=140.0,
        period="FY25-Q1",
    )

    assert pytest.approx(decomp.reported_ratio, abs=1e-6) == 0.030000
    assert pytest.approx(decomp.like_for_like_ratio, abs=1e-6) == 0.053000
    assert pytest.approx(decomp.gap, abs=1e-6) == 0.023000
    assert pytest.approx(decomp.denominator_part, abs=1e-6) == 0.009000
    assert pytest.approx(decomp.writeoff_part, abs=1e-6) == 0.014000

    # Metric M08 identity check
    assert decomp.identity_diff < IDENTITY_TOLERANCE
    assert decomp.identity_holds is True
    assert decomp.writeoffs_disclosed is True


def test_fault_anatomy_identity_random_inputs():
    """Verify the fault anatomy identity holds to 1e-9 across 50 random inputs."""
    rng = np.random.default_rng(12345)
    for _ in range(50):
        b_lag = float(rng.uniform(5000.0, 100000.0))
        b_curr = float(b_lag * rng.uniform(0.9, 1.5))
        g_curr = float(b_curr * rng.uniform(0.01, 0.10))
        w_sum = float(b_lag * rng.uniform(0.001, 0.02))

        decomp = compute_fault_anatomy(
            b_curr=b_curr,
            b_lag=b_lag,
            g_curr=g_curr,
            w_sum=w_sum,
        )
        assert decomp.identity_diff < IDENTITY_TOLERANCE
        assert decomp.identity_holds is True


def test_missing_writeoffs_behavior_rule_r4():
    """Rule R4: Missing write-offs must NOT be silently treated as zero.

    Decomposition is performed without write-offs, labelled as 'write-offs not added back',
    and identity still holds.
    """
    decomp = compute_fault_anatomy(
        b_curr=13000.0,
        b_lag=10000.0,
        g_curr=390.0,
        w_sum=None,  # Not disclosed
        period="FY25-Q2",
    )

    assert decomp.writeoffs_disclosed is False
    assert decomp.writeoff_part == 0.0
    # Gap is purely the denominator expansion effect
    assert pytest.approx(decomp.gap, abs=1e-6) == decomp.denominator_part
    assert decomp.identity_diff < IDENTITY_TOLERANCE
    assert decomp.identity_holds is True


def test_abstain_on_insufficient_data():
    """Rule R4: Engine must return status='abstained' when required inputs are missing."""
    # Only 3 periods provided, but lag is 4
    res = analyze_quality(
        periods=["FY24-Q1", "FY24-Q2", "FY24-Q3"],
        gross_advances=[1000.0, 1100.0, 1200.0],
        gnpa_amount=[30.0, 33.0, 36.0],
        lag=4,
    )
    assert res.status == "abstained"
    assert "Insufficient disclosed data" in (res.abstain_reason or "")
    assert res.persistence_periods == 0


def test_supplementary_ratios_calculated_when_disclosed():
    """Verify Stage 2 share, PCR, and Credit Cost calculation."""
    periods = ["FY24-Q1", "FY24-Q2", "FY24-Q3", "FY24-Q4", "FY25-Q1"]
    advances = [10000.0, 10500.0, 11000.0, 11500.0, 12000.0]
    gnpa = [300.0, 315.0, 330.0, 345.0, 360.0]
    stage2 = [400.0, 420.0, 440.0, 460.0, 480.0]
    provisions = [210.0, 220.0, 231.0, 241.5, 252.0]
    credit_cost = [1.2, 1.25, 1.15, 1.30, 1.22]

    res = analyze_quality(
        periods=periods,
        gross_advances=advances,
        gnpa_amount=gnpa,
        stage2_assets=stage2,
        provisions_total=provisions,
        credit_cost=credit_cost,
        lag=4,
    )

    assert res.status == "tested"
    # Stage 2 share at t=4: 480 / 12000 = 0.04 (4.0%)
    assert pytest.approx(res.series["stage2_share"][4], abs=1e-4) == 0.04
    # PCR at t=4: 252 / 360 = 0.70 (70.0%)
    assert pytest.approx(res.series["pcr"][4], abs=1e-4) == 0.70
    # Credit cost
    assert pytest.approx(res.series["credit_cost"][4], abs=1e-4) == 1.22


def test_robust_z_scores_excludes_latest_two_periods():
    """Verify robust z-score uses history excluding latest 2 periods."""
    # 6 gap values: [0.01, 0.012, 0.011, 0.01, 0.05, 0.06]
    # History (excluding latest 2) = [0.01, 0.012, 0.011, 0.01]
    # median = 0.0105
    # deviations from 0.0105: [0.0005, 0.0015, 0.0005, 0.0005] -> median MAD = 0.0005
    # spread = 1.4826 * 0.0005 = 0.0007413
    gaps = [0.01, 0.012, 0.011, 0.010, 0.050, 0.060]
    z_scores, center, spread = compute_robust_z_scores(gaps, exclude_recent=2)

    assert center is not None and pytest.approx(center, abs=1e-5) == 0.0105
    assert spread is not None and pytest.approx(spread, abs=1e-5) == 0.0007413

    # For the last period (0.060): z = (0.060 - 0.0105) / 0.0007413 ~ 66.7 (huge spike)
    assert z_scores[-1] is not None and z_scores[-1] > 20.0


def test_persistence_calculation():
    """Verify persistence streak calculation."""
    assert calculate_persistence([False, False, False]) == 0
    assert calculate_persistence([True, True, False, True]) == 2
    assert calculate_persistence([False, True, True, True, True, False]) == 4


def test_healthy_fast_grower_negative_control():
    """Canary negative control: A healthy fast grower with consistent history should NOT trigger persistent fault alarms."""
    lenders, gt = generate_canary_dataset(seed=42)
    # Pick a clean fast grower
    fast_growers = [l for l in lenders if l["archetype"] == "clean_fast_grower"]
    sample_fg = fast_growers[0]
    ts = sample_fg["time_series"]

    res = analyze_quality(
        periods=ts["periods"],
        gross_advances=ts["gross_advances"],
        gnpa_amount=ts["gnpa_amount"],
        writeoffs_period=ts["writeoffs_period"],
        stage2_assets=ts["stage2_assets"],
        provisions_total=ts["provisions_total"],
        credit_cost=ts["credit_cost"],
        lag=4,
    )

    assert res.status == "tested"
    # Because fast growth is steady across all 12 quarters, the historical center is high
    # and deviations are modest. It should not exhibit high persistence of fault alarms (e.g. persistence < 3)
    assert res.persistence_periods < 3
