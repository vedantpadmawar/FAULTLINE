"""Canary lab synthetic financial data generator for FAULTLINE.

Source of truth: FAULTLINE_Development_Plan.md Section 7 Task T04.
Generates 40 synthetic lenders x 12 quarters:
- 20 clean lenders (7 are healthy FAST growers as negative controls, 13 standard).
- 20 with controlled planted faults:
  * denominator_masking (5 lenders)
  * writeoff_masking (5 lenders)
  * funding_shift (5 lenders)
  * combined (5 lenders)

Deterministic output given a fixed random seed.
Ensures zero real company names/tickers (Rule R1 compliance).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

# Ensure src/ is importable
src_path = str(Path(__file__).resolve().parent.parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import numpy as np

from faultline.schemas import LedgerEntry

PERIODS = [
    "FY23-Q1",
    "FY23-Q2",
    "FY23-Q3",
    "FY23-Q4",
    "FY24-Q1",
    "FY24-Q2",
    "FY24-Q3",
    "FY24-Q4",
    "FY25-Q1",
    "FY25-Q2",
    "FY25-Q3",
    "FY25-Q4",
]

ARCHETYPES = [
    # 20 clean lenders (13 standard + 7 healthy fast growers as negative controls)
    *([("clean_standard", None, None)] * 13),
    *([("clean_fast_grower", None, None)] * 7),
    # 20 planted faults
    *([("denominator_masking", "denominator_masking", "quality")] * 5),
    *([("writeoff_masking", "writeoff_masking", "quality")] * 5),
    *([("funding_shift", "funding_shift", "funding")] * 5),
    *([("combined", "combined", "both")] * 5),
]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def generate_single_lender(
    lender_id: str,
    archetype: str,
    fault_type: str | None,
    fault_engine: str | None,
    rng: np.random.Generator,
    periods: list[str] = PERIODS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Generate 12 periods of financial data and ground truth for a synthetic lender."""
    n_periods = len(periods)
    profile = "bank" if int(lender_id.split("_")[1]) % 2 == 1 else "nbfc"

    has_fault = fault_type is not None
    # Plant fault after establishing a 2-period clean like-for-like baseline (period index 6 -> FY24-Q3)
    fault_start_idx = 6 if has_fault else None
    fault_start_period = periods[fault_start_idx] if fault_start_idx is not None else None

    # Base figures
    base_advances = float(rng.uniform(25000.0, 120000.0))

    # Quarterly growth trajectory
    advances = [base_advances]
    for t in range(1, n_periods):
        if archetype == "clean_fast_grower":
            # Consistent healthy fast growth across all 12 periods (35-45% CAGR)
            q_growth = float(rng.normal(0.09, 0.008))
        elif archetype == "denominator_masking" or (archetype == "combined" and t >= (fault_start_idx or 99)):
            if t < (fault_start_idx or 99):
                q_growth = float(rng.normal(0.03, 0.005))
            else:
                # Sudden massive loan book surge to mask credit deterioration
                q_growth = float(rng.normal(0.15, 0.015))
        else:
            q_growth = float(rng.normal(0.035, 0.005))

        new_adv = advances[-1] * (1.0 + max(0.005, q_growth))
        advances.append(float(round(new_adv, 2)))

    # Bad loans and write-offs
    gnpa_amount = []
    writeoffs_period = []
    base_gnpa_rate = float(rng.uniform(0.025, 0.035)) if archetype != "clean_fast_grower" else float(rng.uniform(0.015, 0.022))

    for t in range(n_periods):
        b_t = advances[t]
        if archetype in ["clean_standard", "clean_fast_grower"]:
            gnpa = b_t * (base_gnpa_rate + float(rng.normal(0, 0.001)))
            wo = b_t * float(rng.uniform(0.002, 0.0035))
        elif archetype == "denominator_masking":
            if t < (fault_start_idx or 99):
                gnpa = b_t * (base_gnpa_rate + float(rng.normal(0, 0.001)))
                wo = b_t * float(rng.uniform(0.002, 0.0035))
            else:
                # In denominator masking, bad loans surge underlyingly, but fast advance expansion keeps ratio flat
                underlying_bad_ratio = base_gnpa_rate * 2.2
                gnpa = b_t * (underlying_bad_ratio * (advances[t - 1] / b_t))
                wo = b_t * float(rng.uniform(0.002, 0.0035))
        elif archetype == "writeoff_masking":
            if t < (fault_start_idx or 99):
                gnpa = b_t * (base_gnpa_rate + float(rng.normal(0, 0.001)))
                wo = b_t * float(rng.uniform(0.002, 0.0035))
            else:
                # In write-off masking, reported GNPA looks flat or falling, while write-offs surge dramatically
                gnpa = b_t * (base_gnpa_rate * float(rng.uniform(0.75, 0.90)))
                wo = b_t * float(rng.uniform(0.020, 0.032))
        elif archetype == "combined":
            if t < (fault_start_idx or 99):
                gnpa = b_t * (base_gnpa_rate + float(rng.normal(0, 0.001)))
                wo = b_t * float(rng.uniform(0.002, 0.0035))
            else:
                gnpa = b_t * (base_gnpa_rate * float(rng.uniform(0.85, 1.05)))
                wo = b_t * float(rng.uniform(0.018, 0.028))
        else:  # funding_shift only (quality is clean)
            gnpa = b_t * (base_gnpa_rate + float(rng.normal(0, 0.001)))
            wo = b_t * float(rng.uniform(0.002, 0.0035))

        gnpa_amount.append(float(round(max(10.0, gnpa), 2)))
        writeoffs_period.append(float(round(max(2.0, wo), 2)))

    # Derived asset quality fields
    nnpa_amount = []
    provisions_total = []
    stage1_assets = []
    stage2_assets = []
    stage3_assets = []
    credit_cost = []

    for t in range(n_periods):
        g = gnpa_amount[t]
        b = advances[t]
        pcr = float(rng.uniform(0.60, 0.72))
        nnpa = g * (1.0 - pcr)
        prov = (g - nnpa) + b * 0.004

        s3 = g
        s2 = b * float(rng.uniform(0.03, 0.045))
        s1 = b - s2 - s3
        cc = (writeoffs_period[t] + (prov * 0.15)) / b

        nnpa_amount.append(float(round(nnpa, 2)))
        provisions_total.append(float(round(prov, 2)))
        stage1_assets.append(float(round(s1, 2)))
        stage2_assets.append(float(round(s2, 2)))
        stage3_assets.append(float(round(s3, 2)))
        credit_cost.append(float(round(cc * 100, 3)))  # annualized percentage

    # Funding Mix
    total_borrowings = [float(round(adv * float(rng.uniform(0.80, 0.90)), 2)) for adv in advances]
    borrowings_bank = []
    borrowings_bonds = []
    borrowings_deposits = []
    borrowings_securitisation = []
    borrowings_foreign = []
    borrowings_short_term = []
    assigned_loans_outstanding = []
    co_lending_aum = []

    for t in range(n_periods):
        tb = total_borrowings[t]
        adv = advances[t]
        is_funding_fault = (archetype in ["funding_shift", "combined"]) and (t >= (fault_start_idx or 99))

        if profile == "bank":
            if not is_funding_fault:
                s_dep = 0.60
                s_bank = 0.18
                s_bonds = 0.14
                s_sec = 0.04
                s_for = 0.04
                st_share = 0.08
                offbook_share = 0.03
            else:
                # Bank funding fault: bank wholesale borrowings drop, short-term debt spikes
                s_dep = 0.58
                s_bank = 0.04  # sharp fall
                s_bonds = 0.14
                s_sec = 0.12  # off-book securitisation jumps
                s_for = 0.12
                st_share = 0.28  # sharp short-term rise
                offbook_share = 0.18
        else:  # NBFC
            if not is_funding_fault:
                s_bank = 0.45
                s_bonds = 0.35
                s_dep = 0.08
                s_sec = 0.07
                s_for = 0.05
                st_share = 0.10
                offbook_share = 0.07
            else:
                # NBFC funding fault: bank lines frozen, off-book and short-term CP surge
                s_bank = 0.20  # -25 pct drop
                s_bonds = 0.35
                s_dep = 0.05
                s_sec = 0.25  # securitisation replacement
                s_for = 0.15
                st_share = 0.32  # short-term liquidity reliance
                offbook_share = 0.24

        # Add minor random jitter while preserving sum == tb
        raw_shares = np.array([s_bank, s_bonds, s_dep, s_sec, s_for])
        jitter = rng.normal(0, 0.005, size=5)
        adjusted_shares = np.maximum(0.01, raw_shares + jitter)
        norm_shares = adjusted_shares / np.sum(adjusted_shares)

        b_bank = round(float(tb * norm_shares[0]), 2)
        b_bonds = round(float(tb * norm_shares[1]), 2)
        b_dep = round(float(tb * norm_shares[2]), 2)
        b_sec = round(float(tb * norm_shares[3]), 2)
        # Residual to guarantee exact sum to total_borrowings
        b_for = round(tb - (b_bank + b_bonds + b_dep + b_sec), 2)

        st_debt = round(float(tb * st_share), 2)
        assigned = round(float(adv * offbook_share * 0.5), 2)
        co_lend = round(float(adv * offbook_share * 0.5), 2)

        borrowings_bank.append(b_bank)
        borrowings_bonds.append(b_bonds)
        borrowings_deposits.append(b_dep)
        borrowings_securitisation.append(b_sec)
        borrowings_foreign.append(b_for)
        borrowings_short_term.append(st_debt)
        assigned_loans_outstanding.append(assigned)
        co_lending_aum.append(co_lend)

    time_series = {
        "periods": periods,
        "gross_advances": advances,
        "gnpa_amount": gnpa_amount,
        "nnpa_amount": nnpa_amount,
        "writeoffs_period": writeoffs_period,
        "stage1_assets": stage1_assets,
        "stage2_assets": stage2_assets,
        "stage3_assets": stage3_assets,
        "provisions_total": provisions_total,
        "credit_cost": credit_cost,
        "total_borrowings": total_borrowings,
        "borrowings_bank": borrowings_bank,
        "borrowings_bonds": borrowings_bonds,
        "borrowings_deposits": borrowings_deposits,
        "borrowings_securitisation": borrowings_securitisation,
        "borrowings_foreign": borrowings_foreign,
        "borrowings_short_term": borrowings_short_term,
        "assigned_loans_outstanding": assigned_loans_outstanding,
        "co_lending_aum": co_lending_aum,
    }

    lender_data = {
        "lender_id": lender_id,
        "profile": profile,
        "archetype": archetype,
        "time_series": time_series,
    }

    ground_truth = {
        "lender_id": lender_id,
        "profile": profile,
        "archetype": archetype,
        "has_fault": has_fault,
        "fault_type": fault_type,
        "fault_engine": fault_engine,
        "fault_start_idx": fault_start_idx,
        "fault_start_period": fault_start_period,
    }

    return lender_data, ground_truth


def generate_canary_dataset(
    seed: int = 42,
    periods: list[str] = PERIODS,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Generate all 40 synthetic lenders and ground truth labels deterministically."""
    rng = np.random.default_rng(seed)
    lenders_data = []
    ground_truth_map = {}

    for idx, (archetype, fault_type, fault_engine) in enumerate(ARCHETYPES, start=1):
        lender_id = f"SYNTH_{idx:02d}"
        ldata, gt = generate_single_lender(
            lender_id=lender_id,
            archetype=archetype,
            fault_type=fault_type,
            fault_engine=fault_engine,
            rng=rng,
            periods=periods,
        )
        lenders_data.append(ldata)
        ground_truth_map[lender_id] = gt

    return lenders_data, ground_truth_map


def lender_to_ledger_entries(lender_data: dict[str, Any]) -> list[LedgerEntry]:
    """Convert a synthetic lender's time series into validated LedgerEntry objects."""
    lender_id = lender_data["lender_id"]
    ts = lender_data["time_series"]
    periods = ts["periods"]
    entries: list[LedgerEntry] = []

    retrieval_dt = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    mock_url = f"https://canary.faultline.test/filings/{lender_id}.pdf"
    mock_sha256 = _sha256(f"CANARY_SOURCE_{lender_id}")

    entry_num = 1
    for t_idx, period in enumerate(periods):
        for metric, values in ts.items():
            if metric == "periods":
                continue
            val = values[t_idx]
            unit = "pct" if metric == "credit_cost" else "INR_cr"
            snippet = f"{metric} for {period} was {val} {unit}"
            entry = LedgerEntry(
                id=f"{lender_id}_E{entry_num:04d}",
                metric=metric,
                value=float(val),
                unit=unit,
                period=period,
                period_kind="Q",
                source_url=mock_url,
                source_type="filing",
                doc_page=t_idx + 1,
                snippet=snippet,
                source_sha256=mock_sha256,
                retrieved_at=retrieval_dt,
            )
            entries.append(entry)
            entry_num += 1

    return entries


def save_canary_dataset(
    output_dir: Path | str, seed: int = 42
) -> tuple[Path, Path]:
    """Generate and write canary synthetic lenders and ground truth to disk."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    lenders, ground_truth = generate_canary_dataset(seed=seed)

    lenders_path = out / "synthetic_lenders.json"
    gt_path = out / "ground_truth.json"

    with open(lenders_path, "w", encoding="utf-8") as f:
        json.dump(lenders, f, indent=2)

    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2)

    return lenders_path, gt_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FAULTLINE Canary Lab Data Generator")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval/canary",
        help="Directory to save generated datasets",
    )
    args = parser.parse_args()

    l_path, g_path = save_canary_dataset(args.output_dir, seed=args.seed)
    print(f"Generated 40 synthetic lenders saved to: {l_path}")
    print(f"Ground truth labels saved to: {g_path}")
