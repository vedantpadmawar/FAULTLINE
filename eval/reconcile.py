"""Metric M01 evaluation: Reconciliation rate of computed vs published ratios.

Source of truth: FAULTLINE_Development_Plan.md Section 6 & Metric M01.
Reconciliation tolerance: max(0.005 * value, 0.0001) (i.e. within max(0.5% of value, 1 rounding unit)).
Evaluates over >= 4 lenders x >= 6 periods.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

# Ensure src/ and repo root are importable
repo_root = Path(__file__).resolve().parent.parent
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from eval.canary.generate import generate_canary_dataset
from faultline.engines.funding import analyze_funding
from faultline.engines.quality import analyze_quality


def evaluate_reconciliation(
    output_dir: Path | str = "eval/results",
    n_lenders: int = 8,
    min_periods: int = 6,
) -> dict[str, Any]:
    """Verify that computed ratios reconcile with published metrics."""
    lenders, _ = generate_canary_dataset(seed=42)

    total_checks = 0
    passed_checks = 0

    for lender in lenders[:n_lenders]:
        ts = lender["time_series"]
        periods = ts["periods"][:min_periods]
        n = len(periods)

        # 1. Quality Engine Ratios
        q_res = analyze_quality(
            periods=periods,
            gross_advances=ts["gross_advances"][:n],
            gnpa_amount=ts["gnpa_amount"][:n],
            writeoffs_period=ts["writeoffs_period"][:n],
            provisions_total=ts["provisions_total"][:n],
            lag=4,
            baseline_periods=2,
        )

        for t in range(n):
            adv = ts["gross_advances"][t]
            gnpa = ts["gnpa_amount"][t]
            if adv and gnpa and adv > 0:
                expected_gnpa_ratio = gnpa / adv
                computed = q_res.series["reported"][t]
                if computed is not None:
                    total_checks += 1
                    tol = max(0.005 * expected_gnpa_ratio, 0.0001)
                    if abs(computed - expected_gnpa_ratio) <= tol:
                        passed_checks += 1

            # PCR
            prov = ts["provisions_total"][t]
            if prov and gnpa and gnpa > 0:
                expected_pcr = prov / gnpa
                computed_pcr = q_res.series.get("pcr", [None] * n)[t]
                if computed_pcr is not None:
                    total_checks += 1
                    tol = max(0.005 * expected_pcr, 0.0001)
                    if abs(computed_pcr - expected_pcr) <= tol:
                        passed_checks += 1

        # 2. Funding Engine Ratios
        b_type = {
            "borrowings_bank": ts["borrowings_bank"][:n],
            "borrowings_bonds": ts["borrowings_bonds"][:n],
            "borrowings_deposits": ts["borrowings_deposits"][:n],
            "borrowings_securitisation": ts["borrowings_securitisation"][:n],
            "borrowings_foreign": ts["borrowings_foreign"][:n],
        }
        f_res = analyze_funding(
            periods=periods,
            total_borrowings=ts["total_borrowings"][:n],
            borrowings_by_type=b_type,
            borrowings_short_term=ts["borrowings_short_term"][:n],
            lag=4,
        )

        for t in range(n):
            tot = ts["total_borrowings"][t]
            st = ts["borrowings_short_term"][t]
            if tot and st and tot > 0:
                expected_st_share = st / tot
                computed_st = f_res.series["short_term_share"][t]
                if computed_st is not None:
                    total_checks += 1
                    tol = max(0.005 * expected_st_share, 0.0001)
                    if abs(computed_st - expected_st_share) <= tol:
                        passed_checks += 1

            for cat, vals in b_type.items():
                cat_val = vals[t]
                if tot and cat_val and tot > 0:
                    expected_share = cat_val / tot
                    comp_share = f_res.series[f"share_{cat}"][t]
                    if comp_share is not None:
                        total_checks += 1
                        tol = max(0.005 * expected_share, 0.0001)
                        if abs(comp_share - expected_share) <= tol:
                            passed_checks += 1

    rate = (passed_checks / total_checks * 100.0) if total_checks > 0 else 0.0

    result = {
        "metric_id": "M01_reconciliation_rate",
        "value": float(round(rate, 2)),
        "unit": "percent",
        "total_checks": total_checks,
        "passed_checks": passed_checks,
    }

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "m01_reconciliation.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    res = evaluate_reconciliation()
    print(f"M01 Reconciliation Rate: {res['value']}% ({res['passed_checks']}/{res['total_checks']} checks passed)")
