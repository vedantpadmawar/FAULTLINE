"""Metric M08 evaluation: Accounting identity closure rate.

Source of truth: FAULTLINE_Development_Plan.md Section 6 & Metric M08.
Evaluates:
- Fault anatomy identity: gap == denominator_part + writeoff_part (tolerance 1e-9)
- Funding shares sum: sum(share_i) == 1.0 (tolerance 0.005)
Outputs result JSON to eval/results/m08_identities.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

# Ensure src/ and repo root are importable
repo_root = Path(__file__).resolve().parent.parent
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from eval.canary.generate import generate_canary_dataset
from faultline.engines.funding import SHARES_SUM_TOLERANCE, analyze_funding
from faultline.engines.quality import IDENTITY_TOLERANCE, analyze_quality


def evaluate_identities(output_dir: Path | str = "eval/results") -> dict[str, float]:
    """Evaluate identity closure rate across the canary lab dataset."""
    lenders, _ = generate_canary_dataset(seed=42)

    total_checks = 0
    passed_checks = 0

    for lender in lenders:
        ts = lender["time_series"]
        periods = ts["periods"]

        # 1. Quality Engine Identities (t >= 4)
        q_res = analyze_quality(
            periods=periods,
            gross_advances=ts["gross_advances"],
            gnpa_amount=ts["gnpa_amount"],
            writeoffs_period=ts["writeoffs_period"],
            lag=4,
        )

        for t in range(4, len(periods)):
            gap = q_res.series["gap"][t]
            denom = q_res.series["denominator_part"][t]
            wo = q_res.series["writeoff_part"][t]
            if gap is not None and denom is not None and wo is not None:
                total_checks += 1
                if abs(gap - (denom + wo)) < IDENTITY_TOLERANCE:
                    passed_checks += 1

        # 2. Funding Shares Sum (100% +- 0.5 points)
        borrowings_by_type = {
            "borrowings_bank": ts["borrowings_bank"],
            "borrowings_bonds": ts["borrowings_bonds"],
            "borrowings_deposits": ts["borrowings_deposits"],
            "borrowings_securitisation": ts["borrowings_securitisation"],
            "borrowings_foreign": ts["borrowings_foreign"],
        }
        f_res = analyze_funding(
            periods=periods,
            total_borrowings=ts["total_borrowings"],
            borrowings_by_type=borrowings_by_type,
            lag=4,
        )

        for t in range(len(periods)):
            cat_shares = [
                f_res.series[f"share_{c}"][t]
                for c in borrowings_by_type.keys()
                if f_res.series[f"share_{c}"][t] is not None
            ]
            if len(cat_shares) == len(borrowings_by_type):
                total_checks += 1
                if abs(sum(cat_shares) - 1.0) <= SHARES_SUM_TOLERANCE:
                    passed_checks += 1

    rate = (passed_checks / total_checks * 100.0) if total_checks > 0 else 0.0

    result = {
        "metric_id": "M08_identity_closure_rate",
        "value": float(round(rate, 2)),
        "unit": "percent",
        "total_checks": total_checks,
        "passed_checks": passed_checks,
    }

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "m08_identities.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    res = evaluate_identities()
    print(f"M08 Identity Closure Rate: {res['value']}% ({res['passed_checks']}/{res['total_checks']})")
