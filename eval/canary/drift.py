"""Synthetic note pairs generator for disclosure drift canary testing.

Source of truth: FAULTLINE_Development_Plan.md Task T05 & Metrics M12, M13.
Generates this-year and last-year notes with:
- Planted drops (table removed)
- Planted additions (new note section)
- Planted redefinitions (definition materially modified, difflib ratio < 0.85)
- Planted cosmetic negatives (renumbering, formatting, simple number updates)

Stores ground-truth labels for recall and false alarm rate measurement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

# Ensure src/ is importable
src_path = str(Path(__file__).resolve().parent.parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from faultline.engines.drift import NoteSection


def generate_synthetic_note_pairs() -> tuple[list[NoteSection], list[NoteSection], list[dict[str, Any]]]:
    """Generate last-year and this-year synthetic annual report notes with ground truth."""
    # 1. Notes present in Last Year
    ly_notes = [
        # Neutral baseline note
        NoteSection(
            heading="Note 1: Corporate Information and Basis of Preparation",
            text="The financial statements are prepared under historical cost convention in accordance with Indian Accounting Standards.",
            page=42,
        ),
        # Planted DROP: Note 24 (removed in this year)
        NoteSection(
            heading="Note 24: Movement of Stage 2 Assets and Significant Increase in Credit Risk",
            text="Opening balance of Stage 2 loans was INR 4,200 cr. Additions during the year were INR 1,100 cr. Upgrades to Stage 1 stood at INR 650 cr.",
            page=84,
        ),
        # Planted REDEFINITION: Note 18 (material change in definition)
        NoteSection(
            heading="Note 18: Credit Risk and Impairment Policy for Retail Portfolios",
            text="Advances past due by 30 days are automatically classified as Stage 2. Loss given default is determined using a 5-year historical recovery cohort.",
            page=68,
        ),
        # Planted COSMETIC 1: Renumbering only
        NoteSection(
            heading="Note 14.1: Statutory Reserve Allocations",
            text="In accordance with banking regulations, twenty-five percent of statutory profit after tax is allocated to the Special Reserve.",
            page=55,
        ),
        # Planted COSMETIC 2: Reformatting and number changes
        NoteSection(
            heading="Note 21: Related Party Transactions",
            text="Transactions with key managerial personnel during the year included remuneration of INR 14.5 cr and sitting fees of INR 0.8 cr.",
            page=75,
        ),
    ]

    # 2. Notes present in This Year
    ty_notes = [
        # Neutral baseline note
        NoteSection(
            heading="Note 1: Corporate Information and Basis of Preparation",
            text="The financial statements are prepared under historical cost convention in accordance with Indian Accounting Standards.",
            page=44,
        ),
        # Note 24 is REMOVED (omitted from ty_notes)
        # Planted ADDITION: Note 38 (newly added section)
        NoteSection(
            heading="Note 38: Derivative Instruments and Credit Default Swap Exposures",
            text="The company enters into interest rate swaps and forward contracts to manage interest rate risk on offshore foreign borrowings.",
            page=112,
        ),
        # Planted REDEFINITION: Note 18 (materially altered text)
        NoteSection(
            heading="Note 18: Credit Risk and Impairment Policy for Retail Portfolios",
            text="Advances are evaluated on an aggregate portfolio statistical model. Days past due thresholds are flexible and adjusted for seasonal macroeconomic factors.",
            page=71,
        ),
        # Planted COSMETIC 1: Renumbering only (Note 14.1 -> Note 15.2, identical body)
        NoteSection(
            heading="Note 15.2: Statutory Reserve Allocations",
            text="In accordance with banking regulations, twenty-five percent of statutory profit after tax is allocated to the Special Reserve.",
            page=58,
        ),
        # Planted COSMETIC 2: Whitespace, headers, and updated numbers
        NoteSection(
            heading="Note 21: Related Party Transactions",
            text="Annual Report 2026\nPage 78 of 140\nTransactions with key managerial personnel during the year included remuneration of Rs. 16.2 crore and sitting fees of Rs. 0.95 crore.",
            page=78,
        ),
    ]

    # Ground truth labels
    ground_truth = [
        {
            "id": "GT_01",
            "type": "removed",
            "heading": "Note 24: Movement of Stage 2 Assets and Significant Increase in Credit Risk",
            "is_drift": True,
        },
        {
            "id": "GT_02",
            "type": "added",
            "heading": "Note 38: Derivative Instruments and Credit Default Swap Exposures",
            "is_drift": True,
        },
        {
            "id": "GT_03",
            "type": "redefined",
            "heading": "Note 18: Credit Risk and Impairment Policy for Retail Portfolios",
            "is_drift": True,
        },
        {
            "id": "GT_04",
            "type": "cosmetic_renumbering",
            "heading": "Statutory Reserve Allocations",
            "is_drift": False,
        },
        {
            "id": "GT_05",
            "type": "cosmetic_number_update",
            "heading": "Related Party Transactions",
            "is_drift": False,
        },
    ]

    return ly_notes, ty_notes, ground_truth


def save_synthetic_note_pairs(output_dir: Path | str) -> tuple[Path, Path]:
    """Save synthetic note pairs and ground truth to disk."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ly, ty, gt = generate_synthetic_note_pairs()

    data_payload = {
        "last_year_notes": [{"heading": n.heading, "text": n.text, "page": n.page} for n in ly],
        "this_year_notes": [{"heading": n.heading, "text": n.text, "page": n.page} for n in ty],
    }

    pairs_path = out / "note_pairs.json"
    gt_path = out / "drift_ground_truth.json"

    with open(pairs_path, "w", encoding="utf-8") as f:
        json.dump(data_payload, f, indent=2)

    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=2)

    return pairs_path, gt_path


def evaluate_drift(output_dir: Path | str = "eval/results") -> dict[str, Any]:
    """Evaluate M12_drift_recall and M13_drift_false_alarm_rate against ground truth."""
    from faultline.engines.drift import detect_disclosure_drift

    ly, ty, gt = generate_synthetic_note_pairs()
    detected_items = detect_disclosure_drift(ly, ty)

    # Ground truth tracking
    drift_gt_count = sum(1 for item in gt if item["is_drift"])
    cosmetic_gt_count = sum(1 for item in gt if not item["is_drift"])

    detected_headings = {item.heading for item in detected_items}

    # Recall: planted drifts found
    recalled_count = 0
    for item in gt:
        if item["is_drift"]:
            if any(item["heading"] in h or h in item["heading"] for h in detected_headings):
                recalled_count += 1

    # False alarms: cosmetic changes wrongly reported as drift
    false_alarm_count = 0
    for item in gt:
        if not item["is_drift"]:
            if any(item["heading"] in h or h in item["heading"] for h in detected_headings):
                false_alarm_count += 1

    m12_recall = (recalled_count / drift_gt_count * 100.0) if drift_gt_count > 0 else 0.0
    m13_far = (false_alarm_count / cosmetic_gt_count * 100.0) if cosmetic_gt_count > 0 else 0.0

    results = {
        "M12_drift_recall": {
            "value": float(round(m12_recall, 1)),
            "unit": "percent",
            "recalled": recalled_count,
            "total_drift": drift_gt_count,
        },
        "M13_drift_false_alarm_rate": {
            "value": float(round(m13_far, 1)),
            "unit": "percent",
            "false_alarms": false_alarm_count,
            "total_cosmetic": cosmetic_gt_count,
        },
    }

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "drift_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Canary Note Pairs Generator")
    parser.add_argument("--output-dir", type=str, default="eval/canary")
    args = parser.parse_args()

    p_path, g_path = save_synthetic_note_pairs(args.output_dir)
    print(f"Synthetic note pairs saved to: {p_path}")
    print(f"Drift ground truth saved to: {g_path}")

    drift_res = evaluate_drift()
    print(f"M12 Drift Recall:            {drift_res['M12_drift_recall']['value']}%")
    print(f"M13 Drift False Alarm Rate:  {drift_res['M13_drift_false_alarm_rate']['value']}%")
