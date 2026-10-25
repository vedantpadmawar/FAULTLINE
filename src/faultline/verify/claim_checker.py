"""Claim and numeric grounding verification.

Source of truth: FAULTLINE_Development_Plan.md Section 3 (Rules R2 & R3) & Metrics M03, M04.
Verifies that:
- Every number in a dossier maps directly to a verified LedgerEntry ID (M03).
- LLM prose does not author arbitrary ungrounded numbers (M04).
"""

from __future__ import annotations

import re
from typing import Any

from faultline.ledger import FinancialLedger
from faultline.schemas import Dossier


# Regex to extract numeric candidates from prose (e.g., "12.5", "1,500.0", "3.4%", "125000")
NUMERIC_TOKEN_REGEX = re.compile(r"(?<![A-Za-z_])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?")


def parse_numeric_token(token: str) -> float:
    """Parse numeric token into float, stripping commas and percentage signs."""
    clean = token.replace(",", "").rstrip("%")
    return float(clean)


def is_allowable_prose_number(num: float, token: str) -> bool:
    """Allow standard non-financial prose numbering such as bullet ordinals or small counts."""
    # Allow small positive integers used for list items (1 to 10) if they don't have decimals or percent
    if "." not in token and "%" not in token:
        int_val = int(num)
        if 1 <= int_val <= 10 and float(int_val) == num:
            return True
    return False


def verify_dossier_grounding(
    dossier: Dossier, ledger: FinancialLedger
) -> dict[str, Any]:
    """Verify that all numbers in a Dossier are strictly grounded in the ledger.

    Returns dict with:
        is_valid: bool
        unsupported_claims: list[str] (Metric M03)
        llm_authored_numbers: list[str] (Metric M04)
    """
    unsupported_claims: list[str] = []
    llm_authored_numbers: list[str] = []

    # 1. Gather all authorized values from the evidence ledger entries
    authorized_values: set[float] = set()
    authorized_entry_ids: set[str] = set()

    for entry_id in dossier.evidence:
        entry = ledger.get(entry_id)
        if entry is None:
            unsupported_claims.append(f"Evidence ID '{entry_id}' not found in ledger")
        else:
            authorized_entry_ids.add(entry.id)
            authorized_values.add(round(entry.value, 4))
            # Also allow percentage equivalent (e.g., 0.035 -> 3.5)
            authorized_values.add(round(entry.value * 100, 4))

    # 2. Check dossier.numbers mapping
    for name, val in dossier.numbers.items():
        val_rounded = round(val, 4)
        if val_rounded not in authorized_values:
            unsupported_claims.append(
                f"Numeric claim '{name}: {val}' does not match any entry in evidence ledger ({dossier.evidence})"
            )

    # 3. Check LLM text fields for ungrounded numbers (Rule R3 / Metric M04)
    prose_sections = [
        ("discovery", dossier.discovery),
        ("counterargument", dossier.counterargument),
        ("falsification_condition", dossier.falsification_condition),
    ]
    for bullet in dossier.why_it_matters:
        prose_sections.append(("why_it_matters", bullet))

    for field_name, text in prose_sections:
        matches = NUMERIC_TOKEN_REGEX.findall(text)
        for token in matches:
            try:
                num = parse_numeric_token(token)
            except ValueError:
                continue

            # Check if this number is an allowable structural integer (e.g. bullet index)
            if is_allowable_prose_number(num, token):
                continue

            # Verify if this number exists in the authorized ledger values
            num_rounded = round(num, 4)
            if num_rounded not in authorized_values:
                llm_authored_numbers.append(
                    f"Ungrounded number '{token}' found in {field_name}: '{text}'"
                )

    is_valid = len(unsupported_claims) == 0 and len(llm_authored_numbers) == 0
    return {
        "is_valid": is_valid,
        "unsupported_claims": unsupported_claims,
        "llm_authored_numbers": llm_authored_numbers,
    }
