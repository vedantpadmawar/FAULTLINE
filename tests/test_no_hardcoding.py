"""Test Rule R1: No company-specific code.

Source of truth: FAULTLINE_Development_Plan.md Section 3 (Rule R1) & Task T03.
Company names and tickers must not appear in src/.
Acceptance: fail on a deliberate violation, pass on clean code.
"""

from pathlib import Path
import re
import pytest

# Track A universe tickers and names that must NEVER appear in src/
FORBIDDEN_COMPANIES_AND_TICKERS = [
    # Top Private & PSU Banks
    "HDFCBANK",
    "HDFC BANK",
    "ICICIBANK",
    "ICICI BANK",
    "SBIN",
    "STATE BANK OF INDIA",
    "KOTAKBANK",
    "KOTAK MAHINDRA",
    "AXISBANK",
    "AXIS BANK",
    "INDUSINDBK",
    "INDUSIND BANK",
    "BANDHANBNK",
    "BANDHAN BANK",
    "FEDERALBNK",
    "FEDERAL BANK",
    "IDFCFIRSTB",
    "IDFC FIRST",
    "BANKBARODA",
    "BANK OF BARODA",
    "PNB",
    "PUNJAB NATIONAL BANK",
    "CANBK",
    "CANARA BANK",
    # Major NBFCs & Housing Finance
    "BAJFINANCE",
    "BAJAJ FINANCE",
    "BAJAJFINSV",
    "BAJAJ FINSERV",
    "CHOLAFIN",
    "CHOLAMANDALAM",
    "SHRIRAMFIN",
    "SHRIRAM FINANCE",
    "MUTHOOTFIN",
    "MUTHOOT FINANCE",
    "MANAPPURAM",
    "POONAWALLA",
    "LICHSGFIN",
    "LIC HOUSING",
    "PIRAMAL",
    "PEL",
    "L&TFH",
    "LTF",
]


def scan_source_files_for_forbidden_terms(
    src_dir: Path, forbidden_terms: list[str]
) -> list[dict]:
    """Scan all Python files in src_dir for forbidden company names and tickers.

    Uses word-boundary regex to prevent false positives on common English substrings.
    """
    violations = []
    # Compile regex patterns with word boundaries (case-insensitive for names, case-sensitive for tickers)
    compiled_patterns = [
        (term, re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE))
        for term in forbidden_terms
    ]

    for py_file in src_dir.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        lines = text.splitlines()
        for line_no, line in enumerate(lines, start=1):
            for term, pattern in compiled_patterns:
                if pattern.search(line):
                    violations.append(
                        {
                            "file": str(py_file),
                            "line": line_no,
                            "term": term,
                            "content": line.strip(),
                        }
                    )
    return violations


def test_clean_source_code_has_no_hardcoding():
    """Rule R1: Verify that src/ contains ZERO company names or tickers."""
    src_dir = Path(__file__).resolve().parent.parent / "src"
    assert src_dir.exists(), f"Source directory not found at {src_dir}"

    violations = scan_source_files_for_forbidden_terms(
        src_dir, FORBIDDEN_COMPANIES_AND_TICKERS
    )
    assert violations == [], f"Rule R1 Violation: Found hardcoded companies in src/: {violations}"


def test_hardcoding_detector_catches_deliberate_violation(tmp_path):
    """T03 Acceptance: Verifies that the scanner fails on a deliberate violation."""
    dummy_src = tmp_path / "mock_src"
    dummy_src.mkdir()
    violation_file = dummy_src / "bad_module.py"
    violation_file.write_text(
        'def get_default_spread():\n    # Special branch for HDFCBANK\n    return 0.05\n',
        encoding="utf-8",
    )

    violations = scan_source_files_for_forbidden_terms(
        dummy_src, FORBIDDEN_COMPANIES_AND_TICKERS
    )
    assert len(violations) >= 1
    assert violations[0]["term"] == "HDFCBANK"
    assert violations[0]["line"] == 2
