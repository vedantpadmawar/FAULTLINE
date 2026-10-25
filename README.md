# FAULTLINE

**ATMOS 2026: "Finding the Next Signal" — Track A (Banking, NBFCs & Financial Services)**

FAULTLINE is an autonomous forensic agent designed to analyze listed banks, NBFCs, and financial institutions. It restates key asset quality and funding figures on a fixed, like-for-like basis, flags lasting divergences ("faults") between restated figures and reported figures, and tracks year-over-year "disclosure drift" (notes dropped or definitions altered). Faults that persist and are idiosyncratic (not shared by peers) trigger rigorous hypothesis generation, red-teaming, and validation gates before publication.

---

## Core Architecture & Principles

### The Golden Rule
> **Ordinary code computes every number. LLMs plan searches, read documents, and argue. LLMs never produce or edit a number.**

### The Three Forensic Engines
1. **Like-for-Like Quality Ledger (`src/faultline/engines/quality.py`)**: Restates bad loans on a lagged loan book, adds back write-offs, audits Stage 2/3 loans, provision coverage, and credit cost. Computes exact fault anatomy decomposition ($\text{reported} \to \text{like-for-like}$ split into denominator growth vs. write-off effects) with an automatic identity closure check ($\text{tolerance } 10^{-9}$).
2. **Funding Fault Map (`src/faultline/engines/funding.py`)**: Tracks shifts across funding sources (bank borrowings, bonds, deposits, short-term debt, and off-book channels like securitization, direct assignment, and co-lending) and detects abnormal shifts relative to peers with HHI concentration index.
3. **Disclosure Drift (`src/faultline/engines/drift.py`)**: Analyzes normalized text in annual report notes year-over-year to identify dropped tables, newly added sections, and altered accounting definitions. Per Rule R9, drift alone can never publish a finding without numeric fault corroboration.

### Defense Mechanisms
- **Validation Gate (`src/faultline/verify/validation_gate.py`)**: Strict 5-point filter:
  1. Numeric engine persistence $\ge 3$.
  2. Peer-adjusted deviation $\ge 1.5$.
  3. 100% ledger grounding with matching cryptographic source SHA-256 hashes.
  4. Alternative explanations resisted: no benign explanation covers $> 60\%$ of the gap.
  5. Multiple-testing guard: penalty based on number of tested patterns.
- **Skeptic Budget (`src/faultline/agents/hypothesis.py`)**: At least 30% of search queries issued by the agent must actively seek disconfirming evidence (logged to `logs/queries.jsonl`).
- **Living Findings & Falsification Conditions (`src/faultline/report/dossier.py`)**: Findings specify exact machine-checkable conditions that would overturn the hypothesis.

---

## Hard Rules (Section 3 of Development Plan)

- **R1 (No Hardcoding)**: No company names or tickers in `src/`. Enforced by `tests/test_no_hardcoding.py`.
- **R2 (Ledger Grounding)**: Every single numeric token in any output must trace to a specific ledger ID. Enforced by `tests/test_numbers_from_ledger.py` and `eval/grounding.py` (Gate Metric M03).
- **R3 (Template Numbers)**: LLM text cannot contain computed numbers; numbers are rendered strictly via templates. Enforced by `tests/test_numbers_from_ledger.py` (Gate Metric M04).
- **R4 (Abstain on Missing Data)**: Missing data leads to an explicit `ABSTAIN` with a logged reason—never zero-filled or guessed.
- **R5 (Source Depth)**: At least 2 independent source layers (filings, investor presentations, credit rating releases) for any finding; 3 for "Strong".
- **R6 (Public Data Only)**: No paid or proprietary data sources.
- **R7 (Objective Tone)**: No investment recommendations or accusatory phrasing.
- **R8 (Offline Reproducibility)**: Every fetched source cached with its SHA-256 hash.
- **R9 (Drift Boundary)**: Disclosure drift alone can never publish a finding without numeric fault corroboration.

---

## Project Structure

```text
FAULTLINE/
├── config/
│   └── metrics.yaml          # Metrics registry and tier bounds (M01-M23)
├── src/
│   └── faultline/
│       ├── __init__.py
│       ├── main.py           # CLI entrypoint
│       ├── schemas.py        # Pydantic v2 data models
│       ├── ledger.py         # Source-grounded audit ledger
│       ├── llm.py            # Vendor-agnostic LLM client with offline deterministic mode
│       ├── fault_finder.py   # Multi-engine candidate window finder
│       ├── engines/          # Quality, Funding, and Drift engines
│       ├── ingest/           # PDF and financial table extraction with SHA-256 caching
│       ├── agents/           # Hypothesis generation and Red-Team agents
│       ├── verify/           # Validation gate, claim checker, evidence scoring
│       └── report/           # Dossier builder
├── tests/                    # Unit, regression, and hard-rule tests (66 passed)
├── eval/                     # Evaluation harness
│   ├── canary/               # 40 synthetic lenders (healthy fast growers, planted faults)
│   ├── identities.py         # M08 accounting identity closure test
│   ├── reconcile.py          # M01 ratio reconciliation test
│   ├── grounding.py          # M02 claim grounding & M03 unsupported claims test
│   ├── rerun_check.py        # M14 repeat determinism check
│   ├── run_all.py            # Master evaluation harness and scoreboard generator
│   └── results/              # Scoreboard outputs and metric measurements
├── data/                     # Cached public filings with SHA-256 hashes
├── app/                      # Streamlit interactive dashboard
├── logs/                     # JSONL logs for queries and usage
├── pyproject.toml            # Project dependencies and packaging
├── Makefile                  # Build, test, and evaluation automation
└── README.md
```

---

## Installation & Setup

### Requirements
- Python 3.10, 3.11, or 3.12
- PowerShell / Bash / Zsh

### Install Dependencies
```bash
# Editable install with development dependencies
pip install -e ".[dev]"

# Or install core packages directly
pip install pydantic pandas numpy scipy pyyaml streamlit plotly pypdf pytest
```

### Configuration & Environment Variables
The system operates seamlessly in both online and offline environments:
- **Offline Deterministic Mode (Default)**: If no API key is provided, FAULTLINE automatically operates in offline deterministic mode with zero external network dependencies, using template synthesis.
- **Online LLM Mode (Optional)**:
  ```bash
  # Google Gemini
  export GEMINI_API_KEY="your-gemini-key"

  # OpenAI (alternative)
  export OPENAI_API_KEY="your-openai-key"
  ```
  API token consumption and estimated costs are logged to `logs/usage.jsonl`.

---

## Running the Automated Test Suite

Run the full pytest suite:

```bash
# Run pytest directly
python -m pytest -v tests/

# Or via Makefile
make test
```

Current test status: **66 passed in 1.87s** across 11 test modules:
- `test_canary_generator.py`: Canary lab determinism, archetype distribution, identity checks.
- `test_drift_engine.py`: Text normalization, dropped notes, added notes, redefinitions, cosmetic changes.
- `test_funding_engine.py`: Funding mix shares, short-term share, off-book share, HHI index, peer shifts.
- `test_ingestion.py`: Table extraction, Indian numbering format parsing, SHA-256 hashing.
- `test_ledger.py`: Record appending, ID lookups, SHA-256 validation, abstain logging.
- `test_no_hardcoding.py`: Rule R1 audit (strictly 0 company names/tickers in `src/`).
- `test_numbers_from_ledger.py`: Rules R2/R3 audit (Gate Metrics M03 & M04).
- `test_pipeline.py`: Candidate finding, validation gate, red-team challenge, evidence scoring.
- `test_quality_engine.py`: Like-for-like restatement, fault anatomy identity closure ($< 10^{-9}$), robust z-scores.
- `test_schemas.py`: Pydantic v2 schema validation.
- `test_skeleton.py`: Module importability and directory layout.

---

## Running the Evaluation Suite & Scoreboard

Run the complete evaluation harness:

```bash
# Run via python
python eval/run_all.py

# Or via Makefile
make eval
```

This runs all evaluation scripts, reads `config/metrics.yaml`, verifies gate constraints, and writes `eval/results/scoreboard.md` and `eval/results/summary.json`.

### Latest Evaluation Scoreboard (`eval/results/scoreboard.md`)

| Metric ID | Value | Unit | Lower (Floor) | Target | Stretch | Tier | Gate | Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `M01_reconciliation_rate` | **100.0%** | percent | 90 | 95 | 99 | **BEYOND** | - | Verified |
| `M02_claim_grounding_rate` | **100.0%** | percent | 90 | 97 | 100 | **BEYOND** | - | Verified |
| `M03_unsupported_claims` | **0** | count | 0 | 0 | 0 | **BEYOND** | YES | **GATE PASSED** |
| `M04_llm_authored_numbers` | **0** | count | 0 | 0 | 0 | **BEYOND** | YES | **GATE PASSED** |
| `M07_abstain_correctness` | **100.0%** | percent | 95 | 100 | 100 | **BEYOND** | - | Verified |
| `M08_identity_closure_rate` | **100.0%** | percent | 90 | 97 | 100 | **BEYOND** | - | Verified |
| `M09_fault_recall` | **100.0%** | percent | 70 | 85 | 95 | **BEYOND** | - | Verified (20/20) |
| `M10_false_positive_rate` | **0.0%** | percent | 20 | 10 | 5 | **BEYOND** | - | Verified (0/20, all fast growers pass) |
| `M11_fault_start_error` | **0.0** | periods | 1.0 | 0.5 | 0 | **BEYOND** | - | Verified |
| `M12_drift_recall` | **100.0%** | percent | 80 | 90 | 98 | **BEYOND** | - | Verified |
| `M13_drift_false_alarm_rate` | **0.0%** | percent | 15 | 8 | 3 | **BEYOND** | - | Verified |
| `M14_determinism` | **100.0%** | percent | 100 | 100 | 100 | **BEYOND** | YES | **GATE PASSED** |
| `M17_redteam_rejection_band` | **50.0%** | percent | 20 | band | 80 | **TARGET** | - | In Sanity Band [20, 80] |
| `M18_disconfirming_query_share` | **50.0%** | percent | 20 | 30 | 40 | **BEYOND** | - | Verified ($\ge 30\%$ budget) |
| `M22_live_latency_seconds` | **0.26s** | seconds | 600 | 240 | 120 | **BEYOND** | - | Verified |
| `M23_cost_per_run_usd` | **$0.00** | usd | 3.0 | 1.0 | 0.5 | **BEYOND** | - | Verified |

---

## Interactive Streamlit Dashboard

FAULTLINE includes a rich interactive dashboard (`app/streamlit_app.py`) built with Streamlit and Plotly.

### How to Launch
To open the interactive dashboard in your browser:
```bash
streamlit run app/streamlit_app.py
```

### Dashboard Features
1. **Lender Selection**: Interactive selector supporting clean standard lenders, clean fast growers (negative controls), and planted fault archetypes (`denominator_masking`, `writeoff_masking`, `funding_shift`, `combined`).
2. **Executive Discovery Card**: Summary of restated findings and why it matters.
3. **Asset Quality Fault Anatomy**: Interactive Plotly chart showing reported bad-loan ratio vs. like-for-like restated ratio, with exact bar chart decomposition into denominator expansion vs. write-off effects.
4. **Funding Mix & Concentration Map**: Interactive visualization of borrowing mix shares, short-term debt ratio, and Herfindahl-Hirschman Index (HHI) concentration over time.
5. **Disclosure Drift Inspector**: Before-and-after view of removed notes, added disclosures, and redefined accounting policies with page citations.
6. **Skeptic Red-Team & Falsification Log**: Audit log of counterargument tests (seasonality, healthy fast growth, sector-wide shifts, accounting norm changes) and machine-checkable falsification triggers.
7. **Source-Grounded Audit Ledger**: Clickable inspection table displaying source URLs, page numbers, snippets, and cryptographic SHA-256 hashes.
8. **Live Evaluation Scoreboard**: Direct rendering of `eval/results/scoreboard.md` reflecting measured benchmark tiers.

---

## Audit Breakdown & Known Limitations

Per the project definition of done and integrity commitments, here is the exact operational status:

### 1. Fully Implemented Features
- **Ledger & Schemas**: Pydantic v2 schemas (`LedgerEntry`, `EngineResult`, `DriftItem`, `Candidate`, `Dossier`), SHA-256 source hashing, abstain logging.
- **Forensic Engines**: Quality engine with fault anatomy identity ($< 10^{-9}$), Funding engine with HHI and peer adjustments, Drift engine with fuzzy heading and difflib paragraph comparisons.
- **Ingestion**: PDF table parser, Indian number formatting (`1,00,000`), SHA-256 file caching.
- **Agents & Gates**: Hypothesis generator with skeptic search budget ($\ge 30\%$ disconfirming), Red-team agent with 4 alternative explanation tests, 5-point validation gate.
- **Verification**: Evidence scorer (0–100), claim checker (M03/M04 auditor), dossier compiler.
- **Interactive App**: Streamlit dashboard with 8 required views and Plotly charts.
- **Evaluation Harness**: Master runner `eval/run_all.py` producing `scoreboard.md` and `summary.json`.

### 2. Tested Features
- 66 automated tests across 11 test modules (`tests/`).
- 40 synthetic lenders evaluated across 480 periods (`eval/canary/run.py`).
- Synthetic note pairs tested against genuine and cosmetic disclosure changes (`eval/canary/drift.py`).
- Determinism verified across 3,270 pairwise numeric values (`eval/rerun_check.py`).
- Identity closure verified across 800 period checks (`eval/identities.py`).
- Ratio reconciliation verified across 384 period checks (`eval/reconcile.py`).

### 3. Failed Tests
- **0 failed tests**. All 66 pytest tests pass cleanly.

### 4. Incomplete Features (P1/P2 Tasks Deferred)
- **T18 (`eval/backcast.py`)**: Longitudinal Spearman backcast on $\ge 16$ historical cached quarters of real annual filings.
- **T19 (`eval/baselines/`)**: Blind 3-person human evaluation panel on 0–6 rubric comparing FAULTLINE vs. plain LLM and ratio screening.
- **T24 (`eval/falsification_watch.py`)**: Standing watcher daemon re-evaluating living falsification conditions as new quarterly filings are released.

### 5. Evaluation Metrics That Could Not Be Measured
The following metrics are marked **UNMEASURED** on the scoreboard because they require historical multi-year real filing series or external human review panels:
- `M05_lenders_end_to_end`: Requires live batch processing of full NIFTY 50 Track A list.
- `M06_profiler_accuracy`: Requires gold-standard company classification list (`eval/profiler_gold.yaml`).
- `M15_backcast_spearman` & `M16_backcast_lift`: Requires real multi-year quarterly filings across historical banking cycles.
- `M19_baseline_advantage` & `M20_strong_label_precision`: Requires blind external reviewer panel scores.
- `M21_window_stability`: Requires multi-window stability evaluation across variable historical horizons.
