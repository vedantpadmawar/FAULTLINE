# FAULTLINE: Development Plan and Metric Bounds

Audience: a coding agent (Codex, Antigravity or similar) and the human team.
Event: ATMOS 2026 "Finding the Next Signal", Track A (Banking, NBFCs & Financial Services).
Dates: Round 1 report due Mon 19 Oct 2026, 11:59 PM IST. Results Fri 23 Oct. Finals Sun 25 Oct (live run on a company the judges choose).

This is a standalone file. It does not depend on any other plan.

---

## 0. How to use this file (read first)

1. Work on ONE task ID from Section 7 at a time. Finish its acceptance test before the next task.
2. Never edit `config/metrics.yaml` to make a test pass. Fix the code.
3. Run `make test` after every task and `make eval` at the end of each day.
4. If public data is missing for a check, take the abstain path (rule R4). Never invent or zero-fill.
5. Prefer plain Python. If an agent framework is not saving time by day 2, use a sequential pipeline.

Task prompt template:

```
Implement task {ID} from FAULTLINE_Development_Plan.md.
Read Section 3 (hard rules), Section 4 (data contracts) and Section 5 (algorithms) first.
Write the code and tests that satisfy the task's acceptance line.
Do not touch config/metrics.yaml. Report which metric IDs the task moves.
```

---

## 1. What we are building

An autonomous agent that takes any listed Track A company (bank, NBFC or other financial company), restates its key quality and funding numbers on a fixed like-for-like basis, and looks for **faults**: lasting disagreements between the restated view and what is reported. It also compares this year's notes with last year's to find **disclosure drift** (tables dropped, definitions changed). Faults that persist and that peers do not share decide what the agent researches. It then tries to explain and to disprove its own explanation, and publishes only what survives a validation gate.

Design principle: **ordinary code computes every number. LLMs plan searches, read documents and argue. LLMs never produce or edit a number.**

The three engines:
1. Like-for-like quality ledger (bad loans on a lagged loan book, write-offs added back, Stage 2/3, provision coverage, credit cost).
2. Funding fault map (borrowing mix, short-term share, off-book routes such as assignment, securitisation and co-lending, concentration).
3. Disclosure drift (what changed in the notes year on year).

---

## 2. Success definition: lower bound, target, upper bound

Every metric has three bounds. Direction matters.

| Tier | Meaning | Dashboard colour |
|---|---|---|
| BELOW_FLOOR | Worse than the lower bound. Not safe to demo. | red |
| FLOOR | Meets the lower bound. Minimum acceptable. | amber |
| TARGET | Meets the target. The plan. | green |
| BEYOND | Meets or exceeds the upper bound (stretch). Best case. | gold |

Gate metrics (`gate: true`) must be met exactly. Missing a gate means the build is not shippable.

Honesty rules:
- Bounds are goals we set, not results. Never present a bound as a measured result.
- Report counts next to percentages ("17 of 20"). Samples are small.
- Every quoted number must come from `eval/results/*.json` written by `make eval`.

---

## 3. Hard rules

- R1. No company-specific code. Company names and tickers must not appear in `src/`. Test: `tests/test_no_hardcoding.py` greps `src/` against a list of names and fails on a match. (Tickers may appear in `eval/` and `data/`.)
- R2. Every number in any output has a ledger entry. Test: each numeric token in a dossier maps to a ledger id.
- R3. LLM text must not contain computed numbers. Numbers are inserted by templates from ledger values.
- R4. Missing input means ABSTAIN for that check, with a logged reason. A missing value is never zero and never guessed.
- R5. A finding needs at least 2 independent source layers (filings, management, outside). "Strong" needs 3.
- R6. Public data only. No paid or proprietary data.
- R7. No investment recommendation. A fault is never described as an accusation about the company.
- R8. Cache every fetched source with its sha256 so runs are reproducible offline.
- R9. Disclosure drift alone can never publish a finding. It can only corroborate a numeric fault or be reported as a note.

---

## 4. Data contracts (Pydantic v2, `src/faultline/schemas.py`)

```python
class LedgerEntry(BaseModel):
    id: str                      # "L0001"
    metric: str                  # see field list below
    value: float
    unit: str                    # "INR_cr", "pct", "count"
    period: str                  # "FY26-Q2" or "FY26-H1"
    period_kind: Literal["Q", "H", "Y"]
    source_url: str
    source_type: Literal["filing", "management", "outside"]
    doc_page: int | None
    snippet: str                 # exact text containing the value
    source_sha256: str
    retrieved_at: datetime

class EngineResult(BaseModel):
    engine: Literal["quality", "funding", "drift"]
    status: Literal["tested", "abstained"]
    abstain_reason: str | None
    periods: list[str]
    series: dict[str, list[float | None]]   # reported, like_for_like, peer_median, ...
    flags: list[bool]
    persistence_periods: int
    ledger_ids: list[str]

class DriftItem(BaseModel):
    kind: Literal["removed", "added", "redefined"]
    heading: str
    text_before: str | None
    text_after: str | None
    similarity: float | None
    page_before: int | None
    page_after: int | None

class Candidate(BaseModel):
    id: str
    engines: list[str]
    window: tuple[str, str]
    fault_start_period: str
    summary_numbers: dict[str, float]   # code-computed only
    drift_items: list[DriftItem]
    hypotheses: list[str]               # LLM text, no numbers
    red_team_log: list[dict]            # {test, outcome, reason}
    gate: dict
    state: Literal["explained", "unexplained", "no_reliable_finding", "rejected"]

class Dossier(BaseModel):
    company: str
    discovery: str
    why_it_matters: list[str]           # 3 to 5 bullets
    evidence: list[str]                 # ledger ids
    timeline: list[dict]
    numbers: dict[str, float]
    counterargument: str
    falsification_condition: str
    evidence_score: float               # 0 to 100
    evidence_label: Literal["Strong", "Moderate", "Weak"]
    not_tested: list[str]
```

Ledger metric names to support (a field that is not disclosed is simply absent):
`gross_advances`, `gnpa_amount`, `nnpa_amount`, `writeoffs_period`, `stage1_assets`, `stage2_assets`, `stage3_assets`, `provisions_total`, `credit_cost`, `borrowings_bank`, `borrowings_bonds`, `borrowings_deposits`, `borrowings_securitisation`, `borrowings_foreign`, `borrowings_short_term`, `assigned_loans_outstanding`, `co_lending_aum`, `total_borrowings`.

---

## 5. Core algorithms (exact enough to implement)

### 5.1 Fault anatomy (quality engine)

Let `B[t]` = gross advances at period t, `G[t]` = gross bad-loan amount, `W[t]` = write-offs summed over the last L periods (L = 4 for quarterly data, 2 for half-yearly). Let lag = L periods.

```
reported[t]       = G[t] / B[t]
like_for_like[t]  = (G[t] + W[t]) / B[t-lag]
gap[t]            = like_for_like[t] - reported[t]
denominator_part  = G[t] * (1/B[t-lag] - 1/B[t])
writeoff_part     = W[t] / B[t-lag]
identity check    : gap[t] == denominator_part + writeoff_part   (tolerance 1e-9)
```

The identity is an automatic test (metric M08). The decomposition shows how much of a gap is plain growth and how much is write-offs. If `W` is not disclosed for the periods, compute `like_for_like` without write-offs, label it "write-offs not added back", and add "write-off effect" to `not_tested`. Do not treat W as zero silently.

Fault signal: robust z-score of `gap[t]` against the lender's own history (excluding the latest 2 periods), minus the median peer z-score (peer-adjusted).

```
center = median(history); spread = 1.4826 * MAD(history)
z = (gap[t] - center) / spread
peer_adjusted = z - median(peer_z)
flag[t] = z >= 2.0 and peer_adjusted >= 1.5
persistence = longest recent run of consecutive flagged periods
```

Healthy fast growers have a naturally larger `denominator_part`. The negative controls in the canary lab (Section 6) test that they are NOT flagged. Use history of the same lender, so a lender that always grows fast is compared with its own past.

Also compute, when disclosed: Stage 2 share of loans and its change, provision coverage (provisions / gross bad loans), credit cost, and a roll-forward closure check (opening + additions - recoveries - upgrades - write-offs = closing) where the table exists.

### 5.2 Funding fault map

For each period: share of each funder type in total borrowings, `short_term_share = borrowings_short_term / total_borrowings`, `offbook_share = (assigned + securitised + co_lending_aum) / (gross_advances + that off-book amount)`, and a concentration index `HHI = sum(share_i^2)`.

```
shift_i[t]  = share_i[t] - share_i[t-lag]
flag        = |shift_i - median(peer_shift_i)| >= max(0.05, 2 * peer_MAD_i)   # 5 percentage points minimum
persistence = periods the shift keeps the same sign
```

Checks: shares sum to 100% within 0.5 points (metric M08). Funding evidence from outside (rating reports, RBI system data) is attached as `source_type="outside"` ledger entries, never as numbers the LLM invents.

### 5.3 Disclosure drift

Inputs: the notes sections of this year's and last year's annual report, as text with page numbers.

```
normalise: lowercase, collapse whitespace, strip numbering and all digits/currency, drop headers/footers
headings_removed = normalised headings in last year not in this year (fuzzy match ratio < 0.9)
headings_added   = the reverse
redefined        = matched paragraphs (heading match) whose difflib ratio < 0.85 after normalisation
cosmetic         = changes that vanish after normalisation (ignored)
```

Return `DriftItem`s. The LLM may summarise a `DriftItem` in words. It must not add numbers. Per R9, drift cannot publish alone.

### 5.4 Fault finder

Candidate windows are periods where at least 2 engines flag within +-1 period of each other, OR one numeric engine has `persistence >= 3` (drift then only corroborates). Record which engine flagged first. A sequence is a hypothesis, not proof of cause.

### 5.5 Validation gate (after red-team, before claim checking)

All must pass to publish:
1. a numeric engine has `persistence >= 3`
2. peer-adjusted deviation >= 1.5 (if fewer than 3 peers have data: label capped at Weak)
3. data quality: every input has a ledger entry with a matching file hash
4. alternative explanations: no benign explanation covers more than 60 percent of the gap (the arithmetic share is computed by code from the fault anatomy; the red-team supplies the candidate explanations)
5. multiple-testing guard: record how many (engine, window) patterns were tested for this company. Require `peer_adjusted >= 1.5 + 0.25 * log2(patterns_tested)` or downgrade to Weak.

Failure gives `rejected` (logged, shown to the user) or `no_reliable_finding`.

### 5.6 Evidence score (0 to 100)

```
score = 25*persistence_norm        # min(persistence/6, 1)
      + 20*magnitude_norm          # min(z / 6, 1)
      + 20*peer_adjusted_norm      # min(peer_adjusted / 4, 1)
      + 20*independent_sources_norm# min(distinct_source_layers / 3, 1)
      + 15*robustness_norm         # share of alternative explanations resisted
label = Strong   if score >= 70 and layers >= 3 and persistence >= 4
        Moderate if score >= 45 and layers >= 2
        Weak     otherwise
```

Business importance (size of the issue) is stored separately and never feeds the label.

### 5.7 Skeptic budget

At least 30 percent of search queries issued for a candidate must be tagged `disconfirming`. The hypothesis agent tags each query. Metric M16 measures it.

### 5.8 Living findings

`eval/falsification_watch.py` stores each dossier's falsification condition as a machine-checkable rule (metric, comparator, threshold, number of periods) and re-evaluates it when newer cached data arrives, returning `holds`, `overturned` or `not_yet_testable`.

---

## 6. Metrics registry (source of truth: `config/metrics.yaml`)

Create this file exactly. `make eval` reads it and prints the tiered scoreboard.

```yaml
# direction: higher_better | lower_better | band
# lower = worst acceptable (floor). target = plan. stretch = upper bound (beyond is best).
metrics:
  # ---- correctness ----
  - id: M01_reconciliation_rate
    what: computed ratios matching published ratios (bad-loan %, provision coverage, credit cost, shares) within max(0.5% of value, 1 rounding unit)
    unit: percent
    direction: higher_better
    lower: 90
    target: 95
    stretch: 99
    measure: eval/reconcile.py over >=4 lenders x >=6 periods

  - id: M02_claim_grounding_rate
    what: dossier claims with a live URL, page and a cached snippet containing the value
    unit: percent
    direction: higher_better
    lower: 90
    target: 97
    stretch: 100
    measure: eval/grounding.py re-opens each cached source and string-matches the snippet

  - id: M03_unsupported_claims
    what: claims with no ledger entry
    unit: count
    direction: lower_better
    lower: 0
    target: 0
    stretch: 0
    gate: true
    measure: eval/grounding.py

  - id: M04_llm_authored_numbers
    what: numeric tokens in a dossier not inserted from the ledger
    unit: count
    direction: lower_better
    lower: 0
    target: 0
    stretch: 0
    gate: true
    measure: tests/test_numbers_from_ledger.py

  - id: M08_identity_closure_rate
    what: accounting identities that hold (fault anatomy sum, funding shares sum to 100, roll-forward closes where disclosed)
    unit: percent
    direction: higher_better
    lower: 90
    target: 97
    stretch: 100
    measure: eval/identities.py

  - id: M14_determinism
    what: numeric outputs identical across two reruns from cache
    unit: percent
    direction: higher_better
    lower: 100
    target: 100
    stretch: 100
    gate: true
    measure: eval/rerun_check.py

  # ---- generality and honesty ----
  - id: M05_lenders_end_to_end
    what: companies completed with zero code changes (at least 1 bank, 1 NBFC, plus others)
    unit: count
    direction: higher_better
    lower: 4
    target: 6
    stretch: 9
    measure: eval/run_all.py (verify the NIFTY 50 Track A list at build time)

  - id: M06_profiler_accuracy
    what: bank / NBFC / other labelled correctly versus human labels
    unit: percent
    direction: higher_better
    lower: 90
    target: 100
    stretch: 100
    measure: eval/profiler_gold.yaml with >=10 companies including one insurer

  - id: M07_abstain_correctness
    what: planted missing-data cases where the system abstains instead of inventing a value
    unit: percent
    direction: higher_better
    lower: 95
    target: 100
    stretch: 100
    measure: eval/canary/missing_data.py

  # ---- discovery power (canary lab on synthetic lenders) ----
  - id: M09_fault_recall
    what: planted faults detected with the right engine and fault start within +-1 period
    unit: percent
    direction: higher_better
    lower: 70
    target: 85
    stretch: 95
    measure: eval/canary/run.py (types: denominator_masking, writeoff_masking, funding_shift, combined)

  - id: M10_false_positive_rate
    what: clean AND healthy-fast-growth lenders that wrongly publish a finding
    unit: percent
    direction: lower_better
    lower: 20
    target: 10
    stretch: 5
    measure: eval/canary/run.py (negative controls include fast but healthy growers)

  - id: M11_fault_start_error
    what: median absolute error in periods of detected fault start
    unit: periods
    direction: lower_better
    lower: 1.0
    target: 0.5
    stretch: 0
    measure: eval/canary/run.py

  - id: M12_drift_recall
    what: planted disclosure changes found (table dropped, definition changed) in synthetic note pairs
    unit: percent
    direction: higher_better
    lower: 80
    target: 90
    stretch: 98
    measure: eval/canary/drift.py

  - id: M13_drift_false_alarm_rate
    what: cosmetic edits (reformatting, renumbering, number updates) wrongly reported as drift
    unit: percent
    direction: lower_better
    lower: 15
    target: 8
    stretch: 3
    measure: eval/canary/drift.py

  # ---- backcast on real history (cached public data) ----
  - id: M15_backcast_spearman
    what: Spearman correlation between the fault signal at t (data up to t only) and the change in reported bad-loan ratio over the next 4 periods
    unit: correlation
    direction: higher_better
    lower: 0.15
    target: 0.30
    stretch: 0.50
    measure: eval/backcast.py (report n lenders, n periods, bootstrap interval)

  - id: M16_backcast_lift
    what: Spearman of the fault signal minus Spearman of the plain reported-ratio trend (baseline)
    unit: correlation_points
    direction: higher_better
    lower: 0.0
    target: 0.05
    stretch: 0.15
    measure: eval/backcast.py

  # ---- self-challenge ----
  - id: M17_redteam_rejection_band
    what: share of candidates rejected by red-team or gate. Sanity band, not a goal.
    unit: percent
    direction: band
    band: [20, 80]
    note: near 0 means the red-team is toothless; near 100 means the detector is too noisy
    measure: eval/run_all.py

  - id: M18_disconfirming_query_share
    what: share of search queries tagged as looking for evidence against the hypothesis
    unit: percent
    direction: higher_better
    lower: 20
    target: 30
    stretch: 40
    measure: logs/queries.jsonl

  # ---- versus existing approaches ----
  - id: M19_baseline_advantage
    what: mean (FAULTLINE composite minus best baseline composite) on a 0-6 blind rubric
    unit: rubric_points
    direction: higher_better
    lower: 0.5
    target: 1.5
    stretch: 2.5
    measure: eval/baselines/score.py (>=4 companies; reviewer did not build the system)

  - id: M20_strong_label_precision
    what: findings labelled Strong that the blind reviewer agrees are real and supported
    unit: percent
    direction: higher_better
    lower: 80
    target: 90
    stretch: 100
    measure: eval/baselines/score.py (report the count; expect small n)

  - id: M21_window_stability
    what: same verdict class when the history window used for the baseline changes (3 windows)
    unit: percent
    direction: higher_better
    lower: 60
    target: 80
    stretch: 95
    measure: eval/stability.py

  # ---- practicality ----
  - id: M22_live_latency_seconds
    what: seconds from company input to dossier, with cached sources
    unit: seconds
    direction: lower_better
    lower: 600
    target: 240
    stretch: 120
    measure: eval/run_all.py

  - id: M23_cost_per_run_usd
    what: model API cost per company run
    unit: usd
    direction: lower_better
    lower: 3.0
    target: 1.0
    stretch: 0.5
    measure: logs/usage.jsonl
```

Scoreboard output (`eval/results/scoreboard.md`):

```
ID                         value  lower  target  stretch  tier
M01_reconciliation_rate    96.2   90     95      99       TARGET
M10_false_positive_rate    7.0    20     10      5        TARGET
M09_fault_recall           96.0   70     85      95       BEYOND
```

---

## 7. Development plan (task list)

Priority: P0 must exist, P1 should exist, P2 only if P0 and P1 are green.
Cut order if time runs short: T24, T23, T22, T20, T19. Never cut T03, T04, T07, T09, T14, T17.

### Sun 18 and Mon 19 Oct

- **T01 (P0)** Repo skeleton, `pyproject.toml`, `Makefile` (`make test`, `make eval`, `make run COMPANY=...`), `config/metrics.yaml`, empty modules. Acceptance: `make test` runs.
- **T02 (P0)** `schemas.py` (Section 4) and `ledger.py` (append, lookup, hash check). Acceptance: round-trip test; editing a cached file breaks the hash check.
- **T03 (P0)** `tests/test_no_hardcoding.py` (R1) and `tests/test_numbers_from_ledger.py` (R2, R3). Acceptance: fail on a deliberate violation, pass on clean code. Moves M03, M04.
- **T04 (P0)** Canary lab generator `eval/canary/generate.py`: 40 synthetic lenders x 12 periods. Half clean (of those, a third are healthy FAST growers as negative controls), half with ONE planted fault: `denominator_masking` (rapid growth keeps the ratio flat while bad-loan amount rises), `writeoff_masking` (bad-loan amount flat while write-offs rise), `funding_shift` (bank share falls, off-book rises), or `combined`. Seeded noise. Ground-truth file records type and true start. Acceptance: deterministic for a seed. This is the main source of hard numbers because it needs no labelled real data.
- **T05 (P1)** Synthetic note pairs `eval/canary/drift.py`: generate this-year and last-year notes with planted drops and redefinitions, plus cosmetic edits as negatives. Ground truth stored.
- **T06 (P1)** Ingestion smoke test: fetch results and annual report notes for 2 lenders (one bank, one NBFC) from exchange or company pages, cache with hashes. Verify any library with a real call before relying on it; fallback is direct PDF/XBRL download. Acceptance: files cached.
- Round 1 report submitted by Mon 19 Oct, 6 PM IST (buffer before 11:59 PM).

### Tue 20 Oct

- **T07 (P0)** `ingest/pdf_tables.py` and `ingest/results.py`: extract gross advances, bad loans, write-offs, Stage 2/3, provisions, borrowings by type into ledger entries with page and snippet. Absent fields stay absent. Acceptance: ledger populated for 2 lenders x 6 periods where disclosed.
- **T08 (P0)** `engines/quality.py` per 5.1 including the fault anatomy identity test. Acceptance: hand-computed examples pass; identity holds to 1e-9. Moves M08, M01.
- **T09 (P0)** `eval/reconcile.py` and `eval/identities.py`. Acceptance: prints M01 and M08 with counts.

### Wed 21 Oct

- **T10 (P0)** `engines/funding.py` per 5.2. Acceptance: shares sum check passes; planted `funding_shift` is flagged on canary data.
- **T11 (P1)** `peers.py`: choose 3 to 5 peers by profile (bank vs NBFC, size band), never by hard-coded names; compute peer z and shifts. Acceptance: a sector-wide planted fault is labelled "sector story".
- **T12 (P0)** `eval/canary/run.py` producing M09, M10, M11, M07. Acceptance: M09 at or above FLOOR and M10 at or below FLOOR on canary data.

### Thu 22 Oct

- **T13 (P0)** `llm.py` adapter (`complete(prompt, model_tier)`, vendor-agnostic, usage logged) and `profile.py` (bank / NBFC / other). Acceptance: M06 on `profiler_gold.yaml`; an insurer skips loan-book engines with a logged reason.
- **T14 (P0)** `engines/drift.py` per 5.3 with `DriftItem` output; LLM only summarises. Acceptance: M12, M13 on synthetic note pairs.
- **T15 (P0)** `fault_finder.py` (5.4), `agents/hypothesis.py` (research + analysis, queries tagged `supporting` or `disconfirming`, no numbers), `agents/redteam.py` (one-offs, norm changes, seasonality, sector). Acceptance: query log tags exist; M18 computed.
- **T16 (P0)** First end-to-end run on 1 lender, then 2 more with zero code changes. Acceptance: M05 at least 3.

### Fri 23 Oct (results announced)

- **T17 (P0)** `validation_gate.py` (5.5), `verify/claim_checker.py`, `verify/score.py` (5.6), `report/dossier.py` with all eight required items. Acceptance: M02, M03, M04 green; every dossier validates against the schema; unit tests for each gate check.
- **T18 (P1)** Backcast `eval/backcast.py`: for each lender with at least 16 cached periods, compute the fault signal using only data up to t and correlate with the change in reported bad-loan ratio over the next 4 periods; compare with the plain reported-ratio trend; bootstrap interval. Acceptance: M15, M16 computed with n shown.
- **T19 (P1)** Baselines: `eval/baselines/ratio_screen.py` (flags any metric whose year-on-year change exceeds a fixed threshold) and `eval/baselines/plain_llm.py` (one prompt, same filings, no tools). Rubric `eval/baselines/rubric.md`: Real (0 to 2, numbers reconcile), Non-obvious (0 to 2, not stated by management and not visible from a headline ratio), Supported (0 to 2, at least 2 independent sources with working links). Outputs shuffled and anonymised for the blind reviewer. Acceptance: M19, M20 computable.
- **T20 (P1)** `eval/stability.py` (M21) and `eval/rerun_check.py` (M14).

### Sat 24 Oct (hardening, code freeze 6 PM IST)

- **T21 (P0)** Cache sources for 6 or more Track A companies including at least 1 bank, 1 NBFC and 1 non-lender. Acceptance: a cold run works with the network off.
- **T22 (P0)** Streamlit dashboard `app/streamlit_app.py`: discovery card with why-it-matters, key numbers, quality chart (reported vs like-for-like with the fault anatomy split into denominator and write-off parts), funding map (NetworkX or bars), drift log, timeline, clickable evidence trail, red-team log, falsification panel, evidence score with its five inputs. Acceptance: all eight required items visible; every claim clickable.
- **T23 (P2)** **Proof Panel** tab: live scoreboard (Section 6 tiers) and a side-by-side of FAULTLINE, the ratio screen and the plain LLM summary on the same company. Renders from `eval/results` only.
- **T24 (P2)** Living findings (5.8): `eval/falsification_watch.py` shows holds / overturned / not yet testable.
- **T25 (P0)** Latency and cost pass (parallel fetches, streamed agent trace), full `make eval`. Acceptance: M22 and M23 at or above FLOOR.
- **T26 (P0)** Backup demo video (under 2 minutes) and a rehearsed 10-minute talk. Freeze code at 6 PM.

### Sun 25 Oct (finals)

No new features. Morning: `make test`, one cold smoke test, confirm cached data. Demo on the company the judges choose. If the agent abstains on a check, say so and show the not-tested list: that is a feature.

---

## 8. Solo or small-team fallback

- Fewer than 3 people: use the sequential pipeline, skip T23, T24 and T18, and run T19 with only the ratio screen and plain LLM on 3 companies.
- Minimum credible demo: the like-for-like ledger with the fault anatomy split, one funding shift, the red-team log, an evidence score, an abstain case, and the canary lab numbers.

---

## 9. What makes this different (talking points, keep them true)

1. **Fault anatomy**: an exact split of the reported-versus-restated gap into a growth part and a write-off part, with an automatic identity check.
2. **Disclosure drift**: what a lender stopped or started saying is a signal that numbers cannot show.
3. **Healthy-grower negative controls**: the canary lab includes fast but healthy lenders, so the detector is tested against the most common false alarm.
4. **Backcast**: a measured check of whether the signal came before worse reported quality, using only data available at the time.
5. **Proof-carrying numbers**: URL, page, snippet and file hash for every figure, re-opened by the checker.
6. **Skeptic budget and living findings**: a minimum share of searches looks for evidence against us, and each falsification condition is re-checked against newer data.

Claim only what `make eval` actually measured.

---

## 10. Definition of done

- All gate metrics (M03, M04, M14) satisfied.
- Every non-optional metric at FLOOR or better; as many as possible at TARGET; anything at BEYOND is a bonus.
- Cold run works offline from cache for at least 6 companies including a bank, an NBFC and a non-lender.
- Dashboard shows all eight required items with clickable evidence.
- The scoreboard from `make eval` is what we quote on stage.
