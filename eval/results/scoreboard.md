# FAULTLINE Evaluation Scoreboard

Generated: 2026-10-24 20:04:21 UTC

| Metric ID | Value | Unit | Lower (Floor) | Target | Stretch | Tier | Gate |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `M01_reconciliation_rate` | **100.0** | percent | 90 | 95 | 99 | **BEYOND** | - |
| `M02_claim_grounding_rate` | **100.0** | percent | 90 | 97 | 100 | **BEYOND** | - |
| `M03_unsupported_claims` | **0** | count | 0 | 0 | 0 | **BEYOND** | YES |
| `M04_llm_authored_numbers` | **0** | count | 0 | 0 | 0 | **BEYOND** | YES |
| `M08_identity_closure_rate` | **100.0** | percent | 90 | 97 | 100 | **BEYOND** | - |
| `M14_determinism` | **100.0** | percent | 100 | 100 | 100 | **BEYOND** | YES |
| `M05_lenders_end_to_end` | *Unmeasured* | count | 4 | 6 | 9 | **UNMEASURED** | - |
| `M06_profiler_accuracy` | *Unmeasured* | percent | 90 | 100 | 100 | **UNMEASURED** | - |
| `M07_abstain_correctness` | **100.0** | percent | 95 | 100 | 100 | **BEYOND** | - |
| `M09_fault_recall` | **100.0** | percent | 70 | 85 | 95 | **BEYOND** | - |
| `M10_false_positive_rate` | **0.0** | percent | 20 | 10 | 5 | **BEYOND** | - |
| `M11_fault_start_error` | **0.0** | periods | 1.0 | 0.5 | 0 | **BEYOND** | - |
| `M12_drift_recall` | **100.0** | percent | 80 | 90 | 98 | **BEYOND** | - |
| `M13_drift_false_alarm_rate` | **0.0** | percent | 15 | 8 | 3 | **BEYOND** | - |
| `M15_backcast_spearman` | *Unmeasured* | correlation | 0.15 | 0.3 | 0.5 | **UNMEASURED** | - |
| `M16_backcast_lift` | *Unmeasured* | correlation_points | 0.0 | 0.05 | 0.15 | **UNMEASURED** | - |
| `M17_redteam_rejection_band` | **50.0** | percent | 20 | Band | 80 | **TARGET** | - |
| `M18_disconfirming_query_share` | **50.0** | percent | 20 | 30 | 40 | **BEYOND** | - |
| `M19_baseline_advantage` | *Unmeasured* | rubric_points | 0.5 | 1.5 | 2.5 | **UNMEASURED** | - |
| `M20_strong_label_precision` | *Unmeasured* | percent | 80 | 90 | 100 | **UNMEASURED** | - |
| `M21_window_stability` | *Unmeasured* | percent | 60 | 80 | 95 | **UNMEASURED** | - |
| `M22_live_latency_seconds` | **0.29** | seconds | 600 | 240 | 120 | **BEYOND** | - |
| `M23_cost_per_run_usd` | **0.0** | usd | 3.0 | 1.0 | 0.5 | **BEYOND** | - |

## Summary
- Total Metrics: 23
- Measured: 16
- Beyond (Stretch): 15
- At Target: 1
- At Floor: 0
- Below Floor: 0
- Unmeasured: 7

> [!IMPORTANT]
> Per Section 3 and Section 10 of FAULTLINE Development Plan, all gate metrics (M03, M04, M14) are strictly satisfied.
> No unmeasured or historical metrics are marked as passing without cached empirical evaluation data.
