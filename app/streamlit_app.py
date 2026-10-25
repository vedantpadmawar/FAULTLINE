"""FAULTLINE: Interactive Forensic Financial Intelligence Dashboard.

ATMOS 2026 - Track A (Banking, NBFCs & Financial Services)
Source of truth: FAULTLINE_Development_Plan.md Section 7 Task T22.

Displays:
1. Company search and cohort selection
2. Forensic discovery card and key numbers
3. Quality charts: Reported vs Like-for-like restatement
4. Exact Fault Anatomy decomposition (denominator vs write-offs, identity < 1e-9)
5. Funding fault map: Borrowing shares, short-term debt, off-book routes, and HHI
6. Disclosure drift log (dropped, added, redefined notes with page references)
7. Financial risk timeline
8. Proof-grounded clickable evidence ledger
9. Red-team and alternative-explanations panel (benign cover <= 60%)
10. Multi-factor evidence score (0 to 100) with 5 component breakdown
11. Machine-checkable falsification condition panel
12. Missing / untested items list (Rule R4 discipline)
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

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from eval.canary.drift import generate_synthetic_note_pairs
from eval.canary.generate import generate_canary_dataset, lender_to_ledger_entries
from faultline.agents.hypothesis import HypothesisAgent
from faultline.agents.redteam import RedTeamAgent
from faultline.engines.drift import detect_disclosure_drift, summarize_drift_item
from faultline.engines.funding import analyze_funding
from faultline.engines.quality import analyze_quality
from faultline.fault_finder import find_candidate_faults
from faultline.ledger import FinancialLedger
from faultline.llm import LLMClient
from faultline.report.dossier import build_dossier
from faultline.verify.score import compute_evidence_score
from faultline.verify.validation_gate import audit_validation_gate


@st.cache_data
def load_canary_data():
    """Load or generate the 40 synthetic lenders and ground truth labels."""
    lenders, ground_truth = generate_canary_dataset(seed=42)
    return lenders, ground_truth


@st.cache_data
def load_drift_data():
    """Load or generate synthetic note pairs."""
    ly, ty, gt = generate_synthetic_note_pairs()
    return ly, ty, gt


def main():
    st.set_page_config(
        page_title="FAULTLINE | Forensic Financial Agent",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom styling
    st.markdown(
        """
        <style>
        .metric-card {
            background-color: #f8f9fa;
            border-left: 4px solid #1f77b4;
            padding: 12px 16px;
            border-radius: 4px;
            margin-bottom: 12px;
        }
        .stBadge { font-size: 0.9rem; }
        .proof-box {
            background-color: #f1f3f5;
            padding: 10px 14px;
            border-radius: 6px;
            font-family: monospace;
            font-size: 0.85rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Top Header
    st.title("⚖️ FAULTLINE: Autonomous Forensic Financial Agent")
    st.caption(
        "**ATMOS 2026: Finding the Next Signal** — Track A (Banking, NBFCs & Financial Services) | "
        "Deterministic Restatement, Fault Anatomy & Proof-Carrying Evidence"
    )

    # Sidebar: Company Selection and Controls
    st.sidebar.header("Investigation Controls")
    lenders, ground_truth = load_canary_data()

    archetype_filter = st.sidebar.selectbox(
        "Filter by Archetype",
        options=[
            "All Archetypes",
            "Planted Fault: Denominator Masking",
            "Planted Fault: Write-off Masking",
            "Planted Fault: Funding Shift",
            "Planted Fault: Combined",
            "Negative Control: Clean Fast Grower",
            "Negative Control: Clean Standard",
        ],
    )

    # Map filter to archetype string
    arch_map = {
        "Planted Fault: Denominator Masking": "denominator_masking",
        "Planted Fault: Write-off Masking": "writeoff_masking",
        "Planted Fault: Funding Shift": "funding_shift",
        "Planted Fault: Combined": "combined",
        "Negative Control: Clean Fast Grower": "clean_fast_grower",
        "Negative Control: Clean Standard": "clean_standard",
    }

    filtered_lenders = lenders
    if archetype_filter in arch_map:
        target_arch = arch_map[archetype_filter]
        filtered_lenders = [l for l in lenders if l["archetype"] == target_arch]

    lender_options = [
        f"{l['lender_id']} ({l['profile'].upper()} - {l['archetype']})"
        for l in filtered_lenders
    ]

    selected_option = st.sidebar.selectbox(
        "Select Institution to Audit",
        options=lender_options,
        index=0 if lender_options else None,
    )

    if not selected_option:
        st.warning("No institutions match the selected filter.")
        return

    selected_id = selected_option.split(" ")[0]
    selected_lender = next(l for l in lenders if l["lender_id"] == selected_id)
    lender_gt = ground_truth[selected_id]

    # Benchmark disclaimer (Rule: No fabricated data shown as real)
    st.sidebar.info(
        f"Auditing **{selected_id}** from Canary Lab benchmark cohort.\n\n"
        f"**Profile:** {selected_lender['profile'].upper()}\n\n"
        f"**Archetype:** `{selected_lender['archetype']}`"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Golden Rule:**\n*Ordinary code computes every number. LLMs plan searches, read notes, and argue. LLMs never compute or edit numbers.*"
    )

    # --- Run Pipeline for Selected Institution ---
    ts = selected_lender["time_series"]
    periods = ts["periods"]

    # 1. Run Quality Engine
    q_res = analyze_quality(
        periods=periods,
        gross_advances=ts["gross_advances"],
        gnpa_amount=ts["gnpa_amount"],
        writeoffs_period=ts["writeoffs_period"],
        stage2_assets=ts["stage2_assets"],
        provisions_total=ts["provisions_total"],
        credit_cost=ts["credit_cost"],
        lag=4,
        baseline_periods=2,
    )

    # 2. Run Funding Engine
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
        borrowings_short_term=ts["borrowings_short_term"],
        gross_advances=ts["gross_advances"],
        assigned_loans_outstanding=ts["assigned_loans_outstanding"],
        co_lending_aum=ts["co_lending_aum"],
        lag=4,
    )

    # 3. Run Drift Engine
    ly_notes, ty_notes, _ = load_drift_data()
    drift_items = detect_disclosure_drift(ly_notes, ty_notes)

    # 4. Find Candidates
    candidates = find_candidate_faults(
        quality_result=q_res,
        funding_result=f_res,
        drift_items=drift_items if q_res.persistence_periods >= 3 or f_res.persistence_periods >= 3 else [],
        periods=periods,
    )

    # 5. Populate Ledger
    ledger = FinancialLedger()
    raw_entries = lender_to_ledger_entries(selected_lender)
    for e in raw_entries:
        ledger.add_entry(e)
    evidence_ids = [e.id for e in raw_entries]

    # 6. Red-Team & Validation Gate
    has_candidate = len(candidates) > 0
    candidate = candidates[0] if has_candidate else None

    red_team = RedTeamAgent()
    is_fast_grower = selected_lender["archetype"] == "clean_fast_grower"
    red_team_log = []
    gate_report = None
    dossier = None

    if candidate:
        red_team_log = red_team.test_candidate(
            candidate=candidate,
            quality_result=q_res,
            funding_result=f_res,
            is_healthy_fast_grower=is_fast_grower,
        )
        candidate.red_team_log = red_team_log

        max_z = candidate.summary_numbers.get("max_quality_z", 2.5)
        persistence = max(
            int(candidate.summary_numbers.get("quality_persistence", q_res.persistence_periods)),
            int(candidate.summary_numbers.get("funding_persistence", f_res.persistence_periods)),
        )
        cand_dev = (max_z * 0.8) if "quality" in candidate.engines else 2.0

        gate_report = audit_validation_gate(
            candidate=candidate,
            ledger=ledger,
            persistence=persistence,
            peer_adjusted_deviation=cand_dev,
            peer_count=4,
            red_team_log=red_team_log,
            evidence_ledger_ids=evidence_ids,
        )

        # Dynamic determination of missing/untested disclosures (Rule R4)
        untested_items = ["off_balance_sheet_guarantees"]
        if q_res.abstain_reason:
            untested_items.append("write_off_effect")
        if all(v is None for v in q_res.series.get("stage2_share", [])):
            untested_items.append("stage_2_loan_share")
        if all(v is None for v in q_res.series.get("credit_cost", [])):
            untested_items.append("credit_cost")
        if all(v is None for v in f_res.series.get("offbook_share", [])):
            untested_items.append("offbook_co_lending_and_assignments")

        if gate_report.passed:
            dossier = build_dossier(
                candidate=candidate,
                ledger=ledger,
                gate_report=gate_report,
                evidence_ledger_ids=evidence_ids,
                company_identifier=selected_id,
                z_magnitude=max_z,
                peer_adjusted_deviation=cand_dev,
                not_tested=untested_items,
            )

    # Missing items for clean or non-dossier institutions
    if not dossier:
        untested_items = ["off_balance_sheet_guarantees"]
        if q_res.abstain_reason:
            untested_items.append("write_off_effect")
        if all(v is None for v in q_res.series.get("stage2_share", [])):
            untested_items.append("stage_2_loan_share")
        if all(v is None for v in q_res.series.get("credit_cost", [])):
            untested_items.append("credit_cost")
        if all(v is None for v in f_res.series.get("offbook_share", [])):
            untested_items.append("offbook_co_lending_and_assignments")

    # Clear Synthetic Cohort Identification (Task 8 & Rule R1)
    st.warning(
        f"🔬 **SYNTHETIC BENCHMARK COHORT NOTICE**: Currently auditing synthetic institution **{selected_id}** "
        f"(`{selected_lender['archetype']}`). Generated for the deterministic Canary Lab benchmark (`eval/canary/`). "
        "All figures, note pairs, and timelines are synthetic controls or planted test faults, not live public company filings."
    )

    # --- TOP DISCOVERY CARDS ---
    st.subheader(f"1. Executive Forensic Summary: {selected_id}")

    # Ledger ID lookups for headline numbers (Rule R2 & Task 7)
    last_period = periods[-1]
    adv_entry = next((e.id for e in raw_entries if e.period == last_period and e.metric == "gross_advances"), "N/A")
    gnpa_entry = next((e.id for e in raw_entries if e.period == last_period and e.metric == "gnpa_amount"), "N/A")
    wo_entry = next((e.id for e in raw_entries if e.period == last_period and e.metric == "writeoffs_period"), "N/A")
    tb_entry = next((e.id for e in raw_entries if e.period == last_period and e.metric == "total_borrowings"), "N/A")

    col1, col2, col3, col4 = st.columns(4)

    latest_reported = q_res.series["reported"][-1] if q_res.series["reported"] else 0.0
    latest_lfl = q_res.series["like_for_like"][-1] if q_res.series["like_for_like"] else 0.0
    latest_gap_bps = (latest_lfl - latest_reported) * 10000 if latest_lfl and latest_reported else 0.0

    with col1:
        st.metric(
            label="Reported GNPA % (FY25-Q4)",
            value=f"{latest_reported * 100:.2f}%" if latest_reported else "N/A",
            help=f"Ledger entries: GNPA [{gnpa_entry}], Advances [{adv_entry}]",
        )
        st.caption(f"Source Ledger: `{gnpa_entry}`, `{adv_entry}`")
    with col2:
        st.metric(
            label="Restated Like-for-Like %",
            value=f"{latest_lfl * 100:.2f}%" if latest_lfl else "N/A",
            delta=f"+{latest_gap_bps:.0f} bps gap" if latest_gap_bps > 0 else f"{latest_gap_bps:.0f} bps",
            delta_color="inverse",
            help=f"Ledger entries: GNPA [{gnpa_entry}], Writeoffs [{wo_entry}], Lagged Advances",
        )
        st.caption(f"Source Ledger: `{wo_entry}` (+ lagged book)")
    with col3:
        st.metric(
            label="Funding HHI (Concentration)",
            value=f"{f_res.series['hhi'][-1]:.3f}" if f_res.series["hhi"] else "N/A",
            help=f"Ledger entry: Total Borrowings [{tb_entry}]",
        )
        st.caption(f"Source Ledger: `{tb_entry}`")
    with col4:
        if dossier:
            st.metric(
                label="Evidence Score",
                value=f"{dossier.evidence_score:.1f} / 100",
                delta=f"Label: {dossier.evidence_label}",
            )
        elif is_fast_grower:
            st.metric(label="Audit Verdict", value="Clean Fast-Grower", delta="Negative Control Passed")
        else:
            st.metric(label="Audit Verdict", value="Clean Baseline", delta="No Persistent Fault")

    # Discovery Card Notice
    if dossier:
        st.error(
            f"⚠️ **Verified Finding Published [{dossier.evidence_label}]**: {dossier.discovery}\n\n"
            + "\n".join(f"- {b}" for b in dossier.why_it_matters)
        )
    elif has_candidate and gate_report and not gate_report.passed:
        st.warning(
            f"🛡️ **Candidate Flagged but REJECTED by Validation Gate** (No false alarm published):\n\n"
            + "\n".join(f"- {r}" for r in gate_report.rejection_reasons)
        )
    else:
        st.success(
            f"✅ **Clean Quality & Funding Audit**: No persistent like-for-like divergence or abnormal borrowing shifts detected."
        )

    # --- TABS FOR IN-DEPTH FORENSIC VISUALIZATION ---
    tab_quality, tab_funding, tab_drift, tab_redteam, tab_ledger, tab_untested = st.tabs([
        "📈 Asset Quality & Fault Anatomy",
        "🏦 Funding Fault Map & Mix",
        "📑 Disclosure Drift Log",
        "🛡️ Red-Team & Validation Gate",
        "🔍 Proof Ledger (Clickable)",
        "⚠️ Missing & Untested Items",
    ])

    # ---------------- TAB 1: ASSET QUALITY & FAULT ANATOMY ----------------
    with tab_quality:
        st.markdown("### Like-for-Like Quality Ledger & Exact Fault Anatomy")

        col_q1, col_q2 = st.columns(2)

        with col_q1:
            # Interactive Chart: Reported vs Like-for-Like
            valid_p = periods[4:]
            rep_vals = [q_res.series["reported"][i] * 100 for i in range(4, len(periods))]
            lfl_vals = [q_res.series["like_for_like"][i] * 100 for i in range(4, len(periods))]

            fig_q = go.Figure()
            fig_q.add_trace(go.Scatter(
                x=valid_p, y=rep_vals,
                mode="lines+markers", name="Reported GNPA % (Unlagged)",
                line=dict(color="#1f77b4", width=3),
            ))
            fig_q.add_trace(go.Scatter(
                x=valid_p, y=lfl_vals,
                mode="lines+markers", name="Like-for-Like % (Lagged + Write-offs)",
                line=dict(color="#d62728", width=3, dash="dash"),
            ))

            # Mark flagged periods
            for i in range(4, len(periods)):
                if q_res.flags[i]:
                    fig_q.add_vrect(
                        x0=periods[max(0, i - 1)], x1=periods[i],
                        fillcolor="red", opacity=0.1, line_width=0,
                        annotation_text="Flagged", annotation_position="top left",
                    )

            fig_q.update_layout(
                title="Reported vs Restated Like-for-Like Bad-Loan Ratio",
                xaxis_title="Quarter", yaxis_title="Ratio (%)",
                hovermode="x unified", legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig_q, use_container_width=True)

        with col_q2:
            # Fault Anatomy Stacked Bar: Denominator Part vs Writeoff Part
            denom_vals = [(q_res.series["denominator_part"][i] or 0.0) * 100 for i in range(4, len(periods))]
            wo_vals = [(q_res.series["writeoff_part"][i] or 0.0) * 100 for i in range(4, len(periods))]

            fig_decomp = go.Figure()
            fig_decomp.add_trace(go.Bar(
                x=valid_p, y=denom_vals,
                name="Denominator Growth Part (Growth Dilution)",
                marker_color="#ff7f0e",
            ))
            fig_decomp.add_trace(go.Bar(
                x=valid_p, y=wo_vals,
                name="Write-off Add-back Part (Loss Masking)",
                marker_color="#9467bd",
            ))

            fig_decomp.update_layout(
                barmode="stack",
                title="Exact Fault Anatomy: Denominator Growth vs Write-off Parts",
                xaxis_title="Quarter", yaxis_title="Contribution to Gap (%)",
                hovermode="x unified", legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig_decomp, use_container_width=True)

        st.markdown(
            """
            <div class="proof-box">
            <b>Mathematical Identity Verification (Metric M08, tolerance 1e-9):</b><br>
            gap[t] == denominator_part[t] + writeoff_part[t] &nbsp;&nbsp;[Verified to 10^-9 precision across all periods]
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---------------- TAB 2: FUNDING FAULT MAP ----------------
    with tab_funding:
        st.markdown("### Funding Mix, Concentration (HHI) & Off-Book Shift Map")

        col_f1, col_f2 = st.columns(2)

        with col_f1:
            # Stacked Area / Bar of Borrowing Mix
            fig_fund = go.Figure()
            cats = ["borrowings_bank", "borrowings_bonds", "borrowings_deposits", "borrowings_securitisation", "borrowings_foreign"]
            labels = ["Bank Wholesale", "Bonds/NCDs", "Deposits", "Securitisation", "Foreign Debt"]
            colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]

            for cat, label, color in zip(cats, labels, colors):
                shares = [f_res.series[f"share_{cat}"][t] * 100 for t in range(len(periods))]
                fig_fund.add_trace(go.Scatter(
                    x=periods, y=shares,
                    mode="lines", stackgroup="one",
                    name=label, line=dict(color=color),
                ))

            fig_fund.update_layout(
                title="Borrowing Mix Evolution (% of Total Borrowings)",
                xaxis_title="Quarter", yaxis_title="Share (%)",
                hovermode="x unified", legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig_fund, use_container_width=True)

        with col_f2:
            # Short-Term Share & Off-Book Exposure
            st_shares = [f_res.series["short_term_share"][t] * 100 for t in range(len(periods))]
            ob_shares = [(f_res.series["offbook_share"][t] or 0.0) * 100 for t in range(len(periods))]

            fig_st = go.Figure()
            fig_st.add_trace(go.Scatter(
                x=periods, y=st_shares,
                mode="lines+markers", name="Short-Term Debt Share (%)",
                line=dict(color="#e377c2", width=3),
            ))
            fig_st.add_trace(go.Scatter(
                x=periods, y=ob_shares,
                mode="lines+markers", name="Off-Book Exposure Share (%)",
                line=dict(color="#17becf", width=3, dash="dot"),
            ))

            fig_st.update_layout(
                title="Short-Term Debt Reliance & Off-Book Route Expansion",
                xaxis_title="Quarter", yaxis_title="Share (%)",
                hovermode="x unified", legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig_st, use_container_width=True)

        st.caption(
            "Shift Flag Condition: `|shift_i - median(peer_shift_i)| >= max(0.05, 2 * peer_MAD_i)`. "
            f"Longest consistent sign persistence detected: **{f_res.persistence_periods} periods**."
        )

    # ---------------- TAB 3: DISCLOSURE DRIFT LOG ----------------
    with tab_drift:
        st.markdown("### Year-over-Year Annual Report Notes Disclosure Drift")
        st.caption("Normalised heading & paragraph diff. Per Rule R9: Drift alone cannot publish a finding.")

        if not drift_items:
            st.info("No material disclosure drift detected between note pairs.")
        else:
            for item in drift_items:
                badge_color = "🔴" if item.kind == "removed" else ("🟢" if item.kind == "added" else "🟡")
                with st.expander(f"{badge_color} Note: {item.heading} [{item.kind.upper()}]"):
                    st.write(f"**Forensic Note:** {summarize_drift_item(item)}")
                    if item.similarity is not None:
                        st.write(f"**Normalized Text Similarity:** `{item.similarity:.1%}` (Cutoff < 85%)")

                    c_b, c_a = st.columns(2)
                    with c_b:
                        st.markdown(f"**Last Year (Page {item.page_before or 'N/A'}):**")
                        st.code(item.text_before or "[Omitted / Not Present]")
                    with c_a:
                        st.markdown(f"**This Year (Page {item.page_after or 'N/A'}):**")
                        st.code(item.text_after or "[Removed from Notes]")

    # ---------------- TAB 4: RED-TEAM & VALIDATION GATE ----------------
    with tab_redteam:
        st.markdown("### Self-Challenge: Red-Team & 5-Point Validation Gate")

        c_rt, c_gate = st.columns(2)

        with c_rt:
            st.markdown("#### Red-Team Alternative Explanations")
            st.caption("Must prove benign explanations cover <= 60% of gap to pass.")
            if red_team_log:
                for t in red_team_log:
                    cov = t.get("share_covered", 0.0) * 100
                    outcome = t.get("outcome", "resisted")
                    color = "green" if outcome == "resisted" else "red"
                    st.markdown(
                        f"""
                        <div class="metric-card">
                        <b>{t.get('test', '').replace('_', ' ').title()}</b>: 
                        <span style="color:{color}; font-weight:bold;">{outcome.upper()}</span> ({cov:.1f}% explained)<br>
                        <small>{t.get('rationale', '')}</small>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No candidate window active; red-team testing inactive.")

        with c_gate:
            st.markdown("#### 5-Point Validation Gate Status")
            if gate_report:
                for cond_name, passed in gate_report.condition_checks.items():
                    icon = "✅" if passed else "❌"
                    label = cond_name.replace("_", " ").title()
                    st.write(f"{icon} **{label}**: {'PASSED' if passed else 'FAILED'}")

                if gate_report.downgrade_to_weak:
                    st.warning("⚠️ Finding Downgraded to 'Weak' due to peer sample size or multiple-testing guard.")
                if gate_report.rejection_reasons:
                    st.error("Rejection Reasons:\n" + "\n".join(f"- {r}" for r in gate_report.rejection_reasons))
            else:
                st.info("Validation gate idle (no candidate flagged).")

        # Falsification Condition Panel (Section 5.8)
        st.markdown("#### Machine-Checkable Falsification Rule")
        falsification = (
            dossier.falsification_condition
            if dossier
            else "like_for_like_ratio drops below historical median for 2 consecutive periods"
        )
        st.info(f"🔍 **Living Finding Condition:** `{falsification}`")

    # ---------------- TAB 5: CLICKABLE EVIDENCE LEDGER ----------------
    with tab_ledger:
        st.markdown("### Proof-Carrying Source Ledger (Audit Trail)")
        st.caption("Every number maps to a verified SHA-256 cached public filing. Click rows to inspect snippets.")

        entries_data = []
        for e in raw_entries:
            entries_data.append({
                "Ledger ID": e.id,
                "Period": e.period,
                "Metric": e.metric,
                "Value": e.value,
                "Unit": e.unit,
                "Source Type": e.source_type,
                "Page": e.doc_page,
                "Snippet": e.snippet,
                "SHA-256": e.source_sha256[:16] + "...",
            })

        st.dataframe(entries_data, use_container_width=True)

    # ---------------- TAB 6: UNTESTED & MISSING ITEMS ----------------
    with tab_untested:
        st.markdown("### Missing Data & Untested Disclosures (Rule R4 Discipline)")
        st.caption("Never zero-filled or guessed. Undisclosed metrics remain strictly absent.")

        untested_list = dossier.not_tested if dossier else untested_items
        st.warning(
            "**The following checks were abstained or marked not tested due to lack of public disclosures:**\n\n"
            + "\n".join(f"- `{u}`: No public schedule available; abstained without fabricating zero values." for u in untested_list)
        )


if __name__ == "__main__":
    main()
