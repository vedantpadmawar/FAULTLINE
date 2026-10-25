"""Evidence scoring and classification logic for FAULTLINE.

Source of truth: FAULTLINE_Development_Plan.md Section 5.6 & Hard Rule R5.
Formula (0 to 100):
score = 25 * persistence_norm          # min(persistence / 6, 1.0)
      + 20 * magnitude_norm            # min(z / 6, 1.0)
      + 20 * peer_adjusted_norm        # min(peer_adjusted / 4, 1.0)
      + 20 * independent_sources_norm  # min(distinct_source_layers / 3, 1.0)
      + 15 * robustness_norm           # share of alternative explanations resisted

Classification:
- Strong:   if score >= 70 and distinct_source_layers >= 3 and persistence >= 4
- Moderate: if score >= 45 and distinct_source_layers >= 2
- Weak:     otherwise
"""

from __future__ import annotations

from typing import Literal


def compute_evidence_score(
    persistence: int,
    z_magnitude: float,
    peer_adjusted: float,
    distinct_source_layers: int,
    share_explanations_resisted: float,  # 0.0 to 1.0
) -> tuple[float, Literal["Strong", "Moderate", "Weak"], dict[str, float]]:
    """Compute the multi-factor evidence score and label.

    Returns:
        (score, label, component_breakdown)
    """
    persistence_norm = min(max(0.0, persistence) / 6.0, 1.0)
    magnitude_norm = min(max(0.0, z_magnitude) / 6.0, 1.0)
    peer_adjusted_norm = min(max(0.0, peer_adjusted) / 4.0, 1.0)
    independent_sources_norm = min(max(0, distinct_source_layers) / 3.0, 1.0)
    robustness_norm = min(max(0.0, share_explanations_resisted), 1.0)

    score = float(
        round(
            25.0 * persistence_norm
            + 20.0 * magnitude_norm
            + 20.0 * peer_adjusted_norm
            + 20.0 * independent_sources_norm
            + 15.0 * robustness_norm,
            2,
        )
    )

    # Classification label (Section 5.6 & Rule R5)
    if score >= 70.0 and distinct_source_layers >= 3 and persistence >= 4:
        label: Literal["Strong", "Moderate", "Weak"] = "Strong"
    elif score >= 45.0 and distinct_source_layers >= 2:
        label = "Moderate"
    else:
        label = "Weak"

    breakdown = {
        "persistence_component": float(round(25.0 * persistence_norm, 2)),
        "magnitude_component": float(round(20.0 * magnitude_norm, 2)),
        "peer_adjusted_component": float(round(20.0 * peer_adjusted_norm, 2)),
        "sources_component": float(round(20.0 * independent_sources_norm, 2)),
        "robustness_component": float(round(15.0 * robustness_norm, 2)),
        "total_score": score,
    }

    return score, label, breakdown
