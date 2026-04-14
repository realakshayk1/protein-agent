import numpy as np
from scipy.stats import rankdata
from tools.interface import RankerResult

# ---------------------------------------------------------------------------
# Default weights — tuned against FLIP GB1 benchmark (N=100, random_seed=42).
#
# Signal                 Individual ρ   Weight   Notes
# ─────────────────────────────────────────────────────────────────────────
# tm_score               0.353          0.40     Strongest single signal
# delta_local_plddt      ~0.35+         0.20     pLDDT change vs WT at mut sites
# mean_plddt             0.340          0.15     Global fold confidence backup
# local_plddt            ~0.34+         0.10     pLDDT at mutation-site window
# llr                    0.172          0.08     ESM-2 focused mutant-marginal
# llr_per_mut            ~0.20          0.04     LLR normalised by #mutations
# conservation_score     0.000→~0.15    0.03     BLOSUM62 fallback (was BLAST)
# neg_rmsd               ~0.25          0.00     Correlated with tm_score; off by
#                                                default, enable if decorrelated
# ─────────────────────────────────────────────────────────────────────────
# Sum = 1.00
#
# Weights for signals that are absent from candidates_data are effectively 0
# (the column imputes to its mean → uniform rank → no discriminating power).
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS = {
    "neg_rmsd":           0.30,
    "delta_local_plddt":  0.25,
    "tm_score":           0.25,
    "local_plddt":        0.10,
    "mean_plddt":         0.10,
    "llr":                0.00,   # negative correlation on GB1 — disabled
    "llr_per_mut":        0.00,   # negative correlation on GB1 — disabled
    "conservation_score": 0.00,   # BLOSUM zero-variance on GB1 — disabled
}

# Canonical metric order for the score matrix (add new signals here)
_METRICS = [
    "tm_score",
    "delta_local_plddt",
    "mean_plddt",
    "local_plddt",
    "llr",
    "llr_per_mut",
    "conservation_score",
    "neg_rmsd",
]


def rank_candidates(candidates: list[dict], weights: dict = None) -> RankerResult:
    """
    Rank protein variant candidates via weighted rank aggregation (Borda count).

    For each metric, candidates are ranked 1…N (1 = worst raw value,
    N = best raw value).  The composite score is the weighted sum of those
    per-metric ranks.  Highest composite → rank 1 in the final ordering.

    Missing metric values are imputed with the column mean so that a failed
    tool call degrades gracefully rather than crashing the pipeline.

    Note on RMSD:  RMSD is lower-is-better.  Store it in candidates_data as
    ``neg_rmsd = −rmsd`` so the ranker's higher-is-better convention applies.

    Args:
        candidates: List of candidate dicts, each with at minimum a
                    ``candidate_id`` key plus any subset of the metric keys
                    listed in DEFAULT_WEIGHTS.
        weights:    Optional override dict.  Must sum to 1.0 ± 0.01.
                    Keys not in _METRICS are silently ignored.

    Returns:
        RankerResult with ranked_candidates (sorted desc by composite_score)
        and weights_used.
    """
    if len(candidates) < 2:
        raise ValueError("rank_candidates requires at least 2 candidates.")

    w = weights or DEFAULT_WEIGHTS

    if abs(sum(w.values()) - 1.0) > 0.01:
        raise ValueError(
            f"Weights must sum to 1.0 ± 0.01, got {sum(w.values()):.4f}"
        )

    # Merge default metrics with any extra keys in the provided weight dict
    # so callers can pass ad-hoc signals without editing this file.
    all_metrics = list(_METRICS)
    for key in w:
        if key not in all_metrics:
            all_metrics.append(key)

    # Build score matrix — missing keys become NaN
    scores = np.array(
        [[c.get(m, np.nan) for m in all_metrics] for c in candidates],
        dtype=float,
    )

    # Impute NaN columns with column mean (or 0.0 if entire column is NaN)
    with np.errstate(all="ignore"):
        col_means = np.nanmean(scores, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    nan_mask = np.isnan(scores)
    for col_idx in range(scores.shape[1]):
        scores[nan_mask[:, col_idx], col_idx] = col_means[col_idx]

    # Rank each metric: rank 1 = smallest value, rank N = largest value
    # (higher raw value → higher rank → contributes more to composite)
    ranks = np.apply_along_axis(rankdata, 0, scores)

    # Weighted composite score
    weight_vec = np.array([w.get(m, 0.0) for m in all_metrics])
    composite = ranks @ weight_vec

    # Final ranking: rank 1 = highest composite_score
    n = len(candidates)
    composite_ranks = rankdata(-composite)  # negate so rank 1 = highest composite

    result_candidates = []
    for i, cand in enumerate(candidates):
        enriched = dict(cand)
        enriched["composite_score"] = float(composite[i])
        enriched["rank"] = int(composite_ranks[i])
        result_candidates.append(enriched)

    result_candidates.sort(key=lambda x: x["composite_score"], reverse=True)

    return RankerResult(
        ranked_candidates=result_candidates,
        weights_used=w,
    )
