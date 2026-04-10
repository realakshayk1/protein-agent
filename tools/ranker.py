import numpy as np
from scipy.stats import rankdata
from tools.interface import RankerResult

# Weight rationale:
#   llr (0.40)          — ESM-2 focused masked marginal, strong sequence signal
#   local_plddt (0.20)  — pLDDT at mutated positions only, more discriminative
#                          than global mean for interface mutations
#   tm_score (0.20)     — fold preservation relative to WT, captures large
#                          structural disruptions
#   conservation (0.10) — BLAST log-odds at mutated positions (evolutionary)
#   mean_plddt (0.10)   — global structural confidence, down-weighted because
#                          it largely correlates with local_plddt
DEFAULT_WEIGHTS = {
    "llr": 0.40,
    "local_plddt": 0.20,
    "tm_score": 0.20,
    "conservation_score": 0.10,
    "mean_plddt": 0.10,
}

def rank_candidates(candidates: list[dict], weights: dict = None) -> RankerResult:
    """
    Rank protein variant candidates via weighted rank aggregation.

    Args:
        candidates: List of candidate dicts, each with keys:
                    llr, local_plddt, mean_plddt, tm_score, conservation_score.
        weights: Optional dict of per-metric weights. Must sum to 1.0 ± 0.01.
                 Defaults to {llr:0.40, local_plddt:0.20, tm_score:0.20,
                 conservation_score:0.10, mean_plddt:0.10}.

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

    metrics = list(w.keys())

    # Build score matrix — missing keys become NaN
    scores = np.array(
        [[c.get(m, np.nan) for m in metrics] for c in candidates],
        dtype=float,
    )

    # Replace NaN with column mean
    with np.errstate(all="ignore"):
        col_means = np.nanmean(scores, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    nan_mask = np.isnan(scores)
    for col_idx in range(scores.shape[1]):
        scores[nan_mask[:, col_idx], col_idx] = col_means[col_idx]

    # Rank each metric across candidates (higher raw value = higher rank)
    ranks = np.apply_along_axis(rankdata, 0, scores)

    # Weighted composite score
    weight_vec = np.array([w.get(m, 0.0) for m in metrics])
    composite = ranks @ weight_vec

    # Attach scores and ranks back to candidate dicts (copies, not mutations)
    result_candidates = []
    composite_ranks = rankdata(-composite)  # rank 1 = highest composite

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
