import pytest
from tools.ranker import rank_candidates
from tools.interface import RankerResult

CANDIDATES = [
    {"id": "var_A", "llr": 1.2,  "mean_plddt": 82.0, "tm_score": 0.85, "conservation_score": 0.90},
    {"id": "var_B", "llr": -0.3, "mean_plddt": 71.0, "tm_score": 0.72, "conservation_score": 0.60},
    {"id": "var_C", "llr": 2.1,  "mean_plddt": 88.0, "tm_score": 0.91, "conservation_score": 0.95},
    {"id": "var_D", "llr": 0.5,  "mean_plddt": 65.0, "tm_score": 0.60, "conservation_score": 0.40},
]

def test_return_type():
    result = rank_candidates(CANDIDATES)
    assert isinstance(result, RankerResult)

def test_output_length():
    result = rank_candidates(CANDIDATES)
    assert len(result.ranked_candidates) == len(CANDIDATES)

def test_sorted_descending():
    result = rank_candidates(CANDIDATES)
    scores = [c["composite_score"] for c in result.ranked_candidates]
    assert scores == sorted(scores, reverse=True)

def test_best_candidate_is_var_c():
    """var_C dominates on all metrics — must rank first."""
    result = rank_candidates(CANDIDATES)
    assert result.ranked_candidates[0]["id"] == "var_C"

def test_worst_candidate_is_var_d():
    """var_D is weakest on all metrics — must rank last."""
    result = rank_candidates(CANDIDATES)
    assert result.ranked_candidates[-1]["id"] == "var_D"

def test_rank_field_is_1_for_best():
    result = rank_candidates(CANDIDATES)
    top = result.ranked_candidates[0]
    assert top["rank"] == 1

def test_custom_weights():
    # ESM-2 only weighting
    w = {"llr": 1.0, "mean_plddt": 0.0, "tm_score": 0.0, "conservation_score": 0.0}
    result = rank_candidates(CANDIDATES, weights=w)
    # var_C has highest LLR (2.1), should still be first
    assert result.ranked_candidates[0]["id"] == "var_C"

def test_weights_dont_sum_to_one_raises():
    bad_weights = {"llr": 0.5, "mean_plddt": 0.5, "tm_score": 0.5, "conservation_score": 0.0}
    with pytest.raises(ValueError, match="sum to 1.0"):
        rank_candidates(CANDIDATES, weights=bad_weights)

def test_fewer_than_two_candidates_raises():
    with pytest.raises(ValueError, match="at least 2"):
        rank_candidates([CANDIDATES[0]])

def test_missing_metric_handled():
    """Candidates missing a metric key should not crash — NaN filled with col mean."""
    sparse = [
        {"id": "x", "llr": 1.0, "mean_plddt": 80.0},  # missing tm_score, conservation_score
        {"id": "y", "llr": 0.5, "mean_plddt": 70.0},
    ]
    result = rank_candidates(sparse)
    assert len(result.ranked_candidates) == 2

def test_weights_used_preserved():
    custom = {"llr": 0.70, "mean_plddt": 0.10, "tm_score": 0.10, "conservation_score": 0.10}
    result = rank_candidates(CANDIDATES, weights=custom)
    assert result.weights_used == custom
