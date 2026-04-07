# tests/test_esm2.py
import pytest
from tools.esm2 import score_sequence_esm2

GB1_WT = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"

# V39F: known beneficial mutation (higher experimental fitness)
# V39A: known deleterious mutation (lower experimental fitness)
# Source: Olson et al. 2014, GB1 combinatorial dataset
V39F = GB1_WT[:38] + "F" + GB1_WT[39:]
V39A = GB1_WT[:38] + "A" + GB1_WT[39:]

def test_esm2_returns_correct_types():
    result = score_sequence_esm2(GB1_WT, V39F)
    assert isinstance(result.llr, float)
    assert isinstance(result.mutated_positions, list)
    assert result.n_mutations == 1
    assert result.mutated_positions == [38]   # 0-indexed

def test_esm2_wildtype_vs_self_is_zero():
    result = score_sequence_esm2(GB1_WT, GB1_WT)
    assert abs(result.llr) < 1e-3, f"Self-comparison LLR should be ~0, got {result.llr}"

def test_esm2_directional():
    """ESM-2 should rank known beneficial mutation above known deleterious."""
    result_beneficial  = score_sequence_esm2(GB1_WT, V39F)
    result_deleterious = score_sequence_esm2(GB1_WT, V39A)
    assert result_beneficial.llr > result_deleterious.llr, (
        f"Directional test failed: V39F LLR={result_beneficial.llr:.3f}, "
        f"V39A LLR={result_deleterious.llr:.3f}. "
        f"ESM-2 should prefer the beneficial mutation."
    )

def test_esm2_length_mismatch_raises():
    with pytest.raises(ValueError):
        score_sequence_esm2(GB1_WT, GB1_WT[:-1])
