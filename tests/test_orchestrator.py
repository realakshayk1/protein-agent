import pytest
from agent.orchestrator import run_agent

GB1_WT = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
CANDIDATES = {
    "V39F": GB1_WT[:38] + "F" + GB1_WT[39:],
    "V39A": GB1_WT[:38] + "A" + GB1_WT[39:],
}
REF_PDB = "reference_pdbs/2GI9.pdb"

@pytest.mark.slow
def test_returns_dict():
    result = run_agent("task", CANDIDATES, GB1_WT, REF_PDB)
    assert isinstance(result, dict)
    for key in ["status", "ranked_candidates", "weights_used", "imputed_count", "tool_call_log"]:
        assert key in result

@pytest.mark.slow
def test_status_is_complete():
    result = run_agent("task", CANDIDATES, GB1_WT, REF_PDB)
    assert result["status"] == "complete"

@pytest.mark.slow
def test_ranked_candidates_length():
    result = run_agent("task", CANDIDATES, GB1_WT, REF_PDB)
    assert len(result["ranked_candidates"]) == len(CANDIDATES)

@pytest.mark.slow
def test_all_candidates_have_rank():
    result = run_agent("task", CANDIDATES, GB1_WT, REF_PDB)
    for cand in result["ranked_candidates"]:
        assert "rank" in cand

@pytest.mark.slow
def test_rank_1_exists():
    result = run_agent("task", CANDIDATES, GB1_WT, REF_PDB)
    ranks = [cand["rank"] for cand in result["ranked_candidates"]]
    assert ranks.count(1) == 1

@pytest.mark.slow
def test_tool_call_log_ordered():
    result = run_agent("task", CANDIDATES, GB1_WT, REF_PDB)
    log = result["tool_call_log"]
    esm2_idx = log.index("score_sequence_esm2")
    esmfold_idx = log.index("predict_structure_esmfold")
    tmalign_idx = log.index("compute_structural_similarity")
    blast_idx = log.index("blast_conservation")
    rank_idx = log.index("rank_candidates")
    
    assert esm2_idx < esmfold_idx < tmalign_idx < blast_idx < rank_idx

def test_invalid_sequence_raises():
    bad_cands = {"bad": "XXX", "good": CANDIDATES["V39F"]}
    with pytest.raises(ValueError):
        run_agent("task", bad_cands, GB1_WT, REF_PDB)

def test_too_few_candidates_raises():
    too_few = {"V39F": CANDIDATES["V39F"]}
    with pytest.raises(ValueError, match="At least 2 candidates required"):
        run_agent("task", too_few, GB1_WT, REF_PDB)
