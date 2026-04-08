import pytest
from tools.esmfold import predict_structure_esmfold, _extract_plddt

# Short, well-folded sequence — villin headpiece, 35 residues
# Known to fold reliably, ESMFold should return high pLDDT (>70)
VILLIN = "LSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF"

# GB1 wildtype for cross-tool consistency
GB1_WT = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"

def test_esmfold_returns_correct_types():
    result = predict_structure_esmfold(VILLIN, "villin_test")
    assert isinstance(result.pdb_string, str)
    assert isinstance(result.mean_plddt, float)
    assert isinstance(result.plddt_per_residue, list)
    assert result.length == len(VILLIN)
    assert result.candidate_id == "villin_test"

def test_esmfold_pdb_is_valid():
    result = predict_structure_esmfold(VILLIN, "villin_test")
    # Valid PDB must contain ATOM records and END
    assert result.pdb_string.count("ATOM") > 0
    assert result.pdb_string.startswith("HEADER") or "ATOM" in result.pdb_string

def test_esmfold_plddt_range():
    result = predict_structure_esmfold(VILLIN, "villin_test")
    assert 0 <= result.mean_plddt <= 100
    assert all(0 <= p <= 100 for p in result.plddt_per_residue)
    # Villin headpiece is well-characterized — ESMFold should be confident
    assert result.mean_plddt > 70, (
        f"Expected pLDDT > 70 for villin headpiece, got {result.mean_plddt:.1f}"
    )

def test_esmfold_residue_count_matches_sequence():
    result = predict_structure_esmfold(VILLIN, "villin_test")
    assert len(result.plddt_per_residue) == len(VILLIN), (
        f"Expected {len(VILLIN)} pLDDT scores, got {len(result.plddt_per_residue)}"
    )

def test_esmfold_too_long_raises():
    long_seq = "A" * 401
    with pytest.raises(ValueError, match="too long"):
        predict_structure_esmfold(long_seq, "too_long")

def test_extract_plddt_parses_correctly():
    # Minimal valid PDB ATOM line with known B-factor
    mock_pdb = (
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 87.50           C\n"
        "ATOM      2  CB  ALA A   1       1.500   2.500   3.500  1.00 85.00           C\n"
        "ATOM      3  CA  GLY A   2       4.000   5.000   6.000  1.00 91.25           C\n"
    )
    scores = _extract_plddt(mock_pdb)
    assert len(scores) == 2           # only CA atoms
    assert scores[0] == pytest.approx(87.50)
    assert scores[1] == pytest.approx(91.25)
