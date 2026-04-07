import pytest
import os
from tools.tmalign import compute_structural_similarity
from tools.interface import TMAlignResult
from tools.esmfold import predict_structure_esmfold

VALID_PDB_STRING = """ATOM      1  N   MET A   1      27.340  24.430   2.614  1.00 24.13           N  
ATOM      2  CA  MET A   1      26.266  25.413   2.842  1.00 27.53           C  
ATOM      3  C   MET A   1      26.913  26.639   3.461  1.00 28.53           C  
ATOM      4  O   MET A   1      27.895  26.561   4.205  1.00 30.56           O  
ATOM      5  CB  MET A   1      25.112  24.880   3.649  1.00 32.25           C  
ATOM      6  CG  MET A   1      25.353  24.860   5.151  1.00 34.61           C  
ATOM      7  SD  MET A   1      23.958  24.326   6.162  1.00 45.45           S  
ATOM      8  CE  MET A   1      24.447  25.419   7.487  1.00 40.54           C  
END
"""

GB1_WT = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"

def test_type_safety(tmp_path):
    ref_pdb = tmp_path / "ref.pdb"
    ref_pdb.write_text(VALID_PDB_STRING)
    result = compute_structural_similarity(VALID_PDB_STRING, str(ref_pdb))
    assert isinstance(result, TMAlignResult)

def test_tm_score_range(tmp_path):
    ref_pdb = tmp_path / "ref.pdb"
    ref_pdb.write_text(VALID_PDB_STRING)
    result = compute_structural_similarity(VALID_PDB_STRING, str(ref_pdb))
    if result.tm_score is not None:
        assert 0.0 <= result.tm_score <= 1.0

@pytest.mark.slow
def test_gb1_wt_vs_2GI9():
    ref_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reference_pdbs", "2GI9.pdb"))
    esmfold_res = predict_structure_esmfold(GB1_WT, "gb1_wt")
    result = compute_structural_similarity(esmfold_res.pdb_string, ref_path)
    assert result.tm_score is not None
    assert result.tm_score > 0.5

def test_missing_reference_raises():
    with pytest.raises(FileNotFoundError):
        compute_structural_similarity(VALID_PDB_STRING, "non_existent_file.pdb")

def test_rmsd_is_float(tmp_path):
    ref_pdb = tmp_path / "ref.pdb"
    ref_pdb.write_text(VALID_PDB_STRING)
    result = compute_structural_similarity(VALID_PDB_STRING, str(ref_pdb))
    assert isinstance(result.rmsd, float) or result.rmsd is None
