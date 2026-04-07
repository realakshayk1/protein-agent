import pytest
from tools.blast import blast_conservation
from tools.interface import BLASTResult

GB1_WT = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"

@pytest.mark.slow
def test_type_safety():
    res = blast_conservation(GB1_WT, [39])
    assert isinstance(res, BLASTResult)

@pytest.mark.slow
def test_conservation_range():
    res = blast_conservation(GB1_WT, [39])
    assert 0.0 <= res.conservation_score <= 1.0

def test_empty_positions_raises():
    with pytest.raises(ValueError):
        blast_conservation(GB1_WT, [])

def test_invalid_sequence_raises():
    with pytest.raises(ValueError):
        blast_conservation(GB1_WT + "X", [0])

def test_short_sequence_raises():
    with pytest.raises(ValueError):
        blast_conservation("ACD", [0])
