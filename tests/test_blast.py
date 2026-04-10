import pytest
from tools.blast import blast_conservation
from tools.interface import BLASTResult

GB1_WT = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"

@pytest.mark.slow
def test_type_safety():
    res = blast_conservation(GB1_WT, [39])
    assert isinstance(res, BLASTResult)

@pytest.mark.slow
def test_conservation_score_is_float():
    # conservation_score is now a log-odds value (unbounded real number),
    # not a [0, 1] frequency — just verify it's a finite float.
    res = blast_conservation(GB1_WT, [39])
    assert isinstance(res.conservation_score, float)
    assert not (res.conservation_score != res.conservation_score)  # not NaN

def test_empty_positions_raises():
    with pytest.raises(ValueError):
        blast_conservation(GB1_WT, [])

def test_invalid_sequence_raises():
    with pytest.raises(ValueError):
        blast_conservation(GB1_WT + "X", [0])

def test_short_sequence_raises():
    with pytest.raises(ValueError):
        blast_conservation("ACD", [0])
