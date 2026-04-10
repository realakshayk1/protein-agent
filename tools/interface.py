# tools/interface.py
# THE CONTRACT — do not change signatures without coordination with Coder B

from dataclasses import dataclass

@dataclass
class ESM2Result:
    llr: float                      # log-likelihood ratio vs wildtype (higher = more fit)
    wildtype_ll: float
    variant_ll: float
    mutated_positions: list[int]    # 0-indexed
    n_mutations: int

@dataclass
class ESMFoldResult:
    candidate_id: str
    pdb_string: str
    mean_plddt: float               # 0-100, higher = more confident
    plddt_per_residue: list[float]
    length: int

@dataclass
class TMAlignResult:
    tm_score: float                 # 0-1, >0.5 = same fold
    rmsd: float | None

@dataclass
class BLASTResult:
    conservation_score: float       # summed log-odds across mutated positions:
                                    # Σ log P(variant_aa|MSA) - log P(wt_aa|MSA)
                                    # positive = evolutionarily favoured substitution
    mutated_positions: list[int]
    n_hits: int

@dataclass
class RankerResult:
    ranked_candidates: list[dict]
    weights_used: dict

@dataclass
class RankedCandidate:
    candidate_id: str
    sequence: str
    rank: int
    composite_score: float
    llr: float | None
    mean_plddt: float | None
    tm_score: float | None
    conservation_score: float | None

# --- FUNCTION SIGNATURES ---
# Implement these in tools/esm2.py, tools/esmfold.py, etc.
# Return types are exact — Coder B depends on these fields by name.

def score_sequence_esm2(
    wildtype_seq: str,
    variant_seq: str,
    model_name: str = "esm2_t12_35M_UR50D"
) -> ESM2Result: ...

def predict_structure_esmfold(
    sequence: str,
    candidate_id: str
) -> ESMFoldResult: ...

def compute_structural_similarity(
    predicted_pdb: str,
    reference_pdb_path: str
) -> TMAlignResult: ...

def blast_conservation(
    sequence: str,
    mutated_positions: list[int],
    db: str = "swissprot"
) -> BLASTResult: ...

def rank_candidates(
    candidates: list[dict],   # each dict has candidate_id, sequence, + optional scores
    weights: dict | None = None
) -> RankerResult: ...