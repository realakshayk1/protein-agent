# tools/mock_implementations.py
# Coder B uses these until Coder A ships real implementations
from tools.interface import ESM2Result, ESMFoldResult, TMAlignResult, BLASTResult

def score_sequence_esm2(wildtype_seq, variant_seq, model_name="esm2_t6_8M_UR50D"):
    return ESM2Result(llr=0.42, wildtype_ll=-120.0, variant_ll=-119.58,
                      mutated_positions=[38], n_mutations=1)

def predict_structure_esmfold(sequence, candidate_id):
    return ESMFoldResult(candidate_id=candidate_id, pdb_string="MOCK_PDB",
                         mean_plddt=78.3, plddt_per_residue=[78.3]*len(sequence),
                         length=len(sequence))

def compute_structural_similarity(predicted_pdb, reference_pdb_path):
    return TMAlignResult(tm_score=0.85, rmsd=1.2)

def blast_conservation(sequence, mutated_positions, db="swissprot"):
    return BLASTResult(conservation_score=0.73, mutated_positions=mutated_positions, n_hits=42)
