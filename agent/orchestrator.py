import concurrent.futures
from typing import Dict, Any, Callable
from tools.esm2 import score_sequence_esm2
from tools.esmfold import predict_structure_esmfold
from tools.tmalign import compute_structural_similarity
from tools.blast import blast_conservation
from tools.ranker import rank_candidates
from tools.interface import ESM2Result, ESMFoldResult, TMAlignResult, BLASTResult, RankerResult
import time
import logging

def run_agent(task: str, sequences: Dict[str, str], wildtype: str, reference_pdb_path: str, stream_callback: Callable = None) -> Dict[str, Any]:
    # 1. Validate inputs
    if len(sequences) < 2:
        raise ValueError("At least 2 candidates required.")
        
    valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
    for cand_id, seq in sequences.items():
        if not (10 <= len(seq) <= 400):
            raise ValueError(f"Sequence {cand_id} length {len(seq)} is out of valid range 10-400.")
        if not set(seq).issubset(valid_aas):
            raise ValueError(f"Sequence {cand_id} contains invalid amino acids.")
            
    import os
    if not os.path.exists(reference_pdb_path):
        raise ValueError(f"reference_pdb_path {reference_pdb_path} does not exist.")

    tool_call_log = []
    
    def emit(event: dict):
        if stream_callback:
            stream_callback(event)

    start_time = time.time()

    candidates_data = {cand_id: {"candidate_id": cand_id, "sequence": seq, "imputed": False} for cand_id, seq in sequences.items()}

    # Stage 1: ESM-2 scoring (serial)
    stage1_start = time.time()
    esm2_results = {}
    for cand_id, seq in sequences.items():
        emit({"type": "tool_call", "tool": "score_sequence_esm2", "candidate_id": cand_id, "stage": 1})
        tool_call_log.append("score_sequence_esm2")
        try:
            res = score_sequence_esm2(wildtype, seq)
            esm2_results[cand_id] = res
            emit({"type": "tool_result", "tool": "score_sequence_esm2", "candidate_id": cand_id, "success": True, "imputed": False})
        except Exception as e:
            logging.warning(f"ESM2 failed for {cand_id}: {e}")
            esm2_results[cand_id] = None
    
    # Impute ESM2 llr
    valid_llrs = [r.llr for r in esm2_results.values() if r is not None and getattr(r, 'llr', None) is not None]
    mean_llr = sum(valid_llrs) / len(valid_llrs) if valid_llrs else 0.0
    
    for cand_id in sequences:
        if esm2_results[cand_id] is None or getattr(esm2_results[cand_id], 'llr', None) is None:
            candidates_data[cand_id]["llr"] = mean_llr
            candidates_data[cand_id]["imputed"] = True
            emit({"type": "tool_result", "tool": "score_sequence_esm2", "candidate_id": cand_id, "success": False, "imputed": True})
        else:
            candidates_data[cand_id]["llr"] = esm2_results[cand_id].llr

    emit({"type": "stage_complete", "stage": 1, "stage_name": "ESM-2", "duration_seconds": time.time() - stage1_start})

    # Stage 2: ESMFold structure prediction (concurrent)
    stage2_start = time.time()
    esmfold_results = {}
    
    def run_esmfold(cand_id, seq):
        emit({"type": "tool_call", "tool": "predict_structure_esmfold", "candidate_id": cand_id, "stage": 2})
        tool_call_log.append("predict_structure_esmfold")
        try:
            res = predict_structure_esmfold(seq, cand_id)
            return cand_id, res, True
        except Exception as e:
            logging.warning(f"ESMFold failed for {cand_id}: {e}")
            return cand_id, None, False

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(run_esmfold, cand_id, seq): cand_id for cand_id, seq in sequences.items()}
        for future in concurrent.futures.as_completed(futures):
            cand_id, res, success = future.result()
            esmfold_results[cand_id] = res
            emit({"type": "tool_result", "tool": "predict_structure_esmfold", "candidate_id": cand_id, "success": success, "imputed": not success})

    valid_plddts = [r.mean_plddt for r in esmfold_results.values() if r is not None and getattr(r, 'mean_plddt', None) is not None]
    mean_plddt = sum(valid_plddts) / len(valid_plddts) if valid_plddts else 0.0

    for cand_id in sequences:
        if esmfold_results.get(cand_id) is None:
            candidates_data[cand_id]["mean_plddt"] = mean_plddt
            candidates_data[cand_id]["pdb_string"] = None
            candidates_data[cand_id]["imputed"] = True
        else:
            candidates_data[cand_id]["mean_plddt"] = esmfold_results[cand_id].mean_plddt
            candidates_data[cand_id]["pdb_string"] = esmfold_results[cand_id].pdb_string

    emit({"type": "stage_complete", "stage": 2, "stage_name": "ESMFold", "duration_seconds": time.time() - stage2_start})

    # Stage 3: TM-align (serial)
    stage3_start = time.time()
    tmalign_results = {}
    for cand_id in sequences:
        pdb_str = candidates_data[cand_id].get("pdb_string")
        if pdb_str:
            emit({"type": "tool_call", "tool": "compute_structural_similarity", "candidate_id": cand_id, "stage": 3})
            tool_call_log.append("compute_structural_similarity")
            try:
                res = compute_structural_similarity(pdb_str, reference_pdb_path)
                tmalign_results[cand_id] = res
                emit({"type": "tool_result", "tool": "compute_structural_similarity", "candidate_id": cand_id, "success": True, "imputed": False})
            except Exception as e:
                logging.warning(f"TM-align failed for {cand_id}: {e}")
                tmalign_results[cand_id] = None
                emit({"type": "tool_result", "tool": "compute_structural_similarity", "candidate_id": cand_id, "success": False, "imputed": True})
        else:
            tmalign_results[cand_id] = None
            emit({"type": "tool_result", "tool": "compute_structural_similarity", "candidate_id": cand_id, "success": False, "imputed": True})

    valid_tms = [r.tm_score for r in tmalign_results.values() if r is not None and getattr(r, 'tm_score', None) is not None]
    mean_tm = sum(valid_tms) / len(valid_tms) if valid_tms else 0.0

    for cand_id in sequences:
        if tmalign_results.get(cand_id) is None:
             candidates_data[cand_id]["tm_score"] = mean_tm
             candidates_data[cand_id]["imputed"] = True
        else:
             candidates_data[cand_id]["tm_score"] = tmalign_results[cand_id].tm_score

    emit({"type": "stage_complete", "stage": 3, "stage_name": "TM-align", "duration_seconds": time.time() - stage3_start})

    # Stage 4: BLAST conservation (serial)
    stage4_start = time.time()
    blast_results = {}
    for cand_id, seq in sequences.items():
        esm2_res = esm2_results.get(cand_id)
        mutated_positions = esm2_res.mutated_positions if esm2_res else []
        if mutated_positions:
            emit({"type": "tool_call", "tool": "blast_conservation", "candidate_id": cand_id, "stage": 4})
            tool_call_log.append("blast_conservation")
            try:
                res = blast_conservation(seq, mutated_positions)
                blast_results[cand_id] = res
                emit({"type": "tool_result", "tool": "blast_conservation", "candidate_id": cand_id, "success": True, "imputed": False})
            except Exception as e:
                logging.warning(f"BLAST failed for {cand_id}: {e}")
                blast_results[cand_id] = None
                emit({"type": "tool_result", "tool": "blast_conservation", "candidate_id": cand_id, "success": False, "imputed": True})
        else:
            blast_results[cand_id] = None
            emit({"type": "tool_result", "tool": "blast_conservation", "candidate_id": cand_id, "success": False, "imputed": True})
            
    valid_blasts = [r.conservation_score for r in blast_results.values() if r is not None and getattr(r, 'conservation_score', None) is not None]
    mean_blast = sum(valid_blasts) / len(valid_blasts) if valid_blasts else 0.0

    for cand_id in sequences:
        if blast_results.get(cand_id) is None:
             candidates_data[cand_id]["conservation_score"] = mean_blast
             candidates_data[cand_id]["imputed"] = True
        else:
             candidates_data[cand_id]["conservation_score"] = blast_results[cand_id].conservation_score

    emit({"type": "stage_complete", "stage": 4, "stage_name": "BLAST", "duration_seconds": time.time() - stage4_start})

    # Stage 5: Rank
    emit({"type": "tool_call", "tool": "rank_candidates", "candidate_id": "all", "stage": 5})
    tool_call_log.append("rank_candidates")
    ranked = rank_candidates(list(candidates_data.values()))
    
    imputed_count = sum(1 for c in candidates_data.values() if c["imputed"])
    
    emit({"type": "ranking_complete", "n_candidates": len(sequences), "imputed_count": imputed_count})

    return {
        "status": "complete",
        "ranked_candidates": ranked.ranked_candidates,
        "weights_used": ranked.weights_used,
        "imputed_count": imputed_count,
        "tool_call_log": tool_call_log
    }
