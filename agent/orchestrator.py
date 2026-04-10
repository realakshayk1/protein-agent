import concurrent.futures
import logging
import os
import time
from typing import Any, Callable, Dict

import numpy as np

from tools.blast import blast_conservation
from tools.esm2 import score_sequence_esm2
from tools.esmfold import predict_structure_esmfold
from tools.interface import BLASTResult, ESM2Result, ESMFoldResult, RankerResult, TMAlignResult
from tools.ranker import rank_candidates
from tools.tmalign import compute_structural_similarity
from tools.validation import assert_not_wildtype, validate_sequence


def run_agent(
    task: str,
    sequences: Dict[str, str],
    wildtype: str,
    reference_pdb_path: str,
    stream_callback: Callable = None,
    blast_workers: int = 3,
    esm2_model: str = "esm2_t33_650M_UR50D",
    use_triage: bool = True,
    triage_center_fraction: float = 0.60,
) -> Dict[str, Any]:
    """
    Execute the protein evaluation pipeline with optional triage routing.

    Stages
    ------
    1. ESM-2 sequence scoring (serial, disk-cached)
    2. [Triage] classify variants as certain/uncertain based on ESM-2 LLR
    3. ESMFold structure prediction (concurrent, 5 workers) — uncertain only if triage enabled
    4. TM-align structural similarity (serial) — uncertain only if triage enabled
    5. BLAST evolutionary log-odds (concurrent, blast_workers)
    6. Composite ranking (weighted aggregation)

    Triage logic
    ------------
    After ESM-2 scoring, variants whose LLR falls in the central
    `triage_center_fraction` of the distribution — or that carry ≥3 mutations
    (high epistatic risk) — are routed to expensive structural analysis.
    Variants with extreme (confident) ESM-2 scores skip ESMFold/TM-align and
    receive structural scores imputed from the mean of the uncertain cohort.
    This reduces API calls by ~40 % in an uncached setting while concentrating
    structural budget where it matters most.

    Args:
        task:                   Natural-language task description (logged only).
        sequences:              Mapping of candidate_id → amino-acid sequence.
        wildtype:               Wild-type reference sequence.
        reference_pdb_path:     Path to wild-type PDB for TM-align comparison.
        stream_callback:        Optional callable receiving progress events.
        blast_workers:          Concurrent BLAST threads (default 3).
        esm2_model:             ESM-2 HuggingFace model ID.
        use_triage:             Enable two-pass triage routing (default True).
        triage_center_fraction: Fraction of the LLR distribution considered
                                uncertain (default 0.60, i.e. middle 60 %).

    Returns:
        Dict with keys: status, ranked_candidates, weights_used,
                        imputed_count, tool_call_log, triage_summary.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if len(sequences) < 2:
        raise ValueError("At least 2 candidates required.")

    for cand_id, seq in sequences.items():
        validate_sequence(seq, candidate_id=cand_id)
        assert_not_wildtype(seq, wildtype, candidate_id=cand_id)

    if not os.path.exists(reference_pdb_path):
        raise ValueError(f"reference_pdb_path '{reference_pdb_path}' does not exist.")

    tool_call_log: list[str] = []

    def emit(event: dict) -> None:
        if stream_callback:
            stream_callback(event)

    start_time = time.time()

    candidates_data: Dict[str, dict] = {
        cand_id: {"candidate_id": cand_id, "sequence": seq, "imputed": False}
        for cand_id, seq in sequences.items()
    }

    # ------------------------------------------------------------------
    # Stage 1: ESM-2 scoring (serial — model load is not thread-safe)
    # ------------------------------------------------------------------
    stage1_start = time.time()
    esm2_results: Dict[str, ESM2Result | None] = {}

    for cand_id, seq in sequences.items():
        emit({"type": "tool_call", "tool": "score_sequence_esm2", "candidate_id": cand_id, "stage": 1})
        tool_call_log.append("score_sequence_esm2")
        try:
            res = score_sequence_esm2(wildtype, seq, model_name=esm2_model)
            esm2_results[cand_id] = res
            emit({"type": "tool_result", "tool": "score_sequence_esm2", "candidate_id": cand_id, "success": True, "imputed": False})
        except Exception as exc:
            logging.warning(f"ESM-2 failed for {cand_id}: {exc}")
            esm2_results[cand_id] = None

    valid_llrs = [r.llr for r in esm2_results.values() if r is not None]
    mean_llr = sum(valid_llrs) / len(valid_llrs) if valid_llrs else 0.0

    for cand_id in sequences:
        result = esm2_results[cand_id]
        if result is None:
            candidates_data[cand_id]["llr"] = mean_llr
            candidates_data[cand_id]["imputed"] = True
            emit({"type": "tool_result", "tool": "score_sequence_esm2", "candidate_id": cand_id, "success": False, "imputed": True})
        else:
            candidates_data[cand_id]["llr"] = result.llr

    emit({"type": "stage_complete", "stage": 1, "stage_name": "ESM-2", "duration_seconds": time.time() - stage1_start})

    # ------------------------------------------------------------------
    # Triage: decide which variants need structural analysis
    # ------------------------------------------------------------------
    if use_triage and valid_llrs:
        half_tail = (1.0 - triage_center_fraction) / 2.0
        q_lo = np.percentile(valid_llrs, half_tail * 100)
        q_hi = np.percentile(valid_llrs, (1.0 - half_tail) * 100)

        def _needs_structure(cand_id: str) -> bool:
            llr = candidates_data[cand_id]["llr"]
            n_muts = (esm2_results[cand_id].n_mutations
                      if esm2_results.get(cand_id) is not None else 1)
            return (q_lo <= llr <= q_hi) or (n_muts >= 3)

        uncertain_ids = [cid for cid in sequences if _needs_structure(cid)]
        confident_ids = [cid for cid in sequences if not _needs_structure(cid)]

        triage_summary = {
            "enabled": True,
            "llr_q_lo": float(q_lo),
            "llr_q_hi": float(q_hi),
            "n_uncertain": len(uncertain_ids),
            "n_confident": len(confident_ids),
            "uncertain_ids": uncertain_ids,
        }
        emit({"type": "triage_complete", "n_uncertain": len(uncertain_ids),
              "n_confident": len(confident_ids), "q_lo": float(q_lo), "q_hi": float(q_hi)})
    else:
        uncertain_ids = list(sequences.keys())
        confident_ids = []
        triage_summary = {"enabled": False}

    # Build the subset dict for structural stages
    uncertain_sequences = {cid: sequences[cid] for cid in uncertain_ids}

    # ------------------------------------------------------------------
    # Stage 2: ESMFold structure prediction (concurrent, uncertain only)
    # ------------------------------------------------------------------
    stage2_start = time.time()
    esmfold_results: Dict[str, ESMFoldResult | None] = {}

    def _run_esmfold(cand_id: str, seq: str):
        emit({"type": "tool_call", "tool": "predict_structure_esmfold", "candidate_id": cand_id, "stage": 2})
        tool_call_log.append("predict_structure_esmfold")
        try:
            res = predict_structure_esmfold(seq, cand_id)
            return cand_id, res, True
        except Exception as exc:
            logging.warning(f"ESMFold failed for {cand_id}: {exc}")
            return cand_id, None, False

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(_run_esmfold, cid, seq): cid for cid, seq in uncertain_sequences.items()}
        for future in concurrent.futures.as_completed(futures):
            cand_id, res, success = future.result()
            esmfold_results[cand_id] = res
            emit({"type": "tool_result", "tool": "predict_structure_esmfold", "candidate_id": cand_id, "success": success, "imputed": not success})

    # Mark confident variants as not-run (None)
    for cid in confident_ids:
        esmfold_results[cid] = None

    valid_plddts = [r.mean_plddt for r in esmfold_results.values() if r is not None]
    mean_plddt = sum(valid_plddts) / len(valid_plddts) if valid_plddts else 0.0

    for cand_id in sequences:
        res = esmfold_results.get(cand_id)
        if res is None:
            candidates_data[cand_id]["mean_plddt"] = mean_plddt
            candidates_data[cand_id]["local_plddt"] = mean_plddt
            candidates_data[cand_id]["pdb_string"] = None
            if cand_id not in confident_ids:
                candidates_data[cand_id]["imputed"] = True
        else:
            candidates_data[cand_id]["mean_plddt"] = res.mean_plddt
            candidates_data[cand_id]["pdb_string"] = res.pdb_string

            # Local pLDDT: mean pLDDT at the mutated positions specifically.
            # This is more discriminative than global mean for binding-interface
            # mutations like those in GB1 (positions 39-54).
            mutated_pos = (
                esm2_results[cand_id].mutated_positions
                if esm2_results.get(cand_id) is not None
                else []
            )
            if mutated_pos and res.plddt_per_residue:
                valid_local = [
                    res.plddt_per_residue[p]
                    for p in mutated_pos
                    if p < len(res.plddt_per_residue)
                ]
                candidates_data[cand_id]["local_plddt"] = (
                    sum(valid_local) / len(valid_local) if valid_local else res.mean_plddt
                )
            else:
                candidates_data[cand_id]["local_plddt"] = res.mean_plddt

    # Impute local_plddt for confident variants using mean of uncertain cohort
    valid_local_plddts = [
        candidates_data[cid]["local_plddt"]
        for cid in uncertain_ids
        if "local_plddt" in candidates_data[cid]
    ]
    mean_local_plddt = (
        sum(valid_local_plddts) / len(valid_local_plddts) if valid_local_plddts else mean_plddt
    )
    for cid in confident_ids:
        candidates_data[cid]["local_plddt"] = mean_local_plddt

    emit({"type": "stage_complete", "stage": 2, "stage_name": "ESMFold", "duration_seconds": time.time() - stage2_start})

    # ------------------------------------------------------------------
    # Stage 3: TM-align structural similarity (serial, uncertain only)
    # ------------------------------------------------------------------
    stage3_start = time.time()
    tmalign_results: Dict[str, TMAlignResult | None] = {}

    for cand_id in uncertain_ids:
        pdb_str = candidates_data[cand_id].get("pdb_string")
        if pdb_str:
            emit({"type": "tool_call", "tool": "compute_structural_similarity", "candidate_id": cand_id, "stage": 3})
            tool_call_log.append("compute_structural_similarity")
            try:
                res = compute_structural_similarity(pdb_str, reference_pdb_path)
                tmalign_results[cand_id] = res
                emit({"type": "tool_result", "tool": "compute_structural_similarity", "candidate_id": cand_id, "success": True, "imputed": False})
            except Exception as exc:
                logging.warning(f"TM-align failed for {cand_id}: {exc}")
                tmalign_results[cand_id] = None
                emit({"type": "tool_result", "tool": "compute_structural_similarity", "candidate_id": cand_id, "success": False, "imputed": True})
        else:
            tmalign_results[cand_id] = None
            emit({"type": "tool_result", "tool": "compute_structural_similarity", "candidate_id": cand_id, "success": False, "imputed": True})

    # Confident variants: TM-align not run
    for cid in confident_ids:
        tmalign_results[cid] = None

    valid_tms = [r.tm_score for r in tmalign_results.values() if r is not None]
    mean_tm = sum(valid_tms) / len(valid_tms) if valid_tms else 0.0

    for cand_id in sequences:
        res = tmalign_results.get(cand_id)
        if res is None:
            candidates_data[cand_id]["tm_score"] = mean_tm
            if cand_id not in confident_ids:
                candidates_data[cand_id]["imputed"] = True
        else:
            candidates_data[cand_id]["tm_score"] = res.tm_score

    emit({"type": "stage_complete", "stage": 3, "stage_name": "TM-align", "duration_seconds": time.time() - stage3_start})

    # ------------------------------------------------------------------
    # Stage 4: BLAST evolutionary log-odds (concurrent — disk-cached, I/O-bound)
    # ------------------------------------------------------------------
    stage4_start = time.time()
    blast_results: Dict[str, BLASTResult | None] = {}

    def _run_blast(cand_id: str, seq: str):
        esm2_res = esm2_results.get(cand_id)
        mutated_positions = esm2_res.mutated_positions if esm2_res else []
        if not mutated_positions:
            return cand_id, None, False

        emit({"type": "tool_call", "tool": "blast_conservation", "candidate_id": cand_id, "stage": 4})
        tool_call_log.append("blast_conservation")
        try:
            res = blast_conservation(seq, mutated_positions, wildtype_seq=wildtype)
            return cand_id, res, True
        except Exception as exc:
            logging.warning(f"BLAST failed for {cand_id}: {exc}")
            return cand_id, None, False

    if blast_workers > 0:
        with concurrent.futures.ThreadPoolExecutor(max_workers=blast_workers) as executor:
            futures = {executor.submit(_run_blast, cid, seq): cid for cid, seq in sequences.items()}
            for future in concurrent.futures.as_completed(futures):
                cand_id, res, success = future.result()
                blast_results[cand_id] = res
                emit({"type": "tool_result", "tool": "blast_conservation", "candidate_id": cand_id, "success": success, "imputed": not success})
    else:
        for cand_id in sequences:
            blast_results[cand_id] = None
            emit({"type": "tool_result", "tool": "blast_conservation", "candidate_id": cand_id, "success": False, "imputed": True})

    valid_blasts = [r.conservation_score for r in blast_results.values() if r is not None]
    mean_blast = sum(valid_blasts) / len(valid_blasts) if valid_blasts else 0.0

    for cand_id in sequences:
        res = blast_results.get(cand_id)
        if res is None:
            candidates_data[cand_id]["conservation_score"] = mean_blast
            candidates_data[cand_id]["imputed"] = True
        else:
            candidates_data[cand_id]["conservation_score"] = res.conservation_score

    emit({"type": "stage_complete", "stage": 4, "stage_name": "BLAST", "duration_seconds": time.time() - stage4_start})

    # ------------------------------------------------------------------
    # Stage 5: Composite ranking
    # ------------------------------------------------------------------
    emit({"type": "tool_call", "tool": "rank_candidates", "candidate_id": "all", "stage": 5})
    tool_call_log.append("rank_candidates")
    ranked: RankerResult = rank_candidates(list(candidates_data.values()))

    imputed_count = sum(1 for c in candidates_data.values() if c["imputed"])
    emit({"type": "ranking_complete", "n_candidates": len(sequences), "imputed_count": imputed_count})

    return {
        "status": "complete",
        "ranked_candidates": ranked.ranked_candidates,
        "weights_used": ranked.weights_used,
        "imputed_count": imputed_count,
        "tool_call_log": tool_call_log,
        "triage_summary": triage_summary,
        "total_duration_seconds": time.time() - start_time,
    }
