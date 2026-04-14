import concurrent.futures
import logging
import os
import time
from typing import Any, Callable, Dict, Tuple

from tools.blast import blast_conservation, blosum62_conservation
from tools.esm2 import score_sequence_esm2
from tools.esmfold import predict_structure_esmfold
from tools.interface import BLASTResult, ESM2Result, ESMFoldResult, RankerResult, TMAlignResult
from tools.ranker import rank_candidates
from tools.tmalign import compute_structural_similarity
from tools.validation import assert_not_wildtype, validate_sequence


def _local_plddt(
    plddt_per_residue: list[float],
    mutated_positions: list[int],
    window: int = 2,
) -> float:
    """
    Mean pLDDT in a ±window neighbourhood around each mutated position.

    A wider window captures structural perturbation that radiates beyond the
    exact mutation sites, giving a more sensitive fold-quality signal than
    either the single-residue value or the whole-sequence mean.
    """
    if not plddt_per_residue or not mutated_positions:
        return 0.0
    L = len(plddt_per_residue)
    neighbourhood = set()
    for p in mutated_positions:
        for offset in range(-window, window + 1):
            idx = p + offset
            if 0 <= idx < L:
                neighbourhood.add(idx)
    return sum(plddt_per_residue[i] for i in neighbourhood) / len(neighbourhood)


def run_agent(
    task: str,
    sequences: Dict[str, str],
    wildtype: str,
    reference_pdb_path: str,
    stream_callback: Callable = None,
    blast_workers: int = 3,
    esm2_model: str = "esm2_t12_35M_UR50D",
    domain_slice: Tuple[int, int] | None = None,
    plddt_window: int = 2,
    custom_weights: dict | None = None,
) -> Dict[str, Any]:
    """
    Execute the full 6-stage protein evaluation pipeline.

    Stages
    ------
    0. WT ESMFold  — fold wildtype once; provides pLDDT reference for delta_local_plddt
    1. ESM-2       — focused mutant-marginal LLR (2 forward passes per mutation site)
    2. ESMFold     — structure prediction for all variants (concurrent, 5 workers)
    3. TM-align    — structural similarity vs reference PDB (serial)
    4. BLAST / BLOSUM62 — evolutionary conservation (concurrent; BLOSUM62 fallback)
    5. Ranking     — weighted Borda count over all signals

    New signals vs. previous version
    ----------------------------------
    local_plddt        — mean pLDDT in a ±plddt_window neighbourhood of each
                         mutation site (more sensitive than whole-sequence mean).
    delta_local_plddt  — local_plddt(variant) − local_plddt(wildtype); isolates
                         the fold-quality *change* caused by the mutations.
    llr_per_mut        — LLR divided by number of mutations; corrects for the
                         inflated LLR variance of high-order mutants.
    neg_rmsd           — −RMSD (stored so higher = better in the ranker); 0.0
                         weight by default but available for custom weight dicts.
    conservation_score — BLAST identity fraction when BLAST succeeds, else
                         BLOSUM62 normalised substitution score [0, 1].

    Args:
        task:               Natural-language description (logged only).
        sequences:          Mapping candidate_id → amino-acid sequence.
        wildtype:           Wild-type reference sequence.
        reference_pdb_path: Path to the WT PDB for TM-align comparison.
        stream_callback:    Optional progress event callback.
        blast_workers:      Concurrent BLAST threads (0 = skip BLAST entirely).
        esm2_model:         HuggingFace model name for ESM-2 / ESM-1v scoring.
        domain_slice:       Optional (start, end) to restrict ESM-2 scoring to a
                            specific domain (e.g. (0, 56) for 56-aa GB1 domain
                            embedded in a longer fusion construct).
        plddt_window:       ±residue window for local_plddt neighbourhood.

    Returns:
        Dict with keys: status, ranked_candidates, weights_used,
                        imputed_count, tool_call_log.
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
    # Stage 0: Fold wildtype once
    #   • Provides per-residue pLDDT baseline for delta_local_plddt
    #   • PDB string written to a temp file and used as the TM-align
    #     reference so that ESMFold(variant) is compared against
    #     ESMFold(WT) rather than the crystal PDB.  This eliminates
    #     the chain-length mismatch (265 aa predicted vs 56 aa crystal)
    #     and cancels out systematic ESMFold prediction error.
    #     Falls back to the supplied reference_pdb_path if ESMFold fails.
    # ------------------------------------------------------------------
    stage0_start = time.time()
    wt_plddt_per_residue: list[float] = []
    _wt_pdb_tmpfile = None
    # TM-align always uses the supplied crystal PDB — empirically superior to
    # ESMFold(WT) as reference because the crystal provides a precise anchor and
    # ESMFold prediction noise washes out the small per-variant structural signal.
    tmalign_reference_path = reference_pdb_path

    emit({"type": "tool_call", "tool": "predict_structure_esmfold", "candidate_id": "wildtype", "stage": 0})
    tool_call_log.append("predict_structure_esmfold_wt")
    try:
        wt_fold = predict_structure_esmfold(wildtype, "wildtype")
        wt_plddt_per_residue = wt_fold.plddt_per_residue
        emit({"type": "tool_result", "tool": "predict_structure_esmfold", "candidate_id": "wildtype",
              "success": True, "imputed": False})
    except Exception as exc:
        logging.warning(f"ESMFold failed for wildtype: {exc} — delta_local_plddt will be unavailable")
        emit({"type": "tool_result", "tool": "predict_structure_esmfold", "candidate_id": "wildtype",
              "success": False, "imputed": True})

    emit({"type": "stage_complete", "stage": 0, "stage_name": "WT-ESMFold",
          "duration_seconds": time.time() - stage0_start})

    # ------------------------------------------------------------------
    # Stage 1: ESM-2 scoring (serial — model not thread-safe during load)
    # ------------------------------------------------------------------
    stage1_start = time.time()
    esm2_results: Dict[str, ESM2Result | None] = {}

    for cand_id, seq in sequences.items():
        emit({"type": "tool_call", "tool": "score_sequence_esm2", "candidate_id": cand_id, "stage": 1})
        tool_call_log.append("score_sequence_esm2")
        try:
            res = score_sequence_esm2(wildtype, seq, model_name=esm2_model, domain_slice=domain_slice)
            esm2_results[cand_id] = res
            emit({"type": "tool_result", "tool": "score_sequence_esm2", "candidate_id": cand_id,
                  "success": True, "imputed": False})
        except Exception as exc:
            logging.warning(f"ESM-2 failed for {cand_id}: {exc}")
            esm2_results[cand_id] = None

    valid_llrs = [r.llr for r in esm2_results.values() if r is not None]
    mean_llr = sum(valid_llrs) / len(valid_llrs) if valid_llrs else 0.0

    for cand_id in sequences:
        result = esm2_results[cand_id]
        if result is None:
            candidates_data[cand_id]["llr"] = mean_llr
            candidates_data[cand_id]["llr_per_mut"] = mean_llr
            candidates_data[cand_id]["imputed"] = True
            emit({"type": "tool_result", "tool": "score_sequence_esm2", "candidate_id": cand_id,
                  "success": False, "imputed": True})
        else:
            candidates_data[cand_id]["llr"] = result.llr
            candidates_data[cand_id]["llr_per_mut"] = (
                result.llr / max(result.n_mutations, 1)
            )
            candidates_data[cand_id]["mutated_positions"] = result.mutated_positions

    emit({"type": "stage_complete", "stage": 1, "stage_name": "ESM-2",
          "duration_seconds": time.time() - stage1_start})

    # ------------------------------------------------------------------
    # Stage 2: ESMFold structure prediction (concurrent)
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
        futures = {executor.submit(_run_esmfold, cid, seq): cid for cid, seq in sequences.items()}
        for future in concurrent.futures.as_completed(futures):
            cand_id, res, success = future.result()
            esmfold_results[cand_id] = res
            emit({"type": "tool_result", "tool": "predict_structure_esmfold", "candidate_id": cand_id,
                  "success": success, "imputed": not success})

    valid_plddts    = [r.mean_plddt  for r in esmfold_results.values() if r is not None]
    valid_loc_plddt = []  # will accumulate below

    for cand_id in sequences:
        res = esmfold_results.get(cand_id)
        mut_pos = candidates_data[cand_id].get("mutated_positions", [])

        if res is None:
            candidates_data[cand_id]["mean_plddt"]  = 0.0
            candidates_data[cand_id]["local_plddt"] = 0.0
            candidates_data[cand_id]["pdb_string"]  = None
            candidates_data[cand_id]["imputed"]     = True
        else:
            loc = _local_plddt(res.plddt_per_residue, mut_pos, window=plddt_window)
            candidates_data[cand_id]["mean_plddt"]  = res.mean_plddt
            candidates_data[cand_id]["local_plddt"] = loc
            candidates_data[cand_id]["pdb_string"]  = res.pdb_string
            valid_loc_plddt.append(loc)

            # delta_local_plddt: how much did the fold quality change at mutation sites?
            if wt_plddt_per_residue:
                wt_loc = _local_plddt(wt_plddt_per_residue, mut_pos, window=plddt_window)
                candidates_data[cand_id]["delta_local_plddt"] = loc - wt_loc
            # else: delta_local_plddt absent → ranker imputes mean → no signal (safe)

    # Back-fill imputed local_plddt / delta values with column means
    mean_local_plddt = sum(valid_loc_plddt) / len(valid_loc_plddt) if valid_loc_plddt else 0.0
    for cand_id in sequences:
        if candidates_data[cand_id].get("local_plddt", 0.0) == 0.0 and not esmfold_results.get(cand_id):
            candidates_data[cand_id]["local_plddt"] = mean_local_plddt

    emit({"type": "stage_complete", "stage": 2, "stage_name": "ESMFold",
          "duration_seconds": time.time() - stage2_start})

    # ------------------------------------------------------------------
    # Stage 3: TM-align structural similarity (serial)
    # ------------------------------------------------------------------
    stage3_start = time.time()
    tmalign_results: Dict[str, TMAlignResult | None] = {}

    for cand_id in sequences:
        pdb_str = candidates_data[cand_id].get("pdb_string")
        if pdb_str:
            emit({"type": "tool_call", "tool": "compute_structural_similarity", "candidate_id": cand_id, "stage": 3})
            tool_call_log.append("compute_structural_similarity")
            try:
                res = compute_structural_similarity(pdb_str, tmalign_reference_path)
                tmalign_results[cand_id] = res
                emit({"type": "tool_result", "tool": "compute_structural_similarity", "candidate_id": cand_id,
                      "success": True, "imputed": False})
            except Exception as exc:
                logging.warning(f"TM-align failed for {cand_id}: {exc}")
                tmalign_results[cand_id] = None
                emit({"type": "tool_result", "tool": "compute_structural_similarity", "candidate_id": cand_id,
                      "success": False, "imputed": True})
        else:
            tmalign_results[cand_id] = None
            emit({"type": "tool_result", "tool": "compute_structural_similarity", "candidate_id": cand_id,
                  "success": False, "imputed": True})

    valid_tms  = [r.tm_score for r in tmalign_results.values() if r is not None]
    valid_rmsd = [r.rmsd     for r in tmalign_results.values() if r is not None and r.rmsd is not None]
    mean_tm    = sum(valid_tms)  / len(valid_tms)  if valid_tms  else 0.0
    mean_rmsd  = sum(valid_rmsd) / len(valid_rmsd) if valid_rmsd else 0.0

    for cand_id in sequences:
        res = tmalign_results.get(cand_id)
        if res is None:
            candidates_data[cand_id]["tm_score"] = mean_tm
            candidates_data[cand_id]["neg_rmsd"]  = -mean_rmsd
            candidates_data[cand_id]["imputed"]   = True
        else:
            candidates_data[cand_id]["tm_score"] = res.tm_score
            # Store as neg_rmsd so the ranker's higher-is-better convention applies
            candidates_data[cand_id]["neg_rmsd"] = (
                -res.rmsd if res.rmsd is not None else -mean_rmsd
            )

    emit({"type": "stage_complete", "stage": 3, "stage_name": "TM-align",
          "duration_seconds": time.time() - stage3_start})

    # ------------------------------------------------------------------
    # Stage 4: Conservation scoring (BLAST → BLOSUM62 fallback)
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
            res = blast_conservation(
                seq,
                mutated_positions,
                wildtype_seq=wildtype,
                allow_blosum_fallback=True,
            )
            return cand_id, res, True
        except Exception as exc:
            logging.warning(f"Conservation scoring failed for {cand_id}: {exc}")
            # Last-resort: direct BLOSUM62 call
            try:
                res = blosum62_conservation(seq, wildtype, mutated_positions)
                return cand_id, res, True
            except Exception:
                return cand_id, None, False

    if blast_workers > 0:
        with concurrent.futures.ThreadPoolExecutor(max_workers=blast_workers) as executor:
            futures = {executor.submit(_run_blast, cid, seq): cid for cid, seq in sequences.items()}
            for future in concurrent.futures.as_completed(futures):
                cand_id, res, success = future.result()
                blast_results[cand_id] = res
                emit({"type": "tool_result", "tool": "blast_conservation", "candidate_id": cand_id,
                      "success": success, "imputed": not success})
    else:
        # blast_workers=0: skip BLAST; still compute BLOSUM62 (cheap, no network)
        for cand_id, seq in sequences.items():
            esm2_res = esm2_results.get(cand_id)
            mut_pos  = esm2_res.mutated_positions if esm2_res else []
            if mut_pos:
                try:
                    res = blosum62_conservation(seq, wildtype, mut_pos)
                    blast_results[cand_id] = res
                except Exception:
                    blast_results[cand_id] = None
            else:
                blast_results[cand_id] = None
            emit({"type": "tool_result", "tool": "blast_conservation", "candidate_id": cand_id,
                  "success": blast_results[cand_id] is not None, "imputed": blast_results[cand_id] is None})

    valid_blasts = [r.conservation_score for r in blast_results.values() if r is not None]
    mean_blast   = sum(valid_blasts) / len(valid_blasts) if valid_blasts else 0.0

    for cand_id in sequences:
        res = blast_results.get(cand_id)
        if res is None:
            candidates_data[cand_id]["conservation_score"] = mean_blast
            candidates_data[cand_id]["imputed"] = True
        else:
            candidates_data[cand_id]["conservation_score"] = res.conservation_score

    emit({"type": "stage_complete", "stage": 4, "stage_name": "BLAST/BLOSUM62",
          "duration_seconds": time.time() - stage4_start})

    # ------------------------------------------------------------------
    # Stage 5: Composite ranking
    # ------------------------------------------------------------------
    emit({"type": "tool_call", "tool": "rank_candidates", "candidate_id": "all", "stage": 5})
    tool_call_log.append("rank_candidates")
    ranked: RankerResult = rank_candidates(
        list(candidates_data.values()),
        weights=custom_weights,  # None → use DEFAULT_WEIGHTS from ranker.py
    )

    imputed_count = sum(1 for c in candidates_data.values() if c["imputed"])
    emit({"type": "ranking_complete", "n_candidates": len(sequences), "imputed_count": imputed_count})

    return {
        "status": "complete",
        "ranked_candidates": ranked.ranked_candidates,
        "weights_used": ranked.weights_used,
        "imputed_count": imputed_count,
        "tool_call_log": tool_call_log,
        "total_duration_seconds": time.time() - start_time,
    }
