import hashlib
import json
import os
from Bio.Blast import NCBIWWW, NCBIXML
from tools.interface import BLASTResult

# ---------------------------------------------------------------------------
# Disk cache – avoids re-querying NCBI for identical (sequence, positions) pairs.
# Each cache entry is a small JSON file keyed by SHA-256 of the query parameters.
# ---------------------------------------------------------------------------
_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "blast_cache")


def _cache_key(sequence: str, mutated_positions: list[int]) -> str:
    payload = sequence.upper() + ":" + ",".join(str(p) for p in sorted(mutated_positions))
    return hashlib.sha256(payload.encode()).hexdigest()


def _load_cache(key: str) -> BLASTResult | None:
    path = os.path.join(_CACHE_DIR, f"{key}.json")
    if os.path.exists(path):
        try:
            with open(path) as fh:
                d = json.load(fh)
            return BLASTResult(
                conservation_score=d["conservation_score"],
                mutated_positions=d["mutated_positions"],
                n_hits=d["n_hits"],
            )
        except Exception:
            return None
    return None


def _save_cache(key: str, result: BLASTResult) -> None:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    path = os.path.join(_CACHE_DIR, f"{key}.json")
    try:
        with open(path, "w") as fh:
            json.dump(
                {
                    "conservation_score": result.conservation_score,
                    "mutated_positions": result.mutated_positions,
                    "n_hits": result.n_hits,
                },
                fh,
            )
    except Exception:
        pass  # cache write failure is non-fatal


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def blast_conservation(
    sequence: str,
    mutated_positions: list[int],
    db: str = "swissprot",
    use_cache: bool = True,
) -> BLASTResult:
    """
    Run BLAST on a sequence to assess evolutionary conservation at mutated positions.

    Results are cached to disk (data/blast_cache/) so repeated runs are instant.

    Args:
        sequence: The protein sequence to query.
        mutated_positions: 0-indexed positions where mutations occurred.
        db: NCBI BLAST database (default: swissprot).
        use_cache: Whether to read/write the disk cache (default True).

    Returns:
        BLASTResult with the mean conservation score across mutated positions.
    """
    if not mutated_positions:
        raise ValueError("mutated_positions cannot be empty")

    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    if not (10 <= len(sequence) <= 1000):
        raise ValueError("Sequence length must be between 10 and 1000 residues")

    if not all(c.upper() in valid_aa for c in sequence):
        raise ValueError("Sequence contains invalid amino acid characters")

    # --- cache read ---
    if use_cache:
        key = _cache_key(sequence, mutated_positions)
        cached = _load_cache(key)
        if cached is not None:
            return cached

    # --- remote BLAST query ---
    try:
        result_handle = NCBIWWW.qblast(
            "blastp",
            db,
            sequence.upper(),
            hitlist_size=50,
        )
    except Exception as e:
        raise RuntimeError(f"BLAST query failed: {e}")

    try:
        blast_record = NCBIXML.read(result_handle)
    finally:
        result_handle.close()

    alignments = blast_record.alignments
    n_hits = len(alignments)

    if n_hits == 0:
        result = BLASTResult(
            conservation_score=0.0,
            mutated_positions=mutated_positions,
            n_hits=0,
        )
        if use_cache:
            _save_cache(key, result)
        return result

    total_conservation = 0.0

    for pos in mutated_positions:
        query_aa = sequence[pos].upper()
        matches = 0
        valid_hits_for_pos = 0

        for alignment in alignments[:50]:
            hsp = alignment.hsps[0]

            q_start = hsp.query_start - 1  # convert to 0-indexed
            q_end = q_start + len(hsp.query.replace("-", ""))

            if q_start <= pos < q_end:
                # Walk the alignment to find the subject AA at this query position
                q_idx = q_start
                aln_idx = 0
                while q_idx < pos and aln_idx < len(hsp.query):
                    if hsp.query[aln_idx] != "-":
                        q_idx += 1
                    aln_idx += 1

                if aln_idx < len(hsp.sbjct):
                    sbjct_aa = hsp.sbjct[aln_idx]
                    if sbjct_aa.upper() == query_aa:
                        matches += 1
                    valid_hits_for_pos += 1

        if valid_hits_for_pos > 0:
            total_conservation += matches / valid_hits_for_pos

    mean_score = total_conservation / len(mutated_positions) if mutated_positions else 0.0

    result = BLASTResult(
        conservation_score=mean_score,
        mutated_positions=mutated_positions,
        n_hits=n_hits,
    )

    if use_cache:
        _save_cache(key, result)

    return result
