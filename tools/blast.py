"""
BLAST conservation scoring for protein variants.

Backend priority:
  1. Local BLAST+ binary (blastp) + local SwissProt DB  — fast, offline, preferred
  2. Remote NCBI BLAST API                              — network fallback
  3. BLOSUM62 substitution score                        — zero-dependency fallback,
     used automatically when both BLAST backends are unavailable or fail.

Local BLAST+ setup (one-time, ~600 MB):
    # Install BLAST+
    sudo apt install ncbi-blast+            # Debian/Ubuntu
    brew install blast                      # macOS

    # Download and format SwissProt
    cd data/blast_db
    update_blastdb.pl --decompress swissprot   # uses BLAST+ helper script
    # OR manually:
    wget https://ftp.ncbi.nlm.nih.gov/blast/db/swissprot.tar.gz
    tar -xzf swissprot.tar.gz

    # Point the code at your DB (default: data/blast_db/swissprot)
    export BLAST_LOCAL_DB=/path/to/swissprot
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import shutil
import subprocess
import tempfile

from Bio.Blast import NCBIWWW, NCBIXML
from tools.interface import BLASTResult

# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------
_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "blast_cache")
_DEFAULT_LOCAL_DB = os.path.join(os.path.dirname(__file__), "..", "data", "blast_db", "swissprot")


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
        pass  # non-fatal


# ---------------------------------------------------------------------------
# Local BLAST+ detection
# ---------------------------------------------------------------------------

def local_blast_available() -> bool:
    """
    Return True if both the blastp binary and a local SwissProt DB are found.

    DB location is resolved from (in priority order):
      1. BLAST_LOCAL_DB environment variable
      2. data/blast_db/swissprot  (relative to project root)
    """
    if not shutil.which("blastp"):
        return False
    db_path = os.environ.get("BLAST_LOCAL_DB", _DEFAULT_LOCAL_DB)
    # BLAST databases exist as <name>.phr / <name>.psq (legacy) or <name>.pdb (v5)
    return any(
        os.path.exists(db_path + ext)
        for ext in (".phr", ".psq", ".pdb", ".pin")
    )


# ---------------------------------------------------------------------------
# BLOSUM62 fallback (zero external dependencies)
# ---------------------------------------------------------------------------

def _get_blosum62():
    """Load BLOSUM62 matrix via BioPython (already a project dependency)."""
    try:
        from Bio.Align import substitution_matrices
        return substitution_matrices.load("BLOSUM62")
    except Exception:
        return None


def blosum62_conservation(
    variant_seq: str,
    wildtype_seq: str,
    mutated_positions: list[int],
) -> BLASTResult:
    """
    Estimate evolutionary conservation via BLOSUM62 substitution scores.

    BLOSUM62 encodes log-odds of observing a substitution in aligned homologs.
    Higher scores mean more conserved / biochemically similar substitutions.
    This is used as a zero-dependency fallback when BLAST is unavailable.

    The per-position BLOSUM62 scores are averaged across all mutated positions
    and min-max normalised to [0, 1] using the empirical score range of the
    matrix (min ≈ −4 typical for common AAs, max = 11 for Cys→Cys).

    Args:
        variant_seq:       Full variant amino-acid sequence.
        wildtype_seq:      Full wildtype amino-acid sequence (same length).
        mutated_positions: 0-indexed positions where mutations occurred.

    Returns:
        BLASTResult with conservation_score in [0, 1] and n_hits = 0 to signal
        that this is a BLOSUM-derived (not BLAST-derived) score.
    """
    blosum = _get_blosum62()
    if blosum is None or not mutated_positions:
        return BLASTResult(conservation_score=0.5, mutated_positions=mutated_positions, n_hits=0)

    # Empirical BLOSUM62 range for common 20 AAs: roughly [-4, 11]
    BLOSUM_MIN = -4.0
    BLOSUM_MAX = 11.0

    total = 0.0
    valid = 0
    for pos in mutated_positions:
        wt_aa  = wildtype_seq[pos].upper()
        var_aa = variant_seq[pos].upper()
        try:
            score = blosum[wt_aa, var_aa]
            # Normalise to [0, 1]
            normalised = (score - BLOSUM_MIN) / (BLOSUM_MAX - BLOSUM_MIN)
            normalised = max(0.0, min(1.0, normalised))
            total += normalised
            valid += 1
        except KeyError:
            continue  # non-standard AA — skip

    mean_score = total / valid if valid > 0 else 0.5

    return BLASTResult(
        conservation_score=mean_score,
        mutated_positions=mutated_positions,
        n_hits=0,  # 0 = BLOSUM-derived, not BLAST-derived
    )


# ---------------------------------------------------------------------------
# Shared conservation computation (same logic for both BLAST backends)
# ---------------------------------------------------------------------------

def _conservation_from_record(
    blast_record,
    sequence: str,
    mutated_positions: list[int],
) -> tuple[float, int]:
    """
    Extract mean conservation score from a parsed BLAST record.

    Returns (mean_conservation_score, n_hits).
    """
    alignments = blast_record.alignments
    n_hits = len(alignments)

    if n_hits == 0:
        return 0.0, 0

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
    return mean_score, n_hits


# ---------------------------------------------------------------------------
# Local BLAST+ backend
# ---------------------------------------------------------------------------

def _blast_local(sequence: str, mutated_positions: list[int]) -> BLASTResult:
    """Run blastp against a local SwissProt database."""
    db_path = os.environ.get("BLAST_LOCAL_DB", _DEFAULT_LOCAL_DB)

    fasta = f">query\n{sequence.upper()}\n"

    with tempfile.TemporaryDirectory() as tmpdir:
        query_path = os.path.join(tmpdir, "query.fasta")
        out_path = os.path.join(tmpdir, "out.xml")

        with open(query_path, "w") as fh:
            fh.write(fasta)

        cmd = [
            "blastp",
            "-query", query_path,
            "-db", db_path,
            "-out", out_path,
            "-outfmt", "5",          # XML output — BioPython-parseable
            "-max_target_seqs", "50",
            "-num_threads", "4",
            "-evalue", "1e-3",
        ]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Local BLAST+ timed out after 120s")
        except FileNotFoundError:
            raise RuntimeError("blastp binary not found on PATH")

        if proc.returncode != 0:
            raise RuntimeError(f"blastp exited {proc.returncode}: {proc.stderr[:300]}")

        with open(out_path) as fh:
            blast_record = NCBIXML.read(fh)

    mean_score, n_hits = _conservation_from_record(blast_record, sequence, mutated_positions)
    return BLASTResult(
        conservation_score=mean_score,
        mutated_positions=mutated_positions,
        n_hits=n_hits,
    )


# ---------------------------------------------------------------------------
# Remote NCBI BLAST backend
# ---------------------------------------------------------------------------

def _blast_remote(sequence: str, mutated_positions: list[int], db: str) -> BLASTResult:
    """Run blastp against the NCBI remote API."""
    try:
        result_handle = NCBIWWW.qblast(
            "blastp",
            db,
            sequence.upper(),
            hitlist_size=50,
        )
    except Exception as exc:
        raise RuntimeError(f"NCBI BLAST query failed: {exc}")

    try:
        blast_record = NCBIXML.read(result_handle)
    finally:
        result_handle.close()

    mean_score, n_hits = _conservation_from_record(blast_record, sequence, mutated_positions)
    return BLASTResult(
        conservation_score=mean_score,
        mutated_positions=mutated_positions,
        n_hits=n_hits,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def blast_conservation(
    sequence: str,
    mutated_positions: list[int],
    db: str = "swissprot",
    use_cache: bool = True,
    wildtype_seq: str | None = None,
    allow_blosum_fallback: bool = True,
) -> BLASTResult:
    """
    Assess evolutionary conservation at mutated positions.

    Backend selection (in order):
      1. Local BLAST+ + SwissProt DB (if available)
      2. Remote NCBI BLAST API
      3. BLOSUM62 substitution scores (if allow_blosum_fallback=True and
         wildtype_seq is provided)

    Results from BLAST backends are cached to data/blast_cache/.

    Args:
        sequence:              Protein amino acid sequence (the variant).
        mutated_positions:     0-indexed positions where mutations occurred.
        db:                    NCBI database name for remote fallback.
        use_cache:             Read/write the disk cache (default True).
        wildtype_seq:          Wildtype sequence — required for BLOSUM62 fallback.
        allow_blosum_fallback: Fall back to BLOSUM62 when BLAST unavailable.

    Returns:
        BLASTResult.  n_hits == 0 indicates a BLOSUM62-derived score.
    """
    if not mutated_positions:
        raise ValueError("mutated_positions cannot be empty")

    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    if not (10 <= len(sequence) <= 1000):
        raise ValueError("Sequence length must be between 10 and 1000 residues")
    if not all(c.upper() in valid_aa for c in sequence):
        raise ValueError("Sequence contains invalid amino acid characters")

    # Cache read (only for real BLAST results)
    key = _cache_key(sequence, mutated_positions) if use_cache else None
    if use_cache:
        cached = _load_cache(key)
        if cached is not None:
            return cached

    # Backend selection
    blast_result: BLASTResult | None = None
    try:
        if local_blast_available():
            logging.info("BLAST: using local BLAST+ backend")
            blast_result = _blast_local(sequence, mutated_positions)
        else:
            logging.info("BLAST: local DB not found — using remote NCBI backend")
            blast_result = _blast_remote(sequence, mutated_positions, db)
    except Exception as exc:
        logging.warning(f"BLAST failed: {exc}")

    if blast_result is not None:
        if use_cache:
            _save_cache(key, blast_result)
        return blast_result

    # BLOSUM62 fallback
    if allow_blosum_fallback and wildtype_seq is not None:
        logging.info("BLAST: all backends failed — using BLOSUM62 fallback")
        return blosum62_conservation(sequence, wildtype_seq, mutated_positions)

    # Hard failure (no fallback available)
    raise RuntimeError(
        "All BLAST backends failed and no BLOSUM62 fallback available. "
        "Pass wildtype_seq to enable BLOSUM62 fallback."
    )
