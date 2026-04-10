"""
BLAST evolutionary log-odds scoring for protein variants.

For each mutated position i the tool computes:

    score_i = log P(variant_aa_i | homologs) - log P(wt_aa_i | homologs)

where P(aa | homologs) is the empirical frequency of amino acid aa at
position i across the top-50 BLAST hits (pseudocount ε = 0.01 added to
avoid log(0)).  The conservation_score stored in BLASTResult is the sum of
these per-position log-odds values across all mutated positions.

Positive scores indicate the variant amino acid is more common in homologs
than the wildtype amino acid at that position (evolutionarily favoured
substitution).  Negative scores indicate the substitution is disfavoured.

This replaces the previous "mean WT-conservation" approach, which measured
how often the *wildtype* amino acid appeared in homologs — a quantity
identical for all variants at the same position and therefore carrying zero
information about fitness differences between variants.

Backend priority:
  1. Local BLAST+ binary (blastp) + local SwissProt DB  — fast, offline, preferred
  2. Remote NCBI BLAST API                              — fallback if local unavailable

Both paths share the same disk cache (data/blast_cache/) so results are free on
any repeat call regardless of which backend ran first.

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
import math
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

_PSEUDOCOUNT = 0.01  # added to AA frequencies before log to avoid log(0)


def _cache_key(sequence: str, mutated_positions: list[int], wildtype_seq: str | None) -> str:
    payload = sequence.upper() + ":" + ",".join(str(p) for p in sorted(mutated_positions))
    if wildtype_seq is not None:
        payload += ":" + wildtype_seq.upper()
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
# Shared log-odds computation (same logic for both backends)
# ---------------------------------------------------------------------------

def _logodds_from_record(
    blast_record,
    sequence: str,
    mutated_positions: list[int],
    wildtype_seq: str | None,
) -> tuple[float, int]:
    """
    Compute summed log-odds score from a parsed BLAST record.

    For each mutated position i, counts the frequency of:
      - variant_aa_i  (from `sequence`)
      - wt_aa_i       (from `wildtype_seq`, if provided)
    across up to 50 BLAST hits.  Returns:

        score = Σ_i  log(P(variant_aa_i) + ε) - log(P(wt_aa_i) + ε)

    When wildtype_seq is None (backward-compat mode), falls back to
    log P(variant_aa_i) — a weaker but still informative single-sided score.

    Returns (score, n_hits).
    """
    alignments = blast_record.alignments
    n_hits = len(alignments)

    if n_hits == 0:
        return 0.0, 0

    total_logodds = 0.0

    for pos in mutated_positions:
        variant_aa = sequence[pos].upper()
        wt_aa = wildtype_seq[pos].upper() if wildtype_seq is not None else None

        variant_count = 0
        wt_count = 0
        valid_hits = 0

        for alignment in alignments[:50]:
            hsp = alignment.hsps[0]

            q_start = hsp.query_start - 1  # convert to 0-indexed
            q_end = q_start + len(hsp.query.replace("-", ""))

            if q_start <= pos < q_end:
                # Walk the alignment to find subject residue at query position
                q_idx = q_start
                aln_idx = 0
                while q_idx < pos and aln_idx < len(hsp.query):
                    if hsp.query[aln_idx] != "-":
                        q_idx += 1
                    aln_idx += 1

                if aln_idx < len(hsp.sbjct):
                    sbjct_aa = hsp.sbjct[aln_idx].upper()
                    if sbjct_aa == variant_aa:
                        variant_count += 1
                    if wt_aa is not None and sbjct_aa == wt_aa:
                        wt_count += 1
                    valid_hits += 1

        if valid_hits > 0:
            p_variant = (variant_count / valid_hits) + _PSEUDOCOUNT
            if wt_aa is not None:
                p_wt = (wt_count / valid_hits) + _PSEUDOCOUNT
                total_logodds += math.log(p_variant) - math.log(p_wt)
            else:
                # Fallback: single-sided log frequency
                total_logodds += math.log(p_variant)

    return total_logodds, n_hits


# ---------------------------------------------------------------------------
# Local BLAST+ backend
# ---------------------------------------------------------------------------

def _blast_local(
    sequence: str,
    mutated_positions: list[int],
    wildtype_seq: str | None,
) -> BLASTResult:
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

    score, n_hits = _logodds_from_record(blast_record, sequence, mutated_positions, wildtype_seq)
    return BLASTResult(
        conservation_score=score,
        mutated_positions=mutated_positions,
        n_hits=n_hits,
    )


# ---------------------------------------------------------------------------
# Remote NCBI BLAST backend
# ---------------------------------------------------------------------------

def _blast_remote(
    sequence: str,
    mutated_positions: list[int],
    wildtype_seq: str | None,
    db: str,
) -> BLASTResult:
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

    score, n_hits = _logodds_from_record(blast_record, sequence, mutated_positions, wildtype_seq)
    return BLASTResult(
        conservation_score=score,
        mutated_positions=mutated_positions,
        n_hits=n_hits,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def blast_conservation(
    sequence: str,
    mutated_positions: list[int],
    wildtype_seq: str | None = None,
    db: str = "swissprot",
    use_cache: bool = True,
) -> BLASTResult:
    """
    Compute evolutionary log-odds scores at mutated positions via BLAST.

    For each mutated position i, the returned conservation_score is:

        Σ_i  log P(variant_aa_i | MSA) - log P(wt_aa_i | MSA)

    where P is estimated from the top-50 BLAST hit frequencies
    (pseudocount ε=0.01).  Positive = evolutionarily favoured substitution.

    Automatically uses the local BLAST+ binary + SwissProt DB if available
    (set BLAST_LOCAL_DB env var or place db at data/blast_db/swissprot).
    Falls back to remote NCBI BLAST when local is not set up.

    Results are cached to data/blast_cache/ keyed by SHA-256 of
    (sequence, mutated_positions, wildtype_seq) — repeat calls are instant.

    Args:
        sequence:          Variant protein sequence.
        mutated_positions: 0-indexed positions where mutations occurred.
        wildtype_seq:      Wildtype sequence (same length as sequence).
                           Required for full log-odds scoring; when None
                           falls back to log P(variant_aa) only.
        db:                NCBI database name (remote fallback only).
        use_cache:         Read/write the disk cache (default True).

    Returns:
        BLASTResult with summed log-odds conservation_score.
    """
    if not mutated_positions:
        raise ValueError("mutated_positions cannot be empty")

    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    if not (10 <= len(sequence) <= 1000):
        raise ValueError("Sequence length must be between 10 and 1000 residues")
    if not all(c.upper() in valid_aa for c in sequence):
        raise ValueError("Sequence contains invalid amino acid characters")

    # Cache read
    key = _cache_key(sequence, mutated_positions, wildtype_seq) if use_cache else None
    if use_cache and key:
        cached = _load_cache(key)
        if cached is not None:
            return cached

    # Backend selection
    if local_blast_available():
        logging.info("BLAST: using local BLAST+ backend")
        result = _blast_local(sequence, mutated_positions, wildtype_seq)
    else:
        logging.info("BLAST: local DB not found — using remote NCBI backend")
        result = _blast_remote(sequence, mutated_positions, wildtype_seq, db)

    if use_cache and key:
        _save_cache(key, result)

    return result
