import hashlib
import os
import requests
from tools.interface import ESMFoldResult

ESMFOLD_API_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"

# ---------------------------------------------------------------------------
# Disk cache – avoids redundant ESMFold API calls for identical sequences.
# PDB strings are cached as plain text files keyed by SHA-256 of the sequence.
# ---------------------------------------------------------------------------
_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "esmfold_cache")


def _seq_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.upper().encode()).hexdigest()


def _load_pdb_cache(seq_hash: str) -> str | None:
    path = os.path.join(_CACHE_DIR, f"{seq_hash}.pdb")
    if os.path.exists(path):
        try:
            with open(path) as fh:
                return fh.read()
        except Exception:
            return None
    return None


def _save_pdb_cache(seq_hash: str, pdb_string: str) -> None:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    path = os.path.join(_CACHE_DIR, f"{seq_hash}.pdb")
    try:
        with open(path, "w") as fh:
            fh.write(pdb_string)
    except Exception:
        pass  # non-fatal


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_structure_esmfold(
    sequence: str,
    candidate_id: str,
    timeout: int = 120,
    use_cache: bool = True,
) -> ESMFoldResult:
    """
    Predict protein structure via the ESMFold API.

    PDB outputs are cached to disk (data/esmfold_cache/) so repeated calls
    for the same sequence are instant and the pipeline works offline after
    a warm run.

    Args:
        sequence: Protein amino acid sequence (max 400 residues).
        candidate_id: Identifier attached to the returned result.
        timeout: HTTP request timeout in seconds.
        use_cache: Whether to read/write the PDB disk cache (default True).

    Returns:
        ESMFoldResult with per-residue pLDDT scores and PDB string.
    """
    if not sequence.isalpha():
        raise ValueError(f"Sequence contains non-alphabetic characters: {sequence}")
    if len(sequence) > 400:
        raise ValueError(
            f"Sequence too long for ESMFold API: {len(sequence)} residues (max 400)"
        )

    # --- cache read ---
    seq_hash = _seq_hash(sequence)
    pdb_string: str | None = None

    if use_cache:
        pdb_string = _load_pdb_cache(seq_hash)

    if pdb_string is None:
        # --- remote API call ---
        response = requests.post(
            ESMFOLD_API_URL,
            data=sequence,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=timeout,
        )
        response.raise_for_status()
        pdb_string = response.text

        if use_cache:
            _save_pdb_cache(seq_hash, pdb_string)

    plddt_scores = _extract_plddt(pdb_string)
    if not plddt_scores:
        raise ValueError(
            f"Could not extract pLDDT scores from ESMFold response for {candidate_id}"
        )

    if all(p <= 1.0 for p in plddt_scores):
        plddt_scores = [p * 100.0 for p in plddt_scores]

    return ESMFoldResult(
        candidate_id=candidate_id,
        pdb_string=pdb_string,
        mean_plddt=sum(plddt_scores) / len(plddt_scores),
        plddt_per_residue=plddt_scores,
        length=len(sequence),
    )


def _extract_plddt(pdb_string: str) -> list[float]:
    """
    Extract per-residue pLDDT from the B-factor column of ATOM records.

    ESMFold encodes pLDDT (0–100) in the B-factor field of CA atoms.
    PDB format: columns 61-66 are B-factor (1-indexed).
    """
    scores = []
    for line in pdb_string.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            try:
                scores.append(float(line[60:66].strip()))
            except ValueError:
                continue
    return scores
