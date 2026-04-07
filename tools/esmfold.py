import requests
from tools.interface import ESMFoldResult

ESMFOLD_API_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"

def predict_structure_esmfold(
    sequence: str,
    candidate_id: str,
    timeout: int = 120
) -> ESMFoldResult:
    if not sequence.isalpha():
        raise ValueError(f"Sequence contains non-alphabetic characters: {sequence}")
    if len(sequence) > 400:
        raise ValueError(f"Sequence too long for ESMFold API: {len(sequence)} residues (max 400)")

    response = requests.post(
        ESMFOLD_API_URL,
        data=sequence,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=timeout
    )
    response.raise_for_status()
    pdb_string = response.text

    plddt_scores = _extract_plddt(pdb_string)
    if not plddt_scores:
        raise ValueError(f"Could not extract pLDDT scores from ESMFold response for {candidate_id}")

    return ESMFoldResult(
        candidate_id=candidate_id,
        pdb_string=pdb_string,
        mean_plddt=sum(plddt_scores) / len(plddt_scores),
        plddt_per_residue=plddt_scores,
        length=len(sequence)
    )

def _extract_plddt(pdb_string: str) -> list[float]:
    """
    Extract per-residue pLDDT from B-factor column of ATOM records.
    ESMFold encodes pLDDT (0-100) in the B-factor field of CA atoms.
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
