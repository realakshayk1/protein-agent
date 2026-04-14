import subprocess
import tempfile
import os
import re
import shutil
from tools.interface import TMAlignResult


def compute_structural_similarity(
    predicted_pdb: str,
    reference_pdb_path: str,
) -> TMAlignResult:
    """
    Compute TM-score and RMSD between a predicted PDB string and a reference PDB file.

    TMalign outputs two TM-score lines:
      Line 0 — normalized by Chain 1 length (the mobile / predicted structure)
      Line 1 — normalized by Chain 2 length (the reference / template structure)

    We take Line 1 (reference-normalized) because it directly answers the
    question "what fraction of the reference fold is reproduced?", which is the
    standard convention and is the meaningful quantity when the two structures
    differ in length (e.g. a long fusion construct vs. a short domain PDB).

    Args:
        predicted_pdb:     PDB-format string of the predicted variant structure.
        reference_pdb_path: Path to the reference PDB file (e.g. wildtype crystal).

    Returns:
        TMAlignResult with reference-normalized tm_score and RMSD (Å).
    """
    if not os.path.exists(reference_pdb_path):
        raise FileNotFoundError(f"Reference PDB not found at {reference_pdb_path}")

    binary = shutil.which("TMalign") or shutil.which("TMalign.exe")
    if not binary:
        raise FileNotFoundError("TMalign binary not found on PATH. Please install it.")

    fd, temp_pred_path = tempfile.mkstemp(suffix=".pdb")
    with os.fdopen(fd, "w") as f:
        f.write(predicted_pdb)

    try:
        result = subprocess.run(
            [binary, temp_pred_path, reference_pdb_path],
            capture_output=True,
            text=True,
            timeout=30,
        )

        output = result.stdout

        # TMalign emits exactly two "TM-score=" lines.
        # Index 0 → normalized by mobile (Chain 1) length
        # Index 1 → normalized by reference (Chain 2) length  ← we want this
        tm_score_matches = re.findall(r"TM-score=\s*([\d.]+)", output)
        rmsd_match = re.search(r"RMSD=\s*([\d.]+)", output)

        if len(tm_score_matches) >= 2:
            # Reference-normalized TM-score (index 1 = normalized by Chain 2 / reference length)
            tm_score = float(tm_score_matches[1])
        elif tm_score_matches:
            tm_score = float(tm_score_matches[0])
        else:
            tm_score = 0.0

        rmsd = float(rmsd_match.group(1)) if rmsd_match else None

        return TMAlignResult(tm_score=tm_score, rmsd=rmsd)

    finally:
        if os.path.exists(temp_pred_path):
            os.remove(temp_pred_path)
