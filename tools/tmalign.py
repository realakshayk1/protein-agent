import subprocess
import tempfile
import os
import re
import shutil
from tools.interface import TMAlignResult

def compute_structural_similarity(predicted_pdb: str, reference_pdb_path: str) -> TMAlignResult:
    """
    Compute TM-score and RMSD between a predicted PDB string and a reference PDB file.
    
    Args:
        predicted_pdb: The PDB string of the predicted structure.
        reference_pdb_path: Path to the reference PDB file.
        
    Returns:
        TMAlignResult containing the TM-score and RMSD.
    """
    if not os.path.exists(reference_pdb_path):
        raise FileNotFoundError(f"Reference PDB not found at {reference_pdb_path}")
        
    binary = shutil.which("TMalign") or shutil.which("TMalign.exe")
    if not binary:
        raise FileNotFoundError("TMalign binary not found on PATH. Please install it.")

    fd, temp_pred_path = tempfile.mkstemp(suffix=".pdb")
    with os.fdopen(fd, 'w') as f:
        f.write(predicted_pdb)

    try:
        result = subprocess.run(
            [binary, temp_pred_path, reference_pdb_path],
            capture_output=True, text=True, timeout=30
        )
        
        output = result.stdout
        
        tm_score_matches = re.findall(r"TM-score=\s*([\d.]+)", output)
        rmsd_match = re.search(r"RMSD=\s*([\d.]+)", output)
        
        tm_score = float(tm_score_matches[0]) if tm_score_matches else 0.0
        rmsd = float(rmsd_match.group(1)) if rmsd_match else None
        
        return TMAlignResult(
            tm_score=tm_score,
            rmsd=rmsd
        )
    finally:
        if os.path.exists(temp_pred_path):
            os.remove(temp_pred_path)
