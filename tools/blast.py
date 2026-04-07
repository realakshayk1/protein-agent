from Bio.Blast import NCBIWWW, NCBIXML
from tools.interface import BLASTResult

def blast_conservation(sequence: str, mutated_positions: list[int], db: str = "swissprot") -> BLASTResult:
    """
    Run BLAST on a sequence to assess evolutionary conservation at mutated positions.
    
    Args:
        sequence: The protein sequence to query.
        mutated_positions: A list of 0-indexed positions where mutations occurred.
        db: The NCBI BLAST database to use.
        
    Returns:
        BLASTResult containing the mean conservation score across the mutated positions.
    """
    if not mutated_positions:
        raise ValueError("mutated_positions cannot be empty")
        
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    if not (10 <= len(sequence) <= 1000):
        raise ValueError("Sequence length must be between 10 and 1000 residues")
        
    if not all(c.upper() in valid_aa for c in sequence):
        raise ValueError("Sequence contains invalid amino acid characters")

    try:
        # Run qblast without extra retries
        result_handle = NCBIWWW.qblast(
            "blastp", 
            db, 
            sequence.upper(),
            hitlist_size=50
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
        return BLASTResult(
            conservation_score=0.0,
            mutated_positions=mutated_positions,
            n_hits=0
        )
    
    total_conservation = 0.0
    
    for pos in mutated_positions:
        query_aa = sequence[pos].upper()
        matches = 0
        valid_hits_for_pos = 0
        
        for alignment in alignments[:50]:
            hsp = alignment.hsps[0] # taking the best HSP for the alignment
            
            q_start = hsp.query_start - 1 # 0-indexed start
            q_end = q_start + len(hsp.query.replace("-", ""))
            
            if q_start <= pos < q_end:
                # Find corresponding index in the alignment string accounting for gaps
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
            total_conservation += (matches / valid_hits_for_pos)
            
    mean_score = total_conservation / len(mutated_positions) if mutated_positions else 0.0
    
    return BLASTResult(
        conservation_score=mean_score,
        mutated_positions=mutated_positions,
        n_hits=n_hits
    )
