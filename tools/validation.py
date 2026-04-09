"""
Scientific-grade input validation for the protein engineering agent.

Implements sequence identity checks, length constraints, and amino acid
character enforcement per T5.4 of the PRD.
"""
from __future__ import annotations

VALID_AAS = set("ACDEFGHIKLMNPQRSTVWY")
MIN_SEQ_LEN = 10
MAX_SEQ_LEN = 400  # ESMFold API hard limit


def validate_sequence(seq: str, candidate_id: str = "unknown") -> None:
    """
    Validate a protein sequence for use in the agent pipeline.

    Raises ValueError with a descriptive message on any violation.
    """
    if not isinstance(seq, str) or len(seq) == 0:
        raise ValueError(f"[{candidate_id}] sequence must be a non-empty string")

    if not (MIN_SEQ_LEN <= len(seq) <= MAX_SEQ_LEN):
        raise ValueError(
            f"[{candidate_id}] length {len(seq)} is outside the valid range "
            f"[{MIN_SEQ_LEN}, {MAX_SEQ_LEN}] (ESMFold API limit)"
        )

    invalid_chars = set(seq.upper()) - VALID_AAS
    if invalid_chars:
        raise ValueError(
            f"[{candidate_id}] sequence contains invalid amino acid characters: "
            f"{sorted(invalid_chars)}.  Only standard 20 AAs are accepted."
        )


def sequence_identity(seq_a: str, seq_b: str, candidate_id: str = "unknown") -> float:
    """
    Compute fractional sequence identity between two equal-length sequences.

    Returns a value in [0, 1] where 1.0 means identical.

    Raises ValueError if lengths differ.
    """
    if len(seq_a) != len(seq_b):
        raise ValueError(
            f"[{candidate_id}] length mismatch: {len(seq_a)} vs {len(seq_b)}"
        )
    if len(seq_a) == 0:
        return 1.0
    matches = sum(a == b for a, b in zip(seq_a.upper(), seq_b.upper()))
    return matches / len(seq_a)


def assert_not_wildtype(
    variant_seq: str, wildtype_seq: str, candidate_id: str = "unknown"
) -> None:
    """
    Raise ValueError if the variant is identical to the wildtype.

    Running the full pipeline on a wildtype-identical sequence produces a
    trivially zero LLR and pollutes ranking statistics.
    """
    if variant_seq.upper() == wildtype_seq.upper():
        raise ValueError(
            f"[{candidate_id}] variant sequence is identical to the wildtype.  "
            "Exclude the wildtype from candidate sets before calling run_agent()."
        )


def mutation_count(variant_seq: str, wildtype_seq: str) -> int:
    """Return the number of single-residue substitutions relative to wildtype."""
    if len(variant_seq) != len(wildtype_seq):
        raise ValueError("Sequences must have equal length to count mutations.")
    return sum(v.upper() != w.upper() for v, w in zip(variant_seq, wildtype_seq))
