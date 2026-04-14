# tools/esm2.py
import torch
from transformers import EsmTokenizer, EsmForMaskedLM
from tools.interface import ESM2Result

_MODEL_CACHE: dict = {}


def _load_model(model_name: str):
    """Load model once, cache in memory for subsequent calls."""
    if model_name not in _MODEL_CACHE:
        tokenizer = EsmTokenizer.from_pretrained(f"facebook/{model_name}")
        model = EsmForMaskedLM.from_pretrained(f"facebook/{model_name}")
        model.eval()
        _MODEL_CACHE[model_name] = (tokenizer, model)
    return _MODEL_CACHE[model_name]


def _focused_mutant_marginal_llr(
    wt_seq: str,
    var_seq: str,
    mutated_positions: list[int],
    tokenizer,
    model,
) -> float:
    """
    Focused mutant-marginal LLR: score only at mutated positions.

    For each mutated position p:
      - Mask p in the *variant* context → log P(var_aa | variant_neighbours)
      - Mask p in the *wildtype* context  → log P(wt_aa  | wildtype_neighbours)
      - Contribution = log P(var_aa | var_ctx) − log P(wt_aa | wt_ctx)

    Sum across all mutated positions gives the LLR.

    This is the standard "mutant-marginal" approach (Meier et al. 2021) and is
    dramatically faster than whole-sequence marginals: 2 forward passes per
    mutation site vs. 2 × sequence_length passes previously.
    """
    wt_tokens  = tokenizer(wt_seq,  return_tensors="pt")["input_ids"]  # [1, L+2]
    var_tokens = tokenizer(var_seq, return_tensors="pt")["input_ids"]

    total_llr = 0.0
    with torch.no_grad():
        for pos in mutated_positions:
            tidx = pos + 1  # +1 for BOS token

            # --- variant AA in variant context ---
            var_masked = var_tokens.clone()
            var_masked[0, tidx] = tokenizer.mask_token_id
            var_log_probs = torch.log_softmax(
                model(input_ids=var_masked).logits[0, tidx], dim=-1
            )
            var_aa_id = var_tokens[0, tidx].item()

            # --- wildtype AA in wildtype context ---
            wt_masked = wt_tokens.clone()
            wt_masked[0, tidx] = tokenizer.mask_token_id
            wt_log_probs = torch.log_softmax(
                model(input_ids=wt_masked).logits[0, tidx], dim=-1
            )
            wt_aa_id = wt_tokens[0, tidx].item()

            total_llr += var_log_probs[var_aa_id].item() - wt_log_probs[wt_aa_id].item()

    return total_llr


def score_sequence_esm2(
    wildtype_seq: str,
    variant_seq: str,
    model_name: str = "esm2_t12_35M_UR50D",
    domain_slice: tuple[int, int] | None = None,
) -> ESM2Result:
    """
    Score a protein variant relative to its wildtype via focused mutant-marginal LLR.

    Args:
        wildtype_seq:  Full wildtype amino-acid sequence.
        variant_seq:   Full variant amino-acid sequence (same length as wildtype).
        model_name:    HuggingFace ESM-2 (or ESM-1v) model identifier.
                       e.g. "esm2_t12_35M_UR50D", "esm2_t33_650M_UR50D",
                            "esm1v_t33_650M_UR90S_1"
        domain_slice:  Optional (start, end) tuple (0-indexed, end exclusive) to
                       restrict scoring to a specific domain.  Both sequences are
                       truncated to this slice before being passed to the model,
                       which keeps the evolutionary context clean when the full
                       sequence contains non-natural scaffold / linker residues.
                       Example: (0, 56) for the 56-aa GB1 domain embedded in a
                       longer fusion construct.

    Returns:
        ESM2Result.  mutated_positions and n_mutations are always reported in the
        original (full-sequence) coordinate frame so downstream tools receive
        correct indices regardless of domain_slice.
    """
    if len(wildtype_seq) != len(variant_seq):
        raise ValueError(
            f"Wildtype and variant must be same length. "
            f"Got {len(wildtype_seq)} and {len(variant_seq)}."
        )

    # Identify mutated positions in full-sequence coordinates
    mutated_positions_global = [
        i for i, (w, v) in enumerate(zip(wildtype_seq, variant_seq)) if w != v
    ]

    # Apply domain slice: truncate sequences and remap positions
    if domain_slice is not None:
        ds_start, ds_end = domain_slice
        wt_scoring  = wildtype_seq[ds_start:ds_end]
        var_scoring = variant_seq[ds_start:ds_end]
        # Only score mutations that fall within the domain
        mutated_positions_local = [
            p - ds_start
            for p in mutated_positions_global
            if ds_start <= p < ds_end
        ]
    else:
        wt_scoring  = wildtype_seq
        var_scoring = variant_seq
        mutated_positions_local = mutated_positions_global

    if not mutated_positions_local:
        # No mutations in the scoring domain — return zero LLR
        return ESM2Result(
            llr=0.0,
            wildtype_ll=0.0,
            variant_ll=0.0,
            mutated_positions=mutated_positions_global,
            n_mutations=len(mutated_positions_global),
        )

    tokenizer, model = _load_model(model_name)

    llr = _focused_mutant_marginal_llr(
        wt_scoring, var_scoring, mutated_positions_local, tokenizer, model
    )

    return ESM2Result(
        llr=llr,
        wildtype_ll=0.0,   # not separately tracked in focused mode
        variant_ll=0.0,
        mutated_positions=mutated_positions_global,
        n_mutations=len(mutated_positions_global),
    )
