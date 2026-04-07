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

def _masked_marginal_ll(sequence: str, tokenizer, model) -> float:
    """
    Compute sum of masked marginal log-likelihoods over all positions.
    For each position i: mask it, compute log P(true_token | all other tokens).
    Sum over all positions gives sequence log-likelihood.
    """
    tokens = tokenizer(sequence, return_tensors="pt")
    input_ids = tokens["input_ids"]          # shape [1, seq_len+2] (+2 for BOS/EOS)
    total_ll = 0.0

    with torch.no_grad():
        for i in range(1, input_ids.shape[1] - 1):   # skip BOS at 0, EOS at -1
            masked_ids = input_ids.clone()
            true_token = masked_ids[0, i].item()
            masked_ids[0, i] = tokenizer.mask_token_id

            logits = model(input_ids=masked_ids).logits    # [1, seq_len+2, vocab]
            log_probs = torch.log_softmax(logits[0, i], dim=-1)
            total_ll += log_probs[true_token].item()

    return total_ll

def score_sequence_esm2(
    wildtype_seq: str,
    variant_seq: str,
    model_name: str = "esm2_t6_8M_UR50D"
) -> ESM2Result:
    if len(wildtype_seq) != len(variant_seq):
        raise ValueError(
            f"Wildtype and variant must be same length. "
            f"Got {len(wildtype_seq)} and {len(variant_seq)}."
        )

    tokenizer, model = _load_model(model_name)

    wt_ll  = _masked_marginal_ll(wildtype_seq, tokenizer, model)
    var_ll = _masked_marginal_ll(variant_seq,  tokenizer, model)

    mutated_positions = [
        i for i, (w, v) in enumerate(zip(wildtype_seq, variant_seq)) if w != v
    ]

    return ESM2Result(
        llr=var_ll - wt_ll,
        wildtype_ll=wt_ll,
        variant_ll=var_ll,
        mutated_positions=mutated_positions,
        n_mutations=len(mutated_positions)
    )
