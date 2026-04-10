# tools/esm2.py
import hashlib
import json
import os

import torch
from transformers import EsmTokenizer, EsmForMaskedLM
from tools.interface import ESM2Result

# ---------------------------------------------------------------------------
# In-memory model cache — loaded once per process
# ---------------------------------------------------------------------------
_MODEL_CACHE: dict = {}

# ---------------------------------------------------------------------------
# Disk cache — avoids re-scoring identical (model, wt, variant) triples.
# Keyed by SHA-256 of "model_name:wt_seq:var_seq".  Results are ~instant on
# repeat runs even for the 650M model.
# ---------------------------------------------------------------------------
_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "esm2_cache")


def _cache_key(model_name: str, wildtype_seq: str, variant_seq: str) -> str:
    payload = f"{model_name}:{wildtype_seq.upper()}:{variant_seq.upper()}"
    return hashlib.sha256(payload.encode()).hexdigest()


def _load_cache(key: str) -> ESM2Result | None:
    path = os.path.join(_CACHE_DIR, f"{key}.json")
    if os.path.exists(path):
        try:
            with open(path) as fh:
                d = json.load(fh)
            return ESM2Result(
                llr=d["llr"],
                wildtype_ll=d["wildtype_ll"],
                variant_ll=d["variant_ll"],
                mutated_positions=d["mutated_positions"],
                n_mutations=d["n_mutations"],
            )
        except Exception:
            return None
    return None


def _save_cache(key: str, result: ESM2Result) -> None:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    path = os.path.join(_CACHE_DIR, f"{key}.json")
    try:
        with open(path, "w") as fh:
            json.dump(
                {
                    "llr": result.llr,
                    "wildtype_ll": result.wildtype_ll,
                    "variant_ll": result.variant_ll,
                    "mutated_positions": result.mutated_positions,
                    "n_mutations": result.n_mutations,
                },
                fh,
            )
    except Exception:
        pass  # non-fatal


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model(model_name: str):
    """Load model once, cache in memory for subsequent calls."""
    if model_name not in _MODEL_CACHE:
        tokenizer = EsmTokenizer.from_pretrained(f"facebook/{model_name}")
        model = EsmForMaskedLM.from_pretrained(f"facebook/{model_name}")
        model.eval()
        _MODEL_CACHE[model_name] = (tokenizer, model)
    return _MODEL_CACHE[model_name]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _focused_masked_marginal_llr(
    wildtype_seq: str,
    variant_seq: str,
    mutated_positions: list[int],
    tokenizer,
    model,
) -> tuple[float, float, float]:
    """
    Focused masked marginal LLR — only scores mutated positions.

    For each mutated position i:
      - Mask position i in the variant, score log P(variant_aa_i | variant_context)
      - Mask position i in the wildtype, score log P(wt_aa_i | wt_context)
      - Contribution: log P(variant) - log P(wt)

    Summing over mutated positions gives the LLR, which concentrates signal
    on the positions that actually differ and avoids dilution from the many
    identical positions that would otherwise dominate a full-sequence LL.

    Returns (llr, variant_ll, wildtype_ll) where the LLs are partial sums
    over mutated positions only.
    """
    wt_tokens = tokenizer(wildtype_seq, return_tensors="pt")["input_ids"]
    var_tokens = tokenizer(variant_seq, return_tensors="pt")["input_ids"]

    variant_ll = 0.0
    wildtype_ll = 0.0

    with torch.no_grad():
        for pos in mutated_positions:
            token_idx = pos + 1  # +1 to skip BOS token at index 0

            # Score variant AA in variant context
            var_masked = var_tokens.clone()
            var_aa_token = int(var_masked[0, token_idx].item())
            var_masked[0, token_idx] = tokenizer.mask_token_id
            logits_var = model(input_ids=var_masked).logits
            log_probs_var = torch.log_softmax(logits_var[0, token_idx], dim=-1)
            variant_ll += log_probs_var[var_aa_token].item()

            # Score WT AA in WT context
            wt_masked = wt_tokens.clone()
            wt_aa_token = int(wt_masked[0, token_idx].item())
            wt_masked[0, token_idx] = tokenizer.mask_token_id
            logits_wt = model(input_ids=wt_masked).logits
            log_probs_wt = torch.log_softmax(logits_wt[0, token_idx], dim=-1)
            wildtype_ll += log_probs_wt[wt_aa_token].item()

    llr = variant_ll - wildtype_ll
    return llr, variant_ll, wildtype_ll


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_sequence_esm2(
    wildtype_seq: str,
    variant_seq: str,
    model_name: str = "esm2_t33_650M_UR50D",
    use_cache: bool = True,
) -> ESM2Result:
    """
    Score a variant sequence relative to the wildtype using ESM-2.

    Uses focused masked marginal scoring: for each mutated position i, the
    model masks that position and scores log P(aa_i | context).  The LLR is
    the sum of per-position score differences (variant - wildtype), which
    concentrates signal on the mutations themselves.

    Results are cached to data/esm2_cache/ so the 650M model forward passes
    only run once per unique (model, wt, variant) triple.

    Args:
        wildtype_seq: Reference amino acid sequence.
        variant_seq:  Mutant sequence (same length as wildtype).
        model_name:   HuggingFace ESM-2 model ID (default: 650M parameter model).
        use_cache:    Read/write disk cache (default True).

    Returns:
        ESM2Result with LLR and mutated position metadata.
    """
    if len(wildtype_seq) != len(variant_seq):
        raise ValueError(
            f"Wildtype and variant must be same length. "
            f"Got {len(wildtype_seq)} and {len(variant_seq)}."
        )

    mutated_positions = [
        i for i, (w, v) in enumerate(zip(wildtype_seq, variant_seq)) if w != v
    ]

    # Identical sequences → LLR = 0 by definition
    if not mutated_positions:
        return ESM2Result(
            llr=0.0,
            wildtype_ll=0.0,
            variant_ll=0.0,
            mutated_positions=[],
            n_mutations=0,
        )

    # Cache read
    key = _cache_key(model_name, wildtype_seq, variant_seq) if use_cache else None
    if use_cache and key:
        cached = _load_cache(key)
        if cached is not None:
            return cached

    tokenizer, model = _load_model(model_name)

    llr, var_ll, wt_ll = _focused_masked_marginal_llr(
        wildtype_seq, variant_seq, mutated_positions, tokenizer, model
    )

    result = ESM2Result(
        llr=llr,
        wildtype_ll=wt_ll,
        variant_ll=var_ll,
        mutated_positions=mutated_positions,
        n_mutations=len(mutated_positions),
    )

    if use_cache and key:
        _save_cache(key, result)

    return result
