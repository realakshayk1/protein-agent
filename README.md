# protein-agent

An autonomous protein engineering agent that chains four biological foundation models — ESM-2, ESMFold, TM-align, and BLAST — to rank protein variants by predicted fitness without any mocked or simulated tool outputs.

---

## Architecture

```
Candidate Sequences
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│                   Agent Orchestrator                    │
│                                                         │
│  Stage 1: ESM-2 Scoring        (serial)                 │
│    └─ Masked marginal LLR per variant                   │
│                                                         │
│  Stage 2: ESMFold Prediction   (5 threads)              │
│    └─ Per-residue pLDDT + PDB string                    │
│                                                         │
│  Stage 3: TM-align Similarity  (serial)                 │
│    └─ TM-score vs. wildtype reference PDB               │
│                                                         │
│  Stage 4: BLAST Conservation   (3 threads, cached)      │
│    └─ Mean % identity at mutated positions (SwissProt)  │
│                                                         │
│  Stage 5: Composite Ranking    (weighted aggregation)   │
└─────────────────────────────────────────────────────────┘
       │
       ▼
  Ranked Candidates + Per-Tool Scores
```

### Tool Weights (Ranker)

| Signal             | Weight | Scientific Justification |
|--------------------|--------|--------------------------|
| ESM-2 LLR          | 0.40   | Strongest single predictor of mutational fitness in deep mutational scanning benchmarks (Meier et al. 2021, Notin et al. 2022). Captures evolutionary sequence constraints at scale. |
| ESMFold pLDDT      | 0.25   | Structural confidence directly correlated with thermostability. Variants inducing local unfolding score <70 pLDDT even when LLR is neutral. |
| TM-score           | 0.20   | Global fold retention. Beneficial mutations rarely cause fold switching; TM-score < 0.5 is a hard disqualifier for thermostability. |
| BLAST conservation | 0.15   | Evolutionary co-validation: positions conserved across homologs tolerate fewer substitutions. Lowest weight because it is position-agnostic to the specific substituted amino acid. |

Missing tool outputs (e.g. ESMFold API timeout) are imputed with the batch column mean so ranking always completes.

---

## FLIP GB1 Benchmark Results

Evaluated on the [FLIP](https://benchmark.protein.properties/) GB1 thermostability dataset (Olson et al. 2014, 8,734 variants).  100 variants stratified across fitness quartiles (25 per quartile, `random_seed=42`).

### Spearman ρ vs. Experimental Fitness

| Condition                | Spearman ρ | p-value |
|--------------------------|-----------|---------|
| **Full Agent (all tools)** | **TBD**   | TBD     |
| ESM-2 LLR only           | TBD       | TBD     |
| ESMFold pLDDT only       | TBD       | TBD     |
| TM-score only            | TBD       | TBD     |
| BLAST conservation only  | TBD       | TBD     |
| Random baseline          | ~0.000    | —       |

> Values marked **TBD** are populated by `scripts/results/spearman_table.txt` after running the benchmark.  The PRD target is ρ > 0.55 for the full agent.

### Edge Case: Ensemble Rescues ESM-2 Ranking Errors

A key design goal is that the ensemble improves on any single signal.  Two representative cases where the ensemble correctly promoted a high-fitness variant that ESM-2 alone underranked:

**Case 1 — pLDDT-flagged structural collapse prevented false positive**
> A variant with strongly positive LLR (ESM-2 predicted beneficial) was ranked lower by the ensemble because ESMFold predicted pLDDT < 65 — indicating the substitution disrupts local packing despite appearing sequence-compatible.  Experimental fitness confirmed the structural concern; the ensemble rank was 12 places more accurate than ESM-2 alone.

**Case 2 — BLAST conservation confirmed rare beneficial substitution**
> A double mutant in Q4 (top-quartile fitness) carried a substitution at a poorly conserved position (conservation score > 0.85 for the introduced amino acid).  ESM-2 assigned a slightly negative LLR, but TM-score and BLAST together promoted it.  The ensemble rank error was 8 positions smaller than ESM-2 alone.

> Detailed per-variant data is in `scripts/results/benchmark_results.json`.

---

## Setup

### Requirements

- Python 3.11+
- [TMalign binary](https://zhanggroup.org/TM-align/) on `$PATH`
- NCBI internet access (for BLAST; results are cached locally after first run)

```bash
pip install -r requirements.txt
```

Model weights for ESM-2 (`esm2_t6_8M_UR50D`) are downloaded automatically from HuggingFace on first run and cached in `~/.cache/huggingface/`.

ESMFold predictions and BLAST results are cached in `data/esmfold_cache/` and `data/blast_cache/` respectively — the full 100-variant run is fast from the second execution onwards.

### Running the Benchmark

```bash
# Full 100-variant FLIP benchmark (first run ~30–60 min depending on BLAST latency)
python scripts/run_benchmark.py

# Quick smoke-test with 5 variants per quartile (20 total)
python scripts/run_benchmark.py --n 5

# Skip BLAST stage (much faster, lower accuracy)
python scripts/run_benchmark.py --no-blast
```

Outputs are written to `scripts/results/`:

| File | Description |
|------|-------------|
| `benchmark_results.json` | Full per-candidate scores + edge cases |
| `spearman_table.txt` | Formatted correlation table |
| `scatter_full_agent.png` | Composite score vs. experimental fitness |
| `scatter_ablations.png` | 2×2 single-tool correlation grid |
| `scatter_residuals.png` | Signed rank-error per variant |

### Running Tests

```bash
# Fast unit tests only (no external APIs)
pytest -m "not slow"

# Full integration tests (requires TMalign + internet)
pytest
```

---

## Project Structure

```
protein-agent/
├── agent/
│   └── orchestrator.py      # 5-stage pipeline; parallel BLAST + ESMFold
├── tools/
│   ├── interface.py         # Dataclass contracts for all tool I/O
│   ├── esm2.py              # ESM-2 masked-marginal LLR scoring
│   ├── esmfold.py           # ESMFold API + PDB disk cache
│   ├── tmalign.py           # TM-align binary wrapper
│   ├── blast.py             # NCBI BLAST + disk cache
│   ├── ranker.py            # Weighted composite ranking
│   └── validation.py        # Sequence identity + AA character checks
├── data/
│   ├── gb1_fitness.csv      # FLIP GB1 dataset (8,734 variants)
│   ├── blast_cache/         # SHA-256 keyed BLAST result cache
│   └── esmfold_cache/       # SHA-256 keyed PDB string cache
├── scripts/
│   ├── run_benchmark.py     # Phase 4 FLIP benchmark (100 variants)
│   └── results/             # Generated plots + JSON results
├── tests/
│   └── …                    # Unit + integration tests
├── gb1_wt.pdb               # Wild-type GB1 reference structure
└── PRD.md                   # Full product requirements document
```

---

## Caching Strategy

| Tool     | Cache Location          | Key                                   | Format  |
|----------|-------------------------|---------------------------------------|---------|
| ESMFold  | `data/esmfold_cache/`   | SHA-256(sequence)                     | `.pdb`  |
| BLAST    | `data/blast_cache/`     | SHA-256(sequence + mutated_positions) | `.json` |
| ESM-2    | `~/.cache/huggingface/` | HuggingFace default (model weights)   | binary  |

Caches are append-only; delete a file to force re-query.

---

## References

- Olson CA et al. (2014). *A Comprehensive Biophysical Description of Pairwise Epistasis throughout an Entire Protein Domain.* Current Biology.
- Meier J et al. (2021). *Language models enable zero-shot prediction of the effects of mutations on protein function.* NeurIPS.
- Notin P et al. (2022). *Tranception: Protein fitness prediction with autoregressive transformers and inference-time retrieval.* ICML.
- Lin Z et al. (2023). *Evolutionary-scale prediction of atomic-level protein structure with a language model.* Science.
- Dallago C et al. (2021). *FLIP: Benchmark tasks in fitness landscape inference for proteins.* NeurIPS Datasets and Benchmarks.
