"""
Phase 4 FLIP GB1 Benchmark
==========================
Evaluates the protein-engineering agent on 100 GB1 variants stratified across
fitness quartiles.  Outputs:

  scripts/results/benchmark_results.json   -- raw scores + metadata
  scripts/results/scatter_full_agent.png   -- composite score vs experimental fitness
  scripts/results/scatter_ablations.png    -- 2x2 grid: single-tool correlations
  scripts/results/scatter_residuals.png    -- signed residual plot for edge-case ID
  scripts/results/spearman_table.txt       -- formatted correlation table

After a successful run the Spearman table in README.md is updated automatically.

Usage (from repo root):
    python scripts/run_benchmark.py                        # full 100-variant run
    python scripts/run_benchmark.py --n 20                 # quick smoke-test
    python scripts/run_benchmark.py --no-blast             # skip BLAST (faster)
    python scripts/run_benchmark.py --model esm2_t6_8M_UR50D  # override ESM-2 model
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Ensure project root on path before local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.orchestrator import run_agent
from tools.blast import local_blast_available
from tools.validation import mutation_count, sequence_identity

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join("data", "gb1_fitness.csv")
REFERENCE_PDB = "gb1_wt.pdb"
README_PATH = os.path.join(os.path.dirname(__file__), "..", "README.md")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
RANDOM_SEED = 42
TARGET_PER_QUARTILE = 25      # 4 x 25 = 100 total variants
# PRD specifies esm2_t12_35M for the production benchmark (vs. 8M for dev/tests)
DEFAULT_ESM2_MODEL = "esm2_t12_35M_UR50D"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_spearman(x: list[float], y: list[float]) -> tuple[float, float]:
    """Return (rho, p_value), returning (0.0, 1.0) if either series is constant."""
    if np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return 0.0, 1.0
    rho, p = spearmanr(x, y)
    return float(rho), float(p)


def callback(event: dict) -> None:
    etype = event.get("type")
    if etype == "stage_complete":
        stage = event.get("stage_name", "?")
        dur = event.get("duration_seconds", 0)
        print(f"  [Stage complete] {stage} — {dur:.1f}s", flush=True)
    elif etype == "tool_call":
        tool = event.get("tool", "?")
        cid = event.get("candidate_id", "?")
        print(f"    → {tool}({cid})", flush=True)


def load_and_sample(data_path: str, n_per_quartile: int) -> tuple[pd.DataFrame, str]:
    """
    Load the GB1 dataset, detect the wildtype sequence, and return a
    stratified sample of *n_per_quartile* variants per fitness quartile.

    Sequences that are identical to the wildtype are excluded.
    """
    df = pd.read_csv(data_path)
    if "target" in df.columns and "fitness" not in df.columns:
        df = df.rename(columns={"target": "fitness"})

    # Wildtype = sequence whose fitness is closest to 1.0
    wt_idx = (df["fitness"] - 1.0).abs().argsort().iloc[0]
    wt_seq: str = df.loc[wt_idx, "sequence"]

    # Exclude wildtype and any WT-identical sequences
    df_vars = df[df["sequence"] != wt_seq].copy()

    # Exclude sequences with zero mutations just in case (redundant but safe)
    df_vars = df_vars[df_vars["sequence"].apply(lambda s: mutation_count(s, wt_seq) > 0)]

    df_vars["quartile"] = pd.qcut(
        df_vars["fitness"], q=4,
        labels=["Q1", "Q2", "Q3", "Q4"],
        duplicates="drop",
    )

    # Use explicit loop instead of groupby().apply() to avoid pandas version
    # differences where the group-key column gets promoted to index level and
    # then lost on reset_index(drop=True).
    sampled = pd.concat(
        [
            group.sample(n=min(len(group), n_per_quartile), random_state=RANDOM_SEED)
            for _, group in df_vars.groupby("quartile", observed=True)
        ]
    ).reset_index(drop=True)

    return sampled, wt_seq


def build_scatter(
    x: list[float],
    y: list[float],
    xlabel: str,
    ylabel: str,
    title: str,
    rho: float,
    p: float,
    out_path: str,
    highlight_ids: list[int] | None = None,
    highlight_label: str = "edge case",
) -> None:
    """Save a single scatter plot with a Spearman annotation."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  [warn] matplotlib not available – skipping plot {out_path}")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    x_arr = np.array(x)
    y_arr = np.array(y)

    ax.scatter(x_arr, y_arr, alpha=0.7, edgecolors="none", s=40, color="#4C72B0", label="variants")

    if highlight_ids:
        ax.scatter(
            x_arr[highlight_ids], y_arr[highlight_ids],
            s=80, color="#DD4444", zorder=5, label=highlight_label
        )
        ax.legend(fontsize=8)

    # Trend line
    if len(x_arr) > 2:
        m, b = np.polyfit(x_arr, y_arr, 1)
        x_line = np.linspace(x_arr.min(), x_arr.max(), 100)
        ax.plot(x_line, m * x_line + b, "--", linewidth=1, color="#999999")

    p_str = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
    ax.set_title(f"{title}\nSpearman ρ = {rho:.3f}  ({p_str})", fontsize=11)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [plot] saved → {out_path}")


def build_ablation_grid(
    fitness: list[float],
    esm2_scores: list[float],
    plddt_scores: list[float],
    tm_scores: list[float],
    blast_scores: list[float],
    out_path: str,
) -> None:
    """2×2 scatter grid showing each single-tool signal vs experimental fitness."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  [warn] matplotlib not available – skipping {out_path}")
        return

    tools = [
        ("ESM-2 LLR", esm2_scores),
        ("ESMFold pLDDT", plddt_scores),
        ("TM-score", tm_scores),
        ("BLAST conservation", blast_scores),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    y = np.array(fitness)

    for ax, (label, scores) in zip(axes.flat, tools):
        x = np.array(scores)
        rho, p = safe_spearman(x.tolist(), y.tolist())
        ax.scatter(x, y, alpha=0.6, s=25, edgecolors="none", color="#2E86AB")
        if len(x) > 2:
            m, b = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, m * x_line + b, "--", linewidth=1, color="#999999")
        p_str = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
        ax.set_title(f"{label}\nρ = {rho:.3f}  ({p_str})", fontsize=10)
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel("Experimental Fitness", fontsize=9)

    fig.suptitle("Single-Tool Ablations vs. Experimental Fitness (GB1 FLIP)", fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [plot] saved → {out_path}")


def identify_edge_cases(
    ranked_cands: list[dict],
    experimental_fitness: dict[str, float],
    n: int = 3,
) -> list[dict]:
    """
    Find variants where the ensemble ranking diverges most from ESM-2 alone.

    'Ensemble rescues' = cases where ESM-2 ranked a variant low but the
    composite score ranked it correctly (higher fitness → higher rank).
    Returns the top-n such cases with explanatory notes.
    """
    records = []
    for cand in ranked_cands:
        cid = cand["candidate_id"]
        exp_fit = experimental_fitness[cid]
        composite_rank = cand.get("rank", 0)

        # Build a provisional ESM-2-only rank (lower LLR = worse rank)
        records.append({
            "candidate_id": cid,
            "experimental_fitness": exp_fit,
            "composite_rank": composite_rank,
            "llr": cand.get("llr", 0.0),
            "mean_plddt": cand.get("mean_plddt", 0.0),
            "tm_score": cand.get("tm_score", 0.0),
            "conservation_score": cand.get("conservation_score", 0.0),
            "composite_score": cand.get("composite_score", 0.0),
        })

    df = pd.DataFrame(records)
    if df.empty:
        return []

    # ESM-2-only rank: sort by LLR descending; rank 1 = highest LLR
    df = df.sort_values("llr", ascending=False).reset_index(drop=True)
    df["esm2_rank"] = df.index + 1

    # Divergence: |esm2_rank - composite_rank|, we want large values where
    # ensemble rank is better aligned with experimental fitness
    df["rank_divergence"] = df["esm2_rank"] - df["composite_rank"]

    # Sort by experimental fitness descending to find high-fitness cases
    # that ESM-2 under-rated but the ensemble promoted
    top_cases = (
        df[df["rank_divergence"] > 0]
        .sort_values(["experimental_fitness", "rank_divergence"], ascending=[False, False])
        .head(n)
    )

    edge_cases = []
    for _, row in top_cases.iterrows():
        note_parts = []
        if row["mean_plddt"] < 70:
            note_parts.append(f"pLDDT={row['mean_plddt']:.1f} flagged structural concern")
        if row["tm_score"] < 0.6:
            note_parts.append(f"TM-score={row['tm_score']:.3f} indicates fold deviation")
        if row["conservation_score"] > 0.8:
            note_parts.append(f"BLAST conservation={row['conservation_score']:.2f} supports substitution")
        if not note_parts:
            note_parts.append("multi-signal ensemble upgrade")

        edge_cases.append({
            "candidate_id": row["candidate_id"],
            "experimental_fitness": round(row["experimental_fitness"], 4),
            "esm2_rank": int(row["esm2_rank"]),
            "composite_rank": int(row["composite_rank"]),
            "rank_improvement": int(row["rank_divergence"]),
            "llr": round(row["llr"], 4),
            "mean_plddt": round(row["mean_plddt"], 2),
            "tm_score": round(row["tm_score"], 4),
            "conservation_score": round(row["conservation_score"], 4),
            "explanation": "; ".join(note_parts),
        })

    return edge_cases


# ---------------------------------------------------------------------------
# README auto-update
# ---------------------------------------------------------------------------

def update_readme(
    spearman: dict,
    n_variants: int,
    esm2_model: str,
    readme_path: str = README_PATH,
) -> None:
    """
    Replace the TBD placeholder rows in the README benchmark table with real
    Spearman values from the completed run.

    Looks for lines of the form:
        | **Full Agent (all tools)** | **TBD**   | TBD     |
    and replaces with actual numbers.  Safe to call even if the format has
    changed — prints a warning and skips rather than corrupting the file.
    """
    import re

    if not os.path.exists(readme_path):
        print(f"  [warn] README not found at {readme_path} — skipping auto-update")
        return

    with open(readme_path, encoding="utf-8") as fh:
        content = fh.read()

    def _fmt_rho(rho: float) -> str:
        return f"**{rho:.3f}**" if rho >= 0.55 else f"{rho:.3f}"

    def _fmt_p(p: float) -> str:
        return "< 0.001" if p < 0.001 else f"{p:.4f}"

    replacements = [
        # (old_fragment, new_fragment)
        (
            r"\|\s*\*\*Full Agent \(all tools\)\*\*\s*\|\s*\*\*TBD\*\*.*?\|.*?\|",
            f"| **Full Agent (all tools)** | {_fmt_rho(spearman['full_agent']['rho'])} "
            f"| {_fmt_p(spearman['full_agent']['p'])} |",
        ),
        (
            r"\|\s*ESM-2 LLR only\s*\|\s*TBD.*?\|.*?\|",
            f"| ESM-2 LLR only           | {spearman['esm2_only']['rho']:.3f} "
            f"| {_fmt_p(spearman['esm2_only']['p'])} |",
        ),
        (
            r"\|\s*ESMFold pLDDT only\s*\|\s*TBD.*?\|.*?\|",
            f"| ESMFold pLDDT only       | {spearman['plddt_only']['rho']:.3f} "
            f"| {_fmt_p(spearman['plddt_only']['p'])} |",
        ),
        (
            r"\|\s*TM-score only\s*\|\s*TBD.*?\|.*?\|",
            f"| TM-score only            | {spearman['tm_only']['rho']:.3f} "
            f"| {_fmt_p(spearman['tm_only']['p'])} |",
        ),
        (
            r"\|\s*BLAST conservation only\s*\|\s*TBD.*?\|.*?\|",
            f"| BLAST conservation only  | {spearman['blast_only']['rho']:.3f} "
            f"| {_fmt_p(spearman['blast_only']['p'])} |",
        ),
    ]

    updated = content
    changed = 0
    for pattern, replacement in replacements:
        new, n = re.subn(pattern, replacement, updated)
        if n:
            updated = new
            changed += n

    if not changed:
        print("  [warn] README benchmark table not found in expected format — skipping auto-update")
        return

    # Append run metadata note beneath the table if not already present
    meta_note = (
        f"\n> Last benchmark run: {n_variants} variants, "
        f"ESM-2 model `{esm2_model}`, `random_seed={RANDOM_SEED}`.\n"
    )
    if "Last benchmark run:" not in updated:
        # Insert after the last TBD/benchmark table line
        marker = "| Random baseline"
        idx = updated.find(marker)
        if idx != -1:
            end = updated.find("\n", idx) + 1
            updated = updated[:end] + meta_note + updated[end:]

    with open(readme_path, "w", encoding="utf-8") as fh:
        fh.write(updated)
    print(f"  [readme] updated {changed} table rows → {readme_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 4 GB1 FLIP benchmark.")
    parser.add_argument("--n", type=int, default=TARGET_PER_QUARTILE,
                        help=f"Variants per quartile (default {TARGET_PER_QUARTILE} -> 100 total)")
    parser.add_argument("--no-blast", action="store_true",
                        help="Skip BLAST stage (much faster, reduces accuracy)")
    parser.add_argument("--model", default=DEFAULT_ESM2_MODEL,
                        help=f"ESM-2 HuggingFace model name (default: {DEFAULT_ESM2_MODEL})")
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 4 FLIP GB1 Benchmark")
    print("=" * 60)
    print(f"ESM-2 model : {args.model}")
    blast_backend = "skipped" if args.no_blast else ("local BLAST+" if local_blast_available() else "remote NCBI")
    print(f"BLAST backend: {blast_backend}")

    # ------------------------------------------------------------------
    # 1. Load dataset + stratified sampling
    # ------------------------------------------------------------------
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Dataset not found at {DATA_PATH}")
        sys.exit(1)
    if not os.path.exists(REFERENCE_PDB):
        print(f"ERROR: Reference PDB not found at {REFERENCE_PDB}")
        sys.exit(1)

    print(f"\nLoading GB1 dataset: {DATA_PATH}")
    sampled_df, wt_seq = load_and_sample(DATA_PATH, n_per_quartile=args.n)

    print(f"Wild-type sequence ({len(wt_seq)} aa): {wt_seq[:20]}…")
    print(f"Sampled {len(sampled_df)} variants across fitness quartiles:")
    print(sampled_df["quartile"].value_counts().sort_index().to_string())
    print(f"Fitness range:  [{sampled_df['fitness'].min():.3f}, {sampled_df['fitness'].max():.3f}]")

    sequences_dict = {
        f"cand_{i:04d}": row["sequence"]
        for i, row in sampled_df.reset_index(drop=True).iterrows()
    }
    experimental_fitness = {
        f"cand_{i:04d}": row["fitness"]
        for i, row in sampled_df.reset_index(drop=True).iterrows()
    }

    # Validate identities — sanity check
    identities = [sequence_identity(seq, wt_seq) for seq in sequences_dict.values()]
    print(f"\nSequence identity to WT:  mean={np.mean(identities):.3f}  "
          f"min={np.min(identities):.3f}  max={np.max(identities):.3f}")

    # ------------------------------------------------------------------
    # 2. Run agent pipeline
    # ------------------------------------------------------------------
    task = (
        "Given these GB1 variant sequences, rank them by expected thermostability/fitness "
        "using ESM-2 sequence scoring, ESMFold structure prediction, TM-align, and BLAST conservation."
    )
    if args.no_blast:
        print("\n[--no-blast] BLAST stage will be skipped via imputation.")
        task += " (BLAST conservation not available for this run.)"

    print(f"\nStarting agent pipeline on {len(sequences_dict)} variants…")
    wall_start = time.time()

    try:
        agent_result = run_agent(
            task=task,
            sequences=sequences_dict,
            wildtype=wt_seq,
            reference_pdb_path=REFERENCE_PDB,
            stream_callback=callback,
            blast_workers=0 if args.no_blast else 3,
            esm2_model=args.model,
        )
    except Exception as exc:
        print(f"\nAgent pipeline failed: {exc}")
        raise

    wall_time = time.time() - wall_start
    print(f"\nPipeline complete in {wall_time:.1f}s  "
          f"(imputed {agent_result.get('imputed_count', 0)} tool calls)")

    # ------------------------------------------------------------------
    # 3. Collect per-candidate scores
    # ------------------------------------------------------------------
    ranked_cands = agent_result["ranked_candidates"]

    fitness_vals: list[float] = []
    composite_vals: list[float] = []
    esm2_vals: list[float] = []
    plddt_vals: list[float] = []
    tm_vals: list[float] = []
    blast_vals: list[float] = []

    for cand in ranked_cands:
        cid = cand["candidate_id"]
        fitness_vals.append(experimental_fitness[cid])
        composite_vals.append(cand.get("composite_score", 0.0))
        esm2_vals.append(cand.get("llr", 0.0))
        plddt_vals.append(cand.get("mean_plddt", 0.0))
        tm_vals.append(cand.get("tm_score", 0.0))
        blast_vals.append(cand.get("conservation_score", 0.0))

    # ------------------------------------------------------------------
    # 4. Spearman correlations
    # ------------------------------------------------------------------
    rho_full, p_full   = safe_spearman(composite_vals, fitness_vals)
    rho_esm2, p_esm2   = safe_spearman(esm2_vals,      fitness_vals)
    rho_plddt, p_plddt = safe_spearman(plddt_vals,     fitness_vals)
    rho_tm, p_tm       = safe_spearman(tm_vals,        fitness_vals)
    rho_blast, p_blast = safe_spearman(blast_vals,     fitness_vals)

    table_lines = [
        "=" * 55,
        "BENCHMARK RESULTS — Spearman ρ vs. Experimental Fitness",
        f"  Dataset: FLIP GB1  |  N = {len(fitness_vals)} variants",
        "=" * 55,
        f"{'Condition':<28} {'Spearman ρ':>10}  {'p-value':>10}",
        "-" * 55,
        f"{'Full Agent (all 4 tools)':<28} {rho_full:>10.3f}  {p_full:>10.4f}",
        f"{'ESM-2 LLR only':<28} {rho_esm2:>10.3f}  {p_esm2:>10.4f}",
        f"{'ESMFold pLDDT only':<28} {rho_plddt:>10.3f}  {p_plddt:>10.4f}",
        f"{'TM-score only':<28} {rho_tm:>10.3f}  {p_tm:>10.4f}",
        f"{'BLAST conservation only':<28} {rho_blast:>10.3f}  {p_blast:>10.4f}",
        f"{'Random baseline':<28} {'~0.000':>10}  {'N/A':>10}",
        "=" * 55,
        f"Wall time: {wall_time:.1f}s  |  Imputed: {agent_result.get('imputed_count', 0)}",
        "=" * 55,
    ]
    table_str = "\n".join(table_lines)
    print("\n" + table_str)

    # ------------------------------------------------------------------
    # 5. Edge-case analysis
    # ------------------------------------------------------------------
    print("\nIdentifying edge cases where ensemble rescued ESM-2 ranking errors…")
    edge_cases = identify_edge_cases(ranked_cands, experimental_fitness, n=3)

    if edge_cases:
        print("\n  Top ensemble rescue cases:")
        for ec in edge_cases:
            print(
                f"    {ec['candidate_id']}  exp_fitness={ec['experimental_fitness']:.3f}  "
                f"ESM2-rank={ec['esm2_rank']}→composite-rank={ec['composite_rank']}  "
                f"(+{ec['rank_improvement']} positions)\n"
                f"      Reason: {ec['explanation']}"
            )
    else:
        print("  No clear rescue cases found in this sample.")

    # ------------------------------------------------------------------
    # 6. Visualisations
    # ------------------------------------------------------------------
    print("\nGenerating plots…")

    build_scatter(
        x=composite_vals,
        y=fitness_vals,
        xlabel="Agent Composite Score",
        ylabel="Experimental Fitness (log enrichment)",
        title="Full Agent: Composite Score vs. Experimental Fitness (GB1)",
        rho=rho_full,
        p=p_full,
        out_path=os.path.join(RESULTS_DIR, "scatter_full_agent.png"),
        highlight_ids=[
            ranked_cands.index(
                next((c for c in ranked_cands if c["candidate_id"] == ec["candidate_id"]), None)
            )
            for ec in edge_cases
            if any(c["candidate_id"] == ec["candidate_id"] for c in ranked_cands)
        ] if edge_cases else None,
        highlight_label="ensemble rescue",
    )

    build_ablation_grid(
        fitness=fitness_vals,
        esm2_scores=esm2_vals,
        plddt_scores=plddt_vals,
        tm_scores=tm_vals,
        blast_scores=blast_vals,
        out_path=os.path.join(RESULTS_DIR, "scatter_ablations.png"),
    )

    # Residual plot: signed rank error (composite rank − fitness rank)
    n = len(fitness_vals)
    fitness_rank = pd.Series(fitness_vals).rank(ascending=False).tolist()
    composite_rank_list = [c.get("rank", 0) for c in ranked_cands]
    residuals = [cr - fr for cr, fr in zip(composite_rank_list, fitness_rank)]
    cand_labels = [c["candidate_id"] for c in ranked_cands]

    build_scatter(
        x=list(range(1, n + 1)),
        y=residuals,
        xlabel="Candidate index (sorted by composite rank)",
        ylabel="Rank error (composite − fitness rank)",
        title="Signed Rank Residuals — Full Agent",
        rho=0.0,
        p=1.0,
        out_path=os.path.join(RESULTS_DIR, "scatter_residuals.png"),
    )

    # ------------------------------------------------------------------
    # 7. Save results to JSON
    # ------------------------------------------------------------------
    results_payload = {
        "meta": {
            "dataset": "FLIP GB1 (Olson et al. 2014)",
            "n_variants": len(fitness_vals),
            "random_seed": RANDOM_SEED,
            "wall_time_seconds": round(wall_time, 1),
            "imputed_count": agent_result.get("imputed_count", 0),
            "weights_used": agent_result.get("weights_used", {}),
        },
        "spearman": {
            "full_agent": {"rho": round(rho_full, 4), "p": round(p_full, 6)},
            "esm2_only":  {"rho": round(rho_esm2, 4), "p": round(p_esm2, 6)},
            "plddt_only": {"rho": round(rho_plddt, 4), "p": round(p_plddt, 6)},
            "tm_only":    {"rho": round(rho_tm, 4), "p": round(p_tm, 6)},
            "blast_only": {"rho": round(rho_blast, 4), "p": round(p_blast, 6)},
        },
        "edge_cases": edge_cases,
        "ranked_candidates": ranked_cands,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_path = os.path.join(RESULTS_DIR, "benchmark_results.json")
    with open(json_path, "w") as fh:
        json.dump(results_payload, fh, indent=2)
    print(f"\n  [results] saved → {json_path}")

    # Save text table
    table_path = os.path.join(RESULTS_DIR, "spearman_table.txt")
    with open(table_path, "w") as fh:
        fh.write(table_str + "\n")
    print(f"  [results] saved → {table_path}")

    # ------------------------------------------------------------------
    # 8. Auto-update README with real Spearman numbers
    # ------------------------------------------------------------------
    update_readme(
        spearman=results_payload["spearman"],
        n_variants=len(fitness_vals),
        esm2_model=args.model,
    )

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
