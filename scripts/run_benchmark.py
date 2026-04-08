import os
import sys
import pandas as pd
import numpy as np
import time
from scipy.stats import spearmanr
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent.orchestrator import run_agent

def callback(event):
    if event.get("type") in ["tool_call", "stage_complete"]:
        print(f"[{event.get('stage_name', event.get('tool'))}] {event}", flush=True)

def main():
    print("Initialising benchmark (Importing large ML models, this may take 1-2 minutes on WSL)...", flush=True)
    data_path = 'data/gb1_fitness.csv'
    reference_pdb = 'gb1_wt.pdb'
    
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return
        
    df = pd.read_csv(data_path)
    # the target column is the fitness
    df = df.rename(columns={'target': 'fitness'})
    
    # 1. Extract WT sequence (fitness is 1.0 or very close)
    wt_row = df.iloc[(df['fitness'] - 1.0).abs().argsort()[:1]].iloc[0]
    wt_seq = wt_row['sequence']
    print(f"Detected WT Sequence (fitness {wt_row['fitness']}): {wt_seq[:20]}... (len {len(wt_seq)})")
    
    # 2. Stratified sampling of 50 variants
    # Remove WT from sampling
    df_vars = df[df['sequence'] != wt_seq].copy()
    
    df_vars['quartile'] = pd.qcut(df_vars['fitness'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
    
    sampled_df = df_vars.groupby('quartile', observed=True).apply(
        lambda x: x.sample(n=min(len(x), 3), random_state=42)
    ).reset_index(drop=True)
    
    # ensure total is around 10
    if len(sampled_df) > 10:
        sampled_df = sampled_df.sample(n=10, random_state=42)
        
    sequences_dict = {f"cand_{i}": row['sequence'] for i, row in sampled_df.iterrows()}
    experimental_fitness = {f"cand_{i}": row['fitness'] for i, row in sampled_df.iterrows()}
    
    print(f"Sampled {len(sequences_dict)} variants for benchmarking.", flush=True)
    
    # 3. Run Agent Orchestrator
    task = ('Given these GB1 variant sequences, rank them by expected thermostability/fitness '
            'using ESM-2 sequence scoring, ESMFold structure prediction, TM-align, and BLAST conservation.')
            
    print("Starting agent pipeline...", flush=True)
    start_time = time.time()
    
    try:
        agent_result = run_agent(
            task=task,
            sequences=sequences_dict,
            wildtype=wt_seq,
            reference_pdb_path=reference_pdb,
            stream_callback=callback
        )
    except Exception as e:
        print(f"Agent failed: {e}")
        return
        
    duration = time.time() - start_time
    print(f"\nPipeline finished in {duration:.1f} seconds.")
    print(f"Imputed count: {agent_result.get('imputed_count', 0)}")
    
    ranked_cands = agent_result['ranked_candidates']
    
    # 4. Correlation Analysis
    real_scores = []
    agent_composite = []
    esm2_llrs = []
    esmfold_plddts = []
    tm_scores = []
    blast_scores = []
    
    for cand in ranked_cands:
        cid = cand['candidate_id']
        real = experimental_fitness[cid]
        
        real_scores.append(real)
        agent_composite.append(cand.get('composite_score', 0))
        esm2_llrs.append(cand.get('llr', 0))
        esmfold_plddts.append(cand.get('mean_plddt', 0))
        tm_scores.append(cand.get('tm_score', 0))
        blast_scores.append(cand.get('conservation_score', 0))
        
    def safe_spearman(x, y):
        if np.std(x) == 0 or np.std(y) == 0:
            return 0.0, 1.0
        return spearmanr(x, y)
        
    rho_full, _ = safe_spearman(agent_composite, real_scores)
    rho_esm2, _ = safe_spearman(esm2_llrs, real_scores)
    rho_plddt, _ = safe_spearman(esmfold_plddts, real_scores)
    rho_tm, _ = safe_spearman(tm_scores, real_scores)
    rho_blast, _ = safe_spearman(blast_scores, real_scores)
    
    print("\n" + "="*50)
    print("BENCHMARK RESULTS (Spearman ρ against experimental fitness)")
    print("="*50)
    print(f"{'Condition':<25} | {'Spearman ρ':<10}")
    print("-" * 40)
    print(f"{'Full Agent (All 4 tools)':<25} | {rho_full:.3f}")
    print(f"{'ESM-2 LLR only':<25} | {rho_esm2:.3f}")
    print(f"{'ESMFold pLDDT only':<25} | {rho_plddt:.3f}")
    print(f"{'TM-score only':<25} | {rho_tm:.3f}")
    print(f"{'BLAST only':<25} | {rho_blast:.3f}")
    print(f"{'Random Baseline':<25} | ~0.000")
    print("="*50)
    print(f"Total sequences evaluated: {len(real_scores)}")

if __name__ == '__main__':
    main()
