import sys
import os
import pickle
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

sys.path.append(".")

def measure_prompt_intro_outro(tokenizer):
    """
    Creates a dummy prompt with NO retrieved descriptions to measure 
    the fixed static cost (Intro + Graph Summary + Instructions).
    """
    # Mock Graph Summary
    summary = (
        f"Target molecule graph summary:\n"
        f"- Number of atoms: 25\n"
        f"- Number of bonds: 27\n"
        f"- Contains aromatic atoms: yes\n"
        f"- Contains rings: yes\n"
        f"- Total formal charge: 0\n"
        f"- Number of heteroatoms: 4\n"
        f"- Contains phosphorus: no\n"
        f"- Contains nitrogen: yes\n"
    )

    # Static parts of the prompt
    intro = (
        "You are a chemistry expert.\n\n"
        f"{summary}\n"
        "Below are descriptions of molecules that are structurally and "
        "functionally similar to the target molecule:\n\n"
    )
    
    outro = (
        "\nBased on the graph summary and the retrieved examples above, "
        "write a concise and accurate description of the target molecule.\n\n"
        "Description:"
    )
    
    # Measure Base Tokens (Intro + Outro)
    full_static_text = intro + outro
    base_tokens = len(tokenizer.encode(full_static_text))
    
    # Measure per-neighbor structure cost (e.g. "[1] \n")
    neighbor_structure = tokenizer.encode("[1] \n")
    per_neighbor_cost = len(neighbor_structure)
    
    return base_tokens, per_neighbor_cost

def compute_percentiles(lengths):
    return {
        "min": np.min(lengths),
        "avg": np.mean(lengths),
        "median": np.median(lengths),
        "p90": np.percentile(lengths, 90),
        "p95": np.percentile(lengths, 95),
        "max": np.max(lengths)
    }

def print_stats(name, stats):
    print(f"\n--- {name} Statistics ---")
    print(f"  Min:    {int(stats['min'])}")
    print(f"  Avg:    {int(stats['avg'])}")
    print(f"  Median: {int(stats['median'])}")
    print(f"  90%:    {int(stats['p90'])}")
    print(f"  95%:    {int(stats['p95'])}")
    print(f"  Max:    {int(stats['max'])}")

def main():
    # Path to data
    pkl_path = "data/raw/train_graphs.pkl"
    
    if not os.path.exists(pkl_path):
        print(f"Error: {pkl_path} not found.")
        return

    print(f"Loading {pkl_path}...")
    with open(pkl_path, "rb") as f:
        graphs = pickle.load(f)
    
    descriptions = [g.description for g in graphs]
    print(f"Loaded {len(descriptions)} descriptions.")

    # ==========================================
    # BERT Tokenizer Stats (For Retrieval)
    # ==========================================
    print("\nTokenizing with BERT (bert-base-uncased)...")
    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    bert_lens = []
    for text in tqdm(descriptions, desc="BERT"):
        # Add special tokens to get accurate input length
        tokens = bert_tokenizer.encode(text, add_special_tokens=True)
        bert_lens.append(len(tokens))
        
    bert_stats = compute_percentiles(bert_lens)
    print_stats("BERT Token Lengths (Encoder)", bert_stats)
    

    # ==========================================
    # GPT-2 Tokenizer Stats (For Generation)
    # ==========================================
    print("\nTokenizing with GPT-2 (gpt2-medium)...")
    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
    
    target_lens = []
    for text in tqdm(descriptions, desc="GPT-2"):
        tokens = gpt2_tokenizer.encode(text)
        target_lens.append(len(tokens))
        
    gpt2_stats = compute_percentiles(target_lens)
    print_stats("GPT-2 Target Lengths (Output)", gpt2_stats)


    # ==========================================
    # RAG Prompt Estimation
    # ==========================================
    K = 2
    avg_desc = gpt2_stats['avg']
    p95_desc = gpt2_stats['p95']
    
    # Measure Prompt Fixed Costs
    base_tokens, per_neighbor_cost = measure_prompt_intro_outro(gpt2_tokenizer)
    
    print(f"\n--- Prompt Structure Costs ---")
    print(f"  Fixed Static Text: {base_tokens} tokens")
    print(f"  Per Neighbor Cost: {per_neighbor_cost} tokens")

    # Estimate Total Input Length (Prompt + K*Neighbors + Target)
    est_len_avg = base_tokens + (K * (avg_desc + per_neighbor_cost)) + avg_desc
    est_len_95 = base_tokens + (K * (p95_desc + per_neighbor_cost)) + p95_desc
    
    print(f"\n--- RAG Context Window Estimation (K={K}) ---")
    print(f"  Avg Case: ~{int(est_len_avg)} tokens")
    print(f"  95% Case: ~{int(est_len_95)} tokens")

if __name__ == "__main__":
    main()