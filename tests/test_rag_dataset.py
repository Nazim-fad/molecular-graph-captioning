import sys
import os
import yaml
import torch
import json
import pandas as pd
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from transformers import AutoTokenizer

# Add root to path
sys.path.append(".")

from src.model_gine import MolGNN
from src.data_utils import load_id2emb, load_descriptions_from_graphs, PreprocessedGraphDataset
from src.prompts import build_prompt
from src.rag_dataset import RAGGenerationDataset

def test_rag_pipeline_subset():
    # ==========================================
    # 1. SETUP CONFIG & DEVICE
    # ==========================================
    config_path = "configs/rag_dataset.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    gcn_config_path = "configs/gcn_train.yaml"
    with open(gcn_config_path, "r") as f:
        gcn_config = yaml.safe_load(f)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running test on: {DEVICE}")

    # ==========================================
    # 2. SIMULATE RAG PIPELINE FOR A SUBSET
    # ==========================================
    print("\n--- 1. Simulating Retrieval & Prompt Build (Subset) ---")
    
    # Load Embeddings & Model
    train_emb_dict = load_id2emb(config['paths']['retrieval_emb_csv'])
    emb_dim = len(next(iter(train_emb_dict.values())))
    
    model = MolGNN(
            out_dim=emb_dim,
            hidden=gcn_config['model']['hidden_dim'], 
            layers=gcn_config['model']['layers'],
            dropout=gcn_config['model']['dropout']
        ).to(DEVICE)
    ckpt = torch.load(config['paths']['model_checkpoint'], map_location=DEVICE)
    model.load_state_dict(ckpt["mol_enc"])
    model.eval()

    # Prepare Retrieval Index
    train_ids = list(train_emb_dict.keys())
    train_embs = torch.stack([train_emb_dict[i] for i in train_ids]).to(DEVICE)
    train_embs = torch.nn.functional.normalize(train_embs, dim=-1)
    
    # Load Descriptions (Knowledge Base)
    train_id2desc = load_descriptions_from_graphs(config['paths']['retrieval_graphs'])
    
    # Load Graphs 
    ds = PreprocessedGraphDataset(config['paths']['train_graphs'])
    
    # temporary JSONL file for the test
    temp_jsonl = "tests/temp_rag_subset.jsonl"
    os.makedirs("tests", exist_ok=True)
    
    SUBSET_SIZE = 4
    TOP_K = 3
    
    print(f"Processing first {SUBSET_SIZE} graphs...")
    
    with open(temp_jsonl, "w") as f:
        for i in range(SUBSET_SIZE):
            graph = ds[i]
            batch = Batch.from_data_list([graph]).to(DEVICE)
            
            # Encode
            with torch.no_grad():
                target_emb = model(batch) # [1, Dim]
            
            # Retrieve
            sims = target_emb @ train_embs.t()
            topk_indices = sims.topk(TOP_K, dim=-1).indices.squeeze(0)
            
            # Neighbor Texts
            retrieved_texts = [train_id2desc[train_ids[idx]] for idx in topk_indices.tolist()]
            
            # Build Prompt
            prompt = build_prompt(graph, retrieved_texts)
            
            record = {
                "id": str(graph.id),
                "prompt": prompt,
                "target": graph.description
            }
            f.write(json.dumps(record) + "\n")
            
    print(f"Saved temp subset to {temp_jsonl}")

    # ==========================================
    # 3. TEST Tokenization & Masking
    # ==========================================
    print("\n Testing RAGGenerationDataset Loading & Masking...")
    
    dataset = RAGGenerationDataset(
        jsonl_path=temp_jsonl,
        tokenizer_name="gpt2-medium",
        max_length=768 
    )
    
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    
    print("\nBatch keys:", batch.keys())
    print("input_ids shape:", batch["input_ids"].shape)
    print("labels shape:",    batch["labels"].shape)
    
    tokenizer = dataset.tokenizer
    sample_ids = batch["input_ids"][0]
    sample_labels = batch["labels"][0]
    
    # Decode Full Input
    decoded = tokenizer.decode(sample_ids, skip_special_tokens=True)
    print("\n===== FULL INPUT TEXT (Prompt + Target) =====")
    print(decoded[:500] + "...\n[...truncated...]")

    # Inspect Masking
    masked_count = (sample_labels == -100).sum().item()
    total_tokens = sample_labels.size(0)
    print(f"\nMasked tokens (Prompt + Padding): {masked_count}")
    print(f"Total tokens: {total_tokens}")
    
    # Find where the masking stops
    try:
        non_masked_idx = (sample_labels != -100).nonzero(as_tuple=True)[0][0].item()
        print(f"\nTarget Generation starts at token index: {non_masked_idx}")
        print("First 50 target tokens decoded:")
        print(tokenizer.decode(sample_ids[non_masked_idx:non_masked_idx+50]))
    except IndexError:
        print("\n[WARNING] All tokens are masked. Check if max_length is too short or prompts are too long.")

    # Clean the temporary file
    os.remove(temp_jsonl)

if __name__ == "__main__":
    test_rag_pipeline_subset()