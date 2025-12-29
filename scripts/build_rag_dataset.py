import sys
import os
import json
import yaml
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

# Add root to path
sys.path.append(".")

from src.model_gcn import MolGNN
from src.data_utils import (
    load_id2emb, 
    load_descriptions_from_graphs, 
    PreprocessedGraphDataset, 
    collate_fn
)
from src.prompts import build_prompt

# =========================
# CONFIG
# =========================
config_path = "configs/rag_dataset.yaml" 
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# GCN Config
gcn_config_path = "configs/gcn_train.yaml"
with open(gcn_config_path, "r") as f:
    gcn_config = yaml.safe_load(f)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_dataset_for_split(split_name, graph_ds, mol_encoder, train_embs, train_ids, train_id2desc, output_path, top_k):
    """
    Generates RAG prompts and targets for a specific data split.
    """
    print(f"\n--- Building RAG Dataset for: {split_name} ---")
    print(f"Total samples: {len(graph_ds)}")
    
    # DataLoader for faster batch encoding
    loader = DataLoader(graph_ds, batch_size=config['params']['batch_size'], shuffle=False, collate_fn=collate_fn)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    mol_encoder.eval()
    
    with open(output_path, "w") as f:
        for batch in tqdm(loader, desc=f"Processing {split_name}"):
            batch = batch.to(DEVICE)
            
            # Encode Target Graphs
            with torch.no_grad():
                target_embs = mol_encoder(batch)
            
            # Retrieve Similar Molecules
            # [Batch, #Train]
            sims = target_embs @ train_embs.t()
            
            # Top-K indices
            topk_indices = sims.topk(top_k, dim=-1).indices
            
            # Construct Prompts
            batch_list = batch.to_data_list()
            
            for i, graph in enumerate(batch_list):
                graph_id = str(graph.id)
                
                # retrieved text descriptions
                current_indices = topk_indices[i].tolist()
                retrieved_texts = [
                    train_id2desc[train_ids[idx]]
                    for idx in current_indices
                ]
                
                # Build Prompt
                prompt = build_prompt(graph, retrieved_texts)
                target_text = graph.description if split_name != "Test" else ""
                
                record = {
                    "id": graph_id,
                    "prompt": prompt,
                    "target": target_text
                }
                
                f.write(json.dumps(record) + "\n")
                
    print(f"Saved {split_name} dataset to {output_path}")


def main():
    print(f"Device: {DEVICE}")
    print(f"Config: {config['experiment_name']}")
    
    # Load Model & Embeddings
    ckpt_path = config['paths']['model_checkpoint']
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return

    print("Loading Embeddings...")
    train_emb_dict = load_id2emb(config['paths']['retrieval_emb_csv'])
    emb_dim = len(next(iter(train_emb_dict.values())))

    print(f"Loading GCN Encoder from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    
    mol_encoder = MolGNN(
            out_dim=emb_dim,
            hidden=gcn_config['model']['hidden_dim'],
            layers=gcn_config['model']['layers'],
            dropout=gcn_config['model']['dropout']
        ).to(DEVICE)
    mol_encoder.load_state_dict(checkpoint["mol_enc"])
    mol_encoder.eval()

    # Retrieval Database (Train Set)
    print("Indexing Retrieval Database...")
    train_ids = list(train_emb_dict.keys())
    train_embs = torch.stack([train_emb_dict[i] for i in train_ids]).to(DEVICE)
    train_embs = torch.nn.functional.normalize(train_embs, dim=-1)
    
    train_id2desc = load_descriptions_from_graphs(config['paths']['retrieval_graphs'])

    # Generate Validation Dataset
    if os.path.exists(config['paths']['val_graphs']):
        val_ds = PreprocessedGraphDataset(config['paths']['val_graphs'])
        build_dataset_for_split(
            "Validation",
            val_ds,
            mol_encoder,
            train_embs,
            train_ids,
            train_id2desc,
            config['paths']['val_output'],
            config['params']['top_k']
        )

    # Generate Training Dataset 
    if os.path.exists(config['paths']['train_graphs']):
        train_ds = PreprocessedGraphDataset(config['paths']['train_graphs'])
        build_dataset_for_split(
            "Train",
            train_ds,
            mol_encoder,
            train_embs,
            train_ids,
            train_id2desc,
            config['paths']['train_output'],
            config['params']['top_k']
        )
    # Generate Test Dataset
    if os.path.exists(config['paths']['test_graphs']):
        test_ds = PreprocessedGraphDataset(config['paths']['test_graphs'])
        build_dataset_for_split(
            "Test",
            test_ds,
            mol_encoder,
            train_embs,
            train_ids,
            train_id2desc,
            config['paths']['test_output'],
            config['params']['top_k']
        )

if __name__ == "__main__":
    main()