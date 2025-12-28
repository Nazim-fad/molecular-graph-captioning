import sys
import os
import yaml
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add root to path
sys.path.append(".")

from src.data_utils import (
    load_id2emb, 
    load_descriptions_from_graphs, 
    PreprocessedGraphDataset, 
    collate_fn
)
from src.model_gcn import MolGNN, MatchingHead

# =========================================================
# CONFIG
# =========================================================
config_path = "configs/gcn_train.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def retrieve_descriptions(model, match_head, train_data, test_data, train_emb_dict, device, output_csv, top_k=10):
    """
    Args:
        model: Trained GNN model
        train_data: Path to train preprocessed graphs
        test_data: Path to test preprocessed graphs
        train_emb_dict: Dictionary mapping train IDs to text embeddings
        device: Device to run on
        output_csv: Path to save retrieved descriptions
    """
    train_id2desc = load_descriptions_from_graphs(train_data)
    
    train_ids = list(train_emb_dict.keys())
    train_embs = torch.stack([train_emb_dict[id_] for id_ in train_ids]).to(device)
    train_embs = F.normalize(train_embs, dim=-1)
    
    print(f"Train set size: {len(train_ids)}")
    
    test_ds = PreprocessedGraphDataset(test_data)
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
    
    print(f"Test set size: {len(test_ds)}")
    
    test_mol_embs = []
    test_ids_ordered = []
    for graphs in test_dl:
        graphs = graphs.to(device)
        mol_emb = model(graphs)
        test_mol_embs.append(mol_emb)
        batch_size = graphs.num_graphs
        start_idx = len(test_ids_ordered)
        test_ids_ordered.extend(test_ds.ids[start_idx:start_idx + batch_size])
    
    test_mol_embs = torch.cat(test_mol_embs, dim=0)
    print(f"Encoded {test_mol_embs.size(0)} test molecules")
    
    similarities = test_mol_embs @ train_embs.t()
    
    topk_scores, topk_indices = similarities.topk(top_k, dim=-1)
    
    results = []
    for i, test_id in enumerate(test_ids_ordered):
        candidate_indices = topk_indices[i]

        graph_emb = test_mol_embs[i].unsqueeze(0).repeat(top_k, 1)
        text_embs = train_embs[candidate_indices]

        pair_emb = torch.cat([graph_emb, text_embs], dim=-1)

        with torch.no_grad():
            itm_scores = match_head(pair_emb).squeeze(-1)

        best_idx = itm_scores.argmax().item()
        train_idx = candidate_indices[best_idx].item()

        retrieved_train_id = train_ids[train_idx]
        retrieved_desc = train_id2desc[retrieved_train_id]
        
        results.append({
            'ID': test_id,
            'description': retrieved_desc
        })
        
        if i < 5:
            print(f"\nTest ID {test_id}: Retrieved from train ID {retrieved_train_id}")
            print(f"Description: {retrieved_desc[:150]}...")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"\n{'='*80}")
    print(f"Saved {len(results)} retrieved descriptions to: {output_csv}")
    
    return results_df


def main():
    print(f"Device: {DEVICE}")
    print(f"Config: {config['experiment_name']}")
    
    #  paths
    paths = config['paths']
    train_graphs = paths['train_graphs']
    test_graphs = paths['test_graphs']
    train_emb_csv = paths['train_emb_csv']
    checkpoint_dir = paths['checkpoint_dir']

    # Define Output path
    output_dir = "outputs/inference"
    output_csv = os.path.join(output_dir, "test_retrieved_descriptions.csv")
    
    # Checkpoint path
    model_path = os.path.join(checkpoint_dir, "gcn_checkpoint.pt")
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint '{model_path}' not found.")
        print("Please train a model first using train_gcn.py")
        return
    
    if not os.path.exists(test_graphs):
        print(f"Error: Preprocessed graphs not found at {test_graphs}")
        return
    
    train_emb = load_id2emb(train_emb_csv)
    
    emb_dim = len(next(iter(train_emb.values())))

    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    model = MolGNN(
        hidden=config['model']['hidden_dim'],
        out_dim=emb_dim,
        layers=config['model']['layers'],
        dropout=config['model']['dropout']
    ).to(DEVICE)
    print(f"Loading model from {model_path}")
    model.load_state_dict(checkpoint["mol_enc"])
    model.eval()

    match_head = MatchingHead(emb_dim).to(DEVICE)
    match_head.load_state_dict(checkpoint["match_head"])
    match_head.eval()
        
    retrieve_descriptions(
        model=model,
        match_head=match_head,
        train_data=train_graphs,
        test_data=test_graphs,
        train_emb_dict=train_emb,
        device=DEVICE,
        output_csv=output_csv,
        top_k=10
    )

if __name__ == "__main__":
    main()

    