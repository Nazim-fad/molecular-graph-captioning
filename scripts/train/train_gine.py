import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir)) 
sys.path.append(project_root)

import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import math
import torch.nn as nn


from src.losses import clip_contrastive_loss, matching_loss
from src.data_utils import load_id2emb, PreprocessedGraphDataset, collate_fn
from src.model_gine import MolGNN, MatchingHead


# CONFIG
with open("configs/gcn_train.yaml", "r") as f:
    config = yaml.safe_load(f)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Training and Evaluation
def train_epoch(mol_enc, match_head, loader, optimizer, device, logit_scale):
    mol_enc.train()
    match_head.train()

    total_loss, total = 0.0, 0

    lambda_itm = config['training']['lambda_itm']
    label_smoothing = config['training'].get('label_smoothing', 0.0)

    for graphs, text_emb in loader:
        graphs = graphs.to(device)
        text_emb = text_emb.to(device)

        mol_vec = mol_enc(graphs)                 # [B,D]
        mol_vec = F.normalize(mol_vec, dim=-1)
        txt_vec = F.normalize(text_emb, dim=-1)   # [B,D]

        # Clamp logit_scale for stability (temperature in [0.01, 0.5])
        logit_scale.data.clamp_(math.log(1/0.5), math.log(1/0.01))

        # Contrastive loss - CLIP style
        loss_itc = clip_contrastive_loss(
            mol_vec,
            txt_vec,
            logit_scale=logit_scale.exp(),
            label_smoothing=label_smoothing
        )

        # Matching loss
        loss_itm = matching_loss(mol_vec, txt_vec, match_head)

        loss = loss_itc + lambda_itm * loss_itm

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = graphs.num_graphs
        total_loss += loss.item() * bs
        total += bs

    return total_loss / total



@torch.no_grad()
def eval_retrieval(data_path, emb_dict, mol_enc, device):
    mol_enc.eval()
    ds = PreprocessedGraphDataset(data_path, emb_dict)
    dl = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    all_mol, all_txt = [], []
    for graphs, text_emb in dl:
        graphs = graphs.to(device)
        text_emb = text_emb.to(device)
        mol_out = mol_enc(graphs)
        all_mol.append(F.normalize(mol_out, dim=-1))
        all_txt.append(F.normalize(text_emb, dim=-1))
    all_mol = torch.cat(all_mol, dim=0)
    all_txt = torch.cat(all_txt, dim=0)

    sims = all_txt @ all_mol.t()
    ranks = sims.argsort(dim=-1, descending=True)

    N = all_txt.size(0)
    device = sims.device
    correct = torch.arange(N, device=device)

    pos = (ranks == correct.unsqueeze(1)).nonzero()[:, 1] + 1

    mrr = (1.0 / pos.float()).mean().item()

    results = {"MRR": mrr}

    for k in (1, 5, 10):
        hitk = (pos <= k).float().mean().item()
        results[f"R@{k}"] = hitk
        results[f"Hit@{k}"] = hitk

    return results

def main():
    print(f"Device: {DEVICE}")

    train_emb_path = config['paths']['train_emb_csv']
    val_emb_path = config['paths']['val_emb_csv']
    train_graphs_path = config['paths']['train_graphs']
    val_graphs_path = config['paths']['val_graphs']
    checkpoint_dir = config['paths']['checkpoint_dir']
    
    batch_size = config['training']['batch_size']
    learning_rate = config['training']['lr']
    epochs = config['training']['epochs']

    train_emb = load_id2emb(train_emb_path)
    val_emb = load_id2emb(val_emb_path) if os.path.exists(val_emb_path) else None

    emb_dim = len(next(iter(train_emb.values())))

    if not os.path.exists(train_graphs_path):
        print(f"Error: Preprocessed graphs not found at {train_graphs_path}")
        print("Please run: python prepare_graph_data.py")
        return
    
    train_ds = PreprocessedGraphDataset(train_graphs_path, train_emb)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # init model
    mol_enc = MolGNN(
        hidden=config['model']['hidden_dim'],
        out_dim=emb_dim,
        layers=config['model']['layers'],
        dropout=config['model']['dropout']
    ).to(DEVICE)
    match_head = MatchingHead(emb_dim).to(DEVICE)

    init_temp = config['training'].get('temperature', 0.07)
    logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_temp), device=DEVICE))

    optimizer = torch.optim.Adam(
    list(mol_enc.parameters()) + list(match_head.parameters()) + [logit_scale],
    lr=learning_rate
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    monitor = config["training"].get("monitor", "R@1")  # "MRR" ou "R@1"
    best_score = -1.0
    best_epoch = -1
    save_path = os.path.join(checkpoint_dir, "gnn_checkpoint.pt")

    for ep in range(epochs):
        train_loss = train_epoch(mol_enc, match_head, train_dl, optimizer, DEVICE, logit_scale)

        if val_emb is not None and os.path.exists(val_graphs_path):
            val_scores = eval_retrieval(val_graphs_path, val_emb, mol_enc, DEVICE)
        else:
            val_scores = {}

        print(f"Epoch {ep+1}/{epochs} - loss={train_loss:.4f} - val={val_scores}")

        if val_scores and monitor in val_scores:
            score = float(val_scores[monitor])
            if score > best_score and ep > epochs / 2:
                best_score = score
                best_epoch = ep + 1

                torch.save({
                    "mol_enc": mol_enc.state_dict(),
                    "match_head": match_head.state_dict(),
                    "epoch": best_epoch,
                }, save_path)

                print(f"Saved new best to {save_path} ({monitor}={best_score:.4f} @ epoch {best_epoch})")


if __name__ == "__main__":
    main()