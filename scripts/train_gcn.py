import sys
import os
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(".")

from src.losses import contrastive_loss, matching_loss
from src.data_utils import load_id2emb, PreprocessedGraphDataset, collate_fn
from src.model_gcn import MolGNN, MatchingHead


# =========================================================
# CONFIG
# =========================================================
with open("configs/gcn_train.yaml", "r") as f:
    config = yaml.safe_load(f)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# Training and Evaluation
# =========================================================

def train_epoch(mol_enc, match_head, loader, optimizer, device):
    mol_enc.train()
    match_head.train()

    total_loss, total = 0.0, 0
    
    # params
    temp = config['training']['temperature']
    lambda_itm = config['training']['lambda_itm']

    for graphs, text_emb in loader:
        graphs = graphs.to(device)
        text_emb = text_emb.to(device)

        mol_vec = mol_enc(graphs)
        mol_vec = F.normalize(mol_vec, dim=-1)
        txt_vec = F.normalize(text_emb, dim=-1)

        # Contrastive loss (ITC)
        loss_itc = contrastive_loss(mol_vec, txt_vec, temperature=temp)

        # Matching loss (ITM)
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

# =========================================================
# Main Training Loop
# =========================================================
def main():
    print(f"Device: {DEVICE}")

    # Load paths from config
    train_emb_path = config['paths']['train_emb_csv']
    val_emb_path = config['paths']['val_emb_csv']
    train_graphs_path = config['paths']['train_graphs']
    val_graphs_path = config['paths']['val_graphs']
    checkpoint_dir = config['paths']['checkpoint_dir']
    
    # training params
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

    optimizer = torch.optim.Adam(
        list(mol_enc.parameters()) + list(match_head.parameters()),
        lr=learning_rate
    )

    for ep in range(epochs):
        train_loss = train_epoch(mol_enc, match_head, train_dl, optimizer, DEVICE)
        if val_emb is not None and os.path.exists(val_graphs_path):
            val_scores = eval_retrieval(val_graphs_path, val_emb, mol_enc, DEVICE)
        else:
            val_scores = {}
        print(f"Epoch {ep+1}/{epochs} - loss={train_loss:.4f} - val={val_scores}")
    
    # Save Model
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = os.path.join(checkpoint_dir, "gcn_checkpoint.pt")
    torch.save({
            "mol_enc": mol_enc.state_dict(),
            "match_head": match_head.state_dict(),
            "config": config
        }, save_path)
    
    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    main()