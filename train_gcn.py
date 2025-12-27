import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from losses import contrastive_loss, matching_loss
from torch.utils.data import DataLoader

from torch_geometric.data import Batch
from torch_geometric.nn import GINEConv, global_add_pool


from data_utils import (
    load_id2emb,
    PreprocessedGraphDataset, collate_fn
)


# =========================================================
# CONFIG
# =========================================================
# Data paths
TRAIN_GRAPHS = "data/train_graphs.pkl"
VAL_GRAPHS   = "data/validation_graphs.pkl"
TEST_GRAPHS  = "data/test_graphs.pkl"

TRAIN_EMB_CSV = "data/train_embeddings.csv"
VAL_EMB_CSV   = "data/validation_embeddings.csv"

# Training parameters
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3

TEMPERATURE = 0.07
LAMBDA_ITM = 1.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# MATCHING HEAD : MLP to predict if graph-text pair matches
# =========================================================
class MatchingHead(nn.Module):
    def __init__(self, emb_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(2 * emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# =========================================================
# MODEL: GNN to encode graphs (simple GCN, no edge features)
# =========================================================
class MolGNN(nn.Module):
    def __init__(self, hidden=128, out_dim=256, layers=3):
        super().__init__()

        # Atom feature embeddings (node features)
        self.node_embeds = nn.ModuleList([
            nn.Embedding(119, hidden),  # atomic_num
            nn.Embedding(9, hidden),    # chirality
            nn.Embedding(11, hidden),   # degree
            nn.Embedding(12, hidden),   # formal_charge
            nn.Embedding(9, hidden),    # num_hs
            nn.Embedding(5, hidden),    # num_radical_electrons
            nn.Embedding(8, hidden),    # hybridization
            nn.Embedding(2, hidden),    # is_aromatic
            nn.Embedding(2, hidden),    # is_in_ring
        ])

        # Bond feature embeddings (edge features)
        self.edge_embeds = nn.ModuleList([
            nn.Embedding(22, hidden),   # bond_type
            nn.Embedding(6, hidden),    # stereo
            nn.Embedding(2, hidden),    # is_conjugated
        ])

        self.convs = nn.ModuleList()
        for _ in range(layers):
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden)
            )

            self.convs.append(GINEConv(mlp))

        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, batch: Batch):
        """
        batch.x         : [num_nodes, 9]
        batch.edge_attr : [num_edges, 3]
        batch.edge_index: [2, num_edges]
        batch.batch     : [num_nodes]
        """
        # Initialize all nodes with the same learnable embedding
        x = batch.x # [node, 9]
        node_emb = 0
        for i, emb_layer in enumerate(self.node_embeds):
            node_emb = node_emb + emb_layer(x[:, i])
        # node_emb: [node, hidden_dim]

        edge_attr = batch.edge_attr  # [Edge, 3]
        edge_emb = 0
        for i, emb_layer in enumerate(self.edge_embeds):
            edge_emb = edge_emb + emb_layer(edge_attr[:, i])
        # edge_emb: [Edge, hidden_dim]
        
        # Mesasage Passing
        h = node_emb
        for conv in self.convs:
            h = conv(h, batch.edge_index, edge_emb)
            h = F.relu(h)
        # Graph pooling
        g = global_add_pool(h, batch.batch) # [num_graphs, hidden_dim]


        g = self.proj(g)
        g = F.normalize(g, dim=-1)
        return g


# =========================================================
# Training and Evaluation
# =========================================================
def train_epoch(mol_enc, match_head, loader, optimizer, device):
    mol_enc.train()
    match_head.train()

    total_loss, total = 0.0, 0
    for graphs, text_emb in loader:
        graphs = graphs.to(device)
        text_emb = text_emb.to(device)

        mol_vec = mol_enc(graphs)
        txt_vec = F.normalize(text_emb, dim=-1)

        # Contrastive loss
        loss_itc = contrastive_loss(
            mol_vec, txt_vec, temperature=TEMPERATURE
        )

        # Matching loss (ITM)
        loss_itm = matching_loss(
            mol_vec, txt_vec, match_head
        )

        loss = loss_itc + LAMBDA_ITM * loss_itm

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = graphs.num_graphs
        total_loss += loss.item() * bs
        total += bs

    return total_loss / total


@torch.no_grad()
def eval_retrieval(data_path, emb_dict, mol_enc, device):
    ds = PreprocessedGraphDataset(data_path, emb_dict)
    dl = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    all_mol, all_txt = [], []
    for graphs, text_emb in dl:
        graphs = graphs.to(device)
        text_emb = text_emb.to(device)
        all_mol.append(mol_enc(graphs))
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

    train_emb = load_id2emb(TRAIN_EMB_CSV)
    val_emb = load_id2emb(VAL_EMB_CSV) if os.path.exists(VAL_EMB_CSV) else None

    emb_dim = len(next(iter(train_emb.values())))

    if not os.path.exists(TRAIN_GRAPHS):
        print(f"Error: Preprocessed graphs not found at {TRAIN_GRAPHS}")
        print("Please run: python prepare_graph_data.py")
        return
    
    train_ds = PreprocessedGraphDataset(TRAIN_GRAPHS, train_emb)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    mol_enc = MolGNN(out_dim=emb_dim).to(DEVICE)
    match_head = MatchingHead(emb_dim).to(DEVICE)

    optimizer = torch.optim.Adam(
        list(mol_enc.parameters()) + list(match_head.parameters()),
        lr=LR
    )

    for ep in range(EPOCHS):
        train_loss = train_epoch(mol_enc, match_head, train_dl, optimizer, DEVICE)
        if val_emb is not None and os.path.exists(VAL_GRAPHS):
            val_scores = eval_retrieval(VAL_GRAPHS, val_emb, mol_enc, DEVICE)
        else:
            val_scores = {}
        print(f"Epoch {ep+1}/{EPOCHS} - loss={train_loss:.4f} - val={val_scores}")
    
    model_path = "model_checkpoint.pt"
    torch.save({
    "mol_enc": mol_enc.state_dict(),
    "match_head": match_head.state_dict()
    }, model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()
