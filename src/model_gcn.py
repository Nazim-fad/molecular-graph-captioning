import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool
from torch_geometric.data import Batch

# =========================================================
# MATCHING HEAD : MLP to predict if graph-text pair matches
# =========================================================
class MatchingHead(nn.Module):
    """
    MLP to predict if a (graph, text) pair matches.
    Used for the Image-Text Matching (ITM) loss.
    """
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
    """
    GNN to encode molecular graphs.
    Uses GINEConv which supports edge features.
    """
    def __init__(self, hidden=128, out_dim=768, layers=3, dropout=0.1):
        super().__init__()
        self.dropout = dropout

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

        # --- Layers ---
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
        # Embed Nodes
        x = batch.x 
        node_emb = 0
        for i, emb_layer in enumerate(self.node_embeds):
            node_emb = node_emb + emb_layer(x[:, i])
        
        # Embed Edges
        edge_attr = batch.edge_attr
        edge_emb = 0
        for i, emb_layer in enumerate(self.edge_embeds):
            edge_emb = edge_emb + emb_layer(edge_attr[:, i])
        
        # Message Passing
        h = node_emb
        for conv in self.convs:
            # GINEConv with edge features
            h = conv(h, batch.edge_index, edge_attr=edge_emb)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        # Pooling (Graph Representation)
        g = global_add_pool(h, batch.batch) 

        # Projection
        g = self.proj(g)
        g = F.normalize(g, dim=-1)
        
        return g