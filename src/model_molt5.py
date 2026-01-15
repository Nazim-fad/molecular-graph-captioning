import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch_geometric.nn import GATv2Conv, global_add_pool, LayerNorm
from torch_geometric.utils import to_dense_batch

class GraphEncoder(nn.Module):
    """
    Encoder: GATv2 (Attention) + Virtual Node (Global Context).
    """
    def __init__(self, hidden_dim=128, num_layers=3, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        
        self.node_embeds = nn.ModuleList([
            nn.Embedding(119, hidden_dim), 
            nn.Embedding(9, hidden_dim),   
            nn.Embedding(11, hidden_dim),  
            nn.Embedding(12, hidden_dim),  
            nn.Embedding(9, hidden_dim),   
            nn.Embedding(5, hidden_dim),  
            nn.Embedding(8, hidden_dim),   
            nn.Embedding(2, hidden_dim), 
            nn.Embedding(2, hidden_dim),  
        ])

        self.edge_embeds = nn.ModuleList([
            nn.Embedding(22, hidden_dim), 
            nn.Embedding(6, hidden_dim),  
            nn.Embedding(2, hidden_dim),  
        ])

        self.virtual_node_emb = nn.Embedding(1, hidden_dim)
        
        self.virtual_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(num_layers):
            self.convs.append(GATv2Conv(
                in_channels=hidden_dim, 
                out_channels=hidden_dim // 4, 
                heads=4, 
                concat=True, 
                edge_dim=hidden_dim, 
                add_self_loops=False 
            ))
            self.norms.append(LayerNorm(hidden_dim))

    def forward(self, batch):
        x = batch.x
        h = 0
        for i, emb_layer in enumerate(self.node_embeds):
            h = h + emb_layer(x[:, i])
        
        edge_attr = batch.edge_attr
        e = 0
        for i, emb_layer in enumerate(self.edge_embeds):
            e = e + emb_layer(edge_attr[:, i])

        batch_size = batch.num_graphs
        virtual_node_feat = self.virtual_node_emb(
            torch.zeros(batch_size, dtype=torch.long, device=h.device)
        )

        for conv, norm in zip(self.convs, self.norms):
            h = h + virtual_node_feat[batch.batch]
            
            h_in = h
            
            h = conv(h, batch.edge_index, edge_attr=e)
            h = norm(h)
            h = F.gelu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            
            h = h + h_in

            aggr = global_add_pool(h, batch.batch)
            virtual_node_feat = virtual_node_feat + self.virtual_mlp(aggr)

        h_dense, mask = to_dense_batch(h, batch.batch)
        
        return h_dense, mask


class MolT5Generator(nn.Module):
    def __init__(self, gnn_hidden=128, model_name="laituan245/molt5-base", project_dim=256):
        super().__init__()
        
        print(f"Loading {model_name}...")
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)
        
        self.t5.gradient_checkpointing_enable()
        
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        
        self.graph_encoder = GraphEncoder(hidden_dim=gnn_hidden, num_layers=3)
        
        self.connector = nn.Sequential(
            nn.Linear(gnn_hidden, gnn_hidden * 2),
            nn.GELU(), 
            nn.Linear(gnn_hidden * 2, self.t5.config.d_model),
            nn.LayerNorm(self.t5.config.d_model)
        )

        self.graph_proj = nn.Linear(gnn_hidden, project_dim)
        self.text_proj = nn.Linear(self.t5.config.d_model, project_dim)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, batch, target_ids, target_mask):
        graph_feats_dense, graph_mask = self.graph_encoder(batch)

        mask_expanded = graph_mask.unsqueeze(-1) 
        sum_embeddings = torch.sum(graph_feats_dense * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        graph_vec = sum_embeddings / sum_mask
        
        clean_input_ids = target_ids.clone()
        clean_input_ids[clean_input_ids == -100] = self.tokenizer.pad_token_id

        encoder_outputs = self.t5.encoder(
            input_ids=clean_input_ids, 
            attention_mask=target_mask
        )
        text_hidden = encoder_outputs.last_hidden_state
        
        text_mask_expanded = target_mask.unsqueeze(-1)
        text_vec = torch.sum(text_hidden * text_mask_expanded, dim=1) / torch.clamp(text_mask_expanded.sum(dim=1), min=1e-9)

        graph_proj = self.graph_proj(graph_vec) 
        text_proj = self.text_proj(text_vec)

        inputs_embeds = self.connector(graph_feats_dense)
        outputs = self.t5(
            inputs_embeds=inputs_embeds,
            attention_mask=graph_mask,
            labels=target_ids,
            decoder_attention_mask=target_mask
        )
        
        return outputs.loss, graph_proj, text_proj, self.logit_scale.exp()

    @torch.no_grad()
    def generate(self, batch, max_length=256, num_beams=4):
        graph_feats, graph_mask = self.graph_encoder(batch)
        inputs_embeds = self.connector(graph_feats)
        
        gen_ids = self.t5.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=graph_mask,
            max_length=max_length,
            num_beams=num_beams,
            repetition_penalty=1.2,
            length_penalty=1.0
        )
        return self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)