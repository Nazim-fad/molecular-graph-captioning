import math
import torch
import torch.nn.functional as F


def clip_contrastive_loss(graph_emb, text_emb, logit_scale, label_smoothing: float = 0.0):
    """
    MolCA-style bidirectional contrastive loss : learn to align graph and text embeddings. Add learnable temperature.

    Args:
        graph_emb: Tensor [Batch, Dim] - graph embeddings
        text_emb:  Tensor [Batch, Dim] - text embeddings
        logit_scale: float, typically exp(logit_scale) ~ 1/temperature
    
    Returns:
        Scalar loss
    """
    graph_emb = F.normalize(graph_emb, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)

    logits = logit_scale * (graph_emb @ text_emb.T) # logit scale replace temperature, and will be able to change during the training

    targets = torch.arange(logits.size(0), device=logits.device)

    loss_g2t = F.cross_entropy(logits, targets, label_smoothing=label_smoothing)
    loss_t2g = F.cross_entropy(logits.T, targets, label_smoothing=label_smoothing)

    return (loss_g2t + loss_t2g) / 2


def contrastive_loss(graph_emb, text_emb, temperature=0.07):
    """
    MolCA-style bidirectional contrastive loss : learn to align graph and text embeddings.

    Args:
        graph_emb: Tensor [Batch, Dim] - graph embeddings
        text_emb:  Tensor [Batch, Dim] - text embeddings
        temperature: float - temperature scaling parameter

    Returns:
        Scalar loss
    """
    # Normalize embeddings
    graph_emb = F.normalize(graph_emb, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)

    # Similarity matrix
    logits = graph_emb @ text_emb.T
    logits = logits / temperature

    batch_size = graph_emb.size(0)
    labels = torch.arange(batch_size, device=graph_emb.device) 

    # Graph to Text
    loss_g2t = F.cross_entropy(logits, labels)

    # Text to Graph
    loss_t2g = F.cross_entropy(logits.T, labels)

    return (loss_g2t + loss_t2g) / 2


def mine_hard_negatives(similarity_matrix):
    """
    Mine hard negatives from similarity matrix.
    For each row i, find the most similar j != i.

    Args:
        similarity_matrix: Tensor [Batch, Batch]

    Returns:
        Tensor [Batch] - indices of hard negative texts
    """
    batch_size = similarity_matrix.size(0)

    # Mask diagonal (ignore true matches)
    mask = torch.eye(batch_size, device=similarity_matrix.device).bool()
    sim = similarity_matrix.masked_fill(mask, -1e9)

    hard_neg_indices = sim.argmax(dim=1)
    return hard_neg_indices


def matching_loss(graph_emb, text_emb, matching_head):
    """
    graph-text matching loss 

    Args:
        graph_emb: Tensor [Batch, Dim]
        text_emb: Tensor [Batch, Dim]
        matching_head: nn.Module that maps concatenated embeddings -> logit

    Returns:
        Scalar loss
    """
    # Normalize for similarity computation
    graph_emb = F.normalize(graph_emb, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)

    # Similarity matrix
    sim_matrix = graph_emb @ text_emb.T

    # Positive pairs
    pos_pairs = torch.cat([graph_emb, text_emb], dim=-1)
    pos_labels = torch.ones(graph_emb.size(0), device=graph_emb.device)

    # Hard negative pairs
    hard_neg_idx = mine_hard_negatives(sim_matrix)
    neg_text_emb = text_emb[hard_neg_idx]
    neg_pairs = torch.cat([graph_emb, neg_text_emb], dim=-1)
    neg_labels = torch.zeros(graph_emb.size(0), device=graph_emb.device)

    # Combine
    pairs = torch.cat([pos_pairs, neg_pairs], dim=0)
    labels = torch.cat([pos_labels, neg_labels], dim=0)

    logits = matching_head(pairs).squeeze(-1)
    loss = F.binary_cross_entropy_with_logits(logits, labels)

    return loss
