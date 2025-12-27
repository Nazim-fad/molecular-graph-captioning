import json
import torch
from tqdm import tqdm
import pandas as pd
from torch_geometric.data import Batch

from train_gcn import MolGNN, DEVICE
from data_utils import (
    PreprocessedGraphDataset,
    load_descriptions_from_graphs,
    load_id2emb
)


# =========================
# CONFIG
# =========================
VAL_GRAPHS = "data/validation_graphs.pkl"
TRAIN_GRAPHS = "data/train_graphs.pkl"
TRAIN_EMB_CSV = "data/train_embeddings.csv"

OUTPUT_PATH = "rag_cache_val.jsonl"
TOP_K = 5


def build_prompt(retrieved_texts):
    prompt = (
        "You are a chemistry expert.\n\n"
        "Below are descriptions of molecules that are structurally and "
        "functionally similar to a target molecule:\n\n"
    )

    for i, txt in enumerate(retrieved_texts, 1):
        prompt += f"[{i}] {txt}\n"

    prompt += (
        "\nBased on the information above, write a concise and accurate "
        "description of the target molecule.\n\n"
        "Description:"
    )
    return prompt


def main():
    print("Device:", DEVICE)

    # =========================
    # Load trained MolGNN
    # =========================
    checkpoint = torch.load("model_checkpoint.pt", map_location=DEVICE)

    df = pd.read_csv(TRAIN_EMB_CSV)
    emb_dim = len(df.iloc[0]["embedding"].split(","))

    mol_encoder = MolGNN(out_dim=emb_dim).to(DEVICE)
    mol_encoder.load_state_dict(checkpoint["mol_enc"])
    mol_encoder.eval()
    for p in mol_encoder.parameters():
        p.requires_grad = False

    # =========================
    # Load TRAIN retrieval pool
    # =========================
    train_emb_dict = load_id2emb(TRAIN_EMB_CSV)
    train_ids = list(train_emb_dict.keys())
    train_embs = torch.stack(
        [train_emb_dict[i] for i in train_ids]
    ).to(DEVICE)
    train_embs = torch.nn.functional.normalize(train_embs, dim=-1)

    train_id2desc = load_descriptions_from_graphs(TRAIN_GRAPHS)

    # =========================
    # Load VALIDATION graphs
    # =========================
    val_ds = PreprocessedGraphDataset(VAL_GRAPHS)
    print(f"Validation set size: {len(val_ds)}")

    # =========================
    # Build cache
    # =========================
    print("Building validation RAG cache...")
    with open(OUTPUT_PATH, "w") as f:
        for i in tqdm(range(len(val_ds))):
            graph = val_ds[i]
            graph_id = graph.id

            batch = Batch.from_data_list([graph]).to(DEVICE)
            with torch.no_grad():
                emb = mol_encoder(batch)
                emb = torch.nn.functional.normalize(emb, dim=-1)

            # retrieve ONLY from TRAIN
            sims = emb @ train_embs.t()
            topk_idx = sims.topk(TOP_K, dim=-1).indices.squeeze(0)

            retrieved_texts = [
                train_id2desc[train_ids[j]]
                for j in topk_idx.tolist()
            ]

            prompt = build_prompt(retrieved_texts)
            target = graph.description

            record = {
                "id": graph_id,
                "prompt": prompt,
                "target": target
            }

            f.write(json.dumps(record) + "\n")

    print(f"\nSaved validation RAG cache to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
