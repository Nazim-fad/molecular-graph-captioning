import json
import torch
from tqdm import tqdm
from torch_geometric.data import Batch

from train_gcn import MolGNN, DEVICE, TRAIN_GRAPHS, TRAIN_EMB_CSV
from rag_dataset import RAGDataset
from data_utils import load_descriptions_from_graphs


OUTPUT_PATH = "rag_cache_train.jsonl"
TOP_K = 5

def main():
    # Load trained MolGNN
    checkpoint = torch.load("model_checkpoint.pt", map_location=DEVICE)

    import pandas as pd
    df = pd.read_csv(TRAIN_EMB_CSV)
    emb_dim = len(df.iloc[0]["embedding"].split(","))

    mol_encoder = MolGNN(out_dim=emb_dim).to(DEVICE)
    mol_encoder.load_state_dict(checkpoint["mol_enc"])
    mol_encoder.eval()
    for p in mol_encoder.parameters():
        p.requires_grad = False

    # Load dataset components
    rag_ds = RAGDataset(
        graph_path=TRAIN_GRAPHS,
        train_graph_path=TRAIN_GRAPHS,
        train_emb_csv=TRAIN_EMB_CSV,
        mol_encoder=mol_encoder,
        top_k=TOP_K,
        max_length=1024,  # max length not relevant for cache building
        device=DEVICE
    )

    print("Building RAG cache...")
    with open(OUTPUT_PATH, "w") as f:
        for i in tqdm(range(len(rag_ds))):
            sample = rag_ds.graph_dataset[i]
            graph_id = sample.id

            # Retrieve texts & build prompt
            batch = Batch.from_data_list([sample]).to(DEVICE)
            with torch.no_grad():
                emb = mol_encoder(batch)
                emb = torch.nn.functional.normalize(emb, dim=-1)

            sims = emb @ rag_ds.train_embs.t()
            topk_idx = sims.topk(TOP_K, dim=-1).indices.squeeze(0)

            retrieved_texts = [
                rag_ds.train_id2desc[rag_ds.train_ids[j]]
                for j in topk_idx.tolist()
            ]

            prompt = rag_ds._build_prompt(retrieved_texts)
            target = sample.description

            record = {
                "id": graph_id,
                "prompt": prompt,
                "target": target
            }

            f.write(json.dumps(record) + "\n")

    print(f"Saved RAG cache to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
