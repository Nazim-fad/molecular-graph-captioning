import torch
from torch.utils.data import DataLoader

from train_gcn import MolGNN, DEVICE, TRAIN_GRAPHS, TRAIN_EMB_CSV
from rag_dataset import RAGDataset

from transformers import GPT2Tokenizer

# 1. Load trained MolGNN
checkpoint = torch.load("model_checkpoint.pt", map_location=DEVICE)

emb_dim = len(next(iter(torch.load(TRAIN_EMB_CSV).values()))) if False else None
# safer way:
import pandas as pd
df = pd.read_csv(TRAIN_EMB_CSV)
emb_dim = len(df.iloc[0]["embedding"].split(","))

mol_encoder = MolGNN(out_dim=emb_dim).to(DEVICE)
mol_encoder.load_state_dict(checkpoint["mol_enc"])
mol_encoder.eval()

# 2. Create RAG dataset (small K, small max_length for test)
dataset = RAGDataset(
    graph_path=TRAIN_GRAPHS,
    train_graph_path=TRAIN_GRAPHS,
    train_emb_csv=TRAIN_EMB_CSV,
    mol_encoder=mol_encoder,
    top_k=3,
    max_length=768,
    device=DEVICE
)

print("Dataset length:", len(dataset))

# 3. Wrap in DataLoader
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# 4. Fetch one batch
batch = next(iter(loader))

print("\nBatch keys:", batch.keys())
print("input_ids shape:", batch["input_ids"].shape)
print("attention_mask shape:", batch["attention_mask"].shape)
print("labels shape:", batch["labels"].shape)


tokenizer = dataset.tokenizer

sample_ids = batch["input_ids"][0]
sample_labels = batch["labels"][0]

decoded = tokenizer.decode(sample_ids, skip_special_tokens=True)
print("\n===== FULL INPUT TEXT =====\n")
print(decoded)

# # Inspect masking
# masked_positions = (sample_labels == -100).sum().item()
# print("\nMasked tokens (prompt):", masked_positions)
# print("Total tokens:", sample_labels.size(0))


# # Show first 50 label values
# print("\nFirst 50 labels:")
# print(sample_labels[:50])

# # Show transition point
# non_masked = (sample_labels != -100).nonzero(as_tuple=True)[0][0].item()
# print("\nFirst unmasked token index:", non_masked)

# print("\nDecoded target start:")
# print(tokenizer.decode(sample_ids[non_masked:non_masked+50]))
