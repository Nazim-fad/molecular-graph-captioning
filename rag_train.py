import os
import torch
from torch.utils.data import DataLoader
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model

from rag_dataset import CachedRAGDataset
from train_gcn import MolGNN, DEVICE, TRAIN_GRAPHS, TRAIN_EMB_CSV


# =========================
# CONFIG
# =========================
MODEL_NAME = "gpt2-medium"
BATCH_SIZE = 2
EPOCHS = 3
LR = 2e-4
WARMUP_RATIO = 0.05
MAX_LENGTH = 896
TOP_K = 5
SAVE_PATH = "rag_gpt2_lora.pt"


# =========================
# Load Stage-1 model
# =========================
checkpoint = torch.load("model_checkpoint.pt", map_location=DEVICE)

# Infer embedding dim
import pandas as pd
df = pd.read_csv(TRAIN_EMB_CSV)
emb_dim = len(df.iloc[0]["embedding"].split(","))

mol_encoder = MolGNN(out_dim=emb_dim).to(DEVICE)
mol_encoder.load_state_dict(checkpoint["mol_enc"])
mol_encoder.eval()
for p in mol_encoder.parameters():
    p.requires_grad = False


# =========================
# Load tokenizer & model
# =========================
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model.resize_token_embeddings(len(tokenizer))
model.to(DEVICE)

# =========================
# Apply LoRA
# =========================
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["c_attn"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# =========================
# Dataset & DataLoader
# =========================
dataset = CachedRAGDataset(
    cache_path="rag_cache_train.jsonl",
    max_length=896
)


loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)


# =========================
# Optimizer & Scheduler
# =========================
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

num_training_steps = EPOCHS * len(loader)
num_warmup_steps = int(WARMUP_RATIO * num_training_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)


# =========================
# Training loop
# =========================
model.train()

for epoch in range(EPOCHS):
    total_loss = 0.0

    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - loss: {avg_loss:.4f}")

# =========================
# Save LoRA weights
# =========================
torch.save(model.state_dict(), SAVE_PATH)
print(f"Saved LoRA model to {SAVE_PATH}")
