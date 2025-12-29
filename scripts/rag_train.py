import sys
import os
import yaml
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

# Add root to path
sys.path.append(".")

from src.rag_dataset import RAGGenerationDataset
from src.generation_eval import validate_generation  

# =========================
# CONFIG
# =========================
config_path = "configs/rag_train.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# MAIN TRAINING LOOP
# =========================
def main():
    print(f"Device: {DEVICE}")
    print(f"Config: {config['experiment_name']}")

    # Load Tokenizer & Model
    model_name = config['model']['name']
    print(f"Loading model: {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    model.to(DEVICE)

    model.gradient_checkpointing_enable()

    # LoRA setup
    print("Applying LoRA...")
    peft_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['alpha'],
        target_modules=config['lora']['target_modules'],
        lora_dropout=config['lora']['dropout'],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Load Datasets
    print("Loading datasets...")
    train_ds = RAGGenerationDataset(
        jsonl_path=config['paths']['train_data'],
        tokenizer_name=model_name,
        max_length=config['model']['max_length']
    )
    
    # Validation Dataset (if exists)
    val_loader = None
    if os.path.exists(config['paths']['val_data']):
        val_ds = RAGGenerationDataset(
            jsonl_path=config['paths']['val_data'],
            tokenizer_name=model_name,
            max_length=config['model']['max_length']
        )
        val_loader = DataLoader(val_ds, batch_size=config['training']['batch_size'], shuffle=False)
    else:
        print("Warning: No validation data found.")

    train_loader = DataLoader(
        train_ds,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=2
    )

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['lr'])
    
    epochs = config['training']['epochs']
    num_training_steps = epochs * len(train_loader)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['training']['warmup_steps'],
        num_training_steps=num_training_steps
    )

    # Training Loop
    best_score = 0.0
    output_dir = config['paths']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Get generation config param
    max_new_tokens = config.get('generation', {}).get('max_new_tokens', 128)
    print(f"Validation generation length: {max_new_tokens}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress_bar:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}")

        # Safety save
        checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}")
        model.save_pretrained(checkpoint_path)
        print(f"Checkpoint model saved after training epoch {epoch+1} to {checkpoint_path}")

        # Validation
        if val_loader:
            metrics = validate_generation(
                model, 
                tokenizer, 
                val_loader, 
                DEVICE, 
                max_new_tokens=max_new_tokens
            )
            print(f"Validation Results: {metrics}")

            # Save best model
            if metrics["BERTScore-F1"] > best_score:
                best_score = metrics["BERTScore-F1"]
                model.save_pretrained(output_dir)
                print(f"New best model saved to {output_dir} with BERTScore-F1: {best_score:.4f}")
        else:
            model.save_pretrained(output_dir)

    print("\nTraining Complete!")

if __name__ == "__main__":
    main()







