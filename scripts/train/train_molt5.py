import sys
import os
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup


script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(project_root)

from src.model_molt5 import MolT5Generator
from src.dataset_molt5 import MolT5Dataset, collate_molt5
from src.losses import clip_contrastive_loss
from src.data_utils import PreprocessedGraphDataset

# CONFIG
CONFIG_PATH = os.path.join(project_root, "configs/molt5_train.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    print(f"--- Training MolT5 ({config['model']['name']}) ---")
    print(f"Device: {DEVICE}")

    print("Loading Preprocessed Graphs...")
    train_graphs = PreprocessedGraphDataset(config['paths']['train_graphs'])
    val_graphs = PreprocessedGraphDataset(config['paths']['val_graphs'])

    model = MolT5Generator(
        gnn_hidden=config['model']['gnn_hidden'],
        model_name=config['model']['name']
    ).to(DEVICE)

    resume_path = config['training'].get('resume_checkpoint', None)
    if resume_path and os.path.exists(os.path.join(project_root, resume_path)):
        full_path = os.path.join(project_root, resume_path)
        print(f"Resuming training from checkpoint: {full_path}")

        checkpoint = torch.load(full_path, map_location=DEVICE)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

    else:
        print("Starting training from scratch (no valid checkpoint found in config).")


    print("Preparing Datasets (Tokenizing)...")
    train_ds = MolT5Dataset(train_graphs, model.tokenizer, max_len=config['model']['max_length'])
    val_ds = MolT5Dataset(val_graphs, model.tokenizer, max_len=config['model']['max_length'])

    train_loader = DataLoader(
        train_ds, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        collate_fn=collate_molt5,
        num_workers=2
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=config['training']['batch_size'], 
        shuffle=False, 
        collate_fn=collate_molt5,
        num_workers=2
    )

    optimizer = AdamW(model.parameters(), lr=float(config['training']['lr']))
    
    total_steps = len(train_loader) * config['training']['epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=100, 
        num_training_steps=total_steps
    )

    best_val_loss = float('inf')
    save_dir = os.path.join(project_root, config['paths']['checkpoint_dir'])
    os.makedirs(save_dir, exist_ok=True)

    contrastive_weight = config['training']['contrastive_weight']

    for epoch in range(config['training']['epochs']):
        model.train()
        train_loss = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")
        for batch in loop:
            batch_graph = batch['batch_graph'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)

            optimizer.zero_grad()

            gen_loss, graph_vec, text_vec, logit_scale = model(batch_graph, labels, mask)

            cont_loss = clip_contrastive_loss(
                            graph_vec, 
                            text_vec, 
                            logit_scale=logit_scale,  
                            label_smoothing=0.0
                        )
            
            total_loss = gen_loss + (contrastive_weight * cont_loss)

            total_loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += total_loss.item()
            loop.set_postfix({
                "Gen": f"{gen_loss.item():.2f}", 
                "CLIP": f"{cont_loss.item():.2f}",
                "Temp": f"{1/logit_scale.item():.2f}"
            })

        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch_graph = batch['batch_graph'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)
                mask = batch['attention_mask'].to(DEVICE)
                
                loss, _, _, _ = model(batch_graph, labels, mask)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(save_dir, "best_model.pt")
            torch.save(model.state_dict(), save_path)
            print(f"Saved Best Model to {save_path}")
            
        last_path = os.path.join(save_dir, "last_model.pt")
        torch.save(model.state_dict(), last_path)

if __name__ == "__main__":
    main()