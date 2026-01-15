import sys
import os
import yaml
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(project_root)

from src.model_molt5 import MolT5Generator
from src.dataset_molt5 import MolT5Dataset, collate_molt5
from src.data_utils import PreprocessedGraphDataset

# CONFIG
CONFIG_PATH = os.path.join(project_root, "configs/molt5_train.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    print("--- Generating Submissions with MolT5 ---")
    
    checkpoint_path = os.path.join(project_root, config['paths']['checkpoint_dir'], "best_model.pt")
    test_data_path = config['paths']['test_graphs']
    
    output_csv = os.path.join(project_root, config['paths']['test_output'])
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    if not os.path.exists(checkpoint_path):
        print(f"Error: Model not found at {checkpoint_path}. Train first!")
        return

    print(f"Loading model from {checkpoint_path}...")
    model = MolT5Generator(
        gnn_hidden=config['model']['gnn_hidden'],
        model_name=config['model']['name']
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()

    print("Loading Test Graphs...")
    test_graphs = PreprocessedGraphDataset(test_data_path)
    
    test_ds = MolT5Dataset(test_graphs, model.tokenizer, max_len=config['model']['max_length'])
    
    test_loader = DataLoader(
        test_ds, 
        batch_size=32, 
        shuffle=False, 
        collate_fn=collate_molt5,
        num_workers=2
    )

    results = []
    print("Starting Generation...")
    
    gen_max_len = config['model']['max_length']
    gen_num_beams = config['model']['num_beams']

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            batch_graph = batch['batch_graph'].to(DEVICE)
            
            texts = model.generate(
                batch_graph, 
                max_length=gen_max_len,   
                num_beams=gen_num_beams
            )
            
            batch_ids = batch['ids']
            
            for i, text in enumerate(texts):
                results.append({
                    "ID": batch_ids[i],
                    "description": text
                })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} predictions to {output_csv}")

if __name__ == "__main__":
    main()