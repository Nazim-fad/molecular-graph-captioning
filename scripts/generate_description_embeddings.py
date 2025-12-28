"""Generate BERT embeddings for molecular descriptions."""
import os
import pickle
import sys
import yaml
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


sys.path.append(".")

# Configuration
with open("configs/data_prep.yaml", "r") as f:
    config = yaml.safe_load(f)


# Load BERT model
print("Loading BERT model...")
model_name = config.get('text_encoder', 'bert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()
print(f"Model loaded on: {device}")

# Directories from config
input_dir = config['paths']['graphs']
output_dir = config['paths']['embeddings']

# Process each split
for split in ['train', 'validation']:
    print(f"\nProcessing {split}...")

    # Paths 
    pkl_path = os.path.join(input_dir, f'{split}_graphs.pkl')
    output_path = os.path.join(output_dir, f'{split}_embeddings.csv')

    # Check if input exists
    if not os.path.exists(pkl_path):
        print(f"Skipping {split}: File not found at {pkl_path}")
        continue
    
    # Load graphs from pkl file
    print(f"Loading from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        graphs = pickle.load(f)
    print(f"Loaded {len(graphs)} graphs")
    
    # Generate embeddings
    ids = []
    embeddings = []
    
    for graph in tqdm(graphs, total=len(graphs)):
        # Get description from graph
        description = graph.description
        
        # Tokenize
        max_length = config.get('max_token_length')
        inputs = tokenizer(description, return_tensors='pt', 
                          truncation=True, max_length=max_length, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get embedding
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
        
        ids.append(graph.id)
        embeddings.append(embedding)
    
    # Save to CSV
    result = pd.DataFrame({
        'ID': ids,
        'embedding': [','.join(map(str, emb)) for emb in embeddings]
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Saved embeddings to {output_path}")

print("\nDone!")

