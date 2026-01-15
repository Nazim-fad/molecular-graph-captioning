import sys
import os
import json
import yaml
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(project_root)

# Config
CONFIG_PATH = os.path.join(project_root, "configs/llm_rag.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class PromptDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def load_data(jsonl_path):
    print(f"Loading data from {jsonl_path}...")
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    print(f"Loaded {len(data)} samples.")
    return data

def main():
    print(f"--- Batched LLM Inference ({config['model']['name']}) ---")

    # Quantized Model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    print("Loading Tokenizer & Model...")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    
    tokenizer.padding_side = "left" 
    tokenizer.pad_token = tokenizer.eos_token 
    
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'],
        quantization_config=bnb_config,
        device_map="auto" 
    )

    raw_data = load_data(os.path.join(project_root, config['paths']['test_dataset']))
    dataset = PromptDataset(raw_data)
    
    BATCH_SIZE = config.get('generation', {}).get('batch_size', 4)
    print(f"Inference Batch Size: {BATCH_SIZE}")

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    results = []
    
    print("Starting Generation...")
    
    for batch in tqdm(loader, desc="Generating"):
        prompts = batch['prompt']
        ids = batch['id']


        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        
        input_length = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config['generation']['max_new_tokens'],
                temperature=config['generation']['temperature'],
                top_p=config['generation']['top_p'],
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )


        generated_tokens = outputs[:, input_length:]
        
        decoded_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        for i, text in enumerate(decoded_texts):
            clean_text = text.replace("Description:", "").strip()
            
            results.append({
                "ID": ids[i],
                "description": clean_text
            })

    output_path = os.path.join(project_root, config['paths']['output_csv'])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} baseline predictions to {output_path}")

if __name__ == "__main__":
    main()