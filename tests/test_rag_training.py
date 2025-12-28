import sys
import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, AutoTokenizer
from peft import LoraConfig, get_peft_model

sys.path.append(".")
from src.rag_dataset import RAGGenerationDataset

def test_rag_train_step():
    print("Testing RAG Generator Training")
    
    # Dummy JSONL
    dummy_jsonl = "tests/dummy_train.jsonl"
    with open(dummy_jsonl, "w") as f:
        f.write('{"id": "1", "prompt": "Describe this molecule:", "target": "benzene."}\n')
        f.write('{"id": "2", "prompt": "Summarize:", "target": "Water molecule."}\n')

    # Load Dataset
    tokenizer_name = "gpt2"
    ds = RAGGenerationDataset(dummy_jsonl, tokenizer_name=tokenizer_name, max_length=32)
    loader = DataLoader(ds, batch_size=2)
    
    batch = next(iter(loader))
    
    # 3. Init Model with LoRA
    model = GPT2LMHeadModel.from_pretrained(tokenizer_name)
    peft_config = LoraConfig(
        r=4, lora_alpha=16, target_modules=["c_attn"], task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    
    # 4. Forward Pass
    outputs = model(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        labels=batch['labels']
    )
    
    loss = outputs.loss
    print(f"Forward pass successful. Loss: {loss.item():.4f}")
    
    loss.backward()
    print("Backward pass successful.")
    
    # Cleanup
    import os
    os.remove(dummy_jsonl)

if __name__ == "__main__":
    test_rag_train_step()