import json
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class RAGGenerationDataset(Dataset):
    """
    Dataset for training the Decoder in a RAG setup.
    """

    def __init__(
        self,
        jsonl_path,
        tokenizer_name,  
        max_length     
    ):
        super().__init__()
        
        self.max_length = max_length
        self.samples = []

        print(f"Loading generation data from {jsonl_path}...")
        try:
            with open(jsonl_path, "r") as f:
                for line in f:
                    try:
                        self.samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            print(f"Error: Dataset file not found at {jsonl_path}")
        
        print(f"Loaded {len(self.samples)} samples.")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        prompt_text = sample["prompt"]
        target_text = sample["target"]

        # Concatenate prompt and target with EOS token
        full_text = f"{prompt_text} {target_text}{self.tokenizer.eos_token}"

        enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # Mask prompt in labels
        prompt_enc = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        prompt_len = prompt_enc["input_ids"].shape[1]

        labels = input_ids.clone()
        
        if prompt_len < len(labels):
            labels[:prompt_len] = -100
        else:
            labels[:] = -100

        labels[input_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }