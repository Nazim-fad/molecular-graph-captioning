import json
from transformers import GPT2Tokenizer
from torch.utils.data import Dataset
import torch
from torch_geometric.data import Batch


from data_utils import (
    PreprocessedGraphDataset,
    load_descriptions_from_graphs,
    load_id2emb
)

class RAGDataset(Dataset):
    """
    Retrieval-Augmented Generation dataset.

    For each molecule:
      - retrieve top-K similar molecule descriptions
      - build a textual prompt
      - append ground-truth description as target
      - tokenize for GPT-2
    """

    def __init__(
        self,
        graph_path,
        train_graph_path,
        train_emb_csv,
        mol_encoder,
        tokenizer_name="gpt2-medium",
        top_k=5,
        max_length=896,
        device="cuda"
    ):
        super().__init__()

        self.device = device
        self.top_k = top_k
        self.max_length = max_length

        # Load tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load graphs
        self.graph_dataset = PreprocessedGraphDataset(graph_path)
        self.train_id2desc = load_descriptions_from_graphs(train_graph_path)
        self.train_emb_dict = load_id2emb(train_emb_csv)

        # Prepare train embeddings tensor
        self.train_ids = list(self.train_emb_dict.keys())
        self.train_embs = torch.stack(
            [self.train_emb_dict[i] for i in self.train_ids]
        ).to(device)
        self.train_embs = torch.nn.functional.normalize(self.train_embs, dim=-1)

        # Freeze mol encoder
        self.mol_encoder = mol_encoder.to(device)
        self.mol_encoder.eval()
        for p in self.mol_encoder.parameters():
            p.requires_grad = False

    def __len__(self):
        return len(self.graph_dataset)

    def _build_prompt(self, retrieved_texts):
        """
        Build the RAG prompt from retrieved descriptions.
        """
        prompt = (
            "You are a chemistry expert.\n\n"
            "Below are descriptions of molecules that are structurally and "
            "functionally similar to a target molecule:\n\n"
        )

        for i, txt in enumerate(retrieved_texts, 1):
            prompt += f"[{i}] {txt}\n"

        prompt += (
            "\nBased on the information above, write a concise and accurate "
            "description of the target molecule.\n\n"
            "Description:"
        )
        return prompt

    def __getitem__(self, idx):
        graph = self.graph_dataset[idx]

        batch = Batch.from_data_list([graph]).to(self.device)

        # Encode graph
        with torch.no_grad():
            graph_emb = self.mol_encoder(batch)
            graph_emb = torch.nn.functional.normalize(graph_emb, dim=-1)

        # Retrieve top-K captions
        sims = graph_emb @ self.train_embs.t()
        topk_idx = sims.topk(self.top_k, dim=-1).indices.squeeze(0)

        retrieved_texts = [
            self.train_id2desc[self.train_ids[i]]
            for i in topk_idx.tolist()
        ]

        # Build prompt
        prompt_text = self._build_prompt(retrieved_texts)
        target_text = graph.description

        # Tokenize prompt + target
        full_text = prompt_text + " " + target_text
        enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # Build labels (mask prompt tokens)
        prompt_len = len(
            self.tokenizer(
                prompt_text,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )["input_ids"].squeeze(0)
        )

        labels = input_ids.clone()
        labels[:prompt_len] = -100  # ignore prompt tokens

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

class CachedRAGDataset(Dataset):
    """
    Dataset that loads precomputed RAG prompt–target pairs.
    """

    def __init__(
        self,
        cache_path,
        tokenizer_name="gpt2-medium",
        max_length=896
    ):
        super().__init__()

        self.max_length = max_length
        self.samples = []

        with open(cache_path, "r") as f:
            for line in f:
                self.samples.append(json.loads(line))

        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loaded {len(self.samples)} cached RAG samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        prompt_text = sample["prompt"]
        target_text = sample["target"]

        full_text = prompt_text + " " + target_text

        enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # Mask prompt tokens
        prompt_len = len(
            self.tokenizer(
                prompt_text,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )["input_ids"].squeeze(0)
        )

        labels = input_ids.clone()
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
