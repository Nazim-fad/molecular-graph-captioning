import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

class MolT5Dataset(Dataset):
    def __init__(self, graph_dataset, tokenizer, max_len=128):
        """
        Args:
            graph_dataset: Your existing PreprocessedGraphDataset
            tokenizer: The T5 Tokenizer
            max_len: Max sequence length for text
        """
        self.graph_dataset = graph_dataset
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.graph_dataset)

    def __getitem__(self, idx):
        graph = self.graph_dataset[idx]
        
        desc = getattr(graph, "description", "")
        
        tokenized = self.tokenizer(
            desc, 
            max_length=self.max_len, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        
        # T5 trick
        labels = tokenized.input_ids.squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "graph": graph,
            "labels": labels,
            "attention_mask": tokenized.attention_mask.squeeze()
        }

def collate_molt5(batch):
    """
    Custom function to batch Graphs AND Text together.
    """
    graphs = [item['graph'] for item in batch]
    labels = [item['labels'] for item in batch]
    masks = [item['attention_mask'] for item in batch]

    ids = [g.id for g in graphs]
    
    return {
        "batch_graph": Batch.from_data_list(graphs),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(masks),
        "ids": ids
    }