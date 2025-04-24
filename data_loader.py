# Handles dataset loading
from datasets import load_dataset
from config import config

class DataLoader:
    def __init__(self):
        self.dataset = load_dataset(config.DATASET_NAME, split="train[:5000]")
    
    def __len__(self):
        return len(self.dataset)
    
    def get_sample(self, idx):
        return self.dataset[idx]