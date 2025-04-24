# Configuration constants
from pathlib import Path

class Config:
    DATASET_NAME = "ashraq/fashion-product-images-small"
    QDRANT_PATH = "./qdrant_data"  # Persistent storage
    COLLECTION_NAME = "fashion_products"
    BATCH_SIZE = 100
    IMAGE_SIZE = (224, 224)  # CLIP input size

config = Config()