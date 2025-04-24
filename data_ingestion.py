# Populates Qdrant DB
from data_loader import DataLoader
from config import config
from qdrant_client.models import PointStruct
from clip_encoder import CLIPEncoder
from qdrant_manager import QdrantManager
from tqdm import tqdm
import uuid
import numpy as np

def main():
    dl = DataLoader()
    encoder = CLIPEncoder()
    qdrant = QdrantManager()
    
    batch = []
    for idx in tqdm(range(len(dl))):
        try:
            sample = dl.get_sample(idx)
            
            # Generate hybrid embedding
            img_emb = encoder.encode_image(sample["image"])
            txt_emb = encoder.encode_text(sample["productDisplayName"])
            combined_emb = (img_emb + txt_emb) / 2
            
            batch.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=combined_emb.tolist(),
                    payload={"image_id": idx}
                )
            )
            
            if len(batch) >= config.BATCH_SIZE:
                qdrant.upsert_batch(batch)
                batch = []
        except Exception as e:
            print(f"Skipping corrupt sample {idx}: {str(e)}")
    
    if batch:
        qdrant.upsert_batch(batch)

if __name__ == "__main__":
    main()