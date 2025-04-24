# Vector DB operations
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from config import config

class QdrantManager:
    def __init__(self):
        self.client = QdrantClient(path=config.QDRANT_PATH)
        self._create_collection()
    
    def _create_collection(self):
        if not self.client.collection_exists(config.COLLECTION_NAME):
            self.client.create_collection(
                collection_name=config.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=512,  # CLIP embedding size
                    distance=Distance.COSINE
                )
            )
    
    def upsert_batch(self, points):
        self.client.upsert(
            collection_name=config.COLLECTION_NAME,
            points=points
        )
    
    def search(self, query_vector, limit=3):
        return self.client.search(
            collection_name=config.COLLECTION_NAME,
            query_vector=query_vector,
            limit=limit
        )