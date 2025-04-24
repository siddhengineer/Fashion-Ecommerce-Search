# CLIP model wrapper
from sentence_transformers import SentenceTransformer
from PIL import Image
from config import config

class CLIPEncoder:
    def __init__(self):
        self.model = SentenceTransformer("clip-ViT-B-32")
    
    def encode_image(self, image: Image.Image):
        return self.model.encode(image)
    
    def encode_text(self, text: str):
        return self.model.encode(text)