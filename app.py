# Streamlit UI
import streamlit as st
from PIL import Image
from io import BytesIO
import numpy as np
from data_loader import DataLoader
from clip_encoder import CLIPEncoder
from qdrant_manager import QdrantManager

# Set page configuration (must be the first Streamlit command)
st.set_page_config(page_title="Fashion Finder", layout="wide")

@st.cache_resource
def load_components():
    return {
        "data": DataLoader(),
        "encoder": CLIPEncoder(),
        "qdrant": QdrantManager()
    }

components = load_components()

# UI Setup
st.title("👗 AI-Powered Fashion Search")

# Search Type
search_type = st.radio("Search using:", ["Image", "Text"], horizontal=True)

# Handle Inputs
query_emb = None
if search_type == "Image":
    uploaded_file = st.file_uploader("Upload product image", type=["jpg", "png"])
    if uploaded_file:
        query_image = Image.open(BytesIO(uploaded_file.read()))
        query_emb = components["encoder"].encode_image(query_image)
else:
    text_query = st.text_input("Describe your desired item:")
    if text_query:
        query_emb = components["encoder"].encode_text(text_query)

# Display Results
if query_emb is not None:
    with st.spinner("Finding similar products..."):
        results = components["qdrant"].search(query_emb.tolist(), limit=3)
    
    cols = st.columns(4)
    cols[0].image(
        query_image if search_type == "Image" else "https://via.placeholder.com/224",
        width=224,
        caption="Your Query"
    )
    
    for idx, hit in enumerate(results, start=1):
        product = components["data"].get_sample(hit.payload["image_id"])
        cols[idx].image(
            product["image"],
            width=224,
            caption=f"Score: {hit.score:.2f}\n{product['productDisplayName'][:30]}..."
        )