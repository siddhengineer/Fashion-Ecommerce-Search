# 👗 AI-Powered Fashion Search Engine

A multimodal search system that helps users find fashion products using either images or text queries — or both. Powered by [CLIP](https://openai.com/research/clip) for embedding and [Qdrant](https://qdrant.tech/) for fast vector search.

---

## 🚀 Features

- 🔍 **Image-to-Image Search**  
  Upload a fashion product image to discover visually similar items.
- 📝 **Text-to-Image Search**  
  Use natural language prompts to search for products (e.g. _“Red formal dress”_).

---

## 📦 Dataset

**Source**: [ashraq/fashion-product-images-small](https://huggingface.co/datasets/ashraq/fashion-product-images-small) on Hugging Face Datasets.

**Sample Entry**:

```python
{
    "image": PIL.Image,           # RGB image
    "productDisplayName": str,    # e.g. "Men Navy Blue Striped Shirt"
    "articleType": str,           # e.g. "Shirt", "Dress"
    "usage": str,                 # e.g. "Casual", "Sports"
}
```

---

## 🧠 Tech Stack

| Component       | Technology                                         |
| --------------- | -------------------------------------------------- |
| Embedding Model | `clip-ViT-B-32` via `SentenceTransformers`         |
| Vector DB       | [Qdrant](https://qdrant.tech/) (Local, persistent) |
| UI Framework    | [Streamlit](https://streamlit.io/)                 |

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/siddhengineer/Fashion-Ecommerce-Search.git

```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install CLIP Model

```bash
pip install sentence-transformers torch
```

---

## 🔀 Workflow

### Step 1: Data Ingestion

```bash
python data_ingestion.py
```

- Loads 5,000 samples from Hugging Face
- Generates hybrid embeddings (image + product title)
- Stores in Qdrant with associated metadata

### Step 2: Run the Search Interface

```bash
streamlit run app.py
```

---

## 🔍 Search Modes

### 🖼️ Image Search

- Upload a product image.
- CLIP encodes it to a 512-dim vector.
- Qdrant returns the top visually similar products.

### ✍️ Text Search

- Enter a description like `"Red floral dress"` or `"Men’s white sneakers"`.
- Get results with a matching visual profile.

### 🧠 Hybrid Search

```python
img_emb = model.encode(image)
txt_emb = model.encode("blue jeans")

```

---

## 🧽 Query Process

1. User uploads an image and/or types a text query.
2. CLIP generates a 512D embedding.
3. Qdrant retrieves nearest vectors using cosine similarity.
4. Streamlit interface displays the top results with similarity scores.

---

## 📌 Future Enhancements

- 👕 Outfit recommendations
- 🎨 Color-based filtering
- 💬 Chatbot-based fashion assistant (image + prompt input)
- 🛒 E-commerce integration

---
