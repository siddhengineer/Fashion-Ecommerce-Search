# Use official Python base
FROM python:3.10-slim

# Install system deps for CLIP/Qdrant
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libjpeg62-turbo \
    && rm -rf /var/lib/apt/lists/*

# Set working dir
WORKDIR /app

# Copy only necessary files
COPY requirements.txt .
COPY *.py ./
COPY streamlit_app/ ./streamlit_app/

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Persist Qdrant data and HuggingFace cache
VOLUME ["/app/qdrant_data", "/root/.cache/huggingface"]

# Single command to load data & start app
CMD ["sh", "-c", "python data_ingestion.py && streamlit run streamlit_app/app.py --server.port=8501 --server.address=0.0.0.0"]