FROM python:3.10-slim

WORKDIR /app

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-cache Hugging Face model (avoids 429 download errors later)
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy all app files (including templates folder)
COPY . .

# Expose Cloud Run port
ENV PORT=8080
CMD ["python", "-m", "uvicorn", "main_cloud:app", "--host", "0.0.0.0", "--port", "8080"]
