from pathlib import Path

FAISS_INDEX_PATH = Path("vectorstore/faiss_index.bin") # MUST be Path object
CHUNKS_PATH = Path("vectorstore/chunks.json")       # MUST be Path object
#MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_NAME = "intfloat/multilingual-e5-large"
#MODEL_NAME = "models/embedding-001"
