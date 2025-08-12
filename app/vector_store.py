import os
import faiss
import pickle
from typing import List, Tuple, Optional, Dict # Consolidated and corrected typing imports
from app.config import FAISS_INDEX_PATH, CHUNKS_PATH
from pathlib import Path


VECTOR_STORE_DIR = Path("./vectorstore")
VECTOR_STORE_DIR.mkdir(exist_ok=True)

def save_vector_db(index: Optional[faiss.Index], chunks: List[Dict[str, str]], session_id: str) -> None:
    """
    Saves the FAISS index and text chunks to local disk using a unique session ID.
    Handles deletion of files if the vector store becomes empty after an operation.
    """
    print(f"DEBUG: [vector_store] Entering save_vector_db for session: {session_id}")
    faiss_index_path = VECTOR_STORE_DIR / f"index_{session_id}.faiss"
    chunks_path = VECTOR_STORE_DIR / f"chunks_{session_id}.pkl"

    if index is not None and hasattr(index, 'ntotal') and index.ntotal > 0:
        faiss.write_index(index, str(faiss_index_path))
        print(f"DEBUG: [vector_store] FAISS index saved to: {faiss_index_path}. Total vectors: {index.ntotal}")
    elif faiss_index_path.exists():
        os.remove(faiss_index_path)
        print(f"DEBUG: [vector_store] Existing FAISS index file deleted: {faiss_index_path}")

    # Chunks are always saved to reflect the current state
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"DEBUG: [vector_store] Chunks saved to: {chunks_path}. Number of chunks: {len(chunks)}")

def load_vector_db(session_id: str) -> Tuple[Optional[faiss.Index], List[Dict[str, str]]]:
    """
    Loads the FAISS index and text chunks from local disk for a specific session ID.
    """
    print(f"DEBUG: [vector_store] Entering load_vector_db for session: {session_id}")
    faiss_index_path = VECTOR_STORE_DIR / f"index_{session_id}.faiss"
    chunks_path = VECTOR_STORE_DIR / f"chunks_{session_id}.pkl"
    
    index = None
    chunks = []
    
    if faiss_index_path.exists():
        index = faiss.read_index(str(faiss_index_path))
        print(f"✅ [vector_store] FAISS index loaded for session {session_id}. Total vectors: {index.ntotal}")
    else:
        print(f"❗ [vector_store] No FAISS index found for session {session_id}.")

    if chunks_path.exists():
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)
        print(f"✅ [vector_store] Chunks loaded for session {session_id}. Number of chunks: {len(chunks)}")
    else:
        print(f"❗ [vector_store] No chunks file found for session {session_id}.")
        
    return index, chunks