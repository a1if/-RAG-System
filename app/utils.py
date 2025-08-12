import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
import re
from nltk.tokenize import sent_tokenize


def mmr_select(
    query_embedding: np.ndarray,     # Renamed to match pipeline.py
    document_embeddings: np.ndarray, # Renamed to match pipeline.py
    lambda_param: float = 0.5,       # Kept as is
    k: int = 3                       # Renamed to match pipeline.py's 'k'
) -> List[int]: # Changed return type to List[int] as pipeline expects indices
    """
    Selects a diverse set of top-k documents using Maximal Marginal Relevance (MMR).
    This version returns the indices of the selected documents from the
    original document_embeddings array.

    Args:
        query_embedding (np.ndarray): Embedding of the query (1D or 2D array, e.g., [1, embedding_dim]).
        document_embeddings (np.ndarray): Document embeddings of shape (n, dim).
        lambda_param (float): Trade-off parameter between relevance and diversity (0 to 1).
                              0 = purely diversity, 1 = purely relevance.
        k (int): The number of top documents (indices) to select.

    Returns:
        List[int]: A list of indices of the selected diverse and relevant documents.
    """
    if k <= 0 or not document_embeddings.size:
        return []

    # Ensure query_embedding is 2D
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    
    # Calculate initial relevance scores (cosine similarity between query and documents)
    # Ensure both inputs to cosine_similarity are 2D
    query_doc_similarity = cosine_similarity(query_embedding, document_embeddings)[0]

    selected_indices = []
    
    # Start with the index of the most relevant document
    if document_embeddings.shape[0] > 0:
        selected_indices.append(np.argmax(query_doc_similarity))
    else:
        return [] # No documents to select from

    # Continue selecting until k documents are chosen or no more documents remain
    while len(selected_indices) < min(k, document_embeddings.shape[0]):
        remaining_indices = list(set(range(document_embeddings.shape[0])) - set(selected_indices))
        if not remaining_indices:
            break # No more documents to select

        mmr_scores = []
        for idx in remaining_indices:
            relevance = query_doc_similarity[idx] # Get relevance score directly

            # Calculate redundancy: max similarity to already selected documents
            redundancy = 0.0
            if selected_indices: # Only calculate if there are already selected documents
                current_doc_embedding = document_embeddings[idx].reshape(1, -1) # Ensure 2D
                selected_embeddings = document_embeddings[selected_indices] # These are already 2D if from np.ndarray
                
                if selected_embeddings.size > 0 and current_doc_embedding.size > 0:
                     redundancy = np.max(cosine_similarity(current_doc_embedding, selected_embeddings))
            
            mmr_score = lambda_param * relevance - (1 - lambda_param) * redundancy
            mmr_scores.append((mmr_score, idx))

        if not mmr_scores:
            break # No scores to compare

        # Select the document with the highest MMR score from the remaining ones
        next_idx = max(mmr_scores, key=lambda x: x[0])[1] # x[0] is mmr_score, x[1] is original index
        selected_indices.append(next_idx)

    return selected_indices


def clean_chunk_text(text: str) -> str:
    """
    Clean OCR-extracted text for semantic relevance before chunking.
    Removes table artifacts, numeric-only lines, and repetitive exam/formatting noise.
    """
    if not text:
        return ""

    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 1. Remove markdown table formatting completely
        if line.startswith("|") and line.endswith("|") and "|" in line:
            continue
        if re.match(r'^\|\s*[-—]+\s*(\|\s*[-—]+\s*)+\|$', line):
            continue

        # 2. Remove exam question numbering
        line = re.sub(r'^\(?\d{1,3}[).]\s*', '', line)

        # 3. Skip lines that are mostly numbers or punctuation
        total_chars = len(line)
        if total_chars > 0:
            non_alpha = sum(1 for ch in line if not ch.isalpha() and not '\u0980' <= ch <= '\u09FF')
            if non_alpha / total_chars > 0.5:
                continue

        # 4. Remove lines with too few actual letters (garbled OCR noise)
        bengali_chars = sum(1 for ch in line if '\u0980' <= ch <= '\u09FF')
        english_chars = sum(1 for ch in line if ch.isalpha() and not '\u0980' <= ch <= '\u09FF')
        if bengali_chars + english_chars < 4:
            continue

        cleaned_lines.append(line)

    # 5. Join and normalize whitespace
    cleaned_text = re.sub(r'\s+', ' ', "\n".join(cleaned_lines)).strip()

    return cleaned_text



def smart_chunk_by_sentence(text: str, max_chars: int = 400, overlap: int = 100) -> List[str]:
    """
    Splits text into chunks by sentence with optional overlap.
    - Bangla-aware: if Bengali script is detected, split on Bangla punctuation (। ? !).
    - Falls back to NLTK's sent_tokenize otherwise.
    """
    text = re.sub(r"\s+", " ", text).strip()

    # Detect presence of Bengali characters
    contains_bangla = bool(re.search(r"[\u0980-\u09FF]", text))

    if contains_bangla:
        # Split using Bangla danda and common punctuation
        sentences = [s.strip() for s in re.split(r"(?<=[।?!])\s+", text) if s.strip()]
    else:
        # Use NLTK for non-Bangla
        sentences = sent_tokenize(text)

    # If tokenizer failed (e.g., single long sentence), hard-wrap every max_chars
    if len(sentences) <= 1 and len(text) > max_chars:
        sentences = [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

    chunks: List[str] = []
    current_chunk: str = ""

    for sentence in sentences:
        if not current_chunk:
            current_chunk = sentence
            continue

        if len(current_chunk) + 1 + len(sentence) > max_chars:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    if overlap > 0 and len(chunks) > 1:
        overlapped_chunks: List[str] = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
                continue
            prev_chunk = chunks[i - 1]
            overlap_text = prev_chunk[-overlap:] if len(prev_chunk) > overlap else prev_chunk
            overlapped_chunks.append((overlap_text + " " + chunk).strip())
        return overlapped_chunks

    return [c.strip() for c in chunks]


