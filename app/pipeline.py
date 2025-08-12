import os
from dotenv import load_dotenv
from app.config import MODEL_NAME
from app.document_loader import extract_text_from_document
from app.vector_store import load_vector_db, save_vector_db
from app.utils import mmr_select, smart_chunk_by_sentence,clean_chunk_text
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai
import time
from pathlib import Path
import hashlib
from typing import List, Optional, Dict, Generator
import json
import nltk
import traceback
import re
import dataclasses
from app.web_search import web_search
from datetime import datetime, timedelta
from groq import Groq
import collections
from rank_bm25 import BM25Okapi


WEB_SEARCH_TTL = timedelta(hours=1)


@dataclasses.dataclass
class WebSearchCache:
    """Stores the results of the last web search."""
    query: str
    context: str
    sources: List[Dict[str, str]]
    timestamp: datetime = dataclasses.field(default_factory=datetime.now)

@dataclasses.dataclass
class SessionData:
    index: Optional[faiss.Index] = None
    chunks: List[Dict] = dataclasses.field(default_factory=list)
    embeddings: Optional[np.ndarray] = None
    bm25_index: Optional[BM25Okapi] = None
    loaded_documents: Dict[str, str] = dataclasses.field(default_factory=dict)
    web_search_cache: Optional[WebSearchCache] = None

_model_cache = None
_loading_status = False

CACHE_DIR = Path(".cache") 
CACHE_DIR.mkdir(exist_ok=True) 

def get_rag_model():
    """Loads and caches the SentenceTransformer model."""
    global _model_cache
    if _model_cache is None:
        print("INFO: [model_loader] Loading SentenceTransformer model...")
        try:
            _model_cache = SentenceTransformer(MODEL_NAME, cache_folder=str(CACHE_DIR))
            print("✅ [model_loader] SentenceTransformer model loaded successfully.")
        except Exception as e:
            print(f"❌ ERROR: Failed to load SentenceTransformer model: {e}")
            traceback.print_exc()
            raise RuntimeError(f"Failed to load embedding model: {e}")
    return _model_cache

WEB_CACHE_DIR = Path(".cache/web_search") 
WEB_CACHE_DIR.mkdir(exist_ok=True, parents=True) 
WEB_CACHE_TTL = timedelta(hours=1)
NON_SEARCHABLE_PATTERNS = [
    r"^\s*(hello|hi|hey)\s*$",
    r"^\s*(thanks|thank you)\s*$",
    r"^\s*(what can you do|help)\s*$",
    r"^\s*(how does this work)\s*$",
]

def simplify_query(query: str) -> str:
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'can'}
    words = re.findall(r'\b\w+\b', query.lower())
    simplified_words = [word for word in words if word not in stop_words]
    return ' '.join(simplified_words)

def should_combine_or_break_down(query: str) -> str:
    return simplify_query(query)


class RAGPipeline:
    def __init__(self):
        load_dotenv()
        nltk.download('punkt')
        global _loading_status
        if not _loading_status:
            _loading_status = True
            get_rag_model()
            _loading_status = False
        self.model = _model_cache
        self.sessions: Dict[str, SessionData] = {}
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            # This is fine now, since we have a Groq fallback
            print("❗ Warning: Gemini API key is missing. Only Groq will be available.")
            self.gmodel = None
        else:
            self.gmodel = genai.GenerativeModel("gemini-1.5-flash")
        
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            self.groq_client = Groq(api_key=groq_key)
        else:
            self.groq_client = None

    def _load_session_data(self, session_id: str) -> SessionData:
        print(f"DEBUG: [pipeline] Attempting to load session data for ID: {session_id}")
        session_data = SessionData()
        try:
            faiss_file_path = Path(f"vectorstore/index_{session_id}.faiss")
            chunks_file_path = Path(f"vectorstore/chunks_{session_id}.pkl")
            if not faiss_file_path.exists() or not chunks_file_path.exists():
                print(f"❗ [pipeline] Session data files not found for ID: {session_id}. Initializing empty session.")
                dimension = self.model.get_sentence_embedding_dimension()
                # Use inner product index with normalized embeddings (cosine similarity)
                session_data.index = faiss.IndexFlatIP(dimension)
                return session_data

            index, chunks = load_vector_db(session_id)
            if index and chunks:
                texts = [chunk['text'] for chunk in chunks if 'text' in chunk]
                embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
                # Rebuild BM25 index from loaded chunks
                tokenized_corpus = [nltk.word_tokenize(doc['text']) for doc in chunks]
                session_data.bm25_index = BM25Okapi(tokenized_corpus)

                session_data.index = index
                session_data.chunks = chunks
                session_data.embeddings = embeddings
                for chunk in chunks:
                    doc_id = chunk.get("document_id")
                    if doc_id:
                        session_data.loaded_documents[doc_id] = chunk.get("document_name", "N/A")
                print(f"✅ [pipeline] Loaded {len(chunks)} chunks for session {session_id} from disk.")
        except Exception as e:
            print(f"❗ [pipeline] No existing data found for session {session_id} or error occurred: {e}. Initializing a new session.")
            dimension = self.model.get_sentence_embedding_dimension()
            session_data.index = faiss.IndexFlatL2(dimension)
            session_data.chunks = []
            session_data.embeddings = np.array([])
            session_data.bm25_index = BM25Okapi([[]])
        return session_data

    def _save_session_data(self, session_id: str, session_data: SessionData):
        print(f"DEBUG: [pipeline] Saving state for session: {session_id}")
        try:
            save_vector_db(session_data.index, session_data.chunks, session_id)
            print(f"✅ [pipeline] Session {session_id} state saved to disk.")
        except Exception as e:
            print(f"❌ [pipeline] Error saving session {session_id} state: {e}")
            traceback.print_exc()

    def rebuild_vector_store(self, pdf_bytes: bytes, session_id: str, file_name: str) -> str | None:
        print(f"DEBUG: [pipeline] Entered rebuild_vector_store for session: {session_id}")
        if session_id not in self.sessions:
            self.sessions[session_id] = self._load_session_data(session_id)
        session_data = self.sessions[session_id]
        try:
            pages = extract_text_from_document(pdf_bytes, file_name=file_name)
            if not pages:
                return None
            
            # We'll use this to check for duplicate documents later
            full_raw_text = "".join(p["text"] for p in pages)
            doc_hash = hashlib.md5(full_raw_text.encode()).hexdigest()
            
            if any(c.get("document_id") == doc_hash for c in session_data.chunks):
                print(f"🔄 Document already exists ({doc_hash}) for session {session_id} – skipping.")
                return doc_hash
            
            new_chunks = []
            for p in pages:
                # CORRECTED: Clean the text of each page before chunking
                cleaned_text = clean_chunk_text(p["text"])
                
                if not cleaned_text.strip():
                    continue
                
                if any(c.get("page_hash") == p["page_hash"] for c in session_data.chunks):
                    continue
                    
                chunks = smart_chunk_by_sentence(cleaned_text)
                for i, ch in enumerate(chunks):
                    new_chunks.append({
                        "document_id": doc_hash,
                        "document_name": file_name,
                        "chunk_id": f"{doc_hash}-{p['page']}-{i}",
                        "text": ch,
                        "page": p["page"],
                        "page_hash": p["page_hash"]              
                    })

            if not new_chunks:
                return None
            embs = self.model.encode([c["text"] for c in new_chunks], convert_to_numpy=True)
            # Normalize embeddings for cosine similarity and cast to float32
            if embs.size > 0:
                embs = embs.astype("float32")
                norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
                embs = embs / norms

            if session_data.index is None or not session_data.chunks:
                dim = embs.shape[1]
                if len(new_chunks) > 10_000:
                    quantizer = faiss.IndexFlatIP(dim)
                    session_data.index = faiss.IndexIVFFlat(quantizer, dim, 256)
                    session_data.index.train(embs)
                else:
                    session_data.index = faiss.IndexFlatL2(dim)
            session_data.index.add(embs)
            session_data.chunks.extend(new_chunks)
            if session_data.embeddings is None or session_data.embeddings.size == 0:
                session_data.embeddings = embs
            else:
                session_data.embeddings = np.vstack([session_data.embeddings, embs])
            session_data.loaded_documents[doc_hash] = file_name
            
            # Update BM25 index with new chunks
            new_tokenized_chunks = [nltk.word_tokenize(c['text']) for c in new_chunks]
            if session_data.bm25_index is None:
                 session_data.bm25_index = BM25Okapi(new_tokenized_chunks)
            else:
                 current_corpus = [nltk.word_tokenize(c['text']) for c in session_data.chunks if c.get("document_id") != doc_hash]
                 updated_corpus = current_corpus + new_tokenized_chunks
                 session_data.bm25_index = BM25Okapi(updated_corpus)
            
            self._save_session_data(session_id, session_data)
            print(f"✅ Document processed and added to session {session_id}. New chunk count: {len(session_data.chunks)}")
            return doc_hash
        except Exception as e:
            traceback.print_exc()
            return None  
    
    def delete_document_data(self, document_id: str, session_id: str, file_name: Optional[str] = None) -> bool:
        print(f"DEBUG: [pipeline] Entering delete_document_data for session {session_id}.")
        if session_id not in self.sessions:
            self.sessions[session_id] = self._load_session_data(session_id)
        session_data = self.sessions[session_id]
        if not session_data.chunks:
            print(f"❗ [pipeline] No chunks loaded for session {session_id}, cannot delete.")
            return False
        chunks_to_keep = []
        embeddings_to_keep = []
        deleted_count = 0
        for i, chunk in enumerate(session_data.chunks):
            if chunk.get("document_id") == document_id:
                deleted_count += 1
            else:
                chunks_to_keep.append(chunk)
                if session_data.embeddings is not None and i < len(session_data.embeddings):
                    embeddings_to_keep.append(session_data.embeddings[i])
        if deleted_count == 0:
            print(f"ℹ️ [pipeline] No chunks found for document ID: {document_id} in session {session_id}. Nothing to delete.")
            return False
        session_data.chunks = chunks_to_keep
        if embeddings_to_keep:
            session_data.embeddings = np.array(embeddings_to_keep).astype(np.float32)
            # Normalize kept embeddings for cosine similarity
            norms = np.linalg.norm(session_data.embeddings, axis=1, keepdims=True) + 1e-12
            session_data.embeddings = session_data.embeddings / norms
            dimension = session_data.embeddings.shape[1]
            session_data.index = faiss.IndexFlatL2(dimension)
            session_data.index.add(session_data.embeddings)
        else:
            dimension = self.model.get_sentence_embedding_dimension()
            session_data.index = faiss.IndexFlatL2(dimension)
            session_data.embeddings = np.array([])
        # Rebuild BM25 index with the remaining chunks
        if session_data.chunks:
            tokenized_corpus = [nltk.word_tokenize(doc['text']) for doc in session_data.chunks]
            session_data.bm25_index = BM25Okapi(tokenized_corpus)
        else:
            session_data.bm25_index = None
        self._save_session_data(session_id, session_data)
        print(f"🗑️ [pipeline] Successfully deleted {deleted_count} chunks for document ID: {document_id} in session {session_id}. Remaining chunks: {len(session_data.chunks)}.")
        return True
    
    def _delete_session_files(self, session_id: str) -> bool:
        print(f"DEBUG: [pipeline] Deleting all session files for ID: {session_id}.")
        try:
            chunks_file = Path(f"data/sessions/{session_id}.chunks.pkl")
            faiss_file = Path(f"data/sessions/{session_id}.faiss")
            
            chunks_deleted = chunks_file.exists() and os.remove(chunks_file)
            faiss_deleted = faiss_file.exists() and os.remove(faiss_file)
            
            if chunks_deleted or faiss_deleted:
                print(f"✅ [pipeline] Files for session {session_id} deleted successfully.")
                return True
            else:
                print(f"ℹ️ [pipeline] No files found for session {session_id}. Nothing to delete.")
                return False
        except Exception as e:
            print(f"❌ [pipeline] Error deleting files for session {session_id}: {e}")
            traceback.print_exc()
            return False

    def delete_all_documents(self, session_id: str) -> bool:
        print(f"DEBUG: [pipeline] Deleting all documents for session: {session_id}")
        if session_id in self.sessions:
            del self.sessions[session_id]
        return self._delete_session_files(session_id)
        
    def get_loaded_documents(self, session_id: Optional[str] = None) -> List[str]:
        if not session_id or session_id not in self.sessions:
            return []
        session_data = self.sessions[session_id]
        if not session_data.chunks:
            return []
        unique_documents = set()
        for chunk in session_data.chunks:
            document_name = chunk.get("document_name")
            if document_name:
                unique_documents.add(document_name)
        return sorted(list(unique_documents))

    def _get_llm_for_search(self, model_choice: str):
        """Selects the correct LLM for search-related tasks based on model_choice."""
        if model_choice == "groq" and self.groq_client:
            return self.groq_client
        else:
            return self.gmodel
        

    def _get_search_query(self, query: str, history: List[str], model_choice: str) -> str:
        combined_prompt = ""
        for i in range(0, len(history), 2):
            combined_prompt += f"User: {history[i]}\n"
            if i + 1 < len(history):
                combined_prompt += f"Model: {history[i+1]}\n"
        combined_prompt += f"User: {query}"

        # --- UPDATED PROMPT ---
        prompt_template = f"""
        You are a highly efficient search query generator. Your sole purpose is to create a concise and effective search query based on the user's conversation history and their most recent question.

        **CRITICAL INSTRUCTIONS:**
        1.  **ONLY return the search query.** Do not include any other text, explanations, or conversational phrases.
        2.  The query must be short and directly focused on what the user wants to know.
        3.  If the query requires a multi-word phrase, enclose it in double quotes (e.g., "UEFA Champions League winners").

        Example 1:
        Conversation history:
        User: Tell me about the political history of Bangladesh.
        Model: Bangladesh's political history is complex...
        User: what happened with this in past 1 year ?
        Search Query: "political history of Bangladesh in the past year"

        Example 2:
        Conversation history:
        User: What is the current stock price of Apple?
        Model: The stock price of Apple is a dynamic value...
        User: what about Google?
        Search Query: "Google stock price"

        Example 3:
        Conversation history:
        User: What are the main ingredients in a margarita?
        Model: A classic margarita is made with tequila, lime juice, and Cointreau...
        Model: how to make a spicy version?
        Search Query: "how to make a spicy margarita"

        Conversation history:
        {combined_prompt}
        Search Query:
        """
        try:
            llm = self._get_llm_for_search(model_choice)
            if isinstance(llm, Groq):
                response = llm.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[{"role": "user", "content": prompt_template}],
                temperature=0.0
                )
                raw_response = response.choices[0].message.content.strip()
            else:
                response = self.gmodel.generate_content(prompt_template, generation_config=genai.GenerationConfig(temperature=0.0))
                raw_response = response.text.strip()
            
            # Post-process to ensure only the query is returned
            search_query = raw_response.strip('"').strip()
            
            print(f"DEBUG: Raw search query response from LLM: {raw_response}")
            print(f"DEBUG: Cleaned search query: {search_query}")
            return search_query
        except Exception as e:
            print(f"❌ Error generating search query with {model_choice}: {e}")
            return query # Fallback to using the original query

    def _get_dynamic_num_results(self, query: str, history: List[str], model_choice: str) -> int:
        combined_prompt = ""
        for i in range(0, len(history), 2):
            combined_prompt += f"User: {history[i]}\n"
            if i + 1 < len(history):
                combined_prompt += f"Model: {history[i+1]}\n"
        combined_prompt += f"User: {query}"

        prompt_template = f"""
        You are an expert search query analyzer. Your task is to determine the ideal number of web search results needed to answer the user's request. Consider the user's query and the conversation history to gauge the complexity and specificity of the request.

        - If the request is a simple, factual question (e.g., "what is the capital of France?"), suggest 1-2 results.
        - If the request is about a recent event or a topic with a clear answer but some nuance (e.g., "what happened in the last election?"), suggest 3 results.
        - If the request is broad, requires a comprehensive overview, or involves multiple perspectives (e.g., "what are the main challenges facing the climate change?"), suggest 4-5 results.
        - Always return a single number from 1 to 5. Do not return any other text.

        Conversation history:
        {combined_prompt}
        Number of search results needed:
        """
        try:
            llm = self._get_llm_for_search(model_choice)
            if isinstance(llm, Groq):
                response = llm.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=[{"role": "user", "content": prompt_template}],
                    temperature=0.0
                )
                num = int(response.choices[0].message.content.strip())
            else:
                response = llm.generate_content(prompt_template, generation_config=genai.GenerationConfig(temperature=0.0))
                num = int(response.text.strip())
            return max(1, min(5, num))
        except (ValueError, TypeError):
            print("❗ LLM failed to return a valid integer for search results. Using default of 3.")
            return 3

    def _check_cache_relevance(self, new_query: str, cached_context: str, model_choice: str) -> bool:
        """
        Uses the LLM to determine if the cached web search results are relevant to the new query.
        """
        prompt_template = f"""
        You are an expert at determining the relevance of provided context to a new user query.
        Your task is to answer "yes" if the new query can likely be answered by the provided context, and "no" if it cannot.
        Do not provide any other text. Just "yes" or "no".

        --- Provided Context ---
        {cached_context}
        --- End Provided Context ---

        New User Query: {new_query}
        Is the context relevant? (yes/no):
        """
        try:
            llm = self._get_llm_for_search(model_choice)
            if isinstance(llm, Groq):
                response = llm.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=[{"role": "user", "content": prompt_template}],
                    temperature=0.0
                )
                decision = response.choices[0].message.content.strip().lower()
            else:
                response = llm.generate_content(prompt_template, generation_config=genai.GenerationConfig(temperature=0.0))
                decision = response.text.strip().lower()
            return decision == 'yes'
        except Exception as e:
            print(f"❗ Error checking cache relevance: {e}. Defaulting to 'no'.")
            return False

    def _check_if_web_search_needed(self, query: str, history: List[str], model_choice: str) -> bool:
        """
        Uses the LLM to determine if a web search is even necessary.
        Returns True if a web search is required, False otherwise.
        """
        combined_prompt = "\n".join(history + [query]) if history else query
        
        prompt_template = f"""
        You are a highly efficient assistant. Your task is to determine whether a web search is necessary to answer the user's query. Answer "yes" if a web search is required for a complete and accurate answer, and "no" if the question can be answered from general knowledge.

        **Important:** You have a knowledge cutoff. If the query asks for information that is time-sensitive or related to an event that may have happened after your last training update (e.g., a recent sports winner, the current year's events), you MUST answer "yes" to perform a web search, even if your internal knowledge suggests a different answer. This is crucial for providing up-to-date information.

        A web search is needed for:
        - Recent events (e.g., "who won the F1 race this year?").
        - Specific, factual details that are likely to change (e.g., "current population of Tokyo").
        - Broad topics that require a summary of many different perspectives or recent developments (e.g., "the latest on AI ethics").

        A web search is NOT needed for:
        - Simple greetings (e.g., "hello").
        - Basic, unchanging facts (e.g., "what is the capital of France?").
        - Questions that are nonsensical.

        Query: {combined_prompt}
        Is a web search needed? (yes/no):
        """
        try:
            llm = self._get_llm_for_search(model_choice)
            if isinstance(llm, Groq):
                response = llm.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=[{"role": "user", "content": prompt_template}],
                    temperature=0.0
                )
                decision = response.choices[0].message.content.strip().lower()
            else:
                response = llm.generate_content(prompt_template, generation_config=genai.GenerationConfig(temperature=0.0))
                decision = response.text.strip().lower()

            print(f"DEBUG: [web_search_check] LLM decision: {decision}")
            # Check if the decision contains the word "yes"
            return 'yes' in decision
        except Exception as e:
            print(f"❗ Error checking if web search is needed: {e}. Defaulting to 'yes'.")
            return True

    def count_tokens(self, text: str, model_choice: str) -> int:
        if model_choice == "groq":
            import tiktoken
            tokenizer = tiktoken.get_encoding("cl100k_base")
            return len(tokenizer.encode(text))
        elif model_choice == "gemini":
            return len(text) // 4  # Approximation
        else:
            return len(text.split())  # Fallback approximation

    def _detect_language(self, query: str, history: List[str], model_choice: str = "gemini") -> str:
        """
        Detect the language of the user's query using the LLM model.
        Returns either "English" or "Bengali" based on the query.
        """
        try:
            # Create a simple prompt for language detection
            language_detection_prompt = f"""Analyze the following text and determine if it's in English or Bengali language. 
            Respond with only "English" or "Bengali".

            Text to analyze: "{query}"

            Language:"""

            if model_choice == "groq" and self.groq_client:
                messages = [
                    {"role": "system", "content": "You are a language detection assistant. Respond only with the language name."},
                    {"role": "user", "content": language_detection_prompt}
                ]
                
                response = self.groq_client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=messages,
                    temperature=0.0,
                    max_tokens=10
                )
                detected_language = response.choices[0].message.content.strip()
            else:
                import google.generativeai as genai
                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(
                    language_detection_prompt,
                    generation_config=genai.GenerationConfig(temperature=0.0, max_output_tokens=10)
                )
                detected_language = response.text.strip()

            print(f"DEBUG: [language_detection] Detected language: {detected_language}")
            # Ensure only English or Bengali is returned
            if "bengali" in detected_language.lower() or "bangla" in detected_language.lower():
                return "Bengali"
            else:
                return "English"
        
        except Exception as e:
            print(f"DEBUG: [language_detection] Error detecting language: {e}")
            return "English"  # Default to English if detection fails
    
    def _hybrid_retrieve(self, query: str, session_data: SessionData, k_bm25: int = 20, k_vector: int = 20, top_k: int = 5):
        """
        Performs a hybrid search using BM25 and vector search, then re-ranks with RRF.
        """
        print(f"DEBUG: [hybrid_retrieve] Performing hybrid search for query: {query}")
        
        # 1. BM25 search
        tokenized_query = nltk.word_tokenize(query)
        bm25_scores = session_data.bm25_index.get_scores(tokenized_query)
        bm25_indices = np.argsort(bm25_scores)[::-1][:k_bm25]
        
        # 2. Vector search
        query_embedding = self.model.encode([query]).astype(np.float32)
        qn = np.linalg.norm(query_embedding, axis=1, keepdims=True) + 1e-12
        query_embedding = query_embedding / qn
        D, vector_indices = session_data.index.search(query_embedding, k=k_vector)

        # 3. Combine and re-rank with RRF
        rank_fusion_scores = collections.defaultdict(float)
        
        # Calculate RRF scores for BM25 results
        for rank, doc_idx in enumerate(bm25_indices):
            rank_fusion_scores[doc_idx] += 1.0 / (rank + 60) # Constant K=60 for typical RRF
            
        # Calculate RRF scores for vector search results
        for rank, doc_idx in enumerate(vector_indices[0]):
            rank_fusion_scores[doc_idx] += 1.0 / (rank + 60)
            
        # Sort documents by combined RRF score
        reranked_indices = sorted(rank_fusion_scores.keys(), key=lambda idx: rank_fusion_scores[idx], reverse=True)
        
        # Get the final top-k chunks
        final_retrieved_chunks = [session_data.chunks[i] for i in reranked_indices[:top_k]]
        
        return final_retrieved_chunks

    def run(self, query: str, history: List[str], session_id: str, use_web: bool = False, model_choice: str = "gemini", output_format: Optional[str] = None) -> Generator[str, None, None]:
        if session_id not in self.sessions:
            self.sessions[session_id] = self._load_session_data(session_id)
        session_data = self.sessions[session_id]

        detected_language = self._detect_language(query, history, model_choice)
        print(f"DEBUG: [pipeline.run] Query language detected as: {detected_language}")

        perform_rag = session_data.index is not None and session_data.index.ntotal > 0 and session_data.chunks and not use_web

        context = ""
        sources = []

        if use_web:
            print(f"DEBUG: [pipeline.run] Web search mode enabled for session {session_id}.")

            if not self._check_if_web_search_needed(query, history, model_choice):
                print("DEBUG: [pipeline.run] LLM determined web search is not needed. Bypassing search.")
            else:
                use_cached_results = False
                if(session_data.web_search_cache and datetime.now() - session_data.web_search_cache.timestamp < WEB_SEARCH_TTL 
                   and self._check_cache_relevance(query, session_data.web_search_cache.context, model_choice)):
                    context = session_data.web_search_cache.context
                    sources = session_data.web_search_cache.sources
                    use_cached_results = True
                    print("✅ [pipeline.run] Using cached web search results.")
            
            if not use_cached_results:
                search_query = self._get_search_query(query, history, model_choice)
                print(f"DEBUG: [pipeline.run] Generated search query: '{search_query}'")
                num_results = self._get_dynamic_num_results(query, history, model_choice)
                print(f"DEBUG: [pipeline.run] Dynamically determined number of search results: {num_results}")
                context, sources = web_search(search_query, num_results=num_results)
                session_data.web_search_cache = WebSearchCache(query=search_query,context=context, sources=sources)

            with open("debug_prompt_output.txt", "w", encoding="utf-8") as f:
                f.write("🔍 Retrieved Context (from Web Search):\n")
                f.write(context)

        elif perform_rag:
            print(f"DEBUG: [pipeline.run] RAG mode enabled for session {session_id}.")
            session_data.web_search_cache = None

            # --- RAG Retrieval with Hybrid Search (BM25 + Vector + RRF) ---
            final_retrieved_chunks = self._hybrid_retrieve(query, session_data, top_k=5)

            retrieved_formatted = []
            for chunk in final_retrieved_chunks:
                text = chunk.get("text", "")
                retrieved_formatted.append(f" {text}")
                sources.append({
                    "name": chunk.get("document_name", "N/A"), 
                    "page": chunk.get("page", "N/A"),
                    "type": "document"
                })
            context = "\n\n".join(retrieved_formatted)

            with open("debug_prompt_output.txt", "w", encoding="utf-8") as f:
                f.write("🔍 Retrieved Chunks (from RAG):\n")
                for i, citation in enumerate(retrieved_formatted):
                    f.write(f"\n--- Citation {i+1} ---\n{citation}\n")
        else:
            print(f"DEBUG: [pipeline.run] No documents uploaded for session {session_id}. Running in general chatbot mode.")
            with open("debug_prompt_output.txt", "w", encoding="utf-8") as f:
                f.write("🔍 No documents for RAG. Running general chat.\n")
            session_data.web_search_cache = None

        # --- Construct final user prompt ---
        language_instruction = f"IMPORTANT: The user's query is in {detected_language}. Please respond in {detected_language} language only."
        
        if perform_rag or (use_web and context):
            final_user_prompt = f"""{language_instruction}
            You are a helpful and informative AI assistant.
            Your task is to provide a detailed and accurate answer to the user's question, using only the provided context. If the answer is not present, state {"in Bengali: 'প্রদত্ত তথ্যে এই প্রশ্নের উত্তর নেই।'" if detected_language == "Bengali" else ": 'The provided information does not contain the answer to this question.'"}
            --- Retrieved Context ---
            {context}
            --- End Retrieved Context ---
            User's Question: {query}
            Remember: Respond ONLY in {detected_language} language."""
        else:
            final_user_prompt = f"""{language_instruction}
            You are a helpful and informative AI assistant.
            Your task is to provide a detailed and accurate answer to the user's question.
            No additional context is provided, so use your general knowledge.
            **Confidence:** If you are highly confident in your inferred answer, provide it directly. If the information is insufficient to make a confident inference, state the specific message provided.
            User's Question: {query}
            Remember: Respond ONLY in {detected_language} language."""

        # <<< CHANGE START: Clean the history before passing it to the model >>>
        cleaned_history = []
        source_separator = "\n\n---\n**" # Generic separator that catches both languages
        for i in range(0, len(history), 2):
            # Add user message as is
            cleaned_history.append(history[i])
            if i + 1 < len(history):
                model_response = history[i+1]
                # Split the response at the source separator and take only the text before it
                clean_response = model_response.split(source_separator)[0].strip()
                cleaned_history.append(clean_response)
        # <<< CHANGE END >>>

        # --- Format history for the model ---
        if model_choice == "groq" and self.groq_client:
            # <<< CHANGE: Use cleaned_history instead of history >>>
            formatted_history = [{"role": "system", "content": f"You are a helpful assistant. Always respond in {detected_language} language."}]
            for i in range(0, len(cleaned_history), 2):
                formatted_history.append({"role": "user", "content": cleaned_history[i]})
                if i + 1 < len(cleaned_history):
                    formatted_history.append({"role": "assistant", "content": cleaned_history[i+1]})
            formatted_history.append({"role": "user", "content": final_user_prompt})

        else: # Gemini
            # <<< CHANGE: Use cleaned_history instead of history >>>
            system_prompt_for_gemini = f"System prompt: You are a helpful and informative AI assistant. {language_instruction}"
            formatted_history = []
            
            # Build history from the cleaned list
            if cleaned_history:
                # Add system prompt to the first user message
                formatted_history.append({"role": "user", "parts": [f"{system_prompt_for_gemini}\n\n{cleaned_history[0]}"]})
                # Add the rest of the cleaned history
                for i in range(1, len(cleaned_history), 2):
                    formatted_history.append({"role": "model", "parts": [cleaned_history[i]]})
                    if i + 1 < len(cleaned_history):
                        formatted_history.append({"role": "user", "parts": [cleaned_history[i+1]]})
                # Add the current query, which is not in the history yet
                formatted_history.append({"role": "user", "parts": [final_user_prompt]})
            else:
                # This is the first turn, no history to clean
                formatted_history.append({"role": "user", "parts": [f"{system_prompt_for_gemini}\n\n{final_user_prompt}"]})

        with open("debug_prompt_output.txt", "a", encoding="utf-8") as f:
            f.write(f"\n\n🌐 Detected Language: {detected_language}")
            f.write("\n\n📜 Final Prompt to LLM (Gemini Format):\n")
            f.write(json.dumps(formatted_history, indent=2, ensure_ascii=False))
            
        # --- Start generation ---
        start_time = time.time()
        total_tokens = 0
        response_text = ""

        try:
            if model_choice == "groq" and self.groq_client:
                print(f"DEBUG: [pipeline.run] Using Groq model:openai/gpt-oss-20b")
                # Corrected streaming loop: use stream=True directly in the create method.
                stream = self.groq_client.chat.completions.create(
                    messages=formatted_history,
                    model="openai/gpt-oss-20b",
                    temperature=0.0,
                    stream=True
                )
                
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content_chunk = chunk.choices[0].delta.content
                        total_tokens += self.count_tokens(content_chunk, model_choice)
                        response_text += content_chunk
                        yield content_chunk

            else: # Gemini
                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                response = genai.GenerativeModel("gemini-1.5-flash").generate_content(
                    contents=formatted_history,
                    generation_config=genai.GenerationConfig(temperature=0.0),
                    stream=True
                )
                for chunk in response:
                    content = chunk.text
                    yield content
                    total_tokens += self.count_tokens(content, model_choice)
                    response_text += content
            
            end_time = time.time()
            duration = end_time - start_time
            tokens_per_second = total_tokens / duration if duration > 0 else 0

            print(f"🧠 {model_choice.capitalize()} response completed for session {session_id}.")
            print(f"Total tokens generated: {total_tokens}")
            print(f"Time taken: {duration:.2f}s")
            print(f"Tokens per second: {tokens_per_second:.2f}")
            
            

            # --- CRUCIAL: This part correctly saves only the clean response text to history for the NEXT turn
            if response_text:
                history.append(query)
                history.append(response_text.strip()) # This is correct, but the yielded sources are added by the calling UI

            # --- This part yields the sources for display in the current turn ONLY
            if sources:
                source_headers = { "English": "Sources:", "Bengali": "উৎস:" }
                document_label = { "English": "Document:", "Bengali": "নথি:" }.get(detected_language, "Document:")
                page_label = { "English": "Page:", "Bengali": "পৃষ্ঠা:" }.get(detected_language, "Page:")
                
                yield f"\n\n---\n**{source_headers.get(detected_language, 'Sources:')}**"
                unique_sources = collections.OrderedDict()
                for s in sources:
                    source_key = f"{s.get('name')}-{s.get('url', s.get('page'))}"
                    if source_key not in unique_sources:
                        unique_sources[source_key] = s
                for i, source in enumerate(unique_sources.values()):
                    if source["type"] == "web":
                        yield f"\n\n{i+1}. [{source['name']}]({source['url']})"
                    else:
                        yield f"\n\n{i+1}. **{document_label}** {source['name']} ({page_label} {source['page']})"

        except Exception as e:
            # ... (Error handling is fine)
            traceback.print_exc()
            error_messages = {
                "English": f"❌ An error occurred during response generation: {e}",
                "Bengali": f"❌ প্রতিক্রিয়া তৈরির সময় একটি ত্রুটি ঘটেছে: {e}"
            }
            yield error_messages.get(detected_language, f"❌ An error occurred during response generation: {e}")