import streamlit as st
import requests
import time
from pathlib import Path
import json
import os
import uuid
import collections


# Define FRONTEND_READY_FILE relative to project root
FRONTEND_READY_FILE = Path(__file__).parent.parent / ".frontend_ready"
FRONTEND_START_TIMESTAMP = time.time()

# --- Configuration ---
API_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# ------------------------- Page Setup -------------------------
st.set_page_config(
    page_title=" RAG Chatbot",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    /* ... (Your existing CSS styling here) ... */
    .st-chat-message-container.st-chat-message-user .st-emotion-cache-1wmy9hp {
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .edit-button {
        background-color: #282828;
        color: #e0e0e0;
        border: 1px solid #4285F4;
        border-radius: 4px;
        padding: 2px 8px;
        cursor: pointer;
        font-size: 12px;
        transition: background-color 0.2s ease;
        margin-left: 10px;
    }
    .edit-button:hover {
        background-color: #4285F4;
    }
    .st-chat-message-container.st-chat-message-user {
        flex-direction: row; 
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("✨Chatbot")

# ------------------------- Session State Management for Chat -------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# FIX: Consolidate session_id initialization here to ensure it only happens once.
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.info(f"New chat session started with ID: {st.session_state.session_id}")
    
# --- Function to clear the chat ---
def reset_chat_session():
    """Resets the chat history and starts a new session."""
    st.session_state.history = []
    st.session_state.session_id = str(uuid.uuid4())
    st.success("Chat has been cleared. Start a new conversation.")

# ------------------------- API URL Setup -------------------------
for attempt in range(6): 
    port_file = Path(".port")
    if port_file.exists():
        backend_port = port_file.read_text().strip()
        if backend_port and backend_port.isdigit():
            API_URL = f"http://localhost:{backend_port}"
            st.sidebar.success(f"Backend API URL set to: {API_URL}")
            break
        else:
            st.sidebar.warning(f"Found .port file but content is invalid: '{backend_port}'. Using default {API_URL}.")
    time.sleep(2 ** attempt * 0.25)
else:
    st.sidebar.error(f"❌ Could not detect backend port from .port file after multiple attempts. Using default {API_URL}. Please ensure your backend is running and creates a '.port' file.")

# Check backend status early and cache it
@st.cache_data(ttl=10)
def get_backend_status(url):
    try:
        response = requests.get(f"{url}/status", timeout=5)
        response.raise_for_status()
        data = response.json()
        status_ok = data.get("status") == "ok"
        backend_startup_time = data.get("backend_startup_time")
        return status_ok, backend_startup_time
    except requests.exceptions.ConnectionError:
        st.error("❌ Backend not reachable. Please ensure the backend server is running and accessible.")
        return False, None
    except requests.exceptions.Timeout:
        st.error("❌ Backend connection timed out.")
        return False, None
    except requests.exceptions.RequestException as e:
        st.error(f"❌ An error occurred while checking backend status: {e}")
        return False, None
    
# ------------------------- Edit Functionality -------------------------
def edit_message(index):
    """
    Handles the editing of a user message.
    Clears all subsequent messages and updates the chat input box.
    """
    if index < len(st.session_state.history) and index % 2 == 0:
        edited_text = st.session_state.history[index]
        st.session_state.history = st.session_state.history[:index + 1]
        st.session_state.edited_query = edited_text
        st.rerun()

# --- CALL STATUS CHECK AND CALCULATE LOAD TIME ---
st.session_state.backend_status_ok, backend_start_time_from_api = get_backend_status(API_URL)

if st.session_state.backend_status_ok and backend_start_time_from_api is not None:
    total_app_load_time = time.time() - backend_start_time_from_api
    st.sidebar.markdown(f"**⚡ Full App Load Time:** <span style='color: #4CAF50;'>{total_app_load_time:.2f} seconds</span>", unsafe_allow_html=True)
    FRONTEND_READY_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(FRONTEND_READY_FILE, "w") as f:
            f.write(str(time.time()))
    except Exception as e:
        st.warning(f"Could not write frontend readiness file '{FRONTEND_READY_FILE}': {e}")
elif not st.session_state.backend_status_ok:
    st.info("Attempting to connect to backend...")


# ------------------------- Session State Init -------------------------
# REMOVED: The second, redundant `session_id` initialization line
st.session_state.setdefault("history", [])
st.session_state.setdefault("pdf_uploaded", False)
st.session_state.setdefault("last_processed_query", "")
st.session_state.setdefault("processed_pdfs", [])
st.session_state.setdefault("file_uploader_key", 0)
st.session_state.setdefault("edited_query", "")
st.session_state.setdefault("model_choice", "gemini")

# --- Helper to save/load processed_pdfs to/from a small file for persistence ---
PROCESSED_PDFS_FILE = Path(".processed_pdfs.json")

def save_processed_pdfs():
    """Saves processed PDF metadata to a file."""
    with open(PROCESSED_PDFS_FILE, "w") as f:
        json.dump(st.session_state.processed_pdfs, f)

def load_processed_pdfs():
    """Loads processed PDF metadata from a file."""
    if PROCESSED_PDFS_FILE.exists():
        try:
            with open(PROCESSED_PDFS_FILE, "r") as f:
                st.session_state.processed_pdfs = json.load(f)
            st.session_state.pdf_uploaded = bool(st.session_state.processed_pdfs)
        except json.JSONDecodeError:
            st.session_state.processed_pdfs = []
            st.session_state.pdf_uploaded = False
            st.error("Error loading .processed_pdfs.json. File might be corrupted.")
    else:
        st.session_state.processed_pdfs = []
        st.session_state.pdf_uploaded = False

load_processed_pdfs()

# Auto-resume previous session if documents exist from a different session
try:
    existing_session_ids = sorted({p.get("session_id") for p in st.session_state.processed_pdfs if p.get("session_id")})
    if existing_session_ids and st.session_state.session_id not in existing_session_ids:
        # Prefer the most recent processed doc's session
        last_session_id = st.session_state.processed_pdfs[-1].get("session_id")
        if last_session_id:
            st.session_state.session_id = last_session_id
            st.info(f"Resuming previous document session: {st.session_state.session_id}")
except Exception:
    pass

# --- Function to delete a document ---
def delete_document(doc_id: str, doc_name: str, session_id: str):
    """
    Sends a request to the backend to delete a specific document and updates the UI.
    Now accepts session_id as an argument.
    """
    try:
        response = requests.post(
            f"{API_URL}/delete_document",
            json={
                "document_id": doc_id,
                "file_name": doc_name,
                "session_id": session_id # Use the correct, stored session_id
            }
        )
        response.raise_for_status()
        st.success(f"🗑️ Document '{doc_name}' deleted successfully.")
        
        st.session_state.processed_pdfs = [
            p for p in st.session_state.processed_pdfs if p["id"] != doc_id
        ]
        save_processed_pdfs()
        
        st.session_state.file_uploader_key += 1
        
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Failed to delete document '{doc_name}': {e}")

# ------------------------- Sidebar: PDF Upload & Document Manager -------------------------
with st.sidebar:
    st.header("📤 Document Manager")

    if st.session_state.backend_status_ok:
        st.markdown("<p style='color: #4CAF50;'>✅ Backend Connected</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='color: #FF6347;'>❌ Backend Disconnected</p>", unsafe_allow_html=True)
        st.warning("Please ensure your backend server is running.")

    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        # <<< MODIFICATION >>>
        "Choose one or more files:",
        type=["pdf", "docx", "txt", "html", "csv", "xlsx", "png", "jpg", "jpeg", "tiff", "bmp"],
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state.file_uploader_key}"
    )

    if uploaded_files and st.session_state.backend_status_ok:
        for uploaded_file in uploaded_files:
            is_already_processed = any(p['name'] == uploaded_file.name for p in st.session_state.processed_pdfs)

            if not is_already_processed:
                progress_bar = st.progress(0, text=f"Uploading {uploaded_file.name}...")

                try:
                    for pct in range(0, 41, 10):
                        time.sleep(0.05)
                        progress_bar.progress(pct, text=f"Uploading {uploaded_file.name}... {pct}%")

                    # <<< MODIFICATION >>>
                    files_to_send = [("files", (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type))]
                    data = {"session_id": st.session_state.session_id}

                    progress_bar.progress(50, text=f"Processing {uploaded_file.name}...")
                    
                    response = requests.post(f"{API_URL}/upload", files=files_to_send, data=data, timeout=300)
                    response.raise_for_status()

                    result = response.json()
                    file_result = next((r for r in result.get("results", []) if r["filename"] == uploaded_file.name), None)

                    if file_result and file_result.get("status") == "✅ Success":
                        document_id = file_result.get("document_id")
                        st.session_state.processed_pdfs.append({
                            "id": document_id,
                            "name": uploaded_file.name,
                            "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "session_id": st.session_state.session_id
                        })
                        save_processed_pdfs()
                        st.session_state.pdf_uploaded = True
                        progress_bar.progress(100, text=f"✅ {uploaded_file.name} processed successfully")
                        st.success(f"✅ Document '{uploaded_file.name}' processed! You can now ask questions about it.")
                    else:
                        progress_bar.empty()
                        st.error(f"❌ Failed to process '{uploaded_file.name}': {file_result.get('detail', 'Unknown error')}")
                except requests.exceptions.Timeout:
                    progress_bar.empty()
                    st.error(f"❌ Upload of '{uploaded_file.name}' timed out.")
                except requests.exceptions.RequestException as req_err:
                    progress_bar.empty()
                    st.error(f"❌ Network error during upload of '{uploaded_file.name}': {req_err}")
                except Exception as e:
                    progress_bar.empty()
                    st.error(f"❌ Unexpected error for '{uploaded_file.name}': {e}")
            else:
                st.info(f"'{uploaded_file.name}' is already uploaded.")
                
    # --- Display Loaded Documents Section ---
    st.subheader("Loaded Documents")
    if st.session_state.processed_pdfs:
        # Only show documents for the active session
        docs_for_session = [d for d in st.session_state.processed_pdfs if d.get("session_id") == st.session_state.session_id]
        for doc in docs_for_session:
            doc_name = doc["name"]
            doc_id = doc["id"]

            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"📄 **{doc_name}**")
                st.caption(f"ID: {doc_id[:8]}... | Uploaded: {doc['uploaded_at']}")
            with col2:
                # Use the document's original session_id when deleting
                st.button("🗑️", key=f"delete_btn_{doc_id}", on_click=delete_document, args=(doc_id, doc_name, doc.get("session_id", st.session_state.session_id)))
    else:
        st.info("No documents uploaded yet.")

    # --- Retrieval Strategy & Model Selection ---
    st.subheader("🔎 Retrieval Strategy")
    st.session_state.use_web_search = st.checkbox("Use Web Search if no documents", value=False)
    
    st.markdown("---")
    
    st.subheader("🤖 Model Selection")
    st.session_state.model_choice = st.selectbox(
        "Choose a Language Model",
        options=["gemini", "groq"]
    )
    
    st.markdown("---")
    st.button("Clear Chat", on_click=reset_chat_session, use_container_width=True)


# ------------------------- Main Chat Interface -------------------------
for i, chat_item in enumerate(st.session_state.history):
    is_user_message = i % 2 == 0
    with st.chat_message("user" if is_user_message else "assistant"):
        if is_user_message:
            col1, col2 = st.columns([1, 10])
            with col2:
                st.markdown(chat_item)
            with col1:
                if st.button("✏️", key=f"edit_{i}"):
                    if i < len(st.session_state.history) and i % 2 == 0:
                        edited_text = st.session_state.history[i]
                        st.session_state.history = st.session_state.history[:i + 1]
                        st.session_state.edited_query = edited_text
                        st.rerun()
        else:
            st.markdown(chat_item)

if not st.session_state.backend_status_ok:
    st.info("Please start your backend server to enable chat functionality.")

if "edited_query" not in st.session_state:
    st.session_state.edited_query = ""

CHAT_KEY = "query_input"

if st.session_state.get("edited_query"):
    st.session_state[CHAT_KEY] = st.session_state.pop("edited_query", "")

prompt = st.chat_input(
    placeholder="Ask a question about the document or have a general chat…",
    key=CHAT_KEY,
)

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.history.append(prompt)

    with st.chat_message("assistant"):
        search_message_placeholder = st.empty()
        message_placeholder = st.empty()
        full_response = ""

        if not st.session_state.backend_status_ok:
            full_response = "❌ Backend is not running or not accessible. Please ensure the backend services are started."
            message_placeholder.markdown(full_response)
        else:
            try:
                payload = {
                    "query": prompt,
                    "session_id": st.session_state.session_id, # This is the crucial part
                    "history": st.session_state.history[:-1],
                    "use_web": st.session_state.use_web_search,
                    # Pass the selected model to the backend
                    "model_choice": st.session_state.model_choice
                }
                
                if st.session_state.use_web_search:
                    search_message_placeholder.markdown(f"Searching the web using {st.session_state.model_choice}...")

                with requests.post(f"{API_URL}/rag", json=payload, stream=True, timeout=120) as response:
                    response.raise_for_status()

                    search_message_placeholder.empty()

                    for chunk in response.iter_content(chunk_size=1, decode_unicode=True):
                        if chunk:
                            full_response += chunk
                            message_placeholder.markdown(full_response + "▌")
                    
                    message_placeholder.markdown(full_response)
                
                st.session_state.history.append(full_response)

            except requests.exceptions.RequestException as e:
                search_message_placeholder.empty()
                full_response = f"❌ An error occurred: {e}"
                message_placeholder.markdown(full_response)
                st.session_state.history.append(full_response)
            except Exception as e:
                search_message_placeholder.empty()
                full_response = f"❌ An unexpected error occurred: {e}"
                message_placeholder.markdown(full_response)
                st.session_state.history.append(full_response)