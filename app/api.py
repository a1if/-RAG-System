import time
import traceback
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, Form, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from app.pipeline import RAGPipeline
from app.models import QueryRequest, DeleteDocumentRequest
from pathlib import Path

"""Use app.state.rag_pipeline instead of creating a new instance per request."""
def get_rag_pipeline(request: Request) -> RAGPipeline:
    rag = request.app.state.rag_pipeline
    if rag is None:
        raise HTTPException(status_code=500, detail="RAGPipeline not initialized on server startup.")
    return rag

rag_router = APIRouter()
chat_histories: Dict[str, List[List[str]]] = {}

@rag_router.get("/status")
async def status():
    print("DEBUG: [api] Accessed /status endpoint.")
    return {"status": "ok"}

@rag_router.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
    rag: RAGPipeline = Depends(get_rag_pipeline)
):
    """Uploads and processes one or more files, adding them to the vector store for a specific session."""
    if not session_id:
        raise HTTPException(status_code=400, detail="❌ A session_id is required to upload a document.")

    # Allow both MIME-based and extension-based validation
    allowed_mime_types = [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
        "text/plain",  # .txt
        "text/html",  # .html
        "text/csv",  # .csv
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # .xlsx
        "image/png", "image/jpeg", "image/jpg", "image/tiff", "image/bmp"
    ]
    allowed_exts = {".pdf", ".docx", ".txt", ".html", ".htm", ".csv", ".xlsx", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}

    print(f"DEBUG: [api] Entered multi-file upload endpoint for session: {session_id} with {len(files)} files.")
    
    results = []
    rag._load_session_data(session_id)

    for file in files:
        try:
            # Step 1: Validate MIME type and/or file extension
            ext = Path(file.filename).suffix.lower()
            if (file.content_type not in allowed_mime_types) and (ext not in allowed_exts):
                results.append({
                    "filename": file.filename,
                    "status": "❌ Failed to process",
                    "detail": f"Unsupported file type: {file.content_type or ext}. Allowed extensions: {', '.join(sorted(allowed_exts))}"
                })
                continue
            
            print(f"DEBUG: [api] Processing file: '{file.filename}' with MIME type '{file.content_type}'")
            contents = await file.read()
            start_time = time.time()
            
            # Step 2: Call the refactored RAG pipeline method
            document_id = rag.rebuild_vector_store(contents, session_id, file_name=file.filename)
            duration = time.time() - start_time
            
            if document_id is None:
                results.append({
                    "filename": file.filename,
                    "status": "❌ Failed to process",
                    "detail": "No valid text extracted or document already exists."
                })
            else:
                results.append({
                    "filename": file.filename,
                    "status": "✅ Success",
                    "document_id": document_id,
                    "time_taken": round(duration, 2)
                })
            print(f"📄 Processed file '{file.filename}' for session {session_id} in {duration:.2f}s with ID {document_id}")
            
        except Exception as e:
            traceback.print_exc()
            results.append({
                "filename": file.filename,
                "status": "❌ Error",
                "detail": str(e)
            })

    return JSONResponse(content={
        "message": "Multi-file upload processing complete.",
        "results": results
    })
    
@rag_router.get("/documents", response_model=List[str])
async def get_loaded_documents(session_id: str, rag: RAGPipeline = Depends(get_rag_pipeline)):
    """Gets a list of documents loaded for a given session."""
    documents = rag.get_loaded_documents(session_id=session_id)
    return documents

@rag_router.post("/delete_document")
async def delete_document(request: DeleteDocumentRequest, rag: RAGPipeline = Depends(get_rag_pipeline)):
    print(f"DEBUG: [api] Entered delete_document endpoint for ID: {request.document_id}, File: {request.file_name}, Session: {request.session_id}")
    start_time = time.time()
    try:
        deleted_successfully = rag.delete_document_data(request.document_id, request.session_id, request.file_name)
        duration = time.time() - start_time
        if deleted_successfully:
            print(f"🗑️ Document '{request.file_name}' (ID: {request.document_id}) deleted in {duration:.2f}s")
            return JSONResponse(content={"message": f"🗑️ Document '{request.file_name}' deleted successfully.", "time_taken": round(duration, 2)})
        else:
            raise HTTPException(status_code=404, detail=f"❌ Document '{request.file_name}' (ID: {request.document_id}) not found or could not be deleted.")
    except Exception as e:
        print("❌ [api] Error deleting document:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"❌ Failed to delete document: {str(e)}")

@rag_router.post("/rag")
async def ask_question(payload: QueryRequest, rag: RAGPipeline = Depends(get_rag_pipeline)):
    """Handles a chat query and streams the response back to the client."""
    print(f"DEBUG: [api] Received query for session_id: {payload.session_id}, model: {payload.model_choice}")
    history_from_client = payload.history
    try:
        return StreamingResponse(
            rag.run(
                query=payload.query, 
                history=history_from_client, 
                session_id=payload.session_id, 
                use_web=payload.use_web,
                model_choice=payload.model_choice
            ),
            media_type="text/plain"
        )
    except HTTPException:
        raise
    except Exception as e:
        print("❌ [api] Error generating response:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"❌ Failed to generate response: {str(e)}")

@rag_router.post("/clear_history")
async def clear_history(session_id: str):
    if session_id in chat_histories:
        del chat_histories[session_id]
        return {"message": f"Chat history for session {session_id} cleared."}
    else:
        raise HTTPException(status_code=404, detail=f"No chat history found for session {session_id}.")