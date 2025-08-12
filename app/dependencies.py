# app/dependencies.py
from app.pipeline import RAGPipeline
from fastapi import Request # Only Request is needed now
# import traceback # Traceback is not directly used in this function, so it can be removed

print("DEBUG: app.dependencies module loaded.")

def get_rag_pipeline(request: Request) -> RAGPipeline:
    """Returns the globally managed RAGPipeline instance from app.state."""
    rag_instance_from_state = request.app.state.rag_pipeline
    print(f"DEBUG: [get_rag_pipeline] Called. Current rag_instance_from_state: {rag_instance_from_state}")
    
    # If rag_pipeline is None, it means the startup event failed to initialize it.
    # This indicates a critical error during backend startup that needs to be resolved.
    # We still keep this check as a safeguard for fundamental initialization failures.
    if rag_instance_from_state is None:
        print("❌ [get_rag_pipeline] RAGPipeline is NOT initialized in app.state. This indicates a critical startup error.")
        raise ValueError("RAGPipeline failed to initialize during application startup. Check backend logs for details.")
        # Using ValueError here to signify a core backend issue, which FastAPI will catch and
        # convert to a 500 Internal Server Error, which is more appropriate than 400.

    # If the RAGPipeline instance exists, return it.
    # The decision if it has documents for RAG or should act as a general chatbot
    # is handled within the RAGPipeline.run method.
    return rag_instance_from_state