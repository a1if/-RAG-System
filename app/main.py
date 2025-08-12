import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, Depends
from starlette.middleware.base import BaseHTTPMiddleware
from app.api import rag_router
from app.env_check import validate_env
from app.pipeline import RAGPipeline # Needed for type hinting for rag_instance
from app.dependencies import get_rag_pipeline # Only import get_rag_pipeline
import traceback
import time

BACKEND_START_TIMESTAMP = time.time()
print(f"🚀 Backend script started at: {BACKEND_START_TIMESTAMP}")


validate_env()
load_dotenv()

# Create the FastAPI application instance (ONLY ONCE)
app = FastAPI(title="Bangla-English RAG API")

@app.on_event("startup")
async def startup_event():
    """Initialize the RAGPipeline once on application startup and store it in app.state."""
    print("🚀 Initializing RAGPipeline on startup event...")
    start_time_startup = time.time()
    try:
        print("⏳ Attempting to instantiate RAGPipeline...")
        # Store RAGPipeline instance directly on app.state
        app.state.rag_pipeline = RAGPipeline()
        print(f"✅ RAGPipeline initialized successfully in {time.time() - start_time_startup:.2f} seconds.")
        print(f"DEBUG: app.state.rag_pipeline type after init: {type(app.state.rag_pipeline)}")
    except Exception as e:
        print(f"❌ ERROR: RAGPipeline initialization FAILED on startup: {e}")
        traceback.print_exc()
        app.state.rag_pipeline = None # Ensure it's explicitly None on failure
    print("✅ Startup event finished.")

@app.on_event("shutdown")
async def shutdown_event():
    """Perform cleanup on application shutdown (optional)."""
    print("👋 Shutting down RAGPipeline...")
    # Add any cleanup logic for rag_instance if necessary before the app closes.

# A simple status endpoint for the main app
@app.get("/status")
def status(request: Request): # Accept request to access app.state
    # Check if rag_pipeline is initialized via app.state
    return {"status": "ok", "rag_pipeline_initialized": request.app.state.rag_pipeline is not None,"backend_startup_time": BACKEND_START_TIMESTAMP}

# Add the environment validation middleware
class EnvValidationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        required_keys = ["GEMINI_API_KEY"]
        missing = [key for key in required_keys if not os.getenv(key)]
        if missing:
            print(f"⚠️ Warning: Missing required .env keys: {', '.join(missing)}")
        return await call_next(request)

app.add_middleware(EnvValidationMiddleware)

# Include the RAG router
# This ensures that endpoints in rag_router get the single RAGPipeline instance via dependency injection
app.include_router(rag_router, dependencies=[Depends(get_rag_pipeline)])

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("BACKEND_HOST", "0.0.0.0")
    port = int(os.getenv("BACKEND_PORT", 8000))
    # Note: If running with multiple workers in production, the --reload flag should be removed.
    uvicorn.run("app.main:app", host=host, port=port, reload=True, log_level="debug", workers=1)