from typing import List, Optional
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    query: str
    session_id: str = Field(..., description="A unique identifier for the conversation session.")
    history: Optional[List[str]] = Field(default_factory=list, description="A flat list of alternating user and model messages.") 
    use_web: Optional[bool] = False
    model_choice: str = "gemini"

class QueryResponse(BaseModel):
    question: str
    answer: str
    retrieved: List[str]
    llm_time: float

class DeleteDocumentRequest(BaseModel):
    document_id: str
    file_name: Optional[str] = None
    session_id: str


    
