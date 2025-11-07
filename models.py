# models.py
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Client-generated session identifier")
    message: str = Field(..., description="User message")

class ChatResponse(BaseModel):
    text: str

class ClearRequest(BaseModel):
    session_id: str

class HistorySummary(BaseModel):
    messages: int
