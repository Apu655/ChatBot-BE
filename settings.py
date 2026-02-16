# settings.py
import os
from pydantic import BaseModel

class Settings(BaseModel):
    API_KEY: str = os.getenv("GOOGLE_GENAI_API_KEY", "")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gemini-2.5-flash")
    MAX_OUTPUT_TOKENS: int = int(os.getenv("MAX_OUTPUT_TOKENS", "1200"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.3"))
    TEST:str  = os.getenv("TEST","test")
    # summarization knobs
    SUMMARIZE_OVER_MESSAGES: int = int(os.getenv("SUMMARIZE_OVER_MESSAGES", "20"))
    KEEP_RECENT_MESSAGES: int = int(os.getenv("KEEP_RECENT_MESSAGES", "10"))

    class Config:
        arbitrary_types_allowed = True
