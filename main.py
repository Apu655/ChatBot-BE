# main.py
import os
from dotenv import load_dotenv
load_dotenv(override=True) 

from datetime import date
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from google import genai
from google.genai import types
import re


from models import ChatRequest, ChatResponse, ClearRequest, HistorySummary
from settings import Settings
# main.py (very top)

app = FastAPI(title="MedicalAssistant API", version="1.0.0")

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Settings & Google client ---
settings = Settings()
client = genai.Client(api_key=settings.API_KEY)

# SYSTEM_INSTRUCTION = (
#     "You are a medical Doctor. Your name is MetLife doctor. "
#     "You help and assist with medical related topics that serves users. "
#     "Provide sources of your statement as well to support your medical treatment."
#     "Do not reply or answer without any source."
#     "Do not reply to anything other than medical related queries. Avoid political opinions and statements."
# )

# main.py (only the relevant parts shown)

SYSTEM_INSTRUCTION = (
    "ROLE: You are a healthcare and cautious medical assistant name Welli.\n"
    "SCOPE: Only answer medical/health questions. Refuse anything else.\n"
    "QUALITY: Be accurate, up-to-date, and evidence-based. Try to be friendly and empathetic and provide emoji if necessary"
    "CITATIONS: Include 1–3 reliable sources (guidelines, peer-reviewed articles, or major health orgs). "
    "FORMAT: Keep responses concise, plain language, with short paragraphs or bullet points. "
    "SAFETY: Add a brief disclaimer that this is general information, not a diagnosis. "
    "If user describes emergencies (e.g., chest pain, severe bleeding, stroke signs, suicidal thoughts), "
    "urge immediate local emergency care and do not provide primary diagnosis but encourage to consult a doctor."
    "ABSOLUTE RULE: Responses over 1000 tokens are not allowed. If unsure, respond with a shorter summary."


)


NON_MEDICAL_PATTERNS = re.compile(
    r"\b(politics?|mayor|president|stocks?|crypto|programming|math homework|sports?)\b",
    re.I,
)
EMERGENCY_PATTERNS = re.compile(
    r"(chest pain|shortness of breath|severe bleeding|stroke|suicid|overdose|unconscious)",
    re.I,
)


def is_non_medical(msg: str) -> bool:
    return bool(NON_MEDICAL_PATTERNS.search(msg))

def is_emergency(msg: str) -> bool:
    return bool(EMERGENCY_PATTERNS.search(msg))

def has_sources(text: str) -> bool:
    # accept URLs or DOI or recognizable org refs
    return bool(re.search(r"(https?://\S+|doi:\S+|\b(WHO|CDC|NICE|UpToDate|BMJ|PubMed)\b)", text, re.I))

def trim_to_words(text: str, max_words: int) -> str:
    words = text.split()
    return " ".join(words[:max_words])




# --- Conversation store (in-memory per session_id) ---
# For multi-instance or persistence, back this with Redis or a DB.
# Key: session_id -> list[dict(role, parts=[{text}])]
CONV_STORE: Dict[str, List[Dict[str, Any]]] = {}

def get_history(session_id: str) -> List[Dict[str, Any]]:
    return CONV_STORE.setdefault(session_id, [])

def set_history(session_id: str, history: List[Dict[str, Any]]) -> None:
    CONV_STORE[session_id] = history

def summarize_old_context(session_id: str) -> None:
    history = get_history(session_id)
    if len(history) > settings.SUMMARIZE_OVER_MESSAGES:
        old = history[:-settings.KEEP_RECENT_MESSAGES]
        summary_prompt = f"Summarize this medical conversation concisely:\n{old}"
        summary_resp = client.models.generate_content(
            model=settings.MODEL_NAME,
            contents=[{"role": "user", "parts": [{"text": summary_prompt}]}],
        )
        summarized = summary_resp.text if hasattr(summary_resp, "text") else "(summary unavailable)"
        new_hist: List[Dict[str, Any]] = [
            {"role": "user", "parts": [{"text": f"Previous context: {summarized}"}]},
            *history[-settings.KEEP_RECENT_MESSAGES:],
        ]
        set_history(session_id, new_hist)

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.session_id or not req.message:
        raise HTTPException(status_code=400, detail="session_id and message are required")

    history = get_history(req.session_id)

    # append user message
    history.append({"role": "user", "parts": [{"text": req.message}]})
    set_history(req.session_id, history)
    if is_non_medical(req.message):
        raise HTTPException(status_code=400, detail="Only medical questions are allowed.")

    # generate response
    try:
        response = client.models.generate_content(
            model=settings.MODEL_NAME,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                max_output_tokens=settings.MAX_OUTPUT_TOKENS,
                temperature=settings.TEMPERATURE,
            ),
            contents=history,
        )
        text = response.text if hasattr(response, "text") else ""
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {e}")

    if not text:
        
        raise HTTPException(status_code=502, detail="Empty response from model")

    # append assistant response
    history.append({"role": "model", "parts": [{"text": text}]})
    set_history(req.session_id, history)

    # optional summarization to keep context short
    summarize_old_context(req.session_id)

    return ChatResponse(text=text)

@app.post("/clear")
def clear(req: ClearRequest):
    set_history(req.session_id, [])
    return {"ok": True}

@app.get("/history/{session_id}", response_model=HistorySummary)
def history_summary(session_id: str):
    history = get_history(session_id)
    return HistorySummary(messages=len(history))
