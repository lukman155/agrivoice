from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import io
import os
import wave
import logging
from app.voice_pipeline import VoicePipeline

logger = logging.getLogger(__name__)

try:
    import librosa
    HAVE_LIBROSA = True
except ImportError:
    HAVE_LIBROSA = False

app = FastAPI(title="AgriVoice NAIC 2026", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static/outputs", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize pipeline
pipeline = VoicePipeline()

class TextChatRequest(BaseModel):
    query: str
    language: Optional[str] = "ha"

@app.get("/", response_class=HTMLResponse)
def index():
    with open("web/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/chat")
async def chat(
    audio: Optional[UploadFile] = File(None),
    transcript: Optional[str] = Form(None),
    language: Optional[str] = Form(None)
):
    target_lang = (language or "en").strip()
    # If client passed the transcribed speech text directly from speech recognition
    if transcript and transcript.strip():
        logger.info(f"Processing client-transcribed query ({target_lang}): {transcript}")
        return pipeline.process_text(transcript.strip(), target_lang)

    contents = await audio.read() if audio else b""
    if contents:
        return pipeline.process_audio_bytes(contents, language=target_lang)

    return {
        "transcript": "",
        "language": target_lang,
        "advice": "No audio data received. Please record your question.",
        "source": "AgriVoice Monitor",
        "audio_url": None
    }

@app.post("/api/chat-text")
async def chat_text(req: TextChatRequest):
    result = pipeline.process_text(req.query, req.language)
    return result

class SetEndpointRequest(BaseModel):
    api_url: str

@app.get("/api/status")
def get_status():
    from app.config import settings
    return {
        "status": "online",
        "api_url": settings.natlas_api_url,
        "backend": settings.llm_backend,
        "is_gpu_connected": bool(settings.natlas_api_url and "ngrok" in settings.natlas_api_url)
    }

@app.post("/api/set-endpoint")
def set_endpoint(req: SetEndpointRequest):
    from app.config import settings
    new_url = req.api_url.strip().rstrip('/')
    settings.natlas_api_url = new_url
    os.environ["NATLAS_API_URL"] = new_url
    logger.info(f"Updated live GPU endpoint to: {new_url}")
    return {"status": "updated", "api_url": new_url}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8001, reload=True)
