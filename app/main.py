from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import librosa
import numpy as np
import io
import os
from app.voice_pipeline import VoicePipeline

app = FastAPI(title="AgriVoice NAIC 2026")

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

# Lazy initialize pipeline
pipeline = None

@app.on_event("startup")
def load_pipeline():
    global pipeline
    pipeline = VoicePipeline()

@app.get("/", response_class=HTMLResponse)
def index():
    with open("web/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/chat")
async def chat(audio: UploadFile = File(...)):
    contents = await audio.read()
    
    # Load audio with librosa
    audio_data, sr = librosa.load(io.BytesIO(contents), sr=16000)
    
    result = pipeline.process_audio(audio_data, sr)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
