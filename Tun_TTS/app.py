"""
Tun_TTS — FastAPI server for Tunisian Text-to-Speech
=====================================================
XTTS v2 fine-tuned on TunArTTS corpus (Tunisian Darija).
Designed for Azure Container Apps / Azure App Service deployment.
"""

import os
import re
import io
import tempfile
from pathlib import Path

import torch
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from TTS.api import TTS

# ── Config ────────────────────────────────────────────────────
MODEL_DIR = os.getenv("MODEL_DIR", "./xtts-tun-finetuned")
SPEAKER_WAV = os.getenv("SPEAKER_WAV", "./speaker_reference.wav")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── FastAPI App ───────────────────────────────────────────────
app = FastAPI(
    title="TuniSign TTS API",
    description="Tunisian Darija Text-to-Speech with XTTS v2",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Tunisian Text Normalizer ─────────────────────────────────
FRENCH_TO_ARABIC = {
    "merci": "مرسي",
    "bonne journée": "بون جورني",
    "bonjour": "بونجور",
    "pizza": "بيتزا",
    "climat": "كليما",
    "taxi": "تاكسي",
    "bus": "بيس",
    "internet": "انترنات",
    "télé": "تيلي",
    "portable": "بورتابل",
}

SPELLING_FIXES = {
    "مانيش": "ما نيش",
    "كيفاه": "كيفاش",
}


def normalize_tunisian(text: str) -> str:
    """Normalize Tunisian Darija text for XTTS input."""
    # French loanword mapping
    lower = text
    for fr, ar in FRENCH_TO_ARABIC.items():
        lower = re.sub(re.escape(fr), ar, lower, flags=re.IGNORECASE)

    # Spelling standardization
    for variant, standard in SPELLING_FIXES.items():
        lower = lower.replace(variant, standard)

    # Number expansion (basic)
    number_map = {
        "0": "صفر", "1": "واحد", "2": "زوز", "3": "ثلاثة",
        "4": "أربعة", "5": "خمسة", "6": "ستة", "7": "سبعة",
        "8": "ثمنية", "9": "تسعة", "10": "عشرة",
    }
    for digit, word in number_map.items():
        lower = lower.replace(digit, word)

    return lower.strip()


# ── Load XTTS Model ──────────────────────────────────────────
print(f"Loading XTTS v2 model from {MODEL_DIR} on {DEVICE}...")

# Check if we have a fine-tuned model directory or use the base model
config_path = Path(MODEL_DIR) / "config.json"
if config_path.exists():
    tts = TTS(
        model_path=str(Path(MODEL_DIR) / "model.pth"),
        config_path=str(config_path),
        gpu=torch.cuda.is_available(),
    )
else:
    # Fallback to base XTTS v2
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=torch.cuda.is_available())

print("XTTS v2 model loaded.")


# ── Request Schema ────────────────────────────────────────────
class SynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to synthesize")
    language: str = Field(default="ar", description="Language code (default: ar)")
    normalize: bool = Field(default=True, description="Apply Tunisian text normalization")


# ── API Routes ────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "healthy", "model": MODEL_DIR, "device": DEVICE}


@app.post("/synthesize")
async def synthesize(request: SynthesizeRequest):
    """
    Synthesize Tunisian Darija speech from text.

    Returns a WAV audio file.
    """
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    # Apply Tunisian normalization
    if request.normalize:
        text = normalize_tunisian(text)

    try:
        # Generate speech to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        tts.tts_to_file(
            text=text,
            file_path=tmp_path,
            speaker_wav=SPEAKER_WAV if Path(SPEAKER_WAV).exists() else None,
            language=request.language,
        )

        # Read the generated WAV and return it
        audio_data, sample_rate = sf.read(tmp_path)
        os.unlink(tmp_path)

        # Convert to WAV bytes
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sample_rate, format="WAV")
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=tunisign_tts.wav"},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")


@app.get("/")
async def root():
    return {
        "service": "TuniSign TTS API",
        "version": "1.0.0",
        "endpoints": {
            "/synthesize": "POST — Generate speech from text (JSON body)",
            "/health": "GET — Health check",
        },
    }
