"""
Tun_STT — FastAPI server for Tunisian Speech-to-Text
=====================================================
Whisper-small fine-tuned on Tunisian Darija + T-HSAB profanity filter.
Designed for Azure Container Apps / Azure App Service deployment.
"""

import os
import re
import json
import tempfile
from pathlib import Path

import torch
import librosa
import openpyxl
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# ── Config ────────────────────────────────────────────────────
MODEL_DIR = os.getenv("MODEL_DIR", "./whisper-tun-finetuned")
XLSX_PATH = os.getenv("XLSX_PATH", "./T-HSAB.xlsx")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── FastAPI App ───────────────────────────────────────────────
app = FastAPI(
    title="TuniSign STT API",
    description="Tunisian Darija Speech-to-Text with profanity filtering",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Load Whisper Model ────────────────────────────────────────
print(f"Loading Whisper model from {MODEL_DIR} on {DEVICE}...")
processor = WhisperProcessor.from_pretrained(MODEL_DIR)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()
print("Whisper model loaded.")


# ── Load Profanity Filter ─────────────────────────────────────
def _load_bad_words(xlsx_path: str) -> set:
    """Extract bad words from the T-HSAB hate-speech dataset."""
    if not Path(xlsx_path).exists():
        print(f"Warning: {xlsx_path} not found — profanity filter disabled.")
        return set()

    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb.active

    rows = list(ws.iter_rows(min_row=2, values_only=True))
    wb.close()

    # Collect words from rows labeled as hate/abusive
    words = set()
    for row in rows:
        if len(row) >= 2:
            text, label = row[0], row[1]
            if isinstance(text, str) and isinstance(label, str):
                label_lower = label.strip().lower()
                if label_lower in ("hate", "abusive", "offensive"):
                    for token in text.split():
                        cleaned = token.strip(".,!?؟،؛")
                        if len(cleaned) >= 2:
                            words.add(cleaned)
    return words


BAD_WORDS = _load_bad_words(XLSX_PATH)
print(f"Profanity filter loaded: {len(BAD_WORDS)} entries.")

# Build regex pattern from bad words (sorted longest first)
if BAD_WORDS:
    _sorted_bad = sorted(BAD_WORDS, key=len, reverse=True)
    _PROFANITY_RE = re.compile(
        r"(?<!\w)([وفبلك]?)(" + "|".join(re.escape(w) for w in _sorted_bad) + r")(?!\w)",
        re.UNICODE,
    )
else:
    _PROFANITY_RE = None


def _normalize_arabic(text: str) -> str:
    """Basic Arabic text normalization."""
    text = re.sub(r"[إأآٱ]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)  # Remove diacritics
    return text.strip()


def filter_profanity(text: str, replacement: str = "****") -> str:
    """Censor bad words in the transcription output."""
    if _PROFANITY_RE is None:
        return text
    normalized = _normalize_arabic(text)
    return _PROFANITY_RE.sub(replacement, normalized)


# ── API Routes ────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "healthy", "model": MODEL_DIR, "device": DEVICE}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    filter_bad_words: bool = True,
):
    """
    Transcribe an audio file (WAV, MP3, FLAC, etc.) to Tunisian Arabic text.

    - **file**: Audio file to transcribe.
    - **filter_bad_words**: If true, censor profanity in the output (default: true).
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    # Save uploaded file to a temp path for librosa to read
    suffix = Path(file.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # Load and resample audio to 16kHz
        audio, sr = librosa.load(tmp_path, sr=16000, mono=True)

        # Process through Whisper
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(DEVICE)

        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                language="ar",
                task="transcribe",
            )

        raw_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        # Apply profanity filter
        if filter_bad_words:
            filtered_text = filter_profanity(raw_text)
        else:
            filtered_text = raw_text

        return {
            "text": filtered_text,
            "raw_text": raw_text,
            "profanity_filtered": filter_bad_words,
            "audio_duration_s": round(len(audio) / 16000, 2),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    finally:
        os.unlink(tmp_path)


@app.get("/")
async def root():
    return {
        "service": "TuniSign STT API",
        "version": "1.0.0",
        "endpoints": {
            "/transcribe": "POST — Upload audio for transcription",
            "/health": "GET — Health check",
        },
    }
