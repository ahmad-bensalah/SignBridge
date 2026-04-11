import io
import json
import os
import wave
import zipfile
from pathlib import Path

import requests
import streamlit as st
from vosk import KaldiRecognizer, Model, SetLogLevel
from ui_theme import apply_tunisign_theme, render_primary_sidebar

SetLogLevel(-1)

st.set_page_config(
    page_title="TuniSign AI | Speech to Text",
    page_icon="🎧",
    layout="wide",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Manrope', sans-serif;
    }

    .stApp {
        background: #f6f7f9;
    }

    .shell {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
    }

    .subtle {
        color: #5f6b77;
        font-size: 0.92rem;
    }

    .stButton > button {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

apply_tunisign_theme()

# ── Model download URL & paths ────────────────────────────────
HF_MODEL_URL = (
    "https://huggingface.co/Sali7a8603/Tunisian_STT/resolve/main/STT_Tun_Model.zip"
)
MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
DEFAULT_MODEL_PATH = MODEL_DIR / "whisper-tts-model"
MODEL_ZIP_PATH = MODEL_DIR / "STT_Tun_Model.zip"


def download_model(destination: Path) -> None:
    """Download and extract the Whisper-TTS model from HuggingFace."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    st.info("⬇️ Downloading Tunisian STT model from HuggingFace (≈ 542 MB)…")
    progress_bar = st.progress(0, text="Starting download…")

    with requests.get(HF_MODEL_URL, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0

        with open(MODEL_ZIP_PATH, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = min(downloaded / total, 1.0)
                    progress_bar.progress(
                        pct,
                        text=f"Downloaded {downloaded / 1e6:.0f} / {total / 1e6:.0f} MB",
                    )

    progress_bar.progress(1.0, text="Extracting model…")

    with zipfile.ZipFile(MODEL_ZIP_PATH, "r") as zf:
        zf.extractall(destination.parent)

    # The zip may extract into a subfolder — find and rename it
    extracted_dirs = [
        d for d in destination.parent.iterdir()
        if d.is_dir() and d.name not in ("whisper-tts-model",)
    ]
    # If the zip extracted a single folder (e.g. "STT_Tun_Model"), rename it
    if not destination.exists() and extracted_dirs:
        for d in extracted_dirs:
            # Check if it contains model files (conf/, am/, graph/, etc.)
            if any((d / sub).exists() for sub in ("conf", "am", "graph", "ivector")):
                d.rename(destination)
                break

    # Clean up the zip
    MODEL_ZIP_PATH.unlink(missing_ok=True)

    progress_bar.empty()
    st.success("✅ Model downloaded and extracted successfully!")


def transcribe_wav(audio_bytes, model):
    with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        compression = wav_file.getcomptype()

        if compression != "NONE":
            raise ValueError("WAV must be uncompressed PCM.")
        if channels != 1:
            raise ValueError("WAV must be mono (1 channel).")
        if sample_width != 2:
            raise ValueError("WAV must be 16-bit PCM.")

        recognizer = KaldiRecognizer(model, float(sample_rate))
        recognizer.SetWords(True)

        parts = []
        while True:
            data = wav_file.readframes(4000)
            if not data:
                break
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").strip()
                if text:
                    parts.append(text)

        final_result = json.loads(recognizer.FinalResult())
        final_text = final_result.get("text", "").strip()
        if final_text:
            parts.append(final_text)

    transcript = " ".join(parts).strip()
    return transcript, {
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width_bytes": sample_width,
    }


@st.cache_resource
def load_model(model_path):
    return Model(model_path)


with st.sidebar:
    render_primary_sidebar(show_caption=False)
    st.markdown("---")
    st.header("STT Settings")
    model_path = st.text_input("Whisper model path", value=str(DEFAULT_MODEL_PATH))
    st.caption("Expected folder: Demo/model/whisper-tts-model")


st.title("Speech to Text")
st.caption("Upload or record WAV audio and transcribe with the Tunisian Whisper-TTS model.")

if "stt_transcript" not in st.session_state:
    st.session_state["stt_transcript"] = ""

if "stt_details" not in st.session_state:
    st.session_state["stt_details"] = None

left_col, right_col = st.columns([1.15, 1], gap="large")

with left_col:
    st.markdown('<div class="shell">', unsafe_allow_html=True)
    st.subheader("Input Audio")
    st.markdown("<p class='subtle'>Upload a WAV file or record directly in the browser.</p>", unsafe_allow_html=True)
    audio_upload = st.file_uploader("Upload WAV audio", type=["wav"])
    audio_record = st.audio_input("Or record audio")

    selected_audio = audio_record if audio_record is not None else audio_upload

    transcribe_clicked = st.button("Transcribe Audio", type="primary", use_container_width=True)

    if transcribe_clicked:
        if selected_audio is None:
            st.error("Please upload or record a WAV audio file first.")
        else:
            # Auto-download model if it doesn't exist
            if not Path(model_path).exists():
                try:
                    download_model(Path(model_path))
                except Exception as dl_err:
                    st.error(f"Failed to download model: {dl_err}")

            if Path(model_path).exists():
                try:
                    model = load_model(model_path)
                    audio_bytes = selected_audio.read()
                    transcript, details = transcribe_wav(audio_bytes, model)
                    st.session_state["stt_transcript"] = transcript
                    st.session_state["stt_details"] = details

                    if transcript:
                        st.success("Transcription complete.")
                    else:
                        st.warning("No speech was recognized. Try clearer audio.")

                except wave.Error:
                    st.error("Invalid WAV file. Please provide a valid WAV (PCM mono, 16-bit).")
                except ValueError as err:
                    st.error(str(err))
                except Exception as err:
                    st.error(f"STT error: {err}")
            else:
                st.error(f"Model folder not found: {model_path}")
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="shell">', unsafe_allow_html=True)
    st.subheader("Transcript")
    transcript_value = st.session_state.get("stt_transcript", "")
    st.text_area("Recognized text", value=transcript_value, height=280)

    if transcript_value:
        st.download_button(
            "Download transcript",
            data=transcript_value.encode("utf-8"),
            file_name="transcript.txt",
            mime="text/plain",
            use_container_width=True,
        )
    else:
        st.info("Transcript will appear here after processing.")

    details_value = st.session_state.get("stt_details")
    if details_value:
        with st.expander("Audio details"):
            st.json(details_value)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("TuniSign AI Platform | Speech to Text module")
