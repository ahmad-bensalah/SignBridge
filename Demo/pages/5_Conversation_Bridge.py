import json
import html
import io
import os
import re
import tempfile
import time
import wave
import zipfile
from collections import deque
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from elevenlabs.client import ElevenLabs
from skeleton_sign_renderer import skeleton_html_for_tokens
from text_to_sign_engine import translate_text_to_sign_tokens
from ui_theme import apply_tunisign_theme, render_primary_sidebar

st.set_page_config(
    page_title="TuniSign AI | Conversation Bridge",
    page_icon="💬",
    layout="wide",
)

try:
    import av
    import cv2
    import mediapipe as mp
    import numpy as np
    import tensorflow as tf
    from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer
    from tensorflow import keras
except Exception as import_error:
    st.error("Missing dependencies for the conversation module.")
    st.code("pip install -r requirements.txt")
    st.caption(f"Import error: {import_error}")
    st.stop()

import requests as http_requests

try:
    from vosk import KaldiRecognizer, Model, SetLogLevel

    SetLogLevel(-1)
    WHISPER_IMPORT_ERROR = None
except Exception as whisper_import_error:
    KaldiRecognizer = None
    Model = None
    WHISPER_IMPORT_ERROR = whisper_import_error

API_KEY = "sk_dc270b6a089f34d50834bf3e207addaee1082463f7bd4a84"
VOICE_ID = "InB4o9iYj3MQ4HtGs2KV"

ASSET_DIR = Path(__file__).resolve().parents[1] / "signtotext"
MODEL_PATH = ASSET_DIR / "tunisl_v4.keras"
LABELS_PATH = ASSET_DIR / "tunisl_v4_labels.json"
LANDMARKER_PATH = ASSET_DIR / "hand_landmarker.task"

# ── Whisper-TTS STT model paths & download URL ───────────────
HF_MODEL_URL = (
    "https://huggingface.co/Sali7a8603/Tunisian_STT/resolve/main/STT_Tun_Model.zip"
)
WHISPER_MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
WHISPER_MODEL_PATH = WHISPER_MODEL_DIR / "whisper-tts-model"
WHISPER_MODEL_ZIP = WHISPER_MODEL_DIR / "STT_Tun_Model.zip"


def _ensure_whisper_model() -> bool:
    """Download and extract the Whisper-TTS model if it is not already present."""
    if WHISPER_MODEL_PATH.exists():
        return True
    try:
        WHISPER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        st.info("⬇️ Downloading Tunisian STT model from HuggingFace (≈ 542 MB)…")
        progress = st.progress(0, text="Starting download…")
        with http_requests.get(HF_MODEL_URL, stream=True, timeout=600) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(WHISPER_MODEL_ZIP, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        progress.progress(
                            min(downloaded / total, 1.0),
                            text=f"Downloaded {downloaded / 1e6:.0f} / {total / 1e6:.0f} MB",
                        )
        progress.progress(1.0, text="Extracting model…")
        with zipfile.ZipFile(WHISPER_MODEL_ZIP, "r") as zf:
            zf.extractall(WHISPER_MODEL_DIR)
        # Rename extracted folder if needed
        if not WHISPER_MODEL_PATH.exists():
            for d in WHISPER_MODEL_DIR.iterdir():
                if d.is_dir() and d.name != "whisper-tts-model":
                    if any((d / s).exists() for s in ("conf", "am", "graph", "ivector")):
                        d.rename(WHISPER_MODEL_PATH)
                        break
        WHISPER_MODEL_ZIP.unlink(missing_ok=True)
        progress.empty()
        st.success("✅ Model downloaded and extracted successfully!")
        return WHISPER_MODEL_PATH.exists()
    except Exception as dl_err:
        st.error(f"Failed to download Whisper-TTS model: {dl_err}")
        return False

WINDOW_SIZE = 16
DEFAULT_SIGN_CONFIDENCE = 0.55

LABEL_TO_ARABIC = {
    "Demandes/3aslema": "عسلامة",
    "Demandes/5adamet": "خدمة",
    "Demandes/assam": "اسمك",
    "Demandes/barnamjk": "برنامجك",
    "Demandes/chabeb": "شباب",
    "Demandes/cv": "سيفي",
    "Demandes/demande": "طلب",
    "Demandes/enti": "أنتِ",
    "Demandes/labes": "لاباس",
    "Demandes/lyoum": "اليوم",
    "Demandes/mar7ba": "مرحبا",
    "Demandes/n3awnek": "نعاونك",
    "Demandes/nekteblk": "نكتبلك",
    "Demandes/nemchi": "نمشي",
    "Demandes/non": "لا",
    "Demandes/oui": "آه",
    "Demandes/radio": "راديو",
    "Demandes/se7a": "صحة",
    "Demandes/siye7a": "سياحة",
    "Demandes/t7eb": "تحب",
    "Demandes/ta3lim": "تعليم",
    "Demandes/ta3raf": "تعرف",
    "Demandes/ta9ra": "تقرا",
    "Demandes/telvza": "تلفزة",
    "Demandes/tha9afa": "ثقافة",
    "Destinations/baladya": "بلدية",
    "Destinations/banka": "بنكة",
    "Destinations/bousta": "بوسطة",
    "Destinations/dar": "دار",
    "Destinations/ma7kma": "محكمة",
    "Destinations/mostawsaf": "مستوصف",
    "Destinations/sbitar": "سبيطار",
    "Destinations/wzara": "وزارة",
    "Famille/3ayla": "عيلة",
    "Famille/5al-3am": "خال/عم",
    "Famille/5ou": "خو",
    "Famille/bent": "بنت",
    "Famille/bou": "بو",
    "Famille/eben": "ابن",
    "Famille/jad": "جد",
    "Famille/jadda": "جدة",
    "Famille/mar2a": "مرا",
    "Famille/o5t": "أخت",
    "Famille/om": "أم",
    "Famille/tfol": "طفل",
    "Jours/5mis": "الخميس",
    "Jours/a7ad": "الأحد",
    "Jours/erb3a": "الأربعاء",
    "Jours/jom3a": "الجمعة",
    "Jours/sebt": "السبت",
    "Jours/thleth": "الثلاثاء",
    "Jours/thnin": "الاثنين",
    "Transport/car": "كار",
    "Transport/karhba": "كرهبة",
    "Transport/louage": "لواج",
    "Transport/métro": "ميترو",
    "Transport/taxi": "تاكسي",
    "Transport/train": "تران",
}


def audio_to_bytes(audio_result):
    if isinstance(audio_result, (bytes, bytearray)):
        return bytes(audio_result)

    chunks = []
    for chunk in audio_result:
        if isinstance(chunk, (bytes, bytearray)):
            chunks.append(bytes(chunk))

    if not chunks:
        raise ValueError("No audio data returned from ElevenLabs.")

    return b"".join(chunks)


def parse_elevenlabs_error(error):
    status_code = getattr(error, "status_code", None)
    body = getattr(error, "body", None)
    code = None
    message = str(error)

    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, dict):
            code = detail.get("code")
            message = detail.get("message", message)
        elif isinstance(detail, str):
            message = detail

    return status_code, code, message


def transcribe_wav(audio_bytes, model):
    with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
        if wav_file.getcomptype() != "NONE":
            raise ValueError("WAV must be uncompressed PCM.")
        if wav_file.getnchannels() != 1:
            raise ValueError("WAV must be mono (1 channel).")
        if wav_file.getsampwidth() != 2:
            raise ValueError("WAV must be 16-bit PCM.")

        recognizer = KaldiRecognizer(model, float(wav_file.getframerate()))
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

    return " ".join(parts).strip()


@st.cache_resource
def load_whisper_model(model_path):
    return Model(model_path)


def text_to_sign_script(text):
    return translate_text_to_sign_tokens(text).get("script", "")


def sanitize_chat_text(text):
    if text is None:
        return ""
    # Strip accidental HTML fragments that can leak into chat bubbles.
    cleaned = re.sub(r"</?[^>]+>", "", str(text))
    # Remove any remaining angle-bracket fragments defensively.
    cleaned = cleaned.replace("<", " ").replace(">", " ")
    return cleaned.strip()


def format_chat_time(timestamp):
    if not timestamp:
        return "--:--"
    return time.strftime("%H:%M", time.localtime(timestamp))


class TemporalAttention(keras.layers.Layer):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.score = keras.layers.Dense(1, activation="tanh")

    def call(self, x):
        weights = tf.nn.softmax(self.score(x), axis=1)
        return tf.reduce_sum(x * weights, axis=1)


@st.cache_resource
def load_model_and_labels():
    model = keras.models.load_model(
        str(MODEL_PATH),
        custom_objects={"TemporalAttention": TemporalAttention},
        compile=False,
    )
    with open(LABELS_PATH, "r", encoding="utf-8") as labels_file:
        label_data = json.load(labels_file)
    id_to_label = {int(k): v for k, v in label_data["id_to_label"].items()}
    return model, id_to_label


@st.cache_resource
def load_mediapipe():
    base_options = mp.tasks.BaseOptions
    hand_landmarker = mp.tasks.vision.HandLandmarker
    hand_landmarker_options = mp.tasks.vision.HandLandmarkerOptions
    vision_running_mode = mp.tasks.vision.RunningMode

    options = hand_landmarker_options(
        base_options=base_options(model_asset_path=str(LANDMARKER_PATH)),
        running_mode=vision_running_mode.IMAGE,
        num_hands=1,
    )
    return hand_landmarker.create_from_options(options)


def normalize_window(window):
    window_array = window.reshape(-1, 21, 3).copy()
    window_array -= window_array[:, 0:1, :]
    scale = np.linalg.norm(window_array, axis=2).max(axis=1, keepdims=True)
    scale = np.where(scale < 0.15, 0.15, scale)
    window_array /= scale[:, :, None]

    coords = window_array.reshape(-1, 63)
    tips = [4, 8, 12, 16, 20]
    angles_seq = []

    for frame_landmarks in window_array:
        wrist = frame_landmarks[0]
        vectors = [frame_landmarks[t] - wrist for t in tips]
        angles = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                cosine = np.dot(vectors[i], vectors[j]) / (
                    np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]) + 1e-6
                )
                angles.append(float(np.clip(cosine, -1, 1)))
        angles_seq.append(angles)

    return np.concatenate([coords, np.array(angles_seq, dtype=np.float32)], axis=1)


def extract_landmarks(frame_bgr, landmarker):
    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = landmarker.detect(mp_image)
    if not result.hand_landmarks:
        return None

    landmarks = result.hand_landmarks[0]
    return np.array([value for point in landmarks for value in (point.x, point.y, point.z)], dtype=np.float32)


def translate_uploaded_video(video_bytes, model, id_to_label, landmarker, conf_threshold, filename="uploaded.mp4"):
    if not video_bytes:
        return {"text": "", "words": [], "frames": 0, "hand_frames": 0}

    temp_path = None
    accepted_labels = []
    landmark_buffer = deque(maxlen=WINDOW_SIZE * 3)
    total_frames = 0
    hand_frames = 0
    last_label = None
    last_accept_frame = -1000

    capture = None
    try:
        file_suffix = Path(filename).suffix.lower()
        if file_suffix not in {".mp4", ".mov", ".webm", ".avi", ".mkv", ".gif"}:
            file_suffix = ".mp4"

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as temp_video:
            temp_video.write(video_bytes)
            temp_path = temp_video.name

        capture = cv2.VideoCapture(temp_path)
        if not capture.isOpened():
            return {"text": "", "words": [], "frames": 0, "hand_frames": 0}

        while True:
            success, frame = capture.read()
            if not success:
                break

            total_frames += 1
            if total_frames % 2 != 0:
                continue

            vector = extract_landmarks(frame, landmarker)
            if vector is None:
                continue

            hand_frames += 1
            landmark_buffer.append(vector)

            if len(landmark_buffer) < WINDOW_SIZE or total_frames % 4 != 0:
                continue

            seq = np.array(list(landmark_buffer)[-WINDOW_SIZE:], dtype=np.float32)
            window = normalize_window(seq)[np.newaxis]
            probs = model.predict(window, verbose=0)[0]
            pred_id = int(np.argmax(probs))
            confidence = float(probs[pred_id])
            if confidence < conf_threshold:
                continue

            label = id_to_label.get(pred_id, "unknown")
            if label == "unknown":
                continue

            if label == last_label and (total_frames - last_accept_frame) < 24:
                continue

            accepted_labels.append(label)
            last_label = label
            last_accept_frame = total_frames

    finally:
        if capture is not None:
            capture.release()
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass

    translated_words = []
    for label in accepted_labels:
        word = LABEL_TO_ARABIC.get(label, label.split("/")[-1])
        if not translated_words or translated_words[-1] != word:
            translated_words.append(word)

    return {
        "text": " ".join(translated_words).strip(),
        "words": translated_words,
        "frames": total_frames,
        "hand_frames": hand_frames,
    }


class TunSLProcessor(VideoProcessorBase):
    def __init__(self):
        self.landmark_buffer = deque(maxlen=WINDOW_SIZE * 3)
        self.last_prediction = None
        self.last_confidence = 0.0
        self.last_pred_time = 0.0
        self.frame_count = 0
        self.hand_detected = False
        self.model = None
        self.id_to_label = None
        self.landmarker = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        if self.landmarker is not None:
            vector = extract_landmarks(image, self.landmarker)
            if vector is not None:
                self.landmark_buffer.append(vector)
                self.hand_detected = True
            else:
                self.hand_detected = False

        # Match Sign-to-Text cadence to reduce jitter and CPU usage.
        if self.model is not None and self.frame_count % 8 == 0 and len(self.landmark_buffer) >= WINDOW_SIZE:
            seq = np.array(list(self.landmark_buffer)[-WINDOW_SIZE:], dtype=np.float32)
            window = normalize_window(seq)[np.newaxis]
            probs = self.model.predict(window, verbose=0)[0]
            pred_id = int(np.argmax(probs))
            confidence = float(probs[pred_id])
            if confidence >= DEFAULT_SIGN_CONFIDENCE and self.id_to_label:
                self.last_prediction = self.id_to_label.get(pred_id, "unknown")
                self.last_confidence = confidence
                self.last_pred_time = time.time()

        frame_h, frame_w = image.shape[:2]
        indicator_color = (0, 255, 136) if self.hand_detected else (80, 80, 80)
        cv2.circle(image, (30, 30), 12, indicator_color, -1)
        cv2.putText(
            image,
            "Hand" if self.hand_detected else "No hand",
            (50, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            indicator_color,
            2,
        )

        if self.last_prediction and (time.time() - self.last_pred_time < 15.0):
            confidence_pct = int(self.last_confidence * 100)
            cv2.rectangle(image, (0, frame_h - 70), (frame_w, frame_h), (15, 15, 15), -1)
            bar_width = int(frame_w * self.last_confidence)
            bar_color = (0, 200, 100) if self.last_confidence > 0.7 else (0, 160, 255)
            cv2.rectangle(image, (0, frame_h - 6), (bar_width, frame_h), bar_color, -1)

            label_raw = self.last_prediction.split("/")[-1]
            cv2.putText(
                image,
                f"{label_raw}  {confidence_pct}%",
                (16, frame_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return av.VideoFrame.from_ndarray(image, format="bgr24")


st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;600;700&family=Manrope:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Manrope', sans-serif;
    }

    .stApp {
        background: #f6f7f9;
    }

    .shell {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: none;
    }

    .section-title {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #61707f;
        font-weight: 700;
    }

    .sentence {
        font-family: 'IBM Plex Sans Arabic', sans-serif;
        direction: rtl;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        min-height: 72px;
        padding: 0.9rem 1rem;
        font-size: 1.35rem;
        color: #20374d;
        box-shadow: none;
    }

    .chat-shell {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem 1rem 0.4rem;
        box-shadow: none;
    }

    .composer-shell {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: none;
    }

    .empty-thread {
        background: #ffffff;
        border: 1px dashed #94a3b8;
        border-radius: 10px;
        padding: 1rem;
        color: #5e6e7f;
    }

    .mode-note {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.58rem 0.7rem;
        color: #2f4a63;
        font-size: 0.84rem;
        margin-bottom: 0.65rem;
    }

    .chat-heading {
        font-size: 0.88rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #61707f;
        font-weight: 700;
        margin-bottom: 0.45rem;
    }

    .chat-avatar {
        width: 2.05rem;
        height: 2.05rem;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.78rem;
        font-weight: 800;
        border: 1px solid rgba(15, 76, 129, 0.18);
    }

    .chat-avatar-in {
        background: rgba(15, 76, 129, 0.08);
        color: #0f4c81;
    }

    .chat-avatar-out {
        background: rgba(214, 40, 40, 0.09);
        color: #d62828;
    }

    .msg-wrap {
        margin-bottom: 0.75rem;
    }

    .msg-wrap.outgoing {
        text-align: right;
    }

    .msg-meta {
        font-size: 0.76rem;
        color: #6e7d8d;
        margin-bottom: 0.26rem;
    }

    .msg-bubble {
        border-radius: 12px;
        padding: 0.68rem 0.92rem;
        line-height: 1.4;
        display: inline-block;
        max-width: 100%;
        border: 1px solid transparent;
    }

    .msg-bubble.incoming {
        background: #ffffff;
        border-color: rgba(15, 76, 129, 0.2);
        color: #22374b;
        border-top-left-radius: 6px;
        text-align: left;
    }

    .msg-bubble.outgoing {
        background: #2563eb;
        color: #ffffff;
        border-color: #2563eb;
        border-top-right-radius: 6px;
        text-align: left;
    }

    .msg-tag {
        font-size: 0.73rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        display: block;
        margin-bottom: 0.24rem;
        opacity: 0.92;
    }

    .chat-compose-note {
        font-size: 0.83rem;
        color: #61707f;
    }

    .sign-script-preview {
        margin-top: 0.55rem;
        padding-top: 0.5rem;
        border-top: 1px dashed rgba(15, 76, 129, 0.26);
        font-size: 0.9rem;
        line-height: 1.35;
    }

    .sign-script-preview strong {
        display: block;
        font-size: 0.75rem;
        letter-spacing: 0.03em;
        margin-bottom: 0.22rem;
        opacity: 0.9;
        text-transform: uppercase;
    }

    .stButton > button {
        border-radius: 8px;
    }
</style>
""",
    unsafe_allow_html=True,
)

apply_tunisign_theme()

if "conversation_words" not in st.session_state:
    st.session_state.conversation_words = []
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "conversation_video_nonce" not in st.session_state:
    st.session_state.conversation_video_nonce = 0
if "compose_text_nonce" not in st.session_state:
    st.session_state.compose_text_nonce = 0
if "compose_voice_nonce" not in st.session_state:
    st.session_state.compose_voice_nonce = 0

with st.sidebar:
    render_primary_sidebar()

    st.markdown("---")
    sign_conf_threshold = st.slider(
        "Sign confidence threshold",
        min_value=0.30,
        max_value=0.95,
        value=0.55,
        step=0.05,
    )
    model_options = {
        "Multilingual v2 (Best quality)": "eleven_multilingual_v2",
        "Flash v2.5 (Fastest)": "eleven_flash_v2_5",
        "Turbo v2.5 (Balanced)": "eleven_turbo_v2_5",
    }
    model_choice = st.selectbox("Voice model", list(model_options.keys()))
    model_id = model_options[model_choice]

st.title("Conversation Bridge")
st.caption("Minimal messenger layout for text, voice, camera sign, and uploaded sign video.")

with st.spinner("Loading sign recognition assets..."):
    try:
        if not MODEL_PATH.exists() or not LABELS_PATH.exists() or not LANDMARKER_PATH.exists():
            raise FileNotFoundError("Files missing in tts/signtotext")
        sign_model, id_to_label = load_model_and_labels()
        hand_landmarker = load_mediapipe()
    except FileNotFoundError as file_error:
        st.error("Sign model assets are missing in tts/signtotext.")
        st.caption(str(file_error))
        st.stop()

history = st.session_state.conversation_history

# Clean previously stored messages so legacy malformed text doesn't render tags.
for item in history:
    if "text" in item and isinstance(item.get("text"), str):
        item["text"] = sanitize_chat_text(item["text"])

st.markdown("### Chat")

main_chat_col, main_compose_col = st.columns([1.55, 1], gap="large")

with main_chat_col:
    thread_header_cols = st.columns([4, 1])
    with thread_header_cols[0]:
        st.markdown("#### Conversation Thread")
    with thread_header_cols[1]:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.conversation_history = []
            st.session_state.conversation_words = []
            st.rerun()

    if not history:
        st.markdown(
            """
            <div class="empty-thread">
                No messages yet. Use the composer to send text, voice, camera sign, or uploaded sign video.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="chat-shell">', unsafe_allow_html=True)
        st.markdown('<div class="chat-heading">Live Thread</div>', unsafe_allow_html=True)

        for item in history:
            sender = item.get("sender") or ("deaf" if item.get("type", "").startswith("sign") else "normal")
            is_outgoing = sender == "normal"
            created_at = format_chat_time(item.get("created_at"))
            message_text = html.escape(sanitize_chat_text(item.get("text", ""))).replace("\n", "<br>")
            message_text = message_text if message_text else "(no text)"

            message_type = item.get("type", "")
            if message_type == "normal_voice":
                bubble_tag = "Voice to text + sign"
            elif message_type == "sign_video":
                bubble_tag = "Sign video to text"
            elif message_type == "sign_video_voice":
                bubble_tag = "Sign video to text + speech"
            elif message_type == "sign_voice":
                bubble_tag = "Camera sign to text + speech"
            elif message_type == "sign_text":
                bubble_tag = "Camera sign to text"
            else:
                bubble_tag = "Text to sign"

            if is_outgoing:
                bubble_col, avatar_col = st.columns([14, 1], gap="small")
                with bubble_col:
                    st.markdown(
                        f"""
                        <div class="msg-wrap outgoing">
                            <div class="msg-meta">Normal Person • {created_at}</div>
                            <div class="msg-bubble outgoing">
                                <span class="msg-tag">{bubble_tag}</span>
                                {message_text}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if item.get("audio"):
                        st.audio(item["audio"], format="audio/wav")
                    if item.get("sign_tokens"):
                        skeleton_html = skeleton_html_for_tokens(
                            item["sign_tokens"],
                            fps=6,
                            max_frames_per_token=12,
                        )
                        if skeleton_html:
                            st.caption("Continuous skeleton sign output")
                            components.html(skeleton_html, height=560, scrolling=False)
                with avatar_col:
                    st.markdown('<div class="chat-avatar chat-avatar-out">N</div>', unsafe_allow_html=True)
            else:
                avatar_col, bubble_col = st.columns([1, 14], gap="small")
                with avatar_col:
                    st.markdown('<div class="chat-avatar chat-avatar-in">D</div>', unsafe_allow_html=True)
                with bubble_col:
                    st.markdown(
                        f"""
                        <div class="msg-wrap incoming">
                            <div class="msg-meta">Deaf Signer • {created_at}</div>
                            <div class="msg-bubble incoming">
                                <span class="msg-tag">{bubble_tag}</span>
                                {message_text}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if item.get("video"):
                        if item.get("video_mime") == "image/gif":
                            st.image(item["video"])
                        else:
                            st.video(item["video"])
                    if item.get("audio"):
                        st.audio(item["audio"], format="audio/mp3")
                    if item.get("stats"):
                        stats = item["stats"]
                        st.caption(
                            f"frames: {stats.get('frames', 0)} | "
                            f"hand frames: {stats.get('hand_frames', 0)} | "
                            f"words: {len(stats.get('words', []))}"
                        )

        st.markdown('</div>', unsafe_allow_html=True)

with main_compose_col:
    st.markdown('<div class="composer-shell">', unsafe_allow_html=True)
    st.markdown("#### Message Composer")
    st.markdown(
        '<div class="mode-note">Choose input type based on sender: Normal person uses text/voice; Deaf signer uses camera/video.</div>',
        unsafe_allow_html=True,
    )

    input_mode = st.radio(
        "Input choice",
        options=["Text", "Voice", "Camera", "Upload Video"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if input_mode == "Text":
        text_message = st.text_area(
            "Write a chat message",
            key=f"compose_text_message_{st.session_state.compose_text_nonce}",
            height=130,
        )
        if text_message.strip():
            preview_script = text_to_sign_script(text_message.strip())
            if preview_script:
                st.markdown(
                    f"""
                    <div class="sign-script-preview" style="margin-top:0.1rem">
                        <strong>Sign preview</strong>
                        {html.escape(preview_script)}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        if st.button("Send Text", type="primary", use_container_width=True):
            if not text_message.strip():
                st.warning("Please type a message before sending.")
            else:
                clean_text = sanitize_chat_text(text_message.strip())
                parsed_sign = translate_text_to_sign_tokens(clean_text)
                st.session_state.conversation_history.append(
                    {
                        "type": "normal_text",
                        "sender": "normal",
                        "text": clean_text,
                        "sign_script": parsed_sign.get("script", ""),
                        "sign_tokens": parsed_sign.get("tokens", []),
                        "created_at": time.time(),
                    }
                )
                st.session_state.compose_text_nonce += 1
                st.rerun()

    elif input_mode == "Voice":
        st.caption("Voice is transcribed to text with the Whisper-TTS model, then converted to sign output.")
        recorded_audio = st.audio_input(
            "Record voice message",
            key=f"compose_voice_audio_{st.session_state.compose_voice_nonce}",
        )
        voice_caption = st.text_input(
            "Optional text caption",
            key=f"compose_voice_caption_{st.session_state.compose_voice_nonce}",
        )
        if st.button("Send Voice", type="primary", use_container_width=True):
            if recorded_audio is None:
                st.warning("Please record a voice message first.")
            elif WHISPER_IMPORT_ERROR is not None:
                st.error(f"Whisper-TTS dependency is not available: {WHISPER_IMPORT_ERROR}")
            else:
                # Auto-download model if missing
                if not WHISPER_MODEL_PATH.exists():
                    _ensure_whisper_model()
                if not WHISPER_MODEL_PATH.exists():
                    st.error(f"Whisper-TTS model not found at: {WHISPER_MODEL_PATH}")
                else:
                    try:
                        with st.spinner("Transcribing voice..."):
                            whisper_model = load_whisper_model(str(WHISPER_MODEL_PATH))
                            transcript = transcribe_wav(recorded_audio.getvalue(), whisper_model)
                    except wave.Error:
                        st.error("Recorded audio is not valid WAV PCM for Whisper transcription.")
                        transcript = ""
                    except Exception as stt_error:
                        st.error(f"Speech-to-text error: {stt_error}")
                        transcript = ""

                    if transcript:
                        caption_text = voice_caption.strip()
                        payload_text = transcript
                        if caption_text:
                            payload_text = f"Transcript: {transcript}\nCaption: {caption_text}"
                        payload_text = sanitize_chat_text(payload_text)

                        parsed_sign = translate_text_to_sign_tokens(transcript)

                        st.session_state.conversation_history.append(
                            {
                                "type": "normal_voice",
                                "sender": "normal",
                                "text": payload_text,
                                "audio": recorded_audio.getvalue(),
                                "sign_script": parsed_sign.get("script", ""),
                                "sign_tokens": parsed_sign.get("tokens", []),
                                "created_at": time.time(),
                            }
                        )
                        st.session_state.compose_voice_nonce += 1
                        st.rerun()
                    else:
                        st.warning("No speech transcript detected. Please speak clearly and try again.")

    elif input_mode == "Camera":
        st.markdown('<div class="section-title">Live camera sign capture</div>', unsafe_allow_html=True)

        rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
        ctx = webrtc_streamer(
            key="conversation_sign_cam",
            video_processor_factory=TunSLProcessor,
            rtc_configuration=rtc_config,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        latest_label = None
        latest_confidence = 0.0
        latest_arabic = ""

        if ctx.video_processor:
            ctx.video_processor.model = sign_model
            ctx.video_processor.id_to_label = id_to_label
            ctx.video_processor.landmarker = hand_landmarker

            if ctx.video_processor.last_prediction:
                latest_label = ctx.video_processor.last_prediction
                latest_confidence = ctx.video_processor.last_confidence
                latest_arabic = LABEL_TO_ARABIC.get(latest_label, latest_label.split("/")[-1])

        if latest_label and latest_confidence >= sign_conf_threshold:
            st.success(f"Latest sign: {latest_arabic} ({int(latest_confidence * 100)}%)")
        elif ctx.video_processor and not ctx.video_processor.hand_detected:
            st.info("No hand detected yet. Move your hand clearly into frame.")
        else:
            st.info("Waiting for a stable sign prediction...")

        action_cols = st.columns(3)
        with action_cols[0]:
            if st.button("Capture Word", use_container_width=True):
                if latest_label and latest_confidence >= sign_conf_threshold:
                    cleaned_word = sanitize_chat_text(latest_arabic)
                    if cleaned_word:
                        st.session_state.conversation_words.append(cleaned_word)
                else:
                    st.warning("No confident word to capture.")
        with action_cols[1]:
            if st.button("Undo Last", use_container_width=True):
                if st.session_state.conversation_words:
                    st.session_state.conversation_words.pop()
        with action_cols[2]:
            if st.button("Clear Phrase", use_container_width=True):
                st.session_state.conversation_words = []

        camera_phrase = sanitize_chat_text(" ".join(st.session_state.conversation_words)).strip()
        if camera_phrase:
            st.markdown(f'<div class="sentence">{camera_phrase}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="sentence" style="color:#6e8091">الكلمات ستظهر هنا...</div>', unsafe_allow_html=True)

        if st.button("Send Camera Message", type="primary", use_container_width=True):
            if not camera_phrase:
                st.error("Capture at least one word before sending.")
            else:
                message_payload = {
                    "type": "sign_text",
                    "sender": "deaf",
                    "text": camera_phrase,
                    "created_at": time.time(),
                }

                if not API_KEY:
                    st.warning("Missing ElevenLabs API key. Message sent as text only.")
                else:
                    try:
                        client = ElevenLabs(api_key=API_KEY)
                        audio_stream = client.text_to_speech.convert(
                            text=camera_phrase,
                            voice_id=VOICE_ID,
                            model_id=model_id,
                            output_format="mp3_44100_128",
                        )
                        message_payload["type"] = "sign_voice"
                        message_payload["audio"] = audio_to_bytes(audio_stream)
                    except Exception as speech_error:
                        status_code, error_code, message = parse_elevenlabs_error(speech_error)
                        if status_code == 401:
                            st.error("Invalid ElevenLabs key.")
                        elif status_code == 402 and error_code == "paid_plan_required":
                            st.error("Current voice is not allowed on this ElevenLabs plan.")
                        elif status_code == 429:
                            st.error("Rate limit reached. Please retry.")
                        else:
                            st.error(f"TTS error ({status_code}): {message}")
                        message_payload["type"] = "sign_text"

                st.session_state.conversation_history.append(message_payload)
                st.session_state.conversation_words = []
                st.rerun()

    else:
        uploaded_video = st.file_uploader(
            "Upload sign video",
            type=["mp4", "mov", "webm", "avi", "mkv", "gif"],
            key=f"conversation_video_upload_{st.session_state.conversation_video_nonce}",
        )
        st.caption("Uploaded sign videos are translated to text and converted to speech automatically when possible.")

        if uploaded_video is not None:
            if uploaded_video.type == "image/gif":
                st.image(uploaded_video.getvalue())
            else:
                st.video(uploaded_video.getvalue())

        if st.button("Translate and Send Video", type="primary", use_container_width=True):
            if uploaded_video is None:
                st.warning("Please upload a video file first.")
            else:
                video_bytes = uploaded_video.getvalue()
                with st.spinner("Translating signs from uploaded video..."):
                    translated = translate_uploaded_video(
                        video_bytes=video_bytes,
                        model=sign_model,
                        id_to_label=id_to_label,
                        landmarker=hand_landmarker,
                        conf_threshold=sign_conf_threshold,
                        filename=uploaded_video.name,
                    )

                translated_text = translated["text"]
                if not translated_text:
                    translated_text = "No confident sign translation detected from this video."
                translated_text = sanitize_chat_text(translated_text)

                payload = {
                    "type": "sign_video",
                    "sender": "deaf",
                    "text": translated_text,
                    "video": video_bytes,
                    "video_mime": uploaded_video.type or "video/mp4",
                    "stats": translated,
                    "created_at": time.time(),
                }

                if translated.get("text"):
                    if not API_KEY:
                        st.warning("Missing ElevenLabs API key. Video message sent as text only.")
                    else:
                        try:
                            client = ElevenLabs(api_key=API_KEY)
                            audio_stream = client.text_to_speech.convert(
                                text=translated["text"],
                                voice_id=VOICE_ID,
                                model_id=model_id,
                                output_format="mp3_44100_128",
                            )
                            payload["type"] = "sign_video_voice"
                            payload["audio"] = audio_to_bytes(audio_stream)
                        except Exception as speech_error:
                            status_code, error_code, message = parse_elevenlabs_error(speech_error)
                            if status_code == 401:
                                st.error("Invalid ElevenLabs key.")
                            elif status_code == 402 and error_code == "paid_plan_required":
                                st.error("Current voice is not allowed on this ElevenLabs plan.")
                            elif status_code == 429:
                                st.error("Rate limit reached. Please retry.")
                            else:
                                st.error(f"TTS error ({status_code}): {message}")

                st.session_state.conversation_history.append(payload)
                st.session_state.conversation_video_nonce += 1
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("TuniSign AI Platform | Conversation Bridge chat with input choices: text, voice, camera, video upload")
