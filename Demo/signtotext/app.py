"""
TunSL Live — Real-time Tunisian Sign Language Recognition
Streamlit app: webcam → MediaPipe landmarks → BiLSTM model → Arabic label
"""

import streamlit as st
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from tensorflow import keras
import json
import math
import time
from collections import deque
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TunSL Live",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Arabic label map (Tunisian label → Arabic display) ───────────────────────
LABEL_TO_ARABIC = {
    "Demandes/3aslema":   "عسلامة",
    "Demandes/5adamet":   "خدمة",
    "Demandes/assam":     "اسمك",
    "Demandes/barnamjk":  "برنامجك",
    "Demandes/chabeb":    "شباب",
    "Demandes/cv":        "سيفي",
    "Demandes/demande":   "طلب",
    "Demandes/enti":      "أنتِ",
    "Demandes/labes":     "لاباس",
    "Demandes/lyoum":     "اليوم",
    "Demandes/mar7ba":    "مرحبا",
    "Demandes/n3awnek":   "نعاونك",
    "Demandes/nekteblk":  "نكتبلك",
    "Demandes/nemchi":    "نمشي",
    "Demandes/non":       "لا",
    "Demandes/oui":       "آه",
    "Demandes/radio":     "راديو",
    "Demandes/se7a":      "صحة",
    "Demandes/siye7a":    "سياحة",
    "Demandes/t7eb":      "تحب",
    "Demandes/ta3lim":    "تعليم",
    "Demandes/ta3raf":    "تعرف",
    "Demandes/ta9ra":     "تقرا",
    "Demandes/telvza":    "تلفزة",
    "Demandes/tha9afa":   "ثقافة",
    "Destinations/baladya":   "بلدية",
    "Destinations/banka":     "بنكة",
    "Destinations/bousta":    "بوسطة",
    "Destinations/dar":       "دار",
    "Destinations/ma7kma":    "محكمة",
    "Destinations/mostawsaf": "مستوصف",
    "Destinations/sbitar":    "سبيطار",
    "Destinations/wzara":     "وزارة",
    "Famille/3ayla":   "عيلة",
    "Famille/5al-3am": "خال/عم",
    "Famille/5ou":     "خو",
    "Famille/bent":    "بنت",
    "Famille/bou":     "بو",
    "Famille/eben":    "ابن",
    "Famille/jad":     "جد",
    "Famille/jadda":   "جدة",
    "Famille/mar2a":   "مرا",
    "Famille/o5t":     "أخت",
    "Famille/om":      "أم",
    "Famille/tfol":    "طفل",
    "Jours/5mis":   "الخميس",
    "Jours/a7ad":   "الأحد",
    "Jours/erb3a":  "الأربعاء",
    "Jours/jom3a":  "الجمعة",
    "Jours/sebt":   "السبت",
    "Jours/thleth": "الثلاثاء",
    "Jours/thnin":  "الاثنين",
    "Transport/car":    "كار",
    "Transport/karhba": "كرهبة",
    "Transport/louage": "لواج",
    "Transport/métro":  "ميترو",
    "Transport/taxi":   "تاكسي",
    "Transport/train":  "تران",
}

CATEGORY_ARABIC = {
    "Demandes":     "طلبات",
    "Destinations": "وجهات",
    "Famille":      "عيلة",
    "Jours":        "أيام",
    "Transport":    "نقل",
}

# ─── Constants (must match training config) ───────────────────────────────────
WINDOW_SIZE        = 16
STRIDE             = 4
INPUT_FEATURES     = 73
CONFIDENCE_THRESHOLD = 0.55

# ─── Custom Keras layer (must match saved model) ───────────────────────────────
class TemporalAttention(keras.layers.Layer):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.score = keras.layers.Dense(1, activation='tanh')
    def call(self, x):
        w = tf.nn.softmax(self.score(x), axis=1)
        return tf.reduce_sum(x * w, axis=1)

# ─── Load model + labels (cached) ─────────────────────────────────────────────
@st.cache_resource
def load_model_and_labels():
    # compile=False skips deserializing smooth_loss (a tf.function)
    # which Keras can't reconstruct at load time outside the training env.
    # We only need inference so no recompile needed.
    model = keras.models.load_model(
        "tunisl_v4.keras",
        custom_objects={"TemporalAttention": TemporalAttention},
        compile=False,
    )
    with open("tunisl_v4_labels.json", "r", encoding="utf-8") as f:
        ldata = json.load(f)
    id_to_label = {int(k): v for k, v in ldata["id_to_label"].items()}
    return model, id_to_label

# ─── MediaPipe setup (cached) ─────────────────────────────────────────────────
@st.cache_resource
def load_mediapipe():
    BaseOptions           = mp.tasks.BaseOptions
    HandLandmarker        = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode     = mp.tasks.vision.RunningMode
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
    )
    return HandLandmarker.create_from_options(options)

# ─── Feature engineering (same as training) ───────────────────────────────────
def normalize_window(win):
    """(WINDOW_SIZE, 63) → (WINDOW_SIZE, 73)"""
    w = win.reshape(-1, 21, 3).copy()
    w -= w[:, 0:1, :]
    scale = np.linalg.norm(w, axis=2).max(axis=1, keepdims=True)
    scale = np.where(scale < 0.15, 0.15, scale)
    w /= scale[:, :, None]
    coords = w.reshape(-1, 63)
    tips = [4, 8, 12, 16, 20]
    angles_seq = []
    for frame_lms in w:
        wrist = frame_lms[0]
        vecs  = [frame_lms[t] - wrist for t in tips]
        angles = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                cos_a = np.dot(vecs[i], vecs[j]) / (
                    np.linalg.norm(vecs[i]) * np.linalg.norm(vecs[j]) + 1e-6)
                angles.append(float(np.clip(cos_a, -1, 1)))
        angles_seq.append(angles)
    return np.concatenate([coords, np.array(angles_seq, dtype=np.float32)], axis=1)

def extract_landmarks(frame_bgr, landmarker):
    """BGR frame → (63,) landmark vector or None"""
    img_rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result   = landmarker.detect(mp_image)
    if not result.hand_landmarks:
        return None
    lm = result.hand_landmarks[0]
    return np.array([v for p in lm for v in (p.x, p.y, p.z)], dtype=np.float32)

# ─── Video processor for streamlit-webrtc ─────────────────────────────────────
class TunSLProcessor(VideoProcessorBase):
    def __init__(self):
        self.landmark_buffer = deque(maxlen=WINDOW_SIZE * 3)
        self.last_prediction  = None
        self.last_confidence  = 0.0
        self.last_pred_time   = 0.0
        self.frame_count      = 0
        self.hand_detected    = False
        # These are set from outside after construction
        self.model       = None
        self.id_to_label = None
        self.landmarker  = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        # Extract landmarks every frame
        if self.landmarker is not None:
            vec = extract_landmarks(img, self.landmarker)
            if vec is not None:
                self.landmark_buffer.append(vec)
                self.hand_detected = True
            else:
                self.hand_detected = False

        # Run inference every 8 frames if we have enough landmarks
        if (self.model is not None
                and self.frame_count % 8 == 0
                and len(self.landmark_buffer) >= WINDOW_SIZE):

            seq = np.array(list(self.landmark_buffer)[-WINDOW_SIZE:], dtype=np.float32)
            win = normalize_window(seq)[np.newaxis]           # (1, 16, 73)
            probs = self.model.predict(win, verbose=0)[0]
            pred_id    = int(np.argmax(probs))
            confidence = float(probs[pred_id])

            if confidence >= CONFIDENCE_THRESHOLD and self.id_to_label:
                self.last_prediction = self.id_to_label.get(pred_id, "unknown")
                self.last_confidence = confidence
                self.last_pred_time  = time.time()

        # Draw overlay on frame
        h, w = img.shape[:2]

        # Hand detection indicator
        color = (0, 255, 136) if self.hand_detected else (80, 80, 80)
        cv2.circle(img, (30, 30), 12, color, -1)
        cv2.putText(img, "Hand" if self.hand_detected else "No hand",
                    (50, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Prediction overlay
        if self.last_prediction and (time.time() - self.last_pred_time < 15.0):
            arabic  = LABEL_TO_ARABIC.get(self.last_prediction, self.last_prediction.split("/")[-1])
            conf_pct = int(self.last_confidence * 100)
            # Background bar
            cv2.rectangle(img, (0, h - 70), (w, h), (15, 15, 15), -1)
            # Confidence bar fill
            bar_w = int(w * self.last_confidence)
            bar_color = (0, 200, 100) if self.last_confidence > 0.7 else (0, 160, 255)
            cv2.rectangle(img, (0, h - 6), (bar_w, h), bar_color, -1)
            # Label text (latin fallback since OpenCV can't render Arabic)
            label_raw = self.last_prediction.split("/")[-1]
            cv2.putText(img, f"{label_raw}  {conf_pct}%",
                        (16, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                        (255, 255, 255), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans Arabic', sans-serif; }

.main { background: #0f0f0f; }

.hero {
    text-align: center;
    padding: 2rem 1rem 1rem;
}
.hero h1 {
    font-size: 2.8rem;
    font-weight: 700;
    color: #00e87a;
    letter-spacing: -1px;
    margin: 0;
}
.hero p {
    color: #888;
    font-size: 1rem;
    margin: 0.4rem 0 0;
}

.pred-card {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
    margin-bottom: 1rem;
}
.pred-arabic {
    font-size: 3.5rem;
    font-weight: 700;
    color: #00e87a;
    direction: rtl;
    line-height: 1.2;
}
.pred-latin {
    font-size: 1rem;
    color: #666;
    margin-top: 0.3rem;
}
.pred-conf {
    font-size: 0.85rem;
    color: #444;
    margin-top: 0.2rem;
}

.word-pill {
    display: inline-block;
    background: #1e1e1e;
    border: 1px solid #2e2e2e;
    border-radius: 20px;
    padding: 0.4rem 1rem;
    margin: 0.25rem;
    font-size: 1.3rem;
    color: #e0e0e0;
    direction: rtl;
    cursor: default;
}
.word-pill:hover {
    background: #252525;
    border-color: #00e87a44;
}

.sentence-box {
    background: #111;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    min-height: 60px;
    direction: rtl;
    font-size: 1.8rem;
    color: #f0f0f0;
    text-align: right;
    letter-spacing: 0.5px;
    line-height: 1.6;
}

.stat-chip {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    text-align: center;
    font-size: 0.8rem;
    color: #666;
}
.stat-chip b { color: #00e87a; font-size: 1.1rem; display: block; }

.section-label {
    font-size: 0.75rem;
    color: #444;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🤟 TunSL Live</h1>
    <p>لغة الإشارة التونسية — Tunisian Sign Language Recognition</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    conf_threshold = st.slider(
        "Confidence threshold", 0.30, 0.95, CONFIDENCE_THRESHOLD, 0.05,
        help="Minimum confidence to accept a prediction"
    )
    collect_words = st.toggle("Collect words into sentence", value=True)
    st.divider()

    st.markdown("### 📖 Sign vocabulary")
    category_filter = st.selectbox(
        "Filter by category",
        ["All"] + list(CATEGORY_ARABIC.keys()),
        format_func=lambda x: f"{x} — {CATEGORY_ARABIC.get(x, '')}" if x != "All" else "All categories"
    )
    st.markdown("---")
    for full_label, arabic in LABEL_TO_ARABIC.items():
        cat, word = full_label.split("/")
        if category_filter != "All" and cat != category_filter:
            continue
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;padding:3px 0;"
            f"border-bottom:1px solid #1e1e1e;font-size:0.85rem;'>"
            f"<span style='color:#555'>{word}</span>"
            f"<span style='color:#ccc;direction:rtl'>{arabic}</span></div>",
            unsafe_allow_html=True
        )

# ─── Load resources ───────────────────────────────────────────────────────────
with st.spinner("Loading model..."):
    try:
        model, id_to_label = load_model_and_labels()
        landmarker = load_mediapipe()
        model_loaded = True
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}\n\nMake sure `tunisl_v4.keras` and `tunisl_v4_labels.json` are in the same folder as `app.py`.")
        model_loaded = False
        st.stop()

# ─── Session state ────────────────────────────────────────────────────────────
if "collected_words" not in st.session_state:
    st.session_state.collected_words = []
if "last_added"  not in st.session_state:
    st.session_state.last_added = None
if "last_add_time" not in st.session_state:
    st.session_state.last_add_time = 0.0

# ─── Layout ───────────────────────────────────────────────────────────────────
col_cam, col_result = st.columns([3, 2], gap="large")

with col_cam:
    st.markdown('<div class="section-label">📷 Camera feed</div>', unsafe_allow_html=True)

    RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    ctx = webrtc_streamer(
        key="tunisl",
        video_processor_factory=TunSLProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # Inject model/landmarker into processor after creation
    if ctx.video_processor:
        ctx.video_processor.model        = model
        ctx.video_processor.id_to_label  = id_to_label
        ctx.video_processor.landmarker   = landmarker

    st.caption("Sign a TSL word in front of your camera. Hold the sign for ~15 seconds.")

with col_result:
    st.markdown('<div class="section-label">🔍 Live prediction</div>', unsafe_allow_html=True)

    pred_placeholder  = st.empty()
    conf_placeholder  = st.empty()

    st.markdown('<div class="section-label" style="margin-top:1.5rem">📝 Collected words</div>',
                unsafe_allow_html=True)
    sentence_placeholder = st.empty()

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("➕ Add word", use_container_width=True,
                     help="Manually add the last detected word to the sentence"):
            if (ctx.video_processor
                    and ctx.video_processor.last_prediction
                    and ctx.video_processor.last_confidence >= conf_threshold):
                label = ctx.video_processor.last_prediction
                arabic = LABEL_TO_ARABIC.get(label, label.split("/")[-1])
                st.session_state.collected_words.append(arabic)
    with btn_col2:
        if st.button("🗑 Clear sentence", use_container_width=True):
            st.session_state.collected_words = []

    # Stats row
    st.markdown('<div class="section-label" style="margin-top:1.5rem">📊 Session stats</div>',
                unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)

# ─── Live update loop ─────────────────────────────────────────────────────────
if ctx.state.playing and ctx.video_processor:
    proc = ctx.video_processor

    # Refresh display
    while ctx.state.playing:
        time.sleep(0.15)

        pred   = proc.last_prediction
        conf   = proc.last_confidence
        fresh  = (time.time() - proc.last_pred_time) < 15.0

        # Prediction card
        if pred and fresh and conf >= conf_threshold:
            arabic    = LABEL_TO_ARABIC.get(pred, pred.split("/")[-1])
            latin     = pred.split("/")[-1]
            conf_pct  = int(conf * 100)
            bar_color = "#00e87a" if conf > 0.7 else "#ff9933"

            pred_placeholder.markdown(f"""
            <div class="pred-card">
                <div class="pred-arabic">{arabic}</div>
                <div class="pred-latin">{latin}</div>
                <div class="pred-conf">{conf_pct}% confidence</div>
                <div style="margin-top:0.8rem;height:6px;background:#2a2a2a;border-radius:3px;overflow:hidden;">
                    <div style="width:{conf_pct}%;height:100%;background:{bar_color};
                                border-radius:3px;transition:width 0.3s;"></div>
                </div>
            </div>""", unsafe_allow_html=True)

            # Auto-collect: add word if it's new and confident
            if (collect_words
                    and pred != st.session_state.last_added
                    and time.time() - st.session_state.last_add_time > 15.0
                    and conf >= conf_threshold):
                st.session_state.collected_words.append(arabic)
                st.session_state.last_added    = pred
                st.session_state.last_add_time = time.time()

        elif not proc.hand_detected:
            pred_placeholder.markdown("""
            <div class="pred-card">
                <div style="color:#333;font-size:1.5rem">✋</div>
                <div style="color:#444;font-size:0.9rem;margin-top:0.5rem">No hand detected</div>
            </div>""", unsafe_allow_html=True)
        else:
            pred_placeholder.markdown("""
            <div class="pred-card">
                <div style="color:#555;font-size:1rem">⏳ Signing...</div>
            </div>""", unsafe_allow_html=True)

        # Sentence box
        words = st.session_state.collected_words
        if words:
            sentence_html = " · ".join(
                f'<span class="word-pill">{w}</span>' for w in words
            )
            sentence_placeholder.markdown(
                f'<div class="sentence-box">{sentence_html}</div>',
                unsafe_allow_html=True
            )
        else:
            sentence_placeholder.markdown(
                '<div class="sentence-box" style="color:#333">الكلمات ستظهر هنا...</div>',
                unsafe_allow_html=True
            )

        # Stats
        with s1:
            st.markdown(f'<div class="stat-chip"><b>{len(words)}</b>words</div>',
                        unsafe_allow_html=True)
        with s2:
            st.markdown(f'<div class="stat-chip"><b>{proc.frame_count}</b>frames</div>',
                        unsafe_allow_html=True)
        with s3:
            lm_count = len(proc.landmark_buffer)
            st.markdown(f'<div class="stat-chip"><b>{lm_count}</b>landmarks</div>',
                        unsafe_allow_html=True)

else:
    # Idle state
    pred_placeholder.markdown("""
    <div class="pred-card">
        <div style="color:#333;font-size:2rem">🤟</div>
        <div style="color:#444;margin-top:0.5rem">Start the camera to begin</div>
    </div>""", unsafe_allow_html=True)
    sentence_placeholder.markdown(
        '<div class="sentence-box" style="color:#2a2a2a">الكلمات ستظهر هنا...</div>',
        unsafe_allow_html=True
    )
