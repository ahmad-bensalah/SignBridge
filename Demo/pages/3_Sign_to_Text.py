import json
import math
import tempfile
import time
from collections import deque
from pathlib import Path

import streamlit as st
from ui_theme import apply_tunisign_theme, render_primary_sidebar

st.set_page_config(
    page_title="TuniSign AI | Sign to Text",
    page_icon="🖐️",
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
    st.error("Missing Sign-to-Text dependencies. Install project requirements first.")
    st.code("pip install -r requirements.txt")
    st.caption(f"Import error: {import_error}")
    st.stop()


ASSET_DIR = Path(__file__).resolve().parents[1] / "signtotext"
MODEL_PATH = ASSET_DIR / "tunisl_v4.keras"
LABELS_PATH = ASSET_DIR / "tunisl_v4_labels.json"
LANDMARKER_PATH = ASSET_DIR / "hand_landmarker.task"

WINDOW_SIZE = 16
CONFIDENCE_THRESHOLD = 0.55

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

CATEGORY_ARABIC = {
    "Demandes": "طلبات",
    "Destinations": "وجهات",
    "Famille": "عيلة",
    "Jours": "أيام",
    "Transport": "نقل",
}


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
    """Run sign-to-text on an uploaded video clip and return a best-effort sentence."""
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

            # Sample frames to keep processing responsive for longer clips.
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

            # Prevent flooding with repeated labels when signer holds one pose.
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
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        if self.landmarker is not None:
            vector = extract_landmarks(img, self.landmarker)
            if vector is not None:
                self.landmark_buffer.append(vector)
                self.hand_detected = True
            else:
                self.hand_detected = False

        if self.model is not None and self.frame_count % 8 == 0 and len(self.landmark_buffer) >= WINDOW_SIZE:
            seq = np.array(list(self.landmark_buffer)[-WINDOW_SIZE:], dtype=np.float32)
            window = normalize_window(seq)[np.newaxis]
            probs = self.model.predict(window, verbose=0)[0]
            pred_id = int(np.argmax(probs))
            confidence = float(probs[pred_id])

            if confidence >= CONFIDENCE_THRESHOLD and self.id_to_label:
                self.last_prediction = self.id_to_label.get(pred_id, "unknown")
                self.last_confidence = confidence
                self.last_pred_time = time.time()

        frame_h, frame_w = img.shape[:2]
        indicator_color = (0, 255, 136) if self.hand_detected else (80, 80, 80)
        cv2.circle(img, (30, 30), 12, indicator_color, -1)
        cv2.putText(
            img,
            "Hand" if self.hand_detected else "No hand",
            (50, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            indicator_color,
            2,
        )

        if self.last_prediction and (time.time() - self.last_pred_time < 15.0):
            confidence_pct = int(self.last_confidence * 100)
            cv2.rectangle(img, (0, frame_h - 70), (frame_w, frame_h), (15, 15, 15), -1)
            bar_width = int(frame_w * self.last_confidence)
            bar_color = (0, 200, 100) if self.last_confidence > 0.7 else (0, 160, 255)
            cv2.rectangle(img, (0, frame_h - 6), (bar_width, frame_h), bar_color, -1)

            label_raw = self.last_prediction.split("/")[-1]
            cv2.putText(
                img,
                f"{label_raw}  {confidence_pct}%",
                (16, frame_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;600;700&family=Manrope:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Manrope', sans-serif;
    }

    .stApp {
        background: #f6f7f9;
    }

    .pred-card {
        background: #ffffff;
        border: 1px solid rgba(15, 76, 129, 0.18);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: none;
    }

    .pred-arabic {
        font-family: 'IBM Plex Sans Arabic', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        color: #0f4c81;
        direction: rtl;
        line-height: 1.15;
    }

    .pred-latin {
        font-size: 1rem;
        color: #5b6470;
        margin-top: 0.2rem;
    }

    .pred-conf {
        font-size: 0.86rem;
        color: #65727f;
        margin-top: 0.2rem;
    }

    .word-pill {
        display: inline-block;
        background: #f6f9fc;
        border: 1px solid rgba(15, 76, 129, 0.2);
        border-radius: 20px;
        padding: 0.42rem 1rem;
        margin: 0.22rem;
        font-family: 'IBM Plex Sans Arabic', sans-serif;
        font-size: 1.2rem;
        color: #1f3247;
        direction: rtl;
    }

    .sentence-box {
        background: #ffffff;
        border: 1px solid rgba(15, 76, 129, 0.22);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        min-height: 64px;
        direction: rtl;
        font-family: 'IBM Plex Sans Arabic', sans-serif;
        font-size: 1.5rem;
        color: #1e3348;
        text-align: right;
        line-height: 1.5;
        box-shadow: none;
    }

    .stat-chip {
        background: #ffffff;
        border: 1px solid rgba(15, 76, 129, 0.18);
        border-radius: 10px;
        padding: 0.6rem 0.8rem;
        text-align: center;
        font-size: 0.8rem;
        color: #687483;
        box-shadow: none;
    }

    .stat-chip b {
        color: #0f4c81;
        font-size: 1.1rem;
        display: block;
    }

    .section-label {
        font-size: 0.75rem;
        color: #61707f;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }

    .stButton > button {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

apply_tunisign_theme()


with st.sidebar:
    render_primary_sidebar(show_caption=False)

    st.markdown("---")
    st.markdown("### Sign Settings")
    conf_threshold = st.slider(
        "Confidence threshold",
        0.30,
        0.95,
        CONFIDENCE_THRESHOLD,
        0.05,
        help="Minimum confidence to accept a prediction.",
    )
    collect_words = st.toggle("Auto-collect words", value=True)

    st.markdown("---")
    st.markdown("### Vocabulary")
    category_filter = st.selectbox(
        "Filter by category",
        ["All"] + list(CATEGORY_ARABIC.keys()),
        format_func=lambda x: f"{x} - {CATEGORY_ARABIC.get(x, '')}" if x != "All" else "All categories",
    )

    for full_label, arabic_word in LABEL_TO_ARABIC.items():
        category, word = full_label.split("/")
        if category_filter != "All" and category != category_filter:
            continue
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid #2e2b4a;font-size:0.85rem;'>"
            f"<span style='color:#9a95c1'>{word}</span>"
            f"<span style='color:#e9e6f8;direction:rtl'>{arabic_word}</span></div>",
            unsafe_allow_html=True,
        )


st.title("Sign to Text")
st.caption("Live or uploaded sign recognition to translated text.")


with st.spinner("Loading Sign-to-Text model..."):
    try:
        if not MODEL_PATH.exists() or not LABELS_PATH.exists() or not LANDMARKER_PATH.exists():
            raise FileNotFoundError("One or more files are missing in tts/signtotext")
        model, id_to_label = load_model_and_labels()
        landmarker = load_mediapipe()
    except FileNotFoundError as file_error:
        st.error(
            "Sign-to-Text assets are missing. Ensure these files exist in tts/signtotext: "
            "tunisl_v4.keras, tunisl_v4_labels.json, hand_landmarker.task"
        )
        st.caption(str(file_error))
        st.stop()


if "collected_words" not in st.session_state:
    st.session_state.collected_words = []
if "last_added" not in st.session_state:
    st.session_state.last_added = None
if "last_add_time" not in st.session_state:
    st.session_state.last_add_time = 0.0
if "sent_sign_videos" not in st.session_state:
    st.session_state.sent_sign_videos = []
if "video_uploader_nonce" not in st.session_state:
    st.session_state.video_uploader_nonce = 0


col_cam, col_result = st.columns([3, 2], gap="large")

with col_cam:
    st.markdown('<div class="section-label">Camera feed</div>', unsafe_allow_html=True)

    rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    ctx = webrtc_streamer(
        key="tunisl",
        video_processor_factory=TunSLProcessor,
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if ctx.video_processor:
        ctx.video_processor.model = model
        ctx.video_processor.id_to_label = id_to_label
        ctx.video_processor.landmarker = landmarker

    st.caption("Show one TSL sign clearly in front of your camera and hold briefly for stable detection.")

with col_result:
    st.markdown('<div class="section-label">Live prediction</div>', unsafe_allow_html=True)
    pred_placeholder = st.empty()

    st.markdown('<div class="section-label" style="margin-top:1.2rem">Collected words</div>', unsafe_allow_html=True)
    sentence_placeholder = st.empty()

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("Add word", use_container_width=True):
            if (
                ctx.video_processor
                and ctx.video_processor.last_prediction
                and ctx.video_processor.last_confidence >= conf_threshold
            ):
                label = ctx.video_processor.last_prediction
                arabic_word = LABEL_TO_ARABIC.get(label, label.split("/")[-1])
                st.session_state.collected_words.append(arabic_word)
    with btn_col2:
        if st.button("Clear sentence", use_container_width=True):
            st.session_state.collected_words = []

    st.markdown('<div class="section-label" style="margin-top:1.2rem">Send video</div>', unsafe_allow_html=True)
    st.caption("Upload a recorded sign clip and send it as a video message.")

    uploaded_sign_video = st.file_uploader(
        "Video file",
        type=["mp4", "mov", "webm", "avi", "mkv", "gif"],
        key=f"sign_video_uploader_{st.session_state.video_uploader_nonce}",
        help="Supported formats: mp4, mov, webm, avi, mkv, gif",
    )

    video_note = st.text_input(
        "Optional note",
        value=" ".join(st.session_state.collected_words).strip(),
        placeholder="Example: جملة الإشارة لهذا الفيديو",
        key="sign_video_note",
    )

    if st.button("Send video", type="primary", use_container_width=True):
        if uploaded_sign_video is None:
            st.warning("Please choose a video file first.")
        else:
            video_bytes = uploaded_sign_video.getvalue()
            with st.spinner("Translating signs from uploaded video..."):
                translated = translate_uploaded_video(
                    video_bytes=video_bytes,
                    model=model,
                    id_to_label=id_to_label,
                    landmarker=landmarker,
                    conf_threshold=conf_threshold,
                    filename=uploaded_sign_video.name,
                )

            if not translated["text"]:
                st.warning("No confident signs were detected in this clip. Try better lighting or a closer hand view.")

            st.session_state.sent_sign_videos.append(
                {
                    "name": uploaded_sign_video.name,
                    "mime": uploaded_sign_video.type or "video/mp4",
                    "bytes": video_bytes,
                    "note": video_note,
                    "translated_text": translated["text"],
                    "translated_words": translated["words"],
                    "frame_count": translated["frames"],
                    "hand_frame_count": translated["hand_frames"],
                    "created_at": time.time(),
                }
            )
            st.session_state.video_uploader_nonce += 1
            st.success("Video message sent.")
            st.rerun()

    if st.session_state.sent_sign_videos:
        st.markdown('<div class="section-label" style="margin-top:1.2rem">Video timeline</div>', unsafe_allow_html=True)
        if st.button("Clear sent videos", use_container_width=True):
            st.session_state.sent_sign_videos = []
            st.rerun()

        latest_video = st.session_state.sent_sign_videos[-1]
        if latest_video.get("mime") == "image/gif":
            st.image(latest_video["bytes"])
        else:
            st.video(latest_video["bytes"])
        translated_text = latest_video.get("translated_text", "")
        if translated_text:
            st.markdown(f"**Translated text:** {translated_text}")
        else:
            st.markdown("**Translated text:** No confident translation found.")
        if latest_video["note"]:
            st.markdown(f"**Note:** {latest_video['note']}")
        st.caption(
            f"Latest: {latest_video['name']} | "
            f"frames: {latest_video.get('frame_count', 0)} | "
            f"hand frames: {latest_video.get('hand_frame_count', 0)}"
        )

    st.markdown('<div class="section-label" style="margin-top:1.2rem">Session stats</div>', unsafe_allow_html=True)
    stats_cols = st.columns(3)


if ctx.state.playing and ctx.video_processor:
    processor = ctx.video_processor

    while ctx.state.playing:
        time.sleep(0.15)
        prediction = processor.last_prediction
        confidence = processor.last_confidence
        is_fresh = (time.time() - processor.last_pred_time) < 15.0

        if prediction and is_fresh and confidence >= conf_threshold:
            arabic_word = LABEL_TO_ARABIC.get(prediction, prediction.split("/")[-1])
            latin_word = prediction.split("/")[-1]
            conf_pct = int(confidence * 100)
            bar_color = "#0fb16f" if confidence > 0.7 else "#d68024"

            pred_placeholder.markdown(
                f"""
                <div class="pred-card">
                    <div class="pred-arabic">{arabic_word}</div>
                    <div class="pred-latin">{latin_word}</div>
                    <div class="pred-conf">{conf_pct}% confidence</div>
                    <div style="margin-top:0.8rem;height:6px;background:#edf2f8;border-radius:3px;overflow:hidden;">
                        <div style="width:{conf_pct}%;height:100%;background:{bar_color};border-radius:3px;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if (
                collect_words
                and prediction != st.session_state.last_added
                and time.time() - st.session_state.last_add_time > 1.0
                and confidence >= conf_threshold
            ):
                st.session_state.collected_words.append(arabic_word)
                st.session_state.last_added = prediction
                st.session_state.last_add_time = time.time()

        elif not processor.hand_detected:
            pred_placeholder.markdown(
                """
                <div class="pred-card">
                    <div style="color:#6f7a86;font-size:1.5rem">No hand detected</div>
                    <div style="color:#8492a0;font-size:0.9rem;margin-top:0.4rem">Move your hand into the camera frame.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            pred_placeholder.markdown(
                """
                <div class="pred-card">
                    <div style="color:#647282;font-size:1rem">Signing in progress...</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        words = st.session_state.collected_words
        if words:
            sentence_html = " ".join(f'<span class="word-pill">{word}</span>' for word in words)
            sentence_placeholder.markdown(f'<div class="sentence-box">{sentence_html}</div>', unsafe_allow_html=True)
        else:
            sentence_placeholder.markdown(
                '<div class="sentence-box" style="color:#738497">الكلمات ستظهر هنا...</div>',
                unsafe_allow_html=True,
            )

        with stats_cols[0]:
            st.markdown(f'<div class="stat-chip"><b>{len(words)}</b>words</div>', unsafe_allow_html=True)
        with stats_cols[1]:
            st.markdown(f'<div class="stat-chip"><b>{processor.frame_count}</b>frames</div>', unsafe_allow_html=True)
        with stats_cols[2]:
            st.markdown(
                f'<div class="stat-chip"><b>{len(processor.landmark_buffer)}</b>landmarks</div>',
                unsafe_allow_html=True,
            )
else:
    pred_placeholder.markdown(
        """
        <div class="pred-card">
            <div style="color:#5e6c7b;font-size:1.1rem">Start the camera to begin sign recognition.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    sentence_placeholder.markdown(
        '<div class="sentence-box" style="color:#738497">الكلمات ستظهر هنا...</div>',
        unsafe_allow_html=True,
    )


st.markdown("---")
st.caption("TuniSign AI Platform | Integrated Sign to Text module from tts/signtotext")

if st.session_state.sent_sign_videos:
    st.markdown("### Sent Video Messages")
    for idx, video_item in enumerate(reversed(st.session_state.sent_sign_videos), start=1):
        st.markdown(f"Video #{idx}: {video_item['name']}")
        if video_item.get("translated_text"):
            st.markdown(f"**Translated text:** {video_item['translated_text']}")
        else:
            st.caption("Translated text: No confident translation found.")
        if video_item["note"]:
            st.caption(video_item["note"])
        if video_item.get("mime") == "image/gif":
            st.image(video_item["bytes"])
        else:
            st.video(video_item["bytes"])
