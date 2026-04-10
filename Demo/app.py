import streamlit as st
from elevenlabs.client import ElevenLabs
from ui_theme import apply_tunisign_theme, render_primary_sidebar

# Keep key in app logic (not shown in UI).
API_KEY = "sk_dc270b6a089f34d50834bf3e207addaee1082463f7bd4a84"
VOICE_ID = "InB4o9iYj3MQ4HtGs2KV"

DEFAULT_TEXT = "أهلاً وسهلاً! كيفاش حالك اليوم؟ إن شاء الله لباس."


st.set_page_config(
    page_title="TuniSign AI Platform",
    page_icon="🧩",
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
        color: #1f2a35;
    }

    .workspace-shell {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
    }

    .mini-note {
        color: #4b5563;
        font-size: 0.9rem;
    }

    .stButton > button {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

apply_tunisign_theme()


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
    detail = None
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


if "tts_text" not in st.session_state:
    st.session_state["tts_text"] = DEFAULT_TEXT


with st.sidebar:
    render_primary_sidebar()

    st.markdown("---")
    st.subheader("TTS Engine")
    model_options = {
        "Multilingual v2 (Best quality)": "eleven_multilingual_v2",
        "Flash v2.5 (Fastest)": "eleven_flash_v2_5",
        "Turbo v2.5 (Balanced)": "eleven_turbo_v2_5",
    }
    model_choice = st.selectbox("Model", list(model_options.keys()))
    model_id = model_options[model_choice]

    st.markdown("---")
    st.caption("Credentials and voice are configured in the app backend.")
    st.caption(f"Voice ID: {VOICE_ID}")


st.title("TuniSign AI Platform")
st.caption("Minimal communication workspace for text, voice, and sign." )

metric_cols = st.columns(3)
metric_cols[0].metric("Live modules", "2")
metric_cols[1].metric("Planned modules", "2")
metric_cols[2].metric("Locale", "Tunisia")

st.markdown("### Platform Modules")
st.write("- Text to Speech")
st.write("- Speech to Text")
st.write("- Sign to Text")
st.write("- Text to Sign")

st.markdown("### Text to Speech Studio")
left_col, right_col = st.columns([1.45, 1], gap="large")

samples = {
    "Greeting": "أهلاً وسهلاً! كيفاش حالك اليوم؟ إنشاء الله بخير.",
    "Casual": "واش راك؟ الحمد لله، أنا بصح. شنوة أخبارك؟",
    "Arabizi": "Ahla! Chneya akhbarak? Inchallah bekhir. Wesh taamel?",
    "Food": "الكسكسي بلحم هو أكلتي المفضلة. نحب نوكلو كل جمعة.",
    "Proverb": "اللي ما عندوش كبير، يشري كبير.",
}

with left_col:
    st.markdown('<div class="workspace-shell">', unsafe_allow_html=True)
    st.subheader("Compose")
    sample_btn_cols = st.columns(5)
    for col, (label, phrase) in zip(sample_btn_cols, samples.items()):
        if col.button(label, use_container_width=True):
            st.session_state["tts_text"] = phrase

    text_input = st.text_area(
        "Input Text",
        key="tts_text",
        height=220,
        placeholder="اكتب هنا بالدارجة، بالعربية، أو بالأرابيزي...",
    )
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="workspace-shell">', unsafe_allow_html=True)
    st.subheader("Generate")
    char_count = len(text_input)
    st.markdown(f"<p class='mini-note'>Characters: <b>{char_count}</b> / 5000</p>", unsafe_allow_html=True)
    st.markdown(
        "<p class='mini-note'>Professional tip: keep sentences short and punctuated for more natural Tunisian prosody.</p>",
        unsafe_allow_html=True,
    )

    generate = st.button("Generate Speech", type="primary", use_container_width=True)

    if generate:
        if not API_KEY:
            st.error("API key is missing in app.py. Set API_KEY before generating audio.")
        elif not VOICE_ID:
            st.error("Voice ID is missing in app.py. Set VOICE_ID before generating audio.")
        elif not text_input.strip():
            st.error("Please enter text before generating speech.")
        elif char_count > 5000:
            st.error("Text exceeds the 5000 character limit.")
        else:
            with st.spinner("Generating studio-quality speech..."):
                try:
                    client = ElevenLabs(api_key=API_KEY)
                    audio_stream = client.text_to_speech.convert(
                        text=text_input,
                        voice_id=VOICE_ID,
                        model_id=model_id,
                        output_format="mp3_44100_128",
                    )

                    audio_bytes = audio_to_bytes(audio_stream)
                    st.success("Audio generated successfully.")
                    st.audio(audio_bytes, format="audio/mp3")
                    st.download_button(
                        label="Download MP3",
                        data=audio_bytes,
                        file_name="tunisign_tts.mp3",
                        mime="audio/mpeg",
                        use_container_width=True,
                    )

                    with st.expander("Generation Details"):
                        st.json(
                            {
                                "model": model_id,
                                "voice_id": VOICE_ID,
                                "characters": char_count,
                                "output_format": "mp3_44100_128",
                            }
                        )

                except Exception as error:
                    status_code, error_code, message = parse_elevenlabs_error(error)

                    if status_code == 401:
                        st.error("Invalid API key. Update API_KEY in app.py.")
                    elif status_code == 402 and error_code == "paid_plan_required":
                        st.error("This voice is not available on the current ElevenLabs plan.")
                        st.info("Use a voice from My Voices in your ElevenLabs account or upgrade the plan.")
                    elif status_code == 429:
                        st.error("Rate limit reached. Please try again shortly.")
                    else:
                        st.error(f"ElevenLabs error ({status_code}): {message}")
    st.markdown('</div>', unsafe_allow_html=True)


st.markdown("---")
st.caption("TuniSign AI Platform | Designed for accessible communication workflows in Tunisia")