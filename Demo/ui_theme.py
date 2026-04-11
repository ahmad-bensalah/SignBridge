import streamlit as st


PAGE_LINKS = [
    ("app.py", "Text to Speech", "🔊"),
    ("pages/2_STT_Whisper.py", "Speech to Text", "🎧"),
    ("pages/3_Sign_to_Text.py", "Sign to Text", "🖐️"),
    ("pages/4_Text_to_Sign.py", "Text to Sign", "🤟"),
    ("pages/5_Conversation_Bridge.py", "Conversation Bridge", "💬"),
]


def apply_tunisign_theme():
    st.markdown(
        """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Sora:wght@300;600;800&family=IBM+Plex+Sans+Arabic:wght@400;600;700&display=swap');

    :root {
        --bg-main: #0a0a0f;
        --bg-surface: #111028;
        --bg-soft: #171532;
        --border: #2e2b4a;
        --text-main: #ebe9f7;
        --text-muted: #9894bf;
        --accent-a: #7c6af7;
        --accent-b: #6af7d4;
    }

    html, body, [class*="css"] {
        font-family: 'Sora', sans-serif;
        color: var(--text-main);
    }

    .stApp {
        background: radial-gradient(1200px 500px at 10% -10%, #1b1940 0%, var(--bg-main) 42%), var(--bg-main) !important;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #13112a 0%, #0d0d16 100%) !important;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] * {
        color: var(--text-main);
    }

    h1 {
        font-family: 'Space Mono', monospace;
        font-size: 2.35rem;
        letter-spacing: -1px;
        background: linear-gradient(135deg, var(--accent-a), #f7a6c1, var(--accent-b));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    h2, h3 {
        color: var(--text-main);
    }

    p, li, label, .stCaption, .stMarkdown {
        color: var(--text-main);
    }

    .workspace-shell,
    .shell,
    .chat-shell,
    .composer-shell,
    .pred-card,
    .sentence-box,
    .sentence,
    .empty-thread,
    .mode-note,
    .stat-chip {
        background: var(--bg-surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        box-shadow: none !important;
        color: var(--text-main) !important;
    }

    .pred-latin,
    .pred-conf,
    .subtle,
    .mini-note,
    .chat-compose-note,
    .chat-heading,
    .section-label,
    .section-title,
    .msg-meta {
        color: var(--text-muted) !important;
    }

    .pred-arabic,
    .sentence,
    .sentence-box,
    .word-pill {
        color: #f2efff !important;
    }

    .word-pill {
        background: #1a1830 !important;
        border: 1px solid var(--border) !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, var(--accent-a), var(--accent-b)) !important;
        color: #090912 !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'Space Mono', monospace !important;
        font-weight: 700 !important;
    }

    .stButton > button:hover {
        opacity: 0.9 !important;
    }

    .stTextInput > div > div > input,
    .stTextArea textarea,
    .stSelectbox [data-baseweb="select"],
    .stMultiSelect [data-baseweb="select"],
    .stFileUploader,
    .stAudioInput {
        background: #111028 !important;
        color: var(--text-main) !important;
        border-color: var(--border) !important;
    }

    .stMetric {
        background: var(--bg-soft);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.5rem 0.7rem;
    }
</style>
""",
        unsafe_allow_html=True,
    )


def render_primary_sidebar(show_caption=True):
    st.markdown("### TuniSign AI")
    if show_caption:
        st.caption("Unified communication toolkit")

    for page_path, label, icon in PAGE_LINKS:
        st.page_link(page_path, label=label, icon=icon)
