# 💻 Streamlit Demo Suite

This directory contains the multi-module **Streamlit dashboard** that serves as the unified front-end for all SignBridge AI modules.

---

## 🚀 Quick Start

```bash
cd Demo
streamlit run app.py
```

---

## 📑 Pages

| # | Page | File | Description |
|---|------|------|-------------|
| 1 | **TTS Studio** | `app.py` | Advanced text-to-speech generation with sample Tunisian phrases. |
| 2 | **STT Workspace** | `pages/2_STT_Whisper.py` | Upload or record audio for instant transcription using Whisper-TTS. |
| 3 | **Sign Recognition** | `pages/3_Sign_to_Text.py` | Live webcam feed for real-time Tunisian sign detection. |
| 4 | **Avatar Synthesis** | `pages/4_Text_to_Sign.py` | Enter text (e.g., "3aslema") to see the neon skeleton avatar perform the sign. |
| 5 | **Conversation Bridge** | `pages/5_Conversation_Bridge.py` | A unified interface for bidirectional deaf ↔ hearing interaction. |

---

## 📂 Project Structure

```text
Demo/
├── app.py                      # Main entry point (TTS Studio)
├── ui_theme.py                 # Shared UI theming & navigation
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container deployment
├── skeleton_sign_renderer.py   # Neon skeleton avatar renderer
├── text_to_sign_engine.py      # Text → landmark lookup engine
├── pages/
│   ├── 2_STT_Whisper.py          # Speech-to-Text page
│   ├── 3_Sign_to_Text.py       # Sign Recognition page
│   ├── 4_Text_to_Sign.py       # Avatar Synthesis page
│   └── 5_Conversation_Bridge.py# Full conversation bridge
├── signtotext/                 # Sign-to-Text model assets
└── TextToSign/                 # Text-to-Sign landmark data
```

---

## 📦 Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

> [!NOTE]
> The Whisper-TTS model (~542 MB) is **auto-downloaded** from [HuggingFace](https://huggingface.co/Sali7a8603/Tunisian_STT) on first use. Make sure the other trained models from the **TTS** and **SiTT** notebooks have been exported and placed in the expected paths before launching the demo. See the [main README](../README.md) for full setup instructions.
