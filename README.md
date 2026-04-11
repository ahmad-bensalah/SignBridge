# 🇹🇳 SignBridge: Bridging the Gap in Tunisian Communication

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Mediapipe](https://img.shields.io/badge/Mediapipe-00C853?style=flat&logo=google&logoColor=white)](https://google.github.io/mediapipe/)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=Keras&logoColor=white)](https://keras.io/)

**SignBridge** is a comprehensive accessibility platform designed to break communication barriers between the deaf and hearing communities in Tunisia. By leveraging state-of-the-art Deep Learning models, the platform provides seamless bidirectional translation between **Tunisian Dialect (Darija)** and **Tunisian Sign Language (TSL)**.

---

## 🚀 Project Vision
In Tunisia, the deaf community often faces significant hurdles due to the lack of specialized tools that understand the nuances of the local dialect and native sign language. TuniSign AI aims to provide a "Conversation Bridge" that is:
- **Culturally Aware**: Trained on Tunisian speech and sign datasets.
- **Real-time**: Capable of live sign recognition and instant speech synthesis.
- **Hyper-Visual**: Utilizing neon-skeleton avatars for better clarity and engagement.

---

## 🏗️ Technical Architecture

```mermaid
graph TD
    UserA[Deaf User] -- "Signs (TSL)" --> S2T[Sign-to-Text Model]
    S2T -- "Tunisian Text" --> TTS[Text-to-Speech Model]
    TTS -- "Audio (Darija)" --> UserB[Hearing User]
    
    UserB -- "Speech (Darija)" --> STT[Speech-to-Text Model]
    STT -- "Tunisian Text" --> T2S[Text-to-Sign Engine]
    T2S -- "Animated Avatar" --> UserA
    
    subgraph "AI Core"
    S2T
    TTS
    STT
    T2S
    end
```

---

## 🧠 Core AI Modules

### 1. 🖐️ Sign to Text (SiTT)
- **Technology**: Mediapipe Hand Landmarking + Keras Temporal Attention Model.
- **Process**: Extracts 21 3D points per hand, normalizes coordinates relative to the wrist, and calculates inter-finger angles. A window-based model with an integrated attention mechanism classifies sequences of 16 frames into Tunisian signs.
- **[Deep Dive Into SiTT](./Tun_SiTT/)**

### 2. 🤟 Text to Sign (TTSi)
- **Technology**: Concatenative Synthesis + Mediapipe Holistic.
- **Process**: Looks up sign videos/frames from a curated TSL dataset. Extracts holistic landmarks (Pose + Hands) and renders them as a high-fidelity Neon Skeleton Avatar using an HTML5 Canvas engine.
- **[Deep Dive Into TTSi](./Tun_TTSi/)**

### 3. 🗣️ Text to Speech (Tun_TTS)
- **Technology**: Coqui XTTS v2 Fine-tuning.
- **Process**: Fine-tuned on the **TunArTTS** corpus (3 hours of diacritized Tunisian speech). It features a custom Tunisian normalizer that phonetically maps French loanwords to Arabic script for natural prosody.
- **[Deep Dive Into Tun_TTS](./Tun_TTS/)**

### 4. 🎧 Speech to Text (Tun_STT)
- **Technology**: OpenAI Whisper-small + Tunisian Bad-Word Filter.
- **Process**: Uses the Whisper-small model (244M parameters) for high-accuracy transcription of Tunisian Darija, leveraging its multilingual Transformer architecture trained on 680k hours of diverse audio. The transcribed output is then passed through a **profanity filter** built from a curated Tunisian bad-word dataset (`.xlsx`) for safe content delivery.
- **[Deep Dive Into Tun_STT](./Tun_STT/)**

---

## 💻 Streamlit Demo Suite

The project includes a multi-module Streamlit dashboard located in the `/Demo` directory:

1. **TTS Studio**: Advanced text-to-speech generation with sample Tunisian phrases.
2. **STT Workspace**: Upload or record audio for instant transcription.
3. **Sign Recognition**: Live webcam feed for real-time sign detection.
4. **Avatar Synthesis**: enter text (e.g., "3aslema") to see the neon avatar perform the sign.
5. **Conversation Bridge**: A unified interface for bidirectional interaction.

---

## 🛠️ Installation & Setup

> [!IMPORTANT]
> **Model Training Notebooks must be run on Google Colab.** The training pipelines for **STT**, **TTS**, and **SiTT** require at least **10 GB of VRAM** (GPU memory), and the necessary deep learning libraries come pre-installed in the Colab environment. The **TTSi** module does not require training — it runs directly as a Streamlit interface.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/TuniSign-AI.git
   cd TuniSign-AI
   ```

2. **Train / export models** (on Google Colab):
   - Open each module's notebook (`Tun_STT/`, `Tun_TTS/`, `Tun_SiTT/`) in Colab and run all cells.
   - Download the resulting models and place them in the expected paths.

3. **Verify module-specific assets**:
   Ensure you have the following assets placed correctly:
   - `Demo/signtotext/tunisl_v4.keras` (Sign model — from SiTT notebook)
   - Whisper-small model (auto-downloaded on first run via `whisper.load_model("small")`)
   - `Demo/TextToSign/data/` (Avatar frame dataset — used directly by TTSi)

4. **Launch the platform**:
   ```bash
   cd Demo
   streamlit run app.py
   ```

---

## 🚢 Deployment

SignBridge uses a **hybrid deployment** strategy that balances latency, cost, and scalability by splitting workloads between the mobile device and the cloud.

```mermaid
graph LR
    subgraph "📱 On-Device (Local)"
        TTSi["Text-to-Sign\n(~4 MB)"]
        SiTT["Sign-to-Text\n(~6 MB)"]
    end

    subgraph "☁️ Azure Cloud"
        Nginx["Nginx\nLoad Balancer"]
        STT1["STT Container\n(ACI)"]
        STT2["STT Container\n(ACI)"]
        TTS1["TTS Container\n(ACI)"]
        TTS2["TTS Container\n(ACI)"]
    end

    App["Mobile App"] -->|Sign frames| SiTT
    App -->|Text input| TTSi
    App -->|Audio stream| Nginx
    App -->|Text to speak| Nginx
    Nginx --> STT1
    Nginx --> STT2
    Nginx --> TTS1
    Nginx --> TTS2
```

### Local Modules (On-Device)

| Module | Size | Why Local? |
|--------|------|------------|
| **Text-to-Sign (TTSi)** | ~4 MB | Concatenative lookup from a pre-built landmark dataset — no GPU needed. |
| **Sign-to-Text (SiTT)** | ~6 MB | Lightweight Keras TDNN + Mediapipe hand-landmarker runs comfortably on mobile CPUs. |

> **Total on-device footprint ≈ 10 MB** — small enough to ship inside the APK, eliminating network round-trips and enabling offline sign recognition.

### Cloud Modules (Azure Container Instances)

| Module | Image | Why Cloud? |
|--------|-------|------------|
| **Speech-to-Text (STT)** | `tunisign-stt:latest` | Whisper model (~500 MB) requires significant memory; benefits from GPU-backed ACI. |
| **Text-to-Speech (TTS)** | `tunisign-tts:latest` | XTTS v2 inference is compute-intensive; horizontally scaled behind the load balancer. |

Each module is containerised with its own `Dockerfile` and exposed as a **FastAPI** REST endpoint. **Nginx** sits in front as a reverse-proxy / load balancer, distributing requests across multiple ACI replicas to ensure:

- **Horizontal scalability** — spin up additional replicas during peak demand.
- **High availability** — health-checked containers are automatically replaced on failure.
- **Low latency** — geographically co-located with the target user base (Azure West Europe / North Africa).

---

## 🤝 Acknowledgments
- **TunArTTS Corpus**: For providing the foundation for Tunisian speech tasks.
- **Lingora**: For the Tunisian Transcripted Audios dataset.
- **ATILS (Association tunisienne des interprètes en langue des signes)**: For the Tunisian Sign Language dataset.

---
<p align="center">
  <i>Designed with ❤️ for the Tunisian Deaf Community.</i>
</p>
