# 🎧 Tunisian Speech to Text (Tun_STT)

This module provides offline-first, low-latency transcription for the Tunisian Dialect.

## ⚙️ Technical Core: Kaldi & Vosk

### 1. The Engine
We use **Vosk**, a speech recognition toolkit that allows running the **Kaldi** speech recognition engine in a lightweight, cross-platform package.
- **Offline Inference**: Unlike cloud APIs (Google Cloud STT, OpenAI Whisper), Vosk runs entirely on the local device, ensuring user privacy and zero bandwidth cost once the model is downloaded.
- **Kaldi Compatibility**: Uses HMM-GMM and DNN-based acoustic models which are highly efficient for real-time applications.

### 2. Tunisian Acoustic Modeling
The model is optimized to recognize the phonetic characteristics of Tunisian Arabic:
- **Phoneme Handling**: Specifically tuned for the "Qaf" / "Gaf" distinction and the specific vowel sounds present in Darija.
- **Noise Robustness**: Incorporates Kaldi's signal processing layers to handle background noise in typical mobile environments (street noise, cafe chatter).

### 3. Real-time Implementation
The integration in the platform uses a **streaming recognizer** approach:
- **Sample Rate**: Optimized for 16kHz or 44.1kHz mono PCM audio.
- **Chunk-based processing**: Audio frames are processed in 4000-byte chunks to provide "Partial Results" as the user speaks.
- **Final Result**: Once silence is detected, the engine combines partial hypotheses into a final polished transcript.

---

## 🛠️ Developer Setup

### Model Placement
For the module to work, a Vosk model directory must be present:
```bash
# Example structure
TuniSign-AI/Demo/model/vosk-model/
├── am/ (acoustic model)
├── graph/ (language model graph)
└── conf/
```

### Script Usage
```python
from vosk import Model, KaldiRecognizer
import wave

model = Model("model/vosk-model")
wf = wave.open("audio.wav", "rb")
rec = KaldiRecognizer(model, wf.getframerate())

while True:
    data = wf.readframes(4000)
    if len(data) == 0:
        break
    if rec.AcceptWaveform(data):
        print(rec.Result())
```

---

## 📈 Characteristics
- **Latency**: < 200ms on standard CPUs.
- **Binary Size**: Small footprint (~50MB to 300MB depending on the model chosen).
- **Environment**: Works on Linux, Windows, macOS, Android, and iOS.
