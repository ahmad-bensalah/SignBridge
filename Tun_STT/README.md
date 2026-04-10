# 🎧 Tunisian Speech to Text (Tun_STT)

This module provides high-accuracy transcription for the Tunisian Dialect using OpenAI's Whisper model, with an integrated **Tunisian bad-word filter** for safe output.

## ⚙️ Technical Core: OpenAI Whisper

### 1. The Engine
We use **Whisper-small**, OpenAI's open-source automatic speech recognition model, which provides robust multilingual transcription capabilities.
- **Multilingual Support**: Whisper is trained on 680,000 hours of multilingual audio, giving it strong out-of-the-box performance on Arabic dialects including Tunisian Darija.
- **Transformer Architecture**: Uses an encoder-decoder Transformer that jointly performs speech recognition and language identification, making it naturally suited for code-switched Tunisian speech (Arabic + French loanwords).

### 2. Tunisian Acoustic Modeling
The model handles the phonetic characteristics of Tunisian Arabic effectively:
- **Phoneme Handling**: Whisper's large-scale multilingual pretraining allows it to generalize well to the "Qaf" / "Gaf" distinction and the specific vowel sounds present in Darija.
- **Noise Robustness**: The model's training on diverse audio conditions provides inherent robustness to background noise in typical mobile environments (street noise, cafe chatter).

### 3. Implementation
The integration in the platform processes audio through Whisper's pipeline:
- **Sample Rate**: Whisper internally resamples all audio to 16kHz mono.
- **Log-Mel Spectrogram**: Audio is converted into 80-channel log-Mel spectrograms before being fed to the encoder.
- **Beam Search Decoding**: The decoder uses beam search to produce the final transcript with high accuracy.

---

## 🚫 Tunisian Bad-Word Filter

After transcription, the output passes through a **profanity detection filter** specifically built for Tunisian Darija.

- **Dataset**: The filter uses a curated `.xlsx` dataset containing known Tunisian profane and offensive words/expressions.
- **Matching**: Each transcribed word is checked against the dataset entries. Matches are flagged or censored before displaying the final text to the user.
- **Why a dedicated filter?**: Standard Arabic profanity lists do not cover Tunisian slang and dialect-specific insults. This custom dataset ensures culturally accurate content moderation.

---

## 🛠️ How to Run

> [!IMPORTANT]
> **The notebook is designed to run exclusively on Google Colab** and will only open in the Colab environment. The training/inference pipeline requires at least **10 GB of VRAM** (GPU memory), and the necessary libraries (PyTorch, Whisper, etc.) come pre-installed in Colab.

1. Open the notebook located in this directory directly on **Google Colab**.
2. Run all cells to load the Whisper-small model, perform transcription, and apply the bad-word filter.
3. The resulting model/outputs can then be used in the Streamlit demo.

---

## 📈 Characteristics
- **Model Size**: ~461MB (Whisper-small, 244M parameters).
- **Accuracy**: Strong performance on Arabic dialects thanks to multilingual pretraining.
- **Content Safety**: Integrated Tunisian bad-word filter using a curated `.xlsx` dataset.
- **Environment**: Works on Linux, Windows, and macOS. GPU acceleration supported via CUDA.
