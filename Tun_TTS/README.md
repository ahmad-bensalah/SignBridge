# 🗣️ Tunisian Text to Speech (Tun_TTS)

This module provides high-quality voice synthesis for Tunisian Darija, capable of handling local prosody, expressions, and frequent code-switching with French.

## 🔬 Scientific Approach

### 1. Base Model: XTTS v2
We utilize **Coqui XTTS v2**, a transformer-based text-to-speech model.
- **Why XTTS?**: It is an "auto-regressive" model that uses a GPT-like architecture for text-to-audio latent mapping. Crucially, it supports multi-lingual transfer learning, allowing us to leverage pre-training on Modern Standard Arabic (MSA) for Tunisian Dialect.
- **GPT Encoder**: The model predicts audio "tokens" based on text input, which are then decoded into high-fidelity waveforms using a HiFi-GAN vocoder.

### 2. Fine-Tuning on TunArTTS
The model was fine-tuned on the **TunArTTS Corpus**:
- **Data**: ~3 hours of high-quality Tunisian speech from multiple speakers.
- **Diacritization**: The dataset uses diacritized Arabic script (tashkeel), which is vital for disambiguating Darija words that share the same root but different vowels.
- **Training**: fine-tuned for 2000+ steps on a Tesla T4 GPU, optimizing the GPT weights while keeping the vocoder frozen to preserve audio clarity.

### 3. Tunisian Text Normalization
One of the core challenges in Tunisian TTS is **Standardization**. People write Darija in many ways (e.g., "شنوة", "شنية"). 
Our custom pre-processing pipeline:
- **Spelling Standardization**: Maps variants like "مانيش" -> "ما نيش" for consistency.
- **Phonetic Mapping**: French loanwords (e.g., *merci*, *climat*, *pizza*) are automatically mapped to their Arabic phonetic equivalents to ensure the model pronounces them with a natural Tunisian accent.
- **Number Expansion**: Converts digits into their Tunisian spoken form (e.g., "2" -> "زوز").

```python
# Example Normalization Mapping
{
    "merci": "مرسي",
    "bonne journée": "بون جورني",
    "2": "زوز",
    "كيفاه": "كيفاش"
}
```

---

## 🛠️ Deployment

- **Demo Integration**: The `app.py` script demonstrates the module's capabilities.
- **Real-time Performance**: The model supports streaming synthesis, where audio chunks are played as they are generated, minimizing Time-to-First-Audio (TTFA).

---

## 📈 Performance & Quality
The fine-tuned model significantly outperforms vanilla MSA models in:
1. **Prosody**: Captures the distinct "vibe" and rhythm of Tunisian speech.
2. **Vocabulary**: Correctly pronounces dialect-specific particles (e.g., "باهي", "برشا").
3. **Cloning**: Capable of zero-shot voice cloning given a 6-second reference clip.
