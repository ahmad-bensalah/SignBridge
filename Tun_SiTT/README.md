# 🖐️ Tunisian Sign to Text (SiTT)

This module implements real-time recognition of Tunisian Sign Language (TSL) using computer vision and deep temporal modeling.

## 🧠 Technical Approach

### 1. Feature Extraction (Mediapipe)
Instead of processing raw pixels, which is computationally expensive and sensitive to lighting, we use **Mediapipe Hand Landmarker**. 
- **Landmarks**: 21 3D coordinates $(x, y, z)$ per hand.
- **Normalization**:
  - **Centering**: All coordinates are shifted relative to the wrist (landmark 0).
  - **Scaling**: The hand is scaled based on the maximum distance from the wrist, ensuring invariance to camera distance.
- **Geometric Descriptors**: We calculate the **cosine similarity** between vectors formed by the wrist and each finger tip. This provides a robust representation of hand "shape" independent of global orientation.

### 2. Temporal Modeling
Sign language is inherently temporal. A single static frame is often ambiguous.
- **Windowing**: We process sequences of **16 frames** (approx. 0.5 - 1 second of movement).
- **Buffer**: A sliding window buffer stores the last $N$ normalized landmark sets.

### 3. Model Architecture: Temporal Attention
The core classification model uses a **Temporal Attention Mechanism** implemented in Keras.

- **Rationale**: Not all frames in a 16-frame window are equally important. Some frames capture the "core" of the sign (the target pose), while others are transitional. The attention layer allows the model to "focus" on the most discriminative frames.
- **Backend**: The model is trained on a custom TSL dataset containing 100+ common Tunisian signs across categories like Family, Transport, and Destinations.

---

## 🛠️ How to Run

> [!IMPORTANT]
> **Run the notebook on Google Colab.** The training pipeline requires at least **10 GB of VRAM** (GPU memory), and the necessary libraries (TensorFlow/Keras, Mediapipe, etc.) come pre-installed in the Colab environment.

1. Open the notebook located in this directory on **Google Colab**.
2. Run all cells to train the Temporal Attention model on the TSL dataset and export the `.keras` model.
3. The resulting model is then used in the Streamlit demo (`3_Sign_to_Text.py`) for live sign recognition.

### Platform Integration
- **Live Stream**: Uses `streamlit-webrtc` for zero-latency camera capture.
- **Inference**: Occurs every 8 frames to maintain high FPS while providing smooth predictions.
- **Confidence Filtering**: A configurable threshold (default 0.55) prevents flickering and false positives.

---

## 📊 Dataset Categories
| Category | Examples |
| :--- | :--- |
| **Demandes** | عسلامة (Hello), لاباس (Fine), شكرا (Thank you) |
| **Destinations** | بوسطة (Post Office), سبيطار (Hospital), بلادية (Municipality) |
| **Famille** | أم (Mother), بو (Father), أخت (Sister) |
| **Transport** | تاكسي (Taxi), كرهبة (Car), تران (Train) |

---
