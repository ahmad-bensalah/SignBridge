# 🤟 Tunisian Text to Sign (TTSi) - Neon Avatar

The TTSi module transforms written Tunisian text into a clear, animated Sign Language representation using a custom-built Skeleton Avatar.

## 🏗️ Technical Architecture

### 1. Data-Driven Concatenative Synthesis
Unlike standard 3D avatars that require complex rigging and skeletal animation keys, our approach is **data-driven**:
- **Source Material**: We curated a dataset where native Tunisian signers perform signs.
- **Holistic Extraction**: We used **Mediapipe Holistic** to process every frame of the dataset, extracting 468 Face landmarks, 33 Pose landmarks (upper body), and 21 landmarks per Hand.
- **Keyword Lookup**: When a user enters text, the engine tokenizes the input and looks up corresponding landmark sequences from the `DATA_ROOT`.

### 2. The "Neon Skeleton" Rendering Engine
To provide a focused and high-contrast view (ideal for those with low vision or for clarity in fast movement), we developed a custom **HTML5 Canvas Renderer**.

- **Coordinate Mapping**: 3D normalized $(x, y)$ coordinates from Mediapipe are mapped to a $400 \times 560$ canvas.
- **Visual Distinction**:
  - **Pose (Body)**: Rendered in **Indigo (#7C6AF7)** with subtle glow.
  - **Right Hand**: Rendered in **Pink (#F7A6C1)** with intensive neon glow.
  - **Left Hand**: Rendered in **Teal (#6AF7D4)** for clear distinction during overlapping hand movements.
- **Stylized Head**: A gradient-filled radial circle tracks the pose of the signer to provide a human-like reference.

### 3. Integrated Playback System
- **Dynamic FPS**: A slider allows users to control the speed of the animation ($1 \text{--} 15$ FPS), making it a great tool for learning signs.
- **Multi-Word Support**: The system can chain landmark sequences of multiple words (e.g., "3aslema" + "lebes") into a continuous animation.

---

## 🎨 Design Philosophy: "Clarity over Realism"
Standard 3D avatars often suffer from the "Uncanny Valley" and can distract from the actual sign hand-shapes. Our **Neon Skeleton** approach:
1. **Eliminates Noise**: Removes clothes, face details, and background clutter.
2. **Emphasizes Movement**: Highlights the trajectory of hands and depth (via Z-coordinate mapping).
3. **Optimized for Web**: The landmark data is lightweight (JSON) compared to raw video, enabling fast loading on mobile devices.

---

## 🛠️ How to Run

> [!NOTE]
> Unlike the other modules, **TTSi does not require running a training notebook**. This module is a **Streamlit-based interface** that uses pre-extracted landmark data directly.

Simply launch the Streamlit demo and navigate to the **Avatar Synthesis** page:
```bash
cd Demo
streamlit run app.py
```
The engine will automatically load the landmark dataset from `TextToSign/data/` and make all available signs discoverable via the sidebar.

---

## 📂 Dataset Structure
```text
TextToSign/data/
├── Demandes/
│   ├── 3aslema/
│   │   ├── frame_001.jpg (or landmarks.json)
│   │   └── ...
```
The engine automatically scans categories and makes words available for discovery in the Streamlit Sidebar.
