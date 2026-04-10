"""
Tunisian Sign Language Avatar Viewer
- Supports both video (.mp4) and image sequences
- MediaPipe Holistic extracts landmarks from each frame
- A canvas avatar animates those landmarks as a neon skeleton
"""

import json
import re
import time
from pathlib import Path

import cv2
import mediapipe as mp
import streamlit as st
import streamlit.components.v1 as components
from ui_theme import apply_tunisign_theme, render_primary_sidebar

# Page config
st.set_page_config(
    page_title="TuniSign AI | Text to Sign",
    page_icon="🤟",
    layout="wide",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Sora:wght@300;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Sora', sans-serif; background: #0a0a0f; color: #e8e6f0; }
.stApp { background: #0a0a0f; }
h1 {
    font-family: 'Space Mono', monospace; font-size: 2.4rem; letter-spacing: -1px;
    background: linear-gradient(135deg, #7c6af7, #f7a6c1, #6af7d4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.2rem;
}
.subtitle { font-size: 0.95rem; color: #7a7890; font-weight: 300; margin-bottom: 2rem; font-family: 'Space Mono', monospace; }
.word-chip {
    display: inline-block; background: #1a1830; border: 1px solid #2e2b4a;
    border-radius: 20px; padding: 4px 14px; margin: 3px;
    font-size: 0.8rem; color: #9b96c8; font-family: 'Space Mono', monospace;
}
.stTextInput > div > div > input {
    background: #111028 !important; border: 1px solid #2e2b4a !important;
    border-radius: 10px !important; color: #e8e6f0 !important;
    font-family: 'Space Mono', monospace !important; font-size: 1rem !important; padding: 0.7rem 1rem !important;
}
.stTextInput > div > div > input:focus { border-color: #7c6af7 !important; box-shadow: 0 0 0 2px rgba(124,106,247,0.2) !important; }
.stButton > button {
    background: linear-gradient(135deg, #7c6af7, #6af7d4) !important; border: none !important;
    border-radius: 10px !important; color: #0a0a0f !important;
    font-family: 'Space Mono', monospace !important; font-weight: 700 !important; padding: 0.6rem 2rem !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
</style>
""",
    unsafe_allow_html=True,
)

apply_tunisign_theme()

with st.sidebar:
    render_primary_sidebar(show_caption=False)

# Constants
DATA_ROOT = Path(__file__).resolve().parents[1] / "TextToSign" / "data"

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
POSE_UPPER = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
]
HAND_CONNECTIONS = list(mp_hands.HAND_CONNECTIONS)


@st.cache_resource
def get_detectors():
    pose_detector = mp_pose.Pose(
        static_image_mode=True,
        min_detection_confidence=0.3,
        model_complexity=1,
    )
    hands_detector = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.3,
    )
    return pose_detector, hands_detector


def list_words():
    if not DATA_ROOT.exists():
        return {}
    categories = {}
    for cat in sorted(DATA_ROOT.iterdir()):
        if cat.is_dir():
            words = sorted([d.name for d in cat.iterdir() if d.is_dir()])
            if words:
                categories[cat.name] = words
    return categories


def get_word_folder(word: str):
    for cat in DATA_ROOT.iterdir():
        if cat.is_dir():
            candidate = cat / word
            if candidate.exists():
                return candidate
    return None


def get_video(word: str):
    folder = get_word_folder(word)
    if not folder:
        return None
    videos = list(folder.glob("*.mp4"))
    return videos[0] if videos else None


def get_frames(word: str):
    folder = get_word_folder(word)
    if not folder:
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [f for f in folder.iterdir() if f.suffix.lower() in exts]

    def natural_key(p):
        nums = re.findall(r"\((\d+)\)", p.name)
        return int(nums[0]) if nums else 0

    return sorted(files, key=natural_key)


def landmarks_to_dict(pose_results, hands_results):
    data = {}
    if pose_results and pose_results.pose_landmarks:
        data["pose"] = [
            [lm.x, lm.y, lm.z, lm.visibility] for lm in pose_results.pose_landmarks.landmark
        ]

    if hands_results and hands_results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
            side_label = "right"
            if hands_results.multi_handedness and i < len(hands_results.multi_handedness):
                handedness = hands_results.multi_handedness[i].classification[0].label.lower()
                side_label = "left" if handedness == "left" else "right"

            key = "left_hand" if side_label == "left" else "right_hand"
            data[key] = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]

    return data


@st.cache_data(show_spinner=False)
def process_word(word: str):
    pose_detector, hands_detector = get_detectors()
    all_lm = []
    video_path = get_video(word)

    if video_path:
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = set(int(i * (total - 1) / 9) for i in range(10))
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx in indices:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose_results = pose_detector.process(img_rgb)
                hands_results = hands_detector.process(img_rgb)
                all_lm.append(landmarks_to_dict(pose_results, hands_results))
            idx += 1
        cap.release()
    else:
        for f in get_frames(word):
            img = cv2.imread(str(f))
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pose_results = pose_detector.process(img_rgb)
            hands_results = hands_detector.process(img_rgb)
            all_lm.append(landmarks_to_dict(pose_results, hands_results))

    return all_lm


def build_canvas_html(lm_data: dict, frame_idx: int, total: int) -> str:
    pose = lm_data.get("pose", [])
    left_hand = lm_data.get("left_hand", [])
    right_hand = lm_data.get("right_hand", [])

    return f"""<!DOCTYPE html>
<html>
<head>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#0a0a0f; display:flex; flex-direction:column; align-items:center; padding:8px; }}
  canvas {{ border-radius:14px; box-shadow:0 0 30px rgba(124,106,247,0.25); background:#0d0d1a; }}
  .counter {{ color:#3a3860; font-size:10px; font-family:monospace; margin-top:6px; }}
</style>
</head>
<body>
<canvas id=\"c\" width=\"400\" height=\"560\"></canvas>
<div class=\"counter\">Frame {frame_idx} / {total}</div>
<script>
(function(){{
  const pose       = {json.dumps(pose)};
  const leftHand   = {json.dumps(left_hand)};
  const rightHand  = {json.dumps(right_hand)};
  const poseConns  = {json.dumps(POSE_UPPER)};
  const handConns  = {json.dumps([[a,b] for a,b in HAND_CONNECTIONS])};

  const canvas = document.getElementById('c');
  const ctx    = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;

  function toXY(lm){{ return [lm[0]*W, lm[1]*H]; }}

  ctx.strokeStyle='rgba(255,255,255,0.03)'; ctx.lineWidth=1;
  for(let x=0;x<W;x+=40){{ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke();}}
  for(let y=0;y<H;y+=40){{ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();}}

  function drawPose(lms){{
    if(!lms||!lms.length) return;
    ctx.strokeStyle='#7c6af7'; ctx.lineWidth=3; ctx.shadowColor='#7c6af7'; ctx.shadowBlur=10;
    poseConns.forEach(([a,b])=>{{
      if(a>=lms.length||b>=lms.length) return;
      const [x1,y1]=toXY(lms[a]),[x2,y2]=toXY(lms[b]);
      ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.stroke();
    }});
    ctx.shadowBlur=0; ctx.fillStyle='rgba(200,190,255,0.9)';
    poseConns.flat().filter((v,i,a)=>a.indexOf(v)===i).forEach(idx=>{{
      if(idx>=lms.length) return;
      const [x,y]=toXY(lms[idx]);
      ctx.beginPath(); ctx.arc(x,y,4,0,Math.PI*2); ctx.fill();
    }});
  }}

  function drawHand(lms, baseColor, glowColor){{
    if(!lms||lms.length<21) return;
    const pts=lms.map(lm=>toXY(lm));
    const fingerColors=['#f7a6c1','#ffcc80','#a8f0c6','#80d4ff','#d4a8ff'];
    const fingers=[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20]];
    const palmIdx=[0,1,5,9,13,17];
    ctx.beginPath();
    palmIdx.forEach((i,n)=>{{ const [x,y]=pts[i]; n===0?ctx.moveTo(x,y):ctx.lineTo(x,y); }});
    ctx.closePath();
    const pg=ctx.createRadialGradient(pts[9][0],pts[9][1],2,pts[9][0],pts[9][1],W*0.12);
    pg.addColorStop(0,baseColor+'cc'); pg.addColorStop(1,baseColor+'44');
    ctx.fillStyle=pg; ctx.shadowColor=glowColor; ctx.shadowBlur=15; ctx.fill();
    ctx.strokeStyle=glowColor; ctx.lineWidth=1.5; ctx.stroke(); ctx.shadowBlur=0;
    fingers.forEach((joints,fi)=>{{
      const col=fingerColors[fi];
      for(let j=0;j<joints.length-1;j++){{
        const [x1,y1]=pts[joints[j]],[x2,y2]=pts[joints[j+1]];
        const dx=x2-x1,dy=y2-y1,len=Math.sqrt(dx*dx+dy*dy);
        const nx=-dy/len,ny=dx/len;
        const r1=Math.max(3,W*0.018-j*W*0.003),r2=Math.max(2,r1-W*0.003);
        ctx.beginPath();
        ctx.moveTo(x1+nx*r1,y1+ny*r1); ctx.lineTo(x2+nx*r2,y2+ny*r2);
        ctx.arcTo(x2+nx*r2+dx*0.1,y2+ny*r2+dy*0.1,x2-nx*r2,y2-ny*r2,r2);
        ctx.lineTo(x2-nx*r2,y2-ny*r2); ctx.lineTo(x1-nx*r1,y1-ny*r1);
        ctx.arcTo(x1-nx*r1-dx*0.1,y1-ny*r1-dy*0.1,x1+nx*r1,y1+ny*r1,r1);
        ctx.closePath();
        const g=ctx.createLinearGradient(x1,y1,x2,y2);
        g.addColorStop(0,col+'dd'); g.addColorStop(1,col+'99');
        ctx.fillStyle=g; ctx.shadowColor=col; ctx.shadowBlur=8; ctx.fill();
        ctx.strokeStyle=col+'88'; ctx.lineWidth=1; ctx.stroke(); ctx.shadowBlur=0;
        ctx.beginPath(); ctx.arc(x1,y1,r1*0.6,0,Math.PI*2); ctx.fillStyle=col+'cc'; ctx.fill();
      }}
      const tip=pts[joints[joints.length-1]];
      ctx.beginPath(); ctx.arc(tip[0],tip[1],W*0.012,0,Math.PI*2);
      ctx.fillStyle=fingerColors[fi]; ctx.shadowColor=fingerColors[fi]; ctx.shadowBlur=10;
      ctx.fill(); ctx.shadowBlur=0;
    }});
  }}

  function drawHead(lms){{
    if(!lms||!lms.length) return;
    const [nx,ny]=toXY(lms[0]), r=H*0.055;
    const g=ctx.createRadialGradient(nx,ny-r*.2,r*.1,nx,ny,r);
    g.addColorStop(0,'rgba(200,190,255,0.9)'); g.addColorStop(1,'rgba(124,106,247,0.1)');
    ctx.beginPath(); ctx.arc(nx,ny,r,0,Math.PI*2);
    ctx.fillStyle=g; ctx.shadowColor='#7c6af7'; ctx.shadowBlur=20; ctx.fill(); ctx.shadowBlur=0;
  }}

  drawPose(pose);
  drawHand(leftHand,  '#6af7d4', '#00ffcc');
  drawHand(rightHand, '#f7a6c1', '#ff80aa');
  drawHead(pose);

  ctx.font='10px monospace';
  [['● body','rgba(255,255,255,0.3)',10],['● L.hand','#6af7d4',80],['● R.hand','#f7a6c1',160]]
    .forEach(([t,c,x])=>{{ ctx.fillStyle=c; ctx.fillText(t,x,H-10); }});
}})();
</script>
</body>
</html>"""


st.markdown("<h1>🤟 TSL Avatar</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">Tunisian Sign Language · Skeleton Playback</p>', unsafe_allow_html=True)

words_by_cat = list_words()
words = [w for ws in words_by_cat.values() for w in ws]

if words_by_cat:
    for cat, cat_words in words_by_cat.items():
        with st.expander(f"**{cat}** — {len(cat_words)} words"):
            chips = " ".join(f'<span class="word-chip">{w}</span>' for w in cat_words)
            st.markdown(chips, unsafe_allow_html=True)
else:
    st.warning("No dataset found at data/. Place your category folders there and re-run.")

st.markdown("---")

col_input, col_btn, col_speed = st.columns([3, 1, 2])
with col_input:
    word_input = st.text_input(
        "Enter a word",
        placeholder="e.g. 3aslema",
        label_visibility="collapsed",
    )
with col_btn:
    play = st.button("▶ Play")
with col_speed:
    fps = st.slider("Speed (fps)", 1, 15, 6)

display_slot = st.empty()
progress_bar = st.progress(0)
status_slot = st.empty()

if play and word_input.strip():
    words_input = word_input.strip().split()
    all_landmarks = []
    valid_words = []

    with st.spinner(f"Extracting landmarks for: {' '.join(words_input)} ..."):
        for w in words_input:
            if words_by_cat and w not in words:
                st.warning(f"Word '{w}' not found, skipping.")
                continue
            lms = process_word(w)
            if lms:
                all_landmarks.extend(lms)
                valid_words.append(w)

    if not all_landmarks:
        st.error("No valid words found in dataset.")
    else:
        n = len(all_landmarks)
        delay = 1.0 / fps

        for i, lm in enumerate(all_landmarks):
            html_blob = build_canvas_html(lm, i + 1, n)
            with display_slot:
                components.html(html_blob, height=620, scrolling=False)
            progress_bar.progress((i + 1) / n)
            time.sleep(delay)

        status_slot.success(f"Done - {n} frames for {' '.join(valid_words)}")

elif play and not word_input.strip():
    st.warning("Please enter a word first.")
