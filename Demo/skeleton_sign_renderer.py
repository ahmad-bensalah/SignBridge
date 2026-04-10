import json
import re
from functools import lru_cache
from pathlib import Path

import cv2
import mediapipe as mp

DATA_ROOT = Path(__file__).resolve().parent / "TextToSign" / "data"

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
HAND_CONNECTIONS = list(mp.solutions.hands.HAND_CONNECTIONS)


@lru_cache(maxsize=1)
def _get_detectors():
    pose_detector = mp.solutions.pose.Pose(
        static_image_mode=True,
        min_detection_confidence=0.3,
        model_complexity=1,
    )
    hands_detector = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.3,
    )
    return pose_detector, hands_detector


def _landmarks_to_dict(pose_results, hands_results):
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


def _natural_key(path_obj: Path):
    nums = re.findall(r"\((\d+)\)", path_obj.name)
    return int(nums[0]) if nums else 0


@lru_cache(maxsize=1)
def _dataset_index():
    index = {}
    if not DATA_ROOT.exists():
        return index

    for category_dir in sorted(DATA_ROOT.iterdir()):
        if not category_dir.is_dir():
            continue
        for word_dir in sorted(category_dir.iterdir()):
            if word_dir.is_dir():
                index[word_dir.name.lower()] = word_dir

    return index


def _sample_indices(total: int, max_frames: int):
    if total <= 0:
        return []
    sample_count = min(total, max_frames)
    if sample_count <= 1:
        return [0]
    return sorted({int(i * (total - 1) / (sample_count - 1)) for i in range(sample_count)})


@lru_cache(maxsize=256)
def landmarks_for_token(token: str, max_frames: int = 12):
    folder = _dataset_index().get(token.lower())
    if not folder:
        return []

    pose_detector, hands_detector = _get_detectors()
    all_landmarks = []

    videos = sorted(folder.glob("*.mp4"))
    if videos:
        video_path = videos[0]
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        wanted = set(_sample_indices(total, max_frames))

        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx in wanted:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose_results = pose_detector.process(img_rgb)
                hands_results = hands_detector.process(img_rgb)
                all_landmarks.append(_landmarks_to_dict(pose_results, hands_results))
            idx += 1
        cap.release()
    else:
        frame_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        frames = [f for f in folder.iterdir() if f.suffix.lower() in frame_exts]
        frames = sorted(frames, key=_natural_key)
        for frame_path in frames[:max_frames]:
            image = cv2.imread(str(frame_path))
            if image is None:
                continue
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pose_results = pose_detector.process(img_rgb)
            hands_results = hands_detector.process(img_rgb)
            all_landmarks.append(_landmarks_to_dict(pose_results, hands_results))

    return all_landmarks


def landmarks_for_tokens(tokens, max_frames_per_token: int = 12):
    sequence = []
    if not tokens:
        return sequence

    for token in tokens:
        token_frames = landmarks_for_token(str(token), max_frames=max_frames_per_token)
        if token_frames:
            sequence.extend(token_frames)

    return sequence


def _skeleton_html_for_frames(frames, fps: int):
    if not frames:
        return None

    fps = max(1, min(20, int(fps)))
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset=\"utf-8\" />
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #0a0a0f; display:flex; flex-direction:column; align-items:center; padding:6px; }}
  canvas {{ border-radius: 14px; box-shadow: 0 0 20px rgba(124,106,247,0.25); background:#0d0d1a; }}
  .counter {{ color:#8079b0; font-size:10px; font-family:monospace; margin-top:6px; }}
</style>
</head>
<body>
<canvas id=\"c\" width=\"360\" height=\"520\"></canvas>
<div id=\"counter\" class=\"counter\"></div>
<script>
(function(){{
  const seq = {json.dumps(frames)};
  const fps = {fps};
  const poseConns = {json.dumps(POSE_UPPER)};
  const handConns = {json.dumps([[a, b] for a, b in HAND_CONNECTIONS])};

  const canvas = document.getElementById('c');
  const counter = document.getElementById('counter');
  const ctx = canvas.getContext('2d');
  const W = canvas.width;
  const H = canvas.height;

  let idx = 0;

  function toXY(lm) {{ return [lm[0] * W, lm[1] * H]; }}

  function drawGrid() {{
    ctx.strokeStyle = 'rgba(255,255,255,0.03)';
    ctx.lineWidth = 1;
    for (let x = 0; x < W; x += 40) {{
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
    }}
    for (let y = 0; y < H; y += 40) {{
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    }}
  }}

  function drawPose(lms) {{
    if (!lms || !lms.length) return;
    ctx.strokeStyle = '#7c6af7';
    ctx.lineWidth = 3;
    ctx.shadowColor = '#7c6af7';
    ctx.shadowBlur = 10;
    poseConns.forEach(([a, b]) => {{
      if (a >= lms.length || b >= lms.length) return;
      const [x1, y1] = toXY(lms[a]);
      const [x2, y2] = toXY(lms[b]);
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    }});
    ctx.shadowBlur = 0;
  }}

  function drawHand(lms, baseColor, glowColor) {{
    if (!lms || lms.length < 21) return;

    const pts = lms.map((lm) => toXY(lm));
    const fingerColors = ['#f7a6c1', '#ffcc80', '#a8f0c6', '#80d4ff', '#d4a8ff'];
    const fingers = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20]];
    const palmIdx = [0,1,5,9,13,17];

    ctx.beginPath();
    palmIdx.forEach((i, n) => {{
      const [x, y] = pts[i];
      if (n === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }});
    ctx.closePath();

    const pg = ctx.createRadialGradient(pts[9][0], pts[9][1], 2, pts[9][0], pts[9][1], W * 0.12);
    pg.addColorStop(0, baseColor + 'cc');
    pg.addColorStop(1, baseColor + '44');
    ctx.fillStyle = pg;
    ctx.shadowColor = glowColor;
    ctx.shadowBlur = 15;
    ctx.fill();
    ctx.shadowBlur = 0;

    fingers.forEach((joints, fi) => {{
      const col = fingerColors[fi];
      for (let j = 0; j < joints.length - 1; j++) {{
        const [x1, y1] = pts[joints[j]];
        const [x2, y2] = pts[joints[j + 1]];
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.strokeStyle = col;
        ctx.lineWidth = 3 - j * 0.45;
        ctx.shadowColor = col;
        ctx.shadowBlur = 6;
        ctx.stroke();
        ctx.shadowBlur = 0;
      }}
      const tip = pts[joints[joints.length - 1]];
      ctx.beginPath();
      ctx.arc(tip[0], tip[1], 4, 0, Math.PI * 2);
      ctx.fillStyle = col;
      ctx.fill();
    }});

    handConns.forEach(([a, b]) => {{
      if (a >= pts.length || b >= pts.length) return;
      const [x1, y1] = pts[a];
      const [x2, y2] = pts[b];
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.strokeStyle = baseColor + '88';
      ctx.lineWidth = 1;
      ctx.stroke();
    }});
  }}

  function drawHead(lms) {{
    if (!lms || !lms.length) return;
    const [nx, ny] = toXY(lms[0]);
    const r = H * 0.05;
    const g = ctx.createRadialGradient(nx, ny - r * .2, r * .1, nx, ny, r);
    g.addColorStop(0, 'rgba(200,190,255,0.9)');
    g.addColorStop(1, 'rgba(124,106,247,0.12)');
    ctx.beginPath();
    ctx.arc(nx, ny, r, 0, Math.PI * 2);
    ctx.fillStyle = g;
    ctx.fill();
  }}

  function drawFrame() {{
    if (!seq.length) return;
    const frame = seq[idx];
    const pose = frame.pose || [];
    const left = frame.left_hand || [];
    const right = frame.right_hand || [];

    ctx.clearRect(0, 0, W, H);
    drawGrid();
    drawPose(pose);
    drawHand(left, '#6af7d4', '#00ffcc');
    drawHand(right, '#f7a6c1', '#ff80aa');
    drawHead(pose);

    counter.textContent = `Frame ${{idx + 1}} / ${{seq.length}}`;
    idx = (idx + 1) % seq.length;
  }}

  drawFrame();
  setInterval(drawFrame, Math.max(40, Math.floor(1000 / fps)));
}})();
</script>
</body>
</html>"""


def skeleton_html_for_token(token: str, fps: int = 6, max_frames: int = 12):
    frames = landmarks_for_token(token, max_frames=max_frames)
    return _skeleton_html_for_frames(frames, fps=fps)


def skeleton_html_for_tokens(tokens, fps: int = 6, max_frames_per_token: int = 12):
    frames = landmarks_for_tokens(tokens, max_frames_per_token=max_frames_per_token)
    return _skeleton_html_for_frames(frames, fps=fps)
