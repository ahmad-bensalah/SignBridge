import io
import re
from functools import lru_cache
from pathlib import Path

from PIL import Image

DATA_ROOT = Path(__file__).resolve().parent / "TextToSign" / "data"

# Romanized token -> Arabic display word
TOKEN_TO_ARABIC = {
    "3aslema": "عسلامة",
    "5adamet": "خدمة",
    "assam": "اسمك",
    "barnamjk": "برنامجك",
    "chabeb": "شباب",
    "cv": "سيفي",
    "demande": "طلب",
    "enti": "أنتِ",
    "labes": "لاباس",
    "lyoum": "اليوم",
    "mar7ba": "مرحبا",
    "n3awnek": "نعاونك",
    "nekteblk": "نكتبلك",
    "nemchi": "نمشي",
    "non": "لا",
    "oui": "آه",
    "radio": "راديو",
    "se7a": "صحة",
    "siye7a": "سياحة",
    "t7eb": "تحب",
    "ta3lim": "تعليم",
    "ta3raf": "تعرف",
    "ta9ra": "تقرا",
    "telvza": "تلفزة",
    "tha9afa": "ثقافة",
    "baladya": "بلدية",
    "banka": "بنكة",
    "bousta": "بوسطة",
    "dar": "دار",
    "ma7kma": "محكمة",
    "mostawsaf": "مستوصف",
    "sbitar": "سبيطار",
    "wzara": "وزارة",
    "3ayla": "عيلة",
    "5al-3am": "خال/عم",
    "5ou": "خو",
    "bent": "بنت",
    "bou": "بو",
    "eben": "ابن",
    "jad": "جد",
    "jadda": "جدة",
    "mar2a": "مرا",
    "o5t": "أخت",
    "om": "أم",
    "tfol": "طفل",
    "5mis": "الخميس",
    "a7ad": "الأحد",
    "erb3a": "الأربعاء",
    "jom3a": "الجمعة",
    "sebt": "السبت",
    "thleth": "الثلاثاء",
    "thnin": "الاثنين",
    "car": "كار",
    "karhba": "كرهبة",
    "louage": "لواج",
    "metro": "ميترو",
    "taxi": "تاكسي",
    "train": "تران",
}

ARABIC_TO_TOKEN = {arabic: token for token, arabic in TOKEN_TO_ARABIC.items()}

TOKEN_ALIASES = {
    "métro": "metro",
    "metre": "metro",
    "metrou": "metro",
}


def _natural_key(path_obj: Path):
    nums = re.findall(r"\((\d+)\)", path_obj.name)
    return int(nums[0]) if nums else 0


@lru_cache(maxsize=1)
def dataset_index():
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


def normalize_word(raw_word: str):
    cleaned = re.sub(r"(^[^\w\u0600-\u06FF-]+|[^\w\u0600-\u06FF-]+$)", "", raw_word)
    if not cleaned:
        return ""

    if cleaned in ARABIC_TO_TOKEN:
        return ARABIC_TO_TOKEN[cleaned]

    normalized = cleaned.lower()
    normalized = TOKEN_ALIASES.get(normalized, normalized)
    return normalized


def translate_text_to_sign_tokens(text: str):
    idx = dataset_index()
    tokens = []
    unresolved = []

    for raw_word in text.split():
        token = normalize_word(raw_word)
        if not token:
            continue
        if token in idx:
            tokens.append(token)
        else:
            unresolved.append(raw_word)

    return {
        "tokens": tokens,
        "unresolved": unresolved,
        "script": " ".join(f"🤟 {TOKEN_TO_ARABIC.get(t, t)}" for t in tokens),
    }


def media_for_token(token: str):
    idx = dataset_index()
    folder = idx.get(token.lower())
    if not folder:
        return None

    videos = sorted(folder.glob("*.mp4"))
    if videos:
        return {
            "kind": "video",
            "token": token,
            "label": TOKEN_TO_ARABIC.get(token, token),
            "video_path": str(videos[0]),
        }

    frame_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    frames = [f for f in folder.iterdir() if f.suffix.lower() in frame_exts]
    frames = sorted(frames, key=_natural_key)

    if frames:
        return {
            "kind": "frames",
            "token": token,
            "label": TOKEN_TO_ARABIC.get(token, token),
            "frame_paths": [str(f) for f in frames],
        }

    return None


@lru_cache(maxsize=256)
def gif_bytes_for_token(token: str, max_frames: int = 24, duration_ms: int = 110):
    media = media_for_token(token)
    if not media or media["kind"] != "frames":
        return None

    frame_paths = media["frame_paths"]
    if not frame_paths:
        return None

    step = max(1, len(frame_paths) // max_frames)
    selected = frame_paths[::step][:max_frames]

    images = []
    for frame_path in selected:
        try:
            image = Image.open(frame_path).convert("RGB")
            images.append(image)
        except Exception:
            continue

    if not images:
        return None

    buffer = io.BytesIO()
    images[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    return buffer.getvalue()
