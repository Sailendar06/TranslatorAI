import os
import uuid
import shutil
import logging
import time
from pathlib import Path
import sentencepiece as spm

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageFont, ExifTags
import pytesseract
from pytesseract import Output
import ctranslate2
from IndicTransToolkit import IndicProcessor
from langdetect import detect, LangDetectException

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR
AUDIO_DIR  = BASE_DIR / "audio_outputs"
STATIC_DIR = BASE_DIR / "static"
FONTS_DIR  = BASE_DIR / "fonts"
AUDIO_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)
FONTS_DIR.mkdir(exist_ok=True)

# ── Language maps ──────────────────────────────────────────────────────────────
LANGDETECT_MAP = {
    "ta": "tam_Taml", "hi": "hin_Deva", "en": "eng_Latn",
    "te": "tel_Telu", "bn": "ben_Beng", "ml": "mal_Mlym",
    "mr": "mar_Deva", "gu": "guj_Gujr", "pa": "pan_Guru",
    "kn": "kan_Knda", "or": "ory_Orya", "ur": "urd_Arab",
}

TESSERACT_MAP = {
    "tam_Taml": "tam", "hin_Deva": "hin", "eng_Latn": "eng",
    "tel_Telu": "tel", "ben_Beng": "ben", "mal_Mlym": "mal",
    "mar_Deva": "mar", "guj_Gujr": "guj", "pan_Guru": "pan",
    "kan_Knda": "kan",
}

PIPER_VOICE_MAP = {
    "eng_Latn": "en_US-lessac-medium",
    "hin_Deva": "hi_IN-pratham-medium",
    "tam_Taml": "ta_IN-gowajee-medium",
    "tel_Telu": "te_IN-maya-medium",
    "ben_Beng": "bn_IN-bangla-medium",
    "mal_Mlym": "ml_IN-meera-medium",
    "kan_Knda": "kn_IN-divya-medium",
    "guj_Gujr": "gu_IN-divya-medium",
    "mar_Deva": "mr_IN-divya-medium",
    "pan_Guru": "en_US-lessac-medium",
    "ory_Orya": "en_US-lessac-medium",
    "urd_Arab": "en_US-lessac-medium",
}

ESPEAK_VOICE_MAP = {
    "eng_Latn": "en",  "hin_Deva": "hi",  "tam_Taml": "ta",
    "tel_Telu": "te",  "ben_Beng": "bn",  "mal_Mlym": "ml",
    "kan_Knda": "kn",  "guj_Gujr": "gu",  "mar_Deva": "mr",
    "pan_Guru": "pa",  "ory_Orya": "or",  "urd_Arab": "ur",
}

# Font search order per language (first found wins)
FONT_SEARCH = {
    "eng_Latn": ["NotoSans-Regular.ttf",           "DejaVuSans.ttf",           "FreeSans.ttf"],
    "tam_Taml": ["NotoSansTamil-Regular.ttf",      "lohit_ta.ttf"],
    "hin_Deva": ["NotoSansDevanagari-Regular.ttf", "lohit_hi.ttf"],
    "tel_Telu": ["NotoSansTelugu-Regular.ttf",     "lohit_te.ttf"],
    "ben_Beng": ["NotoSansBengali-Regular.ttf",    "lohit_bn.ttf"],
    "mal_Mlym": ["NotoSansMalayalam-Regular.ttf",  "lohit_ml.ttf"],
    "kan_Knda": ["NotoSansKannada-Regular.ttf",    "lohit_kn.ttf"],
    "guj_Gujr": ["NotoSansGujarati-Regular.ttf",   "lohit_gu.ttf"],
    "mar_Deva": ["NotoSansDevanagari-Regular.ttf", "lohit_mr.ttf"],
    "pan_Guru": ["NotoSansGurmukhi-Regular.ttf",   "lohit_pa.ttf"],
    "ory_Orya": ["NotoSansOriya-Regular.ttf",      "lohit_or.ttf"],
    "urd_Arab": ["NotoSansArabic-Regular.ttf",     "lohit_ur.ttf"],
}

# Directories to search for fonts (in priority order)
SYSTEM_FONT_DIRS = [
    FONTS_DIR,
    Path("/usr/share/fonts/truetype/noto"),
    Path("/usr/share/fonts/opentype/noto"),
    Path("/usr/share/fonts/truetype/lohit-devanagari"),
    Path("/usr/share/fonts/truetype/lohit-tamil"),
    Path("/usr/share/fonts/truetype/dejavu"),
    Path("/usr/share/fonts/truetype/freefont"),
    Path("/usr/share/fonts/truetype"),
    Path("/usr/share/fonts"),
    Path("/usr/local/share/fonts"),
]

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="TranslatorAI — Offline AI Translator")
app.mount("/static",        StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/audio_outputs", StaticFiles(directory=str(AUDIO_DIR)),  name="audio")

# ── Load IndicTrans2 models ────────────────────────────────────────────────────
def _sp_path(model_dir: Path, suffix: str) -> str:
    p = model_dir / "vocab" / f"model.{suffix}"
    if p.exists():
        return str(p)
    return str(model_dir / f"model.{suffix}")

log.info("Loading IndicTrans2  indic-en  (INT8)…")
_INDIC_EN_DIR = MODELS_DIR / "indictrans2-indic-en-dist-200M-ct2-int8"
_translator_indic_en = ctranslate2.Translator(
    str(_INDIC_EN_DIR), device="cpu", compute_type="int8",
    inter_threads=2, intra_threads=2,
)
_sp_indic_en_src = spm.SentencePieceProcessor(_sp_path(_INDIC_EN_DIR, "SRC"))
_sp_indic_en_tgt = spm.SentencePieceProcessor(_sp_path(_INDIC_EN_DIR, "TGT"))

_EN_INDIC_DIR = MODELS_DIR / "indictrans2-en-indic-dist-200M-ct2-int8"
_translator_en_indic = None
_sp_en_indic_src     = None
_sp_en_indic_tgt     = None

if _EN_INDIC_DIR.exists():
    log.info("Loading IndicTrans2  en-indic  (INT8)…")
    _translator_en_indic = ctranslate2.Translator(
        str(_EN_INDIC_DIR), device="cpu", compute_type="int8",
        inter_threads=2, intra_threads=2,
    )
    _sp_en_indic_src = spm.SentencePieceProcessor(_sp_path(_EN_INDIC_DIR, "SRC"))
    _sp_en_indic_tgt = spm.SentencePieceProcessor(_sp_path(_EN_INDIC_DIR, "TGT"))
else:
    log.warning("en-indic model not found — English→Indic and Indic→Indic pivot disabled")

_processor = IndicProcessor(inference=True)
log.info("Models ready ✓")

# ── Whisper (lazy) ─────────────────────────────────────────────────────────────
_whisper_model = None

def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        log.info("Loading Whisper base…")
        _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
        log.info("Whisper ready ✓")
    return _whisper_model

# ── EXIF orientation fix ───────────────────────────────────────────────────────
def fix_image_orientation(img: Image.Image) -> Image.Image:
    try:
        exif = img._getexif()
        if exif is None:
            return img
        orientation_key = next(
            (k for k, v in ExifTags.TAGS.items() if v == "Orientation"), None
        )
        if orientation_key is None or orientation_key not in exif:
            return img
        orientation = exif[orientation_key]
        rotation_map = {3: 180, 6: 270, 8: 90}
        flip_map     = {
            2: (Image.FLIP_LEFT_RIGHT, 0),
            4: (Image.FLIP_TOP_BOTTOM, 0),
            5: (Image.FLIP_LEFT_RIGHT, 90),
            7: (Image.FLIP_LEFT_RIGHT, 270),
        }
        if orientation in rotation_map:
            img = img.rotate(rotation_map[orientation], expand=True)
        elif orientation in flip_map:
            flip_op, angle = flip_map[orientation]
            img = img.transpose(flip_op)
            if angle:
                img = img.rotate(angle, expand=True)
    except Exception as e:
        log.warning(f"EXIF fix failed (non-fatal): {e}")
    return img

# ── Translation ────────────────────────────────────────────────────────────────
def detect_lang(text: str, fallback: str = "tam_Taml") -> str:
    try:
        detected = detect(text)
        result   = LANGDETECT_MAP.get(detected, fallback)
        log.info(f"detect_lang: raw='{detected}' → '{result}'")
        return result
    except LangDetectException as e:
        log.warning(f"langdetect failed: {e} — fallback='{fallback}'")
        return fallback

def is_english(lang: str) -> bool:
    return lang == "eng_Latn"

def _encode_batch(preprocessed: list, sp: spm.SentencePieceProcessor) -> list:
    encoded = []
    for s in preprocessed:
        parts = s.split(" ", 2)
        if len(parts) == 3 and len(parts[0]) == 8 and len(parts[1]) == 8 and "_" in parts[0] and "_" in parts[1]:
            src_tag, tgt_tag, text = parts
            tokens = [src_tag, tgt_tag] + sp.encode(text, out_type=str)
        else:
            tokens = sp.encode(s, out_type=str)
           
        if not tokens:
            log.warning(f"SentencePiece: empty token list for: {repr(s[:80])}")
        encoded.append(tokens)
    return encoded

def translate(text: str, src_lang: str, tgt_lang: str) -> str:
    text = text.strip()
    if not text:
        return text
    if src_lang == tgt_lang:
        return text

    log.info(f"translate: {src_lang} → {tgt_lang} | '{text[:80]}'")

    if not is_english(src_lang) and is_english(tgt_lang):
        try:
            batch    = _processor.preprocess_batch([text], src_lang=src_lang, tgt_lang=tgt_lang)
            tokens   = _encode_batch(batch, _sp_indic_en_src)
            results  = _translator_indic_en.translate_batch(
                tokens, beam_size=4, max_decoding_length=256, no_repeat_ngram_size=2,
            )
            raw_text = [_sp_indic_en_tgt.decode(results[0].hypotheses[0])]
            final    = _processor.postprocess_batch(raw_text, lang=tgt_lang)
            log.info(f"translate result: '{final[0][:80]}'")
            return final[0]
        except Exception as e:
            log.error(f"translate (indic→en) FAILED: {e}", exc_info=True)
            raise

    if is_english(src_lang) and not is_english(tgt_lang):
        if _translator_en_indic is None:
            raise RuntimeError(
                "en-indic model not loaded. "
                "Place 'indictrans2-en-indic-dist-200M-ct2-int8' next to main.py."
            )
        try:
            batch    = _processor.preprocess_batch([text], src_lang=src_lang, tgt_lang=tgt_lang)
            tokens   = _encode_batch(batch, _sp_en_indic_src)
            results  = _translator_en_indic.translate_batch(
                tokens, beam_size=4, max_decoding_length=256, no_repeat_ngram_size=2,
            )
            raw_text = [_sp_en_indic_tgt.decode(results[0].hypotheses[0])]
            final    = _processor.postprocess_batch(raw_text, lang=tgt_lang)
            log.info(f"translate result: '{final[0][:80]}'")
            return final[0]
        except Exception as e:
            log.error(f"translate (en→indic) FAILED: {e}", exc_info=True)
            raise

    log.info(f"Indic→Indic pivot: {src_lang} → eng_Latn → {tgt_lang}")
    en_text = translate(text, src_lang, "eng_Latn")
    return translate(en_text, "eng_Latn", tgt_lang)

# ── Font utilities ─────────────────────────────────────────────────────────────
_font_cache: dict = {}

def _find_font_file(filename: str) -> Path | None:
    for d in SYSTEM_FONT_DIRS:
        if not d.exists():
            continue
        p = d / filename
        if p.exists():
            return p
        for sub in d.iterdir():
            if sub.is_dir():
                sp = sub / filename
                if sp.exists():
                    return sp
    return None

def _load_font(lang_token: str, size: int) -> ImageFont.ImageFont:
    cache_key = (lang_token, size)
    if cache_key in _font_cache:
        return _font_cache[cache_key]

    for fname in FONT_SEARCH.get(lang_token, FONT_SEARCH["eng_Latn"]):
        found = _find_font_file(fname)
        if found:
            try:
                font = ImageFont.truetype(str(found), size=size)
                _font_cache[cache_key] = font
                return font
            except Exception as e:
                log.warning(f"Failed to load {found}: {e}")

    log.error(
        f"No font found for lang={lang_token}. "
        f"Run: sudo apt-get install fonts-noto fonts-noto-extra"
    )
    font = ImageFont.load_default()
    _font_cache[cache_key] = font
    return font

def _audit_fonts():
    log.info("─── Font audit ──────────────────────────────")
    for lang, fnames in FONT_SEARCH.items():
        hit = next((_find_font_file(f) for f in fnames if _find_font_file(f)), None)
        if hit:
            log.info(f"  ✓ {lang:15s} → {hit}")
        else:
            log.warning(f"  ✗ {lang:15s} → MISSING  (run: sudo apt-get install fonts-noto)")
    log.info("─────────────────────────────────────────────")

_audit_fonts()

# ── OCR ────────────────────────────────────────────────────────────────────────
_AUTO_DETECT_LANGS = ["tam", "hin", "tel", "kan", "mal", "ben", "guj", "mar", "eng"]

def ocr_image(image_path: str, src_lang_token: str) -> str:
    tess_lang = TESSERACT_MAP.get(src_lang_token, "eng")
    if tess_lang != "eng":
        tess_lang = tess_lang + "+eng"
    try:
        img  = fix_image_orientation(Image.open(image_path))
        text = pytesseract.image_to_string(img, lang=tess_lang).strip()
        if not text:
            log.warning(f"OCR empty for lang={tess_lang}, retrying with eng")
            text = pytesseract.image_to_string(img, lang="eng").strip()
        log.info(f"OCR: {len(text)} chars (lang={tess_lang})")
        return text
    except pytesseract.TesseractError as e:
        log.warning(f"Tesseract error ({tess_lang}): {e} — retrying eng")
        return pytesseract.image_to_string(
            fix_image_orientation(Image.open(image_path)), lang="eng"
        ).strip()

def ocr_auto_detect(image_path: str) -> str:
    img = fix_image_orientation(Image.open(image_path))
    for attempt in range(len(_AUTO_DETECT_LANGS), 0, -1):
        lang_str = "+".join(_AUTO_DETECT_LANGS[:attempt])
        try:
            text = pytesseract.image_to_string(img, lang=lang_str).strip()
            if text:
                log.info(f"Auto-OCR: {len(text)} chars with langs={lang_str}")
                return text
        except pytesseract.TesseractError as e:
            log.warning(f"Auto-OCR failed (langs={lang_str}): {e}")
    return ""

# ── TTS ────────────────────────────────────────────────────────────────────────
def _unique_audio_name() -> str:
    return f"out_{uuid.uuid4().hex}.wav"

def tts_espeak(text: str, lang_token: str) -> str | None:
    import subprocess
    voice    = ESPEAK_VOICE_MAP.get(lang_token, "en")
    out_file = AUDIO_DIR / _unique_audio_name()
    try:
        r = subprocess.run(
            ["espeak-ng", "-v", voice, "-s", "145", "-w", str(out_file), text],
            capture_output=True, timeout=20,
        )
        if r.returncode == 0 and out_file.exists() and out_file.stat().st_size > 0:
            return f"/audio_outputs/{out_file.name}"
        log.warning(f"eSpeak-NG rc={r.returncode}: {r.stderr.decode()[:200]}")
    except Exception as e:
        log.warning(f"eSpeak-NG exception: {e}")
    return None

def tts_piper(text: str, lang_token: str) -> str | None:
    import subprocess
    voice_stem  = PIPER_VOICE_MAP.get(lang_token)
    if not voice_stem:
        return tts_espeak(text, lang_token)

    voice_path  = BASE_DIR / "voices" / (voice_stem + ".onnx")
    config_path = BASE_DIR / "voices" / (voice_stem + ".onnx.json")

    if not voice_path.exists() or not config_path.exists():
        return tts_espeak(text, lang_token)

    out_file = AUDIO_DIR / _unique_audio_name()
    try:
        r = subprocess.run(
            ["piper", "--model", str(voice_path), "--output_file", str(out_file)],
            input=text.encode("utf-8"),
            capture_output=True, timeout=30,
        )
        if r.returncode == 0 and out_file.exists() and out_file.stat().st_size > 0:
            return f"/audio_outputs/{out_file.name}"
        log.warning(f"Piper rc={r.returncode}: {r.stderr.decode()[:200]}")
    except Exception as e:
        log.warning(f"Piper exception: {e}")
    return tts_espeak(text, lang_token)

# ── In-place image translation ─────────────────────────────────────────────────
def create_translated_image(
    image_path: str, src_lang_token: str, tgt_lang_token: str
) -> str | None:
    tess_lang = TESSERACT_MAP.get(src_lang_token, "eng")
    if tess_lang != "eng":
        tess_lang = tess_lang + "+eng"

    try:
        img  = fix_image_orientation(Image.open(image_path)).convert("RGB")
        draw = ImageDraw.Draw(img)
        data = pytesseract.image_to_data(img, lang=tess_lang, output_type=Output.DICT)

        # Group by PARAGRAPH to maintain translation context
        paragraphs: dict = {}
        for i in range(len(data["text"])):
            if int(data["conf"][i]) > 30:
                word = data["text"][i].strip()
                if word:
                    key = (data["block_num"][i], data["par_num"][i])
                    if key not in paragraphs:
                        paragraphs[key] = {"words": [], "left": [], "top": [], "width": [], "height": []}
                    paragraphs[key]["words"].append(word)
                    paragraphs[key]["left"].append(data["left"][i])
                    paragraphs[key]["top"].append(data["top"][i])
                    paragraphs[key]["width"].append(data["width"][i])
                    paragraphs[key]["height"].append(data["height"][i])

        ok_count = 0
        for key, pd in paragraphs.items():
            original_text = " ".join(pd["words"])
            l = min(pd["left"])
            t = min(pd["top"])
            r = max(x + w for x, w in zip(pd["left"], pd["width"]))
            b = max(y + h for y, h in zip(pd["top"], pd["height"]))

            try:
                translated_text = translate(original_text, src_lang_token, tgt_lang_token)
            except Exception as e:
                log.warning(f"Paragraph translate skipped: {e}")
                continue

            draw.rectangle([l, t, r, b], fill="white")
            box_w = r - l
            box_h = b - t

            # ── Font size: derive from average word height, not box height ────
            # avg word height is a reliable proxy for the original font size
            avg_word_h = sum(pd["height"]) / max(len(pd["height"]), 1)
            max_fs = max(8, int(avg_word_h * 0.85))

            # Dynamic text wrapping & scaling — go down to size 6 if needed
            best_fs = 6
            wrapped_lines = [translated_text]

            for test_fs in range(max_fs, 5, -1):
                font = _load_font(tgt_lang_token, test_fs)
                words = translated_text.split()
                lines = []
                current_line = []

                for word in words:
                    test_line = " ".join(current_line + [word]) if current_line else word
                    length = font.getlength(test_line) if hasattr(font, "getlength") else draw.textlength(test_line, font=font)
                    if length <= box_w:
                        current_line.append(word)
                    else:
                        if not current_line:
                            lines.append(word)
                            current_line = []
                        else:
                            lines.append(" ".join(current_line))
                            current_line = [word]
                if current_line:
                    lines.append(" ".join(current_line))

                line_height  = test_fs * 1.3
                total_height = len(lines) * line_height

                if total_height <= box_h:
                    best_fs       = test_fs
                    wrapped_lines = lines
                    break

                # keep smallest attempt as fallback
                if test_fs == 6:
                    best_fs       = test_fs
                    wrapped_lines = lines

            font        = _load_font(tgt_lang_token, best_fs)
            current_y   = t
            line_height = best_fs * 1.3
            for line in wrapped_lines:
                draw.text((l, current_y), line, font=font, fill="black")
                current_y += line_height

            ok_count += 1

        log.info(f"create_translated_image: {ok_count} paragraphs drawn")
        out_name = f"trans_img_{uuid.uuid4().hex}.jpg"
        img.save(STATIC_DIR / out_name, quality=100, subsampling=0)
        return f"/static/{out_name}"

    except Exception as e:
        log.error(f"create_translated_image error: {e}", exc_info=True)
        return None

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open(STATIC_DIR / "index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/translate/image")
async def translate_image(
    file:        UploadFile = File(...),
    target_lang: str        = Form("eng_Latn"),
    src_lang:    str        = Form("auto"),
):
    tmp = BASE_DIR / f"tmp_{uuid.uuid4().hex}_{file.filename}"
    try:
        with open(tmp, "wb") as buf:
            shutil.copyfileobj(file.file, buf)

        if src_lang == "auto":
            quick        = ocr_auto_detect(str(tmp))
            resolved_src = detect_lang(quick, fallback="tam_Taml") if quick else "tam_Taml"
            log.info(f"Auto-detect → src_lang={resolved_src}")
        else:
            resolved_src = src_lang

        raw_text = ocr_image(str(tmp), resolved_src)
        if not raw_text:
            return {"status": "error", "message": "No text extracted from image."}

        log.info(f"OCR result ({len(raw_text)} chars): {raw_text[:150]}")

        try:
            translated = translate(raw_text, resolved_src, target_lang)
        except Exception as e:
            log.error(f"Translation failed: {e}", exc_info=True)
            return {"status": "error", "message": f"Translation failed: {e}"}

        audio_url            = tts_piper(translated, target_lang)
        translated_image_url = create_translated_image(str(tmp), resolved_src, target_lang)

        return {
            "status":               "success",
            "original_text":        raw_text,
            "translated_text":      translated,
            "detected_lang":        resolved_src,
            "input_type":           "image",
            "audio_url":            audio_url,
            "translated_image_url": translated_image_url,
        }
    except Exception as e:
        log.exception("Image translate error")
        return {"status": "error", "message": str(e)}
    finally:
        if tmp.exists():
            tmp.unlink()

@app.post("/translate/speech")
async def translate_speech(
    file:        UploadFile = File(...),
    target_lang: str        = Form("eng_Latn"),
):
    tmp = BASE_DIR / f"tmp_{uuid.uuid4().hex}_{file.filename}"
    try:
        with open(tmp, "wb") as buf:
            shutil.copyfileobj(file.file, buf)

        whisper      = get_whisper()
        segments, _  = whisper.transcribe(str(tmp), beam_size=3)
        raw_text     = " ".join(seg.text for seg in segments).strip()
        if not raw_text:
            return {"status": "error", "message": "No speech detected."}

        resolved_src = detect_lang(raw_text, fallback="eng_Latn")
        translated   = translate(raw_text, resolved_src, target_lang)
        audio_url    = tts_piper(translated, target_lang)

        return {
            "status":          "success",
            "original_text":   raw_text,
            "translated_text": translated,
            "detected_lang":   resolved_src,
            "input_type":      "speech",
            "audio_url":       audio_url,
        }
    except Exception as e:
        log.exception("Speech translate error")
        return {"status": "error", "message": str(e)}
    finally:
        if tmp.exists():
            tmp.unlink()

@app.post("/translate/text")
async def translate_text(
    text:        str = Form(...),
    target_lang: str = Form("eng_Latn"),
    src_lang:    str = Form("auto"),
):
    try:
        text = text.strip()
        if not text:
            return {"status": "error", "message": "No text provided."}

        resolved_src = detect_lang(text) if src_lang == "auto" else src_lang
        translated   = translate(text, resolved_src, target_lang)
        audio_url    = tts_piper(translated, target_lang)

        return {
            "status":          "success",
            "original_text":   text,
            "translated_text": translated,
            "detected_lang":   resolved_src,
            "input_type":      "text",
            "audio_url":       audio_url,
        }
    except Exception as e:
        log.exception("Text translate error")
        return {"status": "error", "message": str(e)}

@app.get("/health")
async def health():
    font_status = {}
    for lang, fnames in FONT_SEARCH.items():
        hit = next((_find_font_file(f) for f in fnames if _find_font_file(f)), None)
        font_status[lang] = str(hit) if hit else f"MISSING — install fonts-noto"
    return {
        "status":   "ok",
        "indic_en": "loaded",
        "en_indic": "loaded" if _translator_en_indic else "NOT LOADED",
        "whisper":  "lazy",
        "fonts":    font_status,
    }

@app.get("/debug/translate")
async def debug_translate(
    text: str = "Hello world",
    src:  str = "eng_Latn",
    tgt:  str = "tam_Taml",
):
    """Test translation directly in browser: /debug/translate?text=Hello&src=eng_Latn&tgt=tam_Taml"""
    try:
        result = translate(text, src, tgt)
        return {"input": text, "src": src, "tgt": tgt, "output": result, "ok": True}
    except Exception as e:
        return {"input": text, "src": src, "tgt": tgt, "error": str(e), "ok": False}
