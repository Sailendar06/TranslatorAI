# TranslatorAI — Offline Multilingual Translator (Raspberry Pi 5)

Fully offline OCR + Speech + Text translation across 12 Indian languages.
Runs on-device: no cloud, no internet needed after setup.

## Stack
- **OCR:** Tesseract
- **NMT:** IndicTrans2 (CTranslate2 INT8, indic↔en, pivot for indic↔indic)
- **STT:** faster-whisper (base)
- **TTS:** Piper (fallback: eSpeak-NG)
- **Backend:** FastAPI
- **Frontend:** static HTML/JS (`static/index.html`)

## Hardware
Tested on Raspberry Pi 5 (8GB RAM recommended — 200M param models + Whisper base
need headroom). 4GB may work but expect swap.

## Quick Start
```bash
git clone https://github.com/<you>/translator-ai.git
cd translator-ai
bash setup_models.sh          # pulls models, voices, system deps — takes a while
uvicorn main:app --host 0.0.0.0 --port 8000
```
Visit `http://<pi-ip>:8000`.

## Repo Layout
```
translator-ai/
├── main.py              # FastAPI backend — OCR, translate, TTS, image overlay
├── static/index.html    # frontend UI
├── requirements.txt      # python deps
├── setup_models.sh       # downloads/converts models (NOT in git — too big)
├── Dockerfile            # optional containerized run
├── .gitignore            # keeps models/voices out of git
└── README.md
```

## Models (not committed — see setup_models.sh)
| Model | Source | Purpose |
|---|---|---|
| indictrans2-indic-en-dist-200M | ai4bharat (HF) → CT2 int8 | Indic → English |
| indictrans2-en-indic-dist-200M | ai4bharat (HF) → CT2 int8 | English → Indic |
| whisper base | faster-whisper | Speech → text |
| Piper voices (9 langs) | rhasspy/piper-voices | Text → speech |

## Health Check
`GET /health` — reports which models loaded + which fonts found.

## Debug
`GET /debug/translate?text=Hello&src=eng_Latn&tgt=tam_Taml`

## License
<add your license here>
