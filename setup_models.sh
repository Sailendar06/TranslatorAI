#!/bin/bash
# setup_models.sh — re-fetch all models + system deps for TranslatorAI on Raspberry Pi 5
set -e

echo "── Python deps ──"
pip install -r requirements.txt

echo "── System deps (Tesseract, fonts, espeak) ──"
sudo apt-get update
sudo apt-get install -y \
  tesseract-ocr tesseract-ocr-tam tesseract-ocr-hin tesseract-ocr-tel \
  tesseract-ocr-ben tesseract-ocr-mal tesseract-ocr-kan tesseract-ocr-guj \
  tesseract-ocr-mar tesseract-ocr-pan \
  fonts-noto fonts-noto-extra espeak-ng wget git-lfs

echo "── IndicTrans2 CT2 int8 models ──"
mkdir -p models
cd models

if [ ! -d "indictrans2-indic-en-dist-200M-ct2-int8" ]; then
  pip install --quiet huggingface_hub ctranslate2
  python3 - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("ai4bharat/indictrans2-indic-en-dist-200M", local_dir="hf-indic-en")
EOF
  ct2-transformers-converter --model hf-indic-en --output_dir indictrans2-indic-en-dist-200M-ct2-int8 --quantization int8
fi

if [ ! -d "indictrans2-en-indic-dist-200M-ct2-int8" ]; then
  python3 - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("ai4bharat/indictrans2-en-indic-dist-200M", local_dir="hf-en-indic")
EOF
  ct2-transformers-converter --model hf-en-indic --output_dir indictrans2-en-indic-dist-200M-ct2-int8 --quantization int8
fi
cd ..

echo "── Piper TTS voices ──"
mkdir -p voices
cd voices
VOICES="en_US-lessac-medium hi_IN-pratham-medium ta_IN-gowajee-medium te_IN-maya-medium bn_IN-bangla-medium ml_IN-meera-medium kn_IN-divya-medium gu_IN-divya-medium mr_IN-divya-medium"
for v in $VOICES; do
  lang_dir=$(echo $v | cut -d'_' -f1)
  [ -f "${v}.onnx" ] || wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/main/${lang_dir}/${v%-*}/medium/${v}.onnx"
  [ -f "${v}.onnx.json" ] || wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/main/${lang_dir}/${v%-*}/medium/${v}.onnx.json"
done
cd ..

echo "── Done. Verify: curl http://localhost:8000/health ──"
