FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    tesseract-ocr tesseract-ocr-tam tesseract-ocr-hin tesseract-ocr-tel \
    tesseract-ocr-ben tesseract-ocr-mal tesseract-ocr-kan tesseract-ocr-guj \
    tesseract-ocr-mar tesseract-ocr-pan \
    fonts-noto fonts-noto-extra espeak-ng wget git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN bash setup_models.sh

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
