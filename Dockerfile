FROM python:3.9-slim

# ffmpeg  — audio decoding (MP3, M4A, AAC, FLAC via audioread/librosa)
# libsndfile1 — WAV/AIFF decoding via soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api.py djay_sorter.py ./
COPY static/ static/

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
