FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV, MangaOCR
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install fastapi uvicorn python-multipart

# App files
COPY MangaTranslator.py .
COPY server.py .
COPY index.html .
COPY font.ttf . 2>/dev/null || true

# Models are mounted at runtime or downloaded
ENV YOLO_MODEL=/models/comic-speech-bubble-detector.pt
ENV OLLAMA_MODEL=qwen3.5:9b
ENV OLLAMA_HOST=http://host.docker.internal:11434
ENV FONT_PATH=/app/font.ttf

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
