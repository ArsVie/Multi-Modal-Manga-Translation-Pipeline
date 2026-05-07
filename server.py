"""FastAPI server for manga translation.
Single endpoint: POST /translate with manga page image, returns translated page.

Usage:
    pip install fastapi uvicorn python-multipart
    uvicorn server:app --host 0.0.0.0 --port 8000
"""

import tempfile
import os
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse

# Import the translator — assumes MangaTranslator.py is in the same directory
import sys
sys.path.insert(0, os.path.dirname(__file__))

try:
    from MangaTranslator import MangaTranslator
except ImportError:
    # Fallback for different project layouts
    sys.path.insert(0, '/tmp/ars-repos/Multi-Modal-Manga-Translation-Pipeline')
    from MangaTranslator import MangaTranslator

app = FastAPI(
    title="Manga Translator API",
    description="Upload a Japanese manga page, get an English translated page back.",
    version="2.0.0",
)

# Load translator once at startup
translator: MangaTranslator | None = None


@app.on_event("startup")
def load_models():
    """Initialize translation pipeline on server start."""
    global translator
    translator = MangaTranslator(
        yolo_model_path=os.getenv("YOLO_MODEL", "comic-speech-bubble-detector.pt"),
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen3.5:9b"),
        font_path=os.getenv("FONT_PATH", "font.ttf"),
        debug=False,
    )


@app.post("/translate")
async def translate_page(image: UploadFile = File(...)):
    """Translate a single manga page.

    Accepts JPEG/PNG upload, returns the translated image.
    """
    if translator is None:
        raise HTTPException(503, "Models still loading, retry in a moment")

    if image.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(400, "Only JPEG and PNG images supported")

    # Save uploaded file to temp location
    suffix = Path(image.filename).suffix if image.filename else ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
        tmp_in.write(await image.read())
        input_path = tmp_in.name

    output_path = input_path + "_translated.jpg"

    try:
        translator.detect_and_process(input_path, output_dir="/tmp/manga-api-crops", page_id="api")
        # For single image, we need a simpler path than process_chapter
        # Use detect + OCR + translate + typeset directly
        from PIL import Image
        import cv2

        img, data = translator.detect_and_process(
            input_path, output_dir="/tmp/manga-api-crops", page_id="single"
        )
        if data:
            data = translator.run_ocr(data)
            data = translator.translate_batch(data)
            translator.typeset(img, data, output_path)

        return FileResponse(
            output_path,
            media_type="image/jpeg",
            filename=f"translated_{image.filename or 'page.jpg'}",
        )

    except Exception as e:
        raise HTTPException(500, f"Translation failed: {e}")

    finally:
        # Cleanup temp files
        for p in (input_path, output_path):
            try:
                os.unlink(p)
            except OSError:
                pass


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ready" if translator else "loading", "models_loaded": translator is not None}
