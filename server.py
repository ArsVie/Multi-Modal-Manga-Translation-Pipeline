"""FastAPI web app for the manga translation pipeline.

Endpoints:
    GET  /          -> the web UI (index.html)
    GET  /status    -> readiness of the pipeline and the local LLM server
    POST /translate -> upload manga pages, get translated images back (base64)

Run:
    uvicorn server:app --host 0.0.0.0 --port 8000
"""

import base64
import io
import os
import re
import shutil
import tempfile
import threading
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse

from MangaTranslator import MangaTranslator

BASE_DIR = Path(__file__).resolve().parent

# Model assets. Override with environment variables if they live elsewhere.
YOLO_MODEL = os.getenv("YOLO_MODEL", str(BASE_DIR / "comic-speech-bubble-detector.pt"))
FONT_PATH = os.getenv("FONT_PATH", str(BASE_DIR / "animeace2_reg.ttf"))

# Local llama.cpp server (OpenAI-compatible API). Gemma 4 26B-A4B by default.
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8110")
LLM_MODEL = os.getenv("LLM_MODEL", "local")

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
VALID_EXTENSIONS = IMAGE_EXTENSIONS + (".zip",)

translator = None
load_error = None
pipeline_lock = threading.Lock()  # one translate request at a time


def load_models():
    """Load YOLO + LaMa + MangaOCR once at startup."""
    global translator, load_error
    try:
        print(f"Loading models (YOLO={YOLO_MODEL}, font={FONT_PATH})...")
        translator = MangaTranslator(
            yolo_model_path=YOLO_MODEL,
            llm_base_url=LLM_BASE_URL,
            llm_model=LLM_MODEL,
            font_path=FONT_PATH,
            debug=False,
        )
        print("Models loaded. Ready for /translate requests.")
    except Exception as exc:  # keep the server up; /translate reports the error
        load_error = str(exc)
        print(f"ERROR: could not load models: {exc}")
        print("Hint: download comic-speech-bubble-detector.pt and a font into the")
        print("project folder (see README), or set YOLO_MODEL / FONT_PATH.")


@asynccontextmanager
async def lifespan(_app):
    load_models()
    yield


app = FastAPI(title="Manga Translator", lifespan=lifespan)


def _llm_state(base_url):
    """'ready' when the LLM server has a model loaded, 'loading' while it
    loads, 'unreachable' when nothing answers."""
    try:
        health = requests.get(f"{base_url}/health", timeout=3)
        return "ready" if health.status_code == 200 else "loading"
    except requests.RequestException:
        return "unreachable"


@app.get("/", response_class=HTMLResponse)
def index():
    """Serve the web UI."""
    return (BASE_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/status")
def status():
    return {
        "models_loaded": translator is not None,
        "load_error": load_error,
        "llm_base_url": LLM_BASE_URL,
        "llm_state": _llm_state(LLM_BASE_URL),
    }


def _parse_custom_translations(raw):
    """Parse 'jp=English' pairs (one per line or comma-separated, '#' comments allowed)."""
    custom = {}
    for chunk in re.split(r"[\n,]", raw or ""):
        chunk = chunk.strip()
        if not chunk or chunk.startswith("#") or "=" not in chunk:
            continue
        jp, _, en = chunk.partition("=")
        if jp.strip():
            custom[jp.strip()] = en.strip()
    return custom


def _natural_key(name):
    """Natural sort key: page2 sorts before page10."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def _unique_name(name, taken):
    """`name`, or name_2 / name_3... when already taken (case-insensitive)."""
    if name.lower() not in taken:
        taken.add(name.lower())
        return name
    stem, ext = os.path.splitext(name)
    n = 2
    while f"{stem}_{n}{ext}".lower() in taken:
        n += 1
    final = f"{stem}_{n}{ext}"
    taken.add(final.lower())
    return final


def _extract_zip(content, dest_dir, taken, max_entries=2000, max_total_bytes=2 * 1024 ** 3):
    """Extract image files from a zip upload into dest_dir (folders flattened)."""
    out = []
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        total = 0
        for info in zf.infolist()[:max_entries]:
            if info.is_dir():
                continue
            base = os.path.basename(info.filename.replace("\\", "/"))
            if (not base or base.startswith(".") or "__MACOSX" in info.filename
                    or os.path.splitext(base)[1].lower() not in IMAGE_EXTENSIONS):
                continue
            total += info.file_size
            if total > max_total_bytes:
                raise ValueError("zip expands to too much data")
            name = _unique_name(base, taken)
            (Path(dest_dir) / name).write_bytes(zf.read(info))
            out.append(name)
    return out


def _process(files, settings):
    """Run the full pipeline on the uploaded pages. Blocking; runs in a thread."""
    assert translator is not None  # the endpoint checks this before dispatching
    with pipeline_lock:
        translator.set_llm(base_url=settings["llm_base_url"], model=None)
        translator.custom_translations = settings["custom_translations"]
        translator.keep_honorifics = settings["keep_honorifics"]
        translator.font_scale = settings["font_scale"]
        if settings["font_scale"] != 1.0:
            print(f"  (lettering scale {settings['font_scale']})")

        in_dir = tempfile.mkdtemp(prefix="manga_in_")
        out_dir = tempfile.mkdtemp(prefix="manga_out_")
        try:
            page_names = []
            taken = set()
            zip_failures = []
            for name, content in files:
                safe = os.path.basename(name) or f"page_{len(page_names) + 1}.jpg"
                if safe.lower().endswith(".zip"):
                    try:
                        extracted = _extract_zip(content, in_dir, taken)
                    except Exception as exc:
                        zip_failures.append((safe, f"could not unpack {safe}: {exc}"))
                        continue
                    if not extracted:
                        zip_failures.append((safe, f"{safe} contained no images"))
                    page_names.extend(extracted)
                    continue
                final = _unique_name(safe, taken)
                (Path(in_dir) / final).write_bytes(content)
                page_names.append(final)

            page_names.sort(key=_natural_key)

            series_info = None
            if settings["title"] or settings["tags"] or settings["description"]:
                series_info = {
                    "title": settings["title"] or "Unknown",
                    "tags": settings["tags"],
                    "description": settings["description"],
                }

            translator.process_chapter(
                input_folder=in_dir,
                output_folder=out_dir,
                series_info=series_info,
                batch_size=settings["batch_size"],
                conf_threshold=settings["conf_threshold"],
                save_comparisons=settings["save_comparisons"],
            )

            pages = []
            for name in page_names:
                out_path = Path(out_dir) / name
                comparison_path = Path(out_dir) / f"{Path(name).stem}_comparison.png"
                page = {"filename": name, "translated_b64": None, "comparison_b64": None}
                if out_path.exists():
                    page["translated_b64"] = base64.b64encode(out_path.read_bytes()).decode()
                else:
                    page["error"] = "translation failed — check the server logs"
                if comparison_path.exists():
                    page["comparison_b64"] = base64.b64encode(comparison_path.read_bytes()).decode()
                pages.append(page)
            for fname, err in zip_failures:
                pages.append({"filename": fname, "translated_b64": None,
                              "comparison_b64": None, "error": err})
            return pages
        finally:
            shutil.rmtree(in_dir, ignore_errors=True)
            shutil.rmtree(out_dir, ignore_errors=True)


@app.post("/translate")
async def translate(
    files: list[UploadFile] = File(...),
    title: str = Form(""),
    tags: str = Form(""),
    description: str = Form(""),
    custom_translations: str = Form(""),
    keep_honorifics: bool = Form(False),
    conf_threshold: float = Form(0.15),
    batch_size: int = Form(4),
    font_scale: float = Form(1.0),
    save_comparisons: bool = Form(True),
    llm_base_url: str = Form(""),
):
    if translator is None:
        raise HTTPException(503, f"Models are not loaded. {load_error or 'Check the server logs.'}")

    if not files:
        raise HTTPException(400, "No files uploaded")

    for f in files:
        if Path(f.filename or "").suffix.lower() not in VALID_EXTENSIONS:
            raise HTTPException(400, f"Unsupported file type: {f.filename} (use PNG/JPG/WEBP/BMP or a .zip)")

    base_url = (llm_base_url or LLM_BASE_URL).rstrip("/")
    state = _llm_state(base_url)
    if state != "ready":
        detail = {
            "unreachable": f"LLM server at {base_url} is not reachable. Start it first "
                           f"(see README: 'Local LLM server').",
            "loading": f"LLM server at {base_url} is still loading its model. Try again in a minute.",
        }.get(state, state)
        raise HTTPException(503, detail)

    uploads = [(f.filename, await f.read()) for f in files]
    settings = {
        "title": title.strip(),
        "tags": tags.strip(),
        "description": description.strip(),
        "custom_translations": _parse_custom_translations(custom_translations),
        "keep_honorifics": keep_honorifics,
        "conf_threshold": conf_threshold,
        "batch_size": batch_size,
        "font_scale": min(1.0, max(0.5, font_scale)),
        "save_comparisons": save_comparisons,
        "llm_base_url": base_url,
    }

    try:
        pages = await run_in_threadpool(_process, uploads, settings)
    except Exception as exc:
        raise HTTPException(500, f"Processing failed: {exc}")

    return JSONResponse({"pages": pages})
