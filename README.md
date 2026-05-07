# Multi-Modal Manga Translation Pipeline

End-to-end ML pipeline that automatically detects, extracts, translates, and typesets Japanese manga text.

## Quick Start

```bash
# Install
git clone https://github.com/ArsVie/Multi-Modal-Manga-Translation-Pipeline
cd Multi-Modal-Manga-Translation-Pipeline
pip install -r requirements.txt

# Download models
# - YOLO detector: comic-speech-bubble-detector.pt (place in project root)
# - Ollama: ollama pull qwen3.5:9b (or qwen3:8b)
# - Font: animeace2_reg.otf (place in project root)

# Run chapter translation
python3 -c "
from MangaTranslator import MangaTranslator
t = MangaTranslator(
    yolo_model_path='comic-speech-bubble-detector.pt',
    ollama_model='qwen3.5:9b',
    font_path='animeace2_reg.otf',
)
t.process_chapter('raw_chapter/', 'translated_chapter/', series_info={
    'title': 'Your Manga', 'tags': 'Action, Fantasy',
    'description': 'Brief plot summary...'
})
"
```

## Web UI + API

```bash
pip install fastapi uvicorn python-multipart
uvicorn server:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000
```

Drag-and-drop upload, batch translate multiple pages, download results.

API: `POST /translate` with image file → returns translated image.

## Docker

```bash
docker build -t manga-translator .
docker run -p 8000:8000 -v /path/to/models:/models manga-translator
```

## Architecture

```
Input → YOLOv8 (bubble detection) → MangaOCR (text extraction) → Ollama/Qwen (translation) → LaMa+OpenCV (typesetting)
```

## Stack

| Stage | Model | Notes |
|---|---|---|
| Detection | YOLOv8 comic-speech-bubble-detector | text_bubble + text_free classes |
| OCR | MangaOCR | Specialized Japanese manga OCR |
| Translation | Qwen3.5 9B via Ollama | Current best at 7-9B tier for JP |
| Inpainting | LaMa (lama-cleaner) + OpenCV | Dual strategy per bubble type |
| Typesetting | PIL + Anime Ace font | Dynamic sizing, outline, hyphenation |

### Alternative: LiquidAI LFM2

For CPU-only or low-resource deployment, [LFM2-350M-ENJP-MT](https://huggingface.co/LiquidAI/LFM2-350M-ENJP-MT) (350MB) delivers comparable JP-EN quality at 10x smaller size. Also see [LFM2.5-1.2B-JP](https://huggingface.co/LiquidAI) for best-in-class 1B-scale Japanese.

## Known Issues

- LaMa inpainting uses `lama-cleaner` >= 0.31; API changed — see imports in MangaTranslator.py
- PaddleOCR-VL can replace MangaOCR but requires transformers==4.55.0 and specific rope config
- First run downloads big-lama.pt (196MB) and MangaOCR weights automatically

## Future Work

- [ ] Multi-GPU distributed processing
- [ ] Fine-tuned translation model on manga corpus (LFM2 + Manga109Dialog)
- [x] ~~Web UI~~ — Gradio demo (app.py) + FastAPI server (server.py)
- [ ] Support for webtoon/manhwa formats
- [ ] Quality assessment metrics
- [x] ~~FastAPI + deploy~~ — server.py + Dockerfile + index.html

## License

MIT. For educational and personal use. Respect copyright.
