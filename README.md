# Multi-Modal Manga Translation Pipeline

End-to-end ML pipeline that detects, extracts, translates, and typesets Japanese manga — fully local, no paid APIs.

Page image → YOLOv8 bubble detection → MangaOCR → local LLM translation → LaMa/OpenCV cleanup → typesetting.

![Not translated | Translated | Translated + LaMa inpainting](013_comparison.png)

*Left: the original page. Middle: translated with the simple blur cleanup. Right: translated with the OpenCV + LaMa hybrid cleanup — the current default output.*

## How it works

```
┌────────┐   ┌───────────────────┐   ┌──────────┐   ┌──────────────────────┐   ┌────────────────┐   ┌───────────┐
│  Page  │──▶│ YOLOv8 detector   │──▶│ MangaOCR │──▶│ Local LLM            │──▶│ Cleanup:       │──▶│ Typeset   │
│  image │   │ text_bubble +     │   │  (JP     │   │ (llama.cpp server,   │   │ OpenCV bubbles │   │ (PIL,     │
│        │   │ text_free regions │   │  text)   │   │  e.g. Gemma 26B-A4B) │   │ + LaMa free    │   │ auto-size)│
└────────┘   └───────────────────┘   └──────────┘   └──────────────────────┘   │ text           │   └───────────┘
                                                                               └────────────────┘
```

- **Detection** — YOLOv8 (`ogkalu/comic-speech-bubble-detector-yolov8m`) labels each region `text_bubble` or `text_free`, sorted in manga reading order (top-to-bottom rows, right-to-left).
- **OCR** — MangaOCR extracts the Japanese text of each crop.
- **Translation** — pages are translated in batches (shared context for consistency). The LLM gets series context (title/tags/description), a custom names map, and returns strict JSON with one translation per bubble. Fully local via any OpenAI-compatible server (llama.cpp).
- **Cleanup** — hybrid: OpenCV inpainting inside speech bubbles (preserves tails and borders), LaMa inpainting over free text (redraws the background).
- **Typesetting** — dynamic font sizing per bubble, hyphenated wrapping, white outline for readability.

## Stack

| Stage | Tool | Notes |
|---|---|---|
| Detection | ultralytics YOLOv8m — `ogkalu/comic-speech-bubble-detector-yolov8m` | ~52 MB, runs on CPU or GPU |
| OCR | `manga-ocr` (kha-white) | ~410 MB weights, CUDA/CPU |
| Translation | local llama.cpp server (OpenAI-compatible API) | Gemma 4 26B-A4B by default; Qwen3.6 35B-A3B also verified |
| Inpainting | `simple-lama-inpainting` (big-lama) + OpenCV | ~196 MB weights, CUDA/CPU |
| Typesetting | PIL + a comic lettering font (Anime Ace 2.0 recommended) | |

## Quickstart

Python **3.10 – 3.12**, Linux / macOS / WSL2 (Windows-native works too). Roughly **10 minutes** end-to-end on a fast connection — the PyTorch wheel is the long pole (a few GB).

```bash
# 1. clone + venv
git clone https://github.com/ArsVie/Multi-Modal-Manga-Translation-Pipeline
cd Multi-Modal-Manga-Translation-Pipeline
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2. dependencies — install torch FIRST, matching your driver.
#    Pick the right index URL at pytorch.org/get-started/locally (CPU-only: skip --index-url)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt

# 3. detector weights (~52 MB)
pip install -U huggingface_hub                          # provides the `hf` CLI
hf download ogkalu/comic-speech-bubble-detector-yolov8m comic-speech-bubble-detector.pt --local-dir .

# 4. lettering font — any comic-style .ttf/.otf, saved as animeace2_reg.ttf in the
#    project root (Anime Ace 2.0 from blambot.com is what the samples use),
#    or point FONT_PATH= at your own file

# 5. an LLM endpoint — ANY OpenAI-compatible server:
#    Ollama already running?   ollama pull gemma3:27b
#        → base URL http://localhost:11434, model name "gemma3:27b" in the web UI
#    llama.cpp?                see "Local LLM server" below (the tuned Gemma 4 26B recipe)
#    hosted API?               paste the base URL, API key and model name in the web UI
#    sanity check: llama.cpp serves /health (200 ready, 503 still loading);
#    anything else is verified through /v1/models automatically

# 6. run
uvicorn server:app --host 0.0.0.0 --port 8000
# open http://localhost:8000, drop pages / a folder / a .zip, hit Translate

# 7. optional: verify the whole setup in one command (translates the bundled sample page)
python scripts/smoke.py
```

First run auto-downloads MangaOCR (~410 MB) and big-lama (~196 MB). Total pipeline footprint ≈ 5 GB + your LLM weights. A CUDA GPU is recommended; CPU-only works — the device auto-detects.

## Local LLM server

Translation targets **any OpenAI-compatible endpoint** — llama.cpp, Ollama, LM Studio, vLLM, or a hosted API (paste its base URL + API key + model name in the web UI). The reference configuration this repo is tuned on: Gemma 4 26B-A4B served by llama.cpp (`llama-server`), reasoning off —

```bash
llama-server \
  -m /path/to/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf \
  --mtp-head /path/to/gemma-4-26B-A4B-it-assistant.Q4_K_M.gguf \
  --spec-type mtp --draft-block-size 3 --draft-max 8 --draft-min 0 \
  --cpu-moe -ngl 99 -ngld 99 -fa on -ctk q8_0 -ctv q8_0 -c 32000 --no-mmap \
  --reasoning off --jinja --host 127.0.0.1 --port 8110
```

Qwen3.6-35B-A3B runs the same way (any port; pass it via `llm_base_url`). Check readiness with `curl http://localhost:8110/health` — `200` means the model is loaded, `503` means it is still loading.

Measured on an RTX PRO 1000 8 GB + DDR5 (Gemma 26B-A4B Q4_K_M + MTP): ~17–29 tokens/s generation; a typical page (~250 output tokens) translates in ~10–20 s.

## Usage

### Python

```python
from MangaTranslator import MangaTranslator

t = MangaTranslator(
    yolo_model_path="comic-speech-bubble-detector.pt",
    font_path="animeace2_reg.ttf",
    llm_base_url="http://localhost:8110",
    custom_translations={"ルーグ": "Lugh"},  # names map, applied before translation
)

t.process_chapter(
    input_folder="raw_chapter/",
    output_folder="translated_chapter/",
    series_info={
        "title": "Sekai Saikou no Ansatsusha",
        "tags": "Action, Fantasy, Romance",
        "description": "One-paragraph plot summary — helps tone and name consistency.",
    },
    batch_size=4,           # pages translated together in one context
    save_comparisons=True,  # also write <page>_comparison.png per page
)
```

### Web UI

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
# open http://localhost:8000   (any free port works — 8000 is taken on some setups)
```

Drop pages, folders or a `.zip` in (or paste with Ctrl+V), hit Translate, download results (pages appear as they finish; **Stop** cancels a run and keeps whatever finished). After a translation the layout becomes a comparison workspace: **originals on the left** as full-width pages you scroll through, and a **single-page translated viewer on the right** (sticky, fitted to the screen) with ◀ ▶ buttons or ← → arrow keys to flip pages. Queues show the page count and an estimated time (a per-page average learned from your own runs; .zip page counts are read from the zip directory). The UI wears 4chan's "Tomorrow" theme. **Translation can run against any OpenAI-compatible endpoint** — the local llama.cpp server by default, or e.g. OpenAI / OpenRouter via the LLM preset, with the API key and model name set in the settings (detection, OCR and lettering always stay local). Advanced settings (persisted in the browser) cover:

- **Custom names map** — one `jp=English` per line (or comma-separated), applied to OCR output before translation
- **Series context** — title / tags / description for tone and consistency
- **LLM server** — pick the Gemma or Qwen endpoint, or enter a custom URL
- **Lettering size** — 1.0 = default, lower shrinks the English lettering
- **Batch size, detection confidence, keep-honorifics, comparison sheets**

`POST /translate` (multipart: `files`, plus the same fields) streams NDJSON for scripting: one `{"type": "page", "filename", "translated_b64", "comparison_b64"}` line per finished page, then `{"type": "done", "cancelled"}` (or `{"type": "error", "message"}`). `POST /cancel` stops the running job at its next page boundary.

## Stage comparison

`save_comparisons=True` (or the web toggle) writes `<page>_comparison.png`:

1. **Not translated** — the original page
2. **Translated** — legacy cleanup (Gaussian blur per bubble) + typeset
3. **Translated + LaMa inpainting** — OpenCV + LaMa hybrid cleanup + typeset (the default output)

It is a quick visual regression check for cleanup quality — see the sheet at the top of this README.

## Configuration

| Parameter | Default | Meaning |
|---|---|---|
| `yolo_model_path` | `comic-speech-bubble-detector.pt` | YOLOv8 weights |
| `llm_base_url` / `llm_model` | `http://localhost:8110` / `local` | LLM endpoint |
| `font_path` | `animeace2_reg.ttf` | typesetting font |
| `custom_translations` | `{}` | JP→EN names/terms map |
| `keep_honorifics` | `False` | re-attach -san / -chan / … when the source has them |
| `device` | auto | `cuda` / `cpu` |
| `debug` | `True` | write `chapter_data.json` next to the output |
| `process_chapter(conf_threshold=)` | `0.15` | YOLO confidence threshold |
| `process_chapter(batch_size=)` | `4` | pages per translation batch |

## Troubleshooting

- **"LLM server ... is not reachable"** — start llama-server first; `curl http://localhost:8110/health` (503 = still loading, 200 = ready).
- **No bubbles detected** — lower `conf_threshold` (e.g. `0.10`).
- **CUDA out of memory** — pass `device="cpu"`, or free VRAM (the LLM server holds ~2.5 GB).
- **Awkward line breaks** — words are kept whole and the font shrinks instead of hyphenating; syllable hyphenation and hard cuts are only last resorts, and names from the custom map are never split. Adjust the size range in `_calculate_optimal_font_size` / `_wrap_text_dynamic`.
- **Some sound effects stay Japanese** — the detector only finds bubbles and free text it was trained on; SFX are frequently missed. Expected.

## Known limitations

- The detector was trained on ~8k manga/webtoon/comic pages — unusual layouts can be missed.
- One page group is one LLM call; very large batches can exceed the server context window (`-c`).
- No automatic SFX translation.

## Credits

- Bubble detector: [ogkalu/comic-speech-bubble-detector-yolov8m](https://huggingface.co/ogkalu/comic-speech-bubble-detector-yolov8m) (trainer: ogkalu; stack: ultralytics YOLOv8)
- OCR: [kha-white/manga-ocr](https://github.com/kha-white/manga-ocr)
- Inpainting: [simple-lama-inpainting](https://github.com/enesmsahin/simple-lama-inpainting) (LaMa)
- Font: Anime Ace 2.0 by Blambot (not bundled — available from Blambot)
- Local LLM runtime: [llama.cpp](https://github.com/ggml-org/llama.cpp)

## License

MIT. For educational and personal use. Respect copyright.
