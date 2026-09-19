"""Setup smoke test — verifies assets + LLM endpoint, then translates the
bundled sample page (013_og.jpg) end to end.

Usage:
    python scripts/smoke.py [--llm http://localhost:8110] [--page 013_og.jpg]

Writes into smoke_out/: translated.jpg (+ comparison.png).
Exit codes: 0 = ok, 1 = missing asset, 2 = LLM endpoint problem.
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(description="Manga pipeline setup smoke test")
    ap.add_argument("--llm", default="http://localhost:8110",
                    help="OpenAI-compatible base URL (default: local llama.cpp)")
    ap.add_argument("--page", default="013_og.jpg", help="sample page to translate")
    ap.add_argument("--yolo", default=None, help="override detector weights path")
    ap.add_argument("--font", default=None, help="override font path")
    args = ap.parse_args()

    # Asset locations: CLI flag > environment (same vars as the server) > project root
    yolo = (Path(args.yolo) if args.yolo
            else Path(os.environ.get("YOLO_MODEL", str(ROOT / "comic-speech-bubble-detector.pt"))))
    if args.font:
        font = Path(args.font)
    elif os.environ.get("FONT_PATH"):
        font = Path(os.environ["FONT_PATH"])
    else:
        font = next((p for p in (ROOT / "animeace2_reg.ttf", ROOT / "animeace2_reg.otf") if p.exists()), None)
    page = Path(args.page) if Path(args.page).is_absolute() else ROOT / args.page

    problems = []
    if not yolo.exists():
        problems.append(f"missing detector weights: {yolo} (README quickstart step 3)")
    if not font:
        problems.append("missing font animeace2_reg.ttf/.otf in the project root "
                        "(README quickstart step 4, or use --font)")
    if not page.exists():
        problems.append(f"missing sample page: {page}")
    if problems:
        print("\n".join("\u2717 " + p for p in problems))
        sys.exit(1)

    import requests
    base = args.llm.rstrip("/")
    try:
        r = requests.get(f"{base}/health", timeout=5)
        if r.status_code == 503:
            print(f"~ LLM at {base} is still loading its model — wait a bit and rerun")
            sys.exit(2)
        if r.status_code == 404:  # hosted APIs have no /health; probe /v1/models
            r2 = requests.get(f"{base}/v1/models", timeout=5)
            if r2.status_code != 200:
                print(f"\u2717 LLM at {base}: /v1/models -> HTTP {r2.status_code} "
                      f"(bad URL or missing API key)")
                sys.exit(2)
        elif r.status_code != 200:
            print(f"\u2717 LLM at {base}: /health -> HTTP {r.status_code}")
            sys.exit(2)
    except requests.RequestException as exc:
        print(f"\u2717 LLM at {base} not reachable: {exc}")
        print("  start llama-server / Ollama first, or pass --llm <base-url>")
        sys.exit(2)
    print(f"\u2713 assets present, LLM at {base} answers")

    from MangaTranslator import MangaTranslator

    t = MangaTranslator(yolo_model_path=str(yolo), font_path=str(font),
                        llm_base_url=base, debug=False)
    out = ROOT / "smoke_out"
    img, data = t.detect_and_process(str(page), output_dir=str(out), page_id="p001")
    data = t.run_ocr(data)
    data = t.translate_batch(data)
    final_img = t.typeset(img, data, str(out / "translated.jpg"))
    t.save_comparison(img, data, str(out / "comparison.png"), translated_panel=final_img)
    done = sum(1 for e in data if e.get("translated_text"))
    print(f"\u2713 done: {done}/{len(data)} bubbles translated "
          f"-> smoke_out/translated.jpg + comparison.png")
    if done < len(data):
        print("  (some bubbles were left in the original text — see the web UI / logs)")


if __name__ == "__main__":
    main()
