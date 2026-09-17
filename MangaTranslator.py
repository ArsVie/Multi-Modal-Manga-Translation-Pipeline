import os
import json
import cv2
import numpy as np
import pyphen
import re
import requests
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from manga_ocr import MangaOcr
from simple_lama_inpainting import SimpleLama

class MangaTranslator:
    # Below this fraction of "edge ink" a text_bubble region is treated as a
    # rectangular text box (title cards etc.) and cleaned with a full-region
    # mask instead of an ellipse mask (an ellipse leaves the corners behind).
    BOX_EDGE_THRESHOLD = 0.05

    def __init__(self, yolo_model_path='comic-speech-bubble-detector.pt',
                 llm_base_url="http://localhost:8110", llm_model="local",
                 font_path="animeace2_reg.ttf", custom_translations=None,
                 keep_honorifics=False, debug=True, device=None):
        """
        Initialize models.

        Args:
            yolo_model_path: Path to the YOLOv8 speech-bubble detector (.pt).
            llm_base_url: Base URL of a local OpenAI-compatible LLM server
                (llama.cpp). Default: the Gemma server on port 8110.
            llm_model: Model name sent to the LLM server (single-model llama.cpp
                servers ignore it).
            font_path: TrueType/OpenType font used for typesetting.
            custom_translations: Dictionary of Japanese terms -> English equivalents.
            keep_honorifics: Re-attach romaji honorifics (-san, -chan, ...)
                when the source text contains them.
            debug: Save chapter_data.json next to the translated output.
            device: 'cuda', 'cpu' or None to auto-detect.
        """
        print("Loading YOLO model...")
        self.yolo_model = YOLO(yolo_model_path)
        self.font_path = font_path

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading LaMa Inpainting model... (device={self.device})")
        self.lama = SimpleLama(device=torch.device(self.device))

        print("Loading MangaOCR model...")
        self.mocr = MangaOcr(force_cpu=(self.device == "cpu"))

        print("Initializing LLM client...")
        self.llm_base_url = llm_base_url.rstrip('/')
        self.llm_model = llm_model
        self.llm_timeout = 600  # seconds; the first call may load the model server-side
        self.dic = pyphen.Pyphen(lang='en')

        # Font cache for performance
        self.font_cache = {}

        # Custom translation dictionary
        self.custom_translations = custom_translations or {}

        self.keep_honorifics = keep_honorifics
        self.debug = debug
        self.honorifics = ['san', 'chan', 'kun', 'sama', 'senpai', 'sensei', 'dono', 'tan']

        # For romanization fallback
        try:
            import pykakasi
            self.kakasi = pykakasi.kakasi()
        except ImportError:
            print("Warning: pykakasi not installed. Install with 'pip install pykakasi' for romanization support.")
            self.kakasi = None

    def set_llm(self, base_url=None, model=None):
        """Update the local LLM endpoint (cheap; no models are reloaded)."""
        if base_url:
            self.llm_base_url = base_url.rstrip('/')
        if model:
            self.llm_model = model

    def _chat(self, user_message, system_message=None, temperature=None):
        """Send one chat completion request to the local OpenAI-compatible server."""
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_message})

        response = requests.post(
            f"{self.llm_base_url}/v1/chat/completions",
            json={
                "model": self.llm_model,
                "messages": messages,
                "temperature": 0.3 if temperature is None else temperature,
                "stream": False,
            },
            timeout=self.llm_timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def _get_font(self, size):
        """Cache fonts to avoid repeated loading"""
        if size not in self.font_cache:
            try:
                self.font_cache[size] = ImageFont.truetype(self.font_path, size)
            except IOError:
                self.font_cache[size] = ImageFont.load_default()
        return self.font_cache[size]

    def _sort_bubbles(self, detections, row_threshold=50):
        """Sort detections in manga reading order: top-to-bottom rows,
        right-to-left within each row. `detections` is a list of dicts with a
        'bbox' key ([x1, y1, x2, y2])."""
        def y(entry): return entry['bbox'][1]
        def x(entry): return entry['bbox'][0]

        detections.sort(key=y)
        sorted_bubbles = []
        if not detections:
            return sorted_bubbles

        current_row = [detections[0]]
        for i in range(1, len(detections)):
            if abs(y(detections[i]) - y(current_row[-1])) < row_threshold:
                current_row.append(detections[i])
            else:
                current_row.sort(key=x, reverse=True)
                sorted_bubbles.extend(current_row)
                current_row = [detections[i]]

        current_row.sort(key=x, reverse=True)
        sorted_bubbles.extend(current_row)
        return sorted_bubbles

    def _wrap_text_dynamic(self, text, font, max_width, allow_hard_break=True):
        """Wrap text to `max_width`, hyphenating words that do not fit.

        Splits use pyphen syllables, picking the longest prefix that fits a
        line. Returns None when a word would need a hard (non-syllable) break
        and `allow_hard_break` is False - callers use that to prefer a smaller
        font over ugly breaks.
        """
        lines = []
        current = ""

        def fits(s):
            return font.getlength(s) <= max_width

        for word in text.split():
            candidate = (current + " " + word).strip()
            if fits(candidate):
                current = candidate
                continue
            if current:
                lines.append(current)
                current = ""

            if fits(word):
                current = word
                continue

            # The word does not fit on a line of its own: split it up.
            pieces = []
            rest = word
            while not fits(rest):
                splits = [(a, b) for a, b in self.dic.iterate(rest) if fits(a + "-")]
                if splits:
                    start, end = max(splits, key=lambda p: len(p[0]))
                    pieces.append(start + "-")
                    rest = end
                else:
                    if not allow_hard_break:
                        return None
                    cut = 1
                    while cut < len(rest) - 1 and fits(rest[:cut + 1] + "-"):
                        cut += 1
                    pieces.append(rest[:cut] + "-")
                    rest = rest[cut:]
            pieces.append(rest)
            lines.extend(pieces[:-1])
            current = pieces[-1]

        if current:
            lines.append(current)
        return "\n".join(lines)

    def _smart_clean_bubble(self, img, bbox):
        """
        Gaussian blur-based cleaning for transparent effect
        """
        x1, y1, x2, y2 = bbox

        # Ensure coordinates are within image bounds
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return img

        # Extract bubble region
        bubble_region = img[y1:y2, x1:x2].copy()

        if bubble_region.size == 0:
            return img

        # Apply Gaussian blur for softer look
        blurred = cv2.GaussianBlur(bubble_region, (21, 21), 0)

        # Brighten the blurred region slightly
        brightened = cv2.addWeighted(blurred, 0.7,
                                     np.ones_like(blurred) * 255, 0.3, 0)

        # Place back into image
        img[y1:y2, x1:x2] = brightened

        return img

    def _preserve_honorifics(self, original_text, translated_text):
        """
        Detect and preserve Japanese honorifics in romaji form.
        Examples: さん→-san, ちゃん→-chan, 君→-kun, 様→-sama
        """
        if not self.keep_honorifics or not self.kakasi:
            return translated_text

        # Common honorific patterns in Japanese
        honorific_map = {
            'さん': '-san',
            'ちゃん': '-chan',
            'くん': '-kun',
            '君': '-kun',
            '様': '-sama',
            'さま': '-sama',
            '先輩': '-senpai',
            'せんぱい': '-senpai',
            '先生': '-sensei',
            'せんせい': '-sensei',
            '殿': '-dono',
            'どの': '-dono',
            'たん': '-tan',
        }

        # Find honorifics in original text
        found_honorifics = []
        for jp_hon, rom_hon in honorific_map.items():
            if jp_hon in original_text:
                found_honorifics.append(rom_hon)

        # If we found honorifics, try to add them back to names in translation
        if found_honorifics:
            # Split into words and check last word for potential name
            words = translated_text.split()
            if len(words) >= 1:
                # Check if translation already has honorific
                last_word = words[-1].lower()
                has_honorific = any(hon.strip('-') in last_word for hon in self.honorifics)

                if not has_honorific and found_honorifics:
                    # Add the first found honorific to what's likely a name.
                    # Look for capitalized words (likely names) and keep any
                    # trailing punctuation in place: "Lugh," -> "Lugh-san,"
                    for i in range(len(words) - 1, -1, -1):
                        match = re.match(r"^([A-Z][\w'-]*)([,.;:!?…]*)$", words[i])
                        if match:
                            words[i] = f"{match.group(1)}{found_honorifics[0]}{match.group(2)}"
                            translated_text = ' '.join(words)
                            break

        return translated_text

    def _draw_text_with_outline(self, draw, position, text, font,
                                 text_color="black", outline_color="white",
                                 outline_width=2, **kwargs):
        """
        Draw text with outline for better readability
        """
        x, y = position
        # Draw outline
        for adj_x in range(-outline_width, outline_width + 1):
            for adj_y in range(-outline_width, outline_width + 1):
                if adj_x != 0 or adj_y != 0:
                    draw.multiline_text((x + adj_x, y + adj_y), text,
                                       fill=outline_color, font=font, **kwargs)
        # Draw main text
        draw.multiline_text(position, text, fill=text_color, font=font, **kwargs)

    def _sanitize_for_font(self, text):
        """Replace punctuation the lettering font lacks (Anime Ace draws a
        missing-glyph box for em dashes and smart quotes)."""
        replacements = {
            "\u2014": "--",  # em dash
            "\u2013": "-",   # en dash
            "\u2015": "--",  # horizontal bar
            "\u2026": "...", # ellipsis
            "\u201c": '"', "\u201d": '"', "\u301d": '"', "\u301e": '"',
            "\u2018": "'", "\u2019": "'",
        }
        for bad, good in replacements.items():
            text = text.replace(bad, good)
        return text

    def _calculate_optimal_font_size(self, text, bbox, min_size=12, max_size=36):
        x1, y1, x2, y2 = bbox
        box_width = x2 - x1
        box_height = y2 - y1
        text = self._sanitize_for_font(text)

        # Tall regions = vertical text turned sideways. English fits a tall
        # column badly with the default cap, so raise the size cap and wrap to
        # nearly the full width: bigger type fills the height instead of
        # pooling in a narrow strip.
        is_vertical = box_height > (box_width * 1.5)
        if is_vertical:
            max_size = max(max_size, min(96, int(max(box_width, box_height) / 3)))
            target_width_ratio = 0.95
        else:
            target_width_ratio = 0.9

        def try_wrap(size, allow_hard_break):
            font = self._get_font(size)
            wrapped = self._wrap_text_dynamic(
                text, font, int(box_width * target_width_ratio),
                allow_hard_break=allow_hard_break,
            )
            if wrapped is None:
                return None
            temp_draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
            left, top, right, bottom = temp_draw.multiline_textbbox(
                (0, 0), wrapped, font=font, align="center"
            )
            if (bottom - top) <= (box_height - 8) and (right - left) <= (box_width - 4):
                return wrapped
            return None

        # Prefer the largest size without hard breaks; allow them as a fallback
        for allow_hard_break in (False, True):
            for size in range(max_size, min_size - 1, -1):
                wrapped = try_wrap(size, allow_hard_break)
                if wrapped is not None:
                    return size, wrapped

        # Fallback: minimum size
        font = self._get_font(min_size)
        wrapped = self._wrap_text_dynamic(text, font, int(box_width * target_width_ratio))
        return min_size, wrapped

    def _has_japanese_characters(self, text):
        """Check if text contains Japanese characters"""
        japanese_ranges = [
            (0x3040, 0x309F),  # Hiragana
            (0x30A0, 0x30FF),  # Katakana
            (0x4E00, 0x9FFF),  # Kanji
        ]
        for char in text:
            code = ord(char)
            for start, end in japanese_ranges:
                if start <= code <= end:
                    return True
        return False

    def _romanize_japanese(self, text):
        """Convert Japanese text to romaji"""
        if not self.kakasi:
            return text

        try:
            result = self.kakasi.convert(text)
            return ''.join([item['hepburn'] for item in result])
        except Exception as e:
            print(f"    Romanization error: {e}")
            return text

    def _apply_custom_translations(self, text):
        """Apply custom character name translations"""
        for jp_term, en_term in self.custom_translations.items():
            text = text.replace(jp_term, en_term)
        return text

    @staticmethod
    def _overlap_ratio(box_a, box_b):
        """Intersection over the smaller area of two boxes."""
        ix = max(0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
        iy = max(0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
        inter = ix * iy
        area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
        area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
        smaller = min(area_a, area_b)
        return inter / smaller if smaller else 0.0

    def _merge_duplicate_detections(self, detections, overlap_thresh=0.6):
        """Drop duplicate detections of the same text block.

        The model can fire both heads (text_bubble and text_free) or twice on
        one region, and labels flicker between runs. Overlapping boxes merge
        into the larger one; a merged pair containing a text_free detection is
        cleaned as free text so the block is OCR'd, translated and drawn once.
        """
        def area(box):
            return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

        kept = []
        for det in sorted(detections, key=lambda d: area(d['bbox']), reverse=True):
            duplicate = next(
                (k for k in kept
                 if self._overlap_ratio(det['bbox'], k['bbox']) > overlap_thresh),
                None,
            )
            if duplicate is None:
                kept.append(det)
            elif det['label'] == 'text_free':
                # Free text wins: LaMa cleans the whole block (incl. corners)
                duplicate['label'] = 'text_free'
        return kept

    def _edge_ink_ratio(self, image, bbox, band=5):
        """Fraction of dark ink pixels in the outer band of a region.

        High = an outline (bubble border, panel line) runs along the crop edge.
        Near zero = clean text area (free text or a boxed title).
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ink = ink > 0
        ch, cw = ink.shape
        band = max(1, min(band, ch // 4, cw // 4))
        return float(max(
            ink[:band, :].mean(),
            ink[-band:, :].mean(),
            ink[:, :band].mean(),
            ink[:, -band:].mean(),
        ))

    def detect_and_process(self, image_path, output_dir="crops", page_id="",
                           conf_threshold=0.15, imgsz=640):
            image = cv2.imread(image_path)
            if image is None: raise ValueError(f"Not found: {image_path}")

            # 1. Run Prediction
            results = self.yolo_model.predict(source=image, conf=conf_threshold,
                                              imgsz=imgsz, save=False, verbose=False)

            # Get the class names dictionary (e.g., {0: 'text', 1: 'bubble'})
            class_names = results[0].names

            # 2. Extract Boxes AND Classes
            detections = []
            for box in results[0].boxes:
                xyxy = list(map(int, box.xyxy[0].tolist()))
                cls_id = int(box.cls[0])
                label = class_names[cls_id] # e.g., "text" or "bubble" or "face"

                # Filter: We only care about text/bubbles, not faces/bodies if your model detects them
                if label in ['face', 'body']: continue

                detections.append({
                    "bbox": xyxy,
                    "label": label
                })

            # Merge duplicate detections of the same text block
            detections = self._merge_duplicate_detections(detections)

            # Decide how each region gets cleaned:
            #   bubble -> outline touches the crop edge: ellipse mask (keeps tails)
            #   box    -> clean rectangular text region (title cards etc.)
            #   free   -> free-floating text: LaMa rectangle
            for det in detections:
                edge = self._edge_ink_ratio(image, det['bbox'])
                det['edge_ink'] = round(float(edge), 4)
                if det['label'] == 'text_free':
                    det['shape'] = 'free'
                else:
                    det['shape'] = 'box' if edge < self.BOX_EDGE_THRESHOLD else 'bubble'

            # Sort in manga reading order (top-to-bottom rows, right-to-left)
            detections = self._sort_bubbles(detections)

            if not os.path.exists(output_dir): os.makedirs(output_dir)

            manga_data = []
            for i, det in enumerate(detections):
                x_min, y_min, x_max, y_max = det['bbox']
                
                # ... (Cropping logic stays the same) ...
                crop = image[y_min:y_max, x_min:x_max]
                
                # Save crop
                crop_filename = f"bubble_{page_id}_{i+1}.png"
                crop_path = os.path.join(output_dir, crop_filename)
                cv2.imwrite(crop_path, crop)

                manga_data.append({
                    "id": f"{page_id}_{i+1}",
                    "page_id": page_id,
                    "bbox": [x_min, y_min, x_max, y_max],
                    "label": det['label'],
                    "shape": det.get('shape', 'bubble'),
                    "edge_ink": det.get('edge_ink'),
                    "crop_path": crop_path,
                    "original_text": "",
                    "translated_text": ""
                })
                
            return image, manga_data

    def run_ocr(self, manga_data, upscale_small_crops=True):
        for entry in manga_data:
            crop_path = entry['crop_path']
            crop = cv2.imread(crop_path)
            if crop is None:
                entry['original_text'] = ""
                continue

            # The OCR model reads small crops better when upscaled
            if upscale_small_crops and max(crop.shape[:2]) < 300:
                crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

            japanese_text = self.mocr(
                Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            )

            # Apply custom translations to original text
            japanese_text = self._apply_custom_translations(japanese_text)

            entry['original_text'] = japanese_text.replace('\n', '')
        return manga_data

    def _translate_single_bubble(self, text, series_info=None):
        """Translate a single bubble (fallback method)"""
        context_str = ""
        if series_info:
            context_str = f"""
Context: {series_info.get('title', '')} - {series_info.get('tags', '')}
"""

        prompt = f"""{context_str}Translate this Japanese manga text to natural English. Return ONLY the English translation, nothing else:
{text}"""

        try:
            translation = self._chat(prompt).strip()

            # Remove common wrapper phrases
            translation = re.sub(r'^(Here\'s the translation:|Translation:|English:)\s*', '', translation, flags=re.IGNORECASE)
            translation = translation.strip('"\'')

            return translation
        except Exception as e:
            print(f"    Translation error: {e}")
            return "[Translation Error]"

    def translate_batch(self, manga_data, series_info=None):
        valid_entries = [e for e in manga_data if e['original_text'].strip()]
        if not valid_entries:
            return manga_data

        input_payload = [
            {"bubble_id": e["id"], "text": e["original_text"]}
            for e in valid_entries
        ]

        json_string = json.dumps(input_payload, ensure_ascii=False)

        context_str = ""
        if series_info:
            context_str = f"""
SERIES CONTEXT:
Title: {series_info.get('title', 'Unknown')}
Tags/Genre: {series_info.get('tags', 'Unknown')}
Description: {series_info.get('description', 'None')}
"""

        system_prompt = (
            "You are a professional manga translator."
            f"{context_str}"
            "Translate the following Japanese sentences into natural English."
            "Use the Series Context to determine tone, slang, and character voices. "
            "Maintain dialogue consistency across pages. "
            "\n\nCRITICAL: Return ONLY a valid JSON array. No explanations, no markdown, no code blocks. "
            'Format: [{{"bubble_id": "p001_1", "translation": "English text here"}}, ...]\n'
            "Your entire response must be parseable JSON."
        )

        max_retries = 3
        content = ""
        try:
            content = self._chat(
                f"Translate this JSON array:\n{json_string}",
                system_message=system_prompt,
            ).strip()

            # More aggressive JSON extraction
            if "```" in content:
                parts = content.split("```")
                for part in parts:
                    part = part.strip()
                    if part.startswith("json"):
                        part = part[4:].strip()
                    if part.startswith("[") and part.endswith("]"):
                        content = part
                        break

            # Find first [ and last ]
            start_idx = content.find("[")
            end_idx = content.rfind("]")

            if start_idx != -1 and end_idx != -1:
                content = content[start_idx:end_idx+1]

            translated_list = json.loads(content)

            if not isinstance(translated_list, list):
                raise ValueError("Response is not a list")

            translation_map = {item['bubble_id']: item['translation'] for item in translated_list}

            for entry in manga_data:
                if entry['id'] in translation_map:
                    translation = translation_map[entry['id']]

                    # Check if translation contains Japanese characters
                    if self._has_japanese_characters(translation):
                        print(f"    ⚠ Translation for {entry['id']} contains Japanese. Retrying (Max {max_retries})...")

                        success = False
                        for attempt in range(max_retries):
                            retry_text = self._translate_single_bubble(
                                entry['original_text'], series_info
                            )

                            # Check if the retry fixed it
                            if not self._has_japanese_characters(retry_text):
                                translation = retry_text
                                success = True
                                print(f"      ✓ Fixed on attempt {attempt + 1}")
                                break
                            else:
                                print(f"      ✗ Attempt {attempt + 1} failed")

                        # Fallback: Romanize if all retries failed
                        if not success:
                            print(f"    ⚠ All retries failed. Romanizing...")
                            translation = self._romanize_japanese(translation)

                    entry['translated_text'] = translation

                # Handle missing translations (fallback to single mode)
                elif entry.get('original_text') and not entry.get('translated_text'):
                     entry['translated_text'] = self._translate_single_bubble(entry['original_text'], series_info)

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"  ⚠ Translation error: {e}")
            if content:
                print(f"  Raw response: {content[:500]}...")
            print(f"  Falling back to individual translations...")

            # Fallback: translate one by one
            for entry in valid_entries:
                try:
                    translation = self._translate_single_bubble(
                        entry['original_text'], series_info
                    )

                    # Check for Japanese in translation
                    if self._has_japanese_characters(translation):
                        print(f"    ⚠ Translation for {entry['id']} has Japanese, romanizing...")
                        translation = self._romanize_japanese(translation)

                    entry['translated_text'] = translation
                except Exception as e2:
                    print(f"    Failed to translate {entry['id']}: {e2}")
                    entry['translated_text'] = "[Translation Error]"

        # Optionally re-attach romaji honorifics to the finished translations
        if self.keep_honorifics:
            for entry in manga_data:
                if entry.get('translated_text') and entry.get('original_text'):
                    entry['translated_text'] = self._preserve_honorifics(
                        entry['original_text'], entry['translated_text']
                    )

        return manga_data



    def clean_page(self, original_image, page_data, inpaint_radius=5):
            """
            Strict Hybrid Cleaning:
            - bubble: OpenCV inpainting of text ink that is NOT connected to the
              crop border (outlines and tails are preserved, corner glyphs are
              cleaned too)
            - box:    same, plus a 3px rim so rectangular borders survive
            - free:   LaMa inpainting on a rectangle mask (redraws background)
            """
            final_image = original_image.copy()
            h, w = original_image.shape[:2]

            # Mask for LaMa (Accumulates all 'text_free' areas)
            lama_mask = np.zeros((h, w), dtype=np.uint8)
            has_lama_work = False

            for entry in page_data:
                # Skip if no translation (optional, but good for speed)
                if not entry.get('translated_text'): continue

                bbox = entry['bbox']
                label = entry.get('label', 'text_free')
                shape = entry.get('shape') or ('free' if label == 'text_free' else 'bubble')

                x1, y1, x2, y2 = bbox

                # Clamp coordinates
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                # Extract crop for analysis
                crop = final_image[y1:y2, x1:x2]
                if crop.size == 0: continue

                gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

                # --- STRATEGY 1: CLEAN TEXT INK NOT CONNECTED TO THE CROP BORDER ---
                if shape in ('bubble', 'box'):
                    ch, cw = crop.shape[:2]

                    # A. Find the text pixels (dark ink)
                    binary_text = cv2.adaptiveThreshold(
                        gray_crop, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                        cv2.THRESH_BINARY_INV, 21, 10
                    )

                    # B. Keep ink connected to the crop border: that is bubble
                    # outline / tails / panel lines. Everything else is text
                    # (this also cleans corner glyphs an ellipse mask missed).
                    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary_text, connectivity=8)
                    cleanable = np.zeros_like(binary_text)
                    for i in range(1, num):
                        bx = stats[i, cv2.CC_STAT_LEFT]
                        by = stats[i, cv2.CC_STAT_TOP]
                        bw_i = stats[i, cv2.CC_STAT_WIDTH]
                        bh_i = stats[i, cv2.CC_STAT_HEIGHT]
                        touches_border = (bx <= 1 or by <= 1
                                          or bx + bw_i >= cw - 1 or by + bh_i >= ch - 1)
                        if not touches_border:
                            cleanable[labels == i] = 255

                    if shape == 'box':
                        # Keep a small rim so faint box borders survive
                        rim = np.zeros((ch, cw), dtype=np.uint8)
                        cv2.rectangle(rim, (3, 3), (max(4, cw - 3), max(4, ch - 3)), 255, -1)
                        cleanable = cv2.bitwise_and(cleanable, rim)

                    # C. Dilate to catch anti-aliasing
                    kernel = np.ones((5,5), np.uint8)
                    final_mask = cv2.dilate(cleanable, kernel, iterations=1)

                    # D. Run OpenCV Inpainting
                    cleaned_crop = cv2.inpaint(crop, final_mask, inpaint_radius, cv2.INPAINT_TELEA)

                    # Paste back
                    final_image[y1:y2, x1:x2] = cleaned_crop

                # --- STRATEGY 2: FREE TEXT (LaMa + Rectangle) ---
                elif shape == 'free':
                    # Keep a small rim when nothing touches the crop edge, so a
                    # faint box border around free text is not painted over
                    inset = 4 if (entry.get('edge_ink') or 0) < 0.05 else 0
                    cv2.rectangle(lama_mask, (x1 + inset, y1 + inset),
                                  (x2 - inset, y2 - inset), 255, -1)
                    has_lama_work = True

            # Run LaMa batch for all free text found
            if has_lama_work:
                # Dilate LaMa mask slightly
                lama_kernel = np.ones((5, 5), np.uint8)
                lama_mask = cv2.dilate(lama_mask, lama_kernel, iterations=1)

                img_pil = Image.fromarray(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
                mask_pil = Image.fromarray(lama_mask)

                try:
                    # 1. Run Model (PIL in, PIL out)
                    result = self.lama(img_pil, mask_pil)
                    cleaned_lama = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

                    # 2. Resize fix (LaMa padding issue)
                    if cleaned_lama.shape[:2] != (h, w):
                        cleaned_lama = cv2.resize(cleaned_lama, (w, h))

                    # 3. Merge LaMa result
                    final_image = np.where(lama_mask[:, :, None] == 255, cleaned_lama, final_image)

                except Exception as e:
                    print(f"    ⚠ LaMa failed: {e}")

            return final_image

    def _render_page(self, original_image, manga_data, cleaning="lama"):
        """Clean the page, then draw the translated text.

        cleaning='lama': hybrid cleanup (current default) — OpenCV inpainting
        inside speech bubbles, LaMa inpainting for free text.
        cleaning='blur': legacy cleanup from the first version — Gaussian blur
        per bubble; kept for the stage comparison sheet.
        """
        if cleaning == "blur":
            working_img = original_image.copy()
            for entry in manga_data:
                if entry.get('translated_text'):
                    working_img = self._smart_clean_bubble(working_img, entry['bbox'])
        else:
            working_img = self.clean_page(original_image, manga_data)

        # Text Drawing with adaptive sizing and outlines
        img_pil = Image.fromarray(cv2.cvtColor(working_img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        for entry in manga_data:
            x1, y1, x2, y2 = entry['bbox']
            text = entry.get('translated_text', '')
            if not text: continue

            # Calculate optimal font size for this bubble
            font_size, wrapped_text = self._calculate_optimal_font_size(
                text, entry['bbox']
            )

            font = self._get_font(font_size)

            # Get text dimensions
            left, top, right, bottom = draw.multiline_textbbox(
                (0, 0), wrapped_text, font=font, align="center"
            )
            text_w, text_h = right - left, bottom - top

            # Center text
            text_x = x1 + ((x2 - x1) - text_w) / 2
            text_y = y1 + ((y2 - y1) - text_h) / 2

            # Draw with outline for readability
            self._draw_text_with_outline(
                draw, (text_x, text_y), wrapped_text, font,
                text_color="black", outline_color="white",
                outline_width=2, align="center", spacing=2
            )

        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def typeset(self, original_image, manga_data, output_path):
        final_img = self._render_page(original_image, manga_data)
        cv2.imwrite(output_path, final_img)
        print(f"  Saved: {output_path}")

    def save_comparison(self, original_image, manga_data, output_path,
                        labels=("Not translated", "Translated", "Translated + LaMa inpainting")):
        """Save a 3-panel sheet: the original page, the page translated with the
        legacy blur cleanup, and the full pipeline output (OpenCV + LaMa)."""
        panels = [
            original_image,
            self._render_page(original_image, manga_data, cleaning="blur"),
            self._render_page(original_image, manga_data, cleaning="lama"),
        ]
        pil_panels = [Image.fromarray(cv2.cvtColor(p, cv2.COLOR_BGR2RGB)) for p in panels]

        label_font = self._get_label_font(28)
        bar_h, gap = 48, 12
        width = sum(p.width for p in pil_panels) + gap * (len(pil_panels) - 1)
        height = max(p.height for p in pil_panels) + bar_h

        sheet = Image.new("RGB", (width, height), (32, 32, 32))
        draw = ImageDraw.Draw(sheet)
        x = 0
        for label, panel in zip(labels, pil_panels):
            draw.rectangle([x, 0, x + panel.width, bar_h], fill=(24, 24, 24))
            text_w = draw.textlength(label, font=label_font)
            draw.text((x + (panel.width - text_w) / 2, 9), label,
                      fill=(238, 238, 238), font=label_font)
            sheet.paste(panel, (x, bar_h))
            x += panel.width + gap

        sheet.save(output_path)
        print(f"  Saved comparison: {output_path}")

    def _get_label_font(self, size):
        """Font for the comparison-sheet labels (falls back to PIL default)."""
        candidates = (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
        )
        for path in candidates:
            if os.path.exists(path):
                try:
                    return ImageFont.truetype(path, size)
                except OSError:
                    continue
        return ImageFont.load_default()

    def process_chapter(self, input_folder, output_folder, series_info=None,
                          batch_size=4, selected_batches=None,
                          conf_threshold=0.15, save_comparisons=False):
            """
            Process manga chapter in batches for better context and efficiency
            """
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            valid_ext = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
            files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_ext)]
            # Sort numerically (p1, p2, p10 instead of p1, p10, p2)
            files.sort(key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else x)

            total_files = len(files)
            total_batches = (total_files + batch_size - 1) // batch_size
            
            # Master list to hold data for the entire chapter
            full_chapter_data = [] 

            print(f"Found {total_files} images in {input_folder}")
            print(f"Total batches: {total_batches} (batch size: {batch_size})")

            if selected_batches:
                print(f"Processing selected batches: {selected_batches}")
            else:
                print(f"Processing all batches\n")

            # Process in batches
            for batch_start in range(0, total_files, batch_size):
                batch_num = batch_start // batch_size + 1

                # Skip if not in selected batches
                if selected_batches and batch_num not in selected_batches:
                    continue

                batch_files = files[batch_start:batch_start + batch_size]
                print(f"=== Batch {batch_num}/{total_batches} ({len(batch_files)} pages) ===")

                # Collect all data for this batch
                batch_data = []
                batch_images = []

                temp_crop_dir = os.path.join(output_folder, "temp_crops")

                for idx, filename in enumerate(batch_files):
                    page_num = batch_start + idx + 1
                    print(f"  [{page_num}/{total_files}] Detecting bubbles in {filename}...")

                    input_path = os.path.join(input_folder, filename)
                    page_id = f"p{page_num:03d}"

                    try:
                        img, data = self.detect_and_process(
                            input_path, output_dir=temp_crop_dir,
                            page_id=page_id, conf_threshold=conf_threshold,
                        )

                        if data:
                            print(f"    Running OCR on {len(data)} bubbles...")
                            data = self.run_ocr(data)
                            batch_data.extend(data)
                        else:
                            print(f"    No bubbles detected")

                        batch_images.append((filename, img, page_id))

                    except Exception as e:
                        print(f"    Error processing {filename}: {e}")
                        continue

                # Translate entire batch at once for context
                if batch_data:
                    print(f"  Translating {len(batch_data)} bubbles from batch...")
                    batch_data = self.translate_batch(batch_data, series_info=series_info)
                    
                    # Add this batch's completed data to the master list
                    full_chapter_data.extend(batch_data)

                # Typeset each page
                print(f"  Typesetting pages...")
                for filename, img, page_id in batch_images:
                    output_path = os.path.join(output_folder, filename)

                    # Filter data for this specific page
                    page_data = [d for d in batch_data if d.get('page_id') == page_id]

                    try:
                        self.typeset(img, page_data, output_path)

                        if save_comparisons:
                            stem = os.path.splitext(filename)[0]
                            comparison_path = os.path.join(
                                output_folder, f"{stem}_comparison.png"
                            )
                            self.save_comparison(img, page_data, comparison_path)
                    except Exception as e:
                        print(f"    Error typesetting {filename}: {e}")

                print()  # Empty line between batches
            
            # --- NEW LOGIC: Save JSON if debug is ON ---
            if self.debug and full_chapter_data:
                json_filename = f"chapter_data.json"
                json_path = os.path.join(output_folder, json_filename)
                
                try:
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(full_chapter_data, f, ensure_ascii=False, indent=2)
                    print(f"  [DEBUG] Saved full chapter data to: {json_filename}")
                except Exception as e:
                    print(f"  [DEBUG] Failed to save JSON: {e}")

            print(f"\n✓ Chapter processing complete! Output saved to: {output_folder}")

if __name__ == "__main__":
    # Define custom character/term translations
    #Example
    custom_translations = {
        "ルーグ": "Lugh",
        "トウアハーデ": "Tuatha Dé",
        "ディア": "Dia",
        "タルト": "Tarte",
        # Add more character names and terms as needed
    }

    translator = MangaTranslator(
        yolo_model_path='comic-speech-bubble-detector.pt',
        llm_base_url="http://localhost:8110",
        llm_model="local",
        font_path="animeace2_reg.ttf",
        custom_translations=custom_translations
    )

    # Define Series Context
    #Example:
    series_context = {
        "title": "Sekai Saikou no Ansatsusha, Isekai Kizoku ni Tensei Suru",
        "tags": "Action, Adventure, Comedy, Drama, Ecchi, Fantasy, Harem, Romance",
        "description": "The most dangerous prey. When the most accomplished assassin on Earth meets his end just before retirement, he finds himself standing before a goddess. Her world is headed for disaster and she has a need for his particular set of skills. Armed with powerful new abilities and decades of deadly knowledge, he begins his second life as Lugh, a young scion of the Tuatha Dé clan of assassins. His mission: to find and eliminate the strongest individual in this world-the hero."
    }

    # Example 1: Process all batches
    translator.process_chapter(
        input_folder='/Your_Chapter_Folder',
        output_folder='/Output_Folder',
        series_info=series_context,
        batch_size=4,          # Depends on your GPU and the LLM context window
        save_comparisons=True  # Also save <name>_comparison.png per page
    )

    # Example 2: Process only specific batches (e.g., batches 1 and 3)
    # translator.process_chapter(
    #     input_folder='/Your_Chapter_Folder',
    #     output_folder='/Output_Folder',
    #     series_info=series_context,
    #     batch_size=4,
    #     selected_batches=[1, 3]  # Only process batches 1 and 3
    # )
