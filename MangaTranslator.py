import os
import json
import cv2
import numpy as np
import textwrap
import pyphen
import re
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from manga_ocr import MangaOcr

class MangaTranslator:
    def __init__(self, yolo_model_path='comic_yolov8m.pt', ollama_model="qwen2.5:7b",
                 font_path="font.ttf", custom_translations=None, keep_honorifics=True, debug=True):
        """
        Initialize models. Defaults to qwen2.5:7b for speed on T4 GPUs.

        Args:
            custom_translations: Dictionary of Japanese terms -> English equivalents
        """
        print("Loading YOLO model...")
        self.yolo_model = YOLO(yolo_model_path)
        self.font_path = font_path

        print("Loading MangaOCR model...")
        self.mocr = MangaOcr()

        print("Initializing LLM...")
        self.llm = ChatOllama(
            model=ollama_model,
            temperature=0.3,
            num_ctx=2048,
            num_gpu=-1
        )
        self.dic = pyphen.Pyphen(lang='en')

        # Font cache for performance
        self.font_cache = {}

        # Custom translation dictionary
        self.custom_translations = custom_translations or {}

        self.keep_honorifics = keep_honorifics
        self.honorifics = ['san', 'chan', 'kun', 'sama', 'senpai', 'sensei', 'dono', 'tan']

        # For romanization fallback
        try:
            import pykakasi
            self.kakasi = pykakasi.kakasi()
        except ImportError:
            print("Warning: pykakasi not installed. Install with 'pip install pykakasi' for romanization support.")
            self.kakasi = None

    def _get_font(self, size):
        """Cache fonts to avoid repeated loading"""
        if size not in self.font_cache:
            try:
                self.font_cache[size] = ImageFont.truetype(self.font_path, size)
            except IOError:
                self.font_cache[size] = ImageFont.load_default()
        return self.font_cache[size]

    def _sort_bubbles(self, bubbles, row_threshold=50):
        bubbles.sort(key=lambda b: b[1])
        sorted_bubbles = []
        if not bubbles:
            return sorted_bubbles

        current_row = [bubbles[0]]
        for i in range(1, len(bubbles)):
            if abs(bubbles[i][1] - current_row[-1][1]) < row_threshold:
                current_row.append(bubbles[i])
            else:
                current_row.sort(key=lambda b: b[2], reverse=True)
                sorted_bubbles.extend(current_row)
                current_row = [bubbles[i]]

        current_row.sort(key=lambda b: b[2], reverse=True)
        sorted_bubbles.extend(current_row)
        return sorted_bubbles

    def _wrap_text_dynamic(self, text, font, max_width):
        words = text.split()
        lines = []
        current_line = []
        current_width = 0
        space_width = font.getlength(" ")

        for word in words:
            word_width = font.getlength(word)
            potential_width = current_width + word_width + (space_width if current_line else 0)

            if potential_width <= max_width:
                current_line.append(word)
                current_width = potential_width
            else:
                splits = list(self.dic.iterate(word))
                found_split = False
                for start, end in reversed(splits):
                    chunk = start + "-"
                    chunk_width = font.getlength(chunk)
                    if current_width + chunk_width + (space_width if current_line else 0) <= max_width:
                        current_line.append(chunk)
                        lines.append(" ".join(current_line))
                        current_line = [end]
                        current_width = font.getlength(end)
                        found_split = True
                        break

                if not found_split:
                    if current_line:
                        lines.append(" ".join(current_line))
                    current_line = [word]
                    current_width = word_width

        if current_line:
            lines.append(" ".join(current_line))
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
                    # Add the first found honorific to what's likely a name
                    # Look for capitalized words (likely names)
                    for i in range(len(words) - 1, -1, -1):
                        if words[i] and words[i][0].isupper():
                            # Add honorific to this name
                            words[i] = words[i] + found_honorifics[0]
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

    def _calculate_optimal_font_size(self, text, bbox, min_size=10, max_size=24):
        """
        Dynamically calculate font size based on bubble dimensions and text length
        """
        x1, y1, x2, y2 = bbox
        box_width = x2 - x1
        box_height = y2 - y1

        # Start with max size and reduce until text fits
        for size in range(max_size, min_size - 1, -1):
            font = self._get_font(size)
            wrapped = self._wrap_text_dynamic(text, font, box_width - 10)

            # Check if it fits height-wise
            temp_draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
            left, top, right, bottom = temp_draw.multiline_textbbox(
                (0, 0), wrapped, font=font, align="center"
            )
            text_height = bottom - top

            if text_height < (box_height - 10):
                return size, wrapped

        # Return minimum size with wrapped text
        font = self._get_font(min_size)
        wrapped = self._wrap_text_dynamic(text, font, box_width - 10)
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

    def detect_and_process(self, image_path, output_dir="crops", page_id="", conf_threshold=0.2):
            image = cv2.imread(image_path)
            if image is None: raise ValueError(f"Not found: {image_path}")

            # 1. Run Prediction
            results = self.yolo_model.predict(source=image, conf=conf_threshold, save=False, verbose=False)
            
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

            # Sort (top to bottom, right to left for manga)
            # Note: We need a custom sort function since detections is now a dict, not just a list of boxes
            detections = sorted(detections, key=lambda x: (x['bbox'][1], -x['bbox'][0]))

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
                    "crop_path": crop_path,
                    "original_text": "",
                    "translated_text": ""
                })
                
            return image, manga_data

    def run_ocr(self, manga_data):
        for entry in manga_data:
            crop_path = entry['crop_path']
            japanese_text = self.mocr(crop_path)

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
            response = self.llm.invoke(prompt)
            translation = response.content.strip()

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
            "You are a professional manga translator. "
            f"{context_str}"
            "Translate the following Japanese text bubbles into natural, concise, colloquial English. "
            "Use the Series Context to determine tone, slang, and character voices. "
            "Maintain dialogue consistency across pages. "
            "\n\nCRITICAL: Return ONLY a valid JSON array. No explanations, no markdown, no code blocks. "
            'Format: [{{"bubble_id": "p001_1", "translation": "English text here"}}, ...]\n'
            "Your entire response must be parseable JSON."
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Translate this JSON array:\n{payload}")
        ])

        chain = prompt | self.llm

        content = ""
        try:
            response = chain.invoke({"payload": json_string})
            content = response.content.strip()

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
                        print(f"    ⚠ Translation for {entry['id']} contains Japanese, retrying...")
                        retry_translation = self._translate_single_bubble(
                            entry['original_text'], series_info
                        )

                        # If still has Japanese, romanize those parts
                        if self._has_japanese_characters(retry_translation):
                            print(f"    ⚠ Still has Japanese, romanizing...")
                            retry_translation = self._romanize_japanese(retry_translation)

                        entry['translated_text'] = retry_translation
                    else:
                        entry['translated_text'] = translation

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

        return manga_data

    def typeset(self, original_image, manga_data, output_path):
        working_img = original_image.copy()

        # 1. Smart Clean with inpainting
        for entry in manga_data:
            if not entry.get('translated_text'): continue
            working_img = self._smart_clean_bubble(working_img, entry['bbox'])

        # 2. Text Drawing with adaptive sizing and outlines
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

        final_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, final_img)
        print(f"  Saved: {output_path}")

    def process_chapter(self, input_folder, output_folder, series_info=None,
                          batch_size=4, selected_batches=None):
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
                        img, data = self.detect_and_process(input_path, output_dir=temp_crop_dir, page_id=page_id)

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
                    except Exception as e:
                        print(f"    Error typesetting {filename}: {e}")

                print()  # Empty line between batches
            
            # --- NEW LOGIC: Save JSON if debug is ON ---
            if self.debug and full_chapter_data:
                json_filename = f"chapter_data_{int(time.time())}.json"
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
        ollama_model="qwen2.5:7b",
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
        batch_size=4 # Depends on your GPU, I found the sweet spot around 3-4 with T4 GPU
    )

    # Example 2: Process only specific batches (e.g., batches 1 and 3)
    # translator.process_chapter(
    #     input_folder='/Your_Chapter_Folder',
    #     output_folder='/Output_Folder',
    #     series_info=series_context,
    #     batch_size=4,
    #     selected_batches=[1, 3]  # Only process batches 1 and 3
    # )

    # Zip result for download
    import subprocess
    subprocess.run(['zip', '-r', 'translated_chapter.zip', 'translated_chapter'])
    print("\nDownload translated_chapter.zip from files panel.")
