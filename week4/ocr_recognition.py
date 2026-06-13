# ============================================================
# Project 4: Image & Text Recognition (OCR Path)
# DecodeLabs Internship | Batch 2026
# Algorithm: pytesseract OCR + OpenCV Pre-Processing
# Pipeline: Load Image → Grayscale → Threshold → OCR → Output
# ============================================================

import cv2
import pytesseract
import numpy as np
import os
import sys

# ─────────────────────────────────────────────
# STEP 1: LOAD IMAGE
# ─────────────────────────────────────────────

def load_image(image_path):
    if not os.path.exists(image_path):
        print(f"[ERROR] File not found: {image_path}")
        sys.exit(1)

    img = cv2.imread(image_path)
    if img is None:
        print("[ERROR] Could not read image. Check file format.")
        sys.exit(1)

    print(f"[OK] Image loaded: {image_path}")
    print(f"     Size: {img.shape[1]}x{img.shape[0]} pixels")
    return img

# ─────────────────────────────────────────────
# STEP 2: PRE-PROCESSING
# ─────────────────────────────────────────────

def preprocess_image(img):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("[OK] Step 1: Converted to grayscale")

    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    print("[OK] Step 2: Denoising applied")

    # Adaptive Thresholding
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    print("[OK] Step 3: Adaptive thresholding applied")

    return thresh

# ─────────────────────────────────────────────
# STEP 3: OCR - EXTRACT TEXT
# ─────────────────────────────────────────────

def extract_text(processed_img):
    config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(processed_img, config=config)
    return text.strip()

# ─────────────────────────────────────────────
# STEP 4: CONFIDENCE SCORE (80% threshold filter)
# ─────────────────────────────────────────────

def get_confidence_data(processed_img):
    data = pytesseract.image_to_data(
        processed_img,
        output_type=pytesseract.Output.DICT
    )

    words = []
    for i, word in enumerate(data['text']):
        conf = int(data['conf'][i])
        if conf > 80 and word.strip():
            words.append((word, conf))

    return words

# ─────────────────────────────────────────────
# STEP 5: SAVE OUTPUT IMAGE
# ─────────────────────────────────────────────

def save_debug_image(processed_img, output_path="output_preprocessed.png"):
    cv2.imwrite(output_path, processed_img)
    print(f"[OK] Pre-processed image saved: {output_path}")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print("   Project 4: OCR Text Recognition - DecodeLabs")
    print("=" * 55)

    image_path = input("\nEnter image path (e.g. test.png): ").strip()

    print("\n[PIPELINE STARTING]")
    print("-" * 40)

    img = load_image(image_path)
    processed = preprocess_image(img)
    save_debug_image(processed)

    print("\n[EXTRACTING TEXT...]")
    extracted_text = extract_text(processed)
    high_conf_words = get_confidence_data(processed)

    print("\n" + "=" * 55)
    print("EXTRACTED TEXT:")
    print("-" * 40)
    if extracted_text:
        print(extracted_text)
    else:
        print("[No text detected]")

    print("\n" + "-" * 40)
    print("HIGH CONFIDENCE WORDS (>80%):")
    if high_conf_words:
        for word, conf in high_conf_words[:10]:
            bar = "█" * (conf // 10)
            print(f"  '{word}' → {conf}%  |{bar}|")
    else:
        print("  [None found above threshold]")

    print("=" * 55)
    print("OCR Pipeline complete.")

if __name__ == "__main__":
    main()
