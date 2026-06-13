# ============================================================
# Project 4: Image & Text Recognition (Object Detection Path)
# DecodeLabs Internship | Batch 2026
# Algorithm: MobileNet-SSD via OpenCV DNN
# Pipeline: Load → DNN Forward Pass → Softmax → 80% Filter → Output
# ============================================================

import cv2
import numpy as np
import urllib.request
import os
import sys

CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def download_model():
    proto_url  = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt"
    model_url  = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel"
    proto_file = "MobileNetSSD_deploy.prototxt"
    model_file = "MobileNetSSD_deploy.caffemodel"

    if not os.path.exists(proto_file):
        print("[DOWNLOADING] Prototxt config file...")
        urllib.request.urlretrieve(proto_url, proto_file)
        print("[OK] Prototxt downloaded.")

    if not os.path.exists(model_file):
        print("[DOWNLOADING] Caffemodel weights (~24MB)...")
        urllib.request.urlretrieve(model_url, model_file)
        print("[OK] Model downloaded.")

    return proto_file, model_file

def load_image(image_path):
    if not os.path.exists(image_path):
        print(f"[ERROR] File not found: {image_path}")
        sys.exit(1)

    img = cv2.imread(image_path)
    if img is None:
        print("[ERROR] Could not read image.")
        sys.exit(1)

    print(f"[OK] Image loaded: {img.shape[1]}x{img.shape[0]}")
    return img

def load_model(proto_file, model_file):
    print("[LOADING] MobileNet-SSD model...")
    net = cv2.dnn.readNetFromCaffe(proto_file, model_file)
    print("[OK] Model loaded.")
    return net

def detect_objects(img, net, confidence_threshold=0.50):
    (h, w) = img.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(img, (300, 300)),
        0.007843, (300, 300), 127.5
    )

    net.setInput(blob)
    detections = net.forward()

    results = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_threshold:
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            results.append({
                "label": label,
                "confidence": float(confidence),
                "box": (startX, startY, endX, endY)
            })

    return results

def draw_and_save(img, results, output_path="output_detected.png"):
    output = img.copy()

    for obj in results:
        label      = obj["label"]
        confidence = obj["confidence"]
        (sX, sY, eX, eY) = obj["box"]
        idx = CLASSES.index(label)
        color = COLORS[idx]

        cv2.rectangle(output, (sX, sY), (eX, eY), color, 2)
        text = f"{label}: {confidence*100:.1f}%"
        y = sY - 10 if sY - 10 > 10 else sY + 10
        cv2.putText(output, text, (sX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imwrite(output_path, output)
    print(f"[OK] Output saved: {output_path}")

def main():
    print("=" * 55)
    print(" Project 4: Object Detection - DecodeLabs")
    print("=" * 55)

    image_path = input("\nEnter image path (e.g. test.jpg): ").strip()

    proto_file, model_file = download_model()
    img = load_image(image_path)
    net = load_model(proto_file, model_file)

    print("\n[RUNNING DETECTION...]")
    results = detect_objects(img, net, confidence_threshold=0.50)
    draw_and_save(img, results)

    print("\n" + "=" * 55)
    print("DETECTED OBJECTS:")
    print("-" * 40)

    if results:
        for rank, obj in enumerate(results, 1):
            conf = obj["confidence"] * 100
            bar  = "█" * int(conf // 5)
            print(f"{rank}. {obj['label'].upper()}")
            print(f"   Confidence: {conf:.1f}%  |{bar}|")
            print(f"   Box: {obj['box']}")
            print()
    else:
        print("  [No objects detected above threshold]")
        print("  Try lowering confidence_threshold to 0.30")

    print(f"Total objects found: {len(results)}")
    print("=" * 55)
    print("Detection complete. Check output_detected.png")

if __name__ == "__main__":
    main()
