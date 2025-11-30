import os
import glob
import csv
from ultralytics import YOLO
import numpy as np
from prettytable import PrettyTable


# ======== CONFIG ========
EDGE_CASE_DIR = "edge_case"            # root folder inside YOLO_demo
MODEL_PATHS = [
    "best.pt",
    "yolov8n.pt"
]
OUTPUT_CSV = "edge_case_results.csv"
IOU_THRESHOLD = 0.5
# ==========================


def iou(boxA, boxB):
    """Compute IoU for two boxes: [x_center, y_center, w, h] (normalized)."""
    # Convert from center to xyxy
    def to_xyxy(box):
        x, y, w, h = box
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        return x1, y1, x2, y2

    A = to_xyxy(boxA)
    B = to_xyxy(boxB)

    # Intersection box
    x1 = max(A[0], B[0])
    y1 = max(A[1], B[1])
    x2 = min(A[2], B[2])
    y2 = min(A[3], B[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0

    areaA = (A[2] - A[0]) * (A[3] - A[1])
    areaB = (B[2] - B[0]) * (B[3] - B[1])
    union = areaA + areaB - inter_area

    return inter_area / union if union > 0 else 0.0


def load_labels(label_path):
    """Load YOLO txt file: cls x y w h."""
    if not os.path.exists(label_path):
        return []
    labels = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            cls = int(parts[0])
            box = list(map(float, parts[1:]))
            labels.append((cls, box))
    return labels


def score_image(model, img_path):
    """Run YOLO and compare with ground truth labels."""
    # Load ground truth
    txt_path = img_path.replace(".jpg", ".txt").replace(".png", ".txt")
    gt = load_labels(txt_path)

    # YOLO inference
    pred = model(img_path)[0]

    pred_boxes = []
    for box in pred.boxes:
        cls = int(box.cls[0])
        xywhn = box.xywhn[0].tolist()   # normalized
        pred_boxes.append((cls, xywhn))

    # Score comparison
    matched = 0
    for gt_cls, gt_box in gt:
        for pred_cls, pred_box in pred_boxes:
            if pred_cls == gt_cls:
                if iou(gt_box, pred_box) >= IOU_THRESHOLD:
                    matched += 1
                    break

    recall = matched / len(gt) if len(gt) > 0 else 1.0
    return recall, len(gt), len(pred_boxes)


def collect_edge_images():
    """Return list of all edge-case image paths."""
    patterns = [
        os.path.join(EDGE_CASE_DIR, "**", "*.jpg"),
        os.path.join(EDGE_CASE_DIR, "**", "*.png")
    ]
    paths = []
    for p in patterns:
        paths.extend(glob.glob(p, recursive=True))
    return sorted(paths)


def main():
    edge_images = collect_edge_images()
    if len(edge_images) == 0:
        print("❌ No edge-case images found.")
        return

    print(f"🔥 Found {len(edge_images)} edge-case images.")

    # Prepare CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "image", "recall", "gt_count", "pred_count"])

        summary = []

        for model_path in MODEL_PATHS:
            print(f"\n🔵 Evaluating model: {model_path}")
            model = YOLO(model_path)

            recalls = []

            for img in edge_images:
                recall, gt_count, pred_count = score_image(model, img)
                recalls.append(recall)

                writer.writerow([model_path, os.path.basename(img), recall, gt_count, pred_count])

            avg_recall = sum(recalls) / len(recalls)
            summary.append((model_path, avg_recall))

    # Summary table
    table = PrettyTable()
    table.field_names = ["Model", "Avg Recall"]
    for m, r in summary:
        table.add_row([m, f"{r:.3f}"])
    print("\n🎯 FINAL SUMMARY")
    print(table)


if __name__ == "__main__":
    main()
