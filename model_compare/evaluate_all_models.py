import os
import subprocess
import cv2
import torch
from ultralytics import YOLO
from datetime import datetime

# ============================================================
# CONFIG
# ============================================================

BASE = os.path.dirname(os.path.abspath(__file__))   # model_compare/
PROJECT_ROOT = os.path.abspath(os.path.join(BASE, ".."))

MODELS_LIST = os.path.join(BASE, "models_to_compare.txt")
DATA_YAML = os.path.join(PROJECT_ROOT, "web_app/game_dataset/data.yaml")

RESULT_DIR = os.path.join(BASE, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

IMAGES_DIR = os.path.join(PROJECT_ROOT, "web_app/game_dataset/images")
LABELS_DIR = os.path.join(PROJECT_ROOT, "web_app/game_dataset/labels")

IOU_THRESHOLD = 0.5
CONF_THRESHOLD = 0.25


# ============================================================
# UTILITIES
# ============================================================

def load_label(path):
    boxes = []
    if not os.path.exists(path):
        return boxes
    with open(path, "r") as f:
        for line in f:
            cls, xc, yc, w, h = map(float, line.strip().split())
            boxes.append([int(cls), xc, yc, w, h])
    return boxes


def yolo_to_xyxy(box, W, H):
    cls, xc, yc, w, h = box
    x1 = (xc - w/2) * W
    y1 = (yc - h/2) * H
    x2 = (xc + w/2) * W
    y2 = (yc + h/2) * H
    return [cls, x1, y1, x2, y2]


def compute_iou(a, b):
    # a, b: [x1,y1,x2,y2]
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (a[2]-a[0]) * (a[3]-a[1])
    areaB = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (areaA + areaB - inter) if (areaA+areaB-inter) else 0


def list_images(root):
    for dp, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                yield os.path.join(dp, f)


# ============================================================
# MAIN EVALUATION
# ============================================================

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
summary_lines = [f"\n=== Model Comparison Report ({timestamp}) ===\n"]

with open(MODELS_LIST, "r") as f:
    model_paths = [line.strip() for line in f if line.strip()]

for model_path in model_paths:

    print(f"\n🚀 Evaluating Model: {model_path}")
    model = YOLO(model_path)

    # --------------------------------------------------------
    # (A) YOLO Official Validation
    # --------------------------------------------------------
    val_output = subprocess.run(
        ["yolo", "detect", "val", f"model={model_path}", f"data={DATA_YAML}", "verbose=False"],
        capture_output=True, text=True
    ).stdout

    with open(os.path.join(RESULT_DIR, f"{os.path.basename(model_path)}_mAP.txt"), "w") as f:
        f.write(val_output)

    # Extract mAP scores
    mAP50 = mAP5095 = "N/A"
    for line in val_output.splitlines():
        if "all" in line:
            parts = line.split()
            mAP50 = parts[-2]
            mAP5095 = parts[-1]

    # --------------------------------------------------------
    # (B) Strict Correctness Evaluation
    # --------------------------------------------------------
    total = correct = fn = fp = mis = low_iou = 0

    for img_path in list_images(IMAGES_DIR):
        total += 1

        img = cv2.imread(img_path)
        H, W = img.shape[:2]

        # load GT
        lbl_path = img_path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        gt_raw = load_label(lbl_path)
        gt_boxes = [yolo_to_xyxy(b, W, H) for b in gt_raw]

        # prediction
        r = model(img, conf=CONF_THRESHOLD, verbose=False)[0]

        pred_boxes = []
        for b in r.boxes.xyxy.cpu().numpy():
            # b = [x1,y1,x2,y2]
            cls = int(r.boxes.cls[pred_boxes.__len__()])
            pred_boxes.append([cls, b[0], b[1], b[2], b[3]])

        used = set()
        case = "correct"

        # match
        for p in pred_boxes:
            cls_p, px1, py1, px2, py2 = p

            best_i = 0
            best_gt = -1
            for i, g in enumerate(gt_boxes):
                iou_score = compute_iou([px1,py1,px2,py2], g[1:])
                if iou_score > best_i:
                    best_i = iou_score
                    best_gt = i

            if best_i < 0.01:
                fp += 1; case = "fp"; break

            gt_cls = gt_boxes[best_gt][0]
            used.add(best_gt)

            if cls_p != gt_cls:
                mis += 1; case = "mis"; break

            if best_i < IOU_THRESHOLD:
                low_iou += 1; case = "low"; break

        # false negatives
        if case == "correct":
            for i in range(len(gt_boxes)):
                if i not in used:
                    fn += 1; case = "fn"; break

        if case == "correct":
            correct += 1

    strict_correct = round(correct / total * 100, 2)

    # --------------------------------------------------------
    # (C) Model File Info
    # --------------------------------------------------------
    size_mb = round(os.path.getsize(model_path) / (1024*1024), 2)

    # summary
    summary_lines.append(
        f"Model: {model_path}\n"
        f"  ✓ Strict Correctness: {strict_correct}%\n"
        f"  ✓ mAP50: {mAP50}\n"
        f"  ✓ mAP50-95: {mAP5095}\n"
        f"  FN:{fn} | FP:{fp} | Mis:{mis} | LowIoU:{low_iou}\n"
        f"  File Size: {size_mb} MB\n"
        f"---------------------------------------------\n"
    )

# save summary
with open(os.path.join(RESULT_DIR, "model_comparison_summary.txt"), "w") as f:
    f.write("\n".join(summary_lines))

print("\n🎉 All Done! Summary saved in:")
print(os.path.join(RESULT_DIR, "model_comparison_summary.txt"))
