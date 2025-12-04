"""
==============================================================
 Automated Edge Case Miner & Next-Train Dataset Builder
 Chloe Lee | Distilled-Vision-Agent Project | 2025
==============================================================

Description:
    This script evaluates a YOLO detection model on the full
    game dataset and automatically extracts all "hard examples"
    (False Negative, False Positive, Misclassification, Low IoU).
    These samples are saved into categorized folders and are
    also used to build the next training dataset (train4, train5, ...).

    This script implements an automated Hard Example Mining
    pipeline used for iterative model improvement.

--------------------------------------------------------------
Usage:
    1. Set MODEL_PATH to the YOLO model you want to evaluate.
    2. Set TRAINX_DIR and TRAIN(X+1)_DIR depending on the stage.
       (Ex: If evaluating train3 → set TRAIN3_DIR and TRAIN4_DIR)
    3. Run:
           python auto_edge_case_finder.py
    4. After running:
        - edge_case_auto/         → categorized mistake folders
        - train4_dataset/         → YOLO-ready dataset
        - train4_summary.txt      → evaluation summary
        - edge_log.txt (append)   → stores all past evaluations

--------------------------------------------------------------
Input Structure:
    ../game_dataset/
        images/train/
        images/val/
        labels/train/
        labels/val/

Output Structure:
    YOLO_demo/train4/
        edge_case_auto/
            false_negative/
            false_positive/
            low_iou/
            misclassified/
            correct/   (only if SAVE_CORRECT=True)
        train4_dataset/
            images/train/
            images/val/
            labels/train/
            labels/val/

--------------------------------------------------------------
Edge Case Rules:
    - False Positive:
          Prediction has IoU < 0.01 with all GT boxes.
    - Misclassified:
          Predicted class != GT class.
    - Low IoU:
          IoU exists but < IOU_THRESHOLD (default: 0.5).
    - False Negative:
          GT exists but detector failed to detect it.
    - Correct:
          High IoU + correct class + no GT missed.

--------------------------------------------------------------
Training Dataset Rule:
    - Only incorrect cases (FN, FP, Low IoU, Misclassified)
      are included in train4_dataset.
    - Correct predictions are excluded by default.

--------------------------------------------------------------
How to create next dataset (train5, train6, ...):
    To generate train5 after evaluating train4:
        - Update MODEL_PATH → runs/detect/train4/weights/best.pt
        - Update TRAIN4_DIR → YOLO_demo/train4
        - Update TRAIN5_DIR → YOLO_demo/train5
        - Update dataset folder name to train5_dataset

--------------------------------------------------------------
Automatic Logging:
    Every run appends a new entry to:
        YOLO_demo/trainX/edge_case_auto/edge_log.txt
    This file builds an evaluation history across all models.

--------------------------------------------------------------
Notes:
    - Supports safe incremental training.
    - Compatible with CPU execution on Apple M-series.
    - Designed for student projects, reproducibility, and clarity.

==============================================================
"""



import os
import shutil
from datetime import datetime
from ultralytics import YOLO
import cv2

# ------------------------------------
# CONFIG
# ------------------------------------

# 🔥 [1] 모델 경로(train3 best.pt)
MODEL_PATH = "../../runs/detect/train3_cpu/weights/best.pt"

# 🔥 [2] 기존 게임 데이터셋 경로
IMAGES_DIR = "../game_dataset/images"
LABELS_DIR = "../game_dataset/labels"

# 🔥 [3] train3 작업 폴더
TRAIN3_DIR = "../YOLO_demo/train3"
TRAIN4_DIR = "../YOLO_demo/train4"

EDGE_DIR = os.path.join(TRAIN4_DIR, "edge_case_auto")
TRAIN4_DATASET = os.path.join(TRAIN4_DIR, "train4_dataset")

IOU_THRESHOLD = 0.5
CONF_THRESHOLD = 0.25
SAVE_CORRECT = False     # train4_dataset에는 correct 포함하지 않음


# ------------------------------------
# Utilities
# ------------------------------------

def ensure(p):
    os.makedirs(p, exist_ok=True)

def list_images(root):
    for dp, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                yield os.path.join(dp, f)

def load_label(path):
    boxes = []
    if not os.path.exists(path):
        return boxes
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls, xc, yc, w, h = map(float, parts)
                boxes.append([int(cls), xc, yc, w, h])
    return boxes

def yolo_to_xyxy(box, W, H):
    cls, xc, yc, w, h = box
    x1 = int((xc - w/2) * W)
    y1 = int((yc - h/2) * H)
    x2 = int((xc + w/2) * W)
    y2 = int((yc + h/2) * H)
    return cls, x1, y1, x2, y2

def iou(a, b):
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (a[2]-a[0]) * (a[3]-a[1])
    areaB = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (areaA + areaB - inter) if (areaA+areaB-inter) else 0


def copy_with_label(src_img, dst_img_folder):
    ensure(dst_img_folder)
    ensure(dst_img_folder.replace("images", "labels"))

    shutil.copy(src_img, dst_img_folder)

    src_lbl = src_img.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
    dst_lbl_folder = dst_img_folder.replace("images", "labels")

    if os.path.exists(src_lbl):
        shutil.copy(src_lbl, dst_lbl_folder)


# ------------------------------------
# MAIN WORKFLOW
# ------------------------------------

print("\n🚀 Loading model:", MODEL_PATH)
model = YOLO(MODEL_PATH)

# Clean and prepare folders
ensure(TRAIN4_DIR)
ensure(EDGE_DIR)
ensure(os.path.join(EDGE_DIR, "images"))
ensure(TRAIN4_DATASET)

# Subfolders
for split in ["train", "val"]:
    ensure(os.path.join(TRAIN4_DATASET, "images", split))
    ensure(os.path.join(TRAIN4_DATASET, "labels", split))

results_report = []

false_pos = false_neg = miscls = low_iou_cnt = correct = 0
total = 0

print("\n🔎 Scanning original dataset...")
for img_path in list_images(IMAGES_DIR):
    total += 1

    img = cv2.imread(img_path)
    H, W = img.shape[:2]

    # GT
    label_path = img_path.replace("/images/", "/labels/").rsplit(".",1)[0] + ".txt"
    gt_boxes_raw = load_label(label_path)
    gt_boxes = [yolo_to_xyxy(b, W, H) for b in gt_boxes_raw]

    # Prediction
    pred = model(img, conf=CONF_THRESHOLD, verbose=False)[0]
    pred_boxes = [[int(b.cls), int(b.xyxy[0][0]), int(b.xyxy[0][1]),
                   int(b.xyxy[0][2]), int(b.xyxy[0][3])] for b in pred.boxes]

    used = set()
    case = "correct"

    # Match predictions
    for pb in pred_boxes:
        cls_p, x1,y1,x2,y2 = pb
        best_iou = 0
        best_gt = -1

        for idx, gb in enumerate(gt_boxes):
            i = iou([x1,y1,x2,y2], gb[1:])
            if i > best_iou:
                best_iou = i
                best_gt = idx

        if best_iou < 0.01:
            case = "false_positive"
            false_pos += 1
            break

        gt_cls = gt_boxes[best_gt][0]
        used.add(best_gt)

        if cls_p != gt_cls:
            case = "misclassified"
            miscls += 1
            break

        if best_iou < IOU_THRESHOLD:
            case = "low_iou"
            low_iou_cnt += 1
            break

    # False negative
    if case == "correct":
        for idx in range(len(gt_boxes)):
            if idx not in used:
                case = "false_negative"
                false_neg += 1
                break

    if case == "correct":
        correct += 1

    # Save to edge_case folder
    dst_case = os.path.join(EDGE_DIR, case)
    ensure(dst_case)
    copy_with_label(img_path, os.path.join(dst_case, "images"))

    # If not correct, save into train4 dataset
    if case != "correct":
        split = "train" if "/train/" in img_path else "val"
        dst_final = os.path.join(TRAIN4_DATASET, "images", split)
        copy_with_label(img_path, dst_final)


# ------------------------------------
# REPORT
# ------------------------------------

ts = datetime.now().strftime("%Y-%m-%d %H:%M")

report = f"""
YOLO train3 test report ({ts})
Model: {MODEL_PATH}

Total images: {total}
Correct: {correct}
False Negative: {false_neg}
False Positive: {false_pos}
Misclassified: {miscls}
Low IoU: {low_iou_cnt}

Correct %: {round(correct/total*100,2)}%

"""
# ⭐ NEW — edge_case 폴더에 계속 이어쓰기
LOG_PATH = os.path.join(EDGE_DIR, "edge_case_log.txt")
with open(LOG_PATH, "a") as logf:
    logf.write(report)

print("\n🔥 DONE!")
print(report)
print(f"📄 Log saved at: {LOG_PATH}")
