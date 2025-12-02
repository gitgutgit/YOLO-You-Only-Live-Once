"""
==============================================================
 Edge Case Miner + Train6 Dataset Builder
 Chloe Lee | Distilled-Vision-Agent Project | 2025
==============================================================

Description:
    Step 1 — Evaluation (Hard Example Mining)
        - Evaluate YOLO model on FULL game_dataset
        - Save FN, FP, Misclassified, Low IoU, Correct → edge_case_auto/

    Step 2 — Build Next Training Dataset (train6)
        - Use ALL images from game_dataset (not edge cases)
        - Randomly split 80/20 → train6_dataset/
        - Write data.yaml

==============================================================
"""

import os
import shutil
import random
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
import cv2

# --------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------

# 1) Model from train5
MODEL_PATH = "../../runs/detect/train5/weights/best.pt"


# 2) game_dataset original
IMAGES_DIR = "../game_dataset/images"
LABELS_DIR = "../game_dataset/labels"

# 3) output for train6
TRAIN6_DIR = "../YOLO_demo/train6"
EDGE_DIR = os.path.join(TRAIN6_DIR, "edge_case_auto")
TRAIN6_DATASET = os.path.join(TRAIN6_DIR, "train6_dataset")

IOU_THRESHOLD = 0.5
CONF_THRESHOLD = 0.25

VAL_RATIO = 0.2       # 80/20 split
SAVE_CORRECT = True   # keep correct in edge cases (optional)

IMG_EXT = (".jpg", ".png", ".jpeg")


# --------------------------------------------------------------
# Utilities
# --------------------------------------------------------------

def ensure(p):
    os.makedirs(p, exist_ok=True)

def all_images(root):
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(IMG_EXT):
                yield os.path.join(dp, f)

def load_label(path):
    boxes = []
    if not os.path.exists(path):
        return boxes
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                c, xc, yc, w, h = map(float, parts)
                boxes.append([int(c), xc, yc, w, h])
    return boxes

def yolo_to_xyxy(box, W, H):
    c, xc, yc, w, h = box
    x1 = int((xc - w/2) * W)
    y1 = int((yc - h/2) * H)
    x2 = int((xc + w/2) * W)
    y2 = int((yc + h/2) * H)
    return c, x1, y1, x2, y2

def IoU(a, b):
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])
    iw = max(0, xB - xA)
    ih = max(0, yB - yA)
    inter = iw * ih

    areaA = (a[2]-a[0]) * (a[3]-a[1])
    areaB = (b[2]-b[0]) * (b[3]-b[1])
    if areaA == 0 or areaB == 0:
        return 0
    return inter / (areaA + areaB - inter)

def copy_img_and_label(src_img, dst_folder):
    ensure(dst_folder)
    shutil.copy(src_img, dst_folder)

    lbl_path = src_img.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
    dst_lbl = dst_folder.replace("images", "labels")

    ensure(dst_lbl)
    if os.path.exists(lbl_path):
        shutil.copy(lbl_path, dst_lbl)


# ==============================================================
# STEP 1 — HARD EXAMPLE MINING
# ==============================================================

print("\n🚀 Loading YOLO model:", MODEL_PATH)
model = YOLO(MODEL_PATH)

ensure(TRAIN6_DIR)
ensure(EDGE_DIR)

for case in ["false_negative", "false_positive", "misclassified", "low_iou", "correct"]:
    ensure(os.path.join(EDGE_DIR, case))


total = correct = fn = fp = miscls = low = 0

print("\n🔍 Running Hard Example Mining...")
for img_path in all_images(IMAGES_DIR):
    total += 1

    img = cv2.imread(img_path)
    if img is None:
        continue

    H, W = img.shape[:2]

    # load label
    lbl = img_path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
    gt_raw = load_label(lbl)
    gt_boxes = [yolo_to_xyxy(b, W, H) for b in gt_raw]

    # prediction
    pred = model(img, conf=CONF_THRESHOLD, verbose=False)[0]
    preds = [[int(b.cls),
              int(b.xyxy[0][0]), int(b.xyxy[0][1]),
              int(b.xyxy[0][2]), int(b.xyxy[0][3])] for b in pred.boxes]

    used = set()
    result = "correct"

    # evaluate
    for pb in preds:
        c, x1, y1, x2, y2 = pb
        best_iou = -1
        best_gt = -1

        for g_i, g in enumerate(gt_boxes):
            i = IoU([x1,y1,x2,y2], g[1:])
            if i > best_iou:
                best_iou = i
                best_gt = g_i

        # False Positive
        if best_iou < 0.01:
            result = "false_positive"
            fp += 1
            break

        # miscls
        if c != gt_boxes[best_gt][0]:
            result = "misclassified"
            miscls += 1
            break

        # low IoU
        if best_iou < IOU_THRESHOLD:
            result = "low_iou"
            low += 1
            break

        used.add(best_gt)

    # FN
    if result == "correct":
        for i in range(len(gt_boxes)):
            if i not in used:
                result = "false_negative"
                fn += 1
                break

    if result == "correct":
        correct += 1

    # Save into edge_case_auto
    dst = os.path.join(EDGE_DIR, result)
    copy_img_and_label(img_path, os.path.join(dst, "images"))


# ==============================================================
# STEP 2 — BUILD TRAIN6 DATASET
# ==============================================================

print("\n📦 Building train6_dataset...")

# init folders
for f in ["images/train", "images/val", "labels/train", "labels/val"]:
    ensure(os.path.join(TRAIN6_DATASET, f))

all_files = list(all_images(IMAGES_DIR))
random.shuffle(all_files)

split = int(len(all_files) * (1 - VAL_RATIO))
train_files = all_files[:split]
val_files = all_files[split:]

print("➡ Train:", len(train_files))
print("➡ Val:  ", len(val_files))

# copy train
for img_path in train_files:
    dst = os.path.join(TRAIN6_DATASET, "images", "train")
    copy_img_and_label(img_path, dst)

# copy val
for img_path in val_files:
    dst = os.path.join(TRAIN6_DATASET, "images", "val")
    copy_img_and_label(img_path, dst)


# Write data.yaml
yaml_path = os.path.join(TRAIN6_DATASET, "data.yaml")
with open(yaml_path, "w") as f:
    f.write(
        f"path: {Path(TRAIN6_DATASET).absolute()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 5\n"
        f"names: ['player','meteor','star','caution_lava','exist_lava']\n"
    )

# ==============================================================
# REPORT
# ==============================================================

ts = datetime.now().strftime("%Y-%m-%d %H:%M")

report = f"""
========================================================
 YOLO Evaluation Report (train5 → build train6)
 Time: {ts}
 Model: {MODEL_PATH}

 Total:          {total}
 Correct:        {correct}
 False Negative: {fn}
 False Positive: {fp}
 Misclassified:  {miscls}
 Low IoU:        {low}

 Correct %: {round(correct/total*100,2)}%
========================================================
"""

log_path = os.path.join(EDGE_DIR, "edge_log.txt")
with open(log_path, "a") as f:
    f.write(report)

print("\n🔥 DONE!")
print(report)
print("📄 Log saved at:", log_path)
print("📂 Train6 dataset saved at:", TRAIN6_DATASET)
