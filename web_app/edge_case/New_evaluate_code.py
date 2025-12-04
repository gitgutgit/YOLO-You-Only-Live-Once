"""
Strict Evaluator — Image-level + Object-level statistics
Chloe Lee ✨ 2025
"""

import os
import shutil
from ultralytics import YOLO
import cv2
import numpy as np

# ------------------------------------
# CONFIG
# ------------------------------------

MODEL_PATH = "../../runs/detect/train6/weights/best.pt"
IMAGES_DIR = "../game_dataset/images"
LABELS_DIR = "../game_dataset/labels"

IOU_THRESHOLD = 0.5
CONF_THRESHOLD = 0.25


# ------------------------------------
# Utilities
# ------------------------------------

def get_all_images(root):
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                yield os.path.join(dirpath, f)

def load_yolo_labels(label_path):
    labels = []
    if not os.path.exists(label_path):
        return labels
    with open(label_path, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 5:
                cls, xc, yc, w, h = map(float, parts)
                labels.append([int(cls), xc, yc, w, h])
    return labels

def yolo_to_xyxy(box, W, H):
    cls, xc, yc, w, h = box
    x1 = int((xc - w/2) * W)
    y1 = int((yc - h/2) * H)
    x2 = int((xc + w/2) * W)
    y2 = int((yc + h/2) * H)
    return cls, x1, y1, x2, y2

def compute_iou(a, b):
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (a[2]-a[0]) * (a[3]-a[1])
    areaB = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (areaA + areaB - inter) if (areaA+areaB-inter) else 0


# ------------------------------------
# MAIN
# ------------------------------------

model = YOLO(MODEL_PATH)
print("✨ Evaluating model:", MODEL_PATH)

# 📌 이미지 단위 통계
total_images = 0
correct_images = 0
incorrect_images = 0

# 📌 객체 단위 통계
fn_objects = 0
fp_objects = 0
mis_objects = 0
low_iou_objects = 0

for img_path in get_all_images(IMAGES_DIR):
    total_images += 1
    img = cv2.imread(img_path)
    if img is None:
        continue
    H, W = img.shape[:2]

    # -------- Load GT --------
    label_path = img_path.replace("/images/", "/labels/").rsplit(".",1)[0] + ".txt"
    gt_raw = load_yolo_labels(label_path)
    gt_boxes = [yolo_to_xyxy(b, W, H) for b in gt_raw]

    # -------- Prediction --------
    r = model(img, conf=CONF_THRESHOLD, verbose=False)[0]
    pred_boxes = [[int(b.cls), int(b.xyxy[0][0]), int(b.xyxy[0][1]),
                   int(b.xyxy[0][2]), int(b.xyxy[0][3])] for b in r.boxes]

    used = set()
    image_is_correct = True   # ← 이미지 단위 체크

    # -------- Match predictions to GT --------
    for pb in pred_boxes:
        cls_p, x1,y1,x2,y2 = pb

        best_iou = 0
        best_gt = -1

        for idx, gb in enumerate(gt_boxes):
            i = compute_iou([x1,y1,x2,y2], gb[1:])
            if i > best_iou:
                best_iou = i
                best_gt = idx

        if best_iou < 0.01:
            fp_objects += 1
            image_is_correct = False
            continue

        gt_cls = gt_boxes[best_gt][0]
        used.add(best_gt)

        if cls_p != gt_cls:
            mis_objects += 1
            image_is_correct = False
            continue

        if best_iou < IOU_THRESHOLD:
            low_iou_objects += 1
            image_is_correct = False
            continue

    # -------- Count false negatives --------
    for idx in range(len(gt_boxes)):
        if idx not in used:
            fn_objects += 1
            image_is_correct = False

    # -------- Update image-level stats --------
    if image_is_correct:
        correct_images += 1
    else:
        incorrect_images += 1


# ------------------------------------
# PRINT REPORT
# ------------------------------------

print("\n============================")
print("        Evaluation Result")
print("============================")
print(f"Total Images       : {total_images}")
print(f"Correct Images     : {correct_images}")
print(f"Incorrect Images   : {incorrect_images}")
print(f"Correct %          : {round(correct_images/total_images*100,2)}%")

print("\n📦 Object-level Statistics")
print(f"FN objects         : {fn_objects}")
print(f"FP objects         : {fp_objects}")
print(f"Misclassified objs : {mis_objects}")
print(f"Low IoU objs       : {low_iou_objects}")
print("============================")
