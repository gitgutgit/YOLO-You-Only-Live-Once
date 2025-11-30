import os
import shutil
from ultralytics import YOLO
import cv2
import numpy as np

# ------------------------------------
# CONFIG
# ------------------------------------

MODEL_PATH = "../../runs/detect/train2/weights/best.pt"
IMAGES_DIR = "../game_dataset/images"
LABELS_DIR = "../game_dataset/labels"

OUTPUT_DIR = "edge_case_auto"
SAVE_CORRECT = True          # 정상 케이스도 저장할지 여부
SAVE_VISUAL = False          # bbox 시각화 이미지 저장 여부

IOU_THRESHOLD = 0.5
CONF_THRESHOLD = 0.25


# ------------------------------------
# Utilities
# ------------------------------------

def get_all_images(root):
    """Recursively get all images in both train/val folders."""
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                yield os.path.join(dirpath, f)

def load_yolo_labels(label_path):
    labels = []
    if not os.path.exists(label_path):
        return labels
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls, xc, yc, w, h = map(float, parts)
                labels.append([int(cls), xc, yc, w, h])
    return labels

def yolo_to_xyxy(box, img_w, img_h):
    cls, xc, yc, w, h = box
    x1 = int((xc - w/2) * img_w)
    y1 = int((yc - h/2) * img_h)
    x2 = int((xc + w/2) * img_w)
    y2 = int((yc + h/2) * img_h)
    return cls, x1, y1, x2, y2

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h
    boxA_area = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxB_area = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    if boxA_area == 0 or boxB_area == 0:
        return 0
    return inter_area / (boxA_area + boxB_area - inter_area)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def save_copy(image_path, case):
    dst = os.path.join(OUTPUT_DIR, case)
    ensure_dir(dst)
    shutil.copy(image_path, dst)

def save_visual(img, pred_boxes, gt_boxes, save_path):
    vis = img.copy()
    for cls,x1,y1,x2,y2 in gt_boxes:
        cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,0),2)
    for cls,x1,y1,x2,y2 in pred_boxes:
        cv2.rectangle(vis,(x1,y1),(x2,y2),(0,0,255),2)
    cv2.imwrite(save_path, vis)


# ------------------------------------
# MAIN
# ------------------------------------

print("🔍 Loading YOLO model...")
model = YOLO(MODEL_PATH)

ensure_dir(OUTPUT_DIR)

false_neg = false_pos = misclassified = low_iou = correct = 0
total_images = 0

print("🚀 Scanning images...")

for img_path in get_all_images(IMAGES_DIR):
    total_images += 1

    img = cv2.imread(img_path)
    if img is None:
        continue

    h, w = img.shape[:2]

    # Label from images → labels
    label_path = img_path.replace("/images/", "/labels/").rsplit(".",1)[0] + ".txt"
    gt_labels = load_yolo_labels(label_path)
    gt_boxes = [yolo_to_xyxy(gt, w, h) for gt in gt_labels]

    # YOLO prediction
    results = model(img, conf=CONF_THRESHOLD, verbose=False)[0]
    pred_boxes = []

    for b in results.boxes:
        cls = int(b.cls)
        x1,y1,x2,y2 = map(int, b.xyxy[0])
        pred_boxes.append([cls, x1,y1,x2,y2])

    used_gt = set()
    local_case = None  # to check if correct case

    # Match predictions
    for pred in pred_boxes:
        p_cls, px1, py1, px2, py2 = pred
        best_iou = 0
        best_gt = -1

        for idx, gt in enumerate(gt_boxes):
            iou = compute_iou([px1,py1,px2,py2], gt[1:])
            if iou > best_iou:
                best_iou = iou
                best_gt = idx

        if best_iou < 0.01:
            false_pos += 1
            save_copy(img_path, "false_positive")
            local_case = "fp"
            continue

        gt_cls = gt_boxes[best_gt][0]

        if p_cls != gt_cls:
            misclassified += 1
            save_copy(img_path, "misclassified")
            local_case = "mc"
        elif best_iou < IOU_THRESHOLD:
            low_iou += 1
            save_copy(img_path, "low_iou")
            local_case = "low"
        else:
            used_gt.add(best_gt)

    # False negative check
    for idx in range(len(gt_boxes)):
        if idx not in used_gt:
            false_neg += 1
            save_copy(img_path, "false_negative")
            local_case = "fn"

    # If none of the above, it's correct case
    if local_case is None and SAVE_CORRECT:
        correct += 1
        save_copy(img_path, "correct")

print("\n🔥 DONE!")
print("=================================")
print(f"Total images:        {total_images}")
print(f"Correct:             {correct}")
print(f"False Negative:      {false_neg}")
print(f"False Positive:      {false_pos}")
print(f"Misclassified:       {misclassified}")
print(f"Low IoU:             {low_iou}")
print("=================================")
print("Correct %:", round(correct / total_images * 100, 2))
print("=================================")
print(f"Edge cases saved in: {OUTPUT_DIR}")
