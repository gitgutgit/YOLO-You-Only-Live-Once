import os
import shutil
from ultralytics import YOLO
import cv2
import numpy as np

# ---------------------------
# CONFIG
# ---------------------------

MODEL_PATH = "../YOLO-dataset-11221748/best.pt"    # fine-tuned model
IMAGES_DIR = "images/val"   # where your test images are
LABELS_DIR = "labels/val"   # YOLO labels (.txt)

OUTPUT_DIR = "edge_case_auto"

IOU_THRESHOLD = 0.5
CONF_THRESHOLD = 0.25   # model confidence threshold


# ---------------------------
# Helper Functions
# ---------------------------

def load_yolo_labels(label_path):
    """Load YOLO labels (class, x_center, y_center, w, h) in normalized coords."""
    labels = []
    if not os.path.exists(label_path):
        return labels
    
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, xc, yc, w, h = map(float, parts)
            labels.append([int(cls), xc, yc, w, h])
    return labels


def yolo_to_xyxy(box, img_w, img_h):
    """Convert YOLO normalized xywh to pixel xyxy."""
    cls, xc, yc, w, h = box
    x1 = int((xc - w/2) * img_w)
    y1 = int((yc - h/2) * img_h)
    x2 = int((xc + w/2) * img_w)
    y2 = int((yc + h/2) * img_h)
    return cls, x1, y1, x2, y2


def compute_iou(boxA, boxB):
    """IoU between two pixel xyxy boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    if boxA_area == 0 or boxB_area == 0:
        return 0

    return inter_area / (boxA_area + boxB_area - inter_area)


def copy_edge_case(image_path, case_type):
    """Copy image into edge_case_auto/<case_type>/"""
    dst = os.path.join(OUTPUT_DIR, case_type)
    os.makedirs(dst, exist_ok=True)
    shutil.copy(image_path, dst)


# ---------------------------
# Main Detection
# ---------------------------

print("🔍 Loading YOLO model...")
model = YOLO(MODEL_PATH)

os.makedirs(OUTPUT_DIR, exist_ok=True)

false_negative_count = 0
false_positive_count = 0
misclassified_count = 0
bad_iou_count = 0

print("🚀 Scanning images...")

for img_name in os.listdir(IMAGES_DIR):
    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(IMAGES_DIR, img_name)
    label_path = os.path.join(LABELS_DIR, img_name.replace(".jpg", ".txt").replace(".png", ".txt"))

    # Load GT labels
    gt_labels = load_yolo_labels(label_path)

    # Load image
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # Inference
    results = model(img, conf=CONF_THRESHOLD, verbose=False)[0]

    pred_boxes = []
    for b in results.boxes:
        cls = int(b.cls)
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        pred_boxes.append([cls, x1, y1, x2, y2])

    # Convert ground truth to xyxy pixel coords
    gt_boxes = [yolo_to_xyxy(gt, w, h) for gt in gt_labels]

    used_gt = set()

    # Compare predictions with GT
    for pred in pred_boxes:
        p_cls, px1, py1, px2, py2 = pred
        best_iou = 0
        best_gt_idx = -1

        for idx, gt in enumerate(gt_boxes):
            g_cls, gx1, gy1, gx2, gy2 = gt
            iou = compute_iou([px1, py1, px2, py2], [gx1, gy1, gx2, gy2])

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx

        if best_iou < 0.01:
            # No matching GT → False Positive
            false_positive_count += 1
            copy_edge_case(img_path, "false_positive")
            continue

        gt_cls = gt_boxes[best_gt_idx][0]

        if p_cls != gt_cls:
            misclassified_count += 1
            copy_edge_case(img_path, "misclassified")
        elif best_iou < IOU_THRESHOLD:
            bad_iou_count += 1
            copy_edge_case(img_path, "low_iou")

        used_gt.add(best_gt_idx)

    # False Negatives = GT but no prediction matched
    for idx, _ in enumerate(gt_boxes):
        if idx not in used_gt:
            false_negative_count += 1
            copy_edge_case(img_path, "false_negative")

print("🔥 DONE!")
print("====================")
print(f"False Negative: {false_negative_count}")
print(f"False Positive: {false_positive_count}")
print(f"Misclassified:  {misclassified_count}")
print(f"Low IoU:        {bad_iou_count}")
print("====================")
print(f"Edge cases saved in: {OUTPUT_DIR}")
