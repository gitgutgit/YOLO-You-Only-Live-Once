import os
import csv
import json
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ------------------------------------------------------
# CONFIG
# ------------------------------------------------------
MODELS = {
    "train2_best": "../runs/detect/train2/weights/best.pt",
    "train3_best": "../runs/detect/train3/weights/best.pt",
    "train4_best": "../runs/detect/train4/weights/best.pt",
    "train5_best": "../runs/detect/train5/weights/best.pt",
    "train6_best": "../runs/detect/train6/weights/best.pt",
}

VAL_IMAGES = "train6/train6_dataset/images/val"
DATA_YAML = "train6/train6_dataset/data.yaml"   # val split 포함된 yaml

SAVE_DIR = "demo_test_results"
os.makedirs(SAVE_DIR, exist_ok=True)

CSV_PATH = os.path.join(SAVE_DIR, "compare_stats.csv")
MD_PATH = os.path.join(SAVE_DIR, "compare_stats.md")

# ------------------------------------------------------
# Helper to convert metrics to dict
# ------------------------------------------------------
def extract_metrics(metrics):
    """Extract useful metrics from Ultralytics result object"""
    return {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map5095": float(metrics.box.map),
    }


# ------------------------------------------------------
# RUN COMPARISON
# ------------------------------------------------------
all_stats = {}

for model_name, model_path in MODELS.items():
    print(f"\n🚀 Running evaluation for: {model_name}")

    model = YOLO(model_path)

    # 1) 예측 이미지 저장
    print("   ▶ Saving prediction images...")
    model(
        VAL_IMAGES,
        project=SAVE_DIR,
        name=f"{model_name}_predictions",
        save=True,
        conf=0.25
    )

    # 2) mAP, Recall 등 계산
    print("   ▶ Computing metrics...")
    metrics = model.val(data=DATA_YAML, split="val")
    stats = extract_metrics(metrics)

    print(f"   ✔ {model_name} metrics: {stats}")
    all_stats[model_name] = stats


# ------------------------------------------------------
# SAVE CSV
# ------------------------------------------------------
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["model", "precision", "recall", "mAP50", "mAP50-95"])
    for name, s in all_stats.items():
        writer.writerow([name, s["precision"], s["recall"], s["map50"], s["map5095"]])

print(f"\n📄 CSV saved at: {CSV_PATH}")

# ------------------------------------------------------
# SAVE MARKDOWN TABLE
# ------------------------------------------------------
with open(MD_PATH, "w") as f:
    f.write("| Model | Precision | Recall | mAP50 | mAP50-95 |\n")
    f.write("|-------|-----------|--------|--------|-----------|\n")
    for name, s in all_stats.items():
        f.write(
            f"| {name} | {s['precision']:.4f} | {s['recall']:.4f} | "
            f"{s['map50']:.4f} | {s['map5095']:.4f} |\n"
        )

print(f"📄 Markdown saved at: {MD_PATH}")

# ------------------------------------------------------
# GRAPH (mAP50 비교)
# ------------------------------------------------------
plt.figure(figsize=(10, 6))
models = list(all_stats.keys())
map50_values = [all_stats[m]["map50"] for m in models]

plt.bar(models, map50_values, color="skyblue")
plt.title("mAP50 Comparison Across Models", fontsize=16)
plt.ylabel("mAP50", fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()

GRAPH_PATH = os.path.join(SAVE_DIR, "map50_compare.png")
plt.savefig(GRAPH_PATH)
print(f"📈 Graph saved at: {GRAPH_PATH}")

print("\n🎉 ALL DONE! Chloe’s full model comparison report generated!")
