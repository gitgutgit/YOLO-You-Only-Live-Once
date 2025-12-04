import os
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt

MODELS = {
    "train2_best": "../runs/detect/train2/weights/best.pt",
    "train3_best": "../runs/detect/train3/weights/best.pt",
    "train4_best": "../runs/detect/train4/weights/best.pt",
    "train5_best": "../runs/detect/train5/weights/best.pt",
    "train6_best": "../runs/detect/train6/weights/best.pt"
}

DATA_YAML = "train6/train6_dataset/data.yaml"

LAVA_CLASSES = [3, 4]  # caution_lava, exist_lava
LAVA_NAMES = ["caution_lava", "exist_lava"]

lava_results = []

for model_name, model_path in MODELS.items():
    print(f"\n🚀 Evaluating lava accuracy for {model_name}")

    model = YOLO(model_path)
    result = model.val(data=DATA_YAML, verbose=False)

    # ✔ Correct attributes
    p = result.box.p
    r = result.box.r
    ap50 = result.box.ap50
    ap_class = result.box.ap   # ← AP50-95 per class

    model_entry = {"model": model_name}

    for i, cls_idx in enumerate(LAVA_CLASSES):
        model_entry[f"{LAVA_NAMES[i]}_precision"] = p[cls_idx]
        model_entry[f"{LAVA_NAMES[i]}_recall"] = r[cls_idx]
        model_entry[f"{LAVA_NAMES[i]}_mAP50"] = ap50[cls_idx]
        model_entry[f"{LAVA_NAMES[i]}_mAP50-95"] = ap_class[cls_idx]

    lava_results.append(model_entry)

df = pd.DataFrame(lava_results)
print("\n🔥 Lava Class Comparison:")
print(df)

df.to_csv("lava_class_comparison.csv", index=False)

df_plot = df.set_index('model')[[f"{n}_mAP50-95" for n in LAVA_NAMES]]
df_plot.plot(kind="bar", figsize=(10, 6))
plt.title("Lava Detection Performance (mAP50-95)")
plt.ylabel("Score")
plt.tight_layout()
plt.savefig("lava_class_compare.png")
plt.show()
