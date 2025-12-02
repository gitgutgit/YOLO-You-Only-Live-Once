import os
import shutil

# ====== CONFIG ======
BASE = os.path.dirname(os.path.abspath(__file__))
EDGE = os.path.join(BASE, "edge_case", "edge_case_auto")
GAME_IMG = os.path.join(BASE, "game_dataset", "images")
GAME_LABEL = os.path.join(BASE, "game_dataset", "labels")

TRAIN3 = os.path.join(BASE, "train3_dataset")
IMG_OUT = os.path.join(TRAIN3, "images")
LBL_OUT = os.path.join(TRAIN3, "labels")
# =====================


def ensure(p):
    os.makedirs(p, exist_ok=True)

def copy_files(src_img_dir, src_lbl_dir):
    for f in os.listdir(src_img_dir):
        if not f.lower().endswith((".jpg", ".png")):
            continue
        img_path = os.path.join(src_img_dir, f)
        lbl_path = os.path.join(src_lbl_dir, f.rsplit(".",1)[0] + ".txt")

        shutil.copy(img_path, IMG_OUT)
        if os.path.exists(lbl_path):
            shutil.copy(lbl_path, LBL_OUT)


def copy_edge_case():
    # 모든 edge_case 종류 다 가져오기
    for case in ["false_negative", "false_positive", "low_iou", "misclassified", "correct"]:
        img_dir = os.path.join(EDGE, case, "images")
        lbl_dir = os.path.join(EDGE, case, "labels")
        if os.path.exists(img_dir):
            copy_files(img_dir, lbl_dir)


def main():
    print("📁 Creating train3 dataset...")
    ensure(IMG_OUT)
    ensure(LBL_OUT)

    # 1. 기본 게임데이터 포함
    print("➡ Adding game_dataset...")
    copy_files(GAME_IMG, GAME_LABEL)

    # 2. 엣지케이스 데이터 포함
    print("➡ Adding edge_case_auto...")
    copy_edge_case()

    print("🎉 train3_dataset 생성 완료!")
    print("위치:", TRAIN3)


if __name__ == "__main__":
    main()
