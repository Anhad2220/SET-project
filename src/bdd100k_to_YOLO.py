import json
import os
import shutil
from PIL import Image

# ---------------- PATHS ----------------
BASE_DIR = "data"

IMAGE_SRC_TRAIN = f"{BASE_DIR}/bdd100k/images/train"
IMAGE_SRC_VAL   = f"{BASE_DIR}/bdd100k/images/val"

LABEL_SRC_TRAIN = f"{BASE_DIR}/bdd100k/labels/train.json"
LABEL_SRC_VAL   = f"{BASE_DIR}/bdd100k/labels/val.json"

YOLO_IMG_TRAIN = f"{BASE_DIR}/dataset_yolo/images/train"
YOLO_IMG_VAL   = f"{BASE_DIR}/dataset_yolo/images/val"

YOLO_LBL_TRAIN = f"{BASE_DIR}/dataset_yolo/labels/train"
YOLO_LBL_VAL   = f"{BASE_DIR}/dataset_yolo/labels/val"

# ------------- CLASS MAP ----------------
CLASS_MAP = {
    "car": 0,
    "bus": 1,
    "truck": 2,
    "person": 3,
    "bike": 4,
    "motor": 5,
    "rider": 6,
    "traffic light": 7,
    "traffic sign": 8,
    "train": 9
}

# ----------- HELPER FUNCTION ------------
def convert_bbox(box, img_w, img_h):
    x1, y1, x2, y2 = box
    xc = ((x1 + x2) / 2) / img_w
    yc = ((y1 + y2) / 2) / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    return xc, yc, w, h

# ------------- MAIN FUNCTION ------------
def process_split(label_path, img_src, img_dst, lbl_dst):
    with open(label_path, "r") as f:
        data = json.load(f)

    for item in data:
        img_name = item["name"]
        img_path = os.path.join(img_src, img_name)

        # skip if image does not exist
        if not os.path.exists(img_path):
            continue

        # read image size safely
        with Image.open(img_path) as img:
            img_w, img_h = img.size

        yolo_labels = []

        # iterate over objects in image
        for obj in item.get("labels", []):
            cls_name = obj.get("category")
            if cls_name not in CLASS_MAP:
                continue

            box = obj.get("box2d")
            if box is None:
                continue

            # convert bbox to YOLO format
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]

            x_center = ((x1 + x2) / 2) / img_w
            y_center = ((y1 + y2) / 2) / img_h
            width = (x2 - x1) / img_w
            height = (y2 - y1) / img_h

            yolo_labels.append(
                f"{CLASS_MAP[cls_name]} {x_center} {y_center} {width} {height}"
            )

        # write YOLO label file (even if empty)
        label_file = os.path.join(lbl_dst, img_name.replace(".jpg", ".txt"))
        with open(label_file, "w") as f:
            f.write("\n".join(yolo_labels))

        # copy image to YOLO folder
        shutil.copy(img_path, img_dst)

# ---------------- RUN -------------------
process_split(LABEL_SRC_TRAIN, IMAGE_SRC_TRAIN, YOLO_IMG_TRAIN, YOLO_LBL_TRAIN)
process_split(LABEL_SRC_VAL, IMAGE_SRC_VAL, YOLO_IMG_VAL, YOLO_LBL_VAL)

print("✅ Conversion completed successfully")