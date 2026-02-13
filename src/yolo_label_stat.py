import os
from collections import defaultdict

# -------- PATHS --------
LABEL_DIR = "data/dataset_yolo/labels/train"

# -------- CLASS MAP (must match your CLASS_MAP) --------
CLASS_NAMES = {
    0: "car",
    1: "bus",
    2: "truck",
    3: "person",
    4: "bike",
    5: "motor",
    6: "rider",
    7: "traffic light",
    8: "traffic sign",
    9: "train"
}

# -------- STATS --------
class_counts = defaultdict(int)
images_with_objects = 0
empty_images = 0
total_objects = 0

label_files = [f for f in os.listdir(LABEL_DIR) if f.endswith(".txt")]

for file in label_files:
    file_path = os.path.join(LABEL_DIR, file)

    with open(file_path, "r") as f:
        lines = f.readlines()

    if len(lines) == 0:
        empty_images += 1
        continue

    images_with_objects += 1

    for line in lines:
        class_id = int(line.strip().split()[0])
        class_counts[class_id] += 1
        total_objects += 1

# -------- PRINT RESULTS --------
print("\n📊 YOLO DATASET SUMMARY (TRAIN SET)\n")

print(f"Total images           : {len(label_files)}")
print(f"Images with objects    : {images_with_objects}")
print(f"Empty images           : {empty_images}")
print(f"Total objects          : {total_objects}")
print(f"Avg objects per image  : {total_objects / max(images_with_objects,1):.2f}\n")

print("🔹 Class-wise object count:")
for cid, count in sorted(class_counts.items()):
    print(f"{cid:2d} ({CLASS_NAMES.get(cid,'unknown'):15s}) : {count}")