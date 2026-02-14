import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")

BDD100K_PATH = os.path.join(DATA_DIR, "bdd100k")
YOLO_DATASET_PATH = os.path.join(DATA_DIR, "dataset_yolo")