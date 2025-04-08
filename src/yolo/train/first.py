from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils import SETTINGS
import os
yolo_path_str = f"{os.environ['HOME']}/.local/yolo"
yolo_path = Path(f"{yolo_path_str}")
yolo_path.mkdir(parents=True, exist_ok=True)

runs_path_str = f"{yolo_path_str}/runs"
runs_path = Path(f"{runs_path_str}")
runs_path.mkdir(parents=True, exist_ok=True)

SETTINGS['weights_dir'] = f"{yolo_path_str}/weights"
SETTINGS['datasets_dir'] = f"{yolo_path_str}/datasets"
SETTINGS['runs_dir'] = runs_path_str

model = YOLO("yolo11n.pt")
# Train the model
results = model.train(data="coco8.yaml", epochs=1, imgsz=640)
